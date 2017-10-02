import sys
import os
import numpy as np
from model import *
from datasets import get_vocabulary, prepare_pair_data
import cPickle as pickle
import shutil


encoder_hidden_size = 512
n_encoder_layers = 2
decoder_hidden_size = 512
embed_size = 128
vocabulary_size = 20000
learning_rate = 0.0001
n_steps = 1500000
grad_clip = 10.0

save_every = n_steps // 20
log_every_n_seconds = 5 * 60
log_every_n_steps = 10000

#kld_start_inc = 0 #.01 * n_steps
kld_start_inc = 10000
kld_weight = 0.0
kld_max = 1.0
kld_inc = (kld_max - kld_weight) / (n_steps // 2)
#kld_inc = 0.
habits_lambda = .2

word_dropout = 0.25

temperature = 1.0
temperature_min = .75
# should get to the temperature around 50% through training, then hold
temperature_dec = (temperature - temperature_min) / (0.5 * n_steps)
#temperature_dec = 0.
USE_CUDA = True


# Training
# ------------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python train.py [filename]")
    sys.exit(1)

reverse = True
csv = False
if sys.argv[1].endswith(".csv"):
    csv = True

if sys.argv[1].endswith(".pkl"):
    cache_path = sys.argv[1]
else:
    tmp_path = "/Tmp/kastner/"
    cache_path = tmp_path + sys.argv[1].split(os.sep)[-1].split(".")[0] + "_stored_info.pkl"

if not os.path.exists(cache_path):
    print("Cached info at {} not found".format(cache_path))
    print("Creating cache... this may take some time")
    input_side, output_side, pairs = prepare_pair_data(sys.argv[1], vocabulary_size, reverse, csv)
    with open(cache_path, "wb") as f:
        pickle.dump((input_side, output_side, pairs), f)
else:
    start_load = time.time()
    print("Fetching cached info at {}".format(cache_path))
    with open(cache_path, "rb") as f:
        input_side, output_side, pairs = pickle.load(f)
    end_load = time.time()
    print("Cache {} loaded, total load time {}".format(cache_path, end_load - start_load))

random_state = np.random.RandomState(1999)
random_state.shuffle(pairs)


def random_training_set():
    pair_i = random_state.choice(len(pairs))
    pair = pairs[pair_i]
    inp = word_tensor(input_side, pair[0])
    target = word_tensor(output_side, pair[1])
    #inp_str = long_word_tensor_to_string(input_side, inp)
    #target_str = long_word_tensor_to_string(output_side, target)
    #from IPython import embed; embed(); raise ValueError()
    return inp, target

n_words = input_side.n_words
e = EncoderRNN(n_words, encoder_hidden_size, embed_size, n_encoder_layers, bidirectional=True)

# custom weights initialization
def rnn_weights_init(m):
    for c in m.children():
        classname = c.__class__.__name__
        if classname.find("GRU") != -1:
            for k, v in c.named_parameters():
                if "weight" in k:
                    v.data.normal_(0.0, 0.02)

d = DecoderRNN(embed_size, decoder_hidden_size, n_words, 1, word_dropout=word_dropout)
rnn_weights_init(d)

vae = VAE(e, d)
if os.path.exists("vae.pt"):
    print("Found saved model {}, continuing...".format("vae.pt"))
    shutil.copyfile("vae.pt", "vae.pt.bak")
    vae = torch.load("vae.pt")
    print("Found model was already trained for {} steps".format(vae.steps_seen))
    temperature = temperature_min
    temperature_min = temperature_min
    temperature_dec = 0.

    kld_weight = kld_max
    kld_inc = 0.

    # change random seed and reshuffle the data, so that we don't repeat the same
    # use hash of the weights and biases? try with float16 to avoid numerical issues in the tails...
    new_seed = hash(tuple([hash(tuple(vae.state_dict()[k].cpu().numpy().ravel().astype("float16"))) for k, v in vae.state_dict().items()]))
    # must be between 0 and 4294967295
    new_seed = abs(new_seed) % 4294967295
    print("Setting new random seed {}".format(new_seed))
    random_state = np.random.RandomState(new_seed)
    print("Reshuffling training data")
    random_state.shuffle(pairs)

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


if USE_CUDA:
    vae.cuda()
    criterion.cuda()
    print("Using CUDA!")


"""
save_every = 5000
job = sconce.Job('vae', {
    'hidden_size': hidden_size,
    'embed_size': embed_size,
    'learning_rate': learning_rate,
    'kld_weight': kld_weight,
    'temperature': temperature,
    'grad_clip': grad_clip,
})

job.log_every = log_every
"""

def save():
    save_filename = 'vae.pt'
    torch.save(vae, save_filename)
    print('Saved as %s' % save_filename)

try:
    # set it so that the first one logs
    start_time = time.time()
    last_log_time = time.time() - log_every_n_seconds
    last_log_step = -log_every_n_steps - 1
    start_steps = vae.steps_seen
    for step in range(n_steps):
        input, target = random_training_set()
        optimizer.zero_grad()

        m, l, z, decoded = vae(input, target, temperature)
        if temperature > temperature_min:
            temperature -= temperature_dec

        ll_loss = criterion(decoded, target)
        #job.record(step, loss.data[0])

        KLD = -0.5 * (2 * l - torch.pow(m, 2) - torch.pow(torch.exp(l), 2) + 1)
        # ha bits , like free bits but over whole layer
        clamp_KLD = torch.clamp(KLD.mean(), min=habits_lambda).squeeze()
        #neg_KLD = -1 * clamp_KLD
        loss = ll_loss + clamp_KLD * kld_weight

        if step > kld_start_inc and kld_weight < kld_max:
            kld_weight += kld_inc

        loss.backward()
        # print('from', next(vae.parameters()).grad.data[0][0])
        ec = torch.nn.utils.clip_grad_norm(vae.parameters(), grad_clip)
        # print('to  ', next(vae.parameters()).grad.data[0][0])
        optimizer.step()

        def log_and_generate(tag, value):
            if tag == "step":
                print('|%s|[%d] %.4f (k=%.4f, t=%.4f, kl=%.4f, ckl=%.4f,  nll=%.4f, ec=%.4f)' % (
                    tag, value, loss.data[0], kld_weight, temperature, KLD.data.mean(), clamp_KLD.data[0], ll_loss.data[0], ec
                ))
            elif tag == "time":
                print('|%s|[%.4f] %.4f (k=%.4f, t=%.4f, kl=%.4f, ckl=%.4f, nll=%.4f,  ec=%.4f)' % (
                    tag, value, loss.data[0], kld_weight, temperature, KLD.data.mean(), clamp_KLD.data[0], ll_loss.data[0], ec
                ))
            inp_str = long_word_tensor_to_string(input_side, input)
            print('    (input {}) "{}"'.format(tag, inp_str))
            target_str = long_word_tensor_to_string(output_side, target)
            if target_str.endswith("EOS "):
               target_str = target_str[:-4]
            #from IPython import embed; embed(); raise ValueError()
            # flip it back
            print('   (target {}) "{}"'.format(tag, target_str[::-1]))
            generated = vae.decoder.generate(z, MAX_LENGTH, temperature)
            generated_str = float_word_tensor_to_string(output_side, generated)
            if generated_str.endswith("EOS "):
               generated_str = generated_str[:-4]
            # flip it back
            print('(generated {}) "{}"'.format(tag, generated_str[::-1]))
            print('')

        if last_log_time <= time.time() - log_every_n_seconds:
            log_and_generate("time", time.time() - start_time)
            last_log_time = time.time()

        if last_log_step <= step - log_every_n_steps:
            log_and_generate("step", step)
            last_log_step = step

        if step > 0 and step % save_every == 0 or step == (n_steps - 1):
            vae.steps_seen = start_steps + step
            save()

    save()

except KeyboardInterrupt as err:
    print("ERROR", err)
    print("Saving before quit...")
    save()

