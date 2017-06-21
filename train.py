import sys
import os
import numpy as np
from model import *
from datasets import get_vocabulary, prepare_pair_data
import cPickle as pickle


encoder_hidden_size = 512
n_encoder_layers = 2
decoder_hidden_size = 512
embed_size = 512
vocabulary_size = 20000
learning_rate = 0.0001
n_steps = 500000
grad_clip = 1.0

save_every = n_steps // 20
log_every_n_seconds = 5 * 60
log_every_n_steps = 10000

kld_start_inc = 0 #.01 * n_steps
kld_weight = 1.0
kld_max = 1.0
kld_inc = 0.
#kld_inc = (kld_max - kld_weight) / (.01 * n_steps)
freebits_lambda = 2.0

word_dropout = 0.1

temperature = .9
temperature_min = .85
#temperature_dec = 0.
# should get to the temperature around 80% through training, then hold
temperature_dec = (temperature - temperature_min) / (0.8 * n_steps)
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


def word_tensor(lang, string):
    split_string = string.split(" ")
    size = len(split_string) + 1
    tensor = torch.zeros(size).long()
    for c in range(len(split_string)):
        tensor[c] = lang.word_to_index(split_string[c])
    tensor[-1] = EOS_token
    tensor = Variable(tensor)
    if USE_CUDA:
        tensor = tensor.cuda()
    return tensor


def index_to_word(lang, top_i):
    if top_i == EOS_token:
        return 'EOS' + " "
    elif top_i == SOS_token:
        return 'SOS' + " "
    elif top_i == UNK_token:
        return 'UNK' + " "
    else:
        return lang.index_to_word(top_i) + " "

def long_word_tensor_to_string(lang, t):
    s = ''
    for i in range(t.size(0)):
        top_i = t.data[i]
        s += index_to_word(lang, top_i)
    return s

def float_word_tensor_to_string(lang, t):
    s = ''
    for i in range(t.size(0)):
        ti = t[i]
        top_k = ti.data.topk(1)
        top_i = top_k[1][0]
        s += index_to_word(lang, top_i)
        if top_i == EOS_token:
            break
    return s

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
e = EncoderRNN(n_words, encoder_hidden_size, embed_size, n_encoder_layers, bidirectional=False)

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
    for step in range(n_steps):
        input, target = random_training_set()
        optimizer.zero_grad()

        m, l, z, decoded = vae(input, target, temperature)
        if temperature > temperature_min:
            temperature -= temperature_dec

        loss = criterion(decoded, target)
        #job.record(step, loss.data[0])

        # free bits
        full_KLD = 0.5 * (l - torch.pow(m, 2) - torch.exp(l) + 1)
        KLD = -1. * torch.clamp(full_KLD.mean(), max=freebits_lambda).squeeze()


        #KLD = (-0.5 * torch.sum(l - torch.pow(m, 2) - torch.exp(l) + 1, 1)).mean().squeeze()
        loss += KLD * kld_weight

        if step > kld_start_inc and kld_weight < kld_max:
            kld_weight += kld_inc

        loss.backward()
        # print('from', next(vae.parameters()).grad.data[0][0])
        ec = torch.nn.utils.clip_grad_norm(vae.parameters(), grad_clip)
        # print('to  ', next(vae.parameters()).grad.data[0][0])
        optimizer.step()

        def log_and_generate(tag, value):
            if tag == "step":
                print('|%s|[%d] %.4f (k=%.4f, t=%.4f, kl=%.4f, ec=%.4f)' % (
                    tag, value, loss.data[0], kld_weight, temperature, KLD.data[0], ec
                ))
            elif tag == "time":
                print('|%s|[%.4f] %.4f (k=%.4f, t=%.4f, kl=%.4f, ec=%.4f)' % (
                    tag, value, loss.data[0], kld_weight, temperature, KLD.data[0], ec
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

        if step > 0 and step % save_every == 0:
            save()

    save()

except KeyboardInterrupt as err:
    print("ERROR", err)
    print("Saving before quit...")
    save()

