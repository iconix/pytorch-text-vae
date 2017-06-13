import sys
import os
import numpy as np
from model import *
from datasets import get_vocabulary, prepare_pair_data
import cPickle as pickle


encoder_hidden_size = 1024
decoder_hidden_size = 256
embed_size = 500
vocabulary_size = 20000
learning_rate = 0.0001
n_epochs = 500000
grad_clip = 1.0

kld_start_inc = 50000
kld_weight = 0.05
kld_max = 1.0
kld_inc = 1E-5 #kld_max / 10000
input_keep = 0.75
temperature = 1.0
temperature_min = 1.0
temperature_dec = temperature / 50000
#temperature_dec = 0.000002
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
cache_path = tmp_path + sys.argv[1].split(".")[0] + "_stored_info.pkl"
if not os.path.exists(cache_path):
    input_side, output_side, pairs = prepare_pair_data(sys.argv[1], vocabulary_size, reverse, csv)
    with open(cache_path, "wb") as f:
        pickle.dump((input_side, output_side, pairs), f)
else:
    print("Fetching cached info at {}".format(cache_path))
    with open(cache_path, "rb") as f:
        input_side, output_side, pairs = pickle.load(f)

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
e = EncoderRNN(n_words, encoder_hidden_size, embed_size, bidirectional=False)
d = DecoderRNN(embed_size, decoder_hidden_size, n_words, 1, input_keep=input_keep)
vae = VAE(e, d)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

if USE_CUDA:
    vae.cuda()
    criterion.cuda()
    print("Using CUDA!")

save_every = 5000
log_every = 200
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
    for epoch in range(n_epochs):
        input, target = random_training_set()
        optimizer.zero_grad()

        m, l, z, decoded = vae(input, target, temperature)
        if temperature > temperature_min:
            temperature -= temperature_dec

        loss = criterion(decoded, target)
        #job.record(epoch, loss.data[0])

        KLD = (-0.5 * torch.sum(l - torch.pow(m, 2) - torch.exp(l) + 1, 1)).mean().squeeze()
        loss += KLD * kld_weight

        if epoch > kld_start_inc and kld_weight < kld_max:
            kld_weight += kld_inc

        loss.backward()
        # print('from', next(vae.parameters()).grad.data[0][0])
        ec = torch.nn.utils.clip_grad_norm(vae.parameters(), grad_clip)
        # print('to  ', next(vae.parameters()).grad.data[0][0])
        optimizer.step()

        if epoch % log_every == 0:
            print('[%d] %.4f (k=%.4f, t=%.4f, kl=%.4f, ec=%.4f)' % (
                epoch, loss.data[0], kld_weight, temperature, KLD.data[0], ec
            ))
            #inp_str = word_tensor_to_string(input_side, inp)
            target_str = long_word_tensor_to_string(output_side, target)
            if target_str.endswith("EOS "):
               target_str = target_str[:-4]
            #from IPython import embed; embed(); raise ValueError()
            # flip it back
            print('   (target) "%s"' % target_str[::-1])
            generated = vae.decoder.generate(z, MAX_LENGTH, temperature)
            generated_str = float_word_tensor_to_string(output_side, generated)
            if generated_str.endswith("EOS "):
               generated_str = generated_str[:-4]
            # flip it back
            print('(generated) "%s"' % generated_str[::-1])
            print('')

        if epoch > 0 and epoch % save_every == 0:
            save()

    save()

except KeyboardInterrupt as err:
    print("ERROR", err)
    print("Saving before quit...")
    save()

