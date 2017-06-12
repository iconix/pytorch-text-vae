#import sconce
import sys
import numpy as np
import re
import unicodedata
from model import *

encoder_hidden_size = 1024
decoder_hidden_size = 256
embed_size = 500
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

SOS_token = 0
EOS_token = 1
UNK_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3 # Count SOS, EOS, UNK

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize(u'NFD', unicode(s))
        if unicodedata.category(c) != u'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

MIN_LENGTH = 5
MAX_LENGTH = 20

def read_langs(reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    f, flen = read_file(sys.argv[1])
    lines = []
    for n, line in enumerate(f.split("\n")):
        if len(line.strip()) > 0:
            l = line.strip().split(",")[1]
            l = re.sub(r'[^\w]', ' ', l)
            if "." in l:
                l = l.split(".")[0]
            l = re.sub(r'\s+', ' ', l).strip().lstrip().rstrip()
            lines.append(l)

    # Split every line into pairs and normalize
    pairs = [[normalize_string(l), normalize_string(l)] for l in lines]

    # Reverse second of pairs, make Lang instances
    if reverse:
        pairs = [(p[0], "".join(list(reversed(p[1])))) for p in pairs]

    input_lang = Lang("in")
    output_lang = Lang("out")

    return input_lang, output_lang, pairs


def filter_pair(p):
    return MIN_LENGTH < len(p[0].split(' ')) < MAX_LENGTH and MIN_LENGTH < len(p[1].split(' ')) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(reverse=False):
    input_side, output_side, pairs = read_langs(reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_side.index_words(pair[0])
        output_side.index_words(pair[1])

    return input_side, output_side, pairs

input_side, output_side, pairs = prepare_data(True)
random_state = np.random.RandomState(1999)
random_state.shuffle(pairs)


def word_tensor(lang, string):
    split_string = string.split(" ")
    size = len(split_string) + 1
    tensor = torch.zeros(size).long()
    for c in range(len(split_string)):
        tensor[c] = lang.word2index[split_string[c]]
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
        return lang.index2word[top_i] + " "

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

