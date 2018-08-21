# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import random
import time
import math
import torch
from torch.autograd import Variable

USE_CUDA = True

# Reading and un-unicode-encoding data
SOS_token = 0
EOS_token = 1
UNK = 2

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

# Turning a string into a tensor

def char_tensor(string):
    size = len(string) + 1
    tensor = torch.zeros(size).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    tensor[-1] = EOS
    tensor = Variable(tensor)
    if USE_CUDA:
        tensor = tensor.cuda()
    return tensor

# Turn a tensor into a string

def index_to_char(top_i):
    if top_i == EOS:
        return '$'
    elif top_i == SOS:
        return '^'
    elif top_i == UNK:
        return '*'
    else:
        return all_characters[top_i]

def tensor_to_string(t):
    s = ''
    for i in range(t.size(0)):
        ti = t[i]
        top_k = ti.data.topk(1)
        top_i = top_k[1][0]
        s += index_to_char(top_i)
    return s.split(index_to_char(EOS))[0]
    #return s

def longtensor_to_string(t):
    s = ''
    for i in range(t.size(0)):
        top_i = t.data[i]
        s += index_to_char(top_i)
    return s

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

