import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from datasets import *

MIN_LENGTH = 10
MAX_LENGTH = 50
MAX_SAMPLE = False
MAX_SAMPLE = True
model_random_state = np.random.RandomState(1988)
torch.manual_seed(1999)

import torch
from torch.nn import Parameter
from functools import wraps


def _decorate(forward, module, name, name_g, name_v):
    @wraps(forward)
    def decorated_forward(*args, **kwargs):
        g = module.__getattr__(name_g)
        v = module.__getattr__(name_v)
        w = v*(g/torch.norm(v)).expand_as(v)
        module.__setattr__(name, w)
        return forward(*args, **kwargs)
    return decorated_forward


def weight_norm(module, name):
    param = module.__getattr__(name)

    # construct g,v such that w = g/||v|| * v
    g = torch.norm(param)
    v = param/g.expand_as(param)
    g = Parameter(g.data)
    v = Parameter(v.data)
    name_g = name + '_g'
    name_v = name + '_v'

    # remove w from parameter list
    del module._parameters[name]

    # add g and v as new parameters
    module.register_parameter(name_g, g)
    module.register_parameter(name_v, v)

    # construct w every time before forward is called
    module.forward = _decorate(module.forward, module, name, name_g, name_v)
    return module


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


class Encoder(nn.Module):
    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        if USE_CUDA:
            eps = eps.cuda()
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

# Encoder
# ------------------------------------------------------------------------------

# Encode into Z with mu and log_var

class EncoderRNN(Encoder):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.1, bidirectional=bidirectional)
        self.o2p = nn.Linear(hidden_size, output_size * 2)

    def forward(self, input):
        embedded = self.embed(input).unsqueeze(1)

        output, hidden = self.gru(embedded, None)
        # mean loses positional info?
        #output = torch.mean(output, 0).squeeze(0) #output[-1] # Take only the last value
        output = output[-1]#.squeeze(0)
        if self.bidirectional:
            output = output[:, :self.hidden_size] + output[: ,self.hidden_size:] # Sum bidirectional outputs
        else:
            output = output[:, :self.hidden_size]

        ps = self.o2p(output)
        mu, logvar = torch.chunk(ps, 2, dim=1)
        z = self.sample(mu, logvar)
        return mu, logvar, z

# Decoder
# ------------------------------------------------------------------------------

# Decode from Z into sequence

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, word_dropout=1.):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.word_dropout = word_dropout

        self.embed = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size + input_size, hidden_size, n_layers)
        self.z2h = nn.Linear(input_size, hidden_size)
        self.i2h = nn.Linear(hidden_size + input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size + input_size, output_size)
        #self.out = nn.Linear(hidden_size, output_size)

    def sample(self, output, temperature):
        if MAX_SAMPLE:
            # Sample top value only
            top_i = output.data.topk(1)[1][0][0]

        else:
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

        input = Variable(torch.LongTensor([top_i]))
        if USE_CUDA:
            input = input.cuda()
        return input, top_i

    def forward(self, z, inputs, temperature):
        n_steps = inputs.size(0)
        outputs = Variable(torch.zeros(n_steps, 1, self.output_size))
        if USE_CUDA:
            outputs = outputs.cuda()

        input = Variable(torch.LongTensor([SOS_token]))
        if USE_CUDA:
            input = input.cuda()

        hidden = self.z2h(z).unsqueeze(0).repeat(self.n_layers, 1, 1)

        for i in range(n_steps):
            output, hidden = self.step(i, z, input, hidden)
            outputs[i] = output

            use_word_dropout = model_random_state.rand() < self.word_dropout
            if use_word_dropout and i < (n_steps - 1):
                unk_input = Variable(torch.LongTensor([UNK_token]))
                if USE_CUDA:
                    unk_input = unk_input.cuda()
                input = unk_input
                continue

            use_teacher_forcing = model_random_state.rand() < temperature
            if use_teacher_forcing:
                input = inputs[i]
            else:
                input, top_i = self.sample(output, temperature)

        return outputs.squeeze(1)

    def generate(self, z, n_steps, temperature):
        outputs = Variable(torch.zeros(n_steps, 1, self.output_size))
        if USE_CUDA:
            outputs = outputs.cuda()

        input = Variable(torch.LongTensor([SOS_token]))
        if USE_CUDA:
            input = input.cuda()
        hidden = self.z2h(z).unsqueeze(0).repeat(self.n_layers, 1, 1)

        for i in range(n_steps):
            output, hidden = self.step(i, z, input, hidden)
            outputs[i] = output
            input, top_i = self.sample(output, temperature)
            #if top_i == EOS: break
        return outputs.squeeze(1)

    def step(self, s, z, input, hidden):
        # print('[DecoderRNN.step] s =', s, 'z =', z.size(), 'i =', input.size(), 'h =', hidden.size())
        input = F.relu(self.embed(input))
        input = torch.cat((input, z), 1)
        input = input.unsqueeze(0)
        output, hidden = self.gru(input, hidden)
        output = output.squeeze(0)
        output = torch.cat((output, z), 1)
        output = self.out(output)
        return output, hidden

# Container
# ------------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.steps_seen = 0

    def encode(self, inputs):
        m, l, z = self.encoder(inputs)
        return m, l, z

    def forward(self, inputs, targets, temperature=1.0):
        m, l, z = self.encoder(inputs)
        decoded = self.decoder(z, targets, temperature)
        return m, l, z, decoded

# Test

if __name__ == '__main__':
    hidden_size = 20
    embed_size = 10
    e = EncoderRNN(n_characters, hidden_size, embed_size)
    d = DecoderRNN(embed_size, hidden_size, n_characters, 2)
    if USE_CUDA:
        e.cuda()
        d.cuda()
    vae = VAE(e, d)
    m, l, z, decoded = vae(char_tensor('@spro'))
    print('m =', m.size())
    print('l =', l.size())
    print('z =', z.size())
    print('decoded', tensor_to_string(decoded))

