import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from functools import wraps

if __package__ is None or __package__ == '':
    from datasets import *
else:
    from pytorchtextvae.datasets import *

MAX_SAMPLE = False
TRUNCATED_SAMPLE = True
model_random_state = np.random.RandomState(1988)
torch.manual_seed(1999)


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
    return tensor

def _pair_to_tensors(input_side, output_side, pair, device):
    inp = word_tensor(input_side, pair[0]).to(device)
    target = word_tensor(output_side, pair[1]).to(device)
    condition = torch.tensor(pair[2], dtype=torch.float).unsqueeze(0).to(device) if len(pair) == 3 else None

    return inp, target, condition

def random_training_set(dataset, random_state, device):
    pair_i = random_state.choice(len(dataset.trn_pairs))
    pair = dataset.trn_pairs[pair_i]
    return _pair_to_tensors(dataset.input_side, dataset.output_side, pair, device)

def random_test_set(dataset, random_state, device):
    pair_i = random_state.choice(len(dataset.test_pairs))
    pair = dataset.test_pairs[pair_i]
    return _pair_to_tensors(dataset.input_side, dataset.output_side, pair, device)

def index_to_word(lang, top_i):
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
    def sample(self, mu, logvar, device):
        eps = Variable(torch.randn(mu.size())).to(device)
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

    def forward(self, input, device):
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
        z = self.sample(mu, logvar, device)
        return mu, logvar, z

# Decoder
# ------------------------------------------------------------------------------

# Decode from Z into sequence

class DecoderRNN(nn.Module):
    def __init__(self, z_size, n_conditions, condition_size, hidden_size, output_size, n_layers=1, word_dropout=1.):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.word_dropout = word_dropout

        input_size = z_size + condition_size

        self.embed = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size + input_size, hidden_size, n_layers)
        self.i2h = nn.Linear(input_size, hidden_size)
        if n_conditions > 0 and condition_size > 0 and n_conditions != condition_size:
            self.c2h = nn.Linear(n_conditions, condition_size)
        #self.dropout = nn.Dropout()
        self.h2o = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size + input_size, output_size)

        print(f'MAX_SAMPLE: {MAX_SAMPLE}; TRUNCATED_SAMPLE: {TRUNCATED_SAMPLE}')

    def sample(self, output, temperature, device, max_sample=MAX_SAMPLE, trunc_sample=TRUNCATED_SAMPLE):
        if max_sample:
            # Sample top value only
            top_i = output.data.topk(1)[1].item()
        else:
            # Sample from the network as a multinomial distribution
            if trunc_sample:
                # Sample from top k values only
                k = 10
                new_output = torch.empty_like(output).fill_(float('-inf'))
                top_v, top_i = output.data.topk(k)
                new_output.data.scatter_(1, top_i, top_v)
                output = new_output

            output_dist = output.data.view(-1).div(temperature).exp()
            if len(torch.nonzero(output_dist)) > 0:
                top_i = torch.multinomial(output_dist, 1)[0]
            else:
                # TODO: how does this happen?
                print(f'[WARNING] output_dist is all zeroes')
                top_i = UNK_token

        input = Variable(torch.LongTensor([top_i])).to(device)
        return input, top_i

    def forward(self, z, condition, inputs, temperature, device):
        n_steps = inputs.size(0)
        outputs = Variable(torch.zeros(n_steps, 1, self.output_size)).to(device)

        input = Variable(torch.LongTensor([SOS_token])).to(device)
        if condition is None:
            decode_embed = z
        else:
            if hasattr(self, 'c2h'):
                #squashed_condition = self.c2h(self.dropout(condition))
                squashed_condition = self.c2h(condition)
                decode_embed = torch.cat([z, squashed_condition], 1)
            else:
                decode_embed = torch.cat([z, condition], 1)


        hidden = self.i2h(decode_embed).unsqueeze(0).repeat(self.n_layers, 1, 1)

        for i in range(n_steps):
            output, hidden = self.step(i, decode_embed, input, hidden)
            outputs[i] = output

            use_word_dropout = model_random_state.rand() < self.word_dropout
            if use_word_dropout and i < (n_steps - 1):
                unk_input = Variable(torch.LongTensor([UNK_token])).to(device)
                input = unk_input
                continue

            use_teacher_forcing = model_random_state.rand() < temperature
            if use_teacher_forcing:
                input = inputs[i]
            else:
                input, top_i = self.sample(output, temperature, device, max_sample=True)

            if input.dim() == 0:
                input = input.unsqueeze(0)

        return outputs.squeeze(1)

    def generate_with_embed(self, embed, n_steps, temperature, device, max_sample=MAX_SAMPLE, trunc_sample=TRUNCATED_SAMPLE):
        outputs = Variable(torch.zeros(n_steps, 1, self.output_size)).to(device)
        input = Variable(torch.LongTensor([SOS_token])).to(device)

        hidden = self.i2h(embed).unsqueeze(0).repeat(self.n_layers, 1, 1)

        for i in range(n_steps):
            output, hidden = self.step(i, embed, input, hidden)
            outputs[i] = output
            input, top_i = self.sample(output, temperature, device, max_sample=max_sample, trunc_sample=trunc_sample)
            #if top_i == EOS: break
        return outputs.squeeze(1)

    def generate(self, z, condition, n_steps, temperature, device, max_sample=MAX_SAMPLE, trunc_sample=TRUNCATED_SAMPLE):
        if condition is None:
            decode_embed = z
        else:
            if condition.dim() == 1:
                condition = condition.unsqueeze(0)

            if hasattr(self, 'c2h'):
                #squashed_condition = self.c2h(self.dropout(condition))
                squashed_condition = self.c2h(condition)
                decode_embed = torch.cat([z, squashed_condition], 1)
            else:
                decode_embed = torch.cat([z, condition], 1)

        return self.generate_with_embed(decode_embed, n_steps, temperature, device, max_sample, trunc_sample)

    def step(self, s, decode_embed, input, hidden):
        # print('[DecoderRNN.step] s =', s, 'decode_embed =', decode_embed.size(), 'i =', input.size(), 'h =', hidden.size())
        input = F.relu(self.embed(input))
        input = torch.cat((input, decode_embed), 1)
        input = input.unsqueeze(0)
        output, hidden = self.gru(input, hidden)
        output = output.squeeze(0)
        output = torch.cat((output, decode_embed), 1)
        output = self.out(output)
        return output, hidden

# Container
# ------------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, encoder, decoder, n_steps=None):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.register_buffer('steps_seen', torch.tensor(0, dtype=torch.long))
        self.register_buffer('kld_max', torch.tensor(1.0, dtype=torch.float))
        self.register_buffer('kld_weight', torch.tensor(0.0, dtype=torch.float))
        if n_steps is not None:
            self.register_buffer('kld_inc', torch.tensor((self.kld_max - self.kld_weight) / (n_steps // 2), dtype=torch.float))
        else:
            self.register_buffer('kld_inc', torch.tensor(0, dtype=torch.float))

    def encode(self, inputs):
        m, l, z = self.encoder(inputs)
        return m, l, z

    def forward(self, inputs, targets, condition, device, temperature=1.0):
        m, l, z = self.encoder(inputs, device)
        decoded = self.decoder(z, condition, targets, temperature, device)
        return m, l, z, decoded

# Test

if __name__ == '__main__':
    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    hidden_size = 20
    z_size = 10
    e = EncoderRNN(n_characters, hidden_size, z_size).to(device)
    d = DecoderRNN(z_size, hidden_size, n_characters, 2).to(device)
    vae = VAE(e, d)
    m, l, z, decoded = vae(char_tensor('@spro'))
    print('m =', m.size())
    print('l =', l.size())
    print('z =', z.size())
    print('decoded', tensor_to_string(decoded))
