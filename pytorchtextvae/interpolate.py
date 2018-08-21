# slerp, lerp and associated from Tom White in plat (https://github.com/dribnet/plat)
from model import *
import numpy as np
import sys
from scipy.stats import norm
import argparse

default_data_path = "books_large_all_stored_info.pkl"
default_vae_path = "vae.pt"
default_temperature = 1.
default_n_samples = 10
default_length = 5
default_path = "slerp"
default_seed = 1999
default_s1 = None
default_s2 = None

parser = argparse.ArgumentParser(description="Interpolation tests for trained RNN-VAE",
#                                 epilog="Simple usage:\n    python minimal_beamsearch.py shakespeare_input.txt -o 10\nFull usage:\n    python minimal_beamsearch.py shakespeare_input.txt -o 10 -d 0 -s 'HOLOFERNES' -e 'crew?\\n' -r 2177",
                                  formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-f", "--filepath", help="Path to pickled dataset info\nDefault: {}".format(default_data_path), default=default_data_path)
parser.add_argument("-s", "--saved", help="Path to saved vae.pt file\nDefault: {}".format(default_vae_path), default=default_vae_path)
parser.add_argument("-l", "--length", help="Length of sample path\nDefault: {}".format(default_length), default=default_length)
parser.add_argument("-p", "--path", help="Path to use for sampling\nDefault: {}".format(default_path), default=default_path)
parser.add_argument("-r", "--seed", help="Random seed to use\nDefault: {}".format(default_seed), default=default_seed)
parser.add_argument("-n", "--nsamples", help="Number of samples\nDefault: {}".format(default_n_samples), default=default_n_samples)
parser.add_argument("-t", "--temperature", help="Temperature to use when sampling\nDefault: {}".format(default_temperature), default=default_temperature)
parser.add_argument("-1", "--s1", help="First sentence on the path, None for random\nDefault: {}".format(default_s1), default=default_s1)
parser.add_argument("-2", "--s2", help="Second sentence on the path, None for random\nDefault: {}".format(default_s2), default=default_s2)

args = parser.parse_args()
filepath = args.filepath
saved = args.saved
length = int(args.length)
path = args.path
seed = int(args.seed)
n_samples = int(args.nsamples)
temperature = float(args.temperature)
s1 = args.s1
s2 = args.s2

# Don't need it for sampling
USE_CUDA = True

vae = torch.load(saved)
vae.train(False)

torch.manual_seed(seed)
random_state = np.random.RandomState(seed)

reverse = True
csv = False

if filepath.endswith(".pkl"):
    cache_path = filepath
    lang_cache_path = filepath.split(os.sep)[-1].split(".")[0] + "_stored_lang.pkl"
else:
    raise ValueError("Must be a pkl file")

if not os.path.exists(cache_path):
    raise ValueError("Must have stored info already!")
else:
    if os.path.exists(lang_cache_path):
        start_load = time.time()
        print("Fetching cached language info at {}".format(lang_cache_path))
        with open(lang_cache_path, "rb") as f:
            input_side, output_side = pickle.load(f)
        end_load = time.time()
        print("Language only cache {} loaded, total load time {}".format(lang_cache_path, end_load - start_load))
    else:
        start_load = time.time()
        print("Fetching cached info at {}".format(cache_path))
        with open(cache_path, "rb") as f:
            input_side, output_side, pairs = pickle.load(f)
        end_load = time.time()
        print("Cache {} loaded, total load time {}".format(cache_path, end_load - start_load))

        with open(lang_cache_path, "wb") as f:
            pickle.dump((input_side, output_side), f)


def encode_sample(encode_sentence=None, stochastic=True):
    size = vae.encoder.output_size
    if encode_sentence is None:
        rm = Variable(torch.FloatTensor(1, size).normal_())
        rl = Variable(torch.FloatTensor(1, size).normal_())
    else:
        inp = word_tensor(input_side, encode_sentence)
        # temporary
        try:
            m, l, z = vae.encode(inp)
        except AttributeError:
            m, l, z = vae.encoder(inp)
        rm = m
        rl = l

    if USE_CUDA:
        rm = rm.cuda()
        rl = rl.cuda()

    if stochastic:
        z = vae.encoder.sample(rm, rl)
    return z



def lerp(val, low, high):
    """Linear interpolation"""
    return low + (high - low) * val


def lerp_gaussian(val, low, high):
    """Linear interpolation with gaussian CDF"""
    low_gau = norm.cdf(low)
    high_gau = norm.cdf(high)
    lerped_gau = lerp(val, low_gau, high_gau)
    return norm.ppf(lerped_gau)


def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


def slerp_gaussian(val, low, high):
    """Spherical interpolation with gaussian CDF (generally not useful)"""
    offset = norm.cdf(np.zeros_like(low))  # offset is just [0.5, 0.5, ...]
    low_gau_shifted = norm.cdf(low) - offset
    high_gau_shifted = norm.cdf(high) - offset
    circle_lerped_gau = slerp(val, low_gau_shifted, high_gau_shifted)
    epsilon = 0.001
    clipped_sum = np.clip(circle_lerped_gau + offset, epsilon, 1.0 - epsilon)
    result = norm.ppf(clipped_sum)
    return result


for s in range(1, n_samples):
    if s1 is None:
        sent0 = None
        z0 = encode_sample()
    else:
        sent0 = input_side.process_sentence(str(s1))
        z0 = encode_sample(sent0, False)

    if s2 is None:
        sent1 = None
        z1 = encode_sample()
    else:
        sent1 = input_side.process_sentence(str(s2))
        z1 = encode_sample(sent1, False)

    z0_np = z0.cpu().data.numpy().ravel()
    z1_np = z1.cpu().data.numpy().ravel()
    last_s = ''

    generated_str = float_word_tensor_to_string(output_side, vae.decoder.generate(z0, MAX_LENGTH, temperature))
    if generated_str.endswith("EOS "):
        generated_str = generated_str[:-4]
    generated_str = generated_str[::-1]

    end_str = float_word_tensor_to_string(output_side, vae.decoder.generate(z1, MAX_LENGTH, temperature))
    if end_str.endswith("EOS "):
        end_str = end_str[:-4]
    end_str = end_str[::-1]

    if sent0 is not None:
        print('(s0)', sent0)
    print('(z0)', generated_str)

    last_s = generated_str

    for i in range(1, length):
        t = i * 1.0 / length

        #sph_z = slerp(t, z0_np, z1_np)
        #sph_z = slerp_gaussian(t, z0_np, z1_np)
        sph_z = lerp(t, z0_np, z1_np)
        interp_z = Variable(torch.FloatTensor(sph_z[None]))
        if USE_CUDA:
            interp_z = interp_z.cuda()
        s = float_word_tensor_to_string(output_side, vae.decoder.generate(interp_z, MAX_LENGTH, temperature))
        generated_str = s
        if generated_str.endswith("EOS "):
            generated_str = generated_str[:-4]
        generated_str = generated_str[::-1]

        if generated_str != last_s and generated_str != end_str:
            print('  .)', generated_str)

        last_s = generated_str

    print('(z1)', end_str)
    if sent1 is not None:
        print('(s1)', sent1)
    print('\n')
