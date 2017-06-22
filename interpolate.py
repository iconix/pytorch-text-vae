# slerp, lerp and associated from Tom White in plat (https://github.com/dribnet/plat)
from model import *
import numpy as np
import sys
from scipy.stats import norm

# Don't need it for sampling
USE_CUDA = True

vae = torch.load('vae_save.pt')
vae.train(False)

TEMPERATURE = .0001
N_SAMPLES = 10
N_STEPS = 10

torch.manual_seed(420)
random_state = np.random.RandomState(2179)

if len(sys.argv) < 2:
    print("Usage: python interpolate.py [filename]")
    sys.exit(1)

reverse = True
csv = False
if sys.argv[1].endswith(".csv"):
    csv = True

if sys.argv[1].endswith(".pkl"):
    cache_path = sys.argv[1]
    lang_cache_path = sys.argv[1].split(os.sep)[-1].split(".")[0] + "_stored_lang.pkl"
else:
    tmp_path = "/Tmp/kastner/"
    cache_path = tmp_path + sys.argv[1].split(os.sep)[-1].split(".")[0] + "_stored_info.pkl"
    lang_cache_path = tmp_path + sys.argv[1].split(os.sep)[-1].split(".")[0] + "_stored_lang.pkl"

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


for s in range(1, N_SAMPLES):
    sent0 = "there is no one else in the world"
    sent1 = "then i turned to him"
    #sent1 = "he was silent for a moment"
    #sent2 = "it was my turn"

    sent0 = input_side.process_sentence(sent0)
    sent1 = input_side.process_sentence(sent1)

    if True:
        z0 = encode_sample(sent0, False)
        z1 = encode_sample(sent1, False)
    else:
        z0 = encode_sample()
        z1 = encode_sample()
        sent1 = None
        sent2 = None
    z0_np = z0.cpu().data.numpy().ravel()
    z1_np = z1.cpu().data.numpy().ravel()
    last_s = ''

    generated_str = float_word_tensor_to_string(output_side, vae.decoder.generate(z0,  MAX_LENGTH, TEMPERATURE))
    if generated_str.endswith("EOS "):
        generated_str = generated_str[:-4]
    generated_str = generated_str[::-1]

    end_str = float_word_tensor_to_string(output_side, vae.decoder.generate(z1,  MAX_LENGTH, TEMPERATURE))
    if end_str.endswith("EOS "):
        end_str = end_str[:-4]
    end_str = end_str[::-1]

    if sent0 is not None:
        print('(s0)', sent0)
    print('(z0)', generated_str)

    last_s = generated_str

    for i in range(1, N_STEPS):
        t = i * 1.0 / N_STEPS

        sph_z = slerp(t, z0_np, z1_np)
        interp_z = Variable(torch.FloatTensor(sph_z[None]))
        if USE_CUDA:
            interp_z = interp_z.cuda()
        s = float_word_tensor_to_string(output_side, vae.decoder.generate(interp_z, MAX_LENGTH, TEMPERATURE))
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
