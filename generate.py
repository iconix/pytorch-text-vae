import argparse
import dill as pickle
import numpy as np
import os
from pathlib import Path
import time
import torch

import model
from datasets import EOS_token

from pdb import set_trace

if __name__ == "__main__":

    '''if len(sys.argv) < 3 or not sys.argv[1].endswith("_stored_info.pkl") or not sys.argv[2].endswith(".pt"):
        print("Usage: python generate.py [modelname]_stored_info.pkl [modelname].pt <optional args>")
        sys.exit(1)'''

    parser = argparse.ArgumentParser(description='pytorch-text-vae:generate')
    parser.add_argument('saved_vae', metavar='SAVED_VAE', help='saved PyTorch vae model')
    parser.add_argument('stored_info', metavar='STORED_INFO', help='pkl of stored info')
    parser.add_argument('--cache-path', default=str(Path('tmp/')), metavar='CACHE',
                        help='cache path (default: tmp/)')
    parser.add_argument('--max-length', type=int, default=50, metavar='LEN',
                        help='max num words per sample (default: 50)')
    parser.add_argument('--num-sample', type=int, default=10, metavar='NS',
                        help='num samplings (default: 10)')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='seed for random number generator (default: None)')
    parser.add_argument('--temp', type=float, default=0.75, metavar='T',
                        help='sample temperature (default: 0.75)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')

    args = parser.parse_args()

    args_passed = locals()['args']
    print(args_passed)

    stored_info = args.stored_info.split(os.sep)[-1]
    cache_file =  os.path.join(args.cache_path, stored_info)

    start_load = time.time()
    print("Fetching cached info at {}".format(cache_file))
    with open(cache_file, "rb") as f:
        input_side, output_side, pairs, EMBED_SIZE, DECODER_HIDDEN_SIZE, ENCODER_HIDDEN_SIZE, N_ENCODER_LAYERS = pickle.load(f)
    end_load = time.time()
    print("Cache {} loaded, total load time {}".format(cache_file, end_load - start_load))

    saved_vae = args.saved_vae.split(os.sep)[-1]
    if os.path.exists(args.saved_vae):
        print("Found saved model {}".format(saved_vae))

        if saved_vae.endswith("_state.pt"):
            e = model.EncoderRNN(input_side.n_words, ENCODER_HIDDEN_SIZE, EMBED_SIZE, N_ENCODER_LAYERS, bidirectional=True)
            d = model.DecoderRNN(EMBED_SIZE, DECODER_HIDDEN_SIZE, input_side.n_words, 1, word_dropout=0)
            vae = model.VAE(e, d)
            if args.use_cuda:
                vae = vae.cuda()
            vae.load_state_dict(torch.load(saved_vae))
        else:
            vae = torch.load(saved_vae)
            print("Trained for {} steps".format(vae.steps_seen))

        print("Setting new random seed")
        if args.seed is None:
            # TODO: torch.manual_seed(1999) in model.py is affecting this
            new_seed = int(time.time())
            new_seed = abs(new_seed) % 4294967295 # must be between 0 and 4294967295
        else:
            new_seed = args.seed
        torch.manual_seed(new_seed)

        for i in range(args.num_sample):
            z = torch.randn(EMBED_SIZE).unsqueeze(0)
            if args.use_cuda:
                z = z.cuda()
            generated = vae.decoder.generate(z, args.max_length, args.temp, args.use_cuda)
            generated_str = model.float_word_tensor_to_string(output_side, generated)

            EOS_str = f' {output_side.index_to_word(torch.LongTensor([EOS_token]))} '

            if generated_str.endswith(EOS_str):
                generated_str = generated_str[:-5]

            # flip it back
            generated_str = generated_str[::-1]

            print('---')
            print(generated_str)
