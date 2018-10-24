import dill as pickle
import numpy as np
import os
from pathlib import Path
import time
import torch

if __package__ is None or __package__ == '':
    # uses current directory visibility
    import model
    from datasets import EOS_token
else:
    # uses current package visibility
    import pytorchtextvae.model as model
    from pytorchtextvae.datasets import EOS_token

def load_model(saved_vae, stored_info, device, cache_path=str(Path('../tmp')), seed=None):
    stored_info = stored_info.split(os.sep)[-1]
    cache_file =  os.path.join(cache_path, stored_info)

    start_load = time.time()
    print(f"Fetching cached info at {cache_file}")
    with open(cache_file, "rb") as f:
        dataset, z_size, condition_size, condition_on, decoder_hidden_size, encoder_hidden_size, n_encoder_layers = pickle.load(f)
    end_load = time.time()
    print(f"Cache {cache_file} loaded (load time: {end_load - start_load:.2f}s)")

    if os.path.exists(saved_vae):
        print(f"Found saved model {saved_vae}")
        start_load_model = time.time()

        e = model.EncoderRNN(dataset.input_side.n_words, encoder_hidden_size, z_size, n_encoder_layers, bidirectional=True)
        d = model.DecoderRNN(z_size, dataset.trn_split.n_conditions, condition_size, decoder_hidden_size, dataset.input_side.n_words, 1, word_dropout=0)
        vae = model.VAE(e, d).to(device)
        vae.load_state_dict(torch.load(saved_vae, map_location=lambda storage, loc: storage))
        vae.eval()
        print(f"Trained for {vae.steps_seen} steps (load time: {time.time() - start_load_model:.2f}s)")

        print("Setting new random seed")
        if seed is None:
            # TODO: torch.manual_seed(1999) in model.py is affecting this
            new_seed = int(time.time())
            new_seed = abs(new_seed) % 4294967295 # must be between 0 and 4294967295
        else:
            new_seed = seed
        torch.manual_seed(new_seed)

        random_state = np.random.RandomState(new_seed)
        #random_state.shuffle(dataset.trn_pairs)

    return vae, dataset, z_size, random_state

def generate(vae, dataset, z_size, random_state, device, condition_inputs=None, max_length=50, num_sample=10, temp=0.75, print_z=False):
    gens = []
    zs = []
    conditions = []

    if dataset.trn_split.n_conditions > -1 and condition_inputs is not None and not isinstance(condition_inputs, list):
        print(f'[WARNING] condition_inputs provided is of type "{type(condition_inputs).__name__}" but should be of type "list". Continuing with random condition_inputs...')

    for i in range(num_sample):
        z = torch.randn(z_size).unsqueeze(0).to(device)

        if dataset.trn_split.n_conditions > -1:
            if isinstance(condition_inputs, list):
                condition = torch.tensor(dataset.encode_conditions(condition_inputs), dtype=torch.float).unsqueeze(0).to(device)
            else:
                condition = model.random_training_set(dataset, random_state, device)[2]
        else:
            condition = None

        generated = vae.decoder.generate(z, condition, max_length, temp, device)
        generated_str = model.float_word_tensor_to_string(dataset.output_side, generated)

        EOS_str = f' {dataset.output_side.index_to_word(torch.LongTensor([EOS_token]))} '

        if generated_str.endswith(EOS_str):
            generated_str = generated_str[:-5]

        # flip it back
        generated_str = generated_str[::-1]

        print('---')
        if dataset.trn_split.n_conditions > -1:
            print(dataset.decode_conditions(condition))
        print(generated_str)
        gens.append(generated_str)
        zs.append(z)
        if dataset.trn_split.n_conditions > -1:
            conditions.append(condition)
        if print_z:
            print(z)

    return gens, zs, conditions

def run(saved_vae, stored_info, cache_path=str(Path(f'..{os.sep}tmp')), condition_inputs=None, max_length=50, num_sample=10, seed=None, temp=0.75,
            use_cuda=True, print_z=False):

    args_passed = locals()
    print(args_passed)

    DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() and use_cuda else 'cpu')

    with torch.no_grad():
        vae, dataset, z_size, random_state = load_model(saved_vae, stored_info, DEVICE, cache_path, seed)
        gens, zs, conditions = generate(vae, dataset, z_size, random_state, DEVICE, condition_inputs, max_length, num_sample, temp, print_z)

if __name__ == "__main__":
    import fire; fire.Fire(run)
