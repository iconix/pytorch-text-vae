import dill as pickle
import numpy as np
import os
import shutil

from datasets import get_vocabulary, prepare_pair_data
from model import *

def train_vae(data_path, tmp_path=f'..{os.sep}tmp',
                encoder_hidden_size=512, n_encoder_layers=2, decoder_hidden_size=512, z_size=128,
                condition_size=16, max_vocab=-1, lr=0.0001, n_steps=1500000, grad_clip=10.0,
                save_every=None, log_every_n_seconds=5*60, log_every_n_steps=1000,
                kld_start_inc=10000, habits_lambda=0.2,
                word_dropout=0.25, temperature=1.0, temperature_min=0.75,
                use_cuda=True, generate_samples=True, generate_interpolations=True, min_gen_len=10, max_gen_len=200):

    args_passed = locals()
    print(args_passed)

    DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() and use_cuda else 'cpu')
    if 'cuda' in DEVICE.type:
        print('Using CUDA!')

    # should get to the temperature around 50% through training, then hold
    temperature_dec = (temperature - temperature_min) / (0.5 * n_steps)
    if save_every is None:
        save_every = log_every_n_steps


    filename = data_path.split(os.sep)[-1].split(".")[0]
    if data_path.endswith(".pkl"):
        cache_path = os.path.join(*data_path.split(os.sep)[:-1])
        cache_file =  os.path.join(cache_path, filename + ".pkl")
    else:
        cache_path = tmp_path
        cache_file = os.path.join(cache_path, filename + "_stored_info.pkl")

    if not os.path.exists(cache_file):
        print("Cached info at {} not found".format(cache_file))
        print("Creating cache... this may take some time")

        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        input_side, output_side, pairs, dataset = prepare_pair_data(data_path, max_vocab, tmp_path, min_gen_len, max_gen_len, reverse=True)

        with open(cache_file, "wb") as f:
            pickle.dump((input_side, output_side, pairs, dataset, z_size, condition_size, decoder_hidden_size, encoder_hidden_size, n_encoder_layers), f)
    else:
        start_load = time.time()
        print("Fetching cached info at {}".format(cache_file))
        with open(cache_file, "rb") as f:
            input_side, output_side, pairs, dataset, z_size, condition_size, decoder_hidden_size, encoder_hidden_size, n_encoder_layers = pickle.load(f)
        end_load = time.time()
        print(f"Cache {cache_file} loaded (load time: {end_load - start_load:.2f}s)")

    print("Shuffling training data")
    random_state = np.random.RandomState(1999)
    random_state.shuffle(pairs)

    print("Initializing model")
    n_words = input_side.n_words
    e = EncoderRNN(n_words, encoder_hidden_size, z_size, n_encoder_layers, bidirectional=True).to(DEVICE)

    # custom weights initialization # TODO: should we do this if using saved_vae?
    def rnn_weights_init(m):
        for c in m.children():
            classname = c.__class__.__name__
            if classname.find("GRU") != -1:
                for k, v in c.named_parameters():
                    if "weight" in k:
                        v.data.normal_(0.0, 0.02)

    d = DecoderRNN(z_size, len(dataset.genre_set) + 1, condition_size, decoder_hidden_size, n_words, 1, word_dropout=word_dropout).to(DEVICE)
    rnn_weights_init(d)

    vae = VAE(e, d, n_steps).to(DEVICE)
    saved_vae = filename + "_state.pt"
    if os.path.exists(saved_vae):
        start_load_model = time.time()
        print("Found saved model {}, continuing...".format(saved_vae))
        shutil.copyfile(saved_vae, saved_vae + ".bak")
        vae.load_state_dict(torch.load(saved_vae))
        print(f"Found model was already trained for {vae.steps_seen} steps (load time: {time.time() - start_load_model:.2f}s)")
        print(f'kld_max: {vae.kld_max}; kld_weight: {vae.kld_weight}')
        temperature = temperature_min
        temperature_min = temperature_min
        temperature_dec = 0.

        print("Setting new random seed")
        # change random seed and reshuffle the data, so that we don't repeat the same
        # use hash of the weights and biases? try with float16 to avoid numerical issues in the tails...
        new_seed = hash(tuple([hash(tuple(vae.state_dict()[k].cpu().numpy().ravel().astype("float16"))) for k, v in vae.state_dict().items()]))
        # must be between 0 and 4294967295
        new_seed = abs(new_seed) % 4294967295
        print(new_seed)
        random_state = np.random.RandomState(new_seed)
        print("Reshuffling training data")
        random_state.shuffle(pairs)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    criterion.to(DEVICE)

    def save():
        save_state_filename = filename + '_state.pt'
        torch.save(vae.state_dict(), save_state_filename)
        print('Saved as %s' % (save_state_filename))

    try:
        # set it so that the first one logs
        start_time = time.time()
        last_log_time = time.time() - log_every_n_seconds
        last_log_step = -log_every_n_steps - 1
        start_steps = vae.steps_seen
        for step in range(start_steps, n_steps):
            input, target, condition = random_training_set(pairs, input_side, output_side, random_state, DEVICE)
            optimizer.zero_grad()

            m, l, z, decoded = vae(input, target, condition, DEVICE, temperature)
            if temperature > temperature_min:
                temperature -= temperature_dec

            ll_loss = criterion(decoded, target)

            KLD = -0.5 * (2 * l - torch.pow(m, 2) - torch.pow(torch.exp(l), 2) + 1)
            # ha bits , like free bits but over whole layer
            clamp_KLD = torch.clamp(KLD.mean(), min=habits_lambda).squeeze()
            loss = ll_loss + clamp_KLD * vae.kld_weight

            loss.backward()

            if step > kld_start_inc and vae.kld_weight < vae.kld_max:
                vae.kld_weight += vae.kld_inc

            ec = torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            optimizer.step()

            def log_and_generate(tag, value):
                if tag == "step":
                    print('|%s|[%d] %.4f (k=%.4f, t=%.4f, kl=%.4f, ckl=%.4f,  nll=%.4f, ec=%.4f)' % (
                        tag, value, loss.item(), vae.kld_weight, temperature, KLD.data.mean(), clamp_KLD.item(), ll_loss.item(), ec
                    ))
                    with open('plots.txt', 'a') as f:
                        f.write(f'{value}\t{loss.item()}\t{ll_loss.item()}\t{KLD.data.mean()}\n')
                elif tag == "time":
                    print('|%s|[%.4f] %.4f (k=%.4f, t=%.4f, kl=%.4f, ckl=%.4f, nll=%.4f,  ec=%.4f)' % (
                        tag, value, loss.item(), vae.kld_weight, temperature, KLD.data.mean(), clamp_KLD.item(), ll_loss.item(), ec
                    ))

                EOS_str = f' {output_side.index_to_word(torch.LongTensor([EOS_token]))} '

                if generate_samples:
                    rand_z = torch.randn(z_size).unsqueeze(0).to(DEVICE)
                    fixed_condition = torch.FloatTensor(dataset.encode_genres(['vapor soul'])).to(DEVICE)

                    generated = vae.decoder.generate(rand_z, fixed_condition, max_gen_len, temperature, DEVICE)
                    generated_str = float_word_tensor_to_string(output_side, generated)

                    if generated_str.endswith(EOS_str):
                        generated_str = generated_str[:-5]

                    # flip it back
                    print('----')
                    print('    (sample {}) "{}"'.format(tag, generated_str[::-1]))

                if generate_interpolations:
                    inp_str = long_word_tensor_to_string(input_side, input)
                    print('----')
                    print('    (input/target {}) "{}"'.format(tag, inp_str))

                    generated = vae.decoder.generate(z, condition, max_gen_len, temperature, DEVICE)
                    generated_str = float_word_tensor_to_string(output_side, generated)
                    if generated_str.endswith(EOS_str):
                        generated_str = generated_str[:-5]

                    # flip it back
                    print('    (interpolation {}) "{}"'.format(tag, generated_str[::-1]))
                    print('----')

            if last_log_time <= time.time() - log_every_n_seconds:
                log_and_generate("time", time.time() - start_time)
                last_log_time = time.time()

            if last_log_step <= step - log_every_n_steps:
                log_and_generate("step", step)
                last_log_step = step

            if step > 0 and step % save_every == 0 or step == (n_steps - 1):
                vae.steps_seen = torch.tensor(step, dtype=torch.long).to(DEVICE)
                save()

        save()

    except KeyboardInterrupt as err:
        print("ERROR", err)
        print("Saving before quit...")
        vae.steps_seen = torch.tensor(step, dtype=torch.long).to(DEVICE)
        save()

if __name__ == "__main__":
    import fire; fire.Fire(train_vae)
