# Author: Nadja Rhodes
# License: BSD 3-Clause
# Modified from Kyle Kastner's example here:
# https://github.com/kastnerkyle/pytorch-text-vae
import time
import os
try:
    import Queue
except ImportError:
    import queue as Queue
import multiprocessing as mp
import dill as pickle

import numpy as np
import re
import sys
import unidecode
import unicodedata
import collections

SOS_token = 0
EOS_token = 1
UNK_token = 2
N_CORE = 24

class Dataset:
    from enum import Enum
    class DataType(Enum):
        DEFAULT = 1
        JSON = 2
        QUILT = 3

    def __init__(self, filename):
        self.filename = filename

        if filename.endswith('.quilt'):
            self.data_type = self.DataType.QUILT
        elif filename.endswith('.json'):
            self.data_type = self.DataType.JSON
        else:
            self.data_type = self.DataType.DEFAULT

    def __iter__(self):
        if self.data_type == self.DataType.QUILT:
            return self.read_quilt_gen()
        elif self.data_type == self.DataType.JSON:
            return self.read_json_gen()
        else:
            return self.read_file_line_gen()

    def read_file_line_gen(self):
        with open(self.filename) as f:
            for line in f:
                yield unidecode.unidecode(line)

    def encode_genres(self, genres):
        e = np.zeros(len(self.genre_set) + 1)
        for g in genres:
            if g in self.genre_to_idx:
                e[self.genre_to_idx[g]] = 1
            else:
                # for unknown genres
                e[len(e) - 1] = 1
        return e

    def decode_genres(self, tensor):
        genres = []
        for i, x in enumerate(tensor.squeeze()):
            if x.item() == 1:
                genres.append(self.idx_to_genre[i])
        return genres

    def read_json_gen(self):
        import pandas as pd
        df = pd.read_json(self.filename)

        self.genre_set = set([g for gg in df.spotify_genres for g in gg])
        self.genre_to_idx = {unique_g: i for i, unique_g in enumerate(sorted(self.genre_set))}
        self.idx_to_genre = {i: unique_g for i, unique_g in enumerate(sorted(self.genre_set))}

        for i, row in df.iterrows():
            gs = self.encode_genres(row.spotify_genres)
            for sent in row.content_sentences:
                yield sent, gs

    def read_quilt_gen(self):
        # TODO: segmentation fault (core dumped) issue

        # read config of format:
            # PACKAGE
            # MODULE_NAME
            # NODE_NAME

        configs = []
        with open(self.filename) as f:
            for line in f:
                configs.append(line.strip())

        if len(configs) != 3:
            print('ERROR: invalid .quilt config file. Expecting 3 lines with PACKAGE, MODULE_NAME, and NODE_NAME.')
            sys.exit()

        import quilt
        pkg_name = f'{configs[0].split(".")[-1]}/{configs[1]}'
        quilt.install(pkg_name, force=True) # overwrites local install

        from importlib import import_module
        import_module(f'{configs[0]}.{configs[1]}') # e.g., import_module('quilt.data.iconix.deephypebot')

        # e.g., df = quilt.data.iconix.deephypebot.reviews_and_metadata_5yrs()
        df = getattr(sys.modules[f'{configs[0]}.{configs[1]}'], configs[2])()

        print(df.head())
        sys.exit()


norvig_list = None
# http://norvig.com/ngrams/count_1w.txt
# TODO: replace with spacy tokenization? or is it better to stick to common words?
'''Things turned to UNK:
- numbers
'''
def get_vocabulary(tmp_path):
    global norvig_list
    global reverse_norvig_list
    if norvig_list == None:
        with open(os.path.join(tmp_path, "count_1w.txt")) as f:
            r = f.readlines()
        norvig_list = [tuple(ri.strip().split("\t")) for ri in r]
    return norvig_list


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize(u'NFD', s)
        if unicodedata.category(c) != u'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"'", r"", s)
    s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"[^\w]", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip().lstrip().rstrip()
    return s


class Lang:
    def __init__(self, name, tmp_path, vocabulary_size=-1, reverse=False):
        self.name = name
        if reverse:
            self.vocabulary = [w[::-1] for w in ["SOS", "EOS", "UNK"]] + [w[0][::-1] for w in get_vocabulary(tmp_path)]
        else:
            self.vocabulary = ["SOS", "EOS", "UNK"] + [w[0] for w in get_vocabulary(tmp_path)]

        if vocabulary_size < 0:
            vocabulary_size = len(self.vocabulary)

        self.reverse = reverse
        self.vocabulary_size = vocabulary_size
        if vocabulary_size < len(self.vocabulary):
            print(f"Trimming vocabulary size from {len(self.vocabulary)} to {vocabulary_size}")
        else:
            print(f"Vocabulary size: {vocabulary_size}")
        self.vocabulary = self.vocabulary[:vocabulary_size]
        self.word2index = {v: k for k, v in enumerate(self.vocabulary)}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words = len(self.vocabulary) # Count SOS, EOS, UNK
        # dict.keys() do not pickle in Python 3.x - convert to list
        # https://groups.google.com/d/msg/pyomo-forum/XOf6zwvEbt4/ZfkbHzvDBgAJ
        self.words = list(self.word2index.keys())
        self.indices = list(self.index2word.keys())

    def index_to_word(self, index):
        try:
            return self.index2word[index.item()]
        except KeyError:
            return self.index2word[self.word2index[self.vocabulary[UNK_token]]]

    def word_to_index(self, word):
        try:
            return self.word2index[word.lower()]
        except KeyError:
            #print(f"[WARNING] {word.lower()}")
            return self.word2index[self.vocabulary[UNK_token]]

    def word_check(self, word):
        if word in self.word2index.keys():
            return word
        else:
            return self.word2index[self.vocabulary[UNK_token]]

    def process_sentence(self, sentence, normalize=True):
        if normalize:
            s = normalize_string(sentence)
        else:
            s = sentence
        return " ".join([w if w in self.words else self.word2index[self.vocabulary[UNK_token]] for w in s.split(" ")])

def filter_pair(p):
    return MIN_LENGTH < len(p[0].split(' ')) < MAX_LENGTH and MIN_LENGTH < len(p[1].split(' ')) < MAX_LENGTH


def process_input_side(s):
    return " ".join([WORDS[w] for w in s.split(" ")])


def process_output_side(s):
    return " ".join([REVERSE_WORDS[w] for w in s.split(" ")])


WORDS = None
REVERSE_WORDS = None

def unk_func():
    return "UNK"

def _get_line(data_type, elem):
    # JSON data can come with extra conditional info
    if data_type == Dataset.DataType.JSON:
        line = elem[0]
    else:
        line = elem

    return line

def _setup(path, vocabulary_size):
    global WORDS
    global REVERSE_WORDS
    wc = collections.Counter()
    dataset = Dataset(path)
    for n, elem in enumerate(iter(dataset)):
        if n % 100000 == 0:
            print("Fetching vocabulary from line {}".format(n))
            print("Current word count {}".format(len(wc.keys())))

        line = _get_line(dataset.data_type, elem)

        l = line.strip().lstrip().rstrip()
        if MIN_LENGTH < len(l.split(' ')) < MAX_LENGTH:
            l = normalize_string(l)
            WORDS = l.split(" ")
            wc.update(WORDS)
        else:
            continue

    the_words = ["SOS", "EOS", "UNK"]
    the_reverse_words = [w[::-1] for w in ["SOS", "EOS", "UNK"]]
    the_words += [wi[0] for wi in wc.most_common()[:vocabulary_size - 3]]
    the_reverse_words += [wi[0][::-1] for wi in wc.most_common()[:vocabulary_size - 3]]

    WORDS = collections.defaultdict(unk_func)
    REVERSE_WORDS = collections.defaultdict(unk_func)
    for k in range(len(the_words)):
        WORDS[the_words[k]] = the_words[k]
        REVERSE_WORDS[the_reverse_words[k]] = the_reverse_words[k]


def proc_line(line, reverse):
    if len(line.strip()) == 0:
        return None
    else:
        l = line.strip().lstrip().rstrip()
        # try to bail as early as possible to minimize processing
        if MIN_LENGTH < len(l.split(' ')) < MAX_LENGTH:
            l = normalize_string(l)
            l2 = l
            pair = (l, l2)

            if filter_pair(pair):
                if reverse:
                    pair = (l, "".join(list(reversed(l2))))
                p0 = process_input_side(pair[0])
                p1 = process_output_side(pair[1])
                return (p0, p1)
            else:
                return None
        else:
            return None


def process(q, oq, iolock):
    while True:
        stuff = q.get()
        if stuff is None:
            break
        r = [(proc_line(s[0], True), s[1]) if isinstance(s, tuple) else proc_line(s, True) for s in stuff]
        r = [ri for ri in r if ri != None and ri[0] != None]
        # flatten any tuples
        r = [ri[0] + (ri[1], ) if isinstance(ri, tuple) else ri for ri in r]
        if len(r) > 0:
            oq.put(r)


# https://stackoverflow.com/questions/43078980/python-multiprocessing-with-generator
def prepare_pair_data(path, vocabulary_size, tmp_path, min_length, max_length, reverse=False):
    global MIN_LENGTH
    global MAX_LENGTH
    MIN_LENGTH, MAX_LENGTH = min_length, max_length

    print("Reading lines...")
    print(f'MIN_LENGTH: {MIN_LENGTH}; MAX_LENGTH: {MAX_LENGTH}')
    pkl_path = path.split(os.sep)[-1].split(".")[0] + "_vocabulary.pkl"
    vocab_cache_path = os.path.join(tmp_path, pkl_path)
    global WORDS
    global REVERSE_WORDS
    if not os.path.exists(vocab_cache_path):
        print("Vocabulary cache {} not found".format(vocab_cache_path))
        print("Prepping vocabulary")
        _setup(path, vocabulary_size)
        with open(vocab_cache_path, "wb") as f:
            pickle.dump((WORDS, REVERSE_WORDS), f)
    else:
        print("Vocabulary cache {} found".format(vocab_cache_path))
        print("Loading...".format(vocab_cache_path))
        with open(vocab_cache_path, "rb") as f:
            r = pickle.load(f)
        WORDS = r[0]
        REVERSE_WORDS = r[1]
    print("Vocabulary prep complete")


    # don't use these for processing, but pass for ease of use later on
    input_side = Lang("in", tmp_path, vocabulary_size)
    output_side = Lang("out", tmp_path, vocabulary_size, reverse)

    print("Setting up queues")
    # some nasty multiprocessing
    # ~ 40 per second was the single core number
    q = mp.Queue(maxsize=1000000 * N_CORE)
    oq = mp.Queue(maxsize=1000000 * N_CORE)
    print("Queue setup complete")
    print("Getting lock")
    iolock = mp.Lock()
    print("Setting up pool")
    pool = mp.Pool(N_CORE, initializer=process, initargs=(q, oq, iolock))
    print("Pool setup complete")

    start_time = time.time()
    pairs = []
    last_empty = time.time()

    curr_block = []
    block_size = 1000
    last_send = 0
    # takes ~ 30s to get a block done
    empty_wait = 2
    avg_time_per_block = 30
    status_every = 100000
    print("Starting block processing")
    dataset = Dataset(path)
    for n, elem in enumerate(iter(dataset)):
        curr_block.append(elem)
        if len(curr_block) > block_size:
            # this could block, oy
            q.put(curr_block)
            curr_block = []

        if last_empty < time.time() - empty_wait:
            try:
                while True:
                    with iolock:
                        r = oq.get(block=True, timeout=.0001)
                    pairs.extend(r)
            except:
                last_empty = time.time()
        if n % status_every == 0:
            with iolock:
                print("Queued line {}".format(n))
                tt = time.time() - start_time
                print("Elapsed time {}".format(tt))
                tl = len(pairs)
                print("Total lines {}".format(tl))
                avg_time_per_block = max(30, block_size * (tt / (tl + 1)))
                print("Approximate lines / s {}".format(tl / tt))
    # finish the queue
    q.put(curr_block)
    print("Finalizing line processing")
    for _ in range(N_CORE):  # tell workers we're done
        q.put(None)
    empty_checks = 0
    prev_len = len(pairs)
    last_status = time.time()
    print("Total lines {}".format(len(pairs)))
    while True:
        if empty_checks > 10:
            break
        if status_every < (len(pairs) - prev_len) or last_status < time.time() - empty_wait:
            print("Total lines {}".format(len(pairs)))
            prev_len = len(pairs)
            last_status = time.time()
        if not oq.empty():
            try:
                while True:
                    with iolock:
                        r = oq.get(block=True, timeout=.0001)
                    pairs.extend(r)
                    empty_checks = 0
            except:
                # Queue.Empty
                pass
        elif oq.empty():
            empty_checks += 1
            time.sleep(empty_wait)
    print("Line processing complete")
    print("Final line count {}".format(len(pairs)))
    pool.close()
    pool.join()
    print("Pair preparation complete")
    return input_side, output_side, pairs, dataset
