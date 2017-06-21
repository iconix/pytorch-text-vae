# Author: Kyle Kastner
# License: BSD 3-Clause
# Modified from Sean Robertson's example here:
# https://github.com/spro/pytorch-text-vae
import time
import os
try:
    import Queue
except ImportError:
    import queue as Queue
import multiprocessing as mp
import cPickle as pickle

import numpy as np
import re
import unidecode
import unicodedata
import collections

USE_CUDA = True
SOS_token = 0
EOS_token = 1
UNK_token = 2
N_CORE = 24


def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


def read_file_line_gen(filename):
    with open(filename) as f:
        for line in f:
            yield unidecode.unidecode(line)


norvig_list = None
# http://norvig.com/ngrams/count_1w.txt
def get_vocabulary():
    global norvig_list
    global reverse_norvig_list
    if norvig_list == None:
        with open("count_1w.txt") as f:
            r = f.readlines()
        norvig_list = [tuple(ri.strip().split("\t")) for ri in r]
    return norvig_list


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize(u'NFD', unicode(s))
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
    def __init__(self, name, vocabulary_size, reverse=False):
        self.name = name
        if reverse:
            self.vocabulary = ["SOS", "EOS", "UNK"] + [w[0][::-1] for w in get_vocabulary()]
        else:
            self.vocabulary = ["SOS", "EOS", "UNK"] + [w[0] for w in get_vocabulary()]

        self.reverse = reverse
        self.vocabulary_size = vocabulary_size
        self.vocabulary = self.vocabulary[:vocabulary_size]
        self.word2index = {v: k for k, v in enumerate(self.vocabulary)}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words = len(self.vocabulary) # Count SOS, EOS, UNK
        self.words = self.word2index.keys()
        self.indices = self.index2word.keys()

    def index_to_word(self, index):
        try:
            return self.index2word[index]
        except KeyError:
            return self.index2word[self.word2index["UNK"]]

    def word_to_index(self, word):
        try:
            return self.word2index[word.lower()]
        except KeyError:
            return self.word2index["UNK"]

    def word_check(self, word):
        if word in self.word2index.keys():
            return word
        else:
            return "UNK"

    def process_sentence(self, sentence, normalize=True):
        if normalize:
            s = normalize_string(sentence)
        else:
            s = sentence
        return " ".join([w if w in self.words else "UNK" for w in s.split(" ")])


MIN_LENGTH = 5
MAX_LENGTH = 15


def filter_pair(p):
    return MIN_LENGTH < len(p[0].split(' ')) < MAX_LENGTH and MIN_LENGTH < len(p[1].split(' ')) < MAX_LENGTH


def process_input_side(s):
    return " ".join([words[w] for w in s.split(" ")])


def process_output_side(s):
    return " ".join([reverse_words[w] for w in s.split(" ")])


words = None
reverse_words = None

def unk_func():
    return "UNK"

def _setup(filepath, vocabulary_size, csv):
    global words
    global reverse_words
    wc = collections.Counter()
    for n, line in enumerate(read_file_line_gen(filepath)):
        if n % 100000 == 0:
            print("Fetching vocabulary from line {}".format(n))
            print("Current word count {}".format(len(wc.keys())))
        l = line.strip().lstrip().rstrip()
        if MIN_LENGTH < len(l.split(' ')) < MAX_LENGTH:
            l = normalize_string(l)
            words = l.split(" ")
            wc.update(words)
        else:
            continue
    the_words = ["SOS", "EOS", "UNK"]
    the_reverse_words = ["SOS", "EOS", "UNK"]
    the_words += [wi[0] for wi in wc.most_common()[:vocabulary_size - 3]]
    the_reverse_words += [wi[0][::-1] for wi in wc.most_common()[:vocabulary_size - 3]]

    words = collections.defaultdict(unk_func)
    reverse_words = collections.defaultdict(unk_func)
    for k in range(len(the_words)):
        words[the_words[k]] = the_words[k]
        reverse_words[the_reverse_words[k]] = the_reverse_words[k]


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
        r = [proc_line(s, True) for s in stuff]
        r = [ri for ri in r if ri != None]
        if len(r) > 0:
            oq.put(r)


# https://stackoverflow.com/questions/43078980/python-multiprocessing-with-generator
def prepare_pair_data(path, vocabulary_size, reverse=False, csv=False):
    print("Reading lines...")
    pkl_path = path.split(os.sep)[-1].split(".")[0] + "_vocabulary.pkl"
    vocab_cache_path = "/Tmp/kastner/" + pkl_path
    global words
    global reverse_words
    if not os.path.exists(vocab_cache_path):
        print("Vocabulary cache {} not found".format(vocab_cache_path))
        print("Prepping vocabulary")
        # Read the file and split into lines
        #f, flen = read_file(path)
        _setup(path, vocabulary_size, csv)
        with open(vocab_cache_path, "wb") as f:
            pickle.dump((words, reverse_words), f)
    else:
        print("Vocabulary cache {} found".format(vocab_cache_path))
        print("Loading...".format(vocab_cache_path))
        with open(vocab_cache_path, "rb") as f:
            r = pickle.load(f)
        words = r[0]
        reverse_words = r[1]
    print("Vocabulary prep complete")


    # don't use these for processing, but pass for ease of use later on
    input_side = Lang("in", vocabulary_size)
    output_side = Lang("out", vocabulary_size, reverse)

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
    for n, line in enumerate(read_file_line_gen(path)):
        curr_block.append(line)
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
    return input_side, output_side, pairs
