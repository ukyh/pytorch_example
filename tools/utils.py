import datetime
import sys
import json
from collections import Counter
from logging import getLogger, INFO, DEBUG, Formatter, StreamHandler, FileHandler


logger = getLogger(__name__)


def trace(*args):
    """ Simple logging. """
    print(datetime.datetime.now().strftime("%H:%M:%S"), " ".join(map(str, args)))


def decorate_logger(args, logger):
    """ Decorate logger. 
        Stream for debug and File for experimental logs.
    """
    logger.setLevel(DEBUG)
    formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # sys.stderr.write = logger.error

    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.log_path != "":
        f_handler = FileHandler(filename=args.log_path, mode="w", encoding="utf-8")
        f_handler.setLevel(INFO)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger


class Vocab:
    """ Defines a vocabulary and convert token <-> index.

    Attributes:
        min_freq: Minimum frequency to add in vocab.
                  Tokens with the lower frequency are treated as "<unk>".
    """
    def __init__(self, min_freq=-1):
        self.tok2idx_dict = {"<pad>":0, "<unk>":1, "<bos>":2, "<eos>":3}
        self.idx2tok_dict = {0:"<pad>", 1:"<unk>", 2:"<bos>", 3:"<eos>"}
        self._special_tokens = list()
        self._tokens = list()
        self.pad_idx = self.tok2idx_dict["<pad>"]
        self.unk_idx = self.tok2idx_dict["<unk>"]
        self.bos_idx = self.tok2idx_dict["<bos>"]
        self.eos_idx = self.tok2idx_dict["<eos>"]
        self.min_freq = min_freq

    def __len__(self):
        return len(self.tok2idx_dict)

    def add_special_token(self, sp_token):
        assert self._special_tokens != None, "Cannot add token after setting or loading vocab"
        assert type(sp_token) == str
        self._special_tokens.append(sp_token)
    
    def add_token(self, token_seq):
        assert self._tokens != None, "Cannot add token after setting or loading vocab"
        assert type(token_seq) == list, "token_seq must be a list of tokens e.g., ['this', 'is', 'an', 'example', '.']"
        self._tokens.extend(token_seq)

    def set_vocab(self):
        """ Set and fix the vocabulary. """
        assert self._special_tokens != None and self._tokens != None, "Vocab is already set or loaded"
        for sp_tok in self._special_tokens:
            self.tok2idx_dict[sp_tok] = len(self.tok2idx_dict)
            self.idx2tok_dict[len(self.idx2tok_dict)] = sp_tok
        sorted_tokens = Counter(self._tokens).most_common(None)
        for tok, freq in sorted_tokens:
            if freq < self.min_freq:
                break
            self.tok2idx_dict[tok] = len(self.tok2idx_dict)
            self.idx2tok_dict[len(self.idx2tok_dict)] = tok
        assert len(self.tok2idx_dict) == len(self.idx2tok_dict)
        self._special_tokens = None
        self._tokens = None
        logger.info("Set vocab: {}".format(len(self.tok2idx_dict)))

    def save_vocab(self, vocab_path):
        """ Save the vocabulary. """
        assert self._special_tokens == None and self._tokens == None, "Vocab is not set yet"
        with open(vocab_path, "w", encoding="utf-8", errors="ignore") as outfile:
            json.dump(self.tok2idx_dict, outfile, indent=4)
        logger.info("Saved vocab to {}".format(vocab_path))

    def load_vocab(self, vocab_path):
        """ Load a saved vocabulary. """
        assert self._special_tokens != None and self._tokens != None, "Vocab is already set or loaded"
        with open(vocab_path, encoding="utf-8", errors="ignore") as infile:
            loaded_dict = json.load(infile)
            for tok, idx in loaded_dict.items():
                self.tok2idx_dict[tok] = int(idx)
                self.idx2tok_dict[int(idx)] = tok
        assert len(self.tok2idx_dict) == len(self.idx2tok_dict)
        self._special_tokens = None
        self._tokens = None
        logger.info("Loaded vocab from {}".format(vocab_path))

    def sorted_vocab(self):
        """ Returns a list of tokens sorted according to their indexes. """
        return [tok for tok, _ in sorted(self.tok2idx_dict.items(), key=lambda x:x[1])]

    def tok2idx(self, token):
        assert type(token) == str
        return self.tok2idx_dict.get(token, self.unk_idx)

    def idx2tok(self, index):
        assert type(index) == int
        if index in self.idx2tok_dict:
            return self.idx2tok_dict[index]
        else:
            raise KeyError("Invalid index: {}".format(index))
