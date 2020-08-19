import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence as pad

import os
import fasttext
import numpy as np
from logging import getLogger

from tools.utils import Vocab


logger = getLogger(__name__)


class Dataset(data.Dataset):
    """ Custom data.Dataset compatible with data.DataLoader. 

    NOTE:
        __getitem__ and __len__ are required.
    """
    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.num_data = len(label)

    def __getitem__(self, index):
        """ Returns a set of items in list.
            See `data` argument in `collate_fn`.
        """
        return self.text[index], self.label[index]

    def __len__(self):
        """ Returns the number of the data. """
        return self.num_data


def collate_fn(data):
    """ Creates mini-batch tensors.

    Args:
        data [() * batch]: List of a tuple (text_i, label_i),
                           made by [dataset[i] for i in indexes]
        text_i [slen]: List of a sentence
        label_i [1]: List of a label

    Returns:
        padded_texts (batch, max_slen): LongTensor sorted according to the sentence length
        lengths (batch): LongTensor sorted according to the sentence length
        lebels (batch): LongTensor sorted according to the sentence length
    """
    def _merge_text(texts):
        """ Merge sentences into a padded tensor.

        Args:
            texts ([slen] * batch): Tuple of sentences

        Returns:
            padded_texts (batch, max_slen): LongTensor of padded sentences
            lenths (batch): LongTensor of sentence lengths
        """
        lengths = [len(s) for s in texts]
        padded_texts = torch.zeros(len(lengths), max(lengths)).long()   # make (batch, max_slen) tensor first
        for i, text in enumerate(texts):
            end = lengths[i]
            padded_texts[i, :end] = torch.LongTensor(text[:end])        # fill the tensor with the given token ids
        return padded_texts, lengths

    # Reshape items
    #   from [(text_i, label_i), (text_j, label_j), ...]
    #   to (text_i, text_j, ...), (label_i, label_j, ...)
    texts, labels = zip(*data)    # ([slen] * batch), (batch)

    # Convert to tensor
    padded_texts, lengths = _merge_text(texts)  # (batch, max_slen), (batch)
    labels = torch.LongTensor(labels)           # (batch)

    # Sort by the sentence length to feed in `pack_padded_sequence`.
    lengths, sort_index = torch.sort(torch.LongTensor(lengths), dim=0, descending=True)
    padded_texts = padded_texts[sort_index]
    labels = labels[sort_index]

    return padded_texts, lengths, labels




def get_vocab(args):
    """ Get a vocabulary and embeddings. """
    # Set or load vocabulary
    vocab = Vocab(args.min_freq)
    if args.run_test:
        vocab.load_vocab(args.vocab_path)
    else:
        with open(args.plot_path, encoding="utf-8", errors="ignore") as plot_file:
            for i, line in enumerate(plot_file):
                if i == args.max_data:
                    break
                vocab.add_token(line.split())
        with open(args.quote_path, encoding="utf-8", errors="ignore") as quote_file:
            for i, line in enumerate(quote_file):
                if i == args.max_data:
                    break
                vocab.add_token(line.split())
        vocab.set_vocab()
        if args.save:
            vocab.save_vocab(args.vocab_path)

    # Get pre-trained embeddings of the vocabulary
    embedder = fasttext.load_model(args.emb_path)
    embs = [embedder.get_word_vector(tok) for tok in vocab.sorted_vocab()]
    embs = np.array(embs, dtype=np.float32) # ndarray (vocab, dim_tok)
    logger.info("Loaded embeddings: {}".format(embs.shape))

    return vocab, embs


def preprocess_data(args, vocab):
    """ Convert tokens to their indexes and split data. """
    # Load data and convert its tokens to indexes
    plot_data = list()
    with open(args.plot_path, encoding="utf-8", errors="ignore") as plot_file:
        for i, line in enumerate(plot_file):
            if i == args.max_data:
                break
            plot_data.append([vocab.tok2idx(tok) for tok in line.split()])
    quote_data = list()
    with open(args.quote_path, encoding="utf-8", errors="ignore") as quote_file:
        for i, line in enumerate(quote_file):
            if i == args.max_data:
                break
            quote_data.append([vocab.tok2idx(tok) for tok in line.split()])
    assert len(plot_data) == len(quote_data)
    
    # Set the number of each split
    n = len(plot_data)
    train_n = int(n * 0.8)
    dev_n = int(n * 0.9) - int(n * 0.8)
    test_n = n - int(n * 0.9)
    assert train_n > 0 and dev_n > 0 and test_n > 0, "max_data is too small"

    # Split data
    # NOTE:
    #   plot (subjective) = 0
    #   quote (objective) = 1
    train_text = plot_data[:int(n * 0.8)] + quote_data[:int(n * 0.8)]
    dev_text = plot_data[int(n * 0.8):int(n * 0.9)] + quote_data[int(n * 0.8):int(n * 0.9)]
    test_text = plot_data[int(n * 0.9):] + quote_data[int(n * 0.9):]
    train_label = [0] * train_n + [1] * train_n
    dev_label = [0] * dev_n + [1] * dev_n
    test_label = [0] * test_n + [1] * test_n
    assert len(train_text) == len(train_label) 
    assert len(dev_text) == len(dev_label)
    assert len(test_text) == len(test_label)
    logger.info("Preprecessed data")
    logger.info("Train size: {}".format(train_n))
    logger.info("Dev size: {}".format(dev_n))
    logger.info("Test size: {}".format(test_n))
    
    return train_text, dev_text, test_text, \
        train_label, dev_label, test_label


def get_data(args):
    """ Get a vocabulary, embeddings and dataloaders. """
    vocab, embs = get_vocab(args)
    train_text, dev_text, test_text, train_label, dev_label, test_label \
        = preprocess_data(args, vocab)
    
    train_data = Dataset(train_text, train_label)
    dev_data = Dataset(dev_text, dev_label)
    test_data = Dataset(test_text, test_label)

    # Get custom dataloaders that return mini-batches made by `collate_fn`
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    logger.info("Processed data")

    return train_loader, dev_loader, test_loader, vocab, embs
