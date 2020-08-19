import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()

    # Data
    parser.add_argument('--data_path', default='rotten_imdb', type=str)
    parser.add_argument('--emb_path', default='./crawl-300d-2M-subword.bin', type=str)
    parser.add_argument('--min_freq', default=-1, type=int)
    parser.add_argument('--max_data', default=-1, type=int)

    # Loop
    parser.add_argument('--epoch_size', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--early_stop', default=-1, type=int)

    # CPU/GPU
    parser.add_argument('--device', default='cpu', type=str)

    # Run
    parser.add_argument('--run_test', default=False, action='store_true')
    parser.add_argument("--log_path", default="", type=str)

    # Model
    parser.add_argument('--init_method', default='default', type=str)
    parser.add_argument('--out_pad', default=0., type=float)
    parser.add_argument('--fix_emb', default=False, action='store_true')

    # Dimension
    parser.add_argument('--dim_tok', default=300, type=int)
    parser.add_argument('--dim_hid', default=128, type=int)
    
    # Dropout
    parser.add_argument('--drop_seq', default=0., type=float)

    # Layers
    parser.add_argument('--nlayer_enc', default=1, type=int)

    # Optimizer
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2_decay', default=0., type=float)

    # Seed
    parser.add_argument('--seed', default=0, type=int)

    # Save and load
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--model_path', default='saved', type=str)

    args = parser.parse_args()

    # Fixed arguments
    args.plot_path = os.path.join(args.data_path, "plot.tok.gt9.5000")
    args.quote_path = os.path.join(args.data_path, "quote.tok.gt9.5000")
    args.model_path = os.path.join("saved_models", args.model_path)
    args.param_path = os.path.join(args.model_path, "model.pth")
    args.vocab_path = os.path.join(args.model_path, "tok2idx.json")
    if args.save and not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if args.log_path != "":
        args.log_path = os.path.join("logs", args.log_path)
        if not os.path.exists("logs"):
            os.makedirs("logs")

    return args
