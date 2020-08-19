from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--log_path', default=None, type=str, required=True, help="Path to a log file")
    parser.add_argument('--metric', default="Accuracy", type=str, help="Metric to take the best epoch")
    args = parser.parse_args()
    return args


args = get_args()


with open (args.log_path) as f:
    best_epoch = 0
    best_val = {args.metric:0}
    tmp_dict = dict()
    for line in f:
        if "Epoch:" in line:
            epoch = line.split()[-1]
            tmp_dict[epoch] = dict()
        elif "Val:" in line:
            _, scores = line.split("\t")
            tmp_dict[epoch]["Val"] = dict()
            for score in scores.split():
                k, v = score.split(":")
                tmp_dict[epoch]["Val"][k] = float(v)
            if tmp_dict[epoch]["Val"][args.metric] > best_val[args.metric]:
                best_epoch = epoch
                best_val = tmp_dict[epoch]["Val"]
        else:
            pass


print("Best {} on validation".format(args.metric))
print("Best epoch:", best_epoch)
print("Best validation scores:")
for k, v in best_val.items():
    print("  {}: {}".format(k, v))
