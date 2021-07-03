from torch import optim
from logging import getLogger


logger = getLogger(__name__)


def get_optim(args, model):
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "AMSGrad":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    else:
        raise KeyError("Unsupported optimizer: {}".format(args.optimizer))
    logger.info("Optimizer:\n{}".format(optimizer))

    return optimizer
