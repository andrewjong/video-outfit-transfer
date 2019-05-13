import os
import torch
from abc import ABC, abstractmethod


class BaseLearner(ABC):
    """
    I define a Learner as the thing that manages and records all of a neural network's
    "learning". Mainly, it takes input to optimize the parameters of its neural
    network. The Learner can also set custom command line arguments, save and load
    models, log train progress for chosen metrics, display visuals of training,
    and adaptively adjust learning rates.
    """

    def __init__(self, opt):
        """
        """
        self.opt = opt
        self.is_train = opt.is_train
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        torch.backends.cudnn.benchmark = True

        self.loss_names = []
        self.loss_names = []

