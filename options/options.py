from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import learners


class BaseOptions:
    def __init__(self):
        """

        """
        self.initialized = False
        self.is_train = None
        self.parser = None

    def initialize(self, parser: ArgumentParser):
        parser.add_argument("--dataroot", required=True, help="path to images")
        parser.add_argument(
            "--name", type=str, default="experiment_name", help="name of experiment"
        )
        parser.add_argument(
            "--checkpoints_dir", default="./checkpoints", help="models are saved here"
        )
        parser.add_argument(
            "--learner", help="chooses which learner to use (see ./learners/*). "
        )
        parser.add_argument(
            "--netD", default="basic", help="what discriminator architecture to use"
        )
        parser.add_argument(
            "--netG", default="basic", help="what generator architecture to use"
        )
        parser.add_argument("--no_dropout")
        parser.add_argument(
            "--data_threads", default=4, type=int, help="number of threads to load data"
        )
        parser.add_argument("--loss", help="what loss function to use")
        parser.add_argument(
            "--batch_size",
            default=1,
            type=int,
            help="batch size to use during training",
        )
        parser.add_argument(
            "-v",
            "--verbosity",
            default=1,
            type=int,
            help="verbosity level (1=low, 3=high)",
            choices=[1, 2, 3],
        )
        parser.add_argument(
            "--from_epoch",
            default="latest",
            help="what epoch to continue training from",
        )
        parser.add_argument(
            "--from_step",
            default=0,
            type=int,
            help="what step to continue training from",
        )
        parser.add_argument(
            "--scale_size",
            type=int,
            default=256,
            help="scale images to this size first",
        )
        parser.add_argument(
            "--crop_size", type=int, default=256, help="then center crop to this size"
        )
        parser.add_argument(
            "--suffix",
            help="append an argument to experiment name, e.g. 'bs_{batch_size'",
        )
        self.initialized = True
        return parser

    def gather_options(self):

        # if the developer doesn't choose a custom parser, choose one for the dev
        if not self.initialized:
            parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt: Namespace
        opt, _ = parser.parse_known_args()

        # modify options based on the learner
        learner_name = opt.learner
        learner_option_setter = learners.get_option_setter(learner_name)
        parser: ArgumentParser = learner_option_setter(parser, self.is_train)
        opt, _ = parser.parse_known_args()

        self.parser = parser
        return parser.parse_args()

    # def set_extra_options(self, parser, module, keyword, *args):
    #     option_setter = module.get_option_setter(keyword)
    #     parser = option_setter(parser, *args)

    def parse(self):
        opt = self.gather_options()
        # add whether we're training
        opt.is_train = self.is_train

        if opt.suffix:
            suffix = "_" + opt.suffix.format(**vars(opt))
            opt.name = opt.name + suffix


class TrainOptions(BaseOptions):
    def initialize(self, parser: ArgumentParser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            "--optimizerG",
            choices=["sgd", "adam", "adabound"],
            help="optimizer to use for generator",
        )
        parser.add_argument(
            "--optimizerD",
            choices=["sgd", "adam", "adabound"],
            help="optimizer to use for discriminator",
        )
        parser.add_argument(
            "--gan_mode",
            help="GAN objective function",
            choices=["vanilla", "wgangp", "dragan"],
        )
        parser.add_argument(
            "--lr", type=float, default=0.0002, help="initial learning rate"
        )
        parser.add_argument(
            "--lr_policy", choices=["linear", "step", "plateau", "cosine"]
        )
        parser.add_argument(
            "--lr_decay_iters", default=50, help="choose when to decay lr"
        )

        self.is_train = True
        return parser


class TestOptions(BaseOptions):
    def initialize(self, parser: ArgumentParser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            "--results_dir", default="results", help="where to save results to"
        )
        self.is_train = False
