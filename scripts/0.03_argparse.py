from argparse import ArgumentParser
from pytorch_lightning import Trainer

def main(args):
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)