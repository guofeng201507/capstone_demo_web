import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--debug", action='store_false')
    parser.add_argument("--model", type=str, default="dnp107")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument('--classifier', type=str, default='base', help='classifier name')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument('--use_bn', action='store_false')

    return parser
