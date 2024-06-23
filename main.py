import argparse
import pprint

import yaml
from ijig_jepa.train.train import train


parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config.yaml file to load',
    default=r'ijig_jepa\config\pretrain_vit.yaml'
)
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='the ID for the device(s) to use.'
)


def process_main(args):

    params = None
    with open(args.fname, 'r') as config_file:
        params = yaml.load(config_file, Loader=yaml.FullLoader)

    params_str = pprint.pformat(params, indent=4)
    pprint.pprint(params_str)

    train(params)



if __name__ == '__main__':
    args = parser.parse_args()
    # single gpu only for now
    process_main(args)