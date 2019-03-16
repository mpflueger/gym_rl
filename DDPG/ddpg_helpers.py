""" Helper functions for code using DDPG """
__author__ = "Max Pflueger"

import argparse
import os
import re


def process_save_path(save_path, mode):
    """ Prevent accidentally overwritng a checkpoint file """
    save_path = os.path.abspath(save_path)

    checkpoint_exists = False
    if os.path.exists(os.path.dirname(save_path)):
        if [f for f in os.listdir(os.path.dirname(save_path))
              if re.match(os.path.basename(save_path), f)]:
            checkpoint_exists = True

    if checkpoint_exists and mode == 'train':
        resp = input("Save path \'{}\' already exists, use anyway?"
                          " [y/N]: ".format(save_path))
        if not re.match(r'[yY](es)?$', resp):
            save_path = None
            print('\033[91m' + "  Checkpoints will NOT be saved!"
                  + '\033[0m')

    if not checkpoint_exists and mode == 'test':
        print("Cannot use 'test' mode without a checkpoint.")

    return save_path


def parse_args(name):
    """ Provide standard argument parsing """
    home = os.path.expanduser('~')
    save_default = os.path.join(home, "tf_data/{}.ckpt".format(name))
    log_default = os.path.join(home, "tf_data/{}-log".format(name))

    parser = argparse.ArgumentParser(description="Run DDPG for {}".format(name))
    parser.add_argument('mode', metavar='mode', default='train',
        choices=['train', 'test'],
        help="What do you want to do? {train, test}")
    parser.add_argument('--save', '-s',
        #default='checkpoints/{}.ckpt'.format(name),
        default=save_default,
        help="Change the default checkpoint save location")
    parser.add_argument('--restore', '-r',
        help="Restore a checkpoint before beginning")
    parser.add_argument('--log',
        default=log_default,
        help="Directory to save log files")
    return parser.parse_args()
