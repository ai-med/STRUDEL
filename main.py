import os
import sys
import torch
import yaml
import configparser
import argparse

from utils.data import data_loader, data_loader_test
from model.octse_net import OctaveSENet
from agent import Agent


def run(args, params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = OctaveSENet(params['NETWORK'])

    agent = Agent(model, device)

    if args.mode == "train":
        try:
            agent.train(params['TRAINING'], data_loader(params['DATA']))
        except KeyboardInterrupt:
            answer = input("Wanna safe the current state? (y/n): ").lower().strip()
            if answer[0] == "y":
                agent.shutdown_safe(params['TRAINING']['CHECKPOINT'])
            else:
                sys.exit(0)
    elif args.mode == "eval":
        agent.eval(params['EVALUATION'], data_loader_test(args.data, params['DATA']))


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", help="either 'train' or 'eval'")
    parser.add_argument('--data', type=str, default="adni", help="either 'adni', 'challenge' or 'concat'")
    parser.add_argument('--config', type=str, default="./config/default.yaml", help="path to config file")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    if os.path.exists(args.config):
        params = read_yaml(file_path=args.config)
    else:
        raise FileNotFoundError('config file not found')
    run(args, params)
