import os
import torch
import argparse
from iti_gen.model import ITI_GEN
torch.backends.cudnn.enabled = True

def parse_args():
    desc = "The hyperparameters for iti-gen"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--ckpt-path', type=str, default='./ckpts',
                        help='path to save the checkpoints')
    parser.add_argument('--prompt', type=str, default='a headshot of a person',
                        help='the text prompt that you want to debias. e.g., use "A natural scene" if you want to use LHQ.')
    parser.add_argument('--attr-list', type=str, default='Male,Skin_tone,Age',
                        help='input the attributes that you want to debias, separated by commas. Eg, Male,Eyeglasses,...')
    parser.add_argument('--data-path', type=str, default='./data', help='path to the reference images')
    parser.add_argument('--epochs', type=int, default=30, help='# of epochs')
    parser.add_argument('--save-ckpt-per-epochs', type=int, default=10, help='save checkpoints per epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=5, help='set # of steps we need in each epoch. We have multiple dataloaders and require updating them iteratively, so steps should be contained the same.')
    parser.add_argument('--refer-size-per-category', type=int, default=200, help='the upper bound number of reference images selected from each category')
    parser.add_argument('--token-length', type=int, default=3, help='length for the learnt token')
    parser.add_argument('--device', type=int, default=1, help='gpu number')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lam', type=float, default=0.8, help='lambda in Equation 7')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    iti_gen = ITI_GEN(args)

    # make directory
    os.makedirs(args.ckpt_path, exist_ok=True)
    folder_path = os.path.join(args.ckpt_path, '{}_{}'.format(args.prompt.replace(' ', '_'), \
                               '_'.join(iti_gen.attr_list)))
    os.makedirs(folder_path, exist_ok=True)

    epoch_saving_list = [(i + 1) * args.save_ckpt_per_epochs for i in range(int(args.epochs // args.save_ckpt_per_epochs))]
    for epoch in range(args.epochs):
        iti_gen.train(epoch, epoch_saving_list, folder_path)