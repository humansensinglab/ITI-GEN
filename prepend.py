import torch
import argparse
import os
from iti_gen.model import ITI_GEN
torch.backends.cudnn.enabled = True

def parse_args():
    desc = "Prepending"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--ckpt-path', type=str, default='./ckpts',
                        help='Path to the checkpoints')
    parser.add_argument('--prompt', type=str, default='a headshot of a person',
                        help='Use the original prompt to find the saved folder.')
    parser.add_argument('--attr-list', type=str, default='Male,Skin_tone,Age',
                        help='Use the attribute list to find the corresponding model.')
    parser.add_argument('--load-model-epoch', type=int, default=9,
                        help='the model epoch loaded.')
    parser.add_argument('--prepended-prompt', type=str, default='a headshot of a person',
                        help='the text prompt used for generation. Tokens trained under "prompt" will be prepended after this prompt to implement Train-once-for-all Generation.')
    parser.add_argument('--data-path', type=str, default='./data', help='path to the reference images')
    parser.add_argument('--steps-per-epoch', type=int, default=5, help='set # of steps we need in each epoch. We have multiple dataloaders and require updating them iteratively, so steps should be contained the same.')
    parser.add_argument('--refer-size-per-category', type=int, default=200, help='the upper bound number of reference images selected from each category')
    parser.add_argument('--token-length', type=int, default=3, help='length for the learnt token')
    parser.add_argument('--device', type=int, default=1, help='gpu number')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    iti_gen = ITI_GEN(args)

    # find the folder
    folder_path = os.path.join(args.ckpt_path, '{}_{}'.format(args.prompt.replace(' ', '_'), \
                               '_'.join(iti_gen.attr_list)))

    # load model
    for idx, attr in enumerate(iti_gen.attr_list):
        state = torch.load(os.path.join(folder_path, 'basis_perturbation_embed_{}_{}.pth'.format(args.load_model_epoch, attr)), map_location='cpu')
        iti_gen.fairtoken_model[idx].load_state_dict(state, strict=False)
        iti_gen.fairtoken_model[idx].eval()

    # Save prepended prompt embeddings
    with torch.no_grad():
        prepend_embeddings = iti_gen.prompt_prepend(args.prepended_prompt)
        basis_np = prepend_embeddings.clone().detach().cpu()
        path = os.path.join(folder_path, 'prepend_prompt_embedding_{}'.format(args.prepended_prompt.replace(' ', '_')))
        os.makedirs(path, exist_ok=True)
        torch.save(basis_np, os.path.join(path, 'basis_final_embed_{}.pt'.format(args.load_model_epoch)))  # (108, 77, 768)
        print("Successfully save Models!")