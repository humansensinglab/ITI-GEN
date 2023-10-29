import os
import glob
import clip
import torch
import argparse
import scipy.stats
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def parse_args():
    desc = "Evaluation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--img-folder', type=str, default='/path/to/image/folder/you/want/to/evaluate',
                        help='path to image folder that you want to evaluate.')
    parser.add_argument('--class-list', nargs='+',
                        help='type of classes that you want to evaluate', required=True, type=str)
    parser.add_argument('--device', type=int, default=1, help='gpu number')

    return parser.parse_args()

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class ImgDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = glob.glob(os.path.join(self.root_dir, '*.png'))
        self.file_list += glob.glob(os.path.join(self.root_dir, '*.jpg'))

        print('Found {} generated images.'.format(len(self.file_list)))

        self.transforms = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        img = self.transforms(img)
        return img

def eval(path, CLASSES, device):

    eval_dataset = ImgDataset(root_dir=path)
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=16,
        num_workers=8,
        drop_last=False,
    )

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    text = clip.tokenize(CLASSES).to(device)

    img_pred_cls_list = []

    for i, data in enumerate(eval_loader):
        img = data.to(device)

        with torch.no_grad():
            logits_per_image, _ = clip_model(img, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # (bs, n_text_query)

            # select the max and set as label
            for j in np.argmax(probs, axis=1):
                img_pred_cls_list.append(j)

    num_each_cls_list = []
    for k in range(len(CLASSES)):
        num_each_cls = len(np.where(np.array(img_pred_cls_list) == k)[0])
        num_each_cls_list.append(num_each_cls)
        print("{}: total pred: {} | ratio: {}".format(CLASSES[k], num_each_cls, num_each_cls / len(eval_dataset)))

    return num_each_cls_list


if __name__ == '__main__':

    args = parse_args()

    CLASSES_prompts = args.class_list
    length = len(CLASSES_prompts)
    device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"

    # evaluate
    num_each_cls_list = eval(args.img_folder, CLASSES_prompts, device)

    # get the ratio
    each_cls_ratio = num_each_cls_list/np.sum(num_each_cls_list)

    # compute KL
    uniform_distribution = np.ones(length)/length

    KL1 = np.sum(scipy.special.kl_div(each_cls_ratio, uniform_distribution))
    KL2 = scipy.stats.entropy(each_cls_ratio, uniform_distribution)
    assert round(KL1, 4) == round(KL2, 4)

    print("For Class {}, KL Divergence is {:4f}".format(CLASSES_prompts, KL1))