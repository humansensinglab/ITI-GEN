import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class ImgDataset(Dataset):
    """
    Construct the dataset
    """
    def __init__(self, root_dir, label, upper_bound=200):
        self.root_dir = root_dir
        self.label = label
        self.upper_bound = upper_bound

        self.file_list = []
        self.prepare_data_list()

        self.transforms = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def prepare_data_list(self):
        # set all images together, such as put positive and negative all into self.file_list
        for i, dir in enumerate(self.root_dir):
            path = glob.glob(os.path.join(dir, '*.jpg'))
            path += glob.glob(os.path.join(dir, '*.png'))

            # random select the images based on the min(upper, folder_size)
            selected_index = np.random.choice(len(path), min(self.upper_bound,len(path)), replace=False)
            for index in selected_index:
                self.file_list.append({'path': path[index], 'label': self.label[i]})

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx]['path'])
        img = self.transforms(img)
        label = self.file_list[idx]['label']
        return dict(img=img, label=label)