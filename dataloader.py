
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset , DataLoader
from roboflow import Roboflow

rf = Roboflow(api_key="TJJkQCgCgKoIpfHHTeRT")
project = rf.workspace("erushiki").project("side-ktyou")
dataset = project.version(5).download("multiclass")

csv_or = pd.read_csv('/content/Side-5/train/_classes.csv')

csv_or.keys()

csv_bottom = csv_or[' Bottom']
csv_left = csv_or[' Left']

csv_right = csv_or[' Right']
csv_top = csv_or[' Top']

csv_file = csv_or['filename']

DEFAULT_PATH_TRAIN = '/content/Side-5/train'
DEFAULT_PATH_TEST = '/content/Side-5/test'


def test(fname, b, l, r, t):
    img = plt.imread(f"{DEFAULT_PATH_TRAIN}/{fname}")

    plt.imshow(img)
    # plt.legend()

    print(f"{b, l, r, t}")

    if b == 1:
        print("Show Bottom")
    if l == 1:
        print("Show Left")
    if r == 1:
        print("Show Right")
    if t == 1:
        print("Show Top")


class dset(Dataset):

    def __init__(self, csv, root, transforms=None):

        super().__init__()

        self.csv = csv
        self.root = root
        self.transforms = transforms

    def __len__(self):

        return len(self.csv)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        readed = pd.read_csv(self.csv)

        csv_bottom = readed[' Bottom']
        csv_left = readed[' Left']
        csv_right = readed[' Right']
        csv_top = readed[' Top']
        csv_file = readed['filename']

        img_name = f'{self.root}/{csv_file[idx]}'
        label = np.array((csv_bottom[idx], csv_left[idx], csv_right[idx], csv_top[idx]))
        img = plt.imread(img_name)
        # img = torch.from_numpy(np.float32(img))
        # label = torch.from_numpy(np.float32(label))
        sample = {'image': img, 'label': label}

        if self.transforms != None:
            img = self.transforms(img)

        return img, label


transform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

Datas = dset('/content/Side-5/train/_classes.csv', '/content/Side-5/train', transform)

DataL = DataLoader(Datas, 1, True)
