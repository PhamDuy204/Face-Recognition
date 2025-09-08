from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import glob
import cv2
import os

class TrainFaceRegDataset(Dataset):
    def __init__(self,data_src,img_transforms=None,phrase='train',size=112) :
        super().__init__()
        self.image_paths = sorted(glob.glob(os.path.join(data_src,phrase,'*.png')))
        self.img_transforms=img_transforms
        self.size_img= size
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        name_image= image_path.split('/')[-1]
        label= int(name_image.split('_')[0])

        image = cv2.imread(image_path,1)
        if image is not None:
            image = cv2.cvtColor(
                cv2.resize(image,(self.size_img,self.size_img),interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB)
        else:
            image = np.random.randint(0,255,(self.size_img,self.size_img,)) 

        if self.img_transforms:
            image=self.img_transforms(Image.fromarray(image))
        return image,torch.tensor(label).long()

