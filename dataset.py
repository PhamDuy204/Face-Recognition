from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import glob
import cv2
import os

def align_face(img, src_landmarks, dst_landmarks, size=(112, 112)):
        # Tính ma trận affine từ landmark thực tế sang landmark chuẩn
        M, _ = cv2.estimateAffinePartial2D(np.array(src_landmarks), dst_landmarks)
        aligned = cv2.warpAffine(img, M, size)
        return aligned


class TrainFaceRegDataset(Dataset):
    def __init__(self,data_src,img_transforms=None,phrase='train',size=112,detector=None) :
        super().__init__()
        self.image_paths = sorted(glob.glob(os.path.join(data_src,phrase,'*.png')))
        self.img_transforms=img_transforms
        self.size_img= size
        self.detector=detector
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
        arcface_template = np.array([
            [38.2946, 51.6963],   # mắt trái
            [73.5318, 51.5014],   # mắt phải
            [56.0252, 71.7366],   # mũi
            [41.5493, 92.3655],   # miệng trái
            [70.7299, 92.2041]    # miệng phải
        ], dtype=np.float32)

        # 2. Dùng MTCNN detect
        if self.detector is not None:
            _, _, landmarks = self.detector.detect(Image.fromarray(image), landmarks=True)
            if landmarks is not None:
                # for lm in landmarks:   # lm có shape (5,2): mắt trái, mắt phải, mũi, miệng trái, miệng phải
                src_landmarks = np.array(landmarks[0], dtype=np.float32)
                # 3. Alignment
                image = align_face(image, src_landmarks, arcface_template, size=(112,112))
        if self.img_transforms:
            image=self.img_transforms(Image.fromarray(image))
        return image,torch.tensor(label).long()

