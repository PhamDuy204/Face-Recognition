import cv2
import numpy as np
from PIL import Image

import torchvision.transforms as T
import torch.nn.functional as F

def align_face(img, src_landmarks, dst_landmarks, size=(112, 112)):

    M, _ = cv2.estimateAffinePartial2D(np.array(src_landmarks), dst_landmarks)
    aligned = cv2.warpAffine(img, M, size)
    return aligned

def compute_embedding(img,model,detector,device='cpu'):
    if isinstance(img,str):
        img =cv2.imread(img,1)
    if img is None:
         raise ValueError("img or img path doesn't exist")
    else:
        img = cv2.cvtColor(
            cv2.resize(img,(112,112),interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB)
        arcface_template = np.array([
        [38.2946, 51.6963],  
        [73.5318, 51.5014],   
        [56.0252, 71.7366],   
        [41.5493, 92.3655],  
        [70.7299, 92.2041]   
        ], dtype=np.float32)

        boxes, _= detector.detect(Image.fromarray(img), landmarks=False)
        if boxes is not None:
            x, y, w, h = boxes[0].astype(int)
            x1, y1 = max(1, x-1), max(1, y-1)
            x2, y2 = min(img.shape[1]-1, x + w-1), min(img.shape[0]-1, y + h-1)
            img = img[y1:y2,x1:x2,:]
            img=cv2.resize(img,(112,112),interpolation=cv2.INTER_LANCZOS4)
            _, _,landmarks= detector.detect(Image.fromarray(img), landmarks=True)
            if landmarks is not None:
                src_landmarks = np.array(landmarks[0], dtype=np.float32)
                img = align_face(img, src_landmarks, arcface_template, size=(112,112))
            img = (T.ToTensor()(img)-0.5)/0.5
            # print(img)
            model.eval()
            model.to(device)
            emb= F.normalize(model(img.unsqueeze(0).to(device)).squeeze().flatten().detach().cpu(),p=2,dim=-1,eps=0).numpy()
            return emb
        else:
            raise ValueError("Don't have any faces in this pic")       