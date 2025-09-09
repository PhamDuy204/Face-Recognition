import numpy as np
import concurrent.futures
import shutil
import os
import glob

import kagglehub

# Download latest version
path = kagglehub.dataset_download("duypok/vn-celeb")

import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import concurrent.futures
# Hàm alignment
def align_face(img, src_landmarks, dst_landmarks, size=(112, 112)):
    # Tính ma trận affine từ landmark thực tế sang landmark chuẩn
    M, _ = cv2.estimateAffinePartial2D(np.array(src_landmarks), dst_landmarks)
    aligned = cv2.warpAffine(img, M, size)
    return aligned


# Template landmark chuẩn cho ArcFace (112x112)


def process(path):
    img = cv2.resize(cv2.imread(path),(112,112),interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arcface_template = np.array([
        [38.2946, 51.6963],   # mắt trái
        [73.5318, 51.5014],   # mắt phải
        [56.0252, 71.7366],   # mũi
        [41.5493, 92.3655],   # miệng trái
        [70.7299, 92.2041]    # miệng phải
    ], dtype=np.float32)
    # 2. Dùng MTCNN detect
    mtcnn = MTCNN(keep_all=True)
    boxes, probs, landmarks = mtcnn.detect(Image.fromarray(img_rgb), landmarks=True)
    if landmarks is not None:
        # for lm in landmarks:   # lm có shape (5,2): mắt trái, mắt phải, mũi, miệng trái, miệng phải
        src_landmarks = np.array(landmarks[0], dtype=np.float32)

        # 3. Alignment
        img_rgb = align_face(img_rgb, src_landmarks, arcface_template, size=(112,112))

        # 4. Lưu ảnh đã căn chỉnh
        cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def custom_format_dataset(data_src=path+'/VN-celeb'):
    root_path=os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]),'datasets')
    shutil.move(data_src,root_path)
    data_src=root_path+'/VN-celeb'
    if data_src[-1]=='/':
        data_src=data_src[:-1]
    def copy_and_rename(src,dst,new_name):
        shutil.copy(src,dst)
        shutil.move(os.path.join(dst,src.split('/')[-1]),os.path.join(dst,new_name))
    root_folder=os.path.dirname(data_src)
    mainfolder:str='custom_dataset'
    subfolders:list[str]=['train','gallery','query']
    for subfolder in subfolders:
        os.makedirs(os.path.join(root_folder,mainfolder,subfolder),exist_ok=True)
    image_paths=glob.glob(os.path.join(data_src,'*/*.png'))
    def copy2dst(image_path):
        id = image_path.split('/')[-2]
        additional_inf= image_path.split('/')[-1]
        new_image_name= str(int(id)-1)+'_'+additional_inf
        prob=np.random.rand()
        if prob < 0.6:
            phrase='train'
        elif prob<0.8:
            phrase='gallery'
        else:
            phrase='query'
        dest=os.path.join(root_folder,mainfolder,phrase)
        # print(dest)    
        copy_and_rename(image_path,dest,new_image_name)
    with concurrent.futures.ThreadPoolExecutor(8) as Executer:
        Executer.map(copy2dst,image_paths)
    return os.path.join(root_folder,mainfolder)
        
new_path = custom_format_dataset()
all_image_paths = glob.glob(os.path. join(new_path,'*/.png'))
if len(all_image_paths)!=0:
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as E:
        E.map(process,all_image_paths)