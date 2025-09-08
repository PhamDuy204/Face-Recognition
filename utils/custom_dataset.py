import numpy as np
import concurrent.futures
import shutil
import os
import glob
def custom_format_dataset(data_src):
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