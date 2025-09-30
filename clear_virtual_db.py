import os
import sys
sys.path.append(
os.path.dirname(__file__)
)
def clear_db():
    os.remove('embs.npy')
    os.remove('ids.npy')
    os.remove('raw_imgs.npy')
clear_db()