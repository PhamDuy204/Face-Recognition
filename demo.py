import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(__file__))

from utils.load_model import *
from utils.inference import * 
import argparse
import numpy as np
from PIL import Image
import torch
import faiss
import io
import shutil
from facenet_pytorch import MTCNN
parser = argparse.ArgumentParser(description="Inference script for Face Recognition")
parser.add_argument("-mn",'--model_name', type=str, default='ghost', help="Name of model")
parser.add_argument("-emb_dim",'--emb_dim', type=int, default=768, help="Embedding Dimension")
parser.add_argument("-ckpt",'--ckpt_model', type=str, default='./ckpts/2/model_e_119.pt', help="Checkpoint of pretranied model")
parser.add_argument("-cldb",'--cleardb', type=bool, default=True, help="Clear all db")
args = parser.parse_args()

if __name__ =='__main__':
    if "db_clear" not in st.session_state:
        st.session_state.db_clear=1
        if args.cleardb:
            try:
                os.remove('embs.npy')
                os.remove('ids.npy')
                os.remove('raw_imgs.npy')
            except:
                pass
    index_IP = faiss.IndexFlatIP(args.emb_dim)
    
    if not os.path.exists('embs.npy') or not os.path.exists('ids.npy') or not os.path.exists('raw_imgs.npy'):
        ids=np.array([])
        embs=np.array([])
        raw_imgs=np.array([])
        np.save("embs.npy", embs)
        np.save("ids.npy", ids)
        np.save("raw_imgs.npy", raw_imgs)
    module = load_model(model_name=args.model_name)
    if module is not None:
        detector = MTCNN(112,keep_all=True)
        model= module(embed_dim=args.emb_dim).cpu()
        if os.path.exists(args.ckpt_model):
            model.load_state_dict(torch.load(args.ckpt_model,weights_only=True,map_location=torch.device("cpu")),strict=False)
    else: raise ValueError('Model name is required')
    st.set_page_config(page_title="Vietnamese Face Recognition Demo", layout="centered")
    st.title("👤 Face Recognition Demo")
    tab1, tab2 = st.tabs(["📝 Register", "🔑 Login"])

    emb_db = np.load("embs.npy")
    id_db = np.load("ids.npy")
    raw_imgs = np.load("raw_imgs.npy")
    if len(emb_db)!=0:
        index_IP.add(emb_db)
    with tab1:
        st.header("Register User")
        user_id = st.text_input("Enter User ID")
        uploaded_file = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"])
        if st.button("Register"):
            if user_id and uploaded_file:
                image = cv2.cvtColor(np.array(Image.open(uploaded_file).convert("RGB")),cv2.COLOR_RGB2BGR)
                img_emb=compute_embedding(image,model,detector).flatten()


                emb_db=emb_db.tolist()
                emb_db.append(img_emb)
                emb_db=np.array(emb_db,dtype=np.float64)

                id_db=id_db.tolist()
                id_db.append(user_id)
                id_db=np.array(id_db,dtype=np.int32).flatten()

                raw_imgs=raw_imgs.tolist()
                raw_imgs.append(uploaded_file.getvalue())
                raw_imgs=np.array(raw_imgs).flatten()

                np.save("embs.npy", emb_db)
                np.save("ids.npy", id_db)
                np.save("raw_imgs.npy", raw_imgs)
                # index_IP.add(np.array([compute_embedding(image,model,detector)]))
                # ids.append(index_IP)
                # raw_imgs[user_id]=uploaded_file.getvalue()
            else:
                st.warning("Please enter User ID and upload an image.")
            # print(len(ids))

        if len(raw_imgs)!=0:
            st.subheader("👥 Registered Users")
            cols = st.columns(len(raw_imgs))  # tạo số cột = số user
            for col, id,data in zip(cols, id_db,raw_imgs):
                image = Image.open(io.BytesIO(data))
                col.image(image, caption=id,width='stretch')
        else:
            st.info("No users registered yet.")
    with tab2:
        st.header("Login")
        uploaded_file = st.file_uploader("Upload face image for login", type=["jpg", "png", "jpeg"], key="login")
        if st.button("Login"):
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Face", use_container_width=True)

                image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
                img_emb=compute_embedding(image,model,detector).flatten()

                D,I = index_IP.search(np.array([img_emb]),1)

                similarity=D[0][0]
                user_id_=id_db[I[0][0]]
                if similarity < 0.2:
                    # st.text(f'score : {similarity}')
                    st.error('unidentified')
                else:
                    # st.text(f'score : {similarity}')
                    st.success(f"Your id in db is : {user_id_}")
                    st.subheader("Your raw image")
                    st.image(Image.open(io.BytesIO(raw_imgs.tolist()[I[0][0]])))
            else:
                st.warning("Please upload an image.")