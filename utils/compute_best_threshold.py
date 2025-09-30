import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import numpy as np

def compute_best_threshold(model,gallery_loader,query_loader,device='cuda'):
    with tqdm(total=len(gallery_loader)+len(query_loader)+4+len(gallery_loader)*gallery_loader.batch_size, desc="Running Eval") as pbar:
        model.to(device)
        model.eval()
        with torch.inference_mode():
            gallery_embs=[]
            gallery_labels=[]
            query_embs=[]
            query_labels=[]
            for (image,label) in gallery_loader:
                image=image.to(device)
                embs = model(image)
                gallery_embs.append(embs.detach())
                gallery_labels.append(label.detach())
                pbar.update(1)
            for (image,label) in query_loader:
                image=image.to(device)
                embs = model(image)
                query_embs.append(embs.detach())
                query_labels.append(label.detach())
                pbar.update(1)
            gallery_embs=F.normalize(torch.cat(gallery_embs),dim=-1).to(device)
            gallery_labels=torch.cat(gallery_labels).flatten().to(device)
            query_embs=F.normalize(torch.cat(query_embs),dim=-1).to(device)
            query_labels=torch.cat(query_labels).flatten().to(device)

            cosine_sim = query_embs@gallery_embs.T
            values,index=cosine_sim.topk(1,dim=-1)
            probs=values.flatten().detach().cpu().numpy()
            true_label=(gallery_labels[index].flatten()==query_labels).long().detach().cpu().numpy()

            fpr,tpr,threshold=roc_curve(true_label,probs)
            roc_auc = auc(fpr, tpr)
            print(threshold[np.argmax(tpr-fpr)])
            # Plot
            plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random chance")

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC)")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()