import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
def eval_model(model,gallery_loader,query_loader,device='cuda'):
    with tqdm(total=len(gallery_loader)+len(query_loader)+4+len(gallery_loader)*gallery_loader.batch_size, desc="Running Eval") as pbar:
        start = time.time()
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
            pos=(gallery_labels.unsqueeze(0).expand(query_labels.size(0),-1)==query_labels.unsqueeze(-1)).sum(-1)
            filter=torch.where(pos!=0)
            pos=pos[filter]
            eval_board={}
            for k in [1,5,10,20]:
                _,index=cosine_sim.topk(k)
                eval_board[f'Top-{k}-Precision']=(((gallery_labels[index]==query_labels.unsqueeze(-1)).sum(-1))[filter]/torch.where(pos<k,pos,k)).mean().item()
                torch.cuda.empty_cache()
                pbar.update(1)
            mAP=0
            for k in range(1,cosine_sim.size(-1)+1):
                _,index=cosine_sim.topk(k)
                mAP+=(((gallery_labels[index]==query_labels.unsqueeze(-1)).sum(-1))[filter]/torch.where(pos<k,pos,k)).mean().item()
                torch.cuda.empty_cache()
                pbar.update(1)
            eval_board[f'mAP']=mAP/cosine_sim.size(-1)
        pbar.set_postfix(time=f"{time.time()-start:.2f}s")
        return eval_board