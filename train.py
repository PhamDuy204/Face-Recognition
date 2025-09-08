from tqdm import tqdm
from accelerate import Accelerator
from eval import eval_model
import os
import torch
import torchvision.transforms as transforms
from utils.load_model import *
from utils.visualize_training import *
import argparse
from dataset import *
from torch.utils.data import DataLoader
from losses import *
from lr_scheduler import *
# from models.vit.vit import *
from models.vit.vit import *
from partial_fc_v2 import *
from eval import eval_model
def train_loop(epochs, model,module_partial_fc, optimizer, train_dataloader,gallery_loader,query_loader, lr_scheduler,device='cuda',ckpt_path='./ckpts'):
    # Initialize accelerator and tensorboard logging
    model.to(device)
    module_partial_fc.to(device)
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1,
    )
    model,module_partial_fc, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,module_partial_fc, optimizer, train_dataloader, lr_scheduler
    )

    global_steps=0
    # Now you train the model
    for epoch in range(epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        module_partial_fc.train()
        model.train()
        for step, (image,label) in enumerate(train_dataloader):
            global_steps+=1
            image = image.to(device)
            label=label.to(device)

            with accelerator.accumulate(model):
                img_emb = model(image)
                emb_dim=img_emb.size(-1)
                loss = module_partial_fc(img_emb, label)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_steps}
            progress_bar.set_postfix(**logs)
        if epoch%50==0:
            unwarp_model=accelerator.unwrap_model(model)
            eval_board=eval_model(unwarp_model,gallery_loader,query_loader,device)
            visualize_tsne(unwarp_model,train_dataloader,device=device,e=epoch)
            print(eval_board)
            os.makedirs(ckpt_path,exist_ok=True)
            torch.save(unwarp_model.state_dict(), os.path.join(ckpt_path,'model_e_{epoch}.pt'))
            print('==>Save<==')

def main():
    parser = argparse.ArgumentParser(description="Training script for Face Recognition")
    # Thêm các argument
    parser.add_argument("-ds","--data_source", type=str, default='./datasets/custom_dataset', help="path to data source")
    parser.add_argument("-nc","--num_class", type=int, default=1050, help="path to data source")
    parser.add_argument("-mn",'--model_name', type=str, default='vit', help="Name of model")
    parser.add_argument("-ims",'--img_size', type=int, default=112, help="Size of image")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight Decay")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--emb_dim", type=int, default=768, help="Embedding Dimension")
    parser.add_argument("--cpkt_path", type=str, default='./ckpts', help="Chekcpoints Path")
    args = parser.parse_args()

    device='cuda'
    train_image_transforms=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    image_transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    train_set=TrainFaceRegDataset(args.data_source,train_image_transforms,'train',args.img_size)
    gallery_set=TrainFaceRegDataset(args.data_source,image_transforms,'gallery',args.img_size)
    query_set=TrainFaceRegDataset(args.data_source,image_transforms,'query',args.img_size)

    train_loader=DataLoader(train_set,args.batch_size,shuffle=True)
    gallery_loader=DataLoader(gallery_set,args.batch_size,shuffle=False)
    query_loader=DataLoader(query_set,args.batch_size,shuffle=False)

    module=load_model(model_name=args.model_name)
    if module is not None:
        model= module(embed_dim=args.emb_dim).cuda()
        model.train()
        margin_loss = CombinedMarginLoss(64,1.0, 0.5, 0.00,0)
        module_partial_fc = PartialFC_V2(
                    margin_loss, args.emb_dim, args.num_class,1,True)
        module_partial_fc.train().cuda()
        optimizer = torch.optim.AdamW(
            params=[{"params": model.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=args.lr, weight_decay=args.weight_decay)
        total_iters=len(train_loader)*args.epochs
        lr_scheduler = PolynomialLRWarmup(
                optimizer=optimizer,
                warmup_iters=0,
                total_iters=total_iters)
        train_loop(args.epochs, model,module_partial_fc, optimizer, train_loader,
                   gallery_loader,query_loader, lr_scheduler,device=device,ckpt_path='./ckpts')

        
    else:
        raise ValueError()
    # Parse arguments
    
if __name__ == "__main__":

    main()
