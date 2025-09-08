import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
import os
def visualize_tsne(model,train_loader, perplexity=30,device='cuda',save_path=
                   os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]),'visualize_training'),e=0):
    print('Saving png file...')
    os.makedirs(save_path,exist_ok=True)
    new_save_path=os.path.join(save_path,str(len(os.listdir(save_path))))
    os.makedirs(new_save_path,exist_ok=True)
    embeddings=[]
    labels=[]
    model.to(device)
    for image,label in train_loader:
        model.eval()
        with torch.inference_mode():
            image = image.to(device)
            label=label.to(device)

            embedding= model(image)
            embeddings.extend(F.normalize(embedding.detach().cpu(),dim=-1).numpy().tolist())
            labels.extend(label.detach().cpu().numpy().tolist())
            if len(embeddings)>900:
                break
    X = np.array(embeddings)
    y = np.array(labels)
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200)
    X_tsne = tsne.fit_transform(X)

    # Normalize to unit circle
    norms = np.linalg.norm(X_tsne, ord=2, axis=1, keepdims=True)
    X_tsne = X_tsne / norms

    # Convert to DataFrame
    df = pd.DataFrame({
        "x": X_tsne[:, 0],
        "y": X_tsne[:, 1],
        "label": y
    })

    # Plot with seaborn
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df,
        x="x", y="y",
        hue="label",
        palette="tab20",
        s=10, alpha=0.7,
        linewidth=0,
        legend=False
    )
    plt.title("t-SNE visualization of embeddings")
    plt.savefig(os.path.join(new_save_path,f"tsne_plot_{e}.png"), dpi=300, bbox_inches="tight")
    plt.close()
