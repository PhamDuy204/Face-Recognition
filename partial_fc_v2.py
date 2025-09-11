
import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import linear, normalize


class PartialFC_V2(torch.nn.Module):
    """
    https://arxiv.org/abs/2203.15565
    A distributed sparsely updating variant of the FC layer, named Partial FC (PFC).
    When sample rate less than 1, in each iteration, positive class centers and a random subset of
    negative class centers are selected to compute the margin-based softmax loss, all class
    centers are still maintained throughout the whole training process, but only a subset is
    selected and updated in each iteration.
    .. note::
        When sample rate equal to 1, Partial FC is equal to model parallelism(default sample rate is 1).
    Example:
    --------
    >>> module_pfc = PartialFC(embedding_size=512, num_classes=8000000, sample_rate=0.2)
    >>> for img, labels in data_loader:
    >>>     embeddings = net(img)
    >>>     loss = module_pfc(embeddings, labels)
    >>>     loss.backward()
    >>>     optimizer.step()
    """
    _version = 2

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
        skip_ce_loss=False
    ):
        self.skip_ce_loss=skip_ce_loss
        """
        Paramenters:
        -----------
        embedding_size: int
            The dimension of embedding, required
        num_classes: int
            Total number of classes, required
        sample_rate: float
            The rate of negative centers participating in the calculation, default is 1.0.
        """
        super(PartialFC_V2, self).__init__()


        # self.cross_entropy = nn.CrossEntropyLoss()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.last_batch_size: int = 0

        self.is_updated: bool = True
        self.init_weight_update: bool = True
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.fc=nn.Linear(embedding_size,num_classes,bias=False)
        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise

    # def sample(self, labels, index_positive):
    #     """
    #         This functions will change the value of labels
    #         Parameters:
    #         -----------
    #         labels: torch.Tensor
    #             pass
    #         index_positive: torch.Tensor
    #             pass
    #         optimizer: torch.optim.Optimizer
    #             pass
    #     """
    #     with torch.no_grad():
    #         positive = torch.unique(labels[index_positive], sorted=True).cuda()
    #         if self.num_sample - positive.size(0) >= 0:
    #             perm = torch.rand(size=[self.num_local]).cuda()
    #             perm[positive] = 2.0
    #             index = torch.topk(perm, k=self.num_sample)[1].cuda()
    #             index = index.sort()[0].cuda()
    #         else:
    #             index = positive
    #         self.weight_index = index

    #         labels[index_positive] = torch.searchsorted(index, labels[index_positive])

    #     return self.weight[self.weight_index]

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
        epoch
    ):
        """
        Parameters:
        ----------
        local_embeddings: torch.Tensor
            feature embeddings on each GPU(Rank).
        local_labels: torch.Tensor
            labels on each GPU(Rank).
        Returns:
        -------
        loss: torch.Tensor
            pass
        """
        local_labels.squeeze_()
        local_labels = local_labels.long()

        # batch_size = local_embeddings.size(0)
        # # if self.last_batch_size == 0:
        # #     self.last_batch_size = batch_size
        # assert self.last_batch_size == batch_size, (
        #     f"last batch size do not equal current batch size: {self.last_batch_size} vs {batch_size}")
        # with torch.no_grad():
        #     positive = torch.unique(labels[index_positive], sorted=True).cuda()
        #     if self.num_sample - positive.size(0) >= 0:
        #         perm = torch.rand(size=[self.num_local]).cuda()
        #         perm[positive] = 2.0
        #         index = torch.topk(perm, k=self.num_sample)[1].cuda()
        #         index = index.sort()[0].cuda()
        #     else:
        #         index = positive
        #     self.weight_index = index

        #     labels[index_positive] = torch.searchsorted(index, labels[index_positive])
        # _gather_embeddings = [
        #     torch.zeros((batch_size, self.embedding_size)).cuda()
        #     for _ in range(self.world_size)
        # ]
        # # _gather_labels = [
        # #     torch.zeros(batch_size).long().cuda() for _ in range(self.world_size)
        # # ]
        # _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        # distributed.all_gather(_gather_labels, local_labels)

        # embeddings = torch.cat(_list_embeddings)
        # labels = torch.cat(_gather_labels)

        # labels = labels.view(-1, 1)
        # index_positive = (self.class_start <= labels) & (
        #     labels < self.class_start + self.num_local
        # )
        # labels[~index_positive] = -1
        # labels[index_positive] -= self.class_start

        norm_embeddings = normalize(local_embeddings,dim=-1)
        norm_weight_activated = normalize(self.weight,dim=-1)
        logits = linear(norm_embeddings, norm_weight_activated)

        logits = self.margin_softmax(logits, local_labels)
        margin_loss=nn.CrossEntropyLoss()(logits,local_labels)
        ce_loss = nn.CrossEntropyLoss()(self.fc(local_embeddings),local_labels)
        if ((epoch <= 30) and (self.skip_ce_loss==False)):
            loss =ce_loss+0.0*margin_loss
        else:
            loss =0.0*ce_loss+margin_loss
        return loss