from collections import Counter
from typing import Dict, List, Tuple, Union, Callable
import nltk
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-(x - self.mu)**2 / (2*self.sigma**2))

class KNRM(torch.nn.Module):
    def __init__(
        self, embedding_matrix: np.ndarray,
        out_layers: List[int],
        freeze_embeddings: bool, kernel_num: int = 21,
        sigma: float = 0.1, exact_sigma: float = 0.001,
    ):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()
        self.mlp = self._get_mlp()
        self.out_activation = torch.nn.Sigmoid()
        pass

        
    def _kernel_mus(self):
        """
        return:
            l_mu: list of kernel means
            for example if self.kernel_num is 5 -> l_mu is [-0.75, -0.25, 0.25, 0.75, 1]
        """

        l_mu = [1]
        bin_size = 2.0 / (self.kernel_num - 1)
        l_mu.append(round(1 - bin_size / 2, 4))
        for i in range(1, self.kernel_num - 1):
            l_mu.append(round(l_mu[i] - bin_size, 4))
        return l_mu[::-1]

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        mus = self._kernel_mus()
        
        for i in range(len(mus)-1):
            kernels.append(GaussianKernel(mus[i], self.sigma))
        
        kernels.append(GaussianKernel(mus[-1], self.exact_sigma))

        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        mlp = torch.nn.Sequential()

        if len(self.out_layers) == 0:
            mlp.add_module(
                f"Linear_{self.kernel_num}_{1}",
                torch.nn.Linear(self.kernel_num, 1)
            )
        else:
            # [i, i+1] -- in_features, out_features for linear layer
            layer_sizes = [self.kernel_num] + self.out_layers + [1]
            # stacking ReLU and Linear Layers
            for i in range(1, len(layer_sizes)):
                mlp.add_module(f"ReLU_{i}", torch.nn.ReLU())
                in_features, out_features = layer_sizes[i-1], layer_sizes[i]
                mlp.add_module(
                    f"Linear_{in_features}_{out_features}",
                    torch.nn.Linear(in_features, out_features)
                )
        
        return mlp
    
    def forward(
        self,
        input_1: Dict[str, torch.Tensor],
        input_2: Dict[str, torch.Tensor]
    ) -> torch.FloatTensor:
        # pairwise loss function
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)

        return out

    def _get_matching_matrix(
        self,
        query: torch.Tensor,
        doc: torch.Tensor,
        eps=1e-8
    ) -> torch.FloatTensor:
        query_emb, doc_emb = self.embeddings(query), self.embeddings(doc)

        query_n = query_emb.norm(dim=-1).unsqueeze(-1)
        query_norm = query_emb / torch.max(query_n, eps * torch.ones_like(query_n))
        doc_n = doc_emb.norm(dim=-1).unsqueeze(-1)
        doc_norm = doc_emb / torch.max(doc_n, eps * torch.ones_like(doc_n))
        
        out = torch.matmul(query_norm, doc_norm.transpose(1, 2))

        return out

    def _apply_kernels(
        self,
        matching_matrix: torch.FloatTensor
    ) -> torch.FloatTensor:
        KM = []

        for kernel in self.kernels:
            # shape = (B)
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = (B, K)
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = (Batch, Left, Embedding), (Batch, Right, Embedding)
        query, doc = inputs['query'], inputs['document']        
        # shape = (Batch, Left, Right)
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = (Batch, Kernels)
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = (Batch)
        out = self.mlp(kernels_out)
        return out