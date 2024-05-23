import torch

import torch.nn.functional as F
from typing import List
    
class FeatureProcessor(torch.nn.Module):
    def __init__(
        self,
        num_layer,
        layer_selections: List[int] = None,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.normalize = normalize

        if layer_selections is not None:
            assert num_layer >= len(layer_selections)
            self.layer_selections = sorted(layer_selections)
        else:
            self.layer_selections = list(range(num_layer))
        self.weights = torch.nn.Parameter(torch.zeros(len(self.layer_selections)))
        self.print_weight = True

    def forward(self, feats):
        assert len(feats) == len(self.layer_selections), print(f"feats lens: {len(feats)} is not equal to layer_selections: {self.layer_selections}")
        if len(feats) == 1:
            return feats[0]
        weighted_feats = self.weighted_sum(feats)
        return weighted_feats

    def weighted_sum(
        self, 
        feats: List[torch.Tensor]
    ):
        '''
        Refers to the code in S3prl Featurizer:
            https://github.com/s3prl/s3prl/blob/main/s3prl/nn/upstream.py#L312
        '''
        stacked_feats = torch.stack(feats, dim=0)
        if self.normalize:
            stacked_feats = F.layer_norm(stacked_feats, (stacked_feats.shape[-1],))

        _, *origin_shape = stacked_feats.shape
        stacked_feats = stacked_feats.view(len(self.layer_selections), -1)  # [L, B*T*D]
        norm_weights = F.softmax(self.weights, dim=-1)  # [L]
        if self.print_weight:
            print("self.weights: ", self.weights)
            print("norm_weights: ", norm_weights)
            self.print_weight = False
        weighted_feats = (norm_weights.unsqueeze(-1) * stacked_feats).sum(dim=0)  # [L*B*T*D]
        weighted_feats = weighted_feats.view(*origin_shape)  # [B, T, D]
        return weighted_feats

    def reset_weight(self, weight: torch.Tensor):
        self.weights = torch.nn.Parameter(weight)
        