# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collection of RENI Losses.
"""
import torch
from torch import nn

class ScaleInvariantLogLoss(nn.Module):
    def __init__(self):
        super(ScaleInvariantLogLoss, self).__init__()

    def forward(self, log_predicted, log_gt):
        R = log_predicted - log_gt

        term1 = torch.mean(R**2)
        term2 = torch.pow(torch.sum(R), 2) / (log_predicted.numel()**2)

        loss = term1 - term2

        return loss
        
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, predicted, gt):
        similarity = self.cosine_similarity(predicted, gt)
        cosine_similarity_loss = 1.0 - similarity.mean()
        return cosine_similarity_loss