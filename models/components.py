import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Expert(nn.Module):
    """
    Class này đại diện cho một lớp expert trong kiến trúc Mixture-of-Expert.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False) # TODO: Sao không cần bias ở layer up-projection này?
        self.linear2 = nn.Linear(d_ff, d_model, bias=False) # TODO: Sao không cần bias ở layer down-projection này?
        self.dropout = nn.Dropout(dropout)  # Sử dụng dropout để tránh bị overfitting. TODO: Có thực sự cần?

        def forward(self, x):
            # linear_1_out = self.linear1(x)
            # activated_1 = F.silu(linear_1_out)
            # dropouted = self.dropout(activated_1)
            # linear_2_out = self.linear2(dropouted)

            return self.linear2(self.dropout(self.silu(self.linear1(x))))