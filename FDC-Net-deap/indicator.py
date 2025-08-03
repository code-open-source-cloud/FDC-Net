import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------- Evaluation index-------------------
def calculate_snr(clean, denoised):
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean((clean - denoised) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    return snr.item()


def calculate_cc(clean, denoised):
    clean_flat = clean.contiguous().view(clean.size(0), -1)
    denoised_flat = denoised.contiguous().view(denoised.size(0), -1)
    clean_mean = clean_flat - clean_flat.mean(dim=1, keepdim=True)
    denoised_mean = denoised_flat - denoised_flat.mean(dim=1, keepdim=True)
    covariance = (clean_mean * denoised_mean).sum(dim=1)
    clean_std = torch.sqrt((clean_mean ** 2).sum(dim=1))
    denoised_std = torch.sqrt((denoised_mean ** 2).sum(dim=1))
    cc = covariance / (clean_std * denoised_std)
    return cc.mean().item()


class AdaptiveBCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        pos_freq = targets.mean(dim=0)
        weights = torch.sqrt(1.0 / (pos_freq + 1e-8))

        loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            weight=weights[None, :].expand_as(targets)
        )
        return loss
