import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# ------------------- Attention Module-------------------
class BandLimitedPositionalEncoding(nn.Module):
    """Position encoding based on the characteristics of EEG frequency bands"""

    def __init__(self, d_model, max_len=128, freq_band=(4, 45)):
        super().__init__()
        self.freq_band = freq_band
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        position = torch.arange(max_len).float()
        freqs = torch.linspace(freq_band[0], freq_band[1], d_model // 2)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(2 * math.pi * position.unsqueeze(1) * freqs.unsqueeze(0) / max_len)
        pe[0, :, 1::2] = torch.cos(2 * math.pi * position.unsqueeze(1) * freqs.unsqueeze(0) / max_len)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.alpha * self.pe[:, :x.size(1)] + self.beta


class ChannelGate(nn.Module):
    """Channel attention gating mechanism"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L, C = x.shape
        y = self.avg_pool(x.transpose(1, 2))
        y = self.fc(y.squeeze(-1)).unsqueeze(1)
        return x * y.expand_as(x)


class EEGSpecificTransformer(nn.Module):

    def __init__(self, d_model=128, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.pos_encoder = BandLimitedPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model))
        self.channel_gate = ChannelGate(d_model)

        # Categorical feedback projection layer
        self.feedback_proj = nn.Sequential(
            nn.Linear(2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, feedback=None):
        B, C, _, L = x.shape
        x = x.squeeze(2).transpose(1, 2)
        x = self.pos_encoder(x)

        # If there is feedback information, fuse it
        if feedback is not None:
            feedback = self.feedback_proj(feedback).unsqueeze(1)  # (B, 1, d_model)
            x = x * (1 + feedback)  # Enhance relevant features

        x = self.transformer(x)
        x = self.channel_gate(x)
        x = x.transpose(1, 2).unsqueeze(2)  # (B, C, 1, L)
        return x


class ChannelAttention(nn.Module):
    """Channel Attention (SE module)"""

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class TemporalAttention(nn.Module):
    """Time-domain Attention (1D Attention)"""

    def __init__(self, length):
        super().__init__()
        self.query = nn.Conv1d(length, length, 1)
        self.key = nn.Conv1d(length, length, 1)
        self.value = nn.Conv1d(length, length, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, _, L = x.shape
        x_flat = x.view(B, C, L)
        q = self.query(x_flat).transpose(1, 2)
        k = self.key(x_flat)
        v = self.value(x_flat)
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))
        return self.gamma * out.view(B, C, 1, L) + x


class DualPathEncoder(nn.Module):
    """Dual-path encoder with classification feedback added"""

    def __init__(self, in_channels, length):
        super().__init__()
        # Shared shallow feature extraction
        self.shared_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        # Denoising path
        self.denoise_path = nn.Sequential(
            nn.Conv2d(128, 128, (1, 3), padding=(0, 1)),
            EEGSpecificTransformer(d_model=128, nhead=4, num_layers=1),  # 修改为可以接受反馈
            nn.Conv2d(128, 128, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        # Classification path
        self.classify_path = nn.Sequential(
            nn.Conv2d(128, 128, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            TemporalAttention(length)
        )

        # Feedback fusion layer
        self.feedback_fusion = nn.Sequential(
            nn.Conv2d(128 + 2, 128, (1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

    def forward(self, x, feedback=None):
        shared = self.shared_stem(x)

        denoise_feat = self.denoise_path(shared)

        if feedback is not None:
            feedback_expanded = feedback.view(feedback.size(0), 1, 1, -1).expand(-1, -1, denoise_feat.size(2), -1)
            denoise_feat = torch.cat([denoise_feat, feedback_expanded], dim=1)
            denoise_feat = self.feedback_fusion(denoise_feat)

        classify_feat = self.classify_path(denoise_feat)

        denoise_feat = denoise_feat / (denoise_feat.norm(dim=1, keepdim=True) + 1e-6)
        classify_feat = classify_feat / (classify_feat.norm(dim=1, keepdim=True) + 1e-6)
        return denoise_feat, classify_feat


class JointDenoiseClassify(nn.Module):
    """The joint denoising classification model incorporates a feedback mechanism"""

    def __init__(self, in_channels=32, length=128, num_classes=2):
        super().__init__()
        self.encoder = DualPathEncoder(in_channels, length)

        # Denoising decoder
        self.denoise_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, in_channels, (1, 3), padding=(0, 1))
        )

        # Classification header
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 256, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ChannelAttention(256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        # Feedback memory module
        self.feedback_memory = None
        self.feedback_proj = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        feedback = None
        if self.feedback_memory is not None:
            feedback = self.feedback_proj(self.feedback_memory.detach())  # Prevent gradient backhaul
        denoise_feat, classify_feat = self.encoder(x, feedback=feedback)
        denoised = self.denoise_decoder(denoise_feat) + x
        logits = self.classifier(classify_feat)
        self.feedback_memory = torch.sigmoid(logits.detach())  # Separation calculation diagram
        return denoised, logits