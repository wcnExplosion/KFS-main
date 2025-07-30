import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np
from layers.RevIN import RevIN

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from layers.StandardNorm import Normalize
from layers.Autoformer_EncDec import series_decomp





class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, seq_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.seq_len = seq_len

        self.value_embedding = nn.Linear(seq_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(2*d_model, d_model)

    def forward(self, x): #B,C,L
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], n_vars, 1))

        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.cat([glb,x], dim=2)
        return self.dropout(self.embedding(x)), n_vars # B,C,D



from kat_rational import KAT_Group
class KAN(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
            bias=True,
            drop=0.1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = KAT_Group(mode = act_cfg['act_init'][0],num_groups=16)
        self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(mode = act_cfg['act_init'][1],num_groups=16)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x


class Gaussian1DFilter(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, padding_mode='reflect'):
        """
        B: Batch size, L: Sequence length, C: Channels
        输入: [B, L, C] → 输出: [B, L, C]
        :param kernel_size: 卷积核大小（奇数）
        :param sigma: 高斯核标准差
        :param padding_mode: 边界填充模式 ('reflect', 'zeros' 等)
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.pad = kernel_size // 2
        self.padding_mode = padding_mode

        # 预生成高斯核（不参与梯度计算）
        self.register_buffer("kernel", self._build_kernel())

    def _build_kernel(self):
        """生成一维高斯核 [1, 1, kernel_size]"""
        x = torch.arange(self.kernel_size, dtype=torch.float) - self.pad
        kernel = torch.exp(-x ** 2 / (2 * self.sigma ** 2))
        kernel = kernel / kernel.sum()  # 归一化
        return kernel.view(1, 1, -1)  # [1, 1, K]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入张量 [B, L, C]
        :return: 滤波后张量 [B, L, C]
        """
        # 调整维度顺序: [B, L, C] → [B, C, L]（PyTorch卷积要求）
        x = x.permute(0, 2, 1)  # [B, C, L]

        # 边界填充（避免序列两端失真）
        x_pad = F.pad(x, (self.pad, self.pad), mode=self.padding_mode)

        # 分组卷积（每个通道独立滤波）
        # 复制核以匹配通道数: [1,1,K] → [C,1,K]
        kernel = self.kernel.repeat(x.shape[1], 1, 1)  # [C, 1, K]

        # 执行一维卷积
        output = F.conv1d(
            x_pad,
            weight=kernel,
            bias=None,
            groups=x.shape[1],  # 关键：分组数=通道数
            padding=0
        )

        # 恢复原始维度顺序: [B, C, L] → [B, L, C]
        return output



def FFT_for_Period(x, k=3):
    # [B, T, C]
    B,T,C = x.shape
    xf = torch.fft.rfft(x, dim=1)

    # find period by amplitudes
    frequency_list = abs(xf).mean(0)#.mean(-1)
    static_list = abs(xf).mean(0).mean(-1)
    frequency_list[0,:] = 0
    _, top_list = torch.topk(frequency_list, k,dim=0)
    _, static_top = torch.topk(static_list, k)
    mask = torch.zeros_like(frequency_list)
    mask[top_list,np.arange(C)] = 1
    xf1 = xf*mask.unsqueeze(0).repeat(x.shape[0],1,1)
    rebuild_x = torch.fft.irfft(xf1, dim=1).permute(0,2,1)



    return rebuild_x


def FFT_for_Period_percent(x,percent, low_k=8):
    B, T, C = x.shape
    if percent ==1:
        return x.permute(0,2,1)
    xf1 = torch.fft.rfft(x, dim=1)  # [B, Freq, C]
    xf = torch.clone(xf1)
    n_freq = xf.shape[1]
    xf[:,0,:] = 0
    # calculate Energy
    energy = torch.abs(xf) ** 2  # [B, Freq, C]
    total_energy = torch.sum(energy, dim=1, keepdim=True)  # [B, 1, C]

    # K-top selection
    k = n_freq
    topk_energy, topk_indices = torch.topk(energy, k, dim=1)  # [B, k, C]

    # count Energy
    cum_topk = torch.cumsum(topk_energy, dim=1)  # [B, k, C]
    target_energy = total_energy * percent  # [B, 1, C]
    mask_cum = cum_topk >= target_energy
    mask_cum[:,:low_k,:] = False
    first_true_index = torch.argmax(mask_cum.int(), dim=1)  # [B, C]
    n_to_keep = torch.clamp(first_true_index + 1, max=k)  # 限制不超过k
    # least K masking
    mask = torch.zeros_like(energy, dtype=torch.float)  # [B, Freq, C]
    k_index = torch.arange(k, device=x.device).view(1, k, 1)  # [1, k, 1]
    keep_mask = (k_index < n_to_keep.unsqueeze(1))  # [B, k, C]
    B, k_val, C = topk_indices.shape
    batch_idx = torch.arange(B, device=x.device)[:, None, None].expand(-1, k_val, C)
    channel_idx = torch.arange(C, device=x.device)[None, None, :].expand(B, k_val, -1)
    mask[batch_idx, topk_indices, channel_idx] = keep_mask.float()
    # rebuild
    xf_masked = xf * mask
    rebuild_x = torch.fft.irfft(xf_masked, dim=1).permute(0,2,1)
    return rebuild_x



class TimeStamp(nn.Module):
    def __init__(self, configs,seq_len):
        super(TimeStamp, self).__init__()
        self.ex_embedding = DataEmbedding_inverted(seq_len, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
    def forward(self, x,x_ex):
        emb = self.ex_embedding(x[:, :, :-4], x_ex)
        return emb

class FReK(nn.Module):
    def __init__(self,configs,l):
        super(FReK, self).__init__()
        self.KAN = KAN(configs.d_model,hidden_features=configs.d_ff)
        if configs.k >=16:
            self.k=configs.k//(l+1)
        else:
            self.k=configs.k
        self.l = configs.seq_len//(2**l)
        self.enc_embedding = EnEmbedding(configs.enc_in, configs.d_model, self.l, configs.dropout)
        self.decomposite = series_decomp(configs.moving_avg)
        self.percent = configs.percent / 100 if configs.percent != 100 else 1

    def forward(self, x):
        B,T,C = x.shape
        #trend,seasonal = self.decomposite(x)
        # rebuild_x = FFT_for_Period(x, self.k)
        rebuild_x = FFT_for_Period_percent(x,self.percent, self.k)
        emb_period,_ = self.enc_embedding(rebuild_x)
        #emb_res,_ = self.enc_embedding(rebuild_x-trend.permute(0,2,1))
        dec_period = self.KAN(emb_period)
        return dec_period


class FTC_fusion(nn.Module):
    def __init__(self,configs):
        super(FTC_fusion, self).__init__()

        self.fusion = nn.Sequential(nn.Linear(2*configs.d_model,configs.d_ff)
                                    ,nn.Dropout(configs.dropout)
                                    ,nn.Linear(configs.d_ff,configs.d_model))
        self.KAN = KAN(in_features=2*configs.d_model,hidden_features=configs.d_ff,out_features=configs.d_model)

    def forward(self,x,xf): #x B,C,d, xf B,C,d w B,C
        B,C,D = x.shape
        x1 = torch.cat((x,xf),dim=-1)
        out = self.KAN(x1)
        return out+x





class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.KAN = KAN(configs.d_model,hidden_features=configs.d_ff)
        self.task_name = configs.task_name

        self.pred_len = configs.pred_len if configs.pred_len > 0 else configs.seq_len
        self.projection_seq = nn.Linear(configs.d_model, self.pred_len)
        self.seq_len = configs.seq_len
        self.dropout = nn.Dropout(configs.dropout)
        self.f = torch.nn.ModuleList(
            [
                FReK(configs,i)
                for i in range(configs.down_sampling_layers)
            ]
        )
        self.t = torch.nn.ModuleList(
            [
                TimeStamp(configs, self.seq_len // (2 ** i))
                for i in range(configs.down_sampling_layers)
            ]
        )


        self.fusion = torch.nn.ModuleList(
            [
                FTC_fusion(configs)
                for i in range(configs.down_sampling_layers)
            ]
        )

        self.use_norm = configs.use_norm
        self.revin_layer = torch.nn.ModuleList([RevIN(configs.enc_in)
                                                for i in range(configs.down_sampling_layers)])

        self.configs = configs

    def multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False).to(x_enc.device)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc


    def forecast(self,x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc_list, x_mark_enc_list = self.multi_scale_process_inputs(x_enc, x_mark_enc)
        # x_dec = []
        # xf_dec = []
        fuse_dec = []
        for i in range(self.configs.down_sampling_layers):
            x_norm = self.revin_layer[i](x_enc_list[i],"norm")
            x_dec_ = self.f[i](x_norm)
            # x_dec.append(self.dropout(x_dec_))
            xf_enc_ = self.dropout(self.t[i](x_norm,x_mark_enc_list[i]))
            # xf_dec.append(self.dropout(xf_enc_))
            fuse_dec.append(self.fusion[i](x_dec_,xf_enc_))
        fuse_dec = torch.stack(fuse_dec,dim=0)
        fuse_dec = torch.mean(fuse_dec,dim=0)
        dec_out = self.projection_seq(fuse_dec).permute(0,2,1)
        dec_out = self.revin_layer[0](dec_out,'denorm')
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return None
