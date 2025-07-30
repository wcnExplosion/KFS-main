# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)

import torch
def FFT_for_Loss(x, k=16):
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
    return mask

def FFT_for_Loss_percent(x, low_k=8):
    B, T, C = x.shape
    xf1 = torch.fft.rfft(x, dim=1)  # [B, Freq, C]
    xf = torch.clone(xf1)
    n_freq = xf.shape[1]
    xf[:,0,:] = 0
    # 计算能量
    energy = torch.abs(xf) ** 2  # [B, Freq, C]
    total_energy = torch.sum(energy, dim=1, keepdim=True)  # [B, 1, C]

    # 确保k不超过频率数
    k = n_freq
    topk_energy, topk_indices = torch.topk(energy, k, dim=1)  # [B, k, C]

    # 计算累积能量并确定保留数
    cum_topk = torch.cumsum(topk_energy, dim=1)  # [B, k, C]
    target_energy = total_energy * 0.95  # [B, 1, C]
    mask_cum = cum_topk >= target_energy
    mask_cum[:,:low_k,:] = False
    first_true_index = torch.argmax(mask_cum.int(), dim=1)  # [B, C]
    n_to_keep = torch.clamp(first_true_index + 1, max=k)  # 限制不超过k
    # 创建保留掩码
    mask = torch.zeros_like(energy, dtype=torch.float)  # [B, Freq, C]
    k_index = torch.arange(k, device=x.device).view(1, k, 1)  # [1, k, 1]
    keep_mask = (k_index < n_to_keep.unsqueeze(1))  # [B, k, C]

    # 使用高级索引赋值（修复scatter_错误）
    B, k_val, C = topk_indices.shape
    batch_idx = torch.arange(B, device=x.device)[:, None, None].expand(-1, k_val, C)
    channel_idx = torch.arange(C, device=x.device)[None, None, :].expand(B, k_val, -1)
    mask[batch_idx, topk_indices, channel_idx] = keep_mask.float()
    # 应用掩码并重建信号
    return mask


def adaptive_fft_loss(x,y):
    mask = FFT_for_Loss(x,32)
    loss_fft = mask*(torch.fft.rfft(x, dim=1) - torch.fft.rfft(y, dim=1))
    loss_fft = loss_fft.abs().mean()
    return loss_fft