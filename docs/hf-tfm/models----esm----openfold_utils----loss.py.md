# `.\models\esm\openfold_utils\loss.py`

```
# 版权声明
# 从包中导入相关类型
from typing import Dict, Optional, Tuple
# 导入torch库

import torch

# 计算每个分桶的中心值
def _calculate_bin_centers(boundaries: torch.Tensor) -> torch.Tensor:
    # 计算分桶边界之间的步长
    step = boundaries[1] - boundaries[0]
    # 计算分桶的中心值
    bin_centers = boundaries + step / 2
    # 将最后一个分桶的中心值加上一个步长，并添加到分桶中心值的张量中
    bin_centers = torch.cat([bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0)
    return bin_centers

# 计算期望的对齐错误
def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,
    aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 计算分桶的中心值
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    # 计算期望的对齐错误和最大值
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )

# 计算预测的对齐错误
def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """计算从logits中得到的对齐置信度指标。

    Args:
      logits: [*, num_res, num_res, num_bins] 来自PredictedAlignedErrorHead的logits输出。
      max_bin: 最大的分桶值
      no_bins: 分桶的数量
    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] 每个残基对的每个分桶的预测对齐错误概率。
      predicted_aligned_error: [*, num_res, num_res] 每对残基的预期对齐距离错误。
      max_predicted_aligned_error: [*] 可能的最大预测错误。
    """
    # 根据最大分桶值和分桶数量在设备上生成分桶边界
    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)

    # 对logits进行softmax操作得到对齐置信度概率
    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    # 计算预测的对齐错误和最大值
    predicted_aligned_error, max_predicted_aligned_error = _calculate_expected_aligned_error(
        alignment_confidence_breaks=boundaries,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }

# 计算TM得分
def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    # 如果残基权重未给定，则初始化为与logits张量形状相同的全1张量
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])  
    # 使用 torch.linspace 在 0 和 max_bin 之间生成间隔均匀的步骤，共计 (no_bins - 1) 步，使用 logits 设备
    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)

    # 调用 _calculate_bin_centers 函数计算每个区间的中心
    bin_centers = _calculate_bin_centers(boundaries)

    # 计算 residue_weights 的总和并丢弃结果
    torch.sum(residue_weights)

    # 获取 logits 的倒数第二个维度的大小
    n = logits.shape[-2]

    # 确定 clipped_n 为 n 和 19 的最大值
    clipped_n = max(n, 19)

    # 计算 d0 的值，根据公式 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    # 使用 softmax 函数计算 logits 的 softmax 结果
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # 计算 tm_per_bin，根据公式 1.0 / (1 + (bin_centers**2) / (d0**2))
    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))

    # 计算 predicted_tm_term，维度为 logits 的倒数第一个维度的和
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    # 计算 normed_residue_mask，为 residue_weights 除以其和再加上一个很小的值 eps 防止除0
    normed_residue_mask = residue_weights / (eps + residue_weights.sum())

    # 计算 per_alignment，为 predicted_tm_term 与 normed_residue_mask 逐元素相乘后再对最后一个维度求和
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    # 计算 weighted，为 per_alignment 与 residue_weights 逐元素相乘
    weighted = per_alignment * residue_weights

    # 找到 weighted 中最大值的索引并返回对应的 per_alignment 的值
    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]
```