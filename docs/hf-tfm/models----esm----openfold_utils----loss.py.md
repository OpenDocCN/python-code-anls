# `.\models\esm\openfold_utils\loss.py`

```py
# 引入必要的模块和类型定义
from typing import Dict, Optional, Tuple
import torch

# 计算直方图的中心点
def _calculate_bin_centers(boundaries: torch.Tensor) -> torch.Tensor:
    step = boundaries[1] - boundaries[0]  # 计算边界间隔
    bin_centers = boundaries + step / 2  # 计算直方图的中心点
    bin_centers = torch.cat([bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0)  # 添加最后一个中心点
    return bin_centers

# 计算期望的对齐误差
def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,
    aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)  # 调用计算中心点的函数
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),  # 计算期望的对齐距离误差
        bin_centers[-1],  # 返回最后一个中心点作为最大值
    )

# 计算预测的对齐误差
def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """从对数输出计算对齐信心度度量。

    Args:
      logits: [*, num_res, num_res, num_bins] PredictedAlignedErrorHead 输出的对数。
      max_bin: 最大 bin 值
      no_bins: bin 的数量
    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] 每个残基对的预测对齐误差概率。
      predicted_aligned_error: [*, num_res, num_res] 每对残基的预期对齐距离误差。
      max_predicted_aligned_error: [*] 可能的最大预测误差。
    """
    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)  # 生成边界值

    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)  # 对 logits 进行 softmax 处理得到对齐信心概率
    predicted_aligned_error, max_predicted_aligned_error = _calculate_expected_aligned_error(
        alignment_confidence_breaks=boundaries,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    return {
        "aligned_confidence_probs": aligned_confidence_probs,  # 返回对齐信心概率
        "predicted_aligned_error": predicted_aligned_error,  # 返回预测的对齐误差
        "max_predicted_aligned_error": max_predicted_aligned_error,  # 返回最大预测误差
    }

# 计算 TM 分数
def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])  # 如果残基权重为空，则初始化为全1张量
    # 在指定设备上生成一个包含从0到max_bin的等间距分割的张量边界
    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)

    # 根据边界计算分箱的中心点
    bin_centers = _calculate_bin_centers(boundaries)

    # 计算残差权重的总和，但是没有将结果赋给任何变量或者使用它
    torch.sum(residue_weights)

    # 获取logits张量的倒数第二维度的大小
    n = logits.shape[-2]

    # 将n与19比较取较大值，并赋给clipped_n
    clipped_n = max(n, 19)

    # 根据公式计算d0的值
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    # 对logits张量在最后一个维度上进行softmax操作，得到概率值
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # 计算每个分箱的时间项值
    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))

    # 计算预测的时间项，即概率加权后每个分箱的加权平均
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    # 计算归一化的残差掩码，即残差权重除以其总和加上一个极小值eps
    normed_residue_mask = residue_weights / (eps + residue_weights.sum())

    # 计算每个对齐的时间项加权和
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    # 计算加权的对齐时间项乘以残差权重
    weighted = per_alignment * residue_weights

    # 找出加权项中值最大的索引
    argmax = (weighted == torch.max(weighted)).nonzero()[0]

    # 返回加权后对齐时间项中值最大的那个值
    return per_alignment[tuple(argmax)]
```