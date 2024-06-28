# `.\models\esm\openfold_utils\feats.py`

```
# 导入必要的模块和类型声明
from typing import Dict, Tuple, overload
import torch
import torch.types
from torch import nn

# 导入自定义模块和函数
from . import residue_constants as rc
from .rigid_utils import Rigid, Rotation
from .tensor_utils import batched_gather

# 定义一个函数重载，接受 torch.Tensor 类型参数并返回 torch.Tensor 类型结果
@overload
def pseudo_beta_fn(aatype: torch.Tensor, all_atom_positions: torch.Tensor, all_atom_masks: None) -> torch.Tensor:
    ...

# 定义另一个函数重载，接受 torch.Tensor 类型参数并返回元组 (torch.Tensor, torch.Tensor)
@overload
def pseudo_beta_fn(
    aatype: torch.Tensor, all_atom_positions: torch.Tensor, all_atom_masks: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...

# 实现函数 pseudo_beta_fn，根据输入参数计算伪β原子位置
def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    # 检查是否为甘氨酸类型
    is_gly = aatype == rc.restype_order["G"]
    # 确定 CA 和 CB 的索引
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    # 根据是否为甘氨酸选择 CA 或 CB 的坐标作为伪β原子的位置
    pseudo_beta = torch.where(
        is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    # 如果提供了原子掩码，则根据甘氨酸类型选择相应的掩码
    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_masks[..., ca_idx],
            all_atom_masks[..., cb_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta

# 定义函数 atom14_to_atom37，将 14 个原子数据映射为 37 个原子数据
def atom14_to_atom37(atom14: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    # 使用 batched_gather 函数将 atom14 数据转换为 atom37 数据
    atom37_data = batched_gather(
        atom14,
        batch["residx_atom37_to_atom14"],
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )

    # 将不存在的原子位置数据置零
    atom37_data = atom37_data * batch["atom37_atom_exists"][..., None]

    return atom37_data

# 定义函数 build_template_angle_feat，构建模板角度特征
def build_template_angle_feat(template_feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    # 获取模板的氨基酸类型和角度正弦余弦值
    template_aatype = template_feats["template_aatype"]
    torsion_angles_sin_cos = template_feats["template_torsion_angles_sin_cos"]
    alt_torsion_angles_sin_cos = template_feats["template_alt_torsion_angles_sin_cos"]
    torsion_angles_mask = template_feats["template_torsion_angles_mask"]
    # 构建模板角度特征张量，包括氨基酸独热编码、主要和备选的角度正弦余弦值以及角度掩码
    template_angle_feat = torch.cat(
        [
            nn.functional.one_hot(template_aatype, 22),
            torsion_angles_sin_cos.reshape(*torsion_angles_sin_cos.shape[:-2], 14),
            alt_torsion_angles_sin_cos.reshape(*alt_torsion_angles_sin_cos.shape[:-2], 14),
            torsion_angles_mask,
        ],
        dim=-1,
    )

    return template_angle_feat

# 定义函数 build_template_pair_feat，构建模板对特征
def build_template_pair_feat(
    batch: Dict[str, torch.Tensor],
    # 继续下一个函数定义
    min_bin: torch.types.Number,
    # 定义变量 min_bin，用于存储最小 bin 的值，类型为 torch.types.Number
    max_bin: torch.types.Number,
    # 定义变量 max_bin，用于存储最大 bin 的值，类型为 torch.types.Number
    no_bins: int,
    # 定义变量 no_bins，用于存储 bin 的数量，类型为整数 int
    use_unit_vector: bool = False,
    # 定义变量 use_unit_vector，用于指示是否使用单位向量，默认为 False，类型为布尔值 bool
    eps: float = 1e-20,
    # 定义变量 eps，用于存储一个小的正数值，用作数值稳定性的参数，默认为 1e-20，类型为浮点数 float
    inf: float = 1e8,
    # 定义变量 inf，用于表示一个较大的数，通常用作无穷大的近似值，默认为 1e8，类型为浮点数 float
def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
) -> Rigid:
    # [*, N, 8, 4, 4]
    # 从 rrgdf 中根据氨基酸类型选择默认的 4x4 变换矩阵
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    # 从 default_4x4 创建 Rigid 对象，包括旋转矩阵和平移矩阵
    default_r = r.from_tensor_4x4(default_4x4)

    # 创建一个新的形状与 alpha 一致的零张量，最后两个维度为 2，表示二维旋转信息
    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    # 将 bb_rot 在第二维度扩展，与 alpha 连接，形成新的张量
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # 创建一个全零张量 all_rots，形状与 default_r 的旋转矩阵形状相同
    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    # 设置旋转矩阵的部分值，形成类似如下结构的旋转矩阵：
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # 这与原始代码保持一致，而不是附加的文档中所用的不同索引方式。

    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    # 使用 default_r 构造所有帧的刚体变换
    all_frames = default_r.compose(Rigid(Rotation(rot_mats=all_rots), None))

    # 从所有帧中提取帧到帧的转移矩阵的特定部分
    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    # 从所有帧中提取帧到背骨坐标系的转移矩阵的特定部分
    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    # 将所有帧的刚体变换连接成一个新的刚体变换序列 all_frames_to_bb
    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    # 将所有帧的刚体变换 all_frames_to_bb 与 r 的全局变换连接起来，形成最终的全局变换
    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    # 返回最终的全局变换结果
    return all_frames_to_global
# 将 group_idx 按照 aatype 中的索引值进行索引，得到形状为 [*, N, 14] 的掩码
group_mask = group_idx[aatype, ...]

# 使用 nn.functional.one_hot 函数将 group_mask 转换为 one-hot 编码，形状为 [*, N, 14, 8]，
# 其中 8 是 default_frames.shape[-3] 的值，表示类别数量
group_mask_one_hot: torch.LongTensor = nn.functional.one_hot(
    group_mask,
    num_classes=default_frames.shape[-3],
)

# 将旋转矩阵 r 与 group_mask_one_hot 相乘，扩展维度以适应广播规则，得到形状为 [*, N, 14, 3] 的张量 t_atoms_to_global
t_atoms_to_global = r[..., None, :] * group_mask_one_hot

# 对 t_atoms_to_global 在最后一个维度上求和，得到形状为 [*, N, 14] 的张量
t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

# 根据 aatype 中的索引值，获取对应的 atom_mask，然后在最后一个维度上添加一个维度，形状变为 [*, N, 14, 1]
atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

# 根据 aatype 中的索引值，获取对应的 lit_positions，形状为 [*, N, 14, 3]
lit_positions = lit_positions[aatype, ...]

# 将 lit_positions 应用到 t_atoms_to_global 上，得到预测的位置 pred_positions，形状为 [*, N, 14, 3]
pred_positions = t_atoms_to_global.apply(lit_positions)

# 将预测的位置 pred_positions 与 atom_mask 相乘，使得未激活的原子位置为零，形状不变 [*, N, 14, 3]
pred_positions = pred_positions * atom_mask

# 返回预测的原子位置 pred_positions
return pred_positions
```