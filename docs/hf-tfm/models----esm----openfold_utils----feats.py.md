# `.\models\esm\openfold_utils\feats.py`

```py
# 导入所需的模块
from typing import Dict, Tuple, overload  # 引入类型提示需要的模块
import torch  # 引入 PyTorch 库
import torch.types  # 引入 PyTorch 类型
from torch import nn  # 从 PyTorch 中引入神经网络模块

# 引入自定义模块
from . import residue_constants as rc  # 从当前目录下的 residue_constants 模块中引入常量 rc
from .rigid_utils import Rigid, Rotation  # 从当前目录下的 rigid_utils 模块中引入 Rigid 和 Rotation 类
from .tensor_utils import batched_gather  # 从当前目录下的 tensor_utils 模块中引入 batched_gather 函数

# 定义用于类型提示的函数重载
@overload  # 函数重载的装饰器
def pseudo_beta_fn(aatype: torch.Tensor, all_atom_positions: torch.Tensor, all_atom_masks: None) -> torch.Tensor:  # 类型提示描述第一种情况的函数签名
    ...

@overload  # 函数重载的装饰器
def pseudo_beta_fn(aatype: torch.Tensor, all_atom_positions: torch.Tensor, all_atom_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # 类型提示描述第二种情况的函数签名
    ...

# 计算伪距离（pseudo beta）的函数，返回伪距离值或伪距离值及掩码
def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    # 判断当前氨基酸是否为甘氨酸
    is_gly = aatype == rc.restype_order["G"]
    # 获取 CA 和 CB 的索引
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    # 计算伪距离
    pseudo_beta = torch.where(
        is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    # 若存在掩码，返回伪距离及掩码，否则只返回伪距离
    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_masks[..., ca_idx],
            all_atom_masks[..., cb_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta

# 计算原子 14 到原子 37 的映射
def atom14_to_atom37(atom14: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    atom37_data = batched_gather(  # 调用 batched_gather 函数
        atom14,
        batch["residx_atom37_to_atom14"],  # 从 batch 字典中获取相应的索引
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )

    atom37_data = atom37_data * batch["atom37_atom_exists"][..., None]  # 通过乘法处理存在的原子数据

    return atom37_data  # 返回原子 37 的数据

# 构建模板角度特征
def build_template_angle_feat(template_feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    # 从特征字典中获取相关数据
    template_aatype = template_feats["template_aatype"]
    torsion_angles_sin_cos = template_feats["template_torsion_angles_sin_cos"]
    alt_torsion_angles_sin_cos = template_feats["template_alt_torsion_angles_sin_cos"]
    torsion_angles_mask = template_feats["template_torsion_angles_mask"]
    # 将相关数据拼接成模板角度特征
    template_angle_feat = torch.cat(
        [
            nn.functional.one_hot(template_aatype, 22),  # 对氨基酸类型进行独热编码
            torsion_angles_sin_cos.reshape(*torsion_angles_sin_cos.shape[:-2], 14),
            alt_torsion_angles_sin_cos.reshape(*alt_torsion_angles_sin_cos.shape[:-2], 14),
            torsion_angles_mask,
        ],
        dim=-1,
    )

    return template_angle_feat  # 返回构建的模板角度特征

# 构建模板配对特征
def build_template_pair_feat(batch: Dict[str, torch.Tensor],  # 参数为包含数据的字典
    # 定义最小值的数据类型为 torch 的 Number 类型
    min_bin: torch.types.Number,
    # 定义最大值的数据类型为 torch 的 Number 类型
    max_bin: torch.types.Number,
    # 定义箱子数量的数据类型为整数
    no_bins: int,
    # 是否使用单位向量的标志，默认为 False
    use_unit_vector: bool = False,
    # 定义一个极小值 eps，默认为 1e-20
    eps: float = 1e-20,
    # 定义一个无穷大的值 inf，默认为 1e8
    inf: float = 1e8,
    # 定义一个函数，接受一个字典类型的参数，并返回一个 torch.Tensor 类型的数据
    ) -> torch.Tensor:

    # 生成模板脊柱伪 beta 掩膜
    template_mask = batch["template_pseudo_beta_mask"]
    # 将二维掩膜扩展为三维掩膜
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

    # 计算距离直方图
    tpb = batch["template_pseudo_beta"]
    dgram = torch.sum((tpb[..., None, :] - tpb[..., None, :, :]) ** 2, dim=-1, keepdim=True)
    lower = torch.linspace(min_bin, max_bin, no_bins, device=tpb.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)

    # 将距离直方图和模板掩膜拼接起来
    to_concat = [dgram, template_mask_2d[..., None]]

    # 将模板氨基酸类型编码为 One-Hot 向量
    aatype_one_hot: torch.LongTensor = nn.functional.one_hot(
        batch["template_aatype"],
        rc.restype_num + 2,
    )

    n_res = batch["template_aatype"].shape[-1]
    # 将氨基酸 One-Hot 向量扩展为三维
    to_concat.append(aatype_one_hot[..., None, :, :].expand(*aatype_one_hot.shape[:-2], n_res, -1, -1))
    to_concat.append(aatype_one_hot[..., None, :].expand(*aatype_one_hot.shape[:-2], -1, n_res, -1))

    n, ca, c = [rc.atom_order[a] for a in ["N", "CA", "C"]]
    # 生成刚体变换
    rigids = Rigid.make_transform_from_reference(
        n_xyz=batch["template_all_atom_positions"][..., n, :],
        ca_xyz=batch["template_all_atom_positions"][..., ca, :],
        c_xyz=batch["template_all_atom_positions"][..., c, :],
        eps=eps,
    )
    points = rigids.get_trans()[..., None, :, :]
    # 计算刚体向量
    rigid_vec = rigids[..., None].invert_apply(points)

    # 计算反距离标量
    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec**2, dim=-1))

    # 生成模板所有原子掩膜
    t_aa_masks = batch["template_all_atom_mask"]
    template_mask = t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

    inv_distance_scalar = inv_distance_scalar * template_mask_2d
    unit_vector = rigid_vec * inv_distance_scalar[..., None]

    # 如果不使用单位向量，则置为 0
    if not use_unit_vector:
        unit_vector = unit_vector * 0.0

    to_concat.extend(torch.unbind(unit_vector[..., None, :], dim=-1))
    to_concat.append(template_mask_2d[..., None])

    # 拼接所有特征
    act = torch.cat(to_concat, dim=-1)
    act = act * template_mask_2d[..., None]

    # 返回合成的特征张量
    return act


# 定义一个函数，接受一个字典类型的参数，并返回一个 torch.Tensor 类型的数据
def build_extra_msa_feat(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    msa_1hot: torch.LongTensor = nn.functional.one_hot(batch["extra_msa"], 23)
    # 构建 MSA 特征
    msa_feat = [
        msa_1hot,
        batch["extra_has_deletion"].unsqueeze(-1),
        batch["extra_deletion_value"].unsqueeze(-1),
    ]
    return torch.cat(msa_feat, dim=-1)


# 定义一个函数，接受 Rigid 类型的参数和几个张量类型的参数，并返回一个 Rigid 类型的数据
def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
) -> Rigid:
    # 获取默认的 4x4 变换矩阵
    default_4x4 = rrgdf[aatype, ...]

    # 用默认的 4x4 变换矩阵生成刚体变换
    default_r = r.from_tensor_4x4(default_4x4)

    # 构造二面角旋转张量
    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # 返回一个 Rigid 类型的数据
    # 将 bb_rot 广播到与 alpha 相同的形状后连接到 alpha 的最后一个维度上
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 8, 3, 3]
    # 生成以下形式的旋转矩阵：
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # 这遵循原始代码而不是补充代码，后者使用不同的索引。

    # 创建全零张量，形状与 default_r.get_rots().get_rot_mats().shape 相同
    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    # 将 1 填充到所有_rots 的第一个维度和第二个维度上的对应位置
    all_rots[..., 0, 0] = 1
    # 使用 alpha 的部分值填充到 all_rots 的特定位置
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    # 使用 all_rots 创建 Rigid 对象
    all_frames = default_r.compose(Rigid(Rotation(rot_mats=all_rots), None))

    # 提取 all_frames 的部分数据作为 chi2_frame_to_frame, chi3_frame_to_frame, chi4_frame_to_frame
    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    # 提取 all_frames 的部分数据作为 chi1_frame_to_bb，并根据之前的计算合成 chi2_frame_to_bb, chi3_frame_to_bb, chi4_frame_to_bb
    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    # 使用 Rigid.cat 方法将所有帧转换到 bb 上的变换连接在一起
    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    # 将所有帧转换到 bb 上的变换与全局变换 r 组合
    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    # 返回所有帧转换到全局坐标系的变换
    return all_frames_to_global
# 将帧和文学位置转换为 14 个原子的位置
def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames: torch.Tensor,
    group_idx: torch.Tensor,
    atom_mask: torch.Tensor,
    lit_positions: torch.Tensor,
) -> torch.Tensor:
    # 根据氨基酸类型和组索引获取组掩码，形状为 [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # 使用组掩码创建独热编码，形状为 [*, N, 14, 8]
    group_mask_one_hot: torch.LongTensor = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # 将旋转矩阵扩展到 3D 坐标空间，形状为 [*, N, 14, 8]，并与组掩码进行相关运算
    t_atoms_to_global = r[..., None, :] * group_mask_one_hot

    # 对 t_atoms_to_global 进行张量映射，将最后一维求和，形状为 [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # 根据氨基酸类型获取原子掩码并在最后一维添加维度，形状为 [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # 根据氨基酸类型获取文学位置信息，形状为 [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    # 将预测位置应用于 t_atoms_to_global，形状为 [*, N, 14, 3]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    # 将预测位置与原子掩码相乘，形状不变
    pred_positions = pred_positions * atom_mask

    # 返回预测位置信息
    return pred_positions
```