# `.\models\esm\openfold_utils\data_transforms.py`

```
# 导入必要的模块和类型
from typing import Dict
import numpy as np
import torch
# 导入其他模块中的常量和工具函数
from . import residue_constants as rc
from .tensor_utils import tensor_tree_map, tree_map

# 定义函数，将原子位置密集化（从37维到14维）
def make_atom14_masks(protein: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # 用于储存不同残基类型的原子14到原子37的映射
    restype_atom14_to_atom37_list = []
    # 用于储存不同残基类型的原子37到原子14的映射
    restype_atom37_to_atom14_list = []
    # 用于储存不同残基类型的原子14的掩码
    restype_atom14_mask_list = []

    # 遍历所有残基类型
    for rt in rc.restypes:
        # 获取原子14的名称列表
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        # 构建原子14到原子37的映射
        restype_atom14_to_atom37_list.append([(rc.atom_order[name] if name else 0) for name in atom_names])
        # 构建原子37到原子14的映射
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14_list.append(
            [(atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0) for name in rc.atom_types]
        )
        # 构建原子14的掩码
        restype_atom14_mask_list.append([(1.0 if name else 0.0) for name in atom_names])

    # 为 'UNK' 残基类型添加虚拟映射和掩码
    restype_atom14_to_atom37_list.append([0] * 14)
    restype_atom37_to_atom14_list.append([0] * 37)
    restype_atom14_mask_list.append([0.0] * 14)

    # 将映射和掩码转换为张量，并移到与蛋白质的设备上
    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37_list,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14_list,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask_list,
        dtype=torch.float32,
        device=protein["aatype"].device,
    )
    protein_aatype = protein["aatype"].to(torch.long)

    # 创建 (residx, atom14) 到 atom37 的映射，即一个形状为 (num_res, 14) 的数组，包含该蛋白质的原子37的索引
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    # 将结果存储到蛋白质字典中
    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # 创建反向映射的索引
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # 创建相应的掩码
    # 创建一个形状为 [21, 37] 的全零张量，表示氨基酸类型和原子类型的掩码
    restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32, device=protein["aatype"].device)
    # 遍历氨基酸类型及其对应的单字母表示
    for restype, restype_letter in enumerate(rc.restypes):
        # 根据单字母表示获取三字母表示的氨基酸名称
        restype_name = rc.restype_1to3[restype_letter]
        # 获取当前氨基酸类型的所有原子名称
        atom_names = rc.residue_atoms[restype_name]
        # 遍历当前氨基酸类型的所有原子名称
        for atom_name in atom_names:
            # 根据原子名称获取对应的原子类型
            atom_type = rc.atom_order[atom_name]
            # 将当前原子的掩码设置为 1，表示存在
            restype_atom37_mask[restype, atom_type] = 1

    # 根据蛋白质的氨基酸类型获取相应的原子类型掩码
    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    # 将原子类型掩码赋值给蛋白质的 "atom37_atom_exists" 字段
    protein["atom37_atom_exists"] = residx_atom37_mask

    # 返回处理后的蛋白质数据
    return protein
# 将输入字典中的所有值转换为在与"batch"键对应的设备上的张量
def make_atom14_masks_np(batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    # 使用tree_map函数将batch字典中的值转换为张量，设备与"aatype"键对应的设备相同
    batch = tree_map(lambda n: torch.tensor(n, device=batch["aatype"].device), batch, np.ndarray)
    # 使用tensor_tree_map函数将张量转换为NumPy数组
    out = tensor_tree_map(lambda t: np.array(t), make_atom14_masks(batch))
    # 返回转换后的字典
    return out
```