# `.\models\esm\openfold_utils\data_transforms.py`

```py
# 导入必要的库和模块
from typing import Dict  # 导入类型提示 Dict

import numpy as np  # 导入 numpy 库
import torch  # 导入 PyTorch 库

from . import residue_constants as rc  # 导入当前包中的 residue_constants 模块
from .tensor_utils import tensor_tree_map, tree_map  # 导入当前包中的 tensor_tree_map 和 tree_map 函数

def make_atom14_masks(protein: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """构建更密集的原子位置掩码（14维而非37维）。"""
    # 初始化三个空列表来存储不同的映射和掩码
    restype_atom14_to_atom37_list = []
    restype_atom37_to_atom14_list = []
    restype_atom14_mask_list = []

    # 遍历所有氨基酸类型
    for rt in rc.restypes:
        # 获取对应氨基酸类型的14维原子名称列表
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        
        # 创建从14维原子到37维原子的映射列表
        restype_atom14_to_atom37_list.append([(rc.atom_order[name] if name else 0) for name in atom_names])
        
        # 创建从37维原子到14维原子的映射列表
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14_list.append(
            [(atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0) for name in rc.atom_types]
        )

        # 创建当前氨基酸类型的14维原子掩码列表
        restype_atom14_mask_list.append([(1.0 if name else 0.0) for name in atom_names])

    # 添加 'UNK' 类型的虚拟映射和掩码
    restype_atom14_to_atom37_list.append([0] * 14)
    restype_atom37_to_atom14_list.append([0] * 37)
    restype_atom14_mask_list.append([0.0] * 14)

    # 将映射列表转换为 PyTorch 张量
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
    
    # 将 protein 字典中的 "aatype" 键的值转换为长整型
    protein_aatype = protein["aatype"].to(torch.long)

    # 创建 (残基索引, 14维原子) --> 37维原子 的映射索引数组
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    # 创建 14维原子掩码数组
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    # 将结果存储回 protein 字典中的相应键
    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # 创建用于反向映射的索引数组
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # 创建相应的掩码
    # 创建一个形状为 [21, 37] 的全零张量，数据类型为 32 位浮点数，存储在指定设备上（由 protein["aatype"].device 决定）
    restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32, device=protein["aatype"].device)
    
    # 遍历 rc.restypes 列表，同时追踪其索引和对应的单字母表示 restype_letter
    for restype, restype_letter in enumerate(rc.restypes):
        # 使用 rc.restype_1to3 字典将单字母表示转换为三字母表示
        restype_name = rc.restype_1to3[restype_letter]
        # 获取当前氨基酸类型对应的原子名列表
        atom_names = rc.residue_atoms[restype_name]
        # 遍历当前氨基酸类型的原子名列表
        for atom_name in atom_names:
            # 使用 rc.atom_order 字典获取原子名对应的类型编号
            atom_type = rc.atom_order[atom_name]
            # 在 restype_atom37_mask 张量中，标记当前氨基酸类型的指定原子类型存在（设为 1）
            restype_atom37_mask[restype, atom_type] = 1
    
    # 根据 protein_aatype 中的索引，选择相应的原子存在掩码，并赋值给 protein 字典中的 "atom37_atom_exists" 键
    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    protein["atom37_atom_exists"] = residx_atom37_mask
    
    # 返回更新后的 protein 字典
    return protein
# 定义函数，接受一个字典类型的参数 batch，值为 torch.Tensor 类型，返回值也是字典类型，其值为 np.ndarray 类型
def make_atom14_masks_np(batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    # 使用 tree_map 函数，将 batch 中的每个值转换为 torch.tensor 类型，设备为 batch["aatype"].device
    batch = tree_map(lambda n: torch.tensor(n, device=batch["aatype"].device), batch, np.ndarray)
    # 使用 tensor_tree_map 函数，对 make_atom14_masks(batch) 的结果进行处理，将其中每个 torch.Tensor 转换为 np.array
    out = tensor_tree_map(lambda t: np.array(t), make_atom14_masks(batch))
    # 返回处理后的结果 out，其中包含了每个键对应的 np.ndarray 数据
    return out
```