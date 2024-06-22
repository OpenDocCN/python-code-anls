# `.\models\esm\openfold_utils\residue_constants.py`

```py
# 导入collections、copy、functools和resources模块
import collections
import copy
import functools
from importlib import resources
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

# 两个连续 CA 之间的距离（反式构象：omega = 180）
ca_ca = 3.80209737096

# AA 类型及其对应的 chi 角列表。每个 AA 类型的列表包含按顺序排列的 chi1、chi2、chi3、chi4（或者从 chi1 开始的相关子集）。ALA 和 GLY 没有 chi 角，因此它们的 chi 角列表为空。
chi_angles_atoms: Dict[str, List[List[str]]] = {
    "ALA": [],
    # 精氨酸的 chi5 角始终为 0 加减 5 度，因此忽略它。
    "ARG": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "NE"], ["CG", "CD", "NE", "CZ"]],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "CE"], ["CG", "CD", "CE", "NZ"]],
    "MET": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "SD"], ["CB", "CG", "SD", "CE"]],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}

# 如果 chi 角以固定长度数组给出，则该矩阵确定如何对每种 AA 类型进行掩码处理。顺序与 restype_order（见下文）相同。
chi_angles_mask: List[List[float]] = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
]
    [1.0, 1.0, 0.0, 0.0],  # HIS，氨基酸组 HIS，包含氨基酸相似性信息
    [1.0, 1.0, 0.0, 0.0],  # ILE，氨基酸组 ILE，包含氨基酸相似性信息
    [1.0, 1.0, 0.0, 0.0],  # LEU，氨基酸组 LEU，包含氨基酸相似性信息
    [1.0, 1.0, 1.0, 1.0],  # LYS，氨基酸组 LYS，包含氨基酸相似性信息
    [1.0, 1.0, 1.0, 0.0],  # MET，氨基酸组 MET，包含氨基酸相似性信息
    [1.0, 1.0, 0.0, 0.0],  # PHE，氨基酸组 PHE，包含氨基酸相似性信息
    [1.0, 1.0, 0.0, 0.0],  # PRO，氨基酸组 PRO，包含氨基酸相似性信息
    [1.0, 0.0, 0.0, 0.0],  # SER，氨基酸组 SER，包含氨基酸相似性信息
    [1.0, 0.0, 0.0, 0.0],  # THR，氨基酸组 THR，包含氨基酸相似性信息
    [1.0, 1.0, 0.0, 0.0],  # TRP，氨基酸组 TRP，包含氨基酸相似性信息
    [1.0, 1.0, 0.0, 0.0],  # TYR，氨基酸组 TYR，包含氨基酸相似性信息
    [1.0, 0.0, 0.0, 0.0],  # VAL，氨基酸组 VAL，包含氨基酸相似性信息
[
# 下面的 chi 角度是 pi 周期的：它们可以被旋转多个 pi 而不影响结构。

# 定义了每个氨基酸的 chi 角度
chi_pi_periodic: List[List[float]] = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

# 相对于 8 个刚性群组的原子位置，由 pre-omega, phi, psi 和 chi 角度定义：
# 0: 'backbone group',
# 1: 'pre-omega-group', (empty)
# 2: 'phi-group', (currently empty, because it defines only hydrogens)
# 3: 'psi-group',
# 4,5,6,7: 'chi1,2,3,4-group'
# 原子位置是相对于相应旋转轴的轴端原子的位置。 x 轴指向旋转轴的方向，y 轴是这样定义的，即二面角角度定义的原子（上面 chi_angles_atoms 中的最后一个条目）在 xy 平面上（具有正的 y 坐标）。
# 格式: [atomname, group_idx, rel_position]
rigid_group_atom_positions: Dict[str, List[Tuple[str, int, Tuple[float, float, float]]]] = {
    # 氨基丙酸 ALA 的原子位置信息
    "ALA": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, -0.000, -0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
    ],
    # 精氨酸 ARG 的原子位置信息
    "ARG": [
        ("N", 0, (-0.524, 1.362, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.525, -0.000, -0.000)),
        ("CB", 0, (-0.524, -0.778, -1.209)),
        ("O", 3, (0.626, 1.062, 0.000)),
        ("CG", 4, (0.616, 1.390, -0.000)),
        ("CD", 5, (0.564, 1.414, 0.000)),
        ("NE", 6, (0.539, 1.357, -0.000)),
        ("NH1", 7, (0.206, 2.301, 0.000)),
        ("NH2", 7, (2.078, 0.978, -0.000)),
        ("CZ", 7, (0.758, 1.093, -0.000)),
    ],
    # 天冬氨酸 ASN 的原子位置信息
    "ASN": [
        ("N", 0, (-0.536, 1.357, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, -0.000, -0.000)),
        ("CB", 0, (-0.531, -0.787, -1.200)),
        ("O", 3, (0.625, 1.062, 0.000)),
        ("CG", 4, (0.584, 1.399, 0.000)),
        ("ND2", 5, (0.593, -1.188, 0.001)),
        ("OD1", 5, (0.633, 1.059, 0.000)),
    ],
    # "ASP" 残基的原子坐标数据
    "ASP": [
        # 残基中的原子名、相对于残基中心的位置、原子的三维坐标
        ("N", 0, (-0.525, 1.362, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.527, 0.000, -0.000)),
        ("CB", 0, (-0.526, -0.778, -1.208)),
        ("O", 3, (0.626, 1.062, -0.000)),
        ("CG", 4, (0.593, 1.398, -0.000)),
        ("OD1", 5, (0.610, 1.091, 0.000)),
        ("OD2", 5, (0.592, -1.101, -0.003)),
    ],
    # "CYS" 残基的原子坐标数据
    "CYS": [
        # 残基中的原子名、相对于残基中心的位置、原子的三维坐标
        ("N", 0, (-0.522, 1.362, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.524, 0.000, 0.000)),
        ("CB", 0, (-0.519, -0.773, -1.212)),
        ("O", 3, (0.625, 1.062, -0.000)),
        ("SG", 4, (0.728, 1.653, 0.000)),
    ],
    # "GLN" 残基的原子坐标数据
    "GLN": [
        # 残基中的原子名、相对于残基中心的位置、原子的三维坐标
        ("N", 0, (-0.526, 1.361, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.525, -0.779, -1.207)),
        ("O", 3, (0.626, 1.062, -0.000)),
        ("CG", 4, (0.615, 1.393, 0.000)),
        ("CD", 5, (0.587, 1.399, -0.000)),
        ("NE2", 6, (0.593, -1.189, -0.001)),
        ("OE1", 6, (0.634, 1.060, 0.000)),
    ],
    # "GLU" 残基的原子坐标数据
    "GLU": [
        # 残基中的原子名、相对于残基中心的位置、原子的三维坐标
        ("N", 0, (-0.528, 1.361, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, -0.000, -0.000)),
        ("CB", 0, (-0.526, -0.781, -1.207)),
        ("O", 3, (0.626, 1.062, 0.000)),
        ("CG", 4, (0.615, 1.392, 0.000)),
        ("CD", 5, (0.600, 1.397, 0.000)),
        ("OE1", 6, (0.607, 1.095, -0.000)),
        ("OE2", 6, (0.589, -1.104, -0.001)),
    ],
    # "GLY" 残基的原子坐标数据
    "GLY": [
        # 残基中的原子名、相对于残基中心的位置、原子的三维坐标
        ("N", 0, (-0.572, 1.337, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.517, -0.000, -0.000)),
        ("O", 3, (0.626, 1.062, -0.000)),
    ],
    # "HIS" 残基的原子坐标数据
    "HIS": [
        # 残基中的原子名、相对于残基中心的位置、原子的三维坐标
        ("N", 0, (-0.527, 1.360, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.525, 0.000, 0.000)),
        ("CB", 0, (-0.525, -0.778, -1.208)),
        ("O", 3, (0.625, 1.063, 0.000)),
        ("CG", 4, (0.600, 1.370, -0.000)),
        ("CD2", 5, (0.889, -1.021, 0.003)),
        ("ND1", 5, (0.744, 1.160, -0.000)),
        ("CE1", 5, (2.030, 0.851, 0.002)),
        ("NE2", 5, (2.145, -0.466, 0.004)),
    ],
    # "ILE" 残基的原子坐标数据
    "ILE": [
        # 残基中的原子名、相对于残基中心的位置、原子的三维坐标
        ("N", 0, (-0.493, 1.373, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.527, -0.000, -0.000)),
        ("CB", 0, (-0.536, -0.793, -1.213)),
        ("O", 3, (0.627, 1.062, -0.000)),
        ("CG1", 4, (0.534, 1.437, -0.000)),
        ("CG2", 4, (0.540, -0.785, -1.199)),
        ("CD1", 5, (0.619, 1.391, 0.000)),
    ],
    # "LEU" 残基的原子坐标数据
    "LEU": [
        # 残基中的原子名、相对于残基中心的位置、原子的三维坐标
        ("N", 0, (-0.520, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.525, -0.000, -0.000)),
        ("CB", 0, (-0.522, -0.773, -1.214)),
        ("O", 3, (0.625, 1.063, -0.000)),
        ("CG", 4, (0.678, 1.371, 0.000)),
        ("CD1", 5, (0.530, 1.430, -0.000)),
        ("CD2", 5, (0.535, -0.774, 1.200)),
    ],
    # 氨基酸Lysine的原子坐标信息
    "LYS": [
        ("N", 0, (-0.526, 1.362, -0.000)),  # 原子类型、序号、坐标
        ("CA", 0, (0.000, 0.000, 0.000)),  # 原子类型、序号、坐标
        ("C", 0, (1.526, 0.000, 0.000)),  # 原子类型、序号、坐标
        ("CB", 0, (-0.524, -0.778, -1.208)),  # 原子类型、序号、坐标
        ("O", 3, (0.626, 1.062, -0.000)),  # 原子类型、序号、坐标
        ("CG", 4, (0.619, 1.390, 0.000)),  # 原子类型、序号、坐标
        ("CD", 5, (0.559, 1.417, 0.000)),  # 原子类型、序号、坐标
        ("CE", 6, (0.560, 1.416, 0.000)),  # 原子类型、序号、坐标
        ("NZ", 7, (0.554, 1.387, 0.000)),  # 原子类型、序号、坐标
    ],
    # 氨基酸Methionine的原子坐标信息
    "MET": [
        ("N", 0, (-0.521, 1.364, -0.000)),  # 原子类型、序号、坐标
        ("CA", 0, (0.000, 0.000, 0.000)),  # 原子类型、序号、坐标
        ("C", 0, (1.525, 0.000, 0.000)),  # 原子类型、序号、坐标
        ("CB", 0, (-0.523, -0.776, -1.210)),  # 原子类型、序号、坐标
        ("O", 3, (0.625, 1.062, -0.000)),  # 原子类型、序号、坐标
        ("CG", 4, (0.613, 1.391, -0.000)),  # 原子类型、序号、坐标
        ("SD", 5, (0.703, 1.695, 0.000)),  # 原子类型、序号、坐标
        ("CE", 6, (0.320, 1.786, -0.000)),  # 原子类型、序号、坐标
    ],
    # 氨基酸Phenylalanine的原子坐标信息
    "PHE": [
        ("N", 0, (-0.518, 1.363, 0.000)),  # 原子类型、序号、坐标
        ("CA", 0, (0.000, 0.000, 0.000)),  # 原子类型、序号、坐标
        ("C", 0, (1.524, 0.000, -0.000)),  # 原子类型、序号、坐标
        ("CB", 0, (-0.525, -0.776, -1.212)),  # 原子类型、序号、坐标
        ("O", 3, (0.626, 1.062, -0.000)),  # 原子类型、序号、坐标
        ("CG", 4, (0.607, 1.377, 0.000)),  # 原子类型、序号、坐标
        ("CD1", 5, (0.709, 1.195, -0.000)),  # 原子类型、序号、坐标
        ("CD2", 5, (0.706, -1.196, 0.000)),  # 原子类型、序号、坐标
        ("CE1", 5, (2.102, 1.198, -0.000)),  # 原子类型、序号、坐标
        ("CE2", 5, (2.098, -1.201, -0.000)),  # 原子类型、序号、坐标
        ("CZ", 5, (2.794, -0.003, -0.001)),  # 原子类型、序号、坐标
    ],
    # 氨基酸Proline的原子坐标信息
    "PRO": [
        ("N", 0, (-0.566, 1.351, -0.000)),  # 原子类型、序号、坐标
        ("CA", 0, (0.000, 0.000, 0.000)),  # 原子类型、序号、坐标
        ("C", 0, (1.527, -0.000, 0.000)),  # 原子类型、序号、坐标
        ("CB", 0, (-0.546, -0.611, -1.293)),  # 原子类型、序号、坐标
        ("O", 3, (0.621, 1.066, 0.000)),  # 原子类型、序号、坐标
        ("CG", 4, (0.382, 1.445, 0.0)),  # 原子类型、序号、坐标
        ("CD", 5, (0.477, 1.424, 0.0)),  # 原子类型、序号、坐标，手动增加2度角度
    ],
    # 氨基酸Serine的原子坐标信息
    "SER": [
        ("N", 0, (-0.529, 1.360, -0.000)),  # 原子类型、序号、坐标
        ("CA", 0, (0.000, 0.000, 0.000)),  # 原子类型、序号、坐标
        ("C", 0, (1.525, -0.000, -0.000)),  # 原子类型、序号、坐标
        ("CB", 0, (-0.518, -0.777, -1.211)),  # 原子类型、序号、坐标
        ("O", 3, (0.626, 1.062, -0.000)),  # 原子类型、序号、坐标
        ("OG", 4, (0.503, 1.325, 0.000)),  # 原子类型、序号、坐标
    ],
    # 氨基酸Threonine的原子坐标信息
    "THR": [
        ("N", 0, (-0.517, 1.364, 0.000)),  # 原子类型、序号、坐标
        ("CA", 0, (0.000, 0.000, 0.000)),  # 原子类型、序号、坐标
        ("C", 0, (1.526, 0.000, -0.000)),  # 原子类型、序号、坐标
        ("CB", 0, (-0.516, -0.793, -1.215)),  # 原子类型、序号、坐标
        ("O", 3, (0.626, 1.062, 0.000)),  # 原子类型、序号、坐标
        ("CG2", 4, (0.550, -0.718, -1.228)),  # 原子类型、序号、坐标
        ("OG1", 4, (0.472, 1.353, 0.000)),  # 原子类型、序号、坐标
    ],
    # 氨基酸Tryptophan的原子坐标信息
    "TRP": [
        ("N", 0, (-0.521, 1.363, 0.000)),  # 原子类型、序号、坐标
        ("CA", 0, (0.000, 0.000, 0.000)),  # 原子类型、序号、坐标
        ("C", 0, (1.525, -0.000, 0.000)),  # 原子类型、序号、坐标
        ("CB", 0, (-0.523, -0.776, -1.212)),  # 原子类型、序号、坐标
        ("O", 3, (0.627, 1.062, 0.000)),  # 原子类型、序号、坐标
        ("CG", 4, (0.609, 1.370, -0.000)),  # 原子类型、序号、坐标
        ("CD1", 5, (0.824, 1.091, 0.000)),  # 原子类型、序号、坐标
        ("CD2", 5, (0.854, -1.148, -0.005)),  # 原子类型、序号、坐标
        ("CE2", 5, (2.186, -0.678, -0.007)),  # 原子类型、序号、坐标
        ("CE3", 5, (0.622, -2.530, -0.007)),  # 原子类型、序号、坐标
        ("NE1", 5, (2.140, 0.690, -0.004)),  # 原子类型、序号、坐标
        ("CH2", 5, (3.028, -2.890, -0.013)),  # 原子类型、序号、坐标
        ("CZ2", 5, (3.283, -1.543, -0.011)),  # 原子类型、序号、坐标
        ("CZ3", 5, (1.715, -3.389, -0.011)),  # 原子类型、序号、坐标
    ],
    # 氨基酸类型"TYR"的原子信息列表，每个原子信息以元组形式存储，包括原子名称、电荷、坐标
    "TYR": [
        # 氢原子"N"，电荷为0，坐标为(-0.522, 1.362, 0.000)
        ("N", 0, (-0.522, 1.362, 0.000)),
        # 钙原子"CA"，电荷为0，坐标为(0.000, 0.000, 0.000)
        ("CA", 0, (0.000, 0.000, 0.000)),
        # 碳原子"C"，电荷为0，坐标为(1.524, -0.000, -0.000)
        ("C", 0, (1.524, -0.000, -0.000)),
        # 氮原子"CB"，电荷为0，坐标为(-0.522, -0.776, -1.213)
        ("CB", 0, (-0.522, -0.776, -1.213)),
        # 氧原子"O"，电荷为3，坐标为(0.627, 1.062, -0.000)
        ("O", 3, (0.627, 1.062, -0.000)),
        # 碳原子"CG"，电荷为4，坐标为(0.607, 1.382, -0.000)
        ("CG", 4, (0.607, 1.382, -0.000)),
        # "CD1"碳原子，电荷为5，坐标为(0.716, 1.195, -0.000)
        ("CD1", 5, (0.716, 1.195, -0.000)),
        # "CD2"碳原子，电荷为5，坐标为(0.713, -1.194, -0.001)
        ("CD2", 5, (0.713, -1.194, -0.001)),
        # "CE1"碳原子，电荷为5，坐标为(2.107, 1.200, -0.002)
        ("CE1", 5, (2.107, 1.200, -0.002)),
        # "CE2"碳原子，电荷为5，坐标为(2.104, -1.201, -0.003)
        ("CE2", 5, (2.104, -1.201, -0.003)),
        # "OH"氧原子，电荷为5，坐标为(4.168, -0.002, -0.005)
        ("OH", 5, (4.168, -0.002, -0.005)),
        # "CZ"碳原子，电荷为5，坐标为(2.791, -0.001, -0.003)
        ("CZ", 5, (2.791, -0.001, -0.003)),
    ],
    
    # 氨基酸类型"VAL"的原子信息列表，每个原子信息以元组形式存储，包括原子名称、电荷、坐标
    "VAL": [
        # 氢原子"N"，电荷为0，坐标为(-0.494, 1.373, -0.000)
        ("N", 0, (-0.494, 1.373, -0.000)),
        # 钙原子"CA"，电荷为0，坐标为(0.000, 0.000, 0.000)
        ("CA", 0, (0.000, 0.000, 0.000)),
        # 碳原子"C"，电荷为0，坐标为(1.527, -0.000, -0.000)
        ("C", 0, (1.527, -0.000, -0.000)),
        # 氮原子"CB"，电荷为0，坐标为(-0.533, -0.795, -1.213)
        ("CB", 0, (-0.533, -0.795, -1.213)),
        # 氧原子"O"，电荷为3，坐标为(0.627, 1.062, -0.000)
        ("O", 3, (0.627, 1.062, -0.000)),
        # "CG1"碳原子，电荷为4，坐标为(0.540, 1.429, -0.000)
        ("CG1", 4, (0.540, 1.429, -0.000)),
        # "CG2"碳原子，电荷为4，坐标为(0.533, -0.776, 1.203)
        ("CG2", 4, (0.533, -0.776, 1.203)),
    ],
# 闭合之前的一个函数定义
}

# 包含每种氨基酸类型的非氢原子列表。PDB 命名约定。
residue_atoms: Dict[str, List[str]] = {
    "ALA": ["C", "CA", "CB", "N", "O"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY": ["C", "CA", "N", "O"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER": ["C", "CA", "CB", "N", "O", "OG"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "N", "NE1", "O"],
    "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
}

# 用于含糊原子命名的命名交换。
residue_atom_renaming_swaps: Dict[str, Dict[str, str]] = {
    "ASP": {"OD1": "OD2"},
    "GLU": {"OE1": "OE2"},
    "PHE": {"CD1": "CD2", "CE1": "CE2"},
    "TYR": {"CD1": "CD2", "CE1": "CE2"},
}

# 原子的范德瓦尔斯半径[埃]（来自维基百科）
van_der_waals_radius: Dict[str, float] = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
}

# 定义了一个名为 Bond 的命名元组
Bond = collections.namedtuple("Bond", ["atom1_name", "atom2_name", "length", "stddev"])
# 定义了一个名为 BondAngle 的命名元组
BondAngle = collections.namedtuple(
    "BondAngle",
    ["atom1_name", "atom2_name", "atom3name", "angle_rad", "stddev"],
)

# 将结构映射到原子顺序的函数
def map_structure_with_atom_order(in_list: list, first_call: bool = True) -> list:
    # 将嵌套列表中的字符串映射到 atom_order 中相应的索引
    if first_call:
        in_list = copy.deepcopy(in_list)
    for i in range(len(in_list)):
        if isinstance(in_list[i], list):
            in_list[i] = map_structure_with_atom_order(in_list[i], first_call=False)
        elif isinstance(in_list[i], str):
            in_list[i] = atom_order[in_list[i]]
        else:
            raise ValueError("Unexpected type when mapping nested lists!")
    return in_list

# 使用 lru_cache 装饰器定义一个函数，用于加载立体化学属性
@functools.lru_cache(maxsize=None)
def load_stereo_chemical_props() -> (
    # 元组类型，包含三个元素，每个元素都是一个映射，键为字符串，值为列表
    Tuple[
        # 映射，键为字符串，值为 Bond 对象列表
        Mapping[str, List[Bond]],
        # 映射，键为字符串，值为 Bond 对象列表
        Mapping[str, List[Bond]],
        # 映射，键为字符串，值为 BondAngle 对象列表
        Mapping[str, List[BondAngle]],
    ]
    # 将 stereo_chemical_props.txt 文件加载到一个良好的结构中。

    # 读取文献中的键长和键角的值，并将键角转换为三角形的对边长度 ("residue_virtual_bonds")。

    # 返回:
    #   residue_bonds: 将 resname 映射到 Bond 元组列表的字典
    #   residue_virtual_bonds: 将 resname 映射到 Bond 元组列表的字典
    #   residue_bond_angles: 将 resname 映射到 BondAngle 元组列表的字典
    # TODO: 应该在设置脚本中下载这个文件
    stereo_chemical_props = resources.read_text("openfold.resources", "stereo_chemical_props.txt")

    # 创建行迭代器
    lines_iter = iter(stereo_chemical_props.splitlines())
    # 加载键长。
    residue_bonds: Dict[str, List[Bond]] = {}
    next(lines_iter)  # 跳过标题行。
    for line in lines_iter:
        # 如果行只包含一个连字符，则退出循环。
        if line.strip() == "-":
            break
        # 分解行，提取键、resname、键长和标准差。
        bond, resname, bond_length, stddev = line.split()
        # 分解键，获取原子1和原子2的名称。
        atom1, atom2 = bond.split("-")
        # 如果 resname 不在 residue_bonds 中，则创建一个空列表。
        if resname not in residue_bonds:
            residue_bonds[resname] = []
        # 将 Bond 对象添加到对应的 resname 的列表中。
        residue_bonds[resname].append(Bond(atom1, atom2, float(bond_length), float(stddev)))
    # 添加一个默认的 "UNK" 键来存储未知的键长。
    residue_bonds["UNK"] = []

    # 加载键角。
    residue_bond_angles: Dict[str, List[BondAngle]] = {}
    next(lines_iter)  # 跳过空行。
    next(lines_iter)  # 跳过标题行。
    for line in lines_iter:
        # 如果行只包含一个连字符，则退出循环。
        if line.strip() == "-":
            break
        # 分解行，提取键、resname、角度和标准差。
        bond, resname, angle_degree, stddev_degree = line.split()
        # 分解键，获取原子1、原子2和原子3的名称。
        atom1, atom2, atom3 = bond.split("-")
        # 如果 resname 不在 residue_bond_angles 中，则创建一个空列表。
        if resname not in residue_bond_angles:
            residue_bond_angles[resname] = []
        # 将 BondAngle 对象添加到对应的 resname 的列表中。
        residue_bond_angles[resname].append(
            BondAngle(
                atom1,
                atom2,
                atom3,
                float(angle_degree) / 180.0 * np.pi,
                float(stddev_degree) / 180.0 * np.pi,
            )
        )
    # 添加一个默认的 "UNK" 键来存储未知的键角。
    residue_bond_angles["UNK"] = []

    def make_bond_key(atom1_name: str, atom2_name: str) -> str:
        """生成用于查找键的唯一键。"""
        # 将原子名称按字母顺序排序并连接成一个字符串作为键。
        return "-".join(sorted([atom1_name, atom2_name]))

    # 将键角转换为距离 ("virtual bonds")。
    residue_virtual_bonds: Dict[str, List[Bond]] = {}
    # 遍历给定的残基键角信息字典
    for resname, bond_angles in residue_bond_angles.items():
        # 创建一个快速查找键长的字典
        bond_cache: Dict[str, Bond] = {}
        # 遍历每个残基键的信息，将键名对应到键对象
        for b in residue_bonds[resname]:
            bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b
        # 初始化残基虚拟键的列表
        residue_virtual_bonds[resname] = []
        # 遍历每个键角信息
        for ba in bond_angles:
            # 查找键对象
            bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
            bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]

            # 根据余弦定理计算 atom1 和 atom3 之间的距离
            # c^2 = a^2 + b^2 - 2ab*cos(gamma).
            gamma = ba.angle_rad
            length = np.sqrt(bond1.length**2 + bond2.length**2 - 2 * bond1.length * bond2.length * np.cos(gamma))

            # 使用未相关错误的不确定性传播
            dl_outer = 0.5 / length
            dl_dgamma = (2 * bond1.length * bond2.length * np.sin(gamma)) * dl_outer
            dl_db1 = (2 * bond1.length - 2 * bond2.length * np.cos(gamma)) * dl_outer
            dl_db2 = (2 * bond2.length - 2 * bond1.length * np.cos(gamma)) * dl_outer
            stddev = np.sqrt(
                (dl_dgamma * ba.stddev) ** 2 + (dl_db1 * bond1.stddev) ** 2 + (dl_db2 * bond2.stddev) ** 2
            )
            # 将计算出的虚拟键信息添加到残基虚拟键列表中
            residue_virtual_bonds[resname].append(Bond(ba.atom1_name, ba.atom3name, length, stddev))

    # 返回结果，包括残基键信息、残基虚拟键信息和残基键角信息
    return (residue_bonds, residue_virtual_bonds, residue_bond_angles)
# 一般键与脯氨酸键（第一个元素）以及脯氨酸（第二个元素）之间的键长。
# 第一个元素表示一般键，第二个元素表示脯氨酸键。
between_res_bond_length_c_n: Tuple[float, float] = (1.329, 1.341)
# 一般键与脯氨酸键之间的键长的标准偏差。
between_res_bond_length_stddev_c_n: Tuple[float, float] = (0.014, 0.016)

# 两个残基之间的 cos 角。
between_res_cos_angles_c_n_ca: Tuple[float, float] = (-0.5203, 0.0353)  # degrees: 121.352 +- 2.315
between_res_cos_angles_ca_c_n: Tuple[float, float] = (-0.4473, 0.0311)  # degrees: 116.568 +- 1.995

# 当需要以需要为每个残基固定原子数据大小的格式存储原子数据时使用此映射（例如 numpy 数组）。
atom_types: List[str] = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
# 将原子类型映射到其索引的字典。
atom_order: Dict[str, int] = {atom_type: i for i, atom_type in enumerate(atom_types)}
# 原子类型的数量。
atom_type_num = len(atom_types)  # := 37.

# 14 列的紧凑原子编码。
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_atom14_names: Dict[str, List[str]] = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
}
    # 定义了键为"VAL"的列表，包含了常见氨基酸VAL的原子名称，其中前四个是主链原子，后面的是侧链原子
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
    # 定义了键为"UNK"的列表，包含了未知氨基酸UNK的原子名称，其中没有已知的原子
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
# pylint: enable=line-too-long
# pylint: enable=bad-whitespace

# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
# 标准的残基顺序，将3字母的氨基酸代码按字母顺序排序得到
restypes: List[str] = [
    "A",  # Alanine
    "R",  # Arginine
    "N",  # Asparagine
    "D",  # Aspartic acid
    "C",  # Cysteine
    "Q",  # Glutamine
    "E",  # Glutamic acid
    "G",  # Glycine
    "H",  # Histidine
    "I",  # Isoleucine
    "L",  # Leucine
    "K",  # Lysine
    "M",  # Methionine
    "F",  # Phenylalanine
    "P",  # Proline
    "S",  # Serine
    "T",  # Threonine
    "W",  # Tryptophan
    "Y",  # Tyrosine
    "V",  # Valine
]
restype_order: Dict[str, int] = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

# Create a list including "X" for unknown amino acids
restypes_with_x: List[str] = restypes + ["X"]
# Create a mapping of amino acids to integers including "X"
restype_order_with_x: Dict[str, int] = {restype: i for i, restype in enumerate(restypes_with_x)}

def sequence_to_onehot(sequence: str, mapping: Mapping[str, int], map_unknown_to_x: bool = False) -> np.ndarray:
    """Maps the given sequence into a one-hot encoded matrix.

    Args:
      sequence: An amino acid sequence.
      mapping: A dictionary mapping amino acids to integers.
      map_unknown_to_x: If True, any amino acid that is not in the mapping will be
        mapped to the unknown amino acid 'X'. If the mapping doesn't contain amino acid 'X', an error will be thrown.
        If False, any amino acid not in the mapping will throw an error.

    Returns:
      A numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of the sequence.

    Raises:
      ValueError: If the mapping doesn't contain values from 0 to
        num_unique_aas - 1 without any gaps.
    """
    num_entries = max(mapping.values()) + 1

    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError(
            "The mapping must have values from 0 to num_unique_aas-1 without any gaps. Got: %s"
            % sorted(mapping.values())
        )

    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

    for aa_index, aa_type in enumerate(sequence):
        if map_unknown_to_x:
            if aa_type.isalpha() and aa_type.isupper():
                aa_id = mapping.get(aa_type, mapping["X"])
            else:
                raise ValueError(f"Invalid character in the sequence: {aa_type}")
        else:
            aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1

    return one_hot_arr

restype_1to3: Dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
# 将restype_1to3字典中的值和键进行反转，生成restype_3to1字典
restype_3to1: Dict[str, str] = {v: k for k, v in restype_1to3.items()}

# 为所有未知残基定义一个restype名称
unk_restype = "UNK"

# 生成包含所有残基名称的列表
resnames: List[str] = [restype_1to3[r] for r in restypes] + [unk_restype]

# 生成一个将残基名映射到索引的字典
resname_to_idx: Dict[str, int] = {resname: i for i, resname in enumerate(resnames)}

# 使用hhblits约定进行映射的字典，其中B映射到D，J和O映射到X，U映射到C，Z映射到E
HHBLITS_AA_TO_ID: Dict[str, int] = {
    # ... (中间代码省略)
}

# HHBLITS_AA_TO_ID的部分反转
ID_TO_HHBLITS_AA: Dict[int, str] = {
    # ... (中间代码省略)
}

# 包含X和-的残基类型列表
restypes_with_x_and_gap: List[str] = restypes + ["X", "-"]

# 将HHBLITS_AA_TO_ID转换为我们的AATYPE的元组
MAP_HHBLITS_AATYPE_TO_OUR_AATYPE: Tuple[int, ...] = tuple(
    restypes_with_x_and_gap.index(ID_TO_HHBLITS_AA[i]) for i in range(len(restypes_with_x_and_gap))

# 生成[restype_num + 1, atom_type_num]的零数组
def _make_standard_atom_mask() -> np.ndarray:
    """返回[num_res_types, num_atom_types]的掩码数组。"""
    # +1用于未知的情况（全为0）。
    # 遍历所有残基，为每个残基的原子类型设置对应的位置为1
    mask = np.zeros([restype_num + 1, atom_type_num], dtype=np.int32)
    for restype, restype_letter in enumerate(restypes):
        restype_name = restype_1to3[restype_letter]
        atom_names = residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            mask[restype, atom_type] = 1
    return mask

# 保存标准原子掩码数组
STANDARD_ATOM_MASK = _make_standard_atom_mask()

#定义每个残基的chi角度的原子
def chi_angle_atom(atom_index: int) -> np.ndarray:
    """通过one-hot表示定义chi角度刚性组。"""
    chi_angles_index = {}
    one_hots = []

    # 生成每个残基的chi角度的索引
    # 生成每个残基的chi角度的one-hot表示
    for k, v in chi_angles_atoms.items():
        indices = [atom_types.index(s[atom_index]) for s in v]
        indices.extend([-1] * (4 - len(indices)))
        chi_angles_index[k] = indices

    # 为每种残基生成one-hot表示
    for r in restypes:
        res3 = restype_1to3[r]
        one_hot = np.eye(atom_type_num)[chi_angles_index[res3]]
        one_hots.append(one_hot
    # 在 one_hots 列表中添加一个全为零的数组，用于表示残基 `X`。
    one_hots.append(np.zeros([4, atom_type_num]))
    # 将 one_hots 列表中的数组堆叠起来，沿着 axis=0 方向
    one_hot = np.stack(one_hots, axis=0)
    # 对堆叠后的数组进行转置，将第二维和第三维交换位置
    one_hot = np.transpose(one_hot, [0, 2, 1])

    # 返回转置后的数组作为结果
    return one_hot
# 使用 chi_angle_atom 函数得到第一个原子的 one-hot 编码
chi_atom_1_one_hot = chi_angle_atom(1)
# 使用 chi_angle_atom 函数得到第二个原子的 one-hot 编码
chi_atom_2_one_hot = chi_angle_atom(2)

# 声明一个三维列表，用于保存氨基酸的 chi 角的原子索引，按照索引顺序而非名称
chi_angles_atom_indices_list: List[List[List[str]]] = [chi_angles_atoms[restype_1to3[r]] for r in restypes]
# 根据原子顺序进行映射结构，得到 chi 角的原子索引
chi_angles_atom_indices_ours: list = map_structure_with_atom_order(chi_angles_atom_indices_list)
# 创建一个 NumPy 数组，保存 chi 角的原子索引并进行填充
chi_angles_atom_indices = np.array(
    [chi_atoms + ([[0, 0, 0, 0]] * (4 - len(chi_atoms))) for chi_atoms in chi_angles_atom_indices_list]
)

# 用于保存原子 (res_name, atom_name) 对应的 chi 组索引和组内原子索引的字典
chi_groups_for_atom: Dict[Tuple[str, str], List[Tuple[int, int]]] = collections.defaultdict(list)
# 遍历氨基酸和其对应的所有 chi 角，填充 chi_groups_for_atom 字典
for res_name, chi_angle_atoms_for_res in chi_angles_atoms.items():
    for chi_group_i, chi_group in enumerate(chi_angle_atoms_for_res):
        for atom_i, atom in enumerate(chi_group):
            chi_groups_for_atom[(res_name, atom)].append((chi_group_i, atom_i))
# 将默认字典转换为普通字典
chi_groups_for_atom = dict(chi_groups_for_atom)

def _make_rigid_transformation_4x4(ex: np.ndarray, ey: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create a rigid 4x4 transformation matrix from two axes and transl."""
    # 归一化 ex 向量
    ex_normalized = ex / np.linalg.norm(ex)

    # 使 ey 向量垂直于 ex 向量
    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)

    # 计算 ez 作为 ex 和 ey 的叉乘
    eznorm = np.cross(ex_normalized, ey_normalized)
    # 构造 4x4 刚体变换矩阵
    m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
    m = np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    return m

# 创建多个 NumPy 数组用于存储数据
restype_atom37_to_rigid_group = np.zeros([21, 37], dtype=int)
restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
restype_atom37_rigid_group_positions = np.zeros([21, 37, 3], dtype=np.float32)
restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=int)
restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)
restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)
restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)

def _make_rigid_group_constants() -> None:
    """Fill the arrays above."""
    # 遍历刚性组中的原子类型和对应的字母表示
    for restype, restype_letter in enumerate(restypes):
        # 根据字母表示获取对应的原子类型名
        resname = restype_1to3[restype_letter]
        # 遍历刚性组原子的名称、组索引和位置
        for atomname, group_idx, atom_position in rigid_group_atom_positions[resname]:
            # 根据原子名获取对应的原子类型
            atomtype = atom_order[atomname]
            # 将原子编号映射到刚性组的索引
            restype_atom37_to_rigid_group[restype, atomtype] = group_idx
            # 设置原子类型为1
            restype_atom37_mask[restype, atomtype] = 1
            # 将原子的刚性组位置信息保存到数组中
            restype_atom37_rigid_group_positions[restype, atomtype, :] = atom_position
    
            # 获取原子名在原子14种的索引
            atom14idx = restype_name_to_atom14_names[resname].index(atomname)
            # 将原子14编号映射到刚性组的索引
            restype_atom14_to_rigid_group[restype, atom14idx] = group_idx
            # 设置原子14类型为1
            restype_atom14_mask[restype, atom14idx] = 1
            # 将原子14的刚性组位置信息保存到数组中
            restype_atom14_rigid_group_positions[restype, atom14idx, :] = atom_position
    # 遍历 restypes 列表，同时获得索引和元素值
    for restype, restype_letter in enumerate(restypes):
        # 将 restype_letter 对应到 restype_1to3 字典里得到三字母缩写
        resname = restype_1to3[restype_letter]
        # 创建一个字典，键是原子名，值是原子位置的 numpy 数组
        atom_positions: Dict[str, np.ndarray] = {
            name: np.array(pos) for name, _, pos in rigid_group_atom_positions[resname]
        }

        # 将刚体组的默认刚性变换矩阵设置为单位矩阵
        restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)

        # 将刚体组的默认刚性变换矩阵设置为单位矩阵
        restype_rigid_group_default_frame[restype, 1, :, :] = np.eye(4)

        # 计算 phi-frame 到 backbone 的刚性变换矩阵
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["N"] - atom_positions["CA"],
            ey=np.array([1.0, 0.0, 0.0]),
            translation=atom_positions["N"],
        )
        restype_rigid_group_default_frame[restype, 2, :, :] = mat

        # 计算 psi-frame 到 backbone 的刚性变换矩阵
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C"] - atom_positions["CA"],
            ey=atom_positions["CA"] - atom_positions["N"],
            translation=atom_positions["C"],
        )
        restype_rigid_group_default_frame[restype, 3, :, :] = mat

        # 如果 chi1 角度存在，计算 chi1-frame 到 backbone 的刚性变换矩阵
        if chi_angles_mask[restype][0]:
            base_atom_names = chi_angles_atoms[resname][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2],
            )
            restype_rigid_group_default_frame[restype, 4, :, :] = mat

        # 遍历计算 chi2-frame 到 chi1-frame，chi3-frame 到 chi2-frame，chi4-frame 到 chi3-frame 的刚性变换矩阵
        # 幸运的是，所有下一个框架的旋转轴都始于前一个框架的 (0,0,0) 点
        for chi_idx in range(1, 4):
            if chi_angles_mask[restype][chi_idx]:
                axis_end_atom_name = chi_angles_atoms[resname][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                mat = _make_rigid_transformation_4x4(
                    ex=axis_end_atom_position,
                    ey=np.array([-1.0, 0.0, 0.0]),
                    translation=axis_end_atom_position,
                )
                restype_rigid_group_default_frame[restype, 4 + chi_idx, :, :] = mat
# 调用函数_make_rigid_group_constants()，初始化一些常数
_make_rigid_group_constants()


def make_atom14_dists_bounds(
    overlap_tolerance: float = 1.5,  # 设置碰撞容忍度的阈值，默认为1.5
    bond_length_tolerance_factor: int = 15,  # 设置键长容忍度因子的阈值，默认为15
) -> Dict[str, np.ndarray]:
    """计算键的上下界以评估违规。"""
    # 初始化键的下界、上界和标准差数组
    restype_atom14_bond_lower_bound = np.zeros([21, 14, 14], np.float32)
    restype_atom14_bond_upper_bound = np.zeros([21, 14, 14], np.float32)
    restype_atom14_bond_stddev = np.zeros([21, 14, 14], np.float32)
    # 载入氨基酸的键和虚拟键信息
    residue_bonds, residue_virtual_bonds, _ = load_stereo_chemical_props()
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        atom_list = restype_name_to_atom14_names[resname]

        # 创建碰撞的下界和上界
        for atom1_idx, atom1_name in enumerate(atom_list):
            if not atom1_name:
                continue
            atom1_radius = van_der_waals_radius[atom1_name[0]]
            for atom2_idx, atom2_name in enumerate(atom_list):
                if (not atom2_name) or atom1_idx == atom2_idx:
                    continue
                atom2_radius = van_der_waals_radius[atom2_name[0]]
                lower = atom1_radius + atom2_radius - overlap_tolerance
                upper = 1e10  # 设置一个大的上界
                restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
                restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
                restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
                restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper

        # 覆盖键和角度的下界和上界
        for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
            atom1_idx = atom_list.index(b.atom1_name)
            atom2_idx = atom_list.index(b.atom2_name)
            lower = b.length - bond_length_tolerance_factor * b.stddev
            upper = b.length + bond_length_tolerance_factor * b.stddev
            restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
            restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
            restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
            restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
            restype_atom14_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
            restype_atom14_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
    # 返回键的下界、上界和标准差的字典
    return {
        "lower_bound": restype_atom14_bond_lower_bound,  # shape (21,14,14)
        "upper_bound": restype_atom14_bond_upper_bound,  # shape (21,14,14)
        "stddev": restype_atom14_bond_stddev,  # shape (21,14,14)
    }


# 初始化含有歧义原子的数组
restype_atom14_ambiguous_atoms = np.zeros((21, 14), dtype=np.float32)
# 创建一个用于交换索引的数组
restype_atom14_ambiguous_atoms_swap_idx: np.ndarray = np.tile(np.arange(14, dtype=int), (21, 1))


def _make_atom14_ambiguity_feats() -> None:
    # 遍历残基原子重命名交换的字典，其中包含残基和其需要重命名的原子对
    for res, pairs in residue_atom_renaming_swaps.items():
        # 根据残基的三字母缩写找到其在残基类型顺序列表中的索引
        res_idx = restype_order[restype_3to1[res]]
        # 遍历每对需要重命名的原子
        for atom1, atom2 in pairs.items():
            # 找到原子1在该残基下的14种原子名称列表中的索引
            atom1_idx = restype_name_to_atom14_names[res].index(atom1)
            # 找到原子2在该残基下的14种原子名称列表中的索引
            atom2_idx = restype_name_to_atom14_names[res].index(atom2)
            # 在残基-14种原子索引的模糊原子标记数组中，标记原子1的索引为1
            restype_atom14_ambiguous_atoms[res_idx, atom1_idx] = 1
            # 在残基-14种原子索引的模糊原子标记数组中，标记原子2的索引为1
            restype_atom14_ambiguous_atoms[res_idx, atom2_idx] = 1
            # 记录原子交换对应的索引，将原子1的索引指向原子2的索引
            restype_atom14_ambiguous_atoms_swap_idx[res_idx, atom1_idx] = atom2_idx
            # 记录原子交换对应的索引，将原子2的索引指向原子1的索引
            restype_atom14_ambiguous_atoms_swap_idx[res_idx, atom2_idx] = atom1_idx
# 调用一个生成原子的14种不确定性特征的函数
_make_atom14_ambiguity_feats()

# 定义一个函数，将氨基酸类型的序列（整数序列）转换为字符串序列
def aatype_to_str_sequence(aatype: Sequence[int]) -> str:
    # 通过列表推导式，将整数索引转换为对应的氨基酸类型的字符串，并将结果连接成一个字符串
    return "".join([restypes_with_x[aatype[i]] for i in range(len(aatype))])
```