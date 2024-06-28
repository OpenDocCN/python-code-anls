# `.\models\esm\openfold_utils\residue_constants.py`

```
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants used in AlphaFold."""

import collections
import copy
import functools
from importlib import resources
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


# Internal import (35fd).


# Distance from one CA to next CA [trans configuration: omega = 180].
ca_ca = 3.80209737096

# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
chi_angles_atoms: Dict[str, List[List[str]]] = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
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

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
chi_angles_mask: List[List[float]] = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    # The following entries are truncated for brevity.
    # They follow the same pattern of defining chi angle masks for each amino acid type.
    # To maintain code block integrity, they are not fully commented here.
    # Please refer to the original source for detailed explanations.
    # Each sublist corresponds to an amino acid type and its associated chi angle masks.
    # 下面是一个二维列表，每行代表一个氨基酸的属性向量
    [
        [1.0, 1.0, 0.0, 0.0],  # HIS - 氨基酸组合：亮度、橙色、宽度、高度
        [1.0, 1.0, 0.0, 0.0],  # ILE - 氨基酸组合：亮度、橙色、宽度、高度
        [1.0, 1.0, 0.0, 0.0],  # LEU - 氨基酸组合：亮度、橙色、宽度、高度
        [1.0, 1.0, 1.0, 1.0],  # LYS - 氨基酸组合：亮度、橙色、宽度、高度
        [1.0, 1.0, 1.0, 0.0],  # MET - 氨基酸组合：亮度、橙色、宽度、高度
        [1.0, 1.0, 0.0, 0.0],  # PHE - 氨基酸组合：亮度、橙色、宽度、高度
        [1.0, 1.0, 0.0, 0.0],  # PRO - 氨基酸组合：亮度、橙色、宽度、高度
        [1.0, 0.0, 0.0, 0.0],  # SER - 氨基酸组合：亮度、橙色、宽度、高度
        [1.0, 0.0, 0.0, 0.0],  # THR - 氨基酸组合：亮度、橙色、宽度、高度
        [1.0, 1.0, 0.0, 0.0],  # TRP - 氨基酸组合：亮度、橙色、宽度、高度
        [1.0, 1.0, 0.0, 0.0],  # TYR - 氨基酸组合：亮度、橙色、宽度、高度
        [1.0, 0.0, 0.0, 0.0],  # VAL - 氨基酸组合：亮度、橙色、宽度、高度
    ]
# 下面的 chi 角度是 pi 周期性的：它们可以通过多个 pi 的旋转而不影响结构。
# 每一行对应一种氨基酸，列出了其四个 chi 角度的初始值。每个角度是以弧度表示的。
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

# 原子的位置相对于8个刚性组的轴端原子，由预omega、phi、psi和chi角度定义：
# 0: '骨架组',
# 1: '预omega组', (空)
# 2: 'phi组', (当前为空，因为它只定义了氢原子)
# 3: 'psi组',
# 4,5,6,7: 'chi1,2,3,4组'
# 原子位置是相对于相应旋转轴的轴端原子的坐标。x轴沿着旋转轴方向，y轴定义为使二面角定义原子（chi_angles_atoms中的最后一个条目）在xy平面上（y坐标为正）。
rigid_group_atom_positions: Dict[str, List[Tuple[str, int, Tuple[float, float, float]]]] = {
    "ALA": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, -0.000, -0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
    ],
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
    # 氨基酸 ASP 的原子坐标信息，每个元素是一个元组，包含原子名称、类型、坐标
    "ASP": [
        ("N", 0, (-0.525, 1.362, -0.000)),  # 氮原子 N，类型 0，坐标 (-0.525, 1.362, -0.000)
        ("CA", 0, (0.000, 0.000, 0.000)),   # 碳α原子 CA，类型 0，坐标 (0.000, 0.000, 0.000)
        ("C", 0, (1.527, 0.000, -0.000)),   # 碳原子 C，类型 0，坐标 (1.527, 0.000, -0.000)
        ("CB", 0, (-0.526, -0.778, -1.208)),# 碳β原子 CB，类型 0，坐标 (-0.526, -0.778, -1.208)
        ("O", 3, (0.626, 1.062, -0.000)),   # 氧原子 O，类型 3，坐标 (0.626, 1.062, -0.000)
        ("CG", 4, (0.593, 1.398, -0.000)),  # 碳γ原子 CG，类型 4，坐标 (0.593, 1.398, -0.000)
        ("OD1", 5, (0.610, 1.091, 0.000)),  # 羟基原子 OD1，类型 5，坐标 (0.610, 1.091, 0.000)
        ("OD2", 5, (0.592, -1.101, -0.003)),# 羟基原子 OD2，类型 5，坐标 (0.592, -1.101, -0.003)
    ],
    # 氨基酸 CYS 的原子坐标信息，每个元素是一个元组，包含原子名称、类型、坐标
    "CYS": [
        ("N", 0, (-0.522, 1.362, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.524, 0.000, 0.000)),
        ("CB", 0, (-0.519, -0.773, -1.212)),
        ("O", 3, (0.625, 1.062, -0.000)),
        ("SG", 4, (0.728, 1.653, 0.000)),
    ],
    # 氨基酸 GLN 的原子坐标信息，每个元素是一个元组，包含原子名称、类型、坐标
    "GLN": [
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
    # 氨基酸 GLU 的原子坐标信息，每个元素是一个元组，包含原子名称、类型、坐标
    "GLU": [
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
    # 氨基酸 GLY 的原子坐标信息，每个元素是一个元组，包含原子名称、类型、坐标
    "GLY": [
        ("N", 0, (-0.572, 1.337, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.517, -0.000, -0.000)),
        ("O", 3, (0.626, 1.062, -0.000)),
    ],
    # 氨基酸 HIS 的原子坐标信息，每个元素是一个元组，包含原子名称、类型、坐标
    "HIS": [
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
    # 氨基酸 ILE 的原子坐标信息，每个元素是一个元组，包含原子名称、类型、坐标
    "ILE": [
        ("N", 0, (-0.493, 1.373, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.527, -0.000, -0.000)),
        ("CB", 0, (-0.536, -0.793, -1.213)),
        ("O", 3, (0.627, 1.062, -0.000)),
        ("CG1", 4, (0.534, 1.437, -0.000)),
        ("CG2", 4, (0.540, -0.785, -1.199)),
        ("CD1", 5, (0.619, 1.391, 0.000)),
    ],
    # 氨基酸 LEU 的原子坐标信息，每个元素是一个元组，包含原子名称、类型、坐标
    "LEU": [
        ("N", 0, (-0.520, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.525, -0.000, -0.000)),
        ("CB", 0, (-0.522, -0.773, -1.214)),
        ("O", 3, (0.625, 1.063, -0.000)),
        ("CG", 4, (0.678, 1.371, 0.000)),
        ("CD1", 5, (0.530, 1.430, -0.000)),
        ("CD2", 5, (0.535, -0.774, 1.200)),
    ],
    "LYS": [
        ("N", 0, (-0.526, 1.362, -0.000)),  # 原子名称"N"，电荷状态0，坐标(-0.526, 1.362, -0.000)
        ("CA", 0, (0.000, 0.000, 0.000)),   # 原子名称"CA"，电荷状态0，坐标(0.000, 0.000, 0.000)
        ("C", 0, (1.526, 0.000, 0.000)),    # 原子名称"C"，电荷状态0，坐标(1.526, 0.000, 0.000)
        ("CB", 0, (-0.524, -0.778, -1.208)),# 原子名称"CB"，电荷状态0，坐标(-0.524, -0.778, -1.208)
        ("O", 3, (0.626, 1.062, -0.000)),   # 原子名称"O"，电荷状态3，坐标(0.626, 1.062, -0.000)
        ("CG", 4, (0.619, 1.390, 0.000)),   # 原子名称"CG"，电荷状态4，坐标(0.619, 1.390, 0.000)
        ("CD", 5, (0.559, 1.417, 0.000)),   # 原子名称"CD"，电荷状态5，坐标(0.559, 1.417, 0.000)
        ("CE", 6, (0.560, 1.416, 0.000)),   # 原子名称"CE"，电荷状态6，坐标(0.560, 1.416, 0.000)
        ("NZ", 7, (0.554, 1.387, 0.000)),   # 原子名称"NZ"，电荷状态7，坐标(0.554, 1.387, 0.000)
    ],
    "MET": [
        ("N", 0, (-0.521, 1.364, -0.000)),  # 原子名称"N"，电荷状态0，坐标(-0.521, 1.364, -0.000)
        ("CA", 0, (0.000, 0.000, 0.000)),   # 原子名称"CA"，电荷状态0，坐标(0.000, 0.000, 0.000)
        ("C", 0, (1.525, 0.000, 0.000)),    # 原子名称"C"，电荷状态0，坐标(1.525, 0.000, 0.000)
        ("CB", 0, (-0.523, -0.776, -1.210)),# 原子名称"CB"，电荷状态0，坐标(-0.523, -0.776, -1.210)
        ("O", 3, (0.625, 1.062, -0.000)),   # 原子名称"O"，电荷状态3，坐标(0.625, 1.062, -0.000)
        ("CG", 4, (0.613, 1.391, -0.000)),   # 原子名称"CG"，电荷状态4，坐标(0.613, 1.391, -0.000)
        ("SD", 5, (0.703, 1.695, 0.000)),    # 原子名称"SD"，电荷状态5，坐标(0.703, 1.695, 0.000)
        ("CE", 6, (0.320, 1.786, -0.000)),   # 原子名称"CE"，电荷状态6，坐标(0.320, 1.786, -0.000)
    ],
    "PHE": [
        ("N", 0, (-0.518, 1.363, 0.000)),   # 原子名称"N"，电荷状态0，坐标(-0.518, 1.363, 0.000)
        ("CA", 0, (0.000, 0.000, 0.000)),   # 原子名称"CA"，电荷状态0，坐标(0.000, 0.000, 0.000)
        ("C", 0, (1.524, 0.000, -0.000)),   # 原子名称"C"，电荷状态0，坐标(1.524, 0.000, -0.000)
        ("CB", 0, (-0.525, -0.776, -1.212)),# 原子名称"CB"，电荷状态0，坐标(-0.525, -0.776, -1.212)
        ("O", 3, (0.626, 1.062, -0.000)),   # 原子名称"O"，电荷状态3，坐标(0.626, 1.062, -0.000)
        ("CG", 4, (0.607, 1.377, 0.000)),   # 原子名称"CG"，电荷状态4，坐标(0.607, 1.377, 0.000)
        ("CD1", 5, (0.709, 1.195, -0.000)), # 原子名称"CD1"，电荷状态5，坐标(0.709, 1.195, -0.000)
        ("CD2", 5, (0.706, -1.196, 0.000)),  # 原子名称"CD2"，电荷状态5，坐标(0.706, -1.196, 0.000)
        ("CE1", 5, (2.102, 1.198, -0.000)),  # 原子名称"CE1"，电荷状态5，坐标(2.102, 1.198, -0.000)
        ("CE2", 5, (2.098, -1.201, -0.000)), # 原子名称"CE2"，电荷状态5，坐标(2.098, -1.201, -0.000)
        ("CZ", 5, (2.794, -0.003, -0.001)),  # 原子名称"CZ"，电荷状态5，坐标(2.794, -0.003, -0.001)
    ],
    "PRO": [
        ("N", 0, (-0.566, 1.351, -0.000)),  # 原子名称"N"，电荷状态0，坐标(-0.566, 1.351, -0.000)
        ("CA", 0, (0.000, 0.000, 0.000)),   # 原子名称"CA"，电荷状态0，坐标(0.000, 0.000, 0.000)
        ("C", 0, (1.527, -0.000, 0.000)),   # 原子名称"C"，电荷状态0，坐标(1.527, -0.000, 0.000)
        ("CB", 0, (-0.546, -0.611, -1.293)),# 原子名称"CB"，电荷状态0，坐标(-0.546, -0.611, -1.293)
        ("O", 3, (0.621, 1.066, 0.000)),    # 原子名称"O"，电荷状态3，坐标(0.621, 1.066, 0.000)
        ("CG", 4, (0.382, 1.445, 0.0)),     # 原子名称"CG"，电荷状态4，坐标(0.382, 1.445, 0.0)
        ("CD", 5, (0.477, 1.424, 0.0)),     # 原子名称"CD"，电荷状态5，坐标(0.477, 1.424, 0.0)
        # ('CD', 5, (0.427, 1.440, 0.0)),   # 注释
    "TYR": [  # TYR 残基的描述开始
        ("N", 0, (-0.522, 1.362, 0.000)),  # 残基的氮原子，索引为 0，坐标为 (-0.522, 1.362, 0.000)
        ("CA", 0, (0.000, 0.000, 0.000)),  # 残基的α-碳原子，索引为 0，坐标为 (0.000, 0.000, 0.000)
        ("C", 0, (1.524, -0.000, -0.000)),  # 残基的碳原子，索引为 0，坐标为 (1.524, -0.000, -0.000)
        ("CB", 0, (-0.522, -0.776, -1.213)),  # 残基的侧链碳原子，索引为 0，坐标为 (-0.522, -0.776, -1.213)
        ("O", 3, (0.627, 1.062, -0.000)),  # 残基的氧原子，索引为 3，坐标为 (0.627, 1.062, -0.000)
        ("CG", 4, (0.607, 1.382, -0.000)),  # 残基的芳香环的碳原子，索引为 4，坐标为 (0.607, 1.382, -0.000)
        ("CD1", 5, (0.716, 1.195, -0.000)),  # 残基的芳香环的第一个碳原子，索引为 5，坐标为 (0.716, 1.195, -0.000)
        ("CD2", 5, (0.713, -1.194, -0.001)),  # 残基的芳香环的第二个碳原子，索引为 5，坐标为 (0.713, -1.194, -0.001)
        ("CE1", 5, (2.107, 1.200, -0.002)),  # 残基的芳香环的第一个环氧基碳原子，索引为 5，坐标为 (2.107, 1.200, -0.002)
        ("CE2", 5, (2.104, -1.201, -0.003)),  # 残基的芳香环的第二个环氧基碳原子，索引为 5，坐标为 (2.104, -1.201, -0.003)
        ("OH", 5, (4.168, -0.002, -0.005)),  # 残基的酚羟基氧原子，索引为 5，坐标为 (4.168, -0.002, -0.005)
        ("CZ", 5, (2.791, -0.001, -0.003)),  # 残基的芳香环的环氧基碳原子，索引为 5，坐标为 (2.791, -0.001, -0.003)
    ],  # TYR 残基的描述结束

    "VAL": [  # VAL 残基的描述开始
        ("N", 0, (-0.494, 1.373, -0.000)),  # 残基的氮原子，索引为 0，坐标为 (-0.494, 1.373, -0.000)
        ("CA", 0, (0.000, 0.000, 0.000)),  # 残基的α-碳原子，索引为 0，坐标为 (0.000, 0.000, 0.000)
        ("C", 0, (1.527, -0.000, -0.000)),  # 残基的碳原子，索引为 0，坐标为 (1.527, -0.000, -0.000)
        ("CB", 0, (-0.533, -0.795, -1.213)),  # 残基的侧链碳原子，索引为 0，坐标为 (-0.533, -0.795, -1.213)
        ("O", 3, (0.627, 1.062, -0.000)),  # 残基的氧原子，索引为 3，坐标为 (0.627, 1.062, -0.000)
        ("CG1", 4, (0.540, 1.429, -0.000)),  # 残基的第一个侧链碳原子，索引为 4，坐标为 (0.540, 1.429, -0.000)
        ("CG2", 4, (0.533, -0.776, 1.203)),  # 残基的第二个侧链碳原子，索引为 4，坐标为 (0.533, -0.776, 1.203)
    ],  # VAL 残基的描述结束
# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
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

# Naming swaps for ambiguous atom names.
# Due to symmetries in the amino acids the naming of atoms is ambiguous in
# 4 of the 20 amino acids.
# (The LDDT paper lists 7 amino acids as ambiguous, but the naming ambiguities
# in LEU, VAL and ARG can be resolved by using the 3d constellations of
# the 'ambiguous' atoms and their neighbours)
# TODO: ^ interpret this
residue_atom_renaming_swaps: Dict[str, Dict[str, str]] = {
    "ASP": {"OD1": "OD2"},
    "GLU": {"OE1": "OE2"},
    "PHE": {"CD1": "CD2", "CE1": "CE2"},
    "TYR": {"CD1": "CD2", "CE1": "CE2"},
}

# Van der Waals radii [Angstroem] of the atoms (from Wikipedia)
van_der_waals_radius: Dict[str, float] = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
}

Bond = collections.namedtuple("Bond", ["atom1_name", "atom2_name", "length", "stddev"])
BondAngle = collections.namedtuple(
    "BondAngle",
    ["atom1_name", "atom2_name", "atom3name", "angle_rad", "stddev"],
)


def map_structure_with_atom_order(in_list: list, first_call: bool = True) -> list:
    # Maps strings in a nested list structure to their corresponding index in atom_order
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


@functools.lru_cache(maxsize=None)
def load_stereo_chemical_props() -> None:
    # Placeholder function, presumably to load stereo chemical properties.
    # No implementation details provided.
    pass
    # 定义一个类型注解，表示函数返回一个元组，包含三个映射结构：
    # 第一个映射结构的键是字符串，值是 Bond 对象的列表
    # 第二个映射结构的键是字符串，值是 Bond 对象的列表
    # 第三个映射结构的键是字符串，值是 BondAngle 对象的列表
    Tuple[
        Mapping[str, List[Bond]],
        Mapping[str, List[Bond]],
        Mapping[str, List[BondAngle]],
    ]
# 将 stereo_chemical_props.txt 文件加载到一个结构化的数据中。

# 从资源管理器中读取 stereo_chemical_props.txt 文件内容
stereo_chemical_props = resources.read_text("openfold.resources", "stereo_chemical_props.txt")

# 创建行迭代器，用于逐行处理文件内容
lines_iter = iter(stereo_chemical_props.splitlines())

# 初始化字典，用于存储残基键值对应的键长信息列表
residue_bonds: Dict[str, List[Bond]] = {}
next(lines_iter)  # 跳过头部信息行

# 遍历文件内容的每一行，处理键长信息
for line in lines_iter:
    if line.strip() == "-":
        break
    bond, resname, bond_length, stddev = line.split()
    atom1, atom2 = bond.split("-")
    # 如果 resname 不在 residue_bonds 字典中，则创建空列表
    if resname not in residue_bonds:
        residue_bonds[resname] = []
    # 向 residue_bonds[resname] 列表中添加 Bond 对象
    residue_bonds[resname].append(Bond(atom1, atom2, float(bond_length), float(stddev)))
residue_bonds["UNK"] = []  # 添加一个默认值

# 初始化字典，用于存储残基键值对应的键角信息列表
residue_bond_angles: Dict[str, List[BondAngle]] = {}
next(lines_iter)  # 跳过空行
next(lines_iter)  # 跳过头部信息行

# 遍历文件内容的每一行，处理键角信息
for line in lines_iter:
    if line.strip() == "-":
        break
    bond, resname, angle_degree, stddev_degree = line.split()
    atom1, atom2, atom3 = bond.split("-")
    # 如果 resname 不在 residue_bond_angles 字典中，则创建空列表
    if resname not in residue_bond_angles:
        residue_bond_angles[resname] = []
    # 向 residue_bond_angles[resname] 列表中添加 BondAngle 对象
    residue_bond_angles[resname].append(
        BondAngle(
            atom1,
            atom2,
            atom3,
            float(angle_degree) / 180.0 * np.pi,
            float(stddev_degree) / 180.0 * np.pi,
        )
    )
residue_bond_angles["UNK"] = []  # 添加一个默认值

def make_bond_key(atom1_name: str, atom2_name: str) -> str:
    """创建用于查找键长的唯一键值。"""
    return "-".join(sorted([atom1_name, atom2_name]))

# 初始化字典，用于存储残基键值对应的虚拟键长信息列表
residue_virtual_bonds: Dict[str, List[Bond]] = {}
    for resname, bond_angles in residue_bond_angles.items():
        # 为键值对(resname, bond_angles)中的每个残基名称(resname)和键角(bond_angles)执行以下操作

        # 创建用于快速查找键长的字典。
        bond_cache: Dict[str, Bond] = {}
        # 遍历给定残基(resname)对应的键的列表，将键对(atom1_name, atom2_name)和键对象存入字典bond_cache中。
        for b in residue_bonds[resname]:
            bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b

        # 将残基的虚拟键列表初始化为空列表。
        residue_virtual_bonds[resname] = []

        # 遍历键角(bond_angles)中的每个键角(ba)。
        for ba in bond_angles:
            # 从bond_cache字典中获取键角的第一个键对应的键对象(bond1)和第二个键对应的键对象(bond2)。
            bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
            bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]

            # 使用余弦定理计算atom1和atom3之间的距离长度。
            gamma = ba.angle_rad
            length = np.sqrt(bond1.length**2 + bond2.length**2 - 2 * bond1.length * bond2.length * np.cos(gamma))

            # 根据假设未关联错误，计算不确定性的传播。
            dl_outer = 0.5 / length
            dl_dgamma = (2 * bond1.length * bond2.length * np.sin(gamma)) * dl_outer
            dl_db1 = (2 * bond1.length - 2 * bond2.length * np.cos(gamma)) * dl_outer
            dl_db2 = (2 * bond2.length - 2 * bond1.length * np.cos(gamma)) * dl_outer
            stddev = np.sqrt(
                (dl_dgamma * ba.stddev) ** 2 + (dl_db1 * bond1.stddev) ** 2 + (dl_db2 * bond2.stddev) ** 2
            )

            # 将计算得到的虚拟键信息添加到residue_virtual_bonds[resname]列表中。
            residue_virtual_bonds[resname].append(Bond(ba.atom1_name, ba.atom3name, length, stddev))

    # 返回包含三个项目的元组：原始键(bonds)、虚拟键(virtual_bonds)和键角(bond_angles)。
    return (residue_bonds, residue_virtual_bonds, residue_bond_angles)
# 一对元组，分别表示普通键合和脯氨酸键合的残基间距长度（以埃为单位）。
between_res_bond_length_c_n: Tuple[float, float] = (1.329, 1.341)
# 一对元组，分别表示普通键合和脯氨酸键合的残基间距长度的标准偏差（以埃为单位）。
between_res_bond_length_stddev_c_n: Tuple[float, float] = (0.014, 0.016)

# 一对元组，分别表示残基间的余弦角度。
between_res_cos_angles_c_n_ca: Tuple[float, float] = (-0.5203, 0.0353)  # degrees: 121.352 +- 2.315
between_res_cos_angles_ca_c_n: Tuple[float, float] = (-0.4473, 0.0311)  # degrees: 116.568 +- 1.995

# 这个列表用于存储原子数据，每个残基需要固定的原子数据大小（例如 numpy 数组）。
atom_types: List[str] = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
    "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
    "CZ3", "NZ", "OXT",
]
# 字典，将原子类型映射到它们在列表中的索引位置。
atom_order: Dict[str, int] = {atom_type: i for i, atom_type in enumerate(atom_types)}
# 原子类型的数量，这里是固定的值 37。
atom_type_num = len(atom_types)  # := 37.

# 字典，将每种氨基酸的简称映射到一个包含14个元素的原子名列表，用于紧凑的原子编码。
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
    # 定义"VAL"键对应的列表，包含了特定的原子名称
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
    # 定义"UNK"键对应的列表，包含了空字符串，用于未知类型的占位符
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
# pylint: enable=line-too-long
# pylint: enable=bad-whitespace

# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
restypes: List[str] = [
    "A",   # Alanine
    "R",   # Arginine
    "N",   # Asparagine
    "D",   # Aspartic acid
    "C",   # Cysteine
    "Q",   # Glutamine
    "E",   # Glutamic acid
    "G",   # Glycine
    "H",   # Histidine
    "I",   # Isoleucine
    "L",   # Leucine
    "K",   # Lysine
    "M",   # Methionine
    "F",   # Phenylalanine
    "P",   # Proline
    "S",   # Serine
    "T",   # Threonine
    "W",   # Tryptophan
    "Y",   # Tyrosine
    "V",   # Valine
]

restype_order: Dict[str, int] = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # Number of standard amino acid types (:= 20).
unk_restype_index = restype_num  # Catch-all index for unknown amino acid types.

restypes_with_x: List[str] = restypes + ["X"]  # Include 'X' for unknown amino acids.
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
                aa_id = mapping.get(aa_type, mapping["X"])  # Map unknown AA to 'X' if allowed.
            else:
                raise ValueError(f"Invalid character in the sequence: {aa_type}")
        else:
            aa_id = mapping[aa_type]  # Map AA based on the provided mapping.
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
# 将 restype_1to3 字典中的键值对反转，创建一个新的字典 restype_3to1，用于将三字母缩写映射回单字母缩写。
restype_3to1: Dict[str, str] = {v: k for k, v in restype_1to3.items()}

# 为所有未知的残基定义一个默认的 restype 名称。
unk_restype = "UNK"

# 根据 restypes 中的单字母缩写列表，加上 unk_restype，创建一个包含所有残基名称的列表 resnames。
resnames: List[str] = [restype_1to3[r] for r in restypes] + [unk_restype]

# 创建一个将残基名称映射到索引的字典 resname_to_idx。
resname_to_idx: Dict[str, int] = {resname: i for i, resname in enumerate(resnames)}

# HHBLITS_AA_TO_ID 和 ID_TO_HHBLITS_AA 是根据 hhblits 约定定义的字母到整数编码的映射。
# HHBLITS_AA_TO_ID 将每个氨基酸字母映射到一个整数 ID。
HHBLITS_AA_TO_ID: Dict[str, int] = {
    "A": 0,
    "B": 2,  # B 被映射到 D 的 ID
    "C": 1,  # C 和 U 共用相同的 ID
    "D": 2,  # D 和 B 共用相同的 ID
    "E": 3,  # E 和 Z 共用相同的 ID
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "J": 20,  # J 被映射到 X 的 ID
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "O": 20,  # O 被映射到 X 的 ID
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "U": 1,   # U 和 C 共用相同的 ID
    "V": 17,
    "W": 18,
    "X": 20,  # X 表示任何氨基酸
    "Y": 19,
    "Z": 3,   # Z 和 E 共用相同的 ID
    "-": 21,  # - 表示在序列比对中缺失的氨基酸
}

# ID_TO_HHBLITS_AA 是 HHBLITS_AA_TO_ID 的部分反转，将整数 ID 映射回对应的氨基酸字母。
ID_TO_HHBLITS_AA: Dict[int, str] = {
    0: "A",
    1: "C",    # 也对应 U
    2: "D",    # 也对应 B
    3: "E",    # 也对应 Z
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",   # 包括 J 和 O
    21: "-",   # 表示缺失的氨基酸
}

# 将 restypes 和 ["X", "-"] 合并，创建一个包含所有氨基酸类型的列表 restypes_with_x_and_gap。
restypes_with_x_and_gap: List[str] = restypes + ["X", "-"]

# 使用 ID_TO_HHBLITS_AA 字典映射 restypes_with_x_and_gap 中的每个氨基酸类型到其在 restypes_with_x_and_gap 中的索引，
# 创建一个元组 MAP_HHBLITS_AATYPE_TO_OUR_AATYPE。
MAP_HHBLITS_AATYPE_TO_OUR_AATYPE: Tuple[int, ...] = tuple(
    restypes_with_x_and_gap.index(ID_TO_HHBLITS_AA[i]) for i in range(len(restypes_with_x_and_gap))
)


def _make_standard_atom_mask() -> np.ndarray:
    """Returns [num_res_types, num_atom_types] mask array."""
    # 创建一个二维数组 mask，维度为 [restype_num + 1, atom_type_num]，初始值都为 0。
    # +1 是为了包括未知类型的残基 (all 0s)。
    mask = np.zeros([restype_num + 1, atom_type_num], dtype=np.int32)
    
    # 遍历 restypes 中的每个单字母缩写 restype_letter。
    for restype, restype_letter in enumerate(restypes):
        # 获取 restype_letter 对应的三字母残基名称。
        restype_name = restype_1to3[restype_letter]
        # 获取该残基的所有原子名称列表。
        atom_names = residue_atoms[restype_name]
        # 遍历该残基的每个原子名称，将对应的原子类型置为 1。
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            mask[restype, atom_type] = 1
    
    return mask


# 调用 _make_standard_atom_mask 函数，生成标准原子掩码数组，并将其赋值给 STANDARD_ATOM_MASK 变量。
STANDARD_ATOM_MASK = _make_standard_atom_mask()


# 定义一个函数 chi_angle_atom，用于生成每个残基中每个 chi 角的轴定义原子的独热表示。
def chi_angle_atom(atom_index: int) -> np.ndarray:
    """Define chi-angle rigid groups via one-hot representations."""
    # 创建一个空字典 chi_angles_index 和一个空列表 one_hots。
    chi_angles_index = {}
    one_hots = []

    # 遍历 chi_angles_atoms 中的键值对 (k, v)。
    for k, v in chi_angles_atoms.items():
        # 对于每个 v 中的序列 s，将其第 atom_index 个原子类型的索引添加到 indices 列表中。
        indices = [atom_types.index(s[atom_index]) for s in v]
        # 如果 indices 的长度不足 4，则用 -1 填充到长度为 4。
        indices.extend([-1] * (4 - len(indices)))
        # 将 indices 列表赋值给 chi_angles_index 字典的键 k。
        chi_angles_index[k] = indices

    # 遍历 restypes 中的每个残基 r。
    for r in restypes:
        # 获取 r 对应的三字母残基名称 res3。
        res3 = restype_1to3[r]
        # 根据 chi_angles_index[res3] 中的每个索引，生成对应的独热表示，并添加到 one_hots 列表中。
        one_hot = np.eye(atom_type_num)[chi_angles_index[res3]]
        one_hots.append(one_hot)
    # 将一个全零的数组添加到 `one_hots` 列表中，数组形状为 [4, atom_type_num]，用于表示残基 `X`。
    one_hots.append(np.zeros([4, atom_type_num]))
    
    # 将 `one_hots` 列表中的数组堆叠成一个新的数组 `one_hot`，沿着第一个轴堆叠。
    one_hot = np.stack(one_hots, axis=0)
    
    # 对 `one_hot` 数组进行转置操作，交换第二个和第三个维度，形状变为 [batch_size, atom_type_num, 4]。
    one_hot = np.transpose(one_hot, [0, 2, 1])
    
    # 返回经过处理的 one_hot 数组作为函数的输出结果。
    return one_hot
# 使用函数 chi_angle_atom(1) 计算第一个原子的氨基酸角度独热编码
chi_atom_1_one_hot = chi_angle_atom(1)
# 使用函数 chi_angle_atom(2) 计算第二个原子的氨基酸角度独热编码
chi_atom_2_one_hot = chi_angle_atom(2)

# 生成一个与 chi_angles_atoms 类似的数组，但使用索引而不是名称
chi_angles_atom_indices_list: List[List[List[str]]] = [chi_angles_atoms[restype_1to3[r]] for r in restypes]
# 使用函数 map_structure_with_atom_order 处理 chi_angles_atom_indices_list 中的结构，并返回结果
chi_angles_atom_indices_ours: list = map_structure_with_atom_order(chi_angles_atom_indices_list)
# 创建一个 numpy 数组，存储每个氨基酸的角度原子索引，用于计算
chi_angles_atom_indices = np.array(
    [chi_atoms + ([[0, 0, 0, 0]] * (4 - len(chi_atoms))) for chi_atoms in chi_angles_atom_indices_list]
)

# 从 (氨基酸名, 原子名) 对映射到原子的 chi 组索引及其在该组内的原子索引
chi_groups_for_atom: Dict[Tuple[str, str], List[Tuple[int, int]]] = collections.defaultdict(list)
for res_name, chi_angle_atoms_for_res in chi_angles_atoms.items():
    for chi_group_i, chi_group in enumerate(chi_angle_atoms_for_res):
        for atom_i, atom in enumerate(chi_group):
            chi_groups_for_atom[(res_name, atom)].append((chi_group_i, atom_i))
chi_groups_for_atom = dict(chi_groups_for_atom)

def _make_rigid_transformation_4x4(ex: np.ndarray, ey: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create a rigid 4x4 transformation matrix from two axes and transl."""
    # 将 ex 向量归一化
    ex_normalized = ex / np.linalg.norm(ex)

    # 使 ey 向量垂直于 ex 向量
    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)

    # 计算 ez 向量作为 ex 和 ey 向量的叉乘
    eznorm = np.cross(ex_normalized, ey_normalized)
    # 创建 4x4 的刚体变换矩阵
    m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
    m = np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    return m

# 创建数组，存储 (氨基酸类型, 原子类型) 到刚体组索引的映射
restype_atom37_to_rigid_group = np.zeros([21, 37], dtype=int)
# 创建数组，存储 (氨基酸类型, 原子类型) 到刚体组位置的掩码
restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
# 创建数组，存储 (氨基酸类型, 原子类型) 到刚体组位置的坐标
restype_atom37_rigid_group_positions = np.zeros([21, 37, 3], dtype=np.float32)
# 创建数组，存储 (氨基酸类型, 原子类型) 到刚体组索引的映射（14 个原子的版本）
restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=int)
# 创建数组，存储 (氨基酸类型, 原子类型) 到刚体组位置的掩码（14 个原子的版本）
restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)
# 创建数组，存储 (氨基酸类型, 原子类型) 到刚体组位置的坐标（14 个原子的版本）
restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)
# 创建数组，存储 (氨基酸类型) 到默认刚体组坐标系的转换矩阵
restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)

def _make_rigid_group_constants() -> None:
    """Fill the arrays above."""
    # 遍历每个残基类型及其对应的字母表示
    for restype, restype_letter in enumerate(restypes):
        # 根据字母找到对应的残基名
        resname = restype_1to3[restype_letter]
        # 遍历该残基名对应的所有原子名称、原子组索引和原子位置信息
        for atomname, group_idx, atom_position in rigid_group_atom_positions[resname]:
            # 根据原子名找到对应的原子类型
            atomtype = atom_order[atomname]

            # 将残基类型和原子类型映射到刚性组索引
            restype_atom37_to_rigid_group[restype, atomtype] = group_idx
            # 设置残基类型和原子类型的掩码为1，表示存在关联
            restype_atom37_mask[restype, atomtype] = 1
            # 设置残基类型和原子类型的刚性组位置信息
            restype_atom37_rigid_group_positions[restype, atomtype, :] = atom_position

            # 在残基名到14原子名列表中找到当前原子名的索引
            atom14idx = restype_name_to_atom14_names[resname].index(atomname)
            # 将残基类型和14原子名索引映射到刚性组索引
            restype_atom14_to_rigid_group[restype, atom14idx] = group_idx
            # 设置残基类型和14原子名索引的掩码为1
            restype_atom14_mask[restype, atom14idx] = 1
            # 设置残基类型和14原子名索引的刚性组位置信息
            restype_atom14_rigid_group_positions[restype, atom14idx, :] = atom_position
    # 遍历氨基酸类型及其对应的缩写
    for restype, restype_letter in enumerate(restypes):
        # 根据缩写获取氨基酸的全名
        resname = restype_1to3[restype_letter]
        # 从预定义的刚性群体原子位置中创建字典，键为原子名，值为位置数组
        atom_positions: Dict[str, np.ndarray] = {
            name: np.array(pos) for name, _, pos in rigid_group_atom_positions[resname]
        }

        # 将刚性群体的默认骨架到骨架的变换矩阵设为单位矩阵（身份变换）
        restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)

        # 将预Ω框架到骨架的变换矩阵设为单位矩阵（虚拟的身份变换）
        restype_rigid_group_default_frame[restype, 1, :, :] = np.eye(4)

        # 将φ框架到骨架的变换矩阵计算为刚性变换矩阵
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["N"] - atom_positions["CA"],  # X轴方向的向量
            ey=np.array([1.0, 0.0, 0.0]),  # Y轴方向的向量
            translation=atom_positions["N"],  # 平移向量
        )
        restype_rigid_group_default_frame[restype, 2, :, :] = mat

        # 将ψ框架到骨架的变换矩阵计算为刚性变换矩阵
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C"] - atom_positions["CA"],  # X轴方向的向量
            ey=atom_positions["CA"] - atom_positions["N"],  # Y轴方向的向量
            translation=atom_positions["C"],  # 平移向量
        )
        restype_rigid_group_default_frame[restype, 3, :, :] = mat

        # 如果存在χ1角度，则计算χ1框架到骨架的变换矩阵
        if chi_angles_mask[restype][0]:
            base_atom_names = chi_angles_atoms[resname][0]  # χ1角度的基础原子名列表
            base_atom_positions = [atom_positions[name] for name in base_atom_names]  # 基础原子的位置列表
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],  # X轴方向的向量
                ey=base_atom_positions[0] - base_atom_positions[1],  # Y轴方向的向量
                translation=base_atom_positions[2],  # 平移向量
            )
            restype_rigid_group_default_frame[restype, 4, :, :] = mat

        # 依次计算χ2到χ4框架到前一框架的刚性变换矩阵
        # 由于所有下一个框架的旋转轴都从前一个框架的(0,0,0)开始，因此这里使用了固定的旋转轴
        for chi_idx in range(1, 4):
            if chi_angles_mask[restype][chi_idx]:
                axis_end_atom_name = chi_angles_atoms[resname][chi_idx][2]  # 当前角度的轴端原子名
                axis_end_atom_position = atom_positions[axis_end_atom_name]  # 轴端原子的位置
                mat = _make_rigid_transformation_4x4(
                    ex=axis_end_atom_position,  # X轴方向的向量
                    ey=np.array([-1.0, 0.0, 0.0]),  # Y轴方向的向量
                    translation=axis_end_atom_position,  # 平移向量
                )
                restype_rigid_group_default_frame[restype, 4 + chi_idx, :, :] = mat
# 调用函数以初始化刚性群组的常量
_make_rigid_group_constants()

# 定义函数，计算原子间的上下界，以评估违规情况
def make_atom14_dists_bounds(
    overlap_tolerance: float = 1.5,  # 碰撞容忍度
    bond_length_tolerance_factor: int = 15,  # 键长容忍因子
) -> Dict[str, np.ndarray]:
    """compute upper and lower bounds for bonds to assess violations."""
    # 初始化数组以存储不同残基类型、原子间距离的下界、上界和标准差
    restype_atom14_bond_lower_bound = np.zeros([21, 14, 14], np.float32)
    restype_atom14_bond_upper_bound = np.zeros([21, 14, 14], np.float32)
    restype_atom14_bond_stddev = np.zeros([21, 14, 14], np.float32)
    
    # 载入化学属性，包括原子间的键和虚拟键
    residue_bonds, residue_virtual_bonds, _ = load_stereo_chemical_props()
    
    # 遍历每种残基类型
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
                upper = 1e10
                # 设置原子间距离的下界和上界
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
            # 设置原子间距离的下界和上界，以及标准差
            restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
            restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
            restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
            restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
            restype_atom14_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
            restype_atom14_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
    
    # 返回包含键的下界、上界和标准差的字典
    return {
        "lower_bound": restype_atom14_bond_lower_bound,  # 形状为 (21,14,14)
        "upper_bound": restype_atom14_bond_upper_bound,  # 形状为 (21,14,14)
        "stddev": restype_atom14_bond_stddev,  # 形状为 (21,14,14)
    }

# 初始化数组以存储不同残基类型的模糊原子信息
restype_atom14_ambiguous_atoms = np.zeros((21, 14), dtype=np.float32)
# 创建索引数组，用于记录不同残基类型的原子序号
restype_atom14_ambiguous_atoms_swap_idx: np.ndarray = np.tile(np.arange(14, dtype=int), (21, 1))

# 定义函数，生成原子的模糊特征
def _make_atom14_ambiguity_feats() -> None:
    # 遍历 residue_atom_renaming_swaps 字典的每一项，其中 res 是键，pairs 是对应的值（另一个字典）
    for res, pairs in residue_atom_renaming_swaps.items():
        # 使用 restype_3to1 字典将三字母氨基酸代码 res 转换为索引 res_idx
        res_idx = restype_order[restype_3to1[res]]
        # 遍历 pairs 字典中的每一对 atom1 和 atom2
        for atom1, atom2 in pairs.items():
            # 在 restype_name_to_atom14_names[res] 列表中找到 atom1 的索引 atom1_idx
            atom1_idx = restype_name_to_atom14_names[res].index(atom1)
            # 在 restype_name_to_atom14_names[res] 列表中找到 atom2 的索引 atom2_idx
            atom2_idx = restype_name_to_atom14_names[res].index(atom2)
            # 将 restype_atom14_ambiguous_atoms 中 (res_idx, atom1_idx) 处置为 1，表示 atom1 是模糊的
            restype_atom14_ambiguous_atoms[res_idx, atom1_idx] = 1
            # 将 restype_atom14_ambiguous_atoms 中 (res_idx, atom2_idx) 处置为 1，表示 atom2 是模糊的
            restype_atom14_ambiguous_atoms[res_idx, atom2_idx] = 1
            # 记录 atom1_idx 处应该交换的索引是 atom2_idx
            restype_atom14_ambiguous_atoms_swap_idx[res_idx, atom1_idx] = atom2_idx
            # 记录 atom2_idx 处应该交换的索引是 atom1_idx
            restype_atom14_ambiguous_atoms_swap_idx[res_idx, atom2_idx] = atom1_idx
# 调用名为 `_make_atom14_ambiguity_feats` 的函数，执行其内部逻辑
_make_atom14_ambiguity_feats()

# 将整数序列 `aatype` 转换为对应的字符串序列，并返回结果
def aatype_to_str_sequence(aatype: Sequence[int]) -> str:
    # 使用列表推导式遍历 `aatype` 序列的每个元素，根据其值从 `restypes_with_x` 字典中获取对应的字符串，并连接成一个字符串
    return "".join([restypes_with_x[aatype[i]] for i in range(len(aatype))])
```