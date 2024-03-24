# `.\lucidrains\geometric-vector-perceptron\examples\data_handler.py`

```py
# 作者：Eric Alcaide

# 从 https://github.com/jonathanking/sidechainnet 借用了大部分代码
# 下面是其许可证：

# 版权所有 2020 Jonathan King
# 允许以源代码和二进制形式重新分发和使用，无论是否进行修改，只要满足以下条件：
#
# 1. 源代码的再分发必须保留上述版权声明、此条件列表和以下免责声明。
#
# 2. 以二进制形式再分发时，必须在提供的文档和/或其他材料中复制上述版权声明、此条件列表和以下免责声明。
#
# 3. 未经特定事先书面许可，不得使用版权持有人或其贡献者的名称来认可或推广从本软件派生的产品。
#
# 版权持有人和贡献者提供的本软件是按原样提供的，不提供任何明示或暗示的担保，包括但不限于对适销性和特定用途的暗示担保。
# 在任何情况下，无论是在合同、严格责任还是侵权（包括疏忽或其他方式）的情况下，版权持有人或贡献者均不对任何直接、间接、附带、特殊、惩罚性或后果性损害（包括但不限于替代商品或服务的采购、使用、数据或利润损失或业务中断）负责，即使已被告知可能发生此类损害。

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np 
from einops import repeat, rearrange

######################
## structural utils ##
######################

def get_dihedral(c1, c2, c3, c4):
    """ 返回弯曲角度（弧度）。
        将使用来自以下链接的 atan2 公式：
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        输入：
        * c1: (batch, 3) 或 (3,)
        * c2: (batch, 3) 或 (3,)
        * c3: (batch, 3) 或 (3,)
        * c4: (batch, 3) 或 (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,  
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) )


def get_angle(c1, c2, c3):
    """ 返回角度（弧度）。
        输入：
        * c1: (batch, 3) 或 (3,)
        * c2: (batch, 3) 或 (3,)
        * c3: (batch, 3) 或 (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2

    # 不使用传统的 arccos，因为它得到的是“最小角度”，不一定是我们想要的
    # return torch.acos( (u1*u2).sum(dim=-1) / (torch.norm(u1, dim=-1)*torch.norm(u2, dim=-1) )+

    # 更好地使用 atan2 公式：atan2(cross, dot) 来自这里：
    # https://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html

    # 添加负号，因为我们希望角度是反向的 - sidechainnet 问题
    return torch.atan2( torch.norm(torch.cross(u1,u2, dim=-1), dim=-1), 
                        -(u1*u2).sum(dim=-1) ) 


def kabsch_torch(X, Y):
    """ 将 X 对齐到 Y 的 Kabsch 对齐。
        假设 X、Y 都是 (D, N) 的形式 - 通常是 (3, N)
    """
    # 将 X 和 Y 居中到原点
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # 计算协方差矩阵（对于每个批次中的蛋白质）
    C = torch.matmul(X_, Y_.t())
    # 通过 SVD 计算最佳旋转矩阵 - 警告！W 必须被转置
    V, S, W = torch.svd(C.detach())
    # 方向校正的行列式符号
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # 创建旋转矩阵 U
    U = torch.matmul(V, W.t())
    # 计算旋转
    X_ = torch.matmul(X_.t(), U).t()
    # 返回居中和对齐后的 X_ 和 Y_
    return X_, Y_
# 计算两个张量之间的均方根偏差，假设 X 和 Y 的形状都是 (batch, d, n)，通常是 (batch, 3, N)
def rmsd_torch(X, Y):
    """ Assumes x,y are both (batch, d, n) - usually (batch, 3, N). """
    return torch.sqrt( torch.mean((X - Y)**2, axis=(-1, -2)) )


############
### INFO ###
############

# 包含了不同氨基酸的构建信息的字典
SC_BUILD_INFO = {
    'A': {
        'angles-names': ['N-CA-CB'],
        'angles-types': ['N -CX-CT'],
        'angles-vals': [1.9146261894377796],
        'atom-names': ['CB'],
        'bonds-names': ['CA-CB'],
        'bonds-types': ['CX-CT'],
        'bonds-vals': [1.526],
        'torsion-names': ['C-N-CA-CB'],
        'torsion-types': ['C -N -CX-CT'],
        'torsion-vals': ['p']
    },
    'R': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-NE', 'CD-NE-CZ', 'NE-CZ-NH1',
            'NE-CZ-NH2'
        ],
        'angles-types': [
            'N -CX-C8', 'CX-C8-C8', 'C8-C8-C8', 'C8-C8-N2', 'C8-N2-CA', 'N2-CA-N2',
            'N2-CA-N2'
        ],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.9408061282176945,
            2.150245638457014, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-NE', 'NE-CZ', 'CZ-NH1', 'CZ-NH2'],
        'bonds-types': ['CX-C8', 'C8-C8', 'C8-C8', 'C8-N2', 'N2-CA', 'CA-N2', 'CA-N2'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.463, 1.34, 1.34, 1.34],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-NE', 'CG-CD-NE-CZ',
            'CD-NE-CZ-NH1', 'CD-NE-CZ-NH2'
        ],
        'torsion-types': [
            'C -N -CX-C8', 'N -CX-C8-C8', 'CX-C8-C8-C8', 'C8-C8-C8-N2', 'C8-C8-N2-CA',
            'C8-N2-CA-N2', 'C8-N2-CA-N2'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'p', 'p', 'i']
    },
    'N': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-OD1', 'CB-CG-ND2'],
        'angles-types': ['N -CX-2C', 'CX-2C-C ', '2C-C -O ', '2C-C -N '],
        'angles-vals': [
            1.9146261894377796, 1.9390607989657, 2.101376419401173, 2.035053907825388
        ],
        'atom-names': ['CB', 'CG', 'OD1', 'ND2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-OD1', 'CG-ND2'],
        'bonds-types': ['CX-2C', '2C-C ', 'C -O ', 'C -N '],
        'bonds-vals': [1.526, 1.522, 1.229, 1.335],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-OD1', 'CA-CB-CG-ND2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-C ', 'CX-2C-C -O ', 'CX-2C-C -N '],
        'torsion-vals': ['p', 'p', 'p', 'i']
    },
    'D': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-OD1', 'CB-CG-OD2'],
        'angles-types': ['N -CX-2C', 'CX-2C-CO', '2C-CO-O2', '2C-CO-O2'],
        'angles-vals': [
            1.9146261894377796, 1.9390607989657, 2.0420352248333655, 2.0420352248333655
        ],
        'atom-names': ['CB', 'CG', 'OD1', 'OD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-OD1', 'CG-OD2'],
        'bonds-types': ['CX-2C', '2C-CO', 'CO-O2', 'CO-O2'],
        'bonds-vals': [1.526, 1.522, 1.25, 1.25],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-OD1', 'CA-CB-CG-OD2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-CO', 'CX-2C-CO-O2', 'CX-2C-CO-O2'],
        'torsion-vals': ['p', 'p', 'p', 'i']
    },
    'C': {
        'angles-names': ['N-CA-CB', 'CA-CB-SG'],
        'angles-types': ['N -CX-2C', 'CX-2C-SH'],
        'angles-vals': [1.9146261894377796, 1.8954275676658419],
        'atom-names': ['CB', 'SG'],
        'bonds-names': ['CA-CB', 'CB-SG'],
        'bonds-types': ['CX-2C', '2C-SH'],
        'bonds-vals': [1.526, 1.81],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-SG'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-SH'],
        'torsion-vals': ['p', 'p']
    },
    'Q': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-OE1', 'CG-CD-NE2'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-C ', '2C-C -O ', '2C-C -N '],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.9390607989657, 2.101376419401173,
            2.035053907825388
        ],
        'atom-names': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-OE1', 'CD-NE2'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-C ', 'C -O ', 'C -N '],
        'bonds-vals': [1.526, 1.526, 1.522, 1.229, 1.335],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-OE1', 'CB-CG-CD-NE2'
        ],
        'torsion-types': [
            'C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-C ', '2C-2C-C -O ', '2C-2C-C -N '
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'i']
    },
    'E': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-OE1', 'CG-CD-OE2'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-CO', '2C-CO-O2', '2C-CO-O2'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.9390607989657, 2.0420352248333655,
            2.0420352248333655
        ],
        'atom-names': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-OE1', 'CD-OE2'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-CO', 'CO-O2', 'CO-O2'],
        'bonds-vals': [1.526, 1.526, 1.522, 1.25, 1.25],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-OE1', 'CB-CG-CD-OE2'
        ],
        'torsion-types': [
            'C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-CO', '2C-2C-CO-O2', '2C-2C-CO-O2'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'i']
    },
    'G': {
        'angles-names': [],
        'angles-types': [],
        'angles-vals': [],
        'atom-names': [],
        'bonds-names': [],
        'bonds-types': [],
        'bonds-vals': [],
        'torsion-names': [],
        'torsion-types': [],
        'torsion-vals': []
    },
    'H': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-ND1', 'CG-ND1-CE1', 'ND1-CE1-NE2', 'CE1-NE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CC', 'CT-CC-NA', 'CC-NA-CR', 'NA-CR-NB', 'CR-NB-CV'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9739673840055867, 2.0943951023931953,
            1.8849555921538759, 1.8849555921538759, 1.8849555921538759
        ],
        'atom-names': ['CB', 'CG', 'ND1', 'CE1', 'NE2', 'CD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-ND1', 'ND1-CE1', 'CE1-NE2', 'NE2-CD2'],
        'bonds-types': ['CX-CT', 'CT-CC', 'CC-NA', 'NA-CR', 'CR-NB', 'NB-CV'],
        'bonds-vals': [1.526, 1.504, 1.385, 1.343, 1.335, 1.394],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-ND1', 'CB-CG-ND1-CE1', 'CG-ND1-CE1-NE2',
            'ND1-CE1-NE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CC', 'CX-CT-CC-NA', 'CT-CC-NA-CR', 'CC-NA-CR-NB',
            'NA-CR-NB-CV'
        ],
        'torsion-vals': ['p', 'p', 'p', 3.141592653589793, 0.0, 0.0]
    },
    'I': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG1', 'CB-CG1-CD1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-2C', '3C-2C-CT', 'CX-3C-CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791
        ],
        'atom-names': ['CB', 'CG1', 'CD1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-CG1', 'CG1-CD1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-2C', '2C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG1', 'CA-CB-CG1-CD1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-2C', 'CX-3C-2C-CT', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p', 'p']
    },
    'L': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CB-CG-CD2'],
        'angles-types': ['N -CX-2C', 'CX-2C-3C', '2C-3C-CT', '2C-3C-CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD1', 'CG-CD2'],
        'bonds-types': ['CX-2C', '2C-3C', '3C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CA-CB-CG-CD2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-3C', 'CX-2C-3C-CT', 'CX-2C-3C-CT'],
        'torsion-vals': ['p', 'p', 'p', 'p']
    },
    'K': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-CE', 'CD-CE-NZ'],
        'angles-types': ['N -CX-C8', 'CX-C8-C8', 'C8-C8-C8', 'C8-C8-C8', 'C8-C8-N3'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791,
            1.9408061282176945
        ],
        'atom-names': ['CB', 'CG', 'CD', 'CE', 'NZ'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-CE', 'CE-NZ'],
        'bonds-types': ['CX-C8', 'C8-C8', 'C8-C8', 'C8-C8', 'C8-N3'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526, 1.471],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-CE', 'CG-CD-CE-NZ'
        ],
        'torsion-types': [
            'C -N -CX-C8', 'N -CX-C8-C8', 'CX-C8-C8-C8', 'C8-C8-C8-C8', 'C8-C8-C8-N3'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'p']
    },
    'M': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-SD', 'CG-SD-CE'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-S ', '2C-S -CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 2.0018926520374962, 1.726130630222392
        ],
        'atom-names': ['CB', 'CG', 'SD', 'CE'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-SD', 'SD-CE'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-S ', 'S -CT'],
        'bonds-vals': [1.526, 1.526, 1.81, 1.81],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-SD', 'CB-CG-SD-CE'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-S ', '2C-2C-S -CT'],
        'torsion-vals': ['p', 'p', 'p', 'p']
    },
    'F': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-CE1', 'CD1-CE1-CZ', 'CE1-CZ-CE2',
            'CZ-CE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CA', 'CT-CA-CA', 'CA-CA-CA', 'CA-CA-CA', 'CA-CA-CA',
            'CA-CA-CA'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9896753472735358, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-CE1', 'CE1-CZ', 'CZ-CE2', 'CE2-CD2'
        ],
        'bonds-types': ['CX-CT', 'CT-CA', 'CA-CA', 'CA-CA', 'CA-CA', 'CA-CA', 'CA-CA'],
        'bonds-vals': [1.526, 1.51, 1.4, 1.4, 1.4, 1.4, 1.4],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-CE1', 'CG-CD1-CE1-CZ',
            'CD1-CE1-CZ-CE2', 'CE1-CZ-CE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CA', 'CX-CT-CA-CA', 'CT-CA-CA-CA', 'CA-CA-CA-CA',
            'CA-CA-CA-CA', 'CA-CA-CA-CA'
        ],
        'torsion-vals': ['p', 'p', 'p', 3.141592653589793, 0.0, 0.0, 0.0]
    },
    'P': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD'],
        'angles-types': ['N -CX-CT', 'CX-CT-CT', 'CT-CT-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'CG', 'CD'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD'],
        'bonds-types': ['CX-CT', 'CT-CT', 'CT-CT'],
        'bonds-vals': [1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD'],
        'torsion-types': ['C -N -CX-CT', 'N -CX-CT-CT', 'CX-CT-CT-CT'],
        'torsion-vals': ['p', 'p', 'p']
    },
    # 定义了氨基酸"S"的键值对，包含了该氨基酸的角度、键和扭转信息
    'S': {
        'angles-names': ['N-CA-CB', 'CA-CB-OG'],
        'angles-types': ['N -CX-2C', 'CX-2C-OH'],
        'angles-vals': [1.9146261894377796, 1.911135530933791],
        'atom-names': ['CB', 'OG'],
        'bonds-names': ['CA-CB', 'CB-OG'],
        'bonds-types': ['CX-2C', '2C-OH'],
        'bonds-vals': [1.526, 1.41],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-OG'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-OH'],
        'torsion-vals': ['p', 'p']
    },
    # 定义了氨基酸"T"的键值对，包含了该氨基酸的角度、键和扭转信息
    'T': {
        'angles-names': ['N-CA-CB', 'CA-CB-OG1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-OH', 'CX-3C-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'OG1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-OG1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-OH', '3C-CT'],
        'bonds-vals': [1.526, 1.41, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-OG1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-OH', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p']
    },
    # 定义了氨基酸"W"的键值对，包含了该氨基酸的角度、键和扭转信息
    'W': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-NE1', 'CD1-NE1-CE2',
            'NE1-CE2-CZ2', 'CE2-CZ2-CH2', 'CZ2-CH2-CZ3', 'CH2-CZ3-CE3', 'CZ3-CE3-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-C*', 'CT-C*-CW', 'C*-CW-NA', 'CW-NA-CN', 'NA-CN-CA',
            'CN-CA-CA', 'CA-CA-CA', 'CA-CA-CA', 'CA-CA-CB'
        ],
        'angles-vals': [
            1.9146261894377796, 2.0176006153054447, 2.181661564992912, 1.8971728969178363,
            1.9477874452256716, 2.3177972466484698, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': [
            'CB', 'CG', 'CD1', 'NE1', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3', 'CD2'
        ],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-NE1', 'NE1-CE2', 'CE2-CZ2', 'CZ2-CH2',
            'CH2-CZ3', 'CZ3-CE3', 'CE3-CD2'
        ],
        'bonds-types': [
            'CX-CT', 'CT-C*', 'C*-CW', 'CW-NA', 'NA-CN', 'CN-CA', 'CA-CA', 'CA-CA',
            'CA-CA', 'CA-CB'
        ],
        'bonds-vals': [1.526, 1.495, 1.352, 1.381, 1.38, 1.4, 1.4, 1.4, 1.4, 1.404],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-NE1', 'CG-CD1-NE1-CE2',
            'CD1-NE1-CE2-CZ2', 'NE1-CE2-CZ2-CH2', 'CE2-CZ2-CH2-CZ3', 'CZ2-CH2-CZ3-CE3',
            'CH2-CZ3-CE3-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-C*', 'CX-CT-C*-CW', 'CT-C*-CW-NA', 'C*-CW-NA-CN',
            'CW-NA-CN-CA', 'NA-CN-CA-CA', 'CN-CA-CA-CA', 'CA-CA-CA-CA', 'CA-CA-CA-CB'
        ],
        'torsion-vals': [
            'p', 'p', 'p', 3.141592653589793, 0.0, 3.141592653589793, 3.141592653589793,
            0.0, 0.0, 0.0
        ]
    },
    'Y': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-CE1', 'CD1-CE1-CZ', 'CE1-CZ-OH',
            'CE1-CZ-CE2', 'CZ-CE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CA', 'CT-CA-CA', 'CA-CA-CA', 'CA-CA-C ', 'CA-C -OH',
            'CA-C -CA', 'C -CA-CA'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9896753472735358, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'OH', 'CE2', 'CD2'],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-CE1', 'CE1-CZ', 'CZ-OH', 'CZ-CE2', 'CE2-CD2'
        ],
        'bonds-types': [
            'CX-CT', 'CT-CA', 'CA-CA', 'CA-CA', 'CA-C ', 'C -OH', 'C -CA', 'CA-CA'
        ],
        'bonds-vals': [1.526, 1.51, 1.4, 1.4, 1.409, 1.364, 1.409, 1.4],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-CE1', 'CG-CD1-CE1-CZ',
            'CD1-CE1-CZ-OH', 'CD1-CE1-CZ-CE2', 'CE1-CZ-CE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CA', 'CX-CT-CA-CA', 'CT-CA-CA-CA', 'CA-CA-CA-C ',
            'CA-CA-C -OH', 'CA-CA-C -CA', 'CA-C -CA-CA'
        ],
        'torsion-vals': [
            'p', 'p', 'p', 3.141592653589793, 0.0, 3.141592653589793, 0.0, 0.0
        ]
    },
    'V': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-CT', 'CX-3C-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'CG1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-CG1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-CT', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p']
    },
    '_': {
        'angles-names': [],
        'angles-types': [],
        'angles-vals': [],
        'atom-names': [],
        'bonds-names': [],
        'bonds-types': [],
        'bonds-vals': [],
        'torsion-names': [],
        'torsion-types': [],
        'torsion-vals': []
    }
# 闭合 BB_BUILD_INFO 字典
}

# 定义 BB_BUILD_INFO 字典，包含键值对表示不同键的键值
BB_BUILD_INFO = {
    "BONDLENS": {
        # 更新的值是根据来自 1DPE_1_A 的晶体数据进行的验证
        # 注释的值是 sidechainnet 的值
        'n-ca': 1.4664931, # 1.442, 
        'ca-c': 1.524119,  # 1.498,
        'c-n': 1.3289373,  # 1.379,
        'c-o': 1.229,  # 来自 parm10.dat || 根据结构有很大的变化
        # 我们从 1DPE_1_A 得到 1.3389416，但也从 2F2H_d2f2hf1 得到 1.2289
        'c-oh': 1.364
    },  # 对于 OXT 来自 parm10.dat
    # 用于放置氧原子
    "BONDANGS": {
        'ca-c-o': 2.0944,  # 近似为 2pi / 3; parm10.dat 表示为 2.0350539
        'ca-c-oh': 2.0944
    },  # 等同于 'ca-c-o'，对于 OXT
    "BONDTORSIONS": {
        'n-ca-c-n': -0.785398163
    }  # 一个简单的近似，不打算精确
}

# 定义函数 make_cloud_mask，返回一个包含相关点为 1，填充点为 0 的数组
def make_cloud_mask(aa):
    """ relevent points will be 1. paddings will be 0. """
    mask = np.zeros(14)
    if aa != "_":
        n_atoms = 4+len( SC_BUILD_INFO[aa]["atom-names"] )
        mask[:n_atoms] = 1
    return mask

# 定义函数 make_bond_mask，返回一个包含每个原子起始键长的数组
def make_bond_mask(aa):
    """ Gives the length of the bond originating each atom. """
    mask = np.zeros(14)
    # backbone
    mask[0] = BB_BUILD_INFO["BONDLENS"]['c-n']
    mask[1] = BB_BUILD_INFO["BONDLENS"]['n-ca']
    mask[2] = BB_BUILD_INFO["BONDLENS"]['ca-c']
    mask[3] = BB_BUILD_INFO["BONDLENS"]['c-o']
    # sidechain - except padding token 
    if aa in SC_BUILD_INFO.keys():
        for i,bond in enumerate(SC_BUILD_INFO[aa]['bonds-vals']):
            mask[4+i] = bond
    return mask

# 定义函数 make_theta_mask，返回一个包含每个原子起始键角度的数组
def make_theta_mask(aa):
    """ Gives the theta of the bond originating each atom. """
    mask = np.zeros(14)
    # backbone
    #
    # sidechain
    for i,theta in enumerate(SC_BUILD_INFO[aa]['angles-vals']):
        mask[4+i] = theta
    return mask

# 定义函数 make_torsion_mask，返回一个包含每个原子起始键二面角的数组
def make_torsion_mask(aa):
    """ Gives the dihedral of the bond originating each atom. """
    mask = np.zeros(14)
    # backbone
    #
    # sidechain
    for i, torsion in enumerate(SC_BUILD_INFO[aa]['torsion-vals']):
        if torsion == 'p':
            mask[4+i] = np.nan 
        elif torsion == "i":
            # https://github.com/jonathanking/sidechainnet/blob/master/sidechainnet/structure/StructureBuilder.py#L372
            mask[4+i] =  999  # anotate to change later # mask[4+i-1] - np.pi
        else:
            mask[4+i] = torsion
    return mask

# 定义函数 make_idx_mask，返回一个包含前三个点的索引的数组
def make_idx_mask(aa):
    """ Gives the idxs of the 3 previous points. """
    mask = np.zeros((11, 3))
    # backbone
    mask[0, :] = np.arange(3) 
    # sidechain
    mapper = {"N": 0, "CA": 1, "C":2,  "CB": 4}
    for i, torsion in enumerate(SC_BUILD_INFO[aa]['torsion-names']):
        # 获取形成二面角的所有原子
        torsions = [x.rstrip(" ") for x in torsion.split("-")]
        # 对于每个原子
        for n, torsion in enumerate(torsions[:-1]):
            # 获取坐标数组中原子的索引
            loc = mapper[torsion] if torsion in mapper.keys() else 4 + SC_BUILD_INFO[aa]['atom-names'].index(torsion)
            # 设置位置为索引
            mask[i+1][n] = loc
    return mask

# 定义 SUPREME_INFO 字典，包含各种信息的字典
SUPREME_INFO = {k: {"cloud_mask": make_cloud_mask(k),
                    "bond_mask": make_bond_mask(k),
                    "theta_mask": make_theta_mask(k),
                    "torsion_mask": make_torsion_mask(k),
                    "idx_mask": make_idx_mask(k),
                    } 
                for k in "ARNDCQEGHILKMFPSTWYV_"}

# 定义函数 scn_cloud_mask，获取原子位置的布尔掩码
def scn_cloud_mask(seq, coords=None):
    """ Gets the boolean mask atom positions (not all aas have same atoms). 
        Inputs: 
        * seqs: (length) iterable of 1-letter aa codes of a protein
        * coords: optional .(batch, lc, 3). sidechainnet coords.
                  returns the true mask (solves potential atoms that might not be provided)
        Outputs: (length, 14) boolean mask 
    """ 
    # 如果坐标不为空
    if coords is not None:
        # 重新排列坐标张量的维度，将最后一维拆分为两个维度
        # 检查是否等于0，然后按最后一个维度求和
        # 检查是否小于坐标张量的最后一个维度的长度
        # 转换为浮点数并移动到 CPU 上
        return ((rearrange(coords, '... (l c) d -> ... l c d', c=14) == 0).sum(dim=-1) < coords.shape[-1]).float().cpu()
    # 如果坐标为空
    # 返回一个张量，其中包含序列中每个氨基酸的云掩码信息
    return torch.tensor([SUPREME_INFO[aa]['cloud_mask'] for aa in seq])
# 定义函数，根据氨基酸序列生成键长掩码
def scn_bond_mask(seq):
    """ Inputs: 
        * seqs: (length). iterable of 1-letter aa codes of a protein
        Outputs: (L, 14) maps point to bond length
    """ 
    # 返回键长掩码的张量
    return torch.tensor([SUPREME_INFO[aa]['bond_mask'] for aa in seq])


# 定义函数，根据氨基酸序列和角度生成角度掩码
def scn_angle_mask(seq, angles):
    """ Inputs: 
        * seq: (length). iterable of 1-letter aa codes of a protein
        * angles: (length, 12). [phi, psi, omega, b_angle(n_ca_c), b_angle(ca_c_n), b_angle(c_n_ca), 6_scn_torsions]
        Outputs: (L, 14) maps point to theta and dihedral.
                 first angle is theta, second is dihedral
    """ 
    # 获取设备和精度
    device, precise = angles.device, angles.type()
    angles = angles
    # 获取角度掩码
    theta_mask   = torch.tensor([SUPREME_INFO[aa]['theta_mask'] for aa in seq]).type(precise)
    torsion_mask = torch.tensor([SUPREME_INFO[aa]['torsion_mask'] for aa in seq]).type(precise)
    # 填充掩码与角度值
    theta_mask[:, 0] = angles[:, 4] # ca_c_n
    theta_mask[1:, 1] = angles[:-1, 5] # c_n_ca
    theta_mask[:, 2] = angles[:, 3] # n_ca_c
    theta_mask[:, 3] = BB_BUILD_INFO["BONDANGS"]["ca-c-o"]
    torsion_mask[:, 0] = angles[:, 1] # n determined by psi of previous
    torsion_mask[1:, 1] = angles[:-1, 2] # ca determined by omega of previous
    torsion_mask[:, 2] = angles[:, 0] # c determined by phi
    torsion_mask[:, 3] = angles[:, 1] - np.pi 
    torsion_mask[-1, 3] += np.pi              
    to_fill = torsion_mask != torsion_mask
    to_pick = torsion_mask == 999
    for i in range(len(seq)):
        number = to_fill[i].long().sum()
        torsion_mask[i, to_fill[i]] = angles[i, 6:6+number]
        for j, val in enumerate(to_pick[i]):
            if val:
                torsion_mask[i, j] = torsion_mask[i, j-1] - np.pi

    return torch.stack([theta_mask, torsion_mask], dim=0).to(device)


# 定义函数，根据氨基酸序列生成索引掩码
def scn_index_mask(seq):
    """ Inputs: 
        * seq: (length). iterable of 1-letter aa codes of a protein
        Outputs: (L, 11, 3) maps point to theta and dihedral.
                 first angle is theta, second is dihedral
    """ 
    # 获取索引掩码
    idxs = torch.tensor([SUPREME_INFO[aa]['idx_mask'] for aa in seq])
    return rearrange(idxs, 'l s d -> d l s')


# 定义函数，根据氨基酸序列和角度生成蛋白质骨架
def build_scaffolds_from_scn_angles(seq, angles, coords=None, device="auto"):
    """ Builds scaffolds for fast access to data
        Inputs: 
        * seq: string of aas (1 letter code)
        * angles: (L, 12) tensor containing the internal angles.
                  Distributed as follows (following sidechainnet convention):
                  * (L, 3) for torsion angles
                  * (L, 3) bond angles
                  * (L, 6) sidechain angles
        * coords: (L, 3) sidechainnet coords. builds the mask with those instead
                  (better accuracy if modified residues present).
        Outputs:
        * cloud_mask: (L, 14 ) mask of points that should be converted to coords 
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, L, 14) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom
    """
    precise = angles.type()
    if device == "auto":
        device = angles.device

    if coords is not None: 
        cloud_mask = scn_cloud_mask(seq, coords=coords)
    else: 
        cloud_mask = scn_cloud_mask(seq)

    cloud_mask = torch.tensor(cloud_mask).bool().to(device)
    # 生成点云索引掩码，将其转换为长整型张量，并移动到指定设备上
    point_ref_mask = torch.tensor(scn_index_mask(seq)).long().to(device)
     
    # 生成角度掩码，将其转换为指定精度类型的张量，并移动到指定设备上
    angles_mask = torch.tensor(scn_angle_mask(seq, angles)).type(precise).to(device)
     
    # 生成键合掩码，将其转换为指定精度类型的张量，并移动到指定设备上
    bond_mask = torch.tensor(scn_bond_mask(seq)).type(precise).to(device)
    # 将所有结果以字典形式返回
    return {"cloud_mask":     cloud_mask, 
            "point_ref_mask": point_ref_mask,
            "angles_mask":    angles_mask,
            "bond_mask":      bond_mask }
#############################
####### ENCODERS ############
#############################


# 修改蛋白质支架的坐标信息
def modify_scaffolds_with_coords(scaffolds, coords):
    """ Gets scaffolds and fills in the right data.
        Inputs: 
        * scaffolds: dict. as returned by `build_scaffolds_from_scn_angles`
        * coords: (L, 14, 3). sidechainnet tensor. same device as scaffolds
        Outputs: corrected scaffolds
    """

    # 计算距离并更新:
    # N, CA, C
    scaffolds["bond_mask"][1:, 0] = torch.norm(coords[1:, 0] - coords[:-1, 2], dim=-1) # N
    scaffolds["bond_mask"][:, 1] = torch.norm(coords[:, 1] - coords[:, 0], dim=-1) # CA
    scaffolds["bond_mask"][:, 2] = torch.norm(coords[:, 2] - coords[:, 1], dim=-1) # C
    # O, CB, 侧链
    selector = np.arange(len(coords))
    for i in range(3, 14):
        # 获取索引
        idx_a, idx_b, idx_c = scaffolds["point_ref_mask"][:, :, i-3] # (3, L, 11) -> 3 * (L, 11)
        # 修正距离
        scaffolds["bond_mask"][:, i] = torch.norm(coords[:, i] - coords[selector, idx_c], dim=-1)
        # 获取角度
        scaffolds["angles_mask"][0, :, i] = get_angle(coords[selector, idx_b], 
                                                      coords[selector, idx_c], 
                                                      coords[:, i])
        # 处理 C-beta，其中请求的 C 来自前一个氨基酸
        if i == 4:
            # 对于第一个氨基酸，使用第二个氨基酸的 N 位置
            first_next_n = coords[1, :1] # 1, 3
            # 请求的 C 来自前一个氨基酸
            main_c_prev_idxs = coords[selector[:-1], idx_a[1:]] # (L-1), 3
            # 连接
            coords_a = torch.cat([first_next_n, main_c_prev_idxs])
        else:
            coords_a = coords[selector, idx_a]
        # 获取二面角
        scaffolds["angles_mask"][1, :, i] = get_dihedral(coords_a,
                                                         coords[selector, idx_b], 
                                                         coords[selector, idx_c], 
                                                         coords[:, i])
    # 为主链修正角度和二面角
    scaffolds["angles_mask"][0, :-1, 0] = get_angle(coords[:-1, 1], coords[:-1, 2], coords[1:, 0]) # ca_c_n
    scaffolds["angles_mask"][0, 1:, 1] = get_angle(coords[:-1, 2], coords[1:, 0], coords[1:, 1]) # c_n_ca
    scaffolds["angles_mask"][0, :, 2] = get_angle(coords[:, 0], coords[:, 1], coords[:, 2]) # n_ca_c
    
    # N 由前一个 psi 决定 = f(n, ca, c, n+1)
    scaffolds["angles_mask"][1, :-1, 0] = get_dihedral(coords[:-1, 0], coords[:-1, 1], coords[:-1, 2], coords[1:, 0])
    # CA 由 omega 决定 = f(ca, c, n+1, ca+1)
    scaffolds["angles_mask"][1, 1:, 1] = get_dihedral(coords[:-1, 1], coords[:-1, 2], coords[1:, 0], coords[1:, 1])
    # C 由 phi 决定 = f(c-1, n, ca, c)
    scaffolds["angles_mask"][1, 1:, 2] = get_dihedral(coords[:-1, 2], coords[1:, 0], coords[1:, 1], coords[1:, 2])

    return scaffolds



if __name__ == "__main__":
    print(scn_cloud_mask("AAAA"))
```