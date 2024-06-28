# `.\models\esm\openfold_utils\__init__.py`

```
# 导入自定义模块中的函数和类

from .chunk_utils import chunk_layer
# 导入 chunk_layer 函数，用于处理数据的分块操作

from .data_transforms import make_atom14_masks
# 导入 make_atom14_masks 函数，用于生成 atom14 掩码

from .feats import atom14_to_atom37, frames_and_literature_positions_to_atom14_pos, torsion_angles_to_frames
# 导入三个函数：atom14_to_atom37、frames_and_literature_positions_to_atom14_pos、torsion_angles_to_frames，
# 用于特征转换和处理

from .loss import compute_predicted_aligned_error, compute_tm
# 导入 compute_predicted_aligned_error 和 compute_tm 函数，用于计算损失

from .protein import Protein as OFProtein
# 导入 Protein 类，并将其命名为 OFProtein，用于处理蛋白质数据

from .protein import to_pdb
# 导入 to_pdb 函数，用于将蛋白质数据输出为 PDB 文件

from .rigid_utils import Rigid, Rotation
# 导入 Rigid 和 Rotation 类，用于刚体和旋转操作的工具函数

from .tensor_utils import dict_multimap, flatten_final_dims, permute_final_dims
# 导入 dict_multimap、flatten_final_dims 和 permute_final_dims 函数，
# 用于处理张量的映射、维度展平和维度置换操作
```