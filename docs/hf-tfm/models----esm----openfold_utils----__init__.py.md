# `.\models\esm\openfold_utils\__init__.py`

```py
# 从相对路径导入模块中的函数或类
from .chunk_utils import chunk_layer
# 从相对路径导入模块中的函数或类
from .data_transforms import make_atom14_masks
# 从相对路径导入模块中的函数或类
from .feats import atom14_to_atom37, frames_and_literature_positions_to_atom14_pos, torsion_angles_to_frames
# 从相对路径导入模块中的函数或类
from .loss import compute_predicted_aligned_error, compute_tm
# 从相对路径导入模块中的类
from .protein import Protein as OFProtein
# 从相对路径导入模块中的函数
from .protein import to_pdb
# 从相对路径导入模块中的类
from .rigid_utils import Rigid, Rotation
# 从相对路径导入模块中的函数
from .tensor_utils import dict_multimap, flatten_final_dims, permute_final_dims
```