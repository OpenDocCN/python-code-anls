# `.\diffusers\models\unets\__init__.py`

```py
# 从工具模块中导入检测 PyTorch 和 Flax 是否可用的函数
from ...utils import is_flax_available, is_torch_available

# 如果 PyTorch 可用，则导入相应的 UNet 模型
if is_torch_available():
    # 导入一维 UNet 模型
    from .unet_1d import UNet1DModel
    # 导入二维 UNet 模型
    from .unet_2d import UNet2DModel
    # 导入条件二维 UNet 模型
    from .unet_2d_condition import UNet2DConditionModel
    # 导入条件三维 UNet 模型
    from .unet_3d_condition import UNet3DConditionModel
    # 导入 I2VGenXL UNet 模型
    from .unet_i2vgen_xl import I2VGenXLUNet
    # 导入 Kandinsky3 UNet 模型
    from .unet_kandinsky3 import Kandinsky3UNet
    # 导入运动模型适配器和 UNet 运动模型
    from .unet_motion_model import MotionAdapter, UNetMotionModel
    # 导入时空条件的 UNet 模型
    from .unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
    # 导入稳定级联 UNet 模型
    from .unet_stable_cascade import StableCascadeUNet
    # 导入二维 UVit 模型
    from .uvit_2d import UVit2DModel

# 如果 Flax 可用，则导入相应的条件二维 UNet 模型
if is_flax_available():
    # 导入条件二维 Flax UNet 模型
    from .unet_2d_condition_flax import FlaxUNet2DConditionModel
```