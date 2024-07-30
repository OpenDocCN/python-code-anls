# `.\yolov8\ultralytics\nn\modules\__init__.py`

```py
# 导入模块和类别，用于Ultralytics YOLO框架的神经网络模块
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxslim {f} {f} && open {f}')  # pip install onnxslim
    ```py
"""

from .block import (
    C1,             # 导入模块中的类别 C1
    C2,             # 导入模块中的类别 C2
    C3,             # 导入模块中的类别 C3
    C3TR,           # 导入模块中的类别 C3TR
    CIB,            # 导入模块中的类别 CIB
    DFL,            # 导入模块中的类别 DFL
    ELAN1,          # 导入模块中的类别 ELAN1
    PSA,            # 导入模块中的类别 PSA
    SPP,            # 导入模块中的类别 SPP
    SPPELAN,        # 导入模块中的类别 SPPELAN
    SPPF,           # 导入模块中的类别 SPPF
    AConv,          # 导入模块中的类别 AConv
    ADown,          # 导入模块中的类别 ADown
    Attention,      # 导入模块中的类别 Attention
    BNContrastiveHead,  # 导入模块中的类别 BNContrastiveHead
    Bottleneck,     # 导入模块中的类别 Bottleneck
    BottleneckCSP,  # 导入模块中的类别 BottleneckCSP
    C2f,            # 导入模块中的类别 C2f
    C2fAttn,        # 导入模块中的类别 C2fAttn
    C2fCIB,         # 导入模块中的类别 C2fCIB
    C3Ghost,        # 导入模块中的类别 C3Ghost
    C3x,            # 导入模块中的类别 C3x
    CBFuse,         # 导入模块中的类别 CBFuse
    CBLinear,       # 导入模块中的类别 CBLinear
    ContrastiveHead,    # 导入模块中的类别 ContrastiveHead
    GhostBottleneck,    # 导入模块中的类别 GhostBottleneck
    HGBlock,        # 导入模块中的类别 HGBlock
    HGStem,         # 导入模块中的类别 HGStem
    ImagePoolingAttn,   # 导入模块中的类别 ImagePoolingAttn
    Proto,          # 导入模块中的类别 Proto
    RepC3,          # 导入模块中的类别 RepC3
    RepNCSPELAN4,   # 导入模块中的类别 RepNCSPELAN4
    RepVGGDW,       # 导入模块中的类别 RepVGGDW
    ResNetLayer,    # 导入模块中的类别 ResNetLayer
    SCDown,         # 导入模块中的类别 SCDown
)

from .conv import (
    CBAM,           # 导入模块中的类别 CBAM
    ChannelAttention,   # 导入模块中的类别 ChannelAttention
    Concat,         # 导入模块中的类别 Concat
    Conv,           # 导入模块中的类别 Conv
    Conv2,          # 导入模块中的类别 Conv2
    ConvTranspose,  # 导入模块中的类别 ConvTranspose
    DWConv,         # 导入模块中的类别 DWConv
    DWConvTranspose2d,  # 导入模块中的类别 DWConvTranspose2d
    Focus,          # 导入模块中的类别 Focus
    GhostConv,      # 导入模块中的类别 GhostConv
    LightConv,      # 导入模块中的类别 LightConv
    RepConv,        # 导入模块中的类别 RepConv
    SpatialAttention,   # 导入模块中的类别 SpatialAttention
)

from .head import (
    OBB,            # 导入模块中的类别 OBB
    Classify,       # 导入模块中的类别 Classify
    Detect,         # 导入模块中的类别 Detect
    Pose,           # 导入模块中的类别 Pose
    RTDETRDecoder,  # 导入模块中的类别 RTDETRDecoder
    Segment,        # 导入模块中的类别 Segment
    WorldDetect,    # 导入模块中的类别 WorldDetect
    v10Detect,      # 导入模块中的类别 v10Detect
)

from .transformer import (
    AIFI,           # 导入模块中的类别 AIFI
    MLP,            # 导入模块中的类别 MLP
    DeformableTransformerDecoder,   # 导入模块中的类别 DeformableTransformerDecoder
    DeformableTransformerDecoderLayer,   # 导入模块中的类别 DeformableTransformerDecoderLayer
    LayerNorm2d,    # 导入模块中的类别 LayerNorm2d
    MLPBlock,       # 导入模块中的类别 MLPBlock
    MSDeformAttn,   # 导入模块中的类别 MSDeformAttn
    TransformerBlock,   # 导入模块中的类别 TransformerBlock
    TransformerEncoderLayer,    # 导入模块中的类别 TransformerEncoderLayer
    TransformerLayer,   # 导入模块中的类别 TransformerLayer
)

__all__ = (
    "Conv",         # 将 Conv 加入到模块的公开 API 列表中
    "Conv2",        # 将 Conv2 加入到模块的公开 API 列表中
    "LightConv",    # 将 LightConv 加入到模块的公开 API 列表中
    "RepConv",      # 将 RepConv 加入到模块的公开 API 列表中
    "DWConv",       # 将 DWConv 加入到模块的公开 API 列表中
    "DWConvTranspose2d",    # 将 DWConvTranspose2d 加入到模块的公开 API 列表中
    "ConvTranspose",    # 将 ConvTranspose 加入到模块的公开 API 列表中
    "Focus",        # 将 Focus 加入到模块的公开 API 列表中
    "GhostConv",    # 将 GhostConv 加入到模块的公开 API 列表中
    "ChannelAttention", # 将 ChannelAttention 加入到模块的公开 API 列表中
    "SpatialAttention",    # 将 SpatialAttention 加入到模块的公开 API 列表中
    "CBAM",         # 将 CBAM 加入到模块的公开 API 列表中
    "Concat",       # 将 Concat 加入到模块的公开 API 列表中
    "TransformerLayer", # 将 TransformerLayer 加入到模块的公开 API 列表中
    "TransformerBlock", # 将 TransformerBlock 加入到模块的公开 API 列表中
    "MLPBlock",     # 将 MLPBlock 加入到模块的公开 API 列表中
    "LayerNorm2d",  # 将 LayerNorm2d 加入到模块的公开 API 列表中
    "DFL",          # 将 DFL 加入到模块的公开 API 列表中
    "HGBlock",      # 将 HGBlock 加入到模块的公开 API 列表中
    "HGStem",       # 将 HGStem 加入到模块的公开 API 列表中
```