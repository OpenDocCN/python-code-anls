# `.\diffusers\models\transformers\__init__.py`

```py
# 从 utils 模块导入判断是否可用的函数
from ...utils import is_torch_available

# 检查是否可用 Torch 库
if is_torch_available():
    # 从当前包导入 2D AuraFlow Transformer 模型
    from .auraflow_transformer_2d import AuraFlowTransformer2DModel
    # 从当前包导入 3D CogVideoX Transformer 模型
    from .cogvideox_transformer_3d import CogVideoXTransformer3DModel
    # 从当前包导入 2D DiT Transformer 模型
    from .dit_transformer_2d import DiTTransformer2DModel
    # 从当前包导入 2D Dual Transformer 模型
    from .dual_transformer_2d import DualTransformer2DModel
    # 从当前包导入 2D Hunyuan DiT Transformer 模型
    from .hunyuan_transformer_2d import HunyuanDiT2DModel
    # 从当前包导入 3D Latte Transformer 模型
    from .latte_transformer_3d import LatteTransformer3DModel
    # 从当前包导入 2D Lumina Next DiT Transformer 模型
    from .lumina_nextdit2d import LuminaNextDiT2DModel
    # 从当前包导入 2D PixArt Transformer 模型
    from .pixart_transformer_2d import PixArtTransformer2DModel
    # 从当前包导入 Prior Transformer 模型
    from .prior_transformer import PriorTransformer
    # 从当前包导入 Stable Audio DiT Transformer 模型
    from .stable_audio_transformer import StableAudioDiTModel
    # 从当前包导入 T5 Film Decoder 模型
    from .t5_film_transformer import T5FilmDecoder
    # 从当前包导入 2D Transformer 模型
    from .transformer_2d import Transformer2DModel
    # 从当前包导入 2D Flux Transformer 模型
    from .transformer_flux import FluxTransformer2DModel
    # 从当前包导入 2D SD3 Transformer 模型
    from .transformer_sd3 import SD3Transformer2DModel
    # 从当前包导入 Temporal Transformer 模型
    from .transformer_temporal import TransformerTemporalModel
```