# `.\lucidrains\DALLE2-pytorch\dalle2_pytorch\__init__.py`

```
# 从dalle2_pytorch版本模块中导入版本号
from dalle2_pytorch.version import __version__
# 从dalle2_pytorch模块中导入DALLE2类、DiffusionPriorNetwork类、DiffusionPrior类、Unet类和Decoder类
from dalle2_pytorch.dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder
# 从dalle2_pytorch模块中导入OpenAIClipAdapter类和OpenClipAdapter类
from dalle2_pytorch.dalle2_pytorch import OpenAIClipAdapter, OpenClipAdapter
# 从dalle2_pytorch模块中导入DecoderTrainer类和DiffusionPriorTrainer类
from dalle2_pytorch.trainer import DecoderTrainer, DiffusionPriorTrainer

# 从dalle2_pytorch模块中导入VQGanVAE类
from dalle2_pytorch.vqgan_vae import VQGanVAE
# 从x_clip模块中导入CLIP类
from x_clip import CLIP
```