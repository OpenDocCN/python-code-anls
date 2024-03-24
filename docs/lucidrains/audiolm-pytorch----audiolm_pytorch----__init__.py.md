# `.\lucidrains\audiolm-pytorch\audiolm_pytorch\__init__.py`

```
# 导入 torch 库
import torch
# 导入版本模块
from packaging import version

# 检查 torch 版本是否大于等于 '2.0.0'，如果是则执行以下操作
if version.parse(torch.__version__) >= version.parse('2.0.0'):
    # 从 einops._torch_specific 模块中导入 allow_ops_in_compiled_graph 函数
    from einops._torch_specific import allow_ops_in_compiled_graph
    # 调用 allow_ops_in_compiled_graph 函数

# 从 audiolm_pytorch.audiolm_pytorch 模块中导入 AudioLM 类
from audiolm_pytorch.audiolm_pytorch import AudioLM
# 从 audiolm_pytorch.soundstream 模块中导入 SoundStream, AudioLMSoundStream, MusicLMSoundStream 类
from audiolm_pytorch.soundstream import SoundStream, AudioLMSoundStream, MusicLMSoundStream
# 从 audiolm_pytorch.encodec 模块中导入 EncodecWrapper 类

# 从 audiolm_pytorch.audiolm_pytorch 模块中导入 SemanticTransformer, CoarseTransformer, FineTransformer 类
from audiolm_pytorch.audiolm_pytorch import SemanticTransformer, CoarseTransformer, FineTransformer
# 从 audiolm_pytorch.audiolm_pytorch 模块中导入 FineTransformerWrapper, CoarseTransformerWrapper, SemanticTransformerWrapper 类

# 从 audiolm_pytorch.vq_wav2vec 模块中导入 FairseqVQWav2Vec 类
from audiolm_pytorch.vq_wav2vec import FairseqVQWav2Vec
# 从 audiolm_pytorch.hubert_kmeans 模块中导入 HubertWithKmeans 类

# 从 audiolm_pytorch.trainer 模块中导入 SoundStreamTrainer, SemanticTransformerTrainer, FineTransformerTrainer, CoarseTransformerTrainer 类

# 从 audiolm_pytorch.audiolm_pytorch 模块中导入 get_embeds 函数
from audiolm_pytorch.audiolm_pytorch import get_embeds
```