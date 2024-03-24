# `.\lucidrains\naturalspeech2-pytorch\naturalspeech2_pytorch\__init__.py`

```
# 导入 torch 库
import torch
# 导入版本比较模块 version
from packaging import version

# 检查 torch 库的版本是否大于等于 '2.0.0'，如果是则执行以下代码
if version.parse(torch.__version__) >= version.parse('2.0.0'):
    # 从 einops._torch_specific 模块中导入 allow_ops_in_compiled_graph 函数
    from einops._torch_specific import allow_ops_in_compiled_graph
    # 调用 allow_ops_in_compiled_graph 函数

# 从 naturalspeech2_pytorch.naturalspeech2_pytorch 模块中导入以下类
from naturalspeech2_pytorch.naturalspeech2_pytorch import (
    NaturalSpeech2,
    Transformer,
    Wavenet,
    Model,
    Trainer,
    PhonemeEncoder,
    DurationPitchPredictor,
    SpeechPromptEncoder,
    Tokenizer,
    ESpeak
)

# 从 audiolm_pytorch 模块中导入以下类
from audiolm_pytorch import (
    SoundStream,
    EncodecWrapper
)
```