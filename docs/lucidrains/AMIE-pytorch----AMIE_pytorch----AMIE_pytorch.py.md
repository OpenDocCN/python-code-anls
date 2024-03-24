# `.\lucidrains\AMIE-pytorch\AMIE_pytorch\AMIE_pytorch.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum 模块
from torch import nn, einsum
# 从 torch.nn 模块中导入 Module, ModuleList 类
from torch.nn import Module, ModuleList

# 导入 einops 库中的 rearrange 函数
from einops import rearrange

# 定义函数

# 判断变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 自我评论提示
# 论文中的图 A.15

PROMPT_EVALUATE_EXPLANATION = """
I have a doctor-patient dialogue and the corresponding rating that quantifies its quality according to
the following criterion: <criterion> (e.g., maintaining patient welfare). The rating of the dialogue is
on a scale of 1 to 5 where:

5: <definition> e.g., “Treats patient respectfully, and ensures comfort, safety and dignity”
1: <definition> e.g., “Causes patient physical or emotional discomfort AND jeopardises patient safety”

First, describe which parts of the dialogue are good with respect to the criterion. Then, describe which parts are bad with respect to the criterion. Lastly, summarise the above to explain the
provided rating, using the following format:

Good: ...
Bad: ...
Summary: ...

DIALOGUE: <dialogue>
Rating: <human rating>
EVALUATION:
"""

# 图 A.16

PROMPT_EVALUATE_QUALITATIVE = """
I have a doctor-patient dialogue which I would like you to evaluate on the following criterion:
<criterion> (e.g., maintaining patient welfare). The dialogue should be rated on a scale of 1-5 with
respect to the criterion where:

5: <definition> e.g., “Treats patient respectfully, and ensures comfort, safety and dignity”
1: <definition> e.g., “Causes patient physical or emotional discomfort AND jeopardises patient safety”

Here are some example dialogues and their ratings:
DIALOGUE: <example dialog>
EVALUATION: <example self-generated explanation>
Rating: <example rating>
...

Now, please rate the following dialogue as instructed below. First, describe which parts of the dialogue
are good with respect to the criterion. Then, describe which parts are bad with respect to the criterion.
Third, summarise the above findings. Lastly, rate the dialogue on a scale of 1-5 with respect to the
criterion, according to this schema:

Good: ...
Bad: ...
Summary: ...
Rating: ...

DIALOGUE: <dialogue>
EVALUATION:
"""

# 自我对弈模块

class OuterSelfPlay(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

class InnerSelfPlay(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

class PatientAgent(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

class ClinicalVignetteGenerator(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

class Moderator(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

class DoctorAgent(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

class SimulatedDialogue(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

class Critic(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

# 主类

class AMIE(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
```