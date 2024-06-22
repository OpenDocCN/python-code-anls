# `.\transformers\data\processors\__init__.py`

```py
# 导入 HuggingFace 库中的相关模块和函数

# 从 HuggingFace 库中的 glue 模块中导入函数和类
from .glue import glue_convert_examples_to_features, glue_output_modes, glue_processors, glue_tasks_num_labels
# 从 HuggingFace 库中的 squad 模块中导入类和函数
from .squad import SquadExample, SquadFeatures, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features
# 从 HuggingFace 库中的 utils 模块中导入类和函数
from .utils import DataProcessor, InputExample, InputFeatures, SingleSentenceClassificationProcessor
# 从 HuggingFace 库中的 xnli 模块中导入函数和类
from .xnli import xnli_output_modes, xnli_processors, xnli_tasks_num_labels
```