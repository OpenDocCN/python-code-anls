# `.\sagemaker\__init__.py`

```py
# 导入 SageMakerTrainer 类从 trainer_sm 模块中
from .trainer_sm import SageMakerTrainer
# 导入 SageMakerTrainingArguments 和 is_sagemaker_dp_enabled 从 training_args_sm 模块中
from .training_args_sm import SageMakerTrainingArguments, is_sagemaker_dp_enabled
```