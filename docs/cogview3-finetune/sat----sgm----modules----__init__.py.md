# `.\cogview3-finetune\sat\sgm\modules\__init__.py`

```py
# 从当前包的编码器模块导入 GeneralConditioner 类
from .encoders.modules import GeneralConditioner

# 定义一个无条件配置字典，包含目标和参数
UNCONDITIONAL_CONFIG = {
    # 设定目标为 sgm.modules.GeneralConditioner
    "target": "sgm.modules.GeneralConditioner",
    # 定义参数，emb_models 为空列表
    "params": {"emb_models": []},
}
```