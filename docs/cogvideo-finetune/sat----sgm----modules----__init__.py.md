# `.\cogvideo-finetune\sat\sgm\modules\__init__.py`

```py
# 从相对路径导入 GeneralConditioner 模块
from .encoders.modules import GeneralConditioner

# 定义一个无条件配置的字典
UNCONDITIONAL_CONFIG = {
    # 设置目标为 GeneralConditioner 模块
    "target": "sgm.modules.GeneralConditioner",
    # 定义参数，初始化 emb_models 为一个空列表
    "params": {"emb_models": []},
}
```