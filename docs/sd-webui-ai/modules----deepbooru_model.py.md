# `stable-diffusion-webui\modules\deepbooru_model.py`

```py
# 导入 torch 库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入神经网络函数模块
import torch.nn.functional as F

# 从自定义模块 modules 中导入 devices
from modules import devices

# 查看更多信息请访问 https://github.com/AUTOMATIC1111/TorchDeepDanbooru

# 定义 DeepDanbooruModel 类，继承自 nn.Module
class DeepDanbooruModel(nn.Module):
    # 重写 load_state_dict 方法
    def load_state_dict(self, state_dict, **kwargs):
        # 从 state_dict 中获取 'tags' 键对应的值，如果不存在则返回空列表
        self.tags = state_dict.get('tags', [])
        
        # 调用父类 nn.Module 的 load_state_dict 方法，传入过滤掉 'tags' 键的 state_dict
        super(DeepDanbooruModel, self).load_state_dict({k: v for k, v in state_dict.items() if k != 'tags'})
```