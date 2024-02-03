# `stable-diffusion-webui\extensions-builtin\Lora\lora_patches.py`

```
# 导入 torch 库
import torch

# 导入自定义的网络模块
import networks
# 导入 patches 模块
from modules import patches

# 定义一个类 LoraPatches
class LoraPatches:
    # 初始化方法
    def __init__(self):
        # 为 torch.nn.Linear 类的 forward 方法添加补丁
        self.Linear_forward = patches.patch(__name__, torch.nn.Linear, 'forward', networks.network_Linear_forward)
        # 为 torch.nn.Linear 类的 _load_from_state_dict 方法添加补丁
        self.Linear_load_state_dict = patches.patch(__name__, torch.nn.Linear, '_load_from_state_dict', networks.network_Linear_load_state_dict)
        # 为 torch.nn.Conv2d 类的 forward 方法添加补丁
        self.Conv2d_forward = patches.patch(__name__, torch.nn.Conv2d, 'forward', networks.network_Conv2d_forward)
        # 为 torch.nn.Conv2d 类的 _load_from_state_dict 方法添加补丁
        self.Conv2d_load_state_dict = patches.patch(__name__, torch.nn.Conv2d, '_load_from_state_dict', networks.network_Conv2d_load_state_dict)
        # 为 torch.nn.GroupNorm 类的 forward 方法添加补丁
        self.GroupNorm_forward = patches.patch(__name__, torch.nn.GroupNorm, 'forward', networks.network_GroupNorm_forward)
        # 为 torch.nn.GroupNorm 类的 _load_from_state_dict 方法添加补丁
        self.GroupNorm_load_state_dict = patches.patch(__name__, torch.nn.GroupNorm, '_load_from_state_dict', networks.network_GroupNorm_load_state_dict)
        # 为 torch.nn.LayerNorm 类的 forward 方法添加补丁
        self.LayerNorm_forward = patches.patch(__name__, torch.nn.LayerNorm, 'forward', networks.network_LayerNorm_forward)
        # 为 torch.nn.LayerNorm 类的 _load_from_state_dict 方法添加补丁
        self.LayerNorm_load_state_dict = patches.patch(__name__, torch.nn.LayerNorm, '_load_from_state_dict', networks.network_LayerNorm_load_state_dict)
        # 为 torch.nn.MultiheadAttention 类的 forward 方法添加补丁
        self.MultiheadAttention_forward = patches.patch(__name__, torch.nn.MultiheadAttention, 'forward', networks.network_MultiheadAttention_forward)
        # 为 torch.nn.MultiheadAttention 类的 _load_from_state_dict 方法添加补丁
        self.MultiheadAttention_load_state_dict = patches.patch(__name__, torch.nn.MultiheadAttention, '_load_from_state_dict', networks.network_MultiheadAttention_load_state_dict)
    # 撤销对 Linear 类的 forward 方法的修改
    self.Linear_forward = patches.undo(__name__, torch.nn.Linear, 'forward')
    # 撤销对 Linear 类的 _load_from_state_dict 方法的修改
    self.Linear_load_state_dict = patches.undo(__name__, torch.nn.Linear, '_load_from_state_dict')
    # 撤销对 Conv2d 类的 forward 方法的修改
    self.Conv2d_forward = patches.undo(__name__, torch.nn.Conv2d, 'forward')
    # 撤销对 Conv2d 类的 _load_from_state_dict 方法的修改
    self.Conv2d_load_state_dict = patches.undo(__name__, torch.nn.Conv2d, '_load_from_state_dict')
    # 撤销对 GroupNorm 类的 forward 方法的修改
    self.GroupNorm_forward = patches.undo(__name__, torch.nn.GroupNorm, 'forward')
    # 撤销对 GroupNorm 类的 _load_from_state_dict 方法的修改
    self.GroupNorm_load_state_dict = patches.undo(__name__, torch.nn.GroupNorm, '_load_from_state_dict')
    # 撤销对 LayerNorm 类的 forward 方法的修改
    self.LayerNorm_forward = patches.undo(__name__, torch.nn.LayerNorm, 'forward')
    # 撤销对 LayerNorm 类的 _load_from_state_dict 方法的修改
    self.LayerNorm_load_state_dict = patches.undo(__name__, torch.nn.LayerNorm, '_load_from_state_dict')
    # 撤销对 MultiheadAttention 类的 forward 方法的修改
    self.MultiheadAttention_forward = patches.undo(__name__, torch.nn.MultiheadAttention, 'forward')
    # 撤销对 MultiheadAttention 类的 _load_from_state_dict 方法的修改
    self.MultiheadAttention_load_state_dict = patches.undo(__name__, torch.nn.MultiheadAttention, '_load_from_state_dict')
```