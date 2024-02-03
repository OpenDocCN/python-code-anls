# `stable-diffusion-webui\extensions-builtin\Lora\ui_extra_networks_lora.py`

```
# 导入 os 模块
import os

# 导入 network 模块
import network

# 导入 networks 模块
import networks

# 从 modules 模块中导入 shared 和 ui_extra_networks
from modules import shared, ui_extra_networks

# 从 modules.ui_extra_networks 模块中导入 quote_js 函数
from modules.ui_extra_networks import quote_js

# 从 ui_edit_user_metadata 模块中导入 LoraUserMetadataEditor 类
from ui_edit_user_metadata import LoraUserMetadataEditor

# 创建 ExtraNetworksPageLora 类，继承自 ExtraNetworksPage 类
class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法，传入 'Lora' 参数
        super().__init__('Lora')

    # 刷新方法
    def refresh(self):
        # 调用 networks 模块的 list_available_networks 方法
        networks.list_available_networks()

    # 列出项目方法
    def list_items(self):
        # 实例化一个列表以防止并发修改
        names = list(networks.available_networks)
        # 遍历 names 列表，获取索引和名称
        for index, name in enumerate(names):
            # 调用 create_item 方法创建项目
            item = self.create_item(name, index)
            # 如果项目不为空，则返回该项目
            if item is not None:
                yield item

    # 获取预览允许的目录列表方法
    def allowed_directories_for_previews(self):
        # 返回允许预览的目录列表
        return [shared.cmd_opts.lora_dir, shared.cmd_opts.lyco_dir_backcompat]

    # 创建用户元数据编辑器方法
    def create_user_metadata_editor(self, ui, tabname):
        # 返回 LoraUserMetadataEditor 类的实例
        return LoraUserMetadataEditor(ui, tabname, self)
```