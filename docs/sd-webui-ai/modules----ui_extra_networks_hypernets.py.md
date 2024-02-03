# `stable-diffusion-webui\modules\ui_extra_networks_hypernets.py`

```py
# 导入 os 模块
import os

# 从 modules 模块中导入 shared 和 ui_extra_networks 模块
from modules import shared, ui_extra_networks
# 从 modules.ui_extra_networks 模块中导入 quote_js 函数
from modules.ui_extra_networks import quote_js
# 从 modules.hashes 模块中导入 sha256_from_cache 函数
from modules.hashes import sha256_from_cache

# 创建 ExtraNetworksPageHypernetworks 类，继承自 ExtraNetworksPage 类
class ExtraNetworksPageHypernetworks(ui_extra_networks.ExtraNetworksPage):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法，设置页面名称为 'Hypernetworks'
        super().__init__('Hypernetworks')

    # 刷新方法
    def refresh(self):
        # 重新加载超网络
        shared.reload_hypernetworks()

    # 创建项目方法
    def create_item(self, name, index=None, enable_filter=True):
        # 获取超网络的完整路径
        full_path = shared.hypernetworks.get(name)
        # 如果路径为空，则返回空
        if full_path is None:
            return

        # 分离路径和扩展名
        path, ext = os.path.splitext(full_path)
        # 计算文件的 SHA256 值
        sha256 = sha256_from_cache(full_path, f'hypernet/{name}')
        # 获取 SHA256 值的前10位作为缩略哈希值
        shorthash = sha256[0:10] if sha256 else None

        # 返回包含项目信息的字典
        return {
            "name": name,
            "filename": full_path,
            "shorthash": shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(path) + " " + (sha256 or ""),
            "prompt": quote_js(f"<hypernet:{name}:") + " + opts.extra_networks_default_multiplier + " + quote_js(">"),
            "local_preview": f"{path}.preview.{shared.opts.samples_format}",
            "sort_keys": {'default': index, **self.get_sort_keys(path + ext)},
        }

    # 列出项目方法
    def list_items(self):
        # 实例化一个列表以防止并发修改
        names = list(shared.hypernetworks)
        # 遍历超网络名称列表
        for index, name in enumerate(names):
            # 创建项目
            item = self.create_item(name, index)
            # 如果项目不为空，则返回项目
            if item is not None:
                yield item

    # 获取预览允许的目录方法
    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.hypernetwork_dir]
```