# `stable-diffusion-webui\modules\ui_extra_networks_checkpoints.py`

```
# 导入所需的模块
import html
import os

# 从模块中导入指定内容
from modules import shared, ui_extra_networks, sd_models
from modules.ui_extra_networks import quote_js
from modules.ui_extra_networks_checkpoints_user_metadata import CheckpointUserMetadataEditor

# 创建一个名为ExtraNetworksPageCheckpoints的类，继承自ExtraNetworksPage类
class ExtraNetworksPageCheckpoints(ui_extra_networks.ExtraNetworksPage):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__('Checkpoints')

        # 设置属性allow_prompt为False
        self.allow_prompt = False

    # 刷新方法
    def refresh(self):
        # 调用shared模块中的refresh_checkpoints方法
        shared.refresh_checkpoints()

    # 创建项目方法
    def create_item(self, name, index=None, enable_filter=True):
        # 获取指定名称的检查点信息
        checkpoint: sd_models.CheckpointInfo = sd_models.checkpoint_aliases.get(name)
        # 如果检查点信息不存在，则返回空
        if checkpoint is None:
            return

        # 获取文件路径和扩展名
        path, ext = os.path.splitext(checkpoint.filename)
        # 返回包含项目信息的字典
        return {
            "name": checkpoint.name_for_extra,
            "filename": checkpoint.filename,
            "shorthash": checkpoint.shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(checkpoint.filename) + " " + (checkpoint.sha256 or ""),
            "onclick": '"' + html.escape(f"""return selectCheckpoint({quote_js(name)})""") + '"',
            "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": checkpoint.metadata,
            "sort_keys": {'default': index, **self.get_sort_keys(checkpoint.filename)},
        }

    # 列出项目方法
    def list_items(self):
        # 实例化一个列表以防止并发修改
        names = list(sd_models.checkpoints_list)
        # 遍历检查点列表，创建项目
        for index, name in enumerate(names):
            item = self.create_item(name, index)
            if item is not None:
                yield item

    # 获取预览允许的目录方法
    def allowed_directories_for_previews(self):
        # 返回允许预览的目录列表
        return [v for v in [shared.cmd_opts.ckpt_dir, sd_models.model_path] if v is not None]

    # 创建用户元数据编辑器方法
    def create_user_metadata_editor(self, ui, tabname):
        # 返回CheckpointUserMetadataEditor对象
        return CheckpointUserMetadataEditor(ui, tabname, self)
```