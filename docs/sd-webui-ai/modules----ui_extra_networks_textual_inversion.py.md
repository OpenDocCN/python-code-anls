# `stable-diffusion-webui\modules\ui_extra_networks_textual_inversion.py`

```py
# 导入所需的模块
import os
from modules import ui_extra_networks, sd_hijack, shared
from modules.ui_extra_networks import quote_js

# 创建一个名为ExtraNetworksPageTextualInversion的类，继承自ExtraNetworksPage类
class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
    # 初始化方法，设置页面名称和是否允许负数提示
    def __init__(self):
        super().__init__('Textual Inversion')
        self.allow_negative_prompt = True

    # 刷新方法，加载文本反转的嵌入数据库
    def refresh(self):
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)

    # 创建项目方法，根据名称创建项目
    def create_item(self, name, index=None, enable_filter=True):
        # 获取名称对应的嵌入
        embedding = sd_hijack.model_hijack.embedding_db.word_embeddings.get(name)
        if embedding is None:
            return

        # 获取嵌入文件的路径和扩展名
        path, ext = os.path.splitext(embedding.filename)
        return {
            "name": name,
            "filename": embedding.filename,
            "shorthash": embedding.shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(embedding.filename) + " " + (embedding.hash or ""),
            "prompt": quote_js(embedding.name),
            "local_preview": f"{path}.preview.{shared.opts.samples_format}",
            "sort_keys": {'default': index, **self.get_sort_keys(embedding.filename)},
        }

    # 列出项目方法，返回项目列表
    def list_items(self):
        # 实例化一个列表以防止并发修改
        names = list(sd_hijack.model_hijack.embedding_db.word_embeddings)
        for index, name in enumerate(names):
            item = self.create_item(name, index)
            if item is not None:
                yield item

    # 允许预览的目录列表
    def allowed_directories_for_previews(self):
        return list(sd_hijack.model_hijack.embedding_db.embedding_dirs)
```