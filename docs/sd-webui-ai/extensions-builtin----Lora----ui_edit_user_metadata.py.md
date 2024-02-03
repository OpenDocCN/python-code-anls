# `stable-diffusion-webui\extensions-builtin\Lora\ui_edit_user_metadata.py`

```py
# 导入 datetime、html、random、gradio 和 re 模块
import datetime
import html
import random

import gradio as gr
import re

# 导入自定义模块 ui_extra_networks_user_metadata
from modules import ui_extra_networks_user_metadata

# 判断标签集是否为非逗号分隔的标签集
def is_non_comma_tagset(tags):
    # 计算标签平均长度
    average_tag_length = sum(len(x) for x in tags.keys()) / len(tags)

    return average_tag_length >= 16

# 编译正则表达式，匹配单词
re_word = re.compile(r"[-_\w']+")
# 编译正则表达式，匹配逗号
re_comma = re.compile(r" *, *")

# 构建标签
def build_tags(metadata):
    tags = {}

    # 遍历元数据中的标签频率信息
    for _, tags_dict in metadata.get("ss_tag_frequency", {}).items():
        for tag, tag_count in tags_dict.items():
            tag = tag.strip()
            tags[tag] = tags.get(tag, 0) + int(tag_count)

    # 如果存在标签且为非逗号分隔的标签集
    if tags and is_non_comma_tagset(tags):
        new_tags = {}

        # 遍历标签，提取单词并统计
        for text, text_count in tags.items():
            for word in re.findall(re_word, text):
                if len(word) < 3:
                    continue

                new_tags[word] = new_tags.get(word, 0) + text_count

        tags = new_tags

    # 根据标签出现次数排序标签
    ordered_tags = sorted(tags.keys(), key=tags.get, reverse=True)

    return [(tag, tags[tag]) for tag in ordered_tags]

# 定义 LoraUserMetadataEditor 类，继承自 UserMetadataEditor 类
class LoraUserMetadataEditor(ui_extra_networks_user_metadata.UserMetadataEditor):
    def __init__(self, ui, tabname, page):
        super().__init__(ui, tabname, page)

        # 初始化属性
        self.select_sd_version = None
        self.taginfo = None
        self.edit_activation_text = None
        self.slider_preferred_weight = None
        self.edit_notes = None

    # 保存 Lora 用户元数据
    def save_lora_user_metadata(self, name, desc, sd_version, activation_text, preferred_weight, notes):
        # 获取用户元数据
        user_metadata = self.get_user_metadata(name)
        # 更新用户元数据信息
        user_metadata["description"] = desc
        user_metadata["sd version"] = sd_version
        user_metadata["activation text"] = activation_text
        user_metadata["preferred weight"] = preferred_weight
        user_metadata["notes"] = notes

        # 写入用户元数据
        self.write_user_metadata(name, user_metadata)
    # 从父类中获取指定名称的元数据表格
    def get_metadata_table(self, name):
        # 调用父类方法获取元数据表格
        table = super().get_metadata_table(name)
        # 获取指定名称对应的项目元数据
        item = self.page.items.get(name, {})
        # 获取项目元数据中的元数据信息
        metadata = item.get("metadata") or {}

        # 定义关键字和对应的标签
        keys = {
            'ss_output_name': "Output name:",
            'ss_sd_model_name': "Model:",
            'ss_clip_skip': "Clip skip:",
            'ss_network_module': "Kohya module:",
        }

        # 遍历关键字和标签，将元数据信息添加到表格中
        for key, label in keys.items():
            value = metadata.get(key, None)
            if value is not None and str(value) != "None":
                table.append((label, html.escape(value)))

        # 获取训练开始时间并添加到表格中
        ss_training_started_at = metadata.get('ss_training_started_at')
        if ss_training_started_at:
            table.append(("Date trained:", datetime.datetime.utcfromtimestamp(float(ss_training_started_at)).strftime('%Y-%m-%d %H:%M'))

        # 获取桶信息并处理分辨率数据
        ss_bucket_info = metadata.get("ss_bucket_info")
        if ss_bucket_info and "buckets" in ss_bucket_info:
            resolutions = {}
            for _, bucket in ss_bucket_info["buckets"].items():
                resolution = bucket["resolution"]
                resolution = f'{resolution[1]}x{resolution[0]}'

                resolutions[resolution] = resolutions.get(resolution, 0) + int(bucket["count"])

            resolutions_list = sorted(resolutions.keys(), key=resolutions.get, reverse=True)
            resolutions_text = html.escape(", ".join(resolutions_list[0:4]))
            if len(resolutions) > 4:
                resolutions_text += ", ..."
                resolutions_text = f"<span title='{html.escape(', '.join(resolutions_list))}'>{resolutions_text}</span>"

            table.append(('Resolutions:' if len(resolutions_list) > 1 else 'Resolution:', resolutions_text))

        # 统计图像数量并添加到表格中
        image_count = 0
        for _, params in metadata.get("ss_dataset_dirs", {}).items():
            image_count += int(params.get("img_count", 0))

        if image_count:
            table.append(("Dataset size:", image_count))

        # 返回填充后的表格
        return table
    # 将值放入组件中
    def put_values_into_components(self, name):
        # 获取用户元数据
        user_metadata = self.get_user_metadata(name)
        # 调用父类方法，将值放入组件中
        values = super().put_values_into_components(name)

        # 获取页面中指定名称的项目
        item = self.page.items.get(name, {})
        # 获取项目的元数据，如果没有则为空字典
        metadata = item.get("metadata") or {}

        # 构建标签
        tags = build_tags(metadata)
        # 从标签中获取前24个标签和对应的计数
        gradio_tags = [(tag, str(count)) for tag, count in tags[0:24]]

        # 返回一组组件值
        return [
            *values[0:5],
            item.get("sd_version", "Unknown"),
            # 更新高亮文本组件的值为标签，如果有标签则可见，否则不可见
            gr.HighlightedText.update(value=gradio_tags, visible=True if tags else False),
            user_metadata.get('activation text', ''),
            float(user_metadata.get('preferred weight', 0.0)),
            # 更新组件可见性，如果有标签则可见，否则不可见
            gr.update(visible=True if tags else False),
            # 更新组件值为从标签生成的随机提示，如果有标签则可见，否则不可见
            gr.update(value=self.generate_random_prompt_from_tags(tags), visible=True if tags else False),
        ]

    # 生成随机提示
    def generate_random_prompt(self, name):
        # 获取页面中指定名称的项目
        item = self.page.items.get(name, {})
        # 获取项目的元数据，如果没有则为空字典
        metadata = item.get("metadata") or {}
        # 构建标签
        tags = build_tags(metadata)

        # 从标签生成随机提示
        return self.generate_random_prompt_from_tags(tags)

    # 根据标签生成随机提示
    def generate_random_prompt_from_tags(self, tags):
        max_count = None
        res = []
        for tag, count in tags:
            if not max_count:
                max_count = count

            v = random.random() * max_count
            if count > v:
                res.append(tag)

        # 将生成的随机提示以逗号分隔并排序
        return ", ".join(sorted(res))

    # 在左侧列中创建额外的默认项目
    def create_extra_default_items_in_left_column(self):

        # 创建下拉菜单组件，用于选择稳定扩散版本
        # 这个组件最好使用 gr.Radio，但我无法使其正常工作
        self.select_sd_version = gr.Dropdown(['SD1', 'SD2', 'SDXL', 'Unknown'], value='Unknown', label='Stable Diffusion version', interactive=True)
```