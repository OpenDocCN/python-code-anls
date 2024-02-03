# `stable-diffusion-webui\modules\options.py`

```
# 导入所需的模块
import json
import sys
from dataclasses import dataclass

# 导入 gradio 模块并重命名为 gr
import gradio as gr

# 导入自定义模块 errors 和 shared_cmd_options
from modules import errors
from modules.shared_cmd_options import cmd_opts

# 定义 OptionInfo 类，用于存储选项信息
class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None, comment_before='', comment_after='', infotext=None, restrict_api=False, category_id=None):
        # 初始化各个属性
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.category_id = category_id
        self.refresh = refresh
        self.do_not_save = False

        # HTML 文本，将添加到 UI 中标签后面
        self.comment_before = comment_before

        # HTML 文本，将添加到 UI 中标签前面
        self.comment_after = comment_after

        # 信息文本
        self.infotext = infotext

        # 是否限制 API 访问
        self.restrict_api = restrict_api

    # 添加链接
    def link(self, label, url):
        self.comment_before += f"[<a href='{url}' target='_blank'>{label}</a>]"
        return self

    # 添加 JavaScript 函数
    def js(self, label, js_func):
        self.comment_before += f"[<a onclick='{js_func}(); return false'>{label}</a>]"
        return self

    # 添加信息文本
    def info(self, info):
        self.comment_after += f"<span class='info'>({info})</span>"
        return self

    # 添加 HTML 文本
    def html(self, html):
        self.comment_after += html
        return self

    # 标记需要重新启动
    def needs_restart(self):
        self.comment_after += " <span class='info'>(requires restart)</span>"
        return self

    # 标记需要重新加载 UI
    def needs_reload_ui(self):
        self.comment_after += " <span class='info'>(requires Reload UI)</span>"
        return self

# OptionHTML 类继承自 OptionInfo 类
class OptionHTML(OptionInfo:
    # 初始化方法，接受一个文本参数
    def __init__(self, text):
        # 调用父类的初始化方法，将文本去除首尾空格后作为参数传入
        super().__init__(str(text).strip(), label='', component=lambda **kwargs: gr.HTML(elem_classes="settings-info", **kwargs))
        
        # 设置属性 do_not_save 为 True，表示不保存
        self.do_not_save = True
# 定义一个函数，用于设置选项部分的标识符和选项字典
def options_section(section_identifier, options_dict):
    # 遍历选项字典的值
    for v in options_dict.values():
        # 如果标识符的长度为2，则将该标识符赋给选项的section属性
        if len(section_identifier) == 2:
            v.section = section_identifier
        # 如果标识符的长度为3，则将前两个字符赋给选项的section属性，将第三个字符赋给选项的category_id属性
        elif len(section_identifier) == 3:
            v.section = section_identifier[0:2]
            v.category_id = section_identifier[2]

    # 返回更新后的选项字典
    return options_dict


# 定义一个包含预设字段的选项类
options_builtin_fields = {"data_labels", "data", "restricted_opts", "typemap"}


# 定义选项类
class Options:
    # 预设的类型映射
    typemap = {int: float}

    # 初始化方法，接受数据标签字典和受限选项作为参数
    def __init__(self, data_labels: dict[str, OptionInfo], restricted_opts):
        # 初始化数据标签
        self.data_labels = data_labels
        # 初始化数据，只包含不需要保存的默认值
        self.data = {k: v.default for k, v in self.data_labels.items() if not v.do_not_save}
        # 初始化受限选项
        self.restricted_opts = restricted_opts

    # 设置属性的方法
    def __setattr__(self, key, value):
        # 如果属性是预设字段，则调用父类的设置属性方法
        if key in options_builtin_fields:
            return super(Options, self).__setattr__(key, value)

        # 如果数据不为空
        if self.data is not None:
            # 如果属性在数据或数据标签中
            if key in self.data or key in self.data_labels:
                # 禁止修改设置
                assert not cmd_opts.freeze_settings, "changing settings is disabled"

                # 获取属性信息
                info = self.data_labels.get(key, None)
                # 如果属性不需要保存，则返回
                if info.do_not_save:
                    return

                # 获取组件参数
                comp_args = info.component_args if info else None
                # 如果组件参数是字典且visible为False，则抛出异常
                if isinstance(comp_args, dict) and comp_args.get('visible', True) is False:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                # 如果隐藏UI目录配置且属性在受限选项中，则抛出异常
                if cmd_opts.hide_ui_dir_config and key in self.restricted_opts:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                # 设置属性值
                self.data[key] = value
                return

        # 调用父类的设置属性方法
        return super(Options, self).__setattr__(key, value)
    # 当访问 Options 类中不存在的属性时，会调用该方法
    def __getattr__(self, item):
        # 如果属性在内置字段中，则返回父类的属性
        if item in options_builtin_fields:
            return super(Options, self).__getattribute__(item)

        # 如果数据不为空，且属性在数据中存在，则返回对应的值
        if self.data is not None:
            if item in self.data:
                return self.data[item]

        # 如果属性在数据标签中存在，则返回默认值
        if item in self.data_labels:
            return self.data_labels[item].default

        # 否则返回父类的属性
        return super(Options, self).__getattribute__(item)

    # 设置选项的值，并调用其 onchange 回调函数，返回是否选项已更改
    def set(self, key, value, is_api=False, run_callbacks=True):
        oldval = self.data.get(key, None)
        # 如果新值与旧值相同，则返回 False
        if oldval == value:
            return False

        option = self.data_labels[key]
        # 如果选项设置为不保存，则返回 False
        if option.do_not_save:
            return False

        # 如果是 API 调用且选项限制了 API，则返回 False
        if is_api and option.restrict_api:
            return False

        # 尝试设置属性值，如果出现 RuntimeError 则返回 False
        try:
            setattr(self, key, value)
        except RuntimeError:
            return False

        # 如果需要运行回调函数且 onchange 不为空，则调用 onchange
        if run_callbacks and option.onchange is not None:
            try:
                option.onchange()
            except Exception as e:
                errors.display(e, f"changing setting {key} to {value}")
                setattr(self, key, oldval)
                return False

        return True

    # 返回指定键的默认值
    def get_default(self, key):
        data_label = self.data_labels.get(key)
        if data_label is None:
            return None

        return data_label.default

    # 将数据保存到指定文件中
    def save(self, filename):
        assert not cmd_opts.freeze_settings, "saving settings is disabled"

        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file, indent=4, ensure_ascii=False)

    # 检查两个值是否为相同类型
    def same_type(self, x, y):
        if x is None or y is None:
            return True

        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))

        return type_x == type_y
    # 从指定文件中加载数据，使用 UTF-8 编码
    def load(self, filename):
        # 使用只读模式打开文件，并指定编码为 UTF-8
        with open(filename, "r", encoding="utf8") as file:
            # 将文件内容加载为 JSON 数据，存储在对象的 data 属性中
            self.data = json.load(file)

        # 检查是否需要进行 1.6.0 版本的 VAE 默认设置
        if self.data.get('sd_vae_as_default') is not None and self.data.get('sd_vae_overrides_per_model_preferences') is None:
            # 如果需要设置默认值，则更新相关属性
            self.data['sd_vae_overrides_per_model_preferences'] = not self.data.get('sd_vae_as_default')

        # 检查是否需要进行 1.1.1 版本的 quicksettings 列表迁移
        if self.data.get('quicksettings') is not None and self.data.get('quicksettings_list') is None:
            # 如果需要迁移列表，则将字符串转换为列表并更新相关属性
            self.data['quicksettings_list'] = [i.strip() for i in self.data.get('quicksettings').split(',')]

        # 检查是否需要进行 1.4.0 版本的 ui_reorder 设置
        if isinstance(self.data.get('ui_reorder'), str) and self.data.get('ui_reorder') and "ui_reorder_list" not in self.data:
            # 如果需要设置 ui_reorder_list，则将字符串转换为列表并更新相关属性
            self.data['ui_reorder_list'] = [i.strip() for i in self.data.get('ui_reorder').split(',')]

        # 记录不匹配的设置数量
        bad_settings = 0
        # 遍历所有数据项
        for k, v in self.data.items():
            # 获取数据项的信息
            info = self.data_labels.get(k, None)
            # 检查数据项是否与默认值类型相同，如果不同则输出警告信息
            if info is not None and not self.same_type(info.default, v):
                print(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})", file=sys.stderr)
                bad_settings += 1

        # 如果存在不匹配的设置，则输出警告信息
        if bad_settings > 0:
            print(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.", file=sys.stderr)

    # 设置指定键的 onchange 回调函数
    def onchange(self, key, func, call=True):
        # 获取指定键的信息
        item = self.data_labels.get(key)
        # 设置指定键的 onchange 回调函数
        item.onchange = func

        # 如果需要立即调用回调函数，则执行回调函数
        if call:
            func()
    # 将数据标签中的键值对转换为字典，如果键不存在则使用默认值
    d = {k: self.data.get(k, v.default) for k, v in self.data_labels.items()}
    # 创建一个字典，包含数据标签中每个键对应的注释（如果存在）
    d["_comments_before"] = {k: v.comment_before for k, v in self.data_labels.items() if v.comment_before is not None}
    d["_comments_after"] = {k: v.comment_after for k, v in self.data_labels.items() if v.comment_after is not None}

    # 创建一个空字典用于存储项目类别
    item_categories = {}
    # 遍历数据标签中的每个项目
    for item in self.data_labels.values():
        # 获取项目的类别
        category = categories.mapping.get(item.category_id)
        # 如果类别不存在，则将其设置为"Uncategorized"
        category = "Uncategorized" if category is None else category.label
        # 如果类别不在项目类别字典中，则将其添加进去
        if category not in item_categories:
            item_categories[category] = item.section[1]

    # 创建一个列表，用于存储每个类别对应的设置页面
    d["_categories"] = [[v, k] for k, v in item_categories.items()] + [["Defaults", "Other"]]

    # 将字典转换为 JSON 格式并返回
    return json.dumps(d)

    # 向数据标签中添加新的选项
    def add_option(self, key, info):
        # 将新的键值对添加到数据标签中
        self.data_labels[key] = info
        # 如果键不存在且不是不保存的选项，则将其添加到数据中
        if key not in self.data and not info.do_not_save:
            self.data[key] = info.default
    # 重新排序设置，确保：
    # - 所有与部分相关的项目总是在一起
    # - 所有属于同一类别的部分总是在一起
    # - 类别内的部分按字母顺序排序
    # - 类别按创建顺序排序

    # 类别是部分的超集：对于类别"postprocessing"，可能有多个部分："face restoration"，"upscaling"。

    # 这个函数还会更改项目的 category_id，以确保所有属于同一部分的项目具有相同的 category_id。
    
    # 初始化空字典，用于存储每个 category_id 对应的索引
    category_ids = {}
    # 初始化空字典，用于存储每个部分对应的 category_id
    section_categories = {}

    # 获取所有设置项的键值对
    settings_items = self.data_labels.items()
    # 遍历每个设置项
    for _, item in settings_items:
        # 如果部分不在 section_categories 中，则将部分与对应的 category_id 存入 section_categories
        if item.section not in section_categories:
            section_categories[item.section] = item.category_id

    # 再次遍历每个设置项
    for _, item in settings_items:
        # 将每个设置项的 category_id 设置为其所属部分的 category_id
        item.category_id = section_categories.get(item.section)

    # 遍历所有类别的映射
    for category_id in categories.mapping:
        # 如果 category_id 不在 category_ids 中，则将其添加到 category_ids 中，并赋予一个索引值
        if category_id not in category_ids:
            category_ids[category_id] = len(category_ids)

    # 定义排序函数，根据 category_id 和部分的顺序进行排序
    def sort_key(x):
        # 获取设置项的信息
        item: OptionInfo = x[1]
        # 获取该设置项所属的 category_id 对应的索引值，如果不存在则使用最后一个索引值
        category_order = category_ids.get(item.category_id, len(category_ids))
        # 获取部分的顺序
        section_order = item.section[1]

        # 返回排序依据
        return category_order, section_order

    # 根据排序函数对设置项进行排序，并更新 self.data_labels
    self.data_labels = dict(sorted(settings_items, key=sort_key))
    # 将一个任意值转换为与此设置的值相同类型的值，使用给定的键
    # 例如：cast_value("eta_noise_seed_delta", "12") -> 返回 12（一个整数而不是字符串）
    def cast_value(self, key, value):
        
        # 如果值为 None，则直接返回 None
        if value is None:
            return None

        # 获取默认值
        default_value = self.data_labels[key].default
        # 如果默认值为 None，则尝试从对象属性中获取
        if default_value is None:
            default_value = getattr(self, key, None)
        # 如果仍然为 None，则返回 None
        if default_value is None:
            return None

        # 获取期望的值类型
        expected_type = type(default_value)
        # 如果期望的类型为布尔型且值为字符串 "False"，则将值转换为 False
        if expected_type == bool and value == "False":
            value = False
        # 否则，将值转换为期望的类型
        else:
            value = expected_type(value)

        # 返回转换后的值
        return value
# 使用 dataclass 装饰器定义 OptionsCategory 类，包含 id 和 label 两个属性
@dataclass
class OptionsCategory:
    id: str
    label: str

# 定义 OptionsCategories 类
class OptionsCategories:
    # 初始化方法，创建一个空字典 mapping
    def __init__(self):
        self.mapping = {}

    # 注册一个新的选项类别，传入 category_id 和 label
    def register_category(self, category_id, label):
        # 如果 category_id 已经存在于 mapping 中，则直接返回 category_id
        if category_id in self.mapping:
            return category_id

        # 如果 category_id 不存在于 mapping 中，则将其添加到 mapping 中，值为一个 OptionsCategory 对象
        self.mapping[category_id] = OptionsCategory(category_id, label)

# 创建一个 OptionsCategories 的实例对象 categories
categories = OptionsCategories()
```