# `.\commands\add_new_model_like.py`

```py
# 导入 difflib 模块，用于生成文本差异的比较结果
import difflib
# 导入 json 模块，用于处理 JSON 数据的编解码
import json
# 导入 os 模块，提供了与操作系统交互的功能
import os
# 导入 re 模块，用于支持正则表达式的操作
import re
# 从 argparse 模块中导入 ArgumentParser 和 Namespace 类，用于解析命令行参数
from argparse import ArgumentParser, Namespace
# 从 dataclasses 模块中导入 dataclass 装饰器，用于简化定义数据类
from dataclasses import dataclass
# 从 datetime 模块中导入 date 类，用于处理日期信息
from datetime import date
# 从 itertools 模块中导入 chain 函数，用于将多个迭代器连接在一起
from itertools import chain
# 从 pathlib 模块中导入 Path 类，用于处理文件路径
from pathlib import Path
# 从 typing 模块中导入各种类型提示，用于静态类型检查
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union

# 导入 yaml 模块，用于处理 YAML 格式的数据
import yaml

# 从 ..models 中导入 auto 模块，可能为自动生成的模型模块
from ..models import auto as auto_module
# 从 ..models.auto.configuration_auto 中导入 model_type_to_module_name 函数，用于获取模型类型对应的模块名称
from ..models.auto.configuration_auto import model_type_to_module_name
# 从 ..utils 中导入 is_flax_available、is_tf_available、is_torch_available、logging 函数和类
from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
# 从 当前目录 的 BaseTransformersCLICommand 模块中导入全部内容
from . import BaseTransformersCLICommand

# 使用 logging 模块获取当前模块的 logger 对象，用于记录日志
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 获取当前年份
CURRENT_YEAR = date.today().year
# 获取 Transformers 模块所在的路径
TRANSFORMERS_PATH = Path(__file__).parent.parent
# 获取代码库的根路径
REPO_PATH = TRANSFORMERS_PATH.parent.parent

@dataclass
class ModelPatterns:
    """
    Holds the basic information about a new model for the add-new-model-like command.
    """
    # 这是一个数据类，用于存储用于 add-new-model-like 命令的新模型的基本信息
    # 函数签名，定义了一个函数，用于初始化模型相关的各种参数和选项
    Args:
        model_name (`str`): 模型名称。
        checkpoint (`str`): 用于文档示例的检查点。
        model_type (`str`, *optional*):
            模型类型，内部库中使用的标识符，如 `bert` 或 `xlm-roberta`。默认为 `model_name` 的小写形式，空格用短横线(-)替换。
        model_lower_cased (`str`, *optional*):
            模型名称的小写形式，用于模块名称或函数名称。默认为 `model_name` 的小写形式，空格和短横线都替换为下划线。
        model_camel_cased (`str`, *optional*):
            模型名称的驼峰式命名形式，用于类名。默认为 `model_name` 的驼峰式命名（考虑空格和短横线都作为单词分隔符）。
        model_upper_cased (`str`, *optional*):
            模型名称的大写形式，用于常量名称。默认为 `model_name` 的大写形式，空格和短横线都替换为下划线。
        config_class (`str`, *optional*):
            与此模型关联的配置类。默认为 `"{model_camel_cased}Config"`。
        tokenizer_class (`str`, *optional*):
            与此模型关联的分词器类（对于不使用分词器的模型，请将其保留为 `None`）。
        image_processor_class (`str`, *optional*):
            与此模型关联的图像处理器类（对于不使用图像处理器的模型，请将其保留为 `None`）。
        feature_extractor_class (`str`, *optional*):
            与此模型关联的特征提取器类（对于不使用特征提取器的模型，请将其保留为 `None`）。
        processor_class (`str`, *optional*):
            与此模型关联的处理器类（对于不使用处理器的模型，请将其保留为 `None`）。
    # 在对象初始化完成后执行的方法，用于设置默认属性
    def __post_init__(self):
        # 如果未指定模型类型，则根据模型名称生成一个小写的类型名称
        if self.model_type is None:
            self.model_type = self.model_name.lower().replace(" ", "-")
        
        # 如果未指定小写模型名称，则根据模型名称生成一个小写且用下划线替换空格和破折号的名称
        if self.model_lower_cased is None:
            self.model_lower_cased = self.model_name.lower().replace(" ", "_").replace("-", "_")
        
        # 如果未指定驼峰式模型名称，则按照一定规则生成驼峰式的模型名称
        if self.model_camel_cased is None:
            # 将模型名称按照空格和破折号拆分成单词列表
            words = self.model_name.split(" ")
            words = list(chain(*[w.split("-") for w in words]))
            # 将每个单词的首字母大写，其余字母小写
            words = [w[0].upper() + w[1:] for w in words]
            self.model_camel_cased = "".join(words)
        
        # 如果未指定大写模型名称，则生成一个大写且用下划线替换空格和破折号的名称
        if self.model_upper_cased is None:
            self.model_upper_cased = self.model_name.upper().replace(" ", "_").replace("-", "_")
        
        # 如果未指定配置类名称，则根据驼峰式模型名称生成一个默认的配置类名称
        if self.config_class is None:
            self.config_class = f"{self.model_camel_cased}Config"
ATTRIBUTE_TO_PLACEHOLDER = {
    "config_class": "[CONFIG_CLASS]",  # 属性到占位符的映射字典，用于标记配置类
    "tokenizer_class": "[TOKENIZER_CLASS]",  # 标记标记器类
    "image_processor_class": "[IMAGE_PROCESSOR_CLASS]",  # 标记图像处理器类
    "feature_extractor_class": "[FEATURE_EXTRACTOR_CLASS]",  # 标记特征提取器类
    "processor_class": "[PROCESSOR_CLASS]",  # 标记处理器类
    "checkpoint": "[CHECKPOINT]",  # 标记检查点
    "model_type": "[MODEL_TYPE]",  # 标记模型类型
    "model_upper_cased": "[MODEL_UPPER_CASED]",  # 标记大写模型名称
    "model_camel_cased": "[MODEL_CAMELCASED]",  # 标记驼峰式模型名称
    "model_lower_cased": "[MODEL_LOWER_CASED]",  # 标记小写模型名称
    "model_name": "[MODEL_NAME]",  # 标记模型名称
}


def is_empty_line(line: str) -> bool:
    """
    Determines whether a line is empty or not.
    判断一行是否为空行。
    """
    return len(line) == 0 or line.isspace()


def find_indent(line: str) -> int:
    """
    Returns the number of spaces that start a line indent.
    返回一行开头的空格数，即缩进量。
    """
    search = re.search(r"^(\s*)(?:\S|$)", line)
    if search is None:
        return 0
    return len(search.groups()[0])


def parse_module_content(content: str) -> List[str]:
    """
    Parse the content of a module in the list of objects it defines.

    Args:
        content (`str`): The content to parse
        要解析的模块内容。

    Returns:
        `List[str]`: The list of objects defined in the module.
        返回模块定义的对象列表。
    """
    objects = []
    current_object = []
    lines = content.split("\n")
    end_markers = [")", "]", "}", '"""']  # 结束标记列表

    for line in lines:
        is_valid_object = len(current_object) > 0
        if is_valid_object and len(current_object) == 1:
            is_valid_object = not current_object[0].startswith("# Copied from")
        if not is_empty_line(line) and find_indent(line) == 0 and is_valid_object:
            if line in end_markers:
                current_object.append(line)
                objects.append("\n".join(current_object))
                current_object = []
            else:
                objects.append("\n".join(current_object))
                current_object = [line]
        else:
            current_object.append(line)

    if len(current_object) > 0:
        objects.append("\n".join(current_object))

    return objects


def extract_block(content: str, indent_level: int = 0) -> str:
    """
    Return the first block in `content` with the indent level `indent_level`.

    The first line in `content` should be indented at `indent_level` level, otherwise an error will be thrown.

    This method will immediately stop the search when a (non-empty) line with indent level less than `indent_level` is
    encountered.

    Args:
        content (`str`): The content to parse
        indent_level (`int`, *optional*, default to 0): The indent level of the blocks to search for

    Returns:
        `str`: The first block in `content` with the indent level `indent_level`.
    返回在`content`中具有缩进级别`indent_level`的第一个块。

    Raises:
        ValueError: If the content does not start with the specified indent level.
        如果内容不以指定的缩进级别开头，则引发 ValueError 异常。
    """
    current_object = []
    lines = content.split("\n")
    # 结束标记列表，用于判断对象结尾的可能字符
    end_markers = [")", "]", "}", '"""']

    # 遍历每一行代码
    for idx, line in enumerate(lines):
        # 如果是第一行且缩进级别大于0，且不是空行，并且第一行的缩进级别与指定的缩进级别不符合，则抛出数值错误
        if idx == 0 and indent_level > 0 and not is_empty_line(line) and find_indent(line) != indent_level:
            raise ValueError(
                f"When `indent_level > 0`, the first line in `content` should have indent level {indent_level}. Got "
                f"{find_indent(line)} instead."
            )

        # 如果当前行的缩进级别小于指定的缩进级别，并且不是空行，则退出循环
        if find_indent(line) < indent_level and not is_empty_line(line):
            break

        # 判断是否为对象的结尾
        is_valid_object = len(current_object) > 0
        if (
            not is_empty_line(line)                          # 不是空行
            and not line.endswith(":")                       # 不是以冒号结尾
            and find_indent(line) == indent_level            # 缩进级别与指定的缩进级别相同
            and is_valid_object                              # 当前对象非空
        ):
            # 如果当前行的左边去除空白后在结束标记列表中，则将该行添加到当前对象中
            if line.lstrip() in end_markers:
                current_object.append(line)
            # 返回当前对象的字符串表示形式
            return "\n".join(current_object)
        else:
            # 将当前行添加到当前对象中
            current_object.append(line)

    # 添加最后一个对象
    if len(current_object) > 0:
        return "\n".join(current_object)
def add_content_to_text(
    text: str,
    content: str,
    add_after: Optional[Union[str, Pattern]] = None,
    add_before: Optional[Union[str, Pattern]] = None,
    exact_match: bool = False,
) -> str:
    """
    A utility to add some content inside a given text.

    Args:
       text (`str`): The text in which we want to insert some content.
       content (`str`): The content to add.
       add_after (`str` or `Pattern`):
           The pattern to test on a line of `text`, the new content is added after the first instance matching it.
       add_before (`str` or `Pattern`):
           The pattern to test on a line of `text`, the new content is added before the first instance matching it.
       exact_match (`bool`, *optional*, defaults to `False`):
           A line is considered a match with `add_after` or `add_before` if it matches exactly when `exact_match=True`,
           otherwise, if `add_after`/`add_before` is present in the line.

    <Tip warning={true}>

    The arguments `add_after` and `add_before` are mutually exclusive, and one exactly needs to be provided.

    </Tip>

    Returns:
        `str`: The text with the new content added if a match was found.
    """
    # 检查是否同时提供了 `add_after` 和 `add_before` 参数
    if add_after is None and add_before is None:
        raise ValueError("You need to pass either `add_after` or `add_before`")
    if add_after is not None and add_before is not None:
        raise ValueError("You can't pass both `add_after` or `add_before`")
    
    # 根据参数设置要匹配的模式
    pattern = add_after if add_before is None else add_before

    def this_is_the_line(line):
        # 检查当前行是否符合模式
        if isinstance(pattern, Pattern):
            return pattern.search(line) is not None
        elif exact_match:
            return pattern == line
        else:
            return pattern in line

    new_lines = []
    # 遍历文本的每一行
    for line in text.split("\n"):
        # 如果当前行符合条件
        if this_is_the_line(line):
            # 根据参数决定添加内容的位置
            if add_before is not None:
                new_lines.append(content)
            new_lines.append(line)
            if add_after is not None:
                new_lines.append(content)
        else:
            # 如果不符合条件，直接将当前行添加到新的文本列表中
            new_lines.append(line)

    # 将新的文本列表合并为一个字符串并返回
    return "\n".join(new_lines)


def add_content_to_file(
    file_name: Union[str, os.PathLike],
    content: str,
    add_after: Optional[Union[str, Pattern]] = None,
    add_before: Optional[Union[str, Pattern]] = None,
    exact_match: bool = False,
):
    """
    A utility to add some content inside a given file.
    
    <Tip warning={true}>

    The arguments `add_after` and `add_before` are mutually exclusive, and one exactly needs to be provided.

    </Tip>
    """
    # 打开指定文件以读取其内容，文件名由参数 `file_name` 指定，使用 UTF-8 编码
    with open(file_name, "r", encoding="utf-8") as f:
        # 将文件全部内容读取到 `old_content` 变量中
        old_content = f.read()
    
    # 调用函数 `add_content_to_text`，将 `content` 添加到 `old_content` 中的指定位置
    new_content = add_content_to_text(
        old_content, content, add_after=add_after, add_before=add_before, exact_match=exact_match
    )
    
    # 以写入模式打开文件 `file_name`，使用 UTF-8 编码
    with open(file_name, "w", encoding="utf-8") as f:
        # 将处理过的 `new_content` 写入文件
        f.write(new_content)
def replace_model_patterns(
    text: str, old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns
) -> Tuple[str, str]:
    """
    Replace all patterns present in a given text.

    Args:
        text (`str`): The text to treat.
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.

    Returns:
        `Tuple(str, str)`: A tuple of with the treated text and the replacement actually done in it.
    """
    # 顺序至关重要，因为我们将按照此顺序检查和替换。例如，配置可能包含驼峰命名，但将在之前处理。
    attributes_to_check = ["config_class"]

    # 添加相关的预处理类
    for attr in ["tokenizer_class", "image_processor_class", "feature_extractor_class", "processor_class"]:
        # 如果旧模型和新模型都有这个属性，则添加到检查列表中
        if getattr(old_model_patterns, attr) is not None and getattr(new_model_patterns, attr) is not None:
            attributes_to_check.append(attr)

    # 特殊情况：checkpoint 和 model_type
    if old_model_patterns.checkpoint not in [old_model_patterns.model_type, old_model_patterns.model_lower_cased]:
        attributes_to_check.append("checkpoint")
    if old_model_patterns.model_type != old_model_patterns.model_lower_cased:
        attributes_to_check.append("model_type")
    else:
        # 在文本中用正则表达式替换旧模型类型为占位符"[MODEL_TYPE]"
        text = re.sub(
            rf'(\s*)model_type = "{old_model_patterns.model_type}"',
            r'\1model_type = "[MODEL_TYPE]"',
            text,
        )

    # 特殊情况：当旧模型的大写驼峰名称与小写驼峰名称相同时（例如对于GPT2），但新模型不同时，需要特殊处理
    if old_model_patterns.model_upper_cased == old_model_patterns.model_camel_cased:
        old_model_value = old_model_patterns.model_upper_cased
        # 如果在文本中找到了旧模型大写驼峰名称的特定格式，用新的大写驼峰占位符替换
        if re.search(rf"{old_model_value}_[A-Z_]*[^A-Z_]", text) is not None:
            text = re.sub(rf"{old_model_value}([A-Z_]*)([^a-zA-Z_])", r"[MODEL_UPPER_CASED]\1\2", text)
    else:
        attributes_to_check.append("model_upper_cased")

    # 添加其他需要检查的属性
    attributes_to_check.extend(["model_camel_cased", "model_lower_cased", "model_name"])

    # 替换每个属性为其占位符
    for attr in attributes_to_check:
        text = text.replace(getattr(old_model_patterns, attr), ATTRIBUTE_TO_PLACEHOLDER[attr])

    # 最后，用新值替换占位符
    replacements = []
    for attr, placeholder in ATTRIBUTE_TO_PLACEHOLDER.items():
        # 如果文本中包含占位符，将其替换为新模型对应的属性值
        if placeholder in text:
            replacements.append((getattr(old_model_patterns, attr), getattr(new_model_patterns, attr)))
            text = text.replace(placeholder, getattr(new_model_patterns, attr))

    # 如果存在两个不一致的替换，则不返回任何值（例如：GPT2->GPT_NEW 和 GPT2->GPTNew）
    # 将 replacements 列表中的第一个元素（旧值）抽取出来形成列表 old_replacement_values
    old_replacement_values = [old for old, new in replacements]
    # 检查 old_replacement_values 是否有重复的元素，如果有，则返回原始文本和空字符串
    if len(set(old_replacement_values)) != len(old_replacement_values):
        return text, ""
    
    # 简化 replacements 列表中的元素，并重新赋值给 replacements
    replacements = simplify_replacements(replacements)
    # 将简化后的 replacements 列表转换为形如 "old->new" 的字符串列表
    replacements = [f"{old}->{new}" for old, new in replacements]
    # 返回原始文本和用逗号连接的替换字符串列表
    return text, ",".join(replacements)
# 将给定的替换模式列表简化，确保没有不必要的模式。
def simplify_replacements(replacements):
    # 如果替换列表长度小于等于1，无需简化，直接返回原列表
    if len(replacements) <= 1:
        return replacements

    # 按照替换模式的长度排序，因为较短的模式可能会被较长的模式"隐含"
    replacements.sort(key=lambda x: len(x[0]))

    idx = 0
    # 遍历替换列表中的每一个模式
    while idx < len(replacements):
        old, new = replacements[idx]
        j = idx + 1
        # 再次遍历当前模式之后的所有模式
        while j < len(replacements):
            old_2, new_2 = replacements[j]
            # 如果当前模式可以被之后的模式"隐含"，则移除之后的模式
            if old_2.replace(old, new) == new_2:
                replacements.pop(j)
            else:
                j += 1
        idx += 1

    return replacements


# 返回指定模块文件对应的模块名称
def get_module_from_file(module_file: Union[str, os.PathLike]) -> str:
    full_module_path = Path(module_file).absolute()
    module_parts = full_module_path.with_suffix("").parts

    idx = len(module_parts) - 1
    # 从文件路径的末尾开始查找第一个名为"transformers"的部分
    while idx >= 0 and module_parts[idx] != "transformers":
        idx -= 1
    # 如果未找到"transformers"，抛出数值错误
    if idx < 0:
        raise ValueError(f"{module_file} is not a transformers module.")

    return ".".join(module_parts[idx:])


# 特殊模式映射字典，将特定字符串替换为相应的类别名称
SPECIAL_PATTERNS = {
    "_CHECKPOINT_FOR_DOC =": "checkpoint",
    "_CONFIG_FOR_DOC =": "config_class",
    "_TOKENIZER_FOR_DOC =": "tokenizer_class",
    "_IMAGE_PROCESSOR_FOR_DOC =": "image_processor_class",
    "_FEAT_EXTRACTOR_FOR_DOC =": "feature_extractor_class",
    "_PROCESSOR_FOR_DOC =": "processor_class",
}


# 正则表达式对象，用于匹配类和函数的定义
_re_class_func = re.compile(r"^(?:class|def)\s+([^\s:\(]+)\s*(?:\(|\:)", flags=re.MULTILINE)


# 从对象中移除指定的属性或方法
def remove_attributes(obj, target_attr):
    lines = obj.split(os.linesep)

    target_idx = None
    for idx, line in enumerate(lines):
        # 查找赋值语句
        if line.lstrip().startswith(f"{target_attr} = "):
            target_idx = idx
            break
        # 查找函数或方法的定义
        elif line.lstrip().startswith(f"def {target_attr}("):
            target_idx = idx
            break

    # 如果未找到目标属性或方法，直接返回原始对象
    if target_idx is None:
        return obj

    line = lines[target_idx]
    indent_level = find_indent(line)
    # 前向传递以找到块的结束位置（包括空行）
    parsed = extract_block("\n".join(lines[target_idx:]), indent_level)
    # 计算解析后的文本以换行符分割后的行数
    num_lines = len(parsed.split("\n"))
    # 将目标索引处后面的 num_lines 行设为 None，表示删除这些行
    for idx in range(num_lines):
        lines[target_idx + idx] = None

    # 逆向遍历以找到注释或装饰器的行
    for idx in range(target_idx - 1, -1, -1):
        line = lines[idx]
        # 如果行以 '#' 或 '@' 开头，并且缩进等级与目标相同，则将该行设为 None
        if (line.lstrip().startswith("#") or line.lstrip().startswith("@")) and find_indent(line) == indent_level:
            lines[idx] = None
        else:
            # 如果不满足条件，退出循环
            break

    # 将列表中非 None 的行连接起来，使用操作系统的换行符分隔
    new_obj = os.linesep.join([x for x in lines if x is not None])

    # 返回处理后的新文本对象
    return new_obj
    """
    Create a new module from an existing one and adapting all function and classes names from old patterns to new ones.

    Args:
        module_file (`str` or `os.PathLike`): Path to the module to duplicate.
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        dest_file (`str` or `os.PathLike`, *optional*): Path to the new module.
        add_copied_from (`bool`, *optional*, defaults to `True`):
            Whether or not to add `# Copied from` statements in the duplicated module.
    """
    # If `dest_file` is not provided, generate it based on `module_file` and replace old model pattern with new one
    if dest_file is None:
        dest_file = str(module_file).replace(
            old_model_patterns.model_lower_cased, new_model_patterns.model_lower_cased
        )

    # Open the existing module file for reading
    with open(module_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Update the year in any copyright statements to the current year
    content = re.sub(r"# Copyright (\d+)\s", f"# Copyright {CURRENT_YEAR} ", content)

    # Parse the module content into individual objects (functions, classes, etc.)
    objects = parse_module_content(content)

    # Loop through each object in the module content
    new_objects = []
    for obj in objects:
        # Handle special case for `PRETRAINED_CONFIG_ARCHIVE_MAP` assignment
        if "PRETRAINED_CONFIG_ARCHIVE_MAP = {" in obj:
            # docstyle-ignore
            # Replace with a new entry specific to the new model patterns
            obj = (
                f"{new_model_patterns.model_upper_cased}_PRETRAINED_CONFIG_ARCHIVE_MAP = "
                + "{"
                + f"""
    "{new_model_patterns.checkpoint}": "https://huggingface.co/{new_model_patterns.checkpoint}/resolve/main/config.json",
"""
                + "}\n"
            )
            new_objects.append(obj)
            continue
        # Handle special case for `PRETRAINED_MODEL_ARCHIVE_LIST` assignment
        elif "PRETRAINED_MODEL_ARCHIVE_LIST = [" in obj:
            if obj.startswith("TF_"):
                prefix = "TF_"
            elif obj.startswith("FLAX_"):
                prefix = "FLAX_"
            else:
                prefix = ""
            # docstyle-ignore
            # Replace with a new list including the new model checkpoint and a reference URL
            obj = f"""{prefix}{new_model_patterns.model_upper_cased}_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "{new_model_patterns.checkpoint}",
    # See all {new_model_patterns.model_name} models at https://huggingface.co/models?filter={new_model_patterns.model_type}
]
"""
        # Collect the updated object into the list of new objects
        new_objects.append(obj)
def filter_framework_files(
    files: List[Union[str, os.PathLike]], frameworks: Optional[List[str]] = None
) -> List[Union[str, os.PathLike]]:
    """
    Filter a list of files to only keep the ones corresponding to a list of frameworks.

    Args:
        files (`List[Union[str, os.PathLike]]`): The list of files to filter.
        frameworks (`List[str]`, *optional*): The list of allowed frameworks.

    Returns:
        `List[Union[str, os.PathLike]]`: The list of filtered files.
    """
    # 如果未提供frameworks参数，则使用默认的框架列表
    if frameworks is None:
        frameworks = get_default_frameworks()

    # 创建一个字典来存储每个框架对应的文件
    framework_to_file = {}
    # 创建一个空列表来存储不属于任何框架的文件
    others = []
    # 遍历每个文件
    for f in files:
        # 将文件路径分割成组成文件名的部分
        parts = Path(f).name.split("_")
        # 如果文件名中不包含"modeling"，将其添加到others列表中并跳过
        if "modeling" not in parts:
            others.append(f)
            continue
        # 根据文件名中的关键词判断框架类型，并将文件路径添加到相应框架的条目中
        if "tf" in parts:
            framework_to_file["tf"] = f
        elif "flax" in parts:
            framework_to_file["flax"] = f
        else:
            framework_to_file["pt"] = f

    # 返回符合给定框架列表的文件路径列表，以及不属于任何框架的文件路径列表
    return [framework_to_file[f] for f in frameworks if f in framework_to_file] + others


def get_model_files(model_type: str, frameworks: Optional[List[str]] = None) -> Dict[str, Union[Path, List[Path]]]:
    """
    Retrieves all the files associated to a model.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
        frameworks (`List[str]`, *optional*):
            If passed, will only keep the model files corresponding to the passed frameworks.

    Returns:
        `Dict[str, Union[Path, List[Path]]]`: A dictionary with the following keys:
        - **doc_file** -- The documentation file for the model.
        - **model_files** -- All the files in the model module.
        - **module_name** -- The name of the module corresponding to the model type.
        - **test_files** -- The test files for the model.
    """
    # Convert model type to its corresponding module name
    module_name = model_type_to_module_name(model_type)

    # Define the path to the model module within TRANSFORMERS_PATH
    model_module = TRANSFORMERS_PATH / "models" / module_name
    # List all Python files within the model module
    model_files = list(model_module.glob("*.py"))
    # Filter model files based on specified frameworks, if provided
    model_files = filter_framework_files(model_files, frameworks=frameworks)

    # Define the path to the documentation file for the model
    doc_file = REPO_PATH / "docs" / "source" / "en" / "model_doc" / f"{model_type}.md"

    # Basic pattern for test files related to the model module
    test_files = [
        f"test_modeling_{module_name}.py",
        f"test_modeling_tf_{module_name}.py",
        f"test_modeling_flax_{module_name}.py",
        f"test_tokenization_{module_name}.py",
        f"test_image_processing_{module_name}.py",
        f"test_feature_extraction_{module_name}.py",
        f"test_processor_{module_name}.py",
    ]
    # Filter test files based on specified frameworks, if provided
    test_files = filter_framework_files(test_files, frameworks=frameworks)
    # Construct full paths for test files within the tests/models/module_name directory
    test_files = [REPO_PATH / "tests" / "models" / module_name / f for f in test_files]
    # Filter out non-existing test files
    test_files = [f for f in test_files if f.exists()]

    # Return a dictionary containing paths to relevant files and module name
    return {"doc_file": doc_file, "model_files": model_files, "module_name": module_name, "test_files": test_files}
# 编译正则表达式，用于匹配文档字符串中的_CHECKPOINT_FOR_DOC赋值语句
_re_checkpoint_for_doc = re.compile(r"^_CHECKPOINT_FOR_DOC\s+=\s+(\S*)\s*$", flags=re.MULTILINE)

# 查找给定模型类型的文档字符串中使用的模型检查点
def find_base_model_checkpoint(
    model_type: str, model_files: Optional[Dict[str, Union[Path, List[Path]]]] = None
) -> str:
    """
    Finds the model checkpoint used in the docstrings for a given model.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
        model_files (`Dict[str, Union[Path, List[Path]]`, *optional*):
            The files associated to `model_type`. Can be passed to speed up the function, otherwise will be computed.

    Returns:
        `str`: The checkpoint used.
    """
    # 如果未提供模型文件，调用函数获取模型文件列表
    if model_files is None:
        model_files = get_model_files(model_type)
    
    # 从模型文件列表中获取模型文件
    module_files = model_files["model_files"]
    
    # 遍历模型文件列表
    for fname in module_files:
        # 如果文件名中不包含"modeling"，跳过该文件
        if "modeling" not in str(fname):
            continue
        
        # 打开文件并读取内容
        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
            # 在文件内容中搜索_CHECKPOINT_FOR_DOC赋值语句
            if _re_checkpoint_for_doc.search(content) is not None:
                # 提取检查点值，并移除可能的引号
                checkpoint = _re_checkpoint_for_doc.search(content).groups()[0]
                checkpoint = checkpoint.replace('"', "")
                checkpoint = checkpoint.replace("'", "")
                return checkpoint

    # 如果未找到_CHECKPOINT_FOR_DOC赋值语句，返回空字符串作为默认值
    # TODO: 如果所有的模型文件中都找不到_CHECKPOINT_FOR_DOC，可能需要一些备用方案
    return ""


# 返回当前环境中安装的默认框架列表（如PyTorch、TensorFlow、Flax）
def get_default_frameworks():
    """
    Returns the list of frameworks (PyTorch, TensorFlow, Flax) that are installed in the environment.
    """
    frameworks = []
    if is_torch_available():  # 如果PyTorch可用，将"pt"添加到框架列表中
        frameworks.append("pt")
    if is_tf_available():  # 如果TensorFlow可用，将"tf"添加到框架列表中
        frameworks.append("tf")
    if is_flax_available():  # 如果Flax可用，将"flax"添加到框架列表中
        frameworks.append("flax")
    return frameworks


# 编译正则表达式，用于匹配模型名称映射中的模型类名
_re_model_mapping = re.compile("MODEL_([A-Z_]*)MAPPING_NAMES")

# 根据给定的模型类型和框架列表，检索相关的模型类
def retrieve_model_classes(model_type: str, frameworks: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Retrieve the model classes associated to a given model.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
        frameworks (`List[str]`, *optional*):
            The frameworks to look for. Will default to `["pt", "tf", "flax"]`, passing a smaller list will restrict
            the classes returned.

    Returns:
        `Dict[str, List[str]]`: A dictionary with one key per framework and the list of model classes associated to
        that framework as values.
    """
    # 如果未提供框架列表，使用默认框架列表
    if frameworks is None:
        frameworks = get_default_frameworks()
    
    # 定义模块字典，包含每种框架对应的模型自动加载模块
    modules = {
        "pt": auto_module.modeling_auto if is_torch_available() else None,
        "tf": auto_module.modeling_tf_auto if is_tf_available() else None,
        "flax": auto_module.modeling_flax_auto if is_flax_available() else None,
    }
    
    # 初始化模型类字典
    model_classes = {}
    # 遍历给定的框架列表
    for framework in frameworks:
        # 初始化一个空列表来存放新的模型类
        new_model_classes = []
        # 检查当前框架是否已安装模块，若未安装则抛出数值错误异常
        if modules[framework] is None:
            raise ValueError(f"You selected {framework} in the frameworks, but it is not installed.")
        # 获取当前框架模块中所有包含模型映射的属性名列表
        model_mappings = [attr for attr in dir(modules[framework]) if _re_model_mapping.search(attr) is not None]
        # 遍历当前框架的模型映射名列表
        for model_mapping_name in model_mappings:
            # 根据模型映射名获取对应的模型映射对象
            model_mapping = getattr(modules[framework], model_mapping_name)
            # 检查给定的模型类型是否在当前模型映射中
            if model_type in model_mapping:
                # 将符合条件的模型类添加到新模型类列表中
                new_model_classes.append(model_mapping[model_type])

        # 如果新模型类列表不为空
        if len(new_model_classes) > 0:
            # 去除重复的模型类，并将结果存入模型类字典中
            model_classes[framework] = list(set(new_model_classes))

    # 返回最终的模型类字典
    return model_classes
    """
    Retrieves all the information from a given model_type.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
        frameworks (`List[str]`, *optional*):
            If passed, will only keep the info corresponding to the passed frameworks.

    Returns:
        `Dict`: A dictionary with the following keys:
        - **frameworks** (`List[str]`): The list of frameworks that back this model type.
        - **model_classes** (`Dict[str, List[str]]`): The model classes implemented for that model type.
        - **model_files** (`Dict[str, Union[Path, List[Path]]]`): The files associated with that model type.
        - **model_patterns** (`ModelPatterns`): The various patterns for the model.
    """
    # Check if the provided model_type exists in the MODEL_NAMES_MAPPING
    if model_type not in auto_module.MODEL_NAMES_MAPPING:
        raise ValueError(f"{model_type} is not a valid model type.")

    # Retrieve the actual model name from the mapping
    model_name = auto_module.MODEL_NAMES_MAPPING[model_type]

    # Retrieve the configuration class name for the given model type
    config_class = auto_module.configuration_auto.CONFIG_MAPPING_NAMES[model_type]

    # Retrieve the archive map if available for the given model type
    archive_map = auto_module.configuration_auto.CONFIG_ARCHIVE_MAP_MAPPING_NAMES.get(model_type, None)

    # Retrieve the tokenizer classes if available for the given model type
    if model_type in auto_module.tokenization_auto.TOKENIZER_MAPPING_NAMES:
        tokenizer_classes = auto_module.tokenization_auto.TOKENIZER_MAPPING_NAMES[model_type]
        tokenizer_class = tokenizer_classes[0] if tokenizer_classes[0] is not None else tokenizer_classes[1]
    else:
        tokenizer_class = None

    # Retrieve the image processor class if available for the given model type
    image_processor_class = auto_module.image_processing_auto.IMAGE_PROCESSOR_MAPPING_NAMES.get(model_type, None)

    # Retrieve the feature extractor class if available for the given model type
    feature_extractor_class = auto_module.feature_extraction_auto.FEATURE_EXTRACTOR_MAPPING_NAMES.get(model_type, None)

    # Retrieve the processor class if available for the given model type
    processor_class = auto_module.processing_auto.PROCESSOR_MAPPING_NAMES.get(model_type, None)

    # Retrieve the files associated with the given model type
    model_files = get_model_files(model_type, frameworks=frameworks)

    # Create a camel-cased version of the config class name without "Config"
    model_camel_cased = config_class.replace("Config", "")

    # Initialize an empty list to store available frameworks
    available_frameworks = []

    # Iterate through the model files and identify the frameworks they belong to
    for fname in model_files["model_files"]:
        if "modeling_tf" in str(fname):
            available_frameworks.append("tf")
        elif "modeling_flax" in str(fname):
            available_frameworks.append("flax")
        elif "modeling" in str(fname):
            available_frameworks.append("pt")

    # If frameworks parameter is None, retrieve default frameworks
    if frameworks is None:
        frameworks = get_default_frameworks()

    # Filter frameworks to include only those available in the model files
    frameworks = [f for f in frameworks if f in available_frameworks]

    # Retrieve model classes based on the model type and selected frameworks
    model_classes = retrieve_model_classes(model_type, frameworks=frameworks)

    # Retrieve model upper-cased name from the constant name of the pretrained archive map, if available
    if archive_map is None:
        model_upper_cased = model_camel_cased.upper()
    # 如果archive_map不包含"PRETRAINED"，则按下面的逻辑处理
    else:
        # 使用下划线分割archive_map，并初始化索引
        parts = archive_map.split("_")
        idx = 0
        # 循环直到找到"PRETRAINED"或者到达末尾
        while idx < len(parts) and parts[idx] != "PRETRAINED":
            idx += 1
        # 如果找到了"PRETRAINED"
        if idx < len(parts):
            # 将"PRETRAINED"之前的部分连接起来作为model_upper_cased
            model_upper_cased = "_".join(parts[:idx])
        else:
            # 如果没有找到"PRETRAINED"，则使用model_camel_cased的大写形式作为model_upper_cased
            model_upper_cased = model_camel_cased.upper()

    # 创建一个ModelPatterns对象，用于存储模型相关的配置和信息
    model_patterns = ModelPatterns(
        model_name,
        # 调用函数find_base_model_checkpoint找到模型的基础检查点
        checkpoint=find_base_model_checkpoint(model_type, model_files=model_files),
        model_type=model_type,
        model_camel_cased=model_camel_cased,
        model_lower_cased=model_files["module_name"],
        model_upper_cased=model_upper_cased,
        config_class=config_class,
        tokenizer_class=tokenizer_class,
        image_processor_class=image_processor_class,
        feature_extractor_class=feature_extractor_class,
        processor_class=processor_class,
    )

    # 返回一个包含各种模型相关信息的字典
    return {
        "frameworks": frameworks,
        "model_classes": model_classes,
        "model_files": model_files,
        "model_patterns": model_patterns,
    }
    # 打开指定路径的初始化文件以供处理
    with open(init_file, "r", encoding="utf-8") as f:
        # 读取整个文件内容
        content = f.read()

    # 将文件内容按行分割成列表
    lines = content.split("\n")
    # 初始化一个空列表，用于存储处理后的新行
    new_lines = []
    # 初始化索引变量，用于迭代处理每一行
    idx = 0
    # 循环处理每一行代码，直到处理完所有行
    while idx < len(lines):
        # 在 try-except-else 块中处理条件导入
        if (re_conditional_imports.search(lines[idx]) is not None) and (re_try.search(lines[idx - 1]) is not None):
            # 移除前面的 `try:` 语句
            new_lines.pop()
            idx += 1
            # 找到下一个 `else:` 之前的空行或者非空行
            while is_empty_line(lines[idx]) or re_else.search(lines[idx]) is None:
                idx += 1
            idx += 1
            # 确定缩进级别
            indent = find_indent(lines[idx])
            # 继续处理直到缩进小于当前缩进或者是空行
            while find_indent(lines[idx]) >= indent or is_empty_line(lines[idx]):
                idx += 1
        # 移除来自 utils 的导入
        elif re_is_xxx_available.search(lines[idx]) is not None:
            line = lines[idx]
            # 替换需要移除的 framework 导入
            for framework in to_remove:
                line = line.replace(f", is_{framework}_available", "")
                line = line.replace(f"is_{framework}_available, ", "")
                line = line.replace(f"is_{framework}_available,", "")
                line = line.replace(f"is_{framework}_available", "")

            # 如果替换后的行不为空，则添加到新行列表中
            if len(line.strip()) > 0:
                new_lines.append(line)
            idx += 1
        # 否则保留该行，除非是关于 tokenizer 导入且不需要保留的情况
        elif keep_processing or (
            re.search(r'^\s*"(tokenization|processing|feature_extraction|image_processing)', lines[idx]) is None
            and re.search(r"^\s*from .(tokenization|processing|feature_extraction|image_processing)", lines[idx])
            is None
        ):
            new_lines.append(lines[idx])
            idx += 1
        else:
            idx += 1

    # 将处理后的新行写入到指定的初始化文件中
    with open(init_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
# 打开 Transformers 库的 __init__.py 文件以进行读取操作，使用 UTF-8 编码
with open(TRANSFORMERS_PATH / "__init__.py", "r", encoding="utf-8") as f:
    # 读取整个文件内容并存储在变量 content 中
    content = f.read()

# 将文件内容按行分割成列表 lines
lines = content.split("\n")
# 初始化索引变量 idx 为 0
idx = 0
# 初始化空列表 new_lines，用于存储处理后的新行内容
new_lines = []
# 初始化 framework 变量为 None，用于存储框架名称
framework = None
    # 当前行号小于文本行数时，继续循环处理文本行
    while idx < len(lines):
        # 新的框架标志置为 False
        new_framework = False
        # 如果当前行不是空行且缩进为0，则将框架设置为 None
        if not is_empty_line(lines[idx]) and find_indent(lines[idx]) == 0:
            framework = None
        # 如果当前行左侧去除空白后以特定字符串开头，则确定框架类型为 "pt"，并设置新框架标志为 True
        elif lines[idx].lstrip().startswith("if not is_torch_available"):
            framework = "pt"
            new_framework = True
        # 如果当前行左侧去除空白后以特定字符串开头，则确定框架类型为 "tf"，并设置新框架标志为 True
        elif lines[idx].lstrip().startswith("if not is_tf_available"):
            framework = "tf"
            new_framework = True
        # 如果当前行左侧去除空白后以特定字符串开头，则确定框架类型为 "flax"，并设置新框架标志为 True
        elif lines[idx].lstrip().startswith("if not is_flax_available"):
            framework = "flax"
            new_framework = True
    
        # 如果是新框架，则需要跳过直到 else: 块以找到导入位置
        if new_framework:
            while lines[idx].strip() != "else:":
                new_lines.append(lines[idx])
                idx += 1
    
        # 如果框架不是所需的框架且框架列表不为空且当前框架不在列表中，则跳过当前行
        if framework is not None and frameworks is not None and framework not in frameworks:
            new_lines.append(lines[idx])
            idx += 1
        # 如果当前行包含旧模型模式的模型引用，则收集整个代码块
        elif re.search(rf'models.{old_model_patterns.model_lower_cased}( |")', lines[idx]) is not None:
            block = [lines[idx]]
            indent = find_indent(lines[idx])
            idx += 1
            # 收集整个缩进块
            while find_indent(lines[idx]) > indent:
                block.append(lines[idx])
                idx += 1
            # 如果当前行的内容是特定列表中的一员，则也添加到块中
            if lines[idx].strip() in [")", "]", "],"]:
                block.append(lines[idx])
                idx += 1
            block = "\n".join(block)
            new_lines.append(block)
    
            add_block = True
            # 如果不需要处理，则只保留非空的处理类
            if not with_processing:
                processing_classes = [
                    old_model_patterns.tokenizer_class,
                    old_model_patterns.image_processor_class,
                    old_model_patterns.feature_extractor_class,
                    old_model_patterns.processor_class,
                ]
                processing_classes = [c for c in processing_classes if c is not None]
                # 遍历处理类列表，将其从块中移除
                for processing_class in processing_classes:
                    block = block.replace(f' "{processing_class}",', "")
                    block = block.replace(f', "{processing_class}"', "")
                    block = block.replace(f" {processing_class},", "")
                    block = block.replace(f", {processing_class}", "")
                    # 如果块中仍包含处理类，则不添加此块
                    if processing_class in block:
                        add_block = False
            # 如果需要添加块，则将替换后的模型模式块添加到新行列表中
            if add_block:
                new_lines.append(replace_model_patterns(block, old_model_patterns, new_model_patterns)[0])
        else:
            # 否则，将当前行直接添加到新行列表中
            new_lines.append(lines[idx])
            idx += 1
    
    # 将新行列表写入到 "__init__.py" 文件中
    with open(TRANSFORMERS_PATH / "__init__.py", "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
# 将模型的标记器添加到自动模块的相关映射中
def insert_tokenizer_in_auto_module(old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns):
    """
    Add a tokenizer to the relevant mappings in the auto module.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
    """
    # 如果旧模型或新模型的标记器类为None，则返回
    if old_model_patterns.tokenizer_class is None or new_model_patterns.tokenizer_class is None:
        return

    # 打开自动模块中的tokenization_auto.py文件，以utf-8编码读取其内容
    with open(TRANSFORMERS_PATH / "models" / "auto" / "tokenization_auto.py", "r", encoding="utf-8") as f:
        content = f.read()

    # 将文件内容按行分割为列表
    lines = content.split("\n")
    idx = 0
    # 首先定位到TOKENIZER_MAPPING_NAMES块
    while not lines[idx].startswith("    TOKENIZER_MAPPING_NAMES = OrderedDict("):
        idx += 1
    idx += 1

    # 定位到TOKENIZER_MAPPING块的结尾
    while not lines[idx].startswith("TOKENIZER_MAPPING = _LazyAutoMapping"):
        # 如果tokenizer块在一行上定义，则以"),结束"
        if lines[idx].endswith(","):
            block = lines[idx]
        # 否则，tokenizer块跨多行，直到找到"),结束"
        else:
            block = []
            while not lines[idx].startswith("            ),"):
                block.append(lines[idx])
                idx += 1
            block = "\n".join(block)
        idx += 1

        # 如果在该块中找到了旧模型类型和标记器类，则找到了旧模型的tokenizer块
        if f'"{old_model_patterns.model_type}"' in block and old_model_patterns.tokenizer_class in block:
            break

    # 将旧模型类型和标记器类替换为新模型类型和标记器类
    new_block = block.replace(old_model_patterns.model_type, new_model_patterns.model_type)
    new_block = new_block.replace(old_model_patterns.tokenizer_class, new_model_patterns.tokenizer_class)

    # 构建新的文件内容行列表，包括更新后的tokenizer块
    new_lines = lines[:idx] + [new_block] + lines[idx:]

    # 将更新后的文件内容写回tokenization_auto.py文件
    with open(TRANSFORMERS_PATH / "models" / "auto" / "tokenization_auto.py", "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))


AUTO_CLASSES_PATTERNS = {
    "configuration_auto.py": [
        '        ("{model_type}", "{model_name}"),',
        '        ("{model_type}", "{config_class}"),',
        '        ("{model_type}", "{pretrained_archive_map}"),',
    ],
    "feature_extraction_auto.py": ['        ("{model_type}", "{feature_extractor_class}"),'],
    "image_processing_auto.py": ['        ("{model_type}", "{image_processor_class}"),'],
    "modeling_auto.py": ['        ("{model_type}", "{any_pt_class}"),'],
    "modeling_tf_auto.py": ['        ("{model_type}", "{any_tf_class}"),'],
    "modeling_flax_auto.py": ['        ("{model_type}", "{any_flax_class}"),'],
    "processing_auto.py": ['        ("{model_type}", "{processor_class}"),'],
}


def add_model_to_auto_classes(
    old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns, model_classes: Dict[str, List[str]]
):
    """
    Add a model to the relevant mappings in the auto module.
    
    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        model_classes (`Dict[str, List[str]]`): A dictionary mapping auto module filenames to lists of model class names.
    """
    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        model_classes (`Dict[str, List[str]]`): A dictionary framework to list of model classes implemented.
    """
    # 调用函数将旧模型模式中的所有分词器插入到新模型模式的自动模块中
    insert_tokenizer_in_auto_module(old_model_patterns, new_model_patterns)
# 模板文档字符串，用于生成新模型的概述性文档
DOC_OVERVIEW_TEMPLATE = """## Overview

The {model_name} model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).

"""


def duplicate_doc_file(
    doc_file: Union[str, os.PathLike],
    old_model_patterns: ModelPatterns,
    new_model_patterns: ModelPatterns,
    dest_file: Optional[Union[str, os.PathLike]] = None,
    frameworks: Optional[List[str]] = None,
):
    """
    Duplicate a documentation file and adapts it for a new model.

    Args:
        module_file (`str` or `os.PathLike`): Path to the doc file to duplicate.
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        dest_file (`str` or `os.PathLike`, *optional*): Path to the new doc file.
            Will default to the a file named `{new_model_patterns.model_type}.md` in the same folder as `module_file`.
        frameworks (`List[str]`, *optional*):
            If passed, will only keep the model classes corresponding to this list of frameworks in the new doc file.
    """
    # 读取原始文档文件内容
    with open(doc_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 更新版权信息为当前年份
    content = re.sub(r"<!--\s*Copyright (\d+)\s", f"<!--Copyright {CURRENT_YEAR} ", content)
    
    # 如果未提供特定框架列表，则使用默认框架列表
    if frameworks is None:
        frameworks = get_default_frameworks()
    
    # 如果未提供目标文件路径，则默认为与原文档文件同目录下，新模型类型命名的文件
    if dest_file is None:
        dest_file = Path(doc_file).parent / f"{new_model_patterns.model_type}.md"

    # 解析文档内容为块。每个块对应一个部分/标题
    lines = content.split("\n")
    blocks = []
    current_block = []

    for line in lines:
        if line.startswith("#"):
            blocks.append("\n".join(current_block))
            current_block = [line]
        else:
            current_block.append(line)
    blocks.append("\n".join(current_block))

    new_blocks = []
    in_classes = False
    # 遍历输入的文本块列表
    for block in blocks:
        # 检查是否以版权声明开始，如果不是则添加到新的文本块列表中
        if not block.startswith("#"):
            new_blocks.append(block)
        # 检查是否为主标题，如果是则替换为新模型名称的标题
        elif re.search(r"^#\s+\S+", block) is not None:
            new_blocks.append(f"# {new_model_patterns.model_name}\n")
        # 检查是否进入类定义部分，根据旧模型配置类来确定
        elif not in_classes and old_model_patterns.config_class in block.split("\n")[0]:
            # 标记已进入类定义部分，并添加文档概述模板及替换后的模型配置块
            in_classes = True
            new_blocks.append(DOC_OVERVIEW_TEMPLATE.format(model_name=new_model_patterns.model_name))
            new_block, _ = replace_model_patterns(block, old_model_patterns, new_model_patterns)
            new_blocks.append(new_block)
        # 处理在类定义部分的情况
        elif in_classes:
            in_classes = True
            # 获取当前文本块的标题，并提取类名
            block_title = block.split("\n")[0]
            block_class = re.search(r"^#+\s+(\S.*)$", block_title).groups()[0]
            new_block, _ = replace_model_patterns(block, old_model_patterns, new_model_patterns)

            # 根据类名条件性地添加新的文本块
            if "Tokenizer" in block_class:
                # 仅在需要时添加标记器类
                if old_model_patterns.tokenizer_class != new_model_patterns.tokenizer_class:
                    new_blocks.append(new_block)
            elif "ImageProcessor" in block_class:
                # 仅在需要时添加图像处理器类
                if old_model_patterns.image_processor_class != new_model_patterns.image_processor_class:
                    new_blocks.append(new_block)
            elif "FeatureExtractor" in block_class:
                # 仅在需要时添加特征提取器类
                if old_model_patterns.feature_extractor_class != new_model_patterns.feature_extractor_class:
                    new_blocks.append(new_block)
            elif "Processor" in block_class:
                # 仅在需要时添加处理器类
                if old_model_patterns.processor_class != new_model_patterns.processor_class:
                    new_blocks.append(new_block)
            elif block_class.startswith("Flax"):
                # 仅在所选框架中包含 Flax 模型时添加
                if "flax" in frameworks:
                    new_blocks.append(new_block)
            elif block_class.startswith("TF"):
                # 仅在所选框架中包含 TF 模型时添加
                if "tf" in frameworks:
                    new_blocks.append(new_block)
            elif len(block_class.split(" ")) == 1:
                # 仅在所选框架中包含 PyTorch 模型时添加
                if "pt" in frameworks:
                    new_blocks.append(new_block)
            else:
                new_blocks.append(new_block)

    # 将新的文本块列表写入目标文件
    with open(dest_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_blocks))
# 在文档目录中插入新模型的条目，与旧模型在同一部分。
def insert_model_in_doc_toc(old_model_patterns, new_model_patterns):
    """
    Insert the new model in the doc TOC, in the same section as the old model.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
    """
    # 指定文档目录文件路径
    toc_file = REPO_PATH / "docs" / "source" / "en" / "_toctree.yml"
    # 打开并加载 YAML 格式的目录文件内容
    with open(toc_file, "r", encoding="utf8") as f:
        content = yaml.safe_load(f)

    # 定位到 API 文档的索引
    api_idx = 0
    while content[api_idx]["title"] != "API":
        api_idx += 1
    # 获取 API 文档下的各个部分
    api_doc = content[api_idx]["sections"]

    # 定位到 Models 部分的索引
    model_idx = 0
    while api_doc[model_idx]["title"] != "Models":
        model_idx += 1
    # 获取 Models 部分下的各个小节
    model_doc = api_doc[model_idx]["sections"]

    # 在目录中查找基础模型的位置
    old_model_type = old_model_patterns.model_type
    section_idx = 0
    while section_idx < len(model_doc):
        # 获取当前小节中的本地目录项列表
        sections = [entry["local"] for entry in model_doc[section_idx]["sections"]]
        # 如果旧模型的目录项在当前小节中，则跳出循环
        if f"model_doc/{old_model_type}" in sections:
            break
        section_idx += 1

    # 如果未找到旧模型的目录项，则输出警告信息并返回
    if section_idx == len(model_doc):
        old_model = old_model_patterns.model_name
        new_model = new_model_patterns.model_name
        print(f"Did not find {old_model} in the table of content, so you will need to add {new_model} manually.")
        return

    # 准备新模型的目录项信息
    toc_entry = {"local": f"model_doc/{new_model_patterns.model_type}", "title": new_model_patterns.model_name}
    # 将新模型的目录项添加到找到的旧模型所在的小节中
    model_doc[section_idx]["sections"].append(toc_entry)
    # 根据标题排序小节中的目录项
    model_doc[section_idx]["sections"] = sorted(model_doc[section_idx]["sections"], key=lambda s: s["title"].lower())
    # 更新 API 文档中的 Models 部分
    api_doc[model_idx]["sections"] = model_doc
    # 更新整体内容中的 API 文档
    content[api_idx]["sections"] = api_doc

    # 将更新后的内容重新写入目录文件
    with open(toc_file, "w", encoding="utf-8") as f:
        f.write(yaml.dump(content, allow_unicode=True))
    # 获取给定模型类型的相关信息，包括模型文件、模型模式等
    model_info = retrieve_info_for_model(model_type, frameworks=frameworks)
    
    # 从模型信息中获取模型文件列表和旧模型模式
    model_files = model_info["model_files"]
    old_model_patterns = model_info["model_patterns"]
    
    # 如果有提供旧的检查点，则更新旧模型模式的检查点属性
    if old_checkpoint is not None:
        old_model_patterns.checkpoint = old_checkpoint
    
    # 检查旧模型模式的检查点属性是否为空，如果是则引发 ValueError
    if len(old_model_patterns.checkpoint) == 0:
        raise ValueError(
            "The old model checkpoint could not be recovered from the model type. Please pass it to the "
            "`old_checkpoint` argument."
        )
    
    # 初始化保持旧处理方式的标志为 True
    keep_old_processing = True
    
    # 检查特定处理属性（如图像处理类、特征提取器类、处理器类、分词器类）是否与新模型模式相同，若有不同则将标志设为 False
    for processing_attr in ["image_processor_class", "feature_extractor_class", "processor_class", "tokenizer_class"]:
        if getattr(old_model_patterns, processing_attr) != getattr(new_model_patterns, processing_attr):
            keep_old_processing = False
    
    # 从模型信息中获取模型类别
    model_classes = model_info["model_classes"]
    
    # 1. 创建新模型的模块
    old_module_name = model_files["module_name"]
    module_folder = TRANSFORMERS_PATH / "models" / new_model_patterns.model_lower_cased
    
    # 确保模块文件夹存在，如果不存在则创建
    os.makedirs(module_folder, exist_ok=True)
    
    # 根据保持旧处理方式的标志筛选要适应的文件列表
    files_to_adapt = model_files["model_files"]
    if keep_old_processing:
        files_to_adapt = [
            f
            for f in files_to_adapt
            if "tokenization" not in str(f)
            and "processing" not in str(f)
            and "feature_extraction" not in str(f)
            and "image_processing" not in str(f)
        ]
    
    # 再次确保模块文件夹存在，如果不存在则创建
    os.makedirs(module_folder, exist_ok=True)
    
    # 遍历要适应的文件列表，生成新的模块文件名并复制到目标位置
    for module_file in files_to_adapt:
        new_module_name = module_file.name.replace(
            old_model_patterns.model_lower_cased, new_model_patterns.model_lower_cased
        )
        dest_file = module_folder / new_module_name
        duplicate_module(
            module_file,
            old_model_patterns,
            new_model_patterns,
            dest_file=dest_file,
            add_copied_from=add_copied_from and "modeling" in new_module_name,
        )
    
    # 清理模块的初始化文件，根据保持旧处理方式的标志更新处理类别
    clean_frameworks_in_init(
        module_folder / "__init__.py", frameworks=frameworks, keep_processing=not keep_old_processing
    )
    
    # 2. 将新模型添加到模型包的初始化文件和主初始化文件中
    add_content_to_file(
        TRANSFORMERS_PATH / "models" / "__init__.py",
        f"    {new_model_patterns.model_lower_cased},",
        add_after=f"    {old_module_name},",
        exact_match=True,
    )
    add_model_to_main_init(
        old_model_patterns, new_model_patterns, frameworks=frameworks, with_processing=not keep_old_processing
    )
    
    # 3. 添加测试文件
    files_to_adapt = model_files["test_files"]
    if keep_old_processing:
        files_to_adapt = [
            f
            for f in files_to_adapt
            if "tokenization" not in str(f)
            and "processor" not in str(f)
            and "feature_extraction" not in str(f)
            and "image_processing" not in str(f)
        ]
    # 定义一个函数，用于禁用与指定文件相关的特定功能测试
    def disable_fx_test(filename: Path) -> bool:
        # 打开文件并读取其内容
        with open(filename) as fp:
            content = fp.read()
        # 使用正则表达式替换文件内容中的特定文本
        new_content = re.sub(r"fx_compatible\s*=\s*True", "fx_compatible = False", content)
        # 将修改后的内容写回到文件中
        with open(filename, "w") as fp:
            fp.write(new_content)
        # 返回值指示是否有内容被修改过
        return content != new_content

    # 初始化一个标志，用于追踪是否禁用了功能测试
    disabled_fx_test = False

    # 创建测试文件夹，如果不存在则创建
    tests_folder = REPO_PATH / "tests" / "models" / new_model_patterns.model_lower_cased
    os.makedirs(tests_folder, exist_ok=True)

    # 创建一个空的 __init__.py 文件
    with open(tests_folder / "__init__.py", "w"):
        pass

    # 遍历需要调整的文件列表
    for test_file in files_to_adapt:
        # 根据模式替换文件名中的旧模型名称为新模型名称
        new_test_file_name = test_file.name.replace(
            old_model_patterns.model_lower_cased, new_model_patterns.model_lower_cased
        )
        # 构建目标文件的路径
        dest_file = test_file.parent.parent / new_model_patterns.model_lower_cased / new_test_file_name
        # 复制指定的测试文件到目标位置，并禁用功能测试
        duplicate_module(
            test_file,
            old_model_patterns,
            new_model_patterns,
            dest_file=dest_file,
            add_copied_from=False,
            attrs_to_remove=["pipeline_model_mapping", "is_pipeline_test_to_skip"],
        )
        # 更新功能测试禁用状态
        disabled_fx_test = disabled_fx_test | disable_fx_test(dest_file)

    # 如果有功能测试被禁用，则输出提示信息
    if disabled_fx_test:
        print(
            "The tests for symbolic tracing with torch.fx were disabled, you can add those once symbolic tracing works"
            " for your new model."
        )

    # 将新模型添加到自动类中
    add_model_to_auto_classes(old_model_patterns, new_model_patterns, model_classes)

    # 添加文档文件
    doc_file = REPO_PATH / "docs" / "source" / "en" / "model_doc" / f"{old_model_patterns.model_type}.md"
    duplicate_doc_file(doc_file, old_model_patterns, new_model_patterns, frameworks=frameworks)
    # 在文档目录中插入新模型
    insert_model_in_doc_toc(old_model_patterns, new_model_patterns)

    # 如果旧模型类型与其检查点名称相同，输出警告信息
    if old_model_patterns.model_type == old_model_patterns.checkpoint:
        print(
            "The model you picked has the same name for the model type and the checkpoint name "
            f"({old_model_patterns.model_type}). As a result, it's possible some places where the new checkpoint "
            f"should be, you have {new_model_patterns.model_type} instead. You should search for all instances of "
            f"{new_model_patterns.model_type} in the new files and check they're not badly used as checkpoints."
        )
    # 如果旧模型名称（小写形式）与其检查点名称相同，输出警告信息
    elif old_model_patterns.model_lower_cased == old_model_patterns.checkpoint:
        print(
            "The model you picked has the same name for the model type and the checkpoint name "
            f"({old_model_patterns.model_lower_cased}). As a result, it's possible some places where the new "
            f"checkpoint should be, you have {new_model_patterns.model_lower_cased} instead. You should search for "
            f"all instances of {new_model_patterns.model_lower_cased} in the new files and check they're not badly "
            "used as checkpoints."
        )
    # 检查旧模型模式的类型是否为小写，并且新模型模式的类型不是小写时
    if (
        old_model_patterns.model_type == old_model_patterns.model_lower_cased
        and new_model_patterns.model_type != new_model_patterns.model_lower_cased
    ):
        # 输出警告信息，说明选择的模型类型和小写模型名称相同，可能导致新模型类型在某些地方被误用为小写模型名称
        print(
            "The model you picked has the same name for the model type and the lowercased model name "
            f"({old_model_patterns.model_lower_cased}). As a result, it's possible some places where the new "
            f"model type should be, you have {new_model_patterns.model_lower_cased} instead. You should search for "
            f"all instances of {new_model_patterns.model_lower_cased} in the new files and check they're not badly "
            "used as the model type."
        )

    # 如果不保留旧的处理逻辑并且旧模型模式的分词器类不为空时
    if not keep_old_processing and old_model_patterns.tokenizer_class is not None:
        # 输出提示信息，指出需要手动修复新分词器文件开头的常量问题。如果新模型有一个快速分词器，还需手动将转换器添加到 `convert_slow_tokenizer.py` 的 `SLOW_TO_FAST_CONVERTERS` 常量中
        print(
            "The constants at the start of the new tokenizer file created needs to be manually fixed. If your new "
            "model has a tokenizer fast, you will also need to manually add the converter in the "
            "`SLOW_TO_FAST_CONVERTERS` constant of `convert_slow_tokenizer.py`."
        )
def add_new_model_like_command_factory(args: Namespace):
    # 创建并返回一个 AddNewModelLikeCommand 对象，使用参数中的配置文件和仓库路径
    return AddNewModelLikeCommand(config_file=args.config_file, path_to_repo=args.path_to_repo)


class AddNewModelLikeCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        # 注册子命令 "add-new-model-like" 到指定的 ArgumentParser 对象
        add_new_model_like_parser = parser.add_parser("add-new-model-like")
        add_new_model_like_parser.add_argument(
            "--config_file", type=str, help="A file with all the information for this model creation."
        )
        add_new_model_like_parser.add_argument(
            "--path_to_repo", type=str, help="When not using an editable install, the path to the Transformers repo."
        )
        # 设置默认的函数处理程序为 add_new_model_like_command_factory 函数
        add_new_model_like_parser.set_defaults(func=add_new_model_like_command_factory)

    def __init__(self, config_file=None, path_to_repo=None, *args):
        if config_file is not None:
            # 如果配置文件不为 None，从配置文件中加载配置信息
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            # 初始化对象的各个属性
            self.old_model_type = config["old_model_type"]
            self.model_patterns = ModelPatterns(**config["new_model_patterns"])
            self.add_copied_from = config.get("add_copied_from", True)
            self.frameworks = config.get("frameworks", get_default_frameworks())
            self.old_checkpoint = config.get("old_checkpoint", None)
        else:
            # 如果配置文件为 None，调用 get_user_input() 函数获取用户输入的属性值
            (
                self.old_model_type,
                self.model_patterns,
                self.add_copied_from,
                self.frameworks,
                self.old_checkpoint,
            ) = get_user_input()

        self.path_to_repo = path_to_repo

    def run(self):
        if self.path_to_repo is not None:
            # 如果仓库路径不为 None，则设定全局变量 TRANSFORMERS_PATH 和 REPO_PATH
            global TRANSFORMERS_PATH
            global REPO_PATH

            REPO_PATH = Path(self.path_to_repo)
            TRANSFORMERS_PATH = REPO_PATH / "src" / "transformers"

        # 调用 create_new_model_like 函数创建新模型
        create_new_model_like(
            model_type=self.old_model_type,
            new_model_patterns=self.model_patterns,
            add_copied_from=self.add_copied_from,
            frameworks=self.frameworks,
            old_checkpoint=self.old_checkpoint,
        )


def get_user_field(
    question: str,
    default_value: Optional[str] = None,
    is_valid_answer: Optional[Callable] = None,
    convert_to: Optional[Callable] = None,
    fallback_message: Optional[str] = None,
) -> Any:
    """
    A utility function that asks a question to the user to get an answer, potentially looping until it gets a valid
    answer.
    """
    # 简单的用户输入获取函数，带有一些可选的参数和验证功能
    # 如果问题字符串不以空格结尾，添加一个空格
    if not question.endswith(" "):
        question = question + " "
    # 如果提供了默认值，将默认值添加到问题的末尾
    if default_value is not None:
        question = f"{question} [{default_value}] "

    # 初始化有效答案为 False，用于循环直到得到有效答案
    valid_answer = False
    while not valid_answer:
        # 提示用户输入问题，并获取用户输入的答案
        answer = input(question)

        # 如果提供了默认值且用户未输入任何内容，则使用默认值
        if default_value is not None and len(answer) == 0:
            answer = default_value

        # 如果提供了自定义的答案验证函数 is_valid_answer
        if is_valid_answer is not None:
            valid_answer = is_valid_answer(answer)
        # 如果提供了转换函数 convert_to
        elif convert_to is not None:
            try:
                # 尝试将答案转换为指定类型
                answer = convert_to(answer)
                valid_answer = True
            except Exception:
                # 如果转换失败，则标记答案为无效，继续循环
                valid_answer = False
        else:
            # 如果没有提供 is_valid_answer 或 convert_to，直接标记答案为有效
            valid_answer = True

        # 如果答案无效，则打印回退消息
        if not valid_answer:
            print(fallback_message)

    # 返回经过验证和可能转换的答案
    return answer
# 将字符串转换为布尔值
def convert_to_bool(x: str) -> bool:
    """
    Converts a string to a bool.
    """
    # 检查字符串是否在可接受的真值列表中，返回对应的布尔值
    if x.lower() in ["1", "y", "yes", "true"]:
        return True
    # 检查字符串是否在可接受的假值列表中，返回对应的布尔值
    if x.lower() in ["0", "n", "no", "false"]:
        return False
    # 如果字符串既不是真值也不是假值，抛出 ValueError 异常
    raise ValueError(f"{x} is not a value that can be converted to a bool.")


# 获取用户输入以添加新模型
def get_user_input():
    """
    Ask the user for the necessary inputs to add the new model.
    """
    # 获取模型类型列表
    model_types = list(auto_module.configuration_auto.MODEL_NAMES_MAPPING.keys())

    # 获取旧模型类型
    valid_model_type = False
    while not valid_model_type:
        # 提示用户输入要复制的模型类型
        old_model_type = input(
            "What is the model you would like to duplicate? Please provide the lowercase `model_type` (e.g. roberta): "
        )
        # 检查用户输入是否在模型类型列表中
        if old_model_type in model_types:
            valid_model_type = True
        else:
            # 如果输入不在列表中，提示用户并尝试提供建议
            print(f"{old_model_type} is not a valid model type.")
            near_choices = difflib.get_close_matches(old_model_type, model_types)
            if len(near_choices) >= 1:
                if len(near_choices) > 1:
                    near_choices = " or ".join(near_choices)
                print(f"Did you mean {near_choices}?")

    # 获取旧模型的详细信息
    old_model_info = retrieve_info_for_model(old_model_type)
    old_tokenizer_class = old_model_info["model_patterns"].tokenizer_class
    old_image_processor_class = old_model_info["model_patterns"].image_processor_class
    old_feature_extractor_class = old_model_info["model_patterns"].feature_extractor_class
    old_processor_class = old_model_info["model_patterns"].processor_class
    old_frameworks = old_model_info["frameworks"]

    # 如果旧模型没有检查点信息，要求用户输入基础检查点的名称
    old_checkpoint = None
    if len(old_model_info["model_patterns"].checkpoint) == 0:
        old_checkpoint = get_user_field(
            "We couldn't find the name of the base checkpoint for that model, please enter it here."
        )

    # 获取新模型的名称
    model_name = get_user_field(
        "What is the name (with no special casing) for your new model in the paper (e.g. RoBERTa)? "
    )
    # 创建默认模型模式对象
    default_patterns = ModelPatterns(model_name, model_name)

    # 获取用户输入的模型标识符
    model_type = get_user_field(
        "What identifier would you like to use for the `model_type` of this model? ",
        default_value=default_patterns.model_type,
    )
    # 获取用户输入的模型模块名（小写）
    model_lower_cased = get_user_field(
        "What lowercase name would you like to use for the module (folder) of this model? ",
        default_value=default_patterns.model_lower_cased,
    )
    # 获取用户输入的模型类的前缀（驼峰命名）
    model_camel_cased = get_user_field(
        "What prefix (camel-cased) would you like to use for the model classes of this model (e.g. Roberta)? ",
        default_value=default_patterns.model_camel_cased,
    )
    # 获取用户输入的模型常量的前缀（大写）
    model_upper_cased = get_user_field(
        "What prefix (upper-cased) would you like to use for the constants relative to this model? ",
        default_value=default_patterns.model_upper_cased,
    )
    # 获取用户输入的配置类名称
    config_class = get_user_field(
        "What will be the name of the config class for this model? ", default_value=f"{model_camel_cased}Config"
    )
    )
    # 调用 get_user_field 函数获取用户输入，用于指定新模型的检查点标识符
    checkpoint = get_user_field(
        "Please give a checkpoint identifier (on the model Hub) for this new model (e.g. facebook/FacebookAI/roberta-base): "
    )

    # 创建旧处理类列表，仅包含非空元素
    old_processing_classes = [
        c
        for c in [old_image_processor_class, old_feature_extractor_class, old_tokenizer_class, old_processor_class]
        if c is not None
    ]
    # 将列表转换为逗号分隔的字符串
    old_processing_classes = ", ".join(old_processing_classes)
    # 获取用户输入，确认新模型是否使用与旧模型相同的处理类
    keep_processing = get_user_field(
        f"Will your new model use the same processing class as {old_model_type} ({old_processing_classes}) (yes/no)? ",
        convert_to=convert_to_bool,
        fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
    )
    # 根据用户的选择，确定新模型的处理类
    if keep_processing:
        image_processor_class = old_image_processor_class
        feature_extractor_class = old_feature_extractor_class
        processor_class = old_processor_class
        tokenizer_class = old_tokenizer_class
    else:
        # 如果不使用与旧模型相同的处理类，则根据需要获取各种处理类的新名称
        if old_tokenizer_class is not None:
            tokenizer_class = get_user_field(
                "What will be the name of the tokenizer class for this model? ",
                default_value=f"{model_camel_cased}Tokenizer",
            )
        else:
            tokenizer_class = None
        if old_image_processor_class is not None:
            image_processor_class = get_user_field(
                "What will be the name of the image processor class for this model? ",
                default_value=f"{model_camel_cased}ImageProcessor",
            )
        else:
            image_processor_class = None
        if old_feature_extractor_class is not None:
            feature_extractor_class = get_user_field(
                "What will be the name of the feature extractor class for this model? ",
                default_value=f"{model_camel_cased}FeatureExtractor",
            )
        else:
            feature_extractor_class = None
        if old_processor_class is not None:
            processor_class = get_user_field(
                "What will be the name of the processor class for this model? ",
                default_value=f"{model_camel_cased}Processor",
            )
        else:
            processor_class = None

    # 创建 ModelPatterns 对象，用于保存新模型的相关属性
    model_patterns = ModelPatterns(
        model_name,
        checkpoint,
        model_type=model_type,
        model_lower_cased=model_lower_cased,
        model_camel_cased=model_camel_cased,
        model_upper_cased=model_upper_cased,
        config_class=config_class,
        tokenizer_class=tokenizer_class,
        image_processor_class=image_processor_class,
        feature_extractor_class=feature_extractor_class,
        processor_class=processor_class,
    )

    # 获取用户输入，确定在创建新建模型文件时是否添加 # Copied from 注释
    add_copied_from = get_user_field(
        "Should we add # Copied from statements when creating the new modeling file (yes/no)? ",
        convert_to=convert_to_bool,
        default_value="yes",
        fallback_message="Please answer yes/no, y/n, true/false or 1/0.",
    )
    # 调用函数获取用户字段，询问是否在所有旧模型类型的框架中添加新模型的版本
    # 用户字段包括确认消息、类型转换函数、默认值和回退消息
    all_frameworks = get_user_field(
        "Should we add a version of your new model in all the frameworks implemented by"
        f" {old_model_type} ({old_frameworks}) (yes/no)? ",
        convert_to=convert_to_bool,  # 将用户输入转换为布尔类型的函数
        default_value="yes",  # 默认值为 "yes"
        fallback_message="Please answer yes/no, y/n, true/false or 1/0.",  # 如果用户输入不合法时的提示消息
    )
    
    # 如果用户选择在所有框架中添加新模型版本
    if all_frameworks:
        frameworks = None  # 框架列表设为 None
    else:
        # 否则，获取用户字段，请求用户输入要使用的框架列表
        frameworks = get_user_field(
            "Please enter the list of framworks you want (pt, tf, flax) separated by spaces",
            # 检查用户输入是否有效，要求所有输入项必须是 ["pt", "tf", "flax"] 中的一种
            is_valid_answer=lambda x: all(p in ["pt", "tf", "flax"] for p in x.split(" ")),
        )
        frameworks = list(set(frameworks.split(" ")))  # 将输入的框架列表转换为集合去重后再转为列表
    
    # 返回元组包含旧模型类型、模型模式、复制来源、框架列表和旧的检查点
    return (old_model_type, model_patterns, add_copied_from, frameworks, old_checkpoint)
```