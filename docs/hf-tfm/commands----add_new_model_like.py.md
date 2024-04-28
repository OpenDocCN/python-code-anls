# `.\transformers\commands\add_new_model_like.py`

```
# 导入模块和库
import difflib  # 用于比较序列之间差异的库
import json  # 用于处理 JSON 格式数据的库
import os  # 用于与操作系统进行交互的库
import re  # 用于正则表达式操作的库
from argparse import ArgumentParser, Namespace  # 用于解析命令行参数的库
from dataclasses import dataclass  # 用于创建数据类的库
from datetime import date  # 用于处理日期的库
from itertools import chain  # 用于迭代工具的库
from pathlib import Path  # 用于操作路径的库
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union  # 用于类型提示的库

import yaml  # 用于处理 YAML 格式数据的库

# 从上层目录中的 models 模块导入 auto 子模块
from ..models import auto as auto_module
# 从上层目录中的 models.auto.configuration_auto 模块导入 model_type_to_module_name 函数
from ..models.auto.configuration_auto import model_type_to_module_name
# 导入 logging 模块
from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
# 从当前目录中的 __init__ 模块导入 BaseTransformersCLICommand 类
from . import BaseTransformersCLICommand

# 获取 logger 对象
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 当前年份
CURRENT_YEAR = date.today().year
# transformers 目录路径
TRANSFORMERS_PATH = Path(__file__).parent.parent
# 代码库路径
REPO_PATH = TRANSFORMERS_PATH.parent.parent

# 定义数据类 ModelPatterns，用于保存新模型的基本信息
@dataclass
class ModelPatterns:
    """
    Holds the basic information about a new model for the add-new-model-like command.
    """
    Args:
        model_name (`str`): The model name.  # 模型名称
        checkpoint (`str`): The checkpoint to use for doc examples.  # 用于文档示例的检查点
        model_type (`str`, *optional*):
            The model type, the identifier used internally in the library like `bert` or `xlm-roberta`. Will default to
            `model_name` lowercased with spaces replaced with minuses (-).  # 模型类型，库内部使用的标识符，如 `bert` 或 `xlm-roberta`。默认为将 `model_name` 转换为小写，并用连字符 (-) 替换空格
        model_lower_cased (`str`, *optional*):
            The lowercased version of the model name, to use for the module name or function names. Will default to
            `model_name` lowercased with spaces and minuses replaced with underscores.  # 模型名称的小写版本，用于模块名称或函数名称。默认为将 `model_name` 转换为小写，并用下划线替换空格和连字符
        model_camel_cased (`str`, *optional*):
            The camel-cased version of the model name, to use for the class names. Will default to `model_name`
            camel-cased (with spaces and minuses both considered as word separators.  # 模型名称的驼峰命名版本，用于类名。默认为将 `model_name` 转换为驼峰命名（空格和连字符都被视为单词分隔符）
        model_upper_cased (`str`, *optional*):
            The uppercased version of the model name, to use for the constant names. Will default to `model_name`
            uppercased with spaces and minuses replaced with underscores.  # 模型名称的大写版本，用于常量名称。默认为将 `model_name` 转换为大写，并用下划线替换空格和连字符
        config_class (`str`, *optional*):
            The tokenizer class associated with this model. Will default to `"{model_camel_cased}Config"`.  # 与此模型关联的分词器类。默认为 `"{model_camel_cased}Config"`
        tokenizer_class (`str`, *optional*):
            The tokenizer class associated with this model (leave to `None` for models that don't use a tokenizer).  # 与此模型关联的分词器类（对于不使用分词器的模型，请将其保留为 `None`）
        image_processor_class (`str`, *optional*):
            The image processor class associated with this model (leave to `None` for models that don't use an image
            processor).  # 与此模型关联的图像处理器类（对于不使用图像处理器的模型，请将其保留为 `None`）
        feature_extractor_class (`str`, *optional*):
            The feature extractor class associated with this model (leave to `None` for models that don't use a feature
            extractor).  # 与此模型关联的特��提取器类（对于不使用特征提取器的模型，请将其保留为 `None`）
        processor_class (`str`, *optional*):
            The processor class associated with this model (leave to `None` for models that don't use a processor).  # 与此模型关联的处理器类（对于不使用处理器的模型，请将其保留为 `None`）
    """

    model_name: str  # 模型名称
    checkpoint: str  # 检查点
    model_type: Optional[str] = None  # 模型类型，默认为 None
    model_lower_cased: Optional[str] = None  # 模型名称的小写版本，默认为 None
    model_camel_cased: Optional[str] = None  # 模型名称的驼峰命名版本，默认为 None
    model_upper_cased: Optional[str] = None  # 模型名称的大写版本，默认为 None
    config_class: Optional[str] = None  # 与模型关联的配置类，默认为 None
    tokenizer_class: Optional[str] = None  # 与模型关联的分词器类，默认为 None
    image_processor_class: Optional[str] = None  # 与模型关联的图像处理器类，默认为 None
    feature_extractor_class: Optional[str] = None  # 与模型关联的特征提取器类，默认为 None
    processor_class: Optional[str] = None  # 与模型关联的处理器类，默认为 None
    # 在初始化对象后，如果模型类型为空，则将模型名称转换为小写并替换空格为破折号
    def __post_init__(self):
        if self.model_type is None:
            self.model_type = self.model_name.lower().replace(" ", "-")
        # 如果模型小写形式为空，则将模型名称转换为小写并替换空格为下划线和破折号为下划线
        if self.model_lower_cased is None:
            self.model_lower_cased = self.model_name.lower().replace(" ", "_").replace("-", "_")
        # 如果模型驼峰形式为空，则将模型名称按空格和破折号拆分成单词，首字母大写
        if self.model_camel_cased is None:
            # 按空格和破折号拆分模型名称
            words = self.model_name.split(" ")
            words = list(chain(*[w.split("-") for w in words]))
            # 确保每个单词首字母大写
            words = [w[0].upper() + w[1:] for w in words]
            self.model_camel_cased = "".join(words)
        # 如果模型大写形式为空，则将模型名称转换为大写并替换空格为下划线和破折号为下划线
        if self.model_upper_cased is None:
            self.model_upper_cased = self.model_name.upper().replace(" ", "_").replace("-", "_")
        # 如果配置类为空，则使用模型驼峰形式加上"Config"作为配置类名称
        if self.config_class is None:
            self.config_class = f"{self.model_camel_cased}Config"
# 将属性名映射到占位符的字典
ATTRIBUTE_TO_PLACEHOLDER = {
    "config_class": "[CONFIG_CLASS]",
    "tokenizer_class": "[TOKENIZER_CLASS]",
    "image_processor_class": "[IMAGE_PROCESSOR_CLASS]",
    "feature_extractor_class": "[FEATURE_EXTRACTOR_CLASS]",
    "processor_class": "[PROCESSOR_CLASS]",
    "checkpoint": "[CHECKPOINT]",
    "model_type": "[MODEL_TYPE]",
    "model_upper_cased": "[MODEL_UPPER_CASED]",
    "model_camel_cased": "[MODEL_CAMELCASED]",
    "model_lower_cased": "[MODEL_LOWER_CASED]",
    "model_name": "[MODEL_NAME]",
}

# 判断一行是否为空行
def is_empty_line(line: str) -> bool:
    """
    Determines whether a line is empty or not.
    """
    return len(line) == 0 or line.isspace()

# 查找行的缩进级别
def find_indent(line: str) -> int:
    """
    Returns the number of spaces that start a line indent.
    """
    search = re.search(r"^(\s*)(?:\S|$)", line)
    if search is None:
        return 0
    return len(search.groups()[0])

# 解析模块内容，返回模块中定义的对象列表
def parse_module_content(content: str) -> List[str]:
    """
    Parse the content of a module in the list of objects it defines.

    Args:
        content (`str`): The content to parse

    Returns:
        `List[str]`: The list of objects defined in the module.
    """
    objects = []  # 保存模块中定义的对象
    current_object = []  # 当前正在解析的对象
    lines = content.split("\n")  # 将内容按行拆分
    # Doc-styler 可以在文档字符串中的三重引号之间接受所有内容，所以我们需要一个假的 """ 来匹配。
    end_markers = [")", "]", "}", '"""']  # 结束标记列表

    for line in lines:
        # 如果当前正在解析的对象非空且长度为1，则需要检查该行是否以 "# Copied from" 开头，如果是，则无效
        is_valid_object = len(current_object) > 0
        if is_valid_object and len(current_object) == 1:
            is_valid_object = not current_object[0].startswith("# Copied from")
        # 如果不是空行且缩进级别为0且对象有效，则当前行是对象的结束标记或新对象的开始
        if not is_empty_line(line) and find_indent(line) == 0 and is_valid_object:
            # 结束标记应该包含在当前对象中
            if line in end_markers:
                current_object.append(line)
                objects.append("\n".join(current_object))  # 将当前对象添加到对象列表中
                current_object = []  # 重置当前对象
            else:
                objects.append("\n".join(current_object))  # 将当前对象添加到对象列表中
                current_object = [line]  # 创建新对象
        else:
            current_object.append(line)  # 继续添加当前行到当前对象中

    # 添加最后一个对象
    if len(current_object) > 0:
        objects.append("\n".join(current_object))

    return objects

# 提取内容中的第一个缩进级别为指定值的代码块
def extract_block(content: str, indent_level: int = 0) -> str:
    """Return the first block in `content` with the indent level `indent_level`.

    The first line in `content` should be indented at `indent_level` level, otherwise an error will be thrown.

    This method will immediately stop the search when a (non-empty) line with indent level less than `indent_level` is
    encountered.

    Args:
        content (`str`): The content to parse
        indent_level (`int`, *optional*, default to 0): The indent level of the blocks to search for

    Returns:
        `str`: The first block in `content` with the indent level `indent_level`.
    """
    current_object = []  # 当前正在解析的代码块
    lines = content.split("\n")  # 将内容按行拆分
    # 定义可能表示对象结束的标记
    end_markers = [")", "]", "}", '"""']

    # 遍历文本行
    for idx, line in enumerate(lines):
        # 检查第一行是否符合预期的缩进级别
        if idx == 0 and indent_level > 0 and not is_empty_line(line) and find_indent(line) != indent_level:
            # 抛出数值错误异常
            raise ValueError(
                f"When `indent_level > 0`, the first line in `content` should have indent level {indent_level}. Got "
                f"{find_indent(line)} instead."
            )

        # 如果当前行的缩进级别小于指定级别且不为空行，则跳出循环
        if find_indent(line) < indent_level and not is_empty_line(line):
            break

        # 判断是否为对象结束
        is_valid_object = len(current_object) > 0
        if (
            not is_empty_line(line)
            and not line.endswith(":")
            and find_indent(line) == indent_level
            and is_valid_object
        ):
            # 如果当前行为对象结束标记，则将其添加到当前对象中并返回
            if line.lstrip() in end_markers:
                current_object.append(line)
            return "\n".join(current_object)
        else:
            current_object.append(line)

    # 添加最后一个对象
    if len(current_object) > 0:
        return "\n".join(current_object)
# 向给定文本中添加内容的实用工具函数

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

    # 如果既没有提供 `add_after` 也没有提供 `add_before`，则抛出 ValueError
    if add_after is None and add_before is None:
        raise ValueError("You need to pass either `add_after` or `add_before`")
    
    # 如果同时提供了 `add_after` 和 `add_before`，则抛出 ValueError
    if add_after is not None and add_before is not None:
        raise ValueError("You can't pass both `add_after` or `add_before`")

    # 根据提供的参数选择合适的模式
    pattern = add_after if add_before is None else add_before

    # 检查是否匹配目标行
    def this_is_the_line(line):
        if isinstance(pattern, Pattern):
            return pattern.search(line) is not None
        elif exact_match:
            return pattern == line
        else:
            return pattern in line

    new_lines = []

    # 遍历文本的每一行
    for line in text.split("\n"):
        # 如果当前行匹配目标模式
        if this_is_the_line(line):
            # 如果需要在目标行之前添加内容，则先添加内容
            if add_before is not None:
                new_lines.append(content)
            # 添加目标行
            new_lines.append(line)
            # 如果需要在目标行之后添加内容，则添加内容
            if add_after is not None:
                new_lines.append(content)
        else:
            # 如果当前行不是目标行，则直接添加到新文本中
            new_lines.append(line)

    # 将新文本行拼接为一个字符串并返回
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
```  
    # 定义一个函数，用于向文件中插入内容
    Args:
       file_name (`str` or `os.PathLike`): 要插入内容的文件名
       content (`str`): 要添加的内容
       add_after (`str` or `Pattern`): 在`text`的一行上测试的模式，新内容将添加在第一个匹配的实例之后
       add_before (`str` or `Pattern`): 在`text`的一行上测试的模式，新内容将添加在第一个匹配的实例之前
       exact_match (`bool`, *optional*, 默认为`False`): 当`exact_match=True`时，如果一行与`add_after`或`add_before`完全匹配，则将其视为匹配，否则，如果一行中存在`add_after`/`add_before`，则将其视为匹配

    <Tip warning={true}>

    参数`add_after`和`add_before`是互斥的，必须提供其中一个。

    </Tip>
    """
    # 以只读模式打开文件
    with open(file_name, "r", encoding="utf-8") as f:
        # 读取文件的旧内容
        old_content = f.read()

    # 调用函数将新内容添加到旧内容中
    new_content = add_content_to_text(
        old_content, content, add_after=add_after, add_before=add_before, exact_match=exact_match
    )

    # 以写入模式打开文件
    with open(file_name, "w", encoding="utf-8") as f:
        # 将新内容写入文件
        f.write(new_content)
# 定义一个函数，用于替换给定文本中的所有模式。
def replace_model_patterns(
    text: str, old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns
) -> Tuple[str, str]:
    """
    Replace all patterns present in a given text.

    Args:
        text (`str`): The text to treat. 待处理的文本。
        old_model_patterns (`ModelPatterns`): The patterns for the old model. 旧模型的模式。
        new_model_patterns (`ModelPatterns`): The patterns for the new model. 新模型的模式。

    Returns:
        `Tuple(str, str)`: A tuple of with the treated text and the replacement actually done in it. 返回一个元组，包含处理后的文本以及实际进行替换的内容。
    """
    # 检查和替换的顺序至关重要。例如，配置可能包含驼峰命名，但会先进行处理。
    attributes_to_check = ["config_class"]
    # 添加相关的预处理类
    for attr in ["tokenizer_class", "image_processor_class", "feature_extractor_class", "processor_class"]:
        if getattr(old_model_patterns, attr) is not None and getattr(new_model_patterns, attr) is not None:
            attributes_to_check.append(attr)

    # 特殊情况的处理：检查点和模型类型
    if old_model_patterns.checkpoint not in [old_model_patterns.model_type, old_model_patterns.model_lower_cased]:
        attributes_to_check.append("checkpoint")
    if old_model_patterns.model_type != old_model_patterns.model_lower_cased:
        attributes_to_check.append("model_type")
    else:
        # 使用正则表达式替换模型类型为占位符
        text = re.sub(
            rf'(\s*)model_type = "{old_model_patterns.model_type}"',
            r'\1model_type = "[MODEL_TYPE]"',
            text,
        )

    # 特殊情况的处理：当旧模型的模型驼峰命名和大写模型名称相同时（例如，对于GPT2），但新模型不同时。
    if old_model_patterns.model_upper_cased == old_model_patterns.model_camel_cased:
        old_model_value = old_model_patterns.model_upper_cased
        # 如果在文本中找到模型大写名称后接非大写字母或下划线的字符，则需要使用特殊的正则表达式替换
        if re.search(rf"{old_model_value}_[A-Z_]*[^A-Z_]", text) is not None:
            text = re.sub(rf"{old_model_value}([A-Z_]*)([^a-zA-Z_])", r"[MODEL_UPPER_CASED]\1\2", text)
    else:
        attributes_to_check.append("model_upper_cased")

    attributes_to_check.extend(["model_camel_cased", "model_lower_cased", "model_name"])

    # 现在让我们用占位符替换每个其他属性
    for attr in attributes_to_check:
        text = text.replace(getattr(old_model_patterns, attr), ATTRIBUTE_TO_PLACEHOLDER[attr])

    # 最后，我们可以用新值替换占位符。
    replacements = []
    for attr, placeholder in ATTRIBUTE_TO_PLACEHOLDER.items():
        if placeholder in text:
            replacements.append((getattr(old_model_patterns, attr), getattr(new_model_patterns, attr)))
            text = text.replace(placeholder, getattr(new_model_patterns, attr))

    # 如果我们有两个不一致的替换，我们不返回任何内容（例如：GPT2->GPT_NEW 和 GPT2->GPTNew）
    # 提取所有旧值，形成列表
    old_replacement_values = [old for old, new in replacements]
    # 如果旧值列表中存在重复值，返回原文本和空字符串
    if len(set(old_replacement_values)) != len(old_replacement_values):
        return text, ""

    # 简化替换规则
    replacements = simplify_replacements(replacements)
    # 构建替换规则字符串列表，格式为"旧值->新值"
    replacements = [f"{old}->{new}" for old, new in replacements]
    # 返回原文本和用逗号连接的替换规则字符串
    return text, ",".join(replacements)
# 简化替换模式列表，确保没有不必要的模式
def simplify_replacements(replacements):
    # 如果替换列表长度小于等于1，则无需简化
    if len(replacements) <= 1:
        return replacements

    # 按照替换模式的长度排序，因为只有长度较短的替换模式才可能"暗示"另一个替换模式
    replacements.sort(key=lambda x: len(x[0]))

    idx = 0
    while idx < len(replacements):
        old, new = replacements[idx]
        # 遍历当前替换模式之后的所有替换模式
        j = idx + 1
        while j < len(replacements):
            old_2, new_2 = replacements[j]
            # 如果当前替换模式暗示了另一个替换模式，则可以删除后者
            if old_2.replace(old, new) == new_2:
                replacements.pop(j)
            else:
                j += 1
        idx += 1

    return replacements


# 根据模块文件返回对应的模块名称
def get_module_from_file(module_file: Union[str, os.PathLike]) -> str:
    full_module_path = Path(module_file).absolute()
    module_parts = full_module_path.with_suffix("").parts

    # 从末尾开始查找第一个名为transformers的部分
    idx = len(module_parts) - 1
    while idx >= 0 and module_parts[idx] != "transformers":
        idx -= 1
    if idx < 0:
        raise ValueError(f"{module_file} is not a transformers module.")

    return ".".join(module_parts[idx:])


# 特殊模式字典，用于指定特殊模式的替换
SPECIAL_PATTERNS = {
    "_CHECKPOINT_FOR_DOC =": "checkpoint",
    "_CONFIG_FOR_DOC =": "config_class",
    "_TOKENIZER_FOR_DOC =": "tokenizer_class",
    "_IMAGE_PROCESSOR_FOR_DOC =": "image_processor_class",
    "_FEAT_EXTRACTOR_FOR_DOC =": "feature_extractor_class",
    "_PROCESSOR_FOR_DOC =": "processor_class",
}


# 编译正则表达式，用于匹配类和函数定义
_re_class_func = re.compile(r"^(?:class|def)\s+([^\s:\(]+)\s*(?:\(|\:)", flags=re.MULTILINE)


# 从对象中移除指定属性
def remove_attributes(obj, target_attr):
    lines = obj.split(os.linesep)

    target_idx = None
    for idx, line in enumerate(lines):
        # 查找赋值语句
        if line.lstrip().startswith(f"{target_attr} = "):
            target_idx = idx
            break
        # 查找函数/方法定义
        elif line.lstrip().startswith(f"def {target_attr}("):
            target_idx = idx
            break

    # 如果未找到目标属性，则返回原对象
    if target_idx is None:
        return obj

    line = lines[target_idx]
    indent_level = find_indent(line)
    # 向前查找块的结束位置（包括空行）
    parsed = extract_block("\n".join(lines[target_idx:]), indent_level)
    # 计算字符串 `parsed` 中包含的行数
    num_lines = len(parsed.split("\n"))
    # 将目标行以及之后 `num_lines` 行的内容设置为 None
    for idx in range(num_lines):
        lines[target_idx + idx] = None

    # 向后遍历以查找注释或装饰器
    for idx in range(target_idx - 1, -1, -1):
        # 获取当前行内容
        line = lines[idx]
        # 如果当前行以 '#' 或 '@' 开头，并且缩进水平与目标行相同，则将该行设置为 None
        if (line.lstrip().startswith("#") or line.lstrip().startswith("@")) and find_indent(line) == indent_level:
            lines[idx] = None
        else:
            # 如果不满足上述条件，则退出循环
            break

    # 使用操作系统的行分隔符连接非 None 的行，形成新的字符串对象
    new_obj = os.linesep.join([x for x in lines if x is not None])

    # 返回新字符串对象
    return new_obj
            if obj.startswith("TF_"):  # 判断对象是否以"TF_"开头
                prefix = "TF_"  # 设置模型类型前缀为"TF_"
            elif obj.startswith("FLAX_"):  # 判断对象是否以"FLAX_"开头
                prefix = "FLAX_"  # 设置模型类型前缀为"FLAX_"
            else:  # 如果以上条件都不满足
                prefix = ""  # 设置模型类型前缀为空字符串
            # docstyle-ignore
            obj = f"""{prefix}{new_model_patterns.model_upper_cased}_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 使用新模型名称创建新对象
    "{new_model_patterns.checkpoint}",
    # See all {new_model_patterns.model_name} models at https://huggingface.co/models?filter={new_model_patterns.model_type}
]

```  # 格式化特殊情况的新对象并添加到新对象列表
# 将新对象添加到列表中并继续下一个对象的处理
            new_objects.append(obj)
            continue

        # 检查是否存在特殊模式，如果存在则替换为新模式
        special_pattern = False
        for pattern, attr in SPECIAL_PATTERNS.items():
            if pattern in obj:
                obj = obj.replace(getattr(old_model_patterns, attr), getattr(new_model_patterns, attr))
                new_objects.append(obj)
                special_pattern = True
                break

        # 如果存在特殊模式，则继续下一个对象的处理
        if special_pattern:
            continue

        # 处理常规类和函数
        old_obj = obj
        obj, replacement = replace_model_patterns(obj, old_model_patterns, new_model_patterns)
        # 检查是否存在“# Copied from”语句，如果不存在且需要添加，则在类/函数定义之前添加
        has_copied_from = re.search(r"^#\s+Copied from", obj, flags=re.MULTILINE) is not None
        if add_copied_from and not has_copied_from and _re_class_func.search(obj) is not None and len(replacement) > 0:
            # 添加“# Copied from”语句，必须在类/函数定义之前添加，可能不是第一行因为装饰器的存在
            module_name = get_module_from_file(module_file)
            old_object_name = _re_class_func.search(old_obj).groups()[0]
            obj = add_content_to_text(
                obj, f"# Copied from {module_name}.{old_object_name} with {replacement}", add_before=_re_class_func
            )
        # 在所有情况下，删除带有方法缩进的“Copied from”语句
        obj = re.sub("\n[ ]+# Copied from [^\n]*\n", "\n", obj)

        # 将处理后的对象添加到列表中
        new_objects.append(obj)

    # 将所有对象内容连接成一个字符串
    content = "\n".join(new_objects)
    # 移除不想复制到新文件中的一些属性
    if attrs_to_remove is not None:
        for attr in attrs_to_remove:
            content = remove_attributes(content, target_attr=attr)

    # 将处理后的内容写入目标文件
    with open(dest_file, "w", encoding="utf-8") as f:
        f.write(content)


def filter_framework_files(
    files: List[Union[str, os.PathLike]], frameworks: Optional[List[str]] = None
) -> List[Union[str, os.PathLike]]:
    """
    从文件列表中筛选出与指定框架对应的文件。

    Args:
        files (`List[Union[str, os.PathLike]]`): 需要筛选的文件列表。
        frameworks (`List[str]`, *optional*): 允许的框架列表。

    Returns:
        `List[Union[str, os.PathLike]]`: 筛选后的文件列表。
    """
    # 如果未提供框架列表，则使用默认框架列表
    if frameworks is None:
        frameworks = get_default_frameworks()

    # 创建框架到文件的映射字典
    framework_to_file = {}
    others = []
    for f in files:
        parts = Path(f).name.split("_")
        # 如果文件名中不包含“modeling”，则将文件添加到“others”列表中
        if "modeling" not in parts:
            others.append(f)
            continue
        # 根据文件名中的关键词确定框架类型，并将文件添加到相应框架的字典中
        if "tf" in parts:
            framework_to_file["tf"] = f
        elif "flax" in parts:
            framework_to_file["flax"] = f
        else:
            framework_to_file["pt"] = f

    # 返回筛选后的文件列表
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
        - **test_files** -- The test files for the model.
    """
    # 根据模型类型获取对应的模块名
    module_name = model_type_to_module_name(model_type)

    # 拼接模型模块的路径
    model_module = TRANSFORMERS_PATH / "models" / module_name
    # 获取模型模块下所有的.py文件
    model_files = list(model_module.glob("*.py"))
    # 根据传入的框架过滤模型文件
    model_files = filter_framework_files(model_files, frameworks=frameworks)

    # 拼接模型文档文件的路径
    doc_file = REPO_PATH / "docs" / "source" / "en" / "model_doc" / f"{model_type}.md"

    # 定义测试文件的基本模式
    test_files = [
        f"test_modeling_{module_name}.py",
        f"test_modeling_tf_{module_name}.py",
        f"test_modeling_flax_{module_name}.py",
        f"test_tokenization_{module_name}.py",
        f"test_image_processing_{module_name}.py",
        f"test_feature_extraction_{module_name}.py",
        f"test_processor_{module_name}.py",
    ]
    # 根据传入的框架过滤测试文件
    test_files = filter_framework_files(test_files, frameworks=frameworks)
    # 添加测试文件的目录路径
    test_files = [REPO_PATH / "tests" / "models" / module_name / f for f in test_files]
    # 过滤存在的文件
    test_files = [f for f in test_files if f.exists()]

    # 返回包含文档文件、模型文件、模块名和测试文件的字典
    return {"doc_file": doc_file, "model_files": model_files, "module_name": module_name, "test_files": test_files}
# 编译正则表达式，用于匹配_CHECKPOINT_FOR_DOC的赋值语句
_re_checkpoint_for_doc = re.compile(r"^_CHECKPOINT_FOR_DOC\s+=\s+(\S*)\s*$", flags=re.MULTILINE)

# 查找给定模型的文档字符串中使用的模型检查点
def find_base_model_checkpoint(
    model_type: str, model_files: Optional[Dict[str, Union[Path, List[Path]]] = None
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
    # 如果未提供model_files，则获取model_files
    if model_files is None:
        model_files = get_model_files(model_type)
    module_files = model_files["model_files"]
    # 遍历模型文件
    for fname in module_files:
        # 如果文件名中不包含"modeling"，则跳过
        if "modeling" not in str(fname):
            continue

        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
            # 搜索_CHECKPOINT_FOR_DOC并提取检查点
            if _re_checkpoint_for_doc.search(content) is not None:
                checkpoint = _re_checkpoint_for_doc.search(content).groups()[0]
                # 去除引号
                checkpoint = checkpoint.replace('"', "")
                checkpoint = checkpoint.replace("'", "")
                return checkpoint

    # TODO: Find some kind of fallback if there is no _CHECKPOINT_FOR_DOC in any of the modeling file.
    return ""

# 返回已安装在环境中的框架列表（PyTorch、TensorFlow、Flax）
def get_default_frameworks():
    """
    Returns the list of frameworks (PyTorch, TensorFlow, Flax) that are installed in the environment.
    """
    frameworks = []
    if is_torch_available():
        frameworks.append("pt")
    if is_tf_available():
        frameworks.append("tf")
    if is_flax_available():
        frameworks.append("flax")
    return frameworks

# 编译正则表达式，用于匹配MODEL_开头的映射名称
_re_model_mapping = re.compile("MODEL_([A-Z_]*)MAPPING_NAMES")

# 检索与给定模型相关联的模型类
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
    # 如果未提供frameworks，则获取默认框架列表
    if frameworks is None:
        frameworks = get_default_frameworks()

    # 根据框架的可用性选择相应的模块
    modules = {
        "pt": auto_module.modeling_auto if is_torch_available() else None,
        "tf": auto_module.modeling_tf_auto if is_tf_available() else None,
        "flax": auto_module.modeling_flax_auto if is_flax_available() else None,
    }

    model_classes = {}
    # 遍历给定的框架列表
    for framework in frameworks:
        # 初始化一个空列表，用于存储新的模型类
        new_model_classes = []
        # 检查给定框架是否已安装
        if modules[framework] is None:
            # 如果未安装，则抛出值错误异常
            raise ValueError(f"You selected {framework} in the frameworks, but it is not installed.")
        # 获取框架模块中包含模型映射的属性列表
        model_mappings = [attr for attr in dir(modules[framework]) if _re_model_mapping.search(attr) is not None]
        # 遍历模型映射列表
        for model_mapping_name in model_mappings:
            # 获取模型映射对象
            model_mapping = getattr(modules[framework], model_mapping_name)
            # 检查模型类型是否在模型映射中
            if model_type in model_mapping:
                # 如果是，则将对应的模型类添加到新模型类列表中
                new_model_classes.append(model_mapping[model_type])

        # 如果新模型类列表不为空
        if len(new_model_classes) > 0:
            # 去除重复的模型类
            model_classes[framework] = list(set(new_model_classes))

    # 返回包含不同框架中模型类的字典
    return model_classes
def retrieve_info_for_model(model_type, frameworks: Optional[List[str]] = None):
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
        - **model_files** (`Dict[str, Union[Path, List[Path]]`): The files associated with that model type.
        - **model_patterns** (`ModelPatterns`): The various patterns for the model.
    """
    # 检查传入的 model_type 是否在 MODEL_NAMES_MAPPING 中
    if model_type not in auto_module.MODEL_NAMES_MAPPING:
        raise ValueError(f"{model_type} is not a valid model type.")

    # 获取 model_type 对应的 model_name, config_class, archive_map
    model_name = auto_module.MODEL_NAMES_MAPPING[model_type]
    config_class = auto_module.configuration_auto.CONFIG_MAPPING_NAMES[model_type]
    archive_map = auto_module.configuration_auto.CONFIG_ARCHIVE_MAP_MAPPING_NAMES.get(model_type, None)
    
    # 获取 model_type 对应的 tokenizer_class, image_processor_class, feature_extractor_class, processor_class
    if model_type in auto_module.tokenization_auto.TOKENIZER_MAPPING_NAMES:
        tokenizer_classes = auto_module.tokenization_auto.TOKENIZER_MAPPING_NAMES[model_type]
        tokenizer_class = tokenizer_classes[0] if tokenizer_classes[0] is not None else tokenizer_classes[1]
    else:
        tokenizer_class = None
    image_processor_class = auto_module.image_processing_auto.IMAGE_PROCESSOR_MAPPING_NAMES.get(model_type, None)
    feature_extractor_class = auto_module.feature_extraction_auto.FEATURE_EXTRACTOR_MAPPING_NAMES.get(model_type, None)
    processor_class = auto_module.processing_auto.PROCESSOR_MAPPING_NAMES.get(model_type, None)

    # 获取 model_files
    model_files = get_model_files(model_type, frameworks=frameworks)
    model_camel_cased = config_class.replace("Config", "")

    # 根据 model_files 中的文件名判断可用的 frameworks
    available_frameworks = []
    for fname in model_files["model_files"]:
        if "modeling_tf" in str(fname):
            available_frameworks.append("tf")
        elif "modeling_flax" in str(fname):
            available_frameworks.append("flax")
        elif "modeling" in str(fname):
            available_frameworks.append("pt")

    # 如果 frameworks 为 None，则获取默认的 frameworks
    if frameworks is None:
        frameworks = get_default_frameworks()

    # 筛选出 frameworks 中存在于 available_frameworks 中的元素
    frameworks = [f for f in frameworks if f in available_frameworks]

    # 获取 model_classes
    model_classes = retrieve_model_classes(model_type, frameworks=frameworks)

    # 如果 archive_map 为 None，则从 pretrained archive map 的常量名中获取 model_upper_cased
    if archive_map is None:
        model_upper_cased = model_camel_cased.upper()
    else:
        # 如果不是预训练模型，则按照下划线分割字符串
        parts = archive_map.split("_")
        idx = 0
        # 在分割后的字符串中查找"PRETRAINED"关键词的位置
        while idx < len(parts) and parts[idx] != "PRETRAINED":
            idx += 1
        # 如果找到"PRETRAINED"关键词，则将之前的部分拼接起来作为模型名称
        if idx < len(parts):
            model_upper_cased = "_".join(parts[:idx])
        else:
            # 如果没有找到"PRETRAINED"关键词，则使用默认的模型驼峰命名并转换为大写
            model_upper_cased = model_camel_cased.upper()

    # 创建模型模式对象，包括模型名称、检查点、模型类型等信息
    model_patterns = ModelPatterns(
        model_name,
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

    # 返回包含框架、模型类、模型文件和模型模式的字典
    return {
        "frameworks": frameworks,
        "model_classes": model_classes,
        "model_files": model_files,
        "model_patterns": model_patterns,
    }
# 定义一个函数，用于清理初始化文件中不属于给定框架列表或与预处理（分词器、特征提取器、图像处理器、处理器）有关的导入行
def clean_frameworks_in_init(
    init_file: Union[str, os.PathLike], frameworks: Optional[List[str]] = None, keep_processing: bool = True
):
    """
    Removes all the import lines that don't belong to a given list of frameworks or concern tokenizers/feature
    extractors/image processors/processors in an init.

    Args:
        init_file (`str` or `os.PathLike`): The path to the init to treat.
        frameworks (`List[str]`, *optional*):
           If passed, this will remove all imports that are subject to a framework not in frameworks
        keep_processing (`bool`, *optional*, defaults to `True`):
            Whether or not to keep the preprocessing (tokenizer, feature extractor, image processor, processor) imports
            in the init.
    """
    # 如果未提供框架列表，则使用默认框架列表
    if frameworks is None:
        frameworks = get_default_frameworks()

    # 定义用于替换的名称映射字典
    names = {"pt": "torch"}
    # 准备待删除的模块列表
    to_remove = [names.get(f, f) for f in ["pt", "tf", "flax"] if f not in frameworks]
    # 如果不保留预处理相关的导入，将其添加到待删除列表中
    if not keep_processing:
        to_remove.extend(["sentencepiece", "tokenizers", "vision"])

    # 如果没有需要删除的导入行，则直接返回
    if len(to_remove) == 0:
        # Nothing to do
        return

    # 构建正则表达式模式，用于匹配待删除的导入行
    remove_pattern = "|".join(to_remove)
    re_conditional_imports = re.compile(rf"^\s*if not is_({remove_pattern})_available\(\):\s*$")
    re_try = re.compile(r"\s*try:")
    re_else = re.compile(r"\s*else:")
    re_is_xxx_available = re.compile(rf"is_({remove_pattern})_available")

    # 打开初始化文件，读取其中内容
    with open(init_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 将文件内容按行拆分
    lines = content.split("\n")
    # 初始化新行列表
    new_lines = []
    # 初始化行索引
    idx = 0
```  
    # 当索引小于行数时，继续循环
    while idx < len(lines):
        # 在 try-except-else 块中进行条件导入
        if (re_conditional_imports.search(lines[idx]) is not None) and (re_try.search(lines[idx - 1]) is not None):
            # 移除前面的 `try:`
            new_lines.pop()
            idx += 1
            # 迭代直到 `else:`
            while is_empty_line(lines[idx]) or re_else.search(lines[idx]) is None:
                idx += 1
            idx += 1
            indent = find_indent(lines[idx])
            while find_indent(lines[idx]) >= indent or is_empty_line(lines[idx]):
                idx += 1
        # 从 utils 中移除导入
        elif re_is_xxx_available.search(lines[idx]) is not None:
            line = lines[idx]
            for framework in to_remove:
                line = line.replace(f", is_{framework}_available", "")
                line = line.replace(f"is_{framework}_available, ", "")
                line = line.replace(f"is_{framework}_available,", "")
                line = line.replace(f"is_{framework}_available", "")

            if len(line.strip()) > 0:
                new_lines.append(line)
            idx += 1
        # 否则保留该行，除非它是一个分词器导入且我们不想保留它
        elif keep_processing or (
            re.search(r'^\s*"(tokenization|processing|feature_extraction|image_processing)', lines[idx]) is None
            and re.search(r"^\s*from .(tokenization|processing|feature_extraction|image_processing)", lines[idx])
            is None
        ):
            new_lines.append(lines[idx])
            idx += 1
        else:
            idx += 1

    # 将新行写入初始化文件
    with open(init_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
def add_model_to_main_init(
    old_model_patterns: ModelPatterns,  # 旧模型的模式
    new_model_patterns: ModelPatterns,  # 新模型的模式
    frameworks: Optional[List[str]] = None,  # 指定的框架列表，默认为 None
    with_processing: bool = True,  # 是否包含处理，默认为 True
):
    """
    Add a model to the main init of Transformers.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
            旧模型的模式
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
            新模型的模式
        frameworks (`List[str]`, *optional*):
            If specified, only the models implemented in those frameworks will be added.
            如果指定，则只添加在这些框架中实现的模型。
        with_processsing (`bool`, *optional*, defaults to `True`):
            Whether the tokenizer/feature extractor/processor of the model should also be added to the init or not.
            是否还应将模型的分词器/特征提取器/处理器添加到初始化中。

    """
    with open(TRANSFORMERS_PATH / "__init__.py", "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")  # 将文件内容按行分割成列表
    idx = 0  # 初始化索引变量
    new_lines = []  # 初始化新行列表
    framework = None  # 初始化框架变量
    while idx < len(lines):
        new_framework = False
        # 如果当前行不是空行且缩进为0，则重置framework为None
        if not is_empty_line(lines[idx]) and find_indent(lines[idx]) == 0:
            framework = None
        # 如果当前行以"if not is_torch_available"开头，则设置framework为"pt"
        elif lines[idx].lstrip().startswith("if not is_torch_available"):
            framework = "pt"
            new_framework = True
        # 如果当前行以"if not is_tf_available"开头，则设置framework为"tf"
        elif lines[idx].lstrip().startswith("if not is_tf_available"):
            framework = "tf"
            new_framework = True
        # 如果当前行以"if not is_flax_available"开头，则设置framework为"flax"
        elif lines[idx].lstrip().startswith("if not is_flax_available"):
            framework = "flax"
            new_framework = True

        # 如果发现了新的framework，则跳过直到找到else:块以确定导入的位置
        if new_framework:
            while lines[idx].strip() != "else:":
                new_lines.append(lines[idx])
                idx += 1

        # 如果当前framework不在所需framework列表中，则跳过
        if framework is not None and frameworks is not None and framework not in frameworks:
            new_lines.append(lines[idx])
            idx += 1
        # 如果当前行包含旧模型名称，则替换为新模型名称
        elif re.search(rf'models.{old_model_patterns.model_lower_cased}( |")', lines[idx]) is not None:
            block = [lines[idx]]
            indent = find_indent(lines[idx])
            idx += 1
            while find_indent(lines[idx]) > indent:
                block.append(lines[idx])
                idx += 1
            # 如果下一行是")", "]", "],"之一，则将其添加到块中
            if lines[idx].strip() in [")", "]", "],"]:
                block.append(lines[idx])
                idx += 1
            block = "\n".join(block)
            new_lines.append(block)

            add_block = True
            # 如果不需要处理，则移除处理类
            if not with_processing:
                processing_classes = [
                    old_model_patterns.tokenizer_class,
                    old_model_patterns.image_processor_class,
                    old_model_patterns.feature_extractor_class,
                    old_model_patterns.processor_class,
                ]
                # 仅保留不为None的处理类
                processing_classes = [c for c in processing_classes if c is not None]
                for processing_class in processing_classes:
                    block = block.replace(f' "{processing_class}",', "")
                    block = block.replace(f', "{processing_class}"', "")
                    block = block.replace(f" {processing_class},", "")
                    block = block.replace(f", {processing_class}", "")

                    if processing_class in block:
                        add_block = False
            if add_block:
                # 将旧模型名称替换为新模型名称
                new_lines.append(replace_model_patterns(block, old_model_patterns, new_model_patterns)[0])
        else:
            new_lines.append(lines[idx])
            idx += 1

    # 将更新后的代码写回到文件中
    with open(TRANSFORMERS_PATH / "__init__.py", "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))
def insert_tokenizer_in_auto_module(old_model_patterns: ModelPatterns, new_model_patterns: ModelPatterns):
    """
    Add a tokenizer to the relevant mappings in the auto module.

    Args:
        old_model_patterns (`ModelPatterns`): The patterns for the old model.
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
    """
    # 如果旧模型或新模型的分词器类为None，则直接返回
    if old_model_patterns.tokenizer_class is None or new_model_patterns.tokenizer_class is None:
        return

    # 读取tokenization_auto.py文件内容
    with open(TRANSFORMERS_PATH / "models" / "auto" / "tokenization_auto.py", "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    idx = 0
    # 找到TOKENIZER_MAPPING_NAMES块
    while not lines[idx].startswith("    TOKENIZER_MAPPING_NAMES = OrderedDict("):
        idx += 1
    idx += 1

    # TOKENIZER_MAPPING_NAMES块结束于此处
    while not lines[idx].startswith("TOKENIZER_MAPPING = _LazyAutoMapping"):
        # 如果分词器块在一行内定义，则以"),结束"
        if lines[idx].endswith(","):
            block = lines[idx]
        # 否则，直到遇到"),"为止
        else:
            block = []
            while not lines[idx].startswith("            ),"):
                block.append(lines[idx])
                idx += 1
            block = "\n".join(block)
        idx += 1

        # 如果在该块中找到旧模型类型和分词器类，则找到旧模型分词器块
        if f'"{old_model_patterns.model_type}"' in block and old_model_patterns.tokenizer_class in block:
            break

    # 替换旧模型类型和分词器类为新模型类型和分词器类
    new_block = block.replace(old_model_patterns.model_type, new_model_patterns.model_type)
    new_block = new_block.replace(old_model_patterns.tokenizer_class, new_model_patterns.tokenizer_class)

    new_lines = lines[:idx] + [new_block] + lines[idx:]
    # 将更新后的内容写回tokenization_auto.py文件
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
        old_model_patterns (`ModelPatterns`): 旧模型的模式。
        new_model_patterns (`ModelPatterns`): 新模型的模式。
        model_classes (`Dict[str, List[str]]`): 一个字典，将框架映射到实现的模型类列表。
    """
    # Tokenizers require special handling
    # 插入分词器到自动模块中
    insert_tokenizer_in_auto_module(old_model_patterns, new_model_patterns)
# 文档模板，包含了模型概述的基本结构，用于生成新的文档
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
    复制文档文件并根据新模型进行调整。

    Args:
        module_file (`str` or `os.PathLike`): 要复制的文档文件的路径。
        old_model_patterns (`ModelPatterns`): 旧模型的模式。
        new_model_patterns (`ModelPatterns`): 新模型的模式。
        dest_file (`str` or `os.PathLike`, *optional*): 新文档文件的路径。
            如果未提供，将默认为与`module_file`相同文件夹中命名为`{new_model_patterns.model_type}.md`的文件。
        frameworks (`List[str]`, *optional*):
            如果传递，新文档文件中只会保留与此框架列表对应的模型类。
    """
    # 读取原始文档文件的内容
    with open(doc_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 更新文档中的版权信息
    content = re.sub(r"<!--\s*Copyright (\d+)\s", f"<!--Copyright {CURRENT_YEAR} ", content)
    # 如果未指定框架列表，则使用默认框架列表
    if frameworks is None:
        frameworks = get_default_frameworks()
    # 如果未指定目标文件路径，则在与原始文档文件相同的文件夹中创建一个文件
    if dest_file is None:
        dest_file = Path(doc_file).parent / f"{new_model_patterns.model_type}.md"

    # 解析文档文件，每个段落/头部为一个块
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
    # 遍历给定的文本块列表
    for block in blocks:
        # 如果文本块不以 '#' 开头，则认为是版权信息，直接添加到新文本块列表中
        if not block.startswith("#"):
            new_blocks.append(block)
        # 如果文本块以 '#' 开头，且为主标题，则替换为新模型的名称
        elif re.search(r"^#\s+\S+", block) is not None:
            new_blocks.append(f"# {new_model_patterns.model_name}\n")
        # 如果文本块中包含旧模型的配置类，则认为接下来是类定义部分
        elif not in_classes and old_model_patterns.config_class in block.split("\n")[0]:
            # 标记已进入类定义部分
            in_classes = True
            # 添加新模型概述模板
            new_blocks.append(DOC_OVERVIEW_TEMPLATE.format(model_name=new_model_patterns.model_name))
            # 替换模型模式中的旧模式为新模式
            new_block, _ = replace_model_patterns(block, old_model_patterns, new_model_patterns)
            new_blocks.append(new_block)
        # 如果已经在类定义部分中
        elif in_classes:
            # 从文本块中提取标题
            block_title = block.split("\n")[0]
            # 从标题中提取类名
            block_class = re.search(r"^#+\s+(\S.*)$", block_title).groups()[0]
            # 替换模型模式中的旧模式为新模式
            new_block, _ = replace_model_patterns(block, old_model_patterns, new_model_patterns)

            # 根据类名判断需要添加的类是否变化，并添加到新文本块列表中
            if "Tokenizer" in block_class:
                if old_model_patterns.tokenizer_class != new_model_patterns.tokenizer_class:
                    new_blocks.append(new_block)
            elif "ImageProcessor" in block_class:
                if old_model_patterns.image_processor_class != new_model_patterns.image_processor_class:
                    new_blocks.append(new_block)
            elif "FeatureExtractor" in block_class:
                if old_model_patterns.feature_extractor_class != new_model_patterns.feature_extractor_class:
                    new_blocks.append(new_block)
            elif "Processor" in block_class:
                if old_model_patterns.processor_class != new_model_patterns.processor_class:
                    new_blocks.append(new_block)
            elif block_class.startswith("Flax"):
                if "flax" in frameworks:
                    new_blocks.append(new_block)
            elif block_class.startswith("TF"):
                if "tf" in frameworks:
                    new_blocks.append(new_block)
            elif len(block_class.split(" ")) == 1:
                if "pt" in frameworks:
                    new_blocks.append(new_block)
            else:
                new_blocks.append(new_block)

    # 将新文本块列表写入目标文件中
    with open(dest_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_blocks))
# 插入新模型到文档目录表中，放置于与旧模型相同的部分中
def insert_model_in_doc_toc(old_model_patterns, new_model_patterns):
    # 获取文档目录文件路径
    toc_file = REPO_PATH / "docs" / "source" / "en" / "_toctree.yml"
    # 以只读模式打开目录文件，加载内容
    with open(toc_file, "r", encoding="utf8") as f:
        content = yaml.safe_load(f)

    # 寻找到 "API" 部分
    api_idx = 0
    while content[api_idx]["title"] != "API":
        api_idx += 1
    api_doc = content[api_idx]["sections"]

    # 寻找到 "Models" 部分
    model_idx = 0
    while api_doc[model_idx]["title"] != "Models":
        model_idx += 1
    model_doc = api_doc[model_idx]["sections"]

    # 在 TOC 中查找基础模型
    old_model_type = old_model_patterns.model_type
    section_idx = 0
    while section_idx < len(model_doc):
        sections = [entry["local"] for entry in model_doc[section_idx]["sections"]]
        # 如果找到了旧模型的部分，则停止搜索
        if f"model_doc/{old_model_type}" in sections:
            break

        section_idx += 1

    # 如果没有找到旧模型的部分，则需要手动添加新模型
    if section_idx == len(model_doc):
        old_model = old_model_patterns.model_name
        new_model = new_model_patterns.model_name
        print(f"Did not find {old_model} in the table of content, so you will need to add {new_model} manually.")
        return

    # 将新模型添加到相同的目录部分中
    toc_entry = {"local": f"model_doc/{new_model_patterns.model_type}", "title": new_model_patterns.model_name}
    model_doc[section_idx]["sections"].append(toc_entry)
    # 按标题的小写字母顺序对目录部分进行排序
    model_doc[section_idx]["sections"] = sorted(model_doc[section_idx]["sections"], key=lambda s: s["title"].lower())
    api_doc[model_idx]["sections"] = model_doc
    content[api_idx]["sections"] = api_doc

    # 将更新后的内容写回目录文件
    with open(toc_file, "w", encoding="utf-8") as f:
        f.write(yaml.dump(content, allow_unicode=True))


# 创建一个与 Transformers 库中给定模型类似的新模型模块
def create_new_model_like(
    model_type: str,
    new_model_patterns: ModelPatterns,
    add_copied_from: bool = True,
    frameworks: Optional[List[str]] = None,
    old_checkpoint: Optional[str] = None,
):
    """
    Args:
        model_type (`str`): The model type to duplicate (like "bert" or "gpt2")
        new_model_patterns (`ModelPatterns`): The patterns for the new model.
        add_copied_from (`bool`, *optional*, defaults to `True`):
            Whether or not to add "Copied from" statements to all classes in the new model modeling files.
        frameworks (`List[str]`, *optional*):
            If passed, will limit the duplicate to the frameworks specified.
        old_checkpoint (`str`, *optional*):
            The name of the base checkpoint for the old model. Should be passed along when it can't be automatically
            recovered from the `model_type`.
    """
    # 检索所有旧模型信息
    # 获取指定模型类型的信息，包括模型文件、模型模式等
    model_info = retrieve_info_for_model(model_type, frameworks=frameworks)
    # 获取模型文件和模型模式
    model_files = model_info["model_files"]
    old_model_patterns = model_info["model_patterns"]
    # 如果存在旧的检查点，则更新模型模式中的检查点
    if old_checkpoint is not None:
        old_model_patterns.checkpoint = old_checkpoint
    # 如果旧的检查点为空，则抛出异常
    if len(old_model_patterns.checkpoint) == 0:
        raise ValueError(
            "The old model checkpoint could not be recovered from the model type. Please pass it to the "
            "`old_checkpoint` argument."
        )

    # 初始化变量，用于判断是否保留旧的处理方式
    keep_old_processing = True
    # 检查新旧模型模式中的处理属性是否一致，若不一致则不保留旧的处理方式
    for processing_attr in ["image_processor_class", "feature_extractor_class", "processor_class", "tokenizer_class"]:
        if getattr(old_model_patterns, processing_attr) != getattr(new_model_patterns, processing_attr):
            keep_old_processing = False

    # 获取模型类信息
    model_classes = model_info["model_classes"]

    # 1. 创建新模型的模块
    old_module_name = model_files["module_name"]
    module_folder = TRANSFORMERS_PATH / "models" / new_model_patterns.model_lower_cased
    # 创建模块文件夹
    os.makedirs(module_folder, exist_ok=True)

    # 获取需要调整的文件列表
    files_to_adapt = model_files["model_files"]
    # 如果保留旧的处理方式，则过滤掉与处理相关的文件
    if keep_old_processing:
        files_to_adapt = [
            f
            for f in files_to_adapt
            if "tokenization" not in str(f)
            and "processing" not in str(f)
            and "feature_extraction" not in str(f)
            and "image_processing" not in str(f)
        ]

    # 创建模块文件夹
    os.makedirs(module_folder, exist_ok=True)
    # 遍历需要调整的文件，复制并调整文件内容
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

    # 清理模块中的处理方式
    clean_frameworks_in_init(
        module_folder / "__init__.py", frameworks=frameworks, keep_processing=not keep_old_processing
    )

    # 2. 将新模型添加到模型初始化文件和主初始化文件中
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
    # 如果保留旧的处理方式，则过滤掉与处理相关的文件
    if keep_old_processing:
        files_to_adapt = [
            f
            for f in files_to_adapt
            if "tokenization" not in str(f)
            and "processor" not in str(f)
            and "feature_extraction" not in str(f)
            and "image_processing" not in str(f)
        ]
    # 定义一个函数，用于禁用指定文件中的某个标志
    def disable_fx_test(filename: Path) -> bool:
        # 以只读模式打开文件
        with open(filename) as fp:
            # 读取文件内容
            content = fp.read()
        # 使用正则表达式替换文件内容中的指定标志
        new_content = re.sub(r"fx_compatible\s*=\s*True", "fx_compatible = False", content)
        # 以写入模式打开文件
        with open(filename, "w") as fp:
            # 将新内容写入文件
            fp.write(new_content)
        # 返回是否文件内容有变化的布尔值
        return content != new_content

    # 初始化一个变量，用于记录是否禁用了指定测试
    disabled_fx_test = False

    # 设置测试文件夹路径，并确保其存在
    tests_folder = REPO_PATH / "tests" / "models" / new_model_patterns.model_lower_cased
    os.makedirs(tests_folder, exist_ok=True)
    # 在测试文件夹中创建一个空的 __init__.py 文件
    with open(tests_folder / "__init__.py", "w"):
        pass

    # 遍历需要调整的文件列表
    for test_file in files_to_adapt:
        # 构造新的测试文件名
        new_test_file_name = test_file.name.replace(
            old_model_patterns.model_lower_cased, new_model_patterns.model_lower_cased
        )
        # 确定目标文件路径
        dest_file = test_file.parent.parent / new_model_patterns.model_lower_cased / new_test_file_name
        # 复制模块文件，并进行必要的修改
        duplicate_module(
            test_file,
            old_model_patterns,
            new_model_patterns,
            dest_file=dest_file,
            add_copied_from=False,
            attrs_to_remove=["pipeline_model_mapping", "is_pipeline_test_to_skip"],
        )
        # 更新是否禁用指定测试的状态
        disabled_fx_test = disabled_fx_test | disable_fx_test(dest_file)

    # 如果有测试被禁用，则输出提示信息
    if disabled_fx_test:
        print(
            "The tests for symbolic tracing with torch.fx were disabled, you can add those once symbolic tracing works"
            " for your new model."
        )

    # 添加模型到自动类列表
    add_model_to_auto_classes(old_model_patterns, new_model_patterns, model_classes)

    # 添加文档文件
    doc_file = REPO_PATH / "docs" / "source" / "en" / "model_doc" / f"{old_model_patterns.model_type}.md"
    duplicate_doc_file(doc_file, old_model_patterns, new_model_patterns, frameworks=frameworks)
    insert_model_in_doc_toc(old_model_patterns, new_model_patterns)

    # 如果模型类型与检查点名称相同，则输出警告信息
    if old_model_patterns.model_type == old_model_patterns.checkpoint:
        print(
            "The model you picked has the same name for the model type and the checkpoint name "
            f"({old_model_patterns.model_type}). As a result, it's possible some places where the new checkpoint "
            f"should be, you have {new_model_patterns.model_type} instead. You should search for all instances of "
            f"{new_model_patterns.model_type} in the new files and check they're not badly used as checkpoints."
        )
    # 如果模型名称（小写形式）与检查点名称相同，则输出警告信息
    elif old_model_patterns.model_lower_cased == old_model_patterns.checkpoint:
        print(
            "The model you picked has the same name for the model type and the checkpoint name "
            f"({old_model_patterns.model_lower_cased}). As a result, it's possible some places where the new "
            f"checkpoint should be, you have {new_model_patterns.model_lower_cased} instead. You should search for "
            f"all instances of {new_model_patterns.model_lower_cased} in the new files and check they're not badly "
            "used as checkpoints."
        )
    # 检查旧模型类型是否为小写模型名称，并且新模型类型不是小写模型名称
    if (
        old_model_patterns.model_type == old_model_patterns.model_lower_cased
        and new_model_patterns.model_type != new_model_patterns.model_lower_cased
    ):
        # 打印警告信息，提示用户选择的模型类型和小写模型名称相同，可能导致新模型类型在某些地方被误用为小写模型名称
        print(
            "The model you picked has the same name for the model type and the lowercased model name "
            f"({old_model_patterns.model_lower_cased}). As a result, it's possible some places where the new "
            f"model type should be, you have {new_model_patterns.model_lower_cased} instead. You should search for "
            f"all instances of {new_model_patterns.model_lower_cased} in the new files and check they're not badly "
            "used as the model type."
        )
    
    # 如果不保留旧的处理方式并且旧模型的分词器类不为空
    if not keep_old_processing and old_model_patterns.tokenizer_class is not None:
        # 打印警告信息，提示需要手动修复新分词器文件开头的常量，如果新模型有一个快速分词器，则需要手动将转换器添加到`convert_slow_tokenizer.py`的`SLOW_TO_FAST_CONVERTERS`常量中
        print(
            "The constants at the start of the new tokenizer file created needs to be manually fixed. If your new "
            "model has a tokenizer fast, you will also need to manually add the converter in the "
            "`SLOW_TO_FAST_CONVERTERS` constant of `convert_slow_tokenizer.py`."
        )
# 创建一个函数，用于生成添加新模型的命令对象
def add_new_model_like_command_factory(args: Namespace):
    return AddNewModelLikeCommand(config_file=args.config_file, path_to_repo=args.path_to_repo)


class AddNewModelLikeCommand(BaseTransformersCLICommand):
    # 静态方法，注册子命令
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        # 添加新模型命令解析器
        add_new_model_like_parser = parser.add_parser("add-new-model-like")
        # 添加配置文件参数
        add_new_model_like_parser.add_argument(
            "--config_file", type=str, help="A file with all the information for this model creation."
        )
        # 添加 Transformers 仓库路径参数
        add_new_model_like_parser.add_argument(
            "--path_to_repo", type=str, help="When not using an editable install, the path to the Transformers repo."
        )
        # 设置默认函数为 add_new_model_like_command_factory
        add_new_model_like_parser.set_defaults(func=add_new_model_like_command_factory)

    # 初始化方法，接收配置文件路径和 Transformers 仓库路径等参数
    def __init__(self, config_file=None, path_to_repo=None, *args):
        # 如果配置文件不为空，则从配置文件中加载配置信息
        if config_file is not None:
            # 从配置文件中加载配置信息
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            # 获取旧模型类型
            self.old_model_type = config["old_model_type"]
            # 获取新模型模式
            self.model_patterns = ModelPatterns(**config["new_model_patterns"])
            # 获取是否添加来源信息的标志
            self.add_copied_from = config.get("add_copied_from", True)
            # 获取框架信息
            self.frameworks = config.get("frameworks", get_default_frameworks())
            # 获取旧检查点信息
            self.old_checkpoint = config.get("old_checkpoint", None)
        else:
            # 从用户输入获取参数
            (
                self.old_model_type,
                self.model_patterns,
                self.add_copied_from,
                self.frameworks,
                self.old_checkpoint,
            ) = get_user_input()

        # 保存 Transformers 仓库路径
        self.path_to_repo = path_to_repo

    # 运行命令方法
    def run(self):
        # 如果指定了 Transformers 仓库路径
        if self.path_to_repo is not None:
            # 调整常量
            global TRANSFORMERS_PATH
            global REPO_PATH

            # 设置仓库路径
            REPO_PATH = Path(self.path_to_repo)
            # 设置 Transformers 路径
            TRANSFORMERS_PATH = REPO_PATH / "src" / "transformers"

        # 创建类似新模型的模型
        create_new_model_like(
            model_type=self.old_model_type,
            new_model_patterns=self.model_patterns,
            add_copied_from=self.add_copied_from,
            frameworks=self.frameworks,
            old_checkpoint=self.old_checkpoint,
        )


# 获取用户字段的实用函数
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
    # 函数定义，用于向用户提出问题并获取回答
    Args:
        # 问题内容，字符串类型
        question (`str`): The question to ask the user.
        # 默认值，如果用户没有提供答案，则使用该值，字符串类型，可选
        default_value (`str`, *optional*): A potential default value that will be used when the answer is empty.
        # 检查答案是否有效的函数，可选
        is_valid_answer (`Callable`, *optional*):
            # 如果设置了此函数，则将一直询问问题，直到该函数对提供的答案返回True
            If set, the question will be asked until this function returns `True` on the provided answer.
        # 将答案转换为指定类型的函数，可选
        convert_to (`Callable`, *optional`):
            # 如果设置了此函数，则将答案传递给该函数。如果该函数在提供的答案上引发错误，则会再次询问问题。
            If set, the answer will be passed to this function. If this function raises an error on the provided
            answer, the question will be asked again.
        # 当问题被再次询问时显示的消息，可选
        fallback_message (`str`, *optional*):
            # 每次再次询问用户问题时显示的消息
            A message that will be displayed each time the question is asked again to the user.

    # 返回用户提供的答案（或默认答案），经过可能的转换函数处理后的值
    Returns:
        `Any`: The answer provided by the user (or the default), passed through the potential conversion function.
    """
    # 如果问题没有以空格结尾，则在问题末尾添加空格
    if not question.endswith(" "):
        question = question + " "
    # 如果提供了默认值，则在问题中包含默认值
    if default_value is not None:
        question = f"{question} [{default_value}] "

    # 循环直到获得有效答案
    valid_answer = False
    while not valid_answer:
        # 询问问题，并获取用户输入的答案
        answer = input(question)
        # 如果提供了默认值且答案为空，则将答案设置为默认值
        if default_value is not None and len(answer) == 0:
            answer = default_value
        # 如果提供了答案有效性检查函数
        if is_valid_answer is not None:
            # 调用该函数检查答案是否有效
            valid_answer = is_valid_answer(answer)
        # 如果提供了答案转换函数
        elif convert_to is not None:
            try:
                # 尝试将答案转换为指定类型
                answer = convert_to(answer)
                # 标记答案有效
                valid_answer = True
            except Exception:
                # 如果转换失败，则答案无效
                valid_answer = False
        else:
            # 如果没有提供答案有效性检查函数和转换函数，则答案有效
            valid_answer = True

        # 如果答案无效，则打印回退消息
        if not valid_answer:
            print(fallback_message)

    # 返回答案
    return answer
def convert_to_bool(x: str) -> bool:
    """
    Converts a string to a bool.
    """
    # 将字符串转换为布尔值
    if x.lower() in ["1", "y", "yes", "true"]:
        return True
    if x.lower() in ["0", "n", "no", "false"]:
        return False
    raise ValueError(f"{x} is not a value that can be converted to a bool."  # 如果无法转换为布尔值，则引发值错误异常


def get_user_input():
    """
    Ask the user for the necessary inputs to add the new model.
    """
    model_types = list(auto_module.configuration_auto.MODEL_NAMES_MAPPING.keys())

    # 获取旧模型类型
    valid_model_type = False
    while not valid_model_type:
        old_model_type = input(
            "What is the model you would like to duplicate? Please provide the lowercase `model_type` (e.g. roberta): "
        )
        if old_model_type in model_types:
            valid_model_type = True
        else:
            print(f"{old_model_type} is not a valid model type.")
            near_choices = difflib.get_close_matches(old_model_type, model_types)
            if len(near_choices) >= 1:
                if len(near_choices) > 1:
                    near_choices = " or ".join(near_choices)
                print(f"Did you mean {near_choices}?")

    old_model_info = retrieve_info_for_model(old_model_type)
    old_tokenizer_class = old_model_info["model_patterns"].tokenizer_class
    old_image_processor_class = old_model_info["model_patterns"].image_processor_class
    old_feature_extractor_class = old_model_info["model_patterns"].feature_extractor_class
    old_processor_class = old_model_info["model_patterns"].processor_class
    old_frameworks = old_model_info["frameworks"]

    old_checkpoint = None
    if len(old_model_info["model_patterns"].checkpoint) == 0:
        old_checkpoint = get_user_field(
            "We couldn't find the name of the base checkpoint for that model, please enter it here."
        )

    model_name = get_user_field(
        "What is the name (with no special casing) for your new model in the paper (e.g. RoBERTa)? "
    )
    default_patterns = ModelPatterns(model_name, model_name)

    model_type = get_user_field(
        "What identifier would you like to use for the `model_type` of this model? ",
        default_value=default_patterns.model_type,
    )
    model_lower_cased = get_user_field(
        "What lowercase name would you like to use for the module (folder) of this model? ",
        default_value=default_patterns.model_lower_cased,
    )
    model_camel_cased = get_user_field(
        "What prefix (camel-cased) would you like to use for the model classes of this model (e.g. Roberta)? ",
        default_value=default_patterns.model_camel_cased,
    )
    model_upper_cased = get_user_field(
        "What prefix (upper-cased) would you like to use for the constants relative to this model? ",
        default_value=default_patterns.model_upper_cased,
    )
    config_class = get_user_field(
        "What will be the name of the config class for this model? ", default_value=f"{model_camel_cased}Config"
    )
    # 询问用户输入一个检查点标识符（在模型 Hub 上）用于这个新模型（例如 facebook/roberta-base）
    checkpoint = get_user_field(
        "Please give a checkpoint identifier (on the model Hub) for this new model (e.g. facebook/roberta-base): "
    )

    # 将旧的处理类组成一个列表，如果类不为空则加入列表中
    old_processing_classes = [
        c
        for c in [old_image_processor_class, old_feature_extractor_class, old_tokenizer_class, old_processor_class]
        if c is not None
    ]
    # 将列表中的类名用逗号连接成字符串
    old_processing_classes = ", ".join(old_processing_classes)
    # 询问用户新模型是否使用与旧模型相同的处理类
    keep_processing = get_user_field(
        f"Will your new model use the same processing class as {old_model_type} ({old_processing_classes}) (yes/no)? ",
        convert_to=convert_to_bool,
        fallback_message="Please answer yes/no, y/n, true/false or 1/0. ",
    )
    # 如果用户选择保持旧的处理类，则将新模型的处理类设置为旧模型的处理类
    if keep_processing:
        image_processor_class = old_image_processor_class
        feature_extractor_class = old_feature_extractor_class
        processor_class = old_processor_class
        tokenizer_class = old_tokenizer_class
    # 如果用户选择不保持旧的处理类，则询问用户新模型的处理类名称
    else:
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

    # 创建模型模式对象，包含模型的各种属性和类名
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

    # 询问用户在创建新建模型文件时是否添加 # Copied from 语句
    add_copied_from = get_user_field(
        "Should we add # Copied from statements when creating the new modeling file (yes/no)? ",
        convert_to=convert_to_bool,
        default_value="yes",
        fallback_message="Please answer yes/no, y/n, true/false or 1/0.",
    )
    # 询问用户是否要在所有旧模型类型已实现的框架中添加新模型的版本，根据用户提供的信息设置默认值为“yes”
    all_frameworks = get_user_field(
        "Should we add a version of your new model in all the frameworks implemented by"
        f" {old_model_type} ({old_frameworks}) (yes/no)? ",
        convert_to=convert_to_bool,
        default_value="yes",
        fallback_message="Please answer yes/no, y/n, true/false or 1/0.",
    )
    # 如果用户选择在所有框架中添加新模型，则将框架列表设为 None
    if all_frameworks:
        frameworks = None
    else:
        # 否则，询问用户要在哪些框架中添加新模型，要求输入以空格分隔的框架列表，只允许输入 "pt", "tf", "flax" 中的一种或多种
        frameworks = get_user_field(
            "Please enter the list of framworks you want (pt, tf, flax) separated by spaces",
            is_valid_answer=lambda x: all(p in ["pt", "tf", "flax"] for p in x.split(" ")),
        )
        # 将用户输入的框架列表转换为集合并转换为列表
        frameworks = list(set(frameworks.split(" ")))
    
    # 返回旧模型类型、模型模式、是否添加副本、框架列表和旧检查点的元组
    return (old_model_type, model_patterns, add_copied_from, frameworks, old_checkpoint)
```