# `.\pytorch\docs\source\scripts\exportdb\generate_example_rst.py`

```py
# 导入模块inspect、os、re以及Path类
import inspect
import os
import re
from pathlib import Path

# 导入torch模块和torchdynamo模块中的私有模块torch._dynamo
import torch
import torch._dynamo as torchdynamo

# 从torch._export.db.case中导入ExportCase类和normalize_inputs函数
from torch._export.db.case import ExportCase, normalize_inputs
# 从torch._export.db.examples中导入all_examples函数
from torch._export.db.examples import all_examples
# 从torch.export中导入export函数

from torch.export import export

# 获取当前文件的绝对路径的父目录
PWD = Path(__file__).absolute().parent
# 获取当前文件的绝对路径的父目录的父目录的父目录的父目录，即项目根目录
ROOT = Path(__file__).absolute().parent.parent.parent.parent
# 定义常量SOURCE为ROOT目录下的source子目录
SOURCE = ROOT / Path("source")
# 定义常量EXPORTDB_SOURCE为SOURCE目录下的generated/exportdb子目录
EXPORTDB_SOURCE = SOURCE / Path("generated") / Path("exportdb")

# 定义函数generate_example_rst，生成指定ExportCase对象的.rst文件内容
def generate_example_rst(example_case: ExportCase):
    """
    Generates the .rst files for all the examples in db/examples/
    """

    # 获取示例案例中的模型
    model = example_case.model

    # 生成标签字符串，包含所有示例案例的标签
    tags = ", ".join(f":doc:`{tag} <{tag}>`" for tag in example_case.tags)

    # 获取模型类所在的源文件路径
    source_file = (
        inspect.getfile(model.__class__)
        if isinstance(model, torch.nn.Module)
        else inspect.getfile(model)
    )
    # 读取源文件内容
    with open(source_file) as file:
        source_code = file.read()
    # 将源代码中的换行符替换为四个空格（用于.rst文件格式）
    source_code = source_code.replace("\n", "\n    ")
    # 根据@export_rewrite_case标记分割源代码
    splitted_source_code = re.split(r"@export_rewrite_case.*\n", source_code)

    # 断言分割后的源代码部分数量为1或2
    assert len(splitted_source_code) in {
        1,
        2,
    }, f"more than one @export_rewrite_case decorator in {source_code}"

    # 生成.rst文件的内容
    title = f"{example_case.name}"
    doc_contents = f"""{title}
{'^' * (len(title))}

.. note::

    Tags: {tags}

    Support Level: {example_case.support_level.name}

Original source code:

.. code-block:: python

    {splitted_source_code[0]}

Result:

.. code-block::

"""

    # 尝试获取动态跟踪的结果图形
    try:
        # 标准化示例输入
        inputs = normalize_inputs(example_case.example_inputs)
        # 导出模型
        exported_program = export(
            model,
            inputs.args,
            inputs.kwargs,
            dynamic_shapes=example_case.dynamic_shapes,
        )
        # 获取导出的图形输出并进行格式化处理
        graph_output = str(exported_program)
        graph_output = re.sub(r"        # File(.|\n)*?\n", "", graph_output)
        graph_output = graph_output.replace("\n", "\n    ")
        output = f"    {graph_output}"
    except torchdynamo.exc.Unsupported as e:
        # 处理torchdynamo.exc.Unsupported异常
        output = "    Unsupported: " + str(e).split("\n")[0]
    except AssertionError as e:
        # 处理AssertionError异常
        output = "    AssertionError: " + str(e).split("\n")[0]
    except RuntimeError as e:
        # 处理RuntimeError异常
        output = "    RuntimeError: " + str(e).split("\n")[0]

    # 将输出内容添加到doc_contents中
    doc_contents += output + "\n"

    # 如果源代码被分成两部分，则添加重写示例的建议到doc_contents中
    if len(splitted_source_code) == 2:
        doc_contents += f"""\n
You can rewrite the example above to something like the following:

.. code-block:: python

{splitted_source_code[1]}

"""

    # 返回生成的.rst文件内容
    return doc_contents


# 定义函数generate_index_rst，生成index.rst文件的内容
def generate_index_rst(example_cases, tag_to_modules, support_level_to_modules):
    """
    Generates the index.rst file
    """

    # 初始化支持内容为空字符串
    # 遍历 support_level_to_modules 字典中的每对键值对，k 是支持级别的枚举值，v 是对应支持级别的模块列表
    for k, v in support_level_to_modules.items():
        # 获取支持级别的枚举值的小写形式，并将下划线替换为空格后转换为标题格式
        support_level = k.name.lower().replace("_", " ").title()
        # 将当前支持级别下的所有模块内容用两个换行符分隔连接成一个字符串
        module_contents = "\n\n".join(v)
        # 将格式化后的支持级别名称和模块内容追加到 support_contents 变量
        support_contents += f"""
# 定义一个字符串，表示支持级别的标题
{support_level}
# 定义一个由 '-' 组成的字符串，长度与支持级别标题相同，用于分隔线
{'-' * (len(support_level))}

# 定义一个字符串，表示模块内容的标题
{module_contents}
"""

# 生成标签名列表，每个标签名前面缩进四个空格
tag_names = "\n    ".join(t for t in tag_to_modules.keys())

# 打开文件 "blurb.txt" 并读取其中的内容作为文档的简介信息
with open(os.path.join(PWD, "blurb.txt")) as file:
    blurb = file.read()

# 生成文档的内容，包括标题、简介信息和标签的目录
doc_contents = f""".. _torch.export_db:

ExportDB
========

{blurb}

.. toctree::
    :maxdepth: 1
    :caption: Tags

    {tag_names}

{support_contents}
"""

# 将生成的文档内容写入到 "index.rst" 文件中
with open(os.path.join(EXPORTDB_SOURCE, "index.rst"), "w") as f:
    f.write(doc_contents)


def generate_tag_rst(tag_to_modules):
    """
    针对每个标签在每个 ExportCase.tag 中出现的情况，生成一个 .rst 文件，
    包含具有该标签的所有示例。
    """

    # 遍历标签到模块的映射项
    for tag, modules_rst in tag_to_modules.items():
        # 生成标签的标题和分隔线
        doc_contents = f"{tag}\n{'=' * (len(tag) + 4)}\n"
        # 将模块内容组合成一个完整的文档
        full_modules_rst = "\n\n".join(modules_rst)
        # 使用正则表达式将标题级别标识（===）替换为分隔线（---）
        full_modules_rst = re.sub(
            r"={3,}", lambda match: "-" * len(match.group()), full_modules_rst
        )
        doc_contents += full_modules_rst

        # 将生成的文档内容写入以标签命名的 .rst 文件中
        with open(os.path.join(EXPORTDB_SOURCE, f"{tag}.rst"), "w") as f:
            f.write(doc_contents)


def generate_rst():
    # 如果 EXPORTDB_SOURCE 目录不存在，则创建它
    if not os.path.exists(EXPORTDB_SOURCE):
        os.makedirs(EXPORTDB_SOURCE)

    # 获取所有示例案例
    example_cases = all_examples()
    # 初始化标签到模块的映射和支持级别到模块的映射
    tag_to_modules = {}
    support_level_to_modules = {}

    # 遍历每个示例案例
    for example_case in example_cases.values():
        # 生成示例案例的 .rst 文件内容
        doc_contents = generate_example_rst(example_case)

        # 将示例案例关联的每个标签添加到标签到模块的映射中
        for tag in example_case.tags:
            tag_to_modules.setdefault(tag, []).append(doc_contents)

        # 将示例案例关联的支持级别添加到支持级别到模块的映射中
        support_level_to_modules.setdefault(example_case.support_level, []).append(
            doc_contents
        )

    # 生成每个标签的 .rst 文件
    generate_tag_rst(tag_to_modules)
    # 生成索引文件 index.rst
    generate_index_rst(example_cases, tag_to_modules, support_level_to_modules)


if __name__ == "__main__":
    # 如果脚本作为主程序运行，则执行生成 .rst 文件的流程
    generate_rst()
```