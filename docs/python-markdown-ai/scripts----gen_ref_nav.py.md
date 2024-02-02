# `markdown\scripts\gen_ref_nav.py`

```py

"""Generate the code reference pages and navigation."""

# 导入所需的模块
import textwrap
import yaml
from pathlib import Path
import mkdocs_gen_files

# 创建导航对象
nav = mkdocs_gen_files.Nav()

# 设置每个模块的选项
per_module_options = {
    "markdown": {"summary": {"attributes": True, "functions": True, "classes": True}}
}

# 获取基础路径
base_path = Path(__file__).resolve().parent.parent

# 定义需要生成文档的模块列表
modules = [
    # 列出所有模块的路径
]

# 遍历每个模块
for src_path in modules:
    # 处理路径
    # 生成文档路径
    # 生成完整文档路径

    # 处理模块路径
    # 如果模块以 "__init__" 结尾，则修改路径和文档路径
    # 如果模块以 "_" 开头，则跳过

    # 生成导航路径
    # 写入文档内容
    # 设置编辑路径

# 写入导航文件

```