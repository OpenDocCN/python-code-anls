# `.\graphrag\tests\unit\indexing\test_init_content.py`

```py
# 导入 re 模块，用于正则表达式操作
# 导入 Any 和 cast 用于类型提示
import re
from typing import Any, cast

# 导入 yaml 模块，用于 YAML 数据的加载和解析
import yaml

# 从 graphrag.config 中导入 GraphRagConfig 和 create_graphrag_config 函数
from graphrag.config import (
    GraphRagConfig,
    create_graphrag_config,
)

# 从 graphrag.index.init_content 导入 INIT_YAML 常量
from graphrag.index.init_content import INIT_YAML

# 定义测试函数 test_init_yaml，用于加载和验证初始化 YAML 数据
def test_init_yaml():
    # 使用 yaml.FullLoader 加载 INIT_YAML 数据
    data = yaml.load(INIT_YAML, Loader=yaml.FullLoader)
    # 使用加载的数据创建 GraphRagConfig 配置对象
    config = create_graphrag_config(data)
    # 使用 strict=True 参数验证配置对象
    GraphRagConfig.model_validate(config, strict=True)

# 定义测试函数 test_init_yaml_uncommented，用于处理未注释的初始化 YAML 数据
def test_init_yaml_uncommented():
    # 将 INIT_YAML 拆分成行
    lines = INIT_YAML.splitlines()
    # 过滤掉包含 "##" 的行，即保留未注释的行
    lines = [line for line in lines if "##" not in line]

    # 定义函数 uncomment_line，用于去除每行开头的注释符号并保留缩进
    def uncomment_line(line: str) -> str:
        # 匹配行首的空白字符，并获取其缩进
        leading_whitespace = cast(Any, re.search(r"^(\s*)", line)).group(1)
        # 去除行首的注释符号和一个空格
        return re.sub(r"^\s*# ", leading_whitespace, line, count=1)

    # 对所有未注释的行应用 uncomment_line 函数，重新构建内容
    content = "\n".join([uncomment_line(line) for line in lines])
    # 使用 yaml.FullLoader 加载修改后的 YAML 数据
    data = yaml.load(content, Loader=yaml.FullLoader)
    # 使用加载的数据创建 GraphRagConfig 配置对象
    config = create_graphrag_config(data)
    # 使用 strict=True 参数验证配置对象
    GraphRagConfig.model_validate(config, strict=True)
```