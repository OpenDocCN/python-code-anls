# `.\pytorch\torch\jit\generate_bytecode.py`

```
# mypy: allow-untyped-defs
# 导入所需的模块和函数
from typing import List
from torch._C import _compile_graph_to_code_table, _generate_upgraders_graph

# 将给定的嵌套元组转换为嵌套列表
def format_bytecode(table):
    def listify(content):
        if not isinstance(content, tuple):
            return content
        return [listify(i) for i in content]
    
    # 初始化格式化后的表格字典
    formatted_table = {}
    # 遍历输入的表格
    for entry in table:
        # 获取表格条目的标识符和内容
        identifier = entry[0]
        content = entry[1]
        # 将内容转换为嵌套列表格式
        content = listify(content)
        # 将标识符及其格式化后的内容存入字典
        formatted_table[identifier] = content
    return formatted_table

# 生成升级器的字节码
def generate_upgraders_bytecode() -> List:
    # 初始化空列表以存储生成的 YAML 内容
    yaml_content = []
    # 获取升级器图谱的字典
    upgraders_graph_map = _generate_upgraders_graph()
    # 遍历每个升级器名称及其对应的图谱
    for upgrader_name, upgrader_graph in upgraders_graph_map.items():
        # 编译升级器图谱为代码表
        bytecode_table = _compile_graph_to_code_table(upgrader_name, upgrader_graph)
        # 格式化字节码表并构建条目字典
        entry = {upgrader_name: format_bytecode(bytecode_table)}
        # 将条目字典添加到 YAML 内容列表中
        yaml_content.append(entry)
    # 返回最终生成的 YAML 内容列表
    return yaml_content

# 如果脚本被直接执行，则引发运行时错误
if __name__ == "__main__":
    raise RuntimeError("This file is not meant to be run directly")
```