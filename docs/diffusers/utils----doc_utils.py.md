# `.\diffusers\utils\doc_utils.py`

```py
# 版权声明，标识该文件属于 HuggingFace 团队，所有权利保留
# 
# 根据 Apache 2.0 许可协议进行许可；
# 除非遵守许可，否则不能使用此文件。
# 可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，软件在“按原样”基础上分发，
# 不提供任何明示或暗示的担保或条件。
# 请参阅许可证以获取有关权限和
# 限制的具体信息。
"""
文档工具：与文档相关的工具
"""

import re  # 导入正则表达式模块，用于字符串匹配和处理


def replace_example_docstring(example_docstring):  # 定义函数，用于替换示例文档字符串
    def docstring_decorator(fn):  # 定义装饰器函数，接受一个函数作为参数
        func_doc = fn.__doc__  # 获取传入函数的文档字符串
        lines = func_doc.split("\n")  # 将文档字符串按行分割成列表
        i = 0  # 初始化索引
        while i < len(lines) and re.search(r"^\s*Examples?:\s*$", lines[i]) is None:  # 查找“Examples:”行
            i += 1  # 移动索引，直到找到“Examples:”或超出列表长度
        if i < len(lines):  # 如果找到“Examples:”行
            lines[i] = example_docstring  # 用新示例文档字符串替换该行
            func_doc = "\n".join(lines)  # 重新组合成完整文档字符串
        else:  # 如果没有找到“Examples:”行
            raise ValueError(  # 抛出错误，提示函数文档字符串需要包含“Examples:”占位符
                f"The function {fn} should have an empty 'Examples:' in its docstring as placeholder, "
                f"current docstring is:\n{func_doc}"
            )
        fn.__doc__ = func_doc  # 更新传入函数的文档字符串
        return fn  # 返回更新后的函数

    return docstring_decorator  # 返回装饰器函数
```