# `.\pytorch\torch\jit\unsupported_tensor_ops.py`

```
# mypy: allow-untyped-defs
# 引入 textwrap 模块中的 dedent 函数，用于去除文本的缩进
from textwrap import dedent

# 引入 Any 和 Dict 类型，用于类型提示
from typing import Any, Dict

# 引入 torch.jit 模块，用于 JIT 编译
import torch.jit


# 定义一个函数 execWrapper，用于执行给定的代码块
def execWrapper(code, glob, loc):
    exec(code, glob, loc)


# 定义一个内部函数 _gen_unsupported_methods_properties，用于生成不支持的 Tensor 方法和属性
def _gen_unsupported_methods_properties():
    # 获取所有非私有属性的集合
    tensor_attrs = set(filter(lambda x: x[0] != "_", dir(torch.Tensor)))
    
    # 创建一个临时的 Tensor 对象
    tensor = torch.tensor([2])
    
    # 定义包含函数模板的字符串，用于生成每个属性对应的函数
    funcs_template = dedent(
        """
    def func(x):
        return x.{op}()
    """
    )
    
    # 定义已弃用的 API 集合
    deprecated_apis = {
        "volatile",
        "resize",
        "reinforce",
        "new",
        "name",
        "map2_",
        "has_names",
        "grad_fn",
        "resize_as",
    }
    
    # 排除已弃用的 API
    tensor_attrs = tensor_attrs - deprecated_apis

    # 初始化方法和属性列表
    properties = []
    methods = []
    
    # 按属性名字母顺序排序属性集合
    sorted_tensor_attrs = sorted(tensor_attrs, key=lambda x: x.lower())
    
    # 遍历每个属性名
    for attr in sorted_tensor_attrs:
        # 使用函数模板生成对应属性的函数字符串
        funcs_str = funcs_template.format(op=attr)
        
        # 创建一个空的作用域字典
        scope: Dict[str, Any] = {}
        
        # 执行生成的函数字符串，将函数定义添加到作用域中
        execWrapper(funcs_str, globals(), scope)
        
        # 尝试使用 Torch 的 JIT 编译单元来编译生成的函数字符串
        try:
            cu = torch.jit.CompilationUnit(funcs_str)
        except Exception as e:
            # 如果编译过程中出现异常，并且异常信息中不包含"nonexistent attribute"，则跳过该属性
            if "nonexistent attribute" not in repr(e):
                continue
            
            # 获取属性的字符串表示
            attr_repr = repr(getattr(tensor, attr))
            
            # 如果属性表示中包含"bound method"或"built-in method"，则将其视为方法，否则视为属性
            if "bound method" in attr_repr or "built-in method" in attr_repr:
                methods.append(attr)
            else:
                properties.append(attr)

    # 将方法和属性列表中的每个项映射为特定格式的字符串，并用换行符连接起来
    mapped_methods = ("\t*  :meth:`~torch.Tensor." + x + r"`" for x in methods)
    mapped_properties = ("\t*  :attr:`~torch.Tensor." + x + r"`" for x in properties)
    
    # 返回映射后的方法和属性字符串
    return "\n".join(mapped_methods), "\n".join(mapped_properties)


# 定义内部函数 _list_unsupported_tensor_ops，用于列出不支持的 Tensor 操作
def _list_unsupported_tensor_ops():
    # 定义标题字符串，标明不支持的 Tensor 方法和属性
    header = """\n\n
Unsupported Tensor Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    
    # 调用 _gen_unsupported_methods_properties 函数获取不支持的方法和属性的字符串表示
    methods, properties = _gen_unsupported_methods_properties()
    
    # 返回格式化后的不支持的 Tensor 方法和属性的字符串
    return (
        header
        + "\n"
        + methods
        + """

Unsupported Tensor Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
        + "\n"
        + properties
    )


# 将 __doc__ 文档字符串设置为列出不支持的 Tensor 操作的结果
__doc__ = _list_unsupported_tensor_ops()
```