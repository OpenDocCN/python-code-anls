# `D:\src\scipysrc\scipy\scipy\sparse\extract.py`

```
# 此文件不是公共使用的，将在 SciPy v2.0.0 中移除。
# 使用 `scipy.sparse` 命名空间来导入下面列出的函数。

# 导入 _sub_module_deprecation 函数，用于处理子模块的废弃警告
from scipy._lib.deprecation import _sub_module_deprecation

# 声明 __all__ 列表，指定了将会公开的模块成员，忽略 F822 类型的警告
__all__ = [
    'coo_matrix',  # 坐标格式稀疏矩阵的构造函数
    'find',        # 在稀疏矩阵中查找非零元素的位置函数
    'tril',        # 返回稀疏矩阵的下三角形式
    'triu',        # 返回稀疏矩阵的上三角形式
]

# 自定义 __dir__() 函数，返回模块的公开成员列表
def __dir__():
    return __all__

# 自定义 __getattr__(name) 函数，用于获取未定义的模块成员
def __getattr__(name):
    return _sub_module_deprecation(
        sub_package="sparse",  # 废弃警告所属的子包名称
        module="extract",       # 废弃警告所属的模块名称
        private_modules=["_extract"],  # 废弃警告中提到的私有模块列表
        all=__all__,           # 当前模块允许的所有成员列表
        attribute=name         # 请求的属性名称
    )
```