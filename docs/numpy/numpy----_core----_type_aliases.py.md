# `.\numpy\numpy\_core\_type_aliases.py`

```
"""
Due to compatibility, numpy has a very large number of different naming
conventions for the scalar types (those subclassing from `numpy.generic`).
This file produces a convoluted set of dictionaries mapping names to types,
and sometimes other mappings too.

.. data:: allTypes
    A dictionary of names to types that will be exposed as attributes through
    ``np._core.numerictypes.*``

.. data:: sctypeDict
    Similar to `allTypes`, but maps a broader set of aliases to their types.

.. data:: sctypes
    A dictionary keyed by a "type group" string, providing a list of types
    under that group.

"""

import numpy._core.multiarray as ma  # 导入numpy的multiarray模块，并简称为ma
from numpy._core.multiarray import typeinfo, dtype  # 从numpy的multiarray模块导入typeinfo和dtype

######################################
# Building `sctypeDict` and `allTypes`
######################################

sctypeDict = {}  # 初始化空字典sctypeDict，用于存储类型别名到类型的映射
allTypes = {}    # 初始化空字典allTypes，用于存储类型名称到类型的映射
c_names_dict = {}  # 初始化空字典c_names_dict，用于存储C语言类型名称到类型的映射

_abstract_type_names = {  # 抽象类型名称的集合
    "generic", "integer", "inexact", "floating", "number",
    "flexible", "character", "complexfloating", "unsignedinteger",
    "signedinteger"
}

for _abstract_type_name in _abstract_type_names:
    allTypes[_abstract_type_name] = getattr(ma, _abstract_type_name)  # 将抽象类型名映射为numpy.ma中对应的类型对象

for k, v in typeinfo.items():
    if k.startswith("NPY_") and v not in c_names_dict:
        c_names_dict[k[4:]] = v  # 将去掉前缀"NPY_"后的名称作为键，类型对象作为值，存入c_names_dict
    else:
        concrete_type = v.type
        allTypes[k] = concrete_type  # 将typeinfo中的键（类型名称）映射为其对应的具体类型
        sctypeDict[k] = concrete_type  # 同时将其映射存入sctypeDict

_aliases = {
    "double": "float64",
    "cdouble": "complex128",
    "single": "float32",
    "csingle": "complex64",
    "half": "float16",
    "bool_": "bool",
    # Default integer:
    "int_": "intp",
    "uint": "uintp",
}

for k, v in _aliases.items():
    sctypeDict[k] = allTypes[v]  # 将_aliases中的键（类型别名）映射为allTypes中对应的类型，并存入sctypeDict
    allTypes[k] = allTypes[v]    # 同时将其映射存入allTypes

# extra aliases are added only to `sctypeDict`
# to support dtype name access, such as`np.dtype("float")`
_extra_aliases = {  
    "float": "float64",
    "complex": "complex128",
    "object": "object_",
    "bytes": "bytes_",
    "a": "bytes_",
    "int": "int_",
    "str": "str_",
    "unicode": "str_",
}

for k, v in _extra_aliases.items():
    sctypeDict[k] = allTypes[v]  # 将_extra_aliases中的键（类型别名）映射为allTypes中对应的类型，并存入sctypeDict

# include extended precision sized aliases
for is_complex, full_name in [(False, "longdouble"), (True, "clongdouble")]:
    longdouble_type: type = allTypes[full_name]

    bits: int = dtype(longdouble_type).itemsize * 8
    base_name: str = "complex" if is_complex else "float"
    extended_prec_name: str = f"{base_name}{bits}"
    if extended_prec_name not in allTypes:
        sctypeDict[extended_prec_name] = longdouble_type  # 将扩展精度类型名称映射为longdouble_type，并存入sctypeDict
        allTypes[extended_prec_name] = longdouble_type    # 同时将其映射存入allTypes


####################
# Building `sctypes`
####################

sctypes = {"int": set(), "uint": set(), "float": set(),
           "complex": set(), "others": set()}  # 初始化sctypes字典，包含各类类型的空集合

for type_info in typeinfo.values():
    if type_info.kind in ["M", "m"]:  # 排除timedelta和datetime类型
        continue

    concrete_type = type_info.type

    # find proper group for each concrete type
    # 为每种具体类型找到适当的类型组
    # 对于每个类型分组和抽象类型的组合，依次进行迭代
    for type_group, abstract_type in [
        ("int", ma.signedinteger), ("uint", ma.unsignedinteger), 
        ("float", ma.floating), ("complex", ma.complexfloating), 
        ("others", ma.generic)
    ]:
        # 检查具体类型是否是抽象类型的子类
        if issubclass(concrete_type, abstract_type):
            # 如果是，则将具体类型添加到相应类型分组的集合中
            sctypes[type_group].add(concrete_type)
            # 一旦找到匹配，就退出循环
            break
# 对 sctype groups 按照位大小进行排序

# 遍历 sctypes 字典的键（sctype_key），表示不同的数据类型组
for sctype_key in sctypes.keys():
    # 将 sctype_key 对应的值转换为列表，并进行排序
    sctype_list = list(sctypes[sctype_key])
    # 使用 lambda 函数作为排序的关键字，按照 dtype(x).itemsize 的大小排序
    sctype_list.sort(key=lambda x: dtype(x).itemsize)
    # 更新 sctype_key 对应的值为排序后的列表
    sctypes[sctype_key] = sctype_list
```