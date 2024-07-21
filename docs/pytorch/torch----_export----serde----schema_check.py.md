# `.\pytorch\torch\_export\serde\schema_check.py`

```
# 使用 mypy 的 allow-untyped-defs 功能，允许不强制类型注解
mypy: allow-untyped-defs
# 导入必要的模块
import dataclasses  # 用于数据类支持
import hashlib  # 用于哈希功能
import re  # 正则表达式库
import typing  # 提供类型注解支持
from enum import IntEnum  # 枚举类型的支持
from typing import Any, Dict, Optional, Union  # 导入特定类型支持

from torch._export.serde import schema  # 导入 Torch 序列化和反序列化模块的 schema 子模块
from torch._export.serde.union import _Union  # 导入 Torch 序列化和反序列化模块的 union 子模块


class SchemaUpdateError(Exception):
    # 自定义异常类，用于处理模式更新错误
    pass


def _check(x, msg):
    # 检查条件 x 是否为真，否则抛出 SchemaUpdateError 异常，并传递错误消息 msg
    if not x:
        raise SchemaUpdateError(msg)


def _staged_schema():
    # 定义内部函数 _staged_schema，返回类型为 Dict[str, Any]
    ret: Dict[str, Any] = {}  # 初始化返回结果的字典
    defs = {}  # 初始化定义的字典

    def _handle_aggregate(ty):
        # 内部函数 _handle_aggregate 处理聚合类型 ty
        def dump_type(t):
            # 内部函数 dump_type 根据类型 t 转换为字符串表示
            if isinstance(t, type):
                return t.__name__  # 如果 t 是类型，返回其类名
            elif isinstance(t, str):
                assert t in defs
                return t  # 如果 t 是字符串，则应该在 defs 中，返回 t
            elif o := typing.get_origin(t):
                # 利用 Python 3.8 的 walrus 操作符，处理泛型类型
                if o == list:
                    head = "List"
                elif o == dict:
                    head = "Dict"
                elif o == tuple:
                    if typing.get_args(t) == ():
                        return "Tuple[()]"  # 空元组的特殊情况
                    head = "Tuple"
                elif o == Union:
                    args = typing.get_args(t)
                    assert len(args) == 2 and args[1] == type(None)
                    return f"Optional[{dump_type(args[0])}]"  # 可选类型的处理
                else:
                    raise AssertionError(f"Type {t} is not supported in export schema.")  # 不支持的类型异常
                return (
                    f"{head}[{', '.join([dump_type(x) for x in typing.get_args(t)])}]"
                )  # 组装泛型类型的字符串表示

            elif t == ():
                return "()"  # 空元组的字符串表示
            else:
                raise AssertionError(f"Type {t} is not supported in export schema.")  # 不支持的类型异常

        def dump_field(f):
            # 内部函数 dump_field 处理字段 f
            t = dump_type(f.type)  # 获取字段的类型字符串表示
            ret = {"type": t}  # 初始化字段信息的字典

            value = dataclasses.MISSING
            if f.default is not dataclasses.MISSING:
                value = f.default  # 如果字段有默认值，使用默认值
            elif f.default_factory is not dataclasses.MISSING:
                value = f.default_factory()  # 如果字段有默认工厂，调用工厂生成默认值

            if t.startswith("Optional[") and value is not None:
                raise AssertionError(
                    f"Optional field {ty.__name__}.{f.name} must have default value to be None."
                )  # 如果是可选类型但没有默认值为 None，则抛出异常

            if value is not dataclasses.MISSING:
                default = str(value)
                ret["default"] = default  # 如果存在默认值，添加到返回信息中
            return ret  # 返回字段信息的字典

        return {f.name: dump_field(f) for f in dataclasses.fields(ty)}  # 返回字段名到字段信息的字典

    def _handle_int_enum(name, ty):
        # 处理整数枚举类型
        ret[name] = {"kind": "enum", "fields": {x.name: x.value for x in ty}}  # 将枚举值和名称存储在返回结果中

    def _handle_struct(name, ty):
        # 处理结构类型
        ret[name] = {"kind": "struct", "fields": _handle_aggregate(ty)}  # 存储结构类型及其字段信息到返回结果中

    def _handle_union(name, ty):
        # 处理联合类型
        ret[name] = {"kind": "union", "fields": _handle_aggregate(ty)}  # 存储联合类型及其字段信息到返回结果中
    # 遍历给定类（schema）的所有属性名
    for name in dir(schema):
        # 如果属性名以下划线开头，则跳过
        if name.startswith("_"):
            continue

        # 获取属性值
        value = getattr(schema, name)

        # 如果属性值具有 "__module__" 属性，并且不属于当前类（schema）的模块，则跳过
        if hasattr(value, "__module__") and value.__module__ != schema.__name__:
            continue

        # 将属性名与属性值加入到字典 defs 中
        defs[name] = value

    # 遍历字典 defs 中的所有项
    for name, value in defs.items():
        # 如果值是一个类对象
        if isinstance(value, type):
            # 如果是 IntEnum 的子类，则调用 _handle_int_enum 处理
            if issubclass(value, IntEnum):
                _handle_int_enum(name, value)
            # 如果是数据类（dataclass）
            elif dataclasses.is_dataclass(value):
                # 如果是 _Union 的子类，则调用 _handle_union 处理
                if issubclass(value, _Union):
                    _handle_union(name, value)
                # 否则调用 _handle_struct 处理
                else:
                    _handle_struct(name, value)
            else:
                # 抛出异常，表示遇到未知的 schema 类型
                raise AssertionError(f"Unknown schema type {name}: {value}")
        # 如果值是整数或元组类型
        elif isinstance(value, (int, tuple)):
            # 确保属性名是 "SCHEMA_VERSION" 或 "TREESPEC_VERSION"
            assert name in ("SCHEMA_VERSION", "TREESPEC_VERSION")
        else:
            # 抛出异常，表示遇到未知的变量类型
            raise AssertionError(f"Unknown variable {name}: {value}")

    # 将 SCHEMA_VERSION 加入到返回字典 ret 中作为列表
    ret["SCHEMA_VERSION"] = list(defs["SCHEMA_VERSION"])
    # 确保 SCHEMA_VERSION 列表中的所有元素都大于 0
    assert all(x > 0 for x in ret["SCHEMA_VERSION"])
    # 将 TREESPEC_VERSION 加入到返回字典 ret 中
    ret["TREESPEC_VERSION"] = defs["TREESPEC_VERSION"]
    # 确保 TREESPEC_VERSION 大于 0
    assert ret["TREESPEC_VERSION"] > 0
    # 返回最终的返回字典 ret
    return ret
# 比较两个数据库模式的差异，并返回增加和删除的项
def _diff_schema(dst, src):
    # 从源模式中提取在目标模式中不存在的键，形成添加项字典
    additions = {key: src[key] for key in src.keys() - dst.keys()}
    # 从目标模式中提取在源模式中不存在的键，形成删除项字典
    subtractions = {key: dst[key] for key in dst.keys() - src.keys()}

    # 找到源模式和目标模式中共有的键
    common_keys = src.keys() & dst.keys()

    # 版本相关的键集合，通常不参与比较
    versions = {"SCHEMA_VERSION", "TREESPEC_VERSION"}
    common_keys -= versions

    # 遍历共有的键
    for key in common_keys:
        # 提取源模式和目标模式中的类型和字段信息
        src_kind = src[key]["kind"]
        src_fields = src[key]["fields"]
        dst_kind = dst[key]["kind"]
        dst_fields = dst[key]["fields"]
        
        # 检查类型是否相同，如果不同则抛出异常
        _check(
            src_kind == dst_kind,
            f"Type {key} changed kind from {dst_kind} to {src_kind}",
        )
        assert isinstance(src_fields, dict) and isinstance(dst_fields, dict)
        
        # 提取新增的字段和删除的字段
        added_fields = {
            key: src_fields[key] for key in src_fields.keys() - dst_fields.keys()
        }
        subtracted_fields = {
            key: dst_fields[key] for key in dst_fields.keys() - src_fields.keys()
        }
        
        # 找到共有的字段
        common_fields = src_fields.keys() & dst_fields.keys()

        # 遍历共有的字段
        for field in common_fields:
            src_field = src_fields[field]
            dst_field = dst_fields[field]
            if src_kind == "struct":
                # 检查结构体字段类型是否相同，如果不同则抛出异常
                _check(
                    src_field["type"] == dst_field["type"],
                    f"Type of the field {key}.{field} changed from {dst_field['type']} to {src_field['type']}",
                )
                # 如果源字段有默认值而目标字段没有，则将其添加到新增字段中
                if "default" in src_field and "default" not in dst_field:
                    added_fields[field] = {}
                    added_fields[field]["default"] = src_field["default"]
                # 如果目标字段有默认值而源字段没有，则将其添加到删除字段中
                if "default" not in src_field and "default" in dst_field:
                    subtracted_fields[field] = {}
                    subtracted_fields[field]["default"] = dst_field["default"]
            elif src_kind == "enum":
                # 检查枚举字段的值是否相同，如果不同则抛出异常
                _check(
                    src_field == dst_field,
                    f"Value of the enum field {key}.{field} changed from {dst_field} to {src_field}",
                )
            elif src_kind == "union":
                # 检查联合类型字段的类型是否相同，如果不同则抛出异常
                _check(
                    src_field["type"] == dst_field["type"],
                    f"Type of the field {key}.{field} changed from {dst_field['type']} to {src_field['type']}",
                )
            else:
                # 如果是未知类型，则抛出异常
                raise AssertionError(f"Unknown kind {src_kind}: {key}")
        
        # 如果有新增字段，则添加到添加项字典中
        if len(added_fields) > 0:
            assert key not in additions
            additions[key] = {}
            additions[key]["fields"] = added_fields
        
        # 如果有删除字段，则添加到删除项字典中
        if len(subtracted_fields) > 0:
            assert key not in subtractions
            subtractions[key] = {}
            subtractions[key]["fields"] = subtracted_fields

    # 返回新增项字典和删除项字典
    return additions, subtractions


# 计算给定对象的哈希值，返回其 SHA-256 散列值的十六进制表示
def _hash_schema(s):
    return hashlib.sha256(repr(s).encode("utf-8")).hexdigest()


# 用于表示版本控制系统中的提交对象，包含了计算出的结果、校验结果、路径、新增项、删除项和基础模式
@dataclasses.dataclass
class _Commit:
    result: Dict[str, Any]
    checksum_result: str
    path: str
    additions: Dict[str, Any]
    subtractions: Dict[str, Any]
    base: Dict[str, Any]
    # 声明一个可选的字符串类型变量 checksum_base
    checksum_base: Optional[str]
def update_schema():
    # 导入 importlib.resources 模块，用于访问包内资源
    import importlib.resources

    # 检查包内是否存在名为 "schema.yaml" 的资源文件
    if importlib.resources.is_resource(__package__, "schema.yaml"):
        # 读取 "schema.yaml" 文件的内容为字符串
        content = importlib.resources.read_text(__package__, "schema.yaml")
        # 在文件内容中查找匹配指定正则表达式的内容，用于提取校验和信息
        match = re.search("checksum<<([A-Fa-f0-9]{64})>>", content)
        # 检查是否成功找到校验和信息，如果未找到则抛出异常信息
        _check(match is not None, "checksum not found in schema.yaml")
        assert match is not None
        # 提取校验和信息并赋值给 checksum_base
        checksum_base = match.group(1)
        # 导入 yaml 模块，使用指定的 Loader 解析 YAML 格式的内容
        from yaml import load, Loader
        dst = load(content, Loader=Loader)
        # 断言解析后的内容是一个字典类型
        assert isinstance(dst, dict)
    else:
        # 如果未找到 "schema.yaml" 文件，则将 checksum_base 设为 None
        checksum_base = None
        # 创建一个默认的字典 dst，用于后续操作
        dst = {"SCHEMA_VERSION": None, "TREESPEC_VERSION": None}

    # 获取 _staged_schema() 返回的数据作为 src
    src = _staged_schema()
    # 比较 dst 和 src 的差异，得到增加和删除的部分
    additions, subtractions = _diff_schema(dst, src)
    # 返回一个 _Commit 对象，包括 src 数据、src 的哈希值、文件路径、增加和删除的部分、base 数据、base 的校验和信息
    return _Commit(
        result=src,
        checksum_result=_hash_schema(src),
        path=__package__.replace(".", "/") + "/schema.yaml",
        additions=additions,
        subtractions=subtractions,
        base=dst,
        checksum_base=checksum_base,
    )


def check(commit: _Commit, force_unsafe: bool = False):
    # 初始化变量
    next_version = None
    reason = ""
    
    # Step 1: 检测主要的模式更新
    if len(commit.additions) > 0:
        # 遍历 commit.additions 字典
        for k, v in commit.additions.items():
            # 如果 k 不在 commit.base 中，则跳过当前迭代
            if k not in commit.base:
                continue
            # 获取 kind 和 fields
            kind = commit.result[k]["kind"]
            fields = v["fields"]
            # 遍历 fields 字典
            for f, d in fields.items():
                # 如果 fields 中的 default 不存在，并且 kind 为 "struct"
                if "default" not in d and kind == "struct":
                    # 构建原因字符串，表示字段缺少默认值，需要进行主版本升级
                    reason += (
                        f"Field {k}.{f} is added to schema.py without a default value as an incomparible change "
                        + "which requires major version bump.\n"
                    )
                    # 设置下一个版本号为当前基础版本号加一
                    next_version = [commit.base["SCHEMA_VERSION"][0] + 1, 1]

    # 检测是否有删除操作
    if len(commit.subtractions) > 0:
        # 遍历 commit.subtractions 字典
        for k, v in commit.subtractions.items():
            # 如果 k 不在 commit.result 中，则跳过当前迭代
            if k not in commit.result:
                continue
            # 遍历 fields 列表
            for f in v["fields"]:
                # 构建原因字符串，表示字段被移除，需要进行主版本升级
                reason = f"Field {k}.{f} is removed from schema.py as an incompatible change which requires major version bump.\n"
            # 设置下一个版本号为当前基础版本号加一
            next_version = [commit.base["SCHEMA_VERSION"][0] + 1, 1]

    # 如果 force_unsafe 为 True，则添加强制标记的原因
    if force_unsafe:
        reason += "--force-unsafe is used."
        # 设置下一个版本号为当前结果版本号
        next_version = commit.result["SCHEMA_VERSION"]
    else:
        # Step 2: Detect minor schema updates.

        # 如果下一个版本号未定义且提交有添加内容
        if next_version is None and len(commit.additions) > 0:
            # 遍历每一个添加的项
            for k, v in commit.additions.items():
                # 遍历每一个字段
                for f in v["fields"]:
                    # 更新原因，说明字段被添加到 schema.py 中作为兼容性修改，但仍需要次版本号的增加
                    reason += (
                        f"Field {k}.{f} is added to schema.py as an compatible change "
                        + "which still requires minor version bump.\n"
                    )
            # 设置下一个版本号为当前基础版本号的主版本号和次版本号增加1
            next_version = [
                commit.base["SCHEMA_VERSION"][0],
                commit.base["SCHEMA_VERSION"][1] + 1,
            ]

        # 如果下一个版本号未定义且提交有移除内容
        if next_version is None and len(commit.subtractions) > 0:
            # 遍历每一个移除的项
            for k, v in commit.subtractions.items():
                # 遍历每一个字段
                for f in v["fields"]:
                    # 更新原因，说明字段从 schema.py 中移除作为兼容性修改，但仍需要次版本号的增加
                    reason += (
                        f"Field {k}.{f} is removed from schema.py as an compatible change "
                        + "which still requires minor version bump.\n"
                    )
            # 设置下一个版本号为当前基础版本号的主版本号和次版本号增加1
            next_version = [
                commit.base["SCHEMA_VERSION"][0],
                commit.base["SCHEMA_VERSION"][1] + 1,
            ]

    # 返回计算出的下一个版本号和更新原因
    return next_version, reason
```