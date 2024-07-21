# `.\pytorch\torch\package\_package_pickler.py`

```
# mypy: allow-untyped-defs
# 导入 pickle 模块的相关组件，其中 type: ignore[attr-defined] 表示忽略类型检查
from pickle import (
    _compat_pickle,
    _extension_registry,
    _getattribute,
    _Pickler,
    EXT1,
    EXT2,
    EXT4,
    GLOBAL,
    Pickler,
    PicklingError,
    STACK_GLOBAL,
)
# 导入 struct 模块的 pack 函数
from struct import pack
# 导入 types 模块的 FunctionType 类型
from types import FunctionType

# 从当前包中导入 Importer 类及相关异常和函数
from .importer import Importer, ObjMismatchError, ObjNotFoundError, sys_importer

# 定义 PackagePickler 类，继承自 _Pickler 类
class PackagePickler(_Pickler):
    """Package-aware pickler.

    This behaves the same as a normal pickler, except it uses an `Importer`
    to find objects and modules to save.
    """

    # 初始化方法，接受一个 Importer 实例作为参数
    def __init__(self, importer: Importer, *args, **kwargs):
        self.importer = importer
        # 调用父类 _Pickler 的初始化方法
        super().__init__(*args, **kwargs)

        # 确保从 _Pickler 中复制的分发表是最新的
        # 曾经遇到过的问题是某些库（例如 dill）会改变 _Pickler.dispatch，
        # PackagePickler 在导入时复制了一份，然后该库移除了它的分发条目，
        # 这样会导致 PackagePickler 拥有一个可能导致不良行为的过时分发表。
        self.dispatch = _Pickler.dispatch.copy()  # type: ignore[misc]
        # 将 FunctionType 类型的对象保存函数绑定到自定义的 save_global 方法上
        self.dispatch[FunctionType] = PackagePickler.save_global  # type: ignore[assignment]
    # 定义一个保存全局变量的方法，接受一个对象和可选的变量名作为参数
    def save_global(self, obj, name=None):
        # 不幸的是，pickler 代码的结构迫使我们复制/粘贴这个函数。
        # 唯一的更改标记为 CHANGED 下面。
        write = self.write  # type: ignore[attr-defined]
        memo = self.memo  # type: ignore[attr-defined]

        # CHANGED: 使用导入器从模块环境中导入模块而不是使用 __import__
        try:
            # 获取对象的模块名和名称
            module_name, name = self.importer.get_name(obj, name)
        except (ObjNotFoundError, ObjMismatchError) as err:
            # 抛出无法 pickle 对象的异常，包括错误信息
            raise PicklingError(f"Can't pickle {obj}: {str(err)}") from None

        # 导入指定模块
        module = self.importer.import_module(module_name)
        # 获取指定模块和名称的属性
        _, parent = _getattribute(module, name)
        # END CHANGED

        # 如果协议版本大于等于 2，则处理扩展注册表
        if self.proto >= 2:  # type: ignore[attr-defined]
            code = _extension_registry.get((module_name, name))
            if code:
                assert code > 0
                # 根据代码大小选择合适的扩展写入操作码
                if code <= 0xFF:
                    write(EXT1 + pack("<B", code))
                elif code <= 0xFFFF:
                    write(EXT2 + pack("<H", code))
                else:
                    write(EXT4 + pack("<i", code))
                return
        
        # 获取最后一个名称部分
        lastname = name.rpartition(".")[2]
        # 如果父对象是模块本身，则使用最后一个名称部分作为名称
        if parent is module:
            name = lastname

        # 如果协议版本大于等于 4，则使用 STACK_GLOBAL 写入操作码
        if self.proto >= 4:  # type: ignore[attr-defined]
            self.save(module_name)  # type: ignore[attr-defined]
            self.save(name)  # type: ignore[attr-defined]
            write(STACK_GLOBAL)
        elif parent is not module:
            # 如果父对象不是模块本身，则使用 save_reduce 方法写入 reduce 操作
            self.save_reduce(getattr, (parent, lastname))  # type: ignore[attr-defined]
        elif self.proto >= 3:  # type: ignore[attr-defined]
            # 如果协议版本大于等于 3，则写入全局操作码及模块名和名称的字节串
            write(
                GLOBAL
                + bytes(module_name, "utf-8")
                + b"\n"
                + bytes(name, "utf-8")
                + b"\n"
            )
        else:
            # 否则，如果协议版本小于 3，并且需要修复导入，则根据协议版本选择适当的编码方式写入全局操作码
            if self.fix_imports:  # type: ignore[attr-defined]
                r_name_mapping = _compat_pickle.REVERSE_NAME_MAPPING
                r_import_mapping = _compat_pickle.REVERSE_IMPORT_MAPPING
                if (module_name, name) in r_name_mapping:
                    module_name, name = r_name_mapping[(module_name, name)]
                elif module_name in r_import_mapping:
                    module_name = r_import_mapping[module_name]
            try:
                # 尝试使用 ASCII 编码方式写入全局操作码及模块名和名称的字节串
                write(
                    GLOBAL
                    + bytes(module_name, "ascii")
                    + b"\n"
                    + bytes(name, "ascii")
                    + b"\n"
                )
            except UnicodeEncodeError:
                # 如果编码失败，则抛出无法 pickle 全局标识符的异常，包括错误信息和协议版本
                raise PicklingError(
                    "can't pickle global identifier '%s.%s' using "
                    "pickle protocol %i" % (module, name, self.proto)  # type: ignore[attr-defined]
                ) from None

        # 将对象加入 memo 缓存
        self.memoize(obj)  # type: ignore[attr-defined]
def create_pickler(data_buf, importer, protocol=4):
    # 检查导入器是否是系统默认的导入器
    if importer is sys_importer:
        # 如果我们正在使用正常的导入库系统，
        # 则可以使用pickle的C实现，这更快
        return Pickler(data_buf, protocol=protocol)
    else:
        # 如果不是系统默认的导入器，则使用PackagePickler
        return PackagePickler(importer, data_buf, protocol=protocol)
```