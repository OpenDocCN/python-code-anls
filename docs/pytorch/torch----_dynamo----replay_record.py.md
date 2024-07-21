# `.\pytorch\torch\_dynamo\replay_record.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import dataclasses  # 导入用于数据类的模块
from dataclasses import field  # 从数据类模块中导入字段函数
from types import CodeType, ModuleType  # 导入 CodeType 和 ModuleType 类型
from typing import Any, Dict  # 导入 Any 和 Dict 类型

from torch.utils._import_utils import import_dill  # 从 torch.utils._import_utils 模块导入 import_dill 函数

dill = import_dill()  # 调用 import_dill 函数并将其结果赋给 dill 变量


@dataclasses.dataclass
class ModuleRecord:
    module: ModuleType  # 类型注解，module 是一个 ModuleType 类的实例
    accessed_attrs: Dict[str, Any] = field(default_factory=dict)  # 类型注解，accessed_attrs 是一个字典，键为 str，值为 Any


@dataclasses.dataclass
class DummyModule:
    name: str  # 类型注解，name 是一个字符串
    is_torch: bool = False  # 类型注解，is_torch 是一个布尔值，默认为 False

    @property
    def __name__(self):
        return self.name  # 返回实例的 name 属性作为 __name__ 属性的值


@dataclasses.dataclass
class ExecutionRecord:
    code: CodeType  # 类型注解，code 是一个 CodeType 类的实例
    globals: Dict[str, Any] = field(default_factory=dict)  # 类型注解，globals 是一个字典，键为 str，值为 Any
    locals: Dict[str, Any] = field(default_factory=dict)  # 类型注解，locals 是一个字典，键为 str，值为 Any
    builtins: Dict[str, Any] = field(default_factory=dict)  # 类型注解，builtins 是一个字典，键为 str，值为 Any
    code_options: Dict[str, Any] = field(default_factory=dict)  # 类型注解，code_options 是一个字典，键为 str，值为 Any

    def dump(self, f):
        assert dill is not None, "replay_record requires `pip install dill`"  # 断言语句，确保 dill 已导入
        dill.dump(self, f)  # 使用 dill 序列化当前实例到文件 f 中

    @classmethod
    def load(cls, f):
        assert dill is not None, "replay_record requires `pip install dill`"  # 断言语句，确保 dill 已导入
        return dill.load(f)  # 使用 dill 从文件 f 中反序列化并返回一个实例


@dataclasses.dataclass
class ExecutionRecorder:
    LOCAL_MOD_PREFIX = "___local_mod_"

    code: CodeType  # 类型注解，code 是一个 CodeType 类的实例
    globals: Dict[str, Any] = field(default_factory=dict)  # 类型注解，globals 是一个字典，键为 str，值为 Any
    locals: Dict[str, Any] = field(default_factory=dict)  # 类型注解，locals 是一个字典，键为 str，值为 Any
    builtins: Dict[str, Any] = field(default_factory=dict)  # 类型注解，builtins 是一个字典，键为 str，值为 Any
    code_options: Dict[str, Any] = field(default_factory=dict)  # 类型注解，code_options 是一个字典，键为 str，值为 Any
    name_to_modrec: Dict[str, Any] = field(default_factory=dict)  # 类型注解，name_to_modrec 是一个字典，键为 str，值为 Any

    def add_local_var(self, name, var):
        if isinstance(var, ModuleType):  # 检查 var 是否是 ModuleType 类的实例
            self.locals[name] = self._add_mod(var)  # 如果是，将 var 添加到 locals 字典中，并使用 _add_mod 方法处理
        else:
            self.locals[name] = var  # 否则，直接将 var 添加到 locals 字典中

    def add_global_var(self, name, var):
        if isinstance(var, ModuleType):  # 检查 var 是否是 ModuleType 类的实例
            self.globals[name] = self._add_mod(var)  # 如果是，将 var 添加到 globals 字典中，并使用 _add_mod 方法处理
        else:
            self.globals[name] = var  # 否则，直接将 var 添加到 globals 字典中

    def add_local_mod(self, name, mod):
        assert isinstance(mod, ModuleType)  # 断言语句，确保 mod 是 ModuleType 类的实例

        self.add_global_var(name, mod)  # 调用 add_global_var 方法，将 mod 添加到 globals 字典中

    def record_module_access(self, mod, name, val):
        if isinstance(val, ModuleType):  # 检查 val 是否是 ModuleType 类的实例
            self.name_to_modrec[mod.__name__].accessed_attrs[name] = self._add_mod(val)  # 如果是，将 val 添加到 name_to_modrec 字典中对应模块的 accessed_attrs 字典中，并使用 _add_mod 方法处理
            return

        if mod.__name__ in self.name_to_modrec:
            self.name_to_modrec[mod.__name__].accessed_attrs[name] = val  # 将 val 添加到 name_to_modrec 字典中对应模块的 accessed_attrs 字典中

    def get_record(self):
        return ExecutionRecord(
            self.code,
            ExecutionRecorder._resolve_modules(self.globals),  # 调用 _resolve_modules 方法，处理并返回 globals 字典中的 ModuleRecord 实例
            ExecutionRecorder._resolve_modules(self.locals),  # 调用 _resolve_modules 方法，处理并返回 locals 字典中的 ModuleRecord 实例
            self.builtins.copy(),  # 复制并返回 builtins 字典
            self.code_options.copy(),  # 复制并返回 code_options 字典
        )

    def _add_mod(self, mod):
        if mod.__name__ not in self.name_to_modrec:  # 检查 mod.__name__ 是否已存在于 name_to_modrec 字典中
            self.name_to_modrec[mod.__name__] = ModuleRecord(mod)  # 如果不存在，将 mod 添加为 ModuleRecord 实例到 name_to_modrec 字典中

        return self.name_to_modrec[mod.__name__]  # 返回 name_to_modrec 字典中 mod.__name__ 对应的 ModuleRecord 实例

    # Convert ModuleRecords -> DummyModule tree
    @classmethod
    # 解析模块变量，替换为虚拟模块对象
    def _resolve_modules(cls, vars):
        # 内部函数：解析单个模块变量，如果不是ModuleRecord类型则直接返回，否则创建DummyModule对象
        def resolve_module(var):
            if not isinstance(var, ModuleRecord):
                return var
            
            # 创建一个虚拟模块对象，名称为原始模块的名称
            dummy_mod = DummyModule(var.module.__name__)
            
            # 遍历模块中访问的属性，递归解析每个属性的值，将其赋值给虚拟模块对象
            for attr_name, attr_value in var.accessed_attrs.items():
                attr_value = resolve_module(attr_value)
                dummy_mod.__setattr__(attr_name, attr_value)
            
            return dummy_mod

        # 对输入的vars字典中的每个键值对应用resolve_module函数进行解析，并返回结果字典
        return {k: resolve_module(v) for k, v in vars.items()}
```