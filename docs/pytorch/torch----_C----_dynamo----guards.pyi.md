# `.\pytorch\torch\_C\_dynamo\guards.pyi`

```
# mypy: allow-untyped-defs
# 引入必要的模块和类型声明
from typing import Any

# 导入 PyTorch 库
import torch

# 定义全局状态守卫类，包含检查和原因方法的声明
class GlobalStateGuard:
    def check(self) -> bool: ...
    def reason(self) -> str: ...

# 空定义 LeafGuard 类
class LeafGuard: ...

# 空定义 GuardDebugInfo 类
class GuardDebugInfo: ...

# 定义守卫管理器类，包含检查和详细检查方法的声明
class GuardManager:
    def check(self, value) -> bool: ...
    def check_verbose(self, value) -> GuardDebugInfo: ...

    # 访问器方法
    def globals_dict_manager(
        self,
        f_globals: dict[str, Any],
        source,
        example_value,
        guard_manager_enum,
    ) -> GuardManager: ...
    def dict_getitem_manager(
        self,
        key,
        source,
        example_value,
        guard_manager_enum,
    ) -> GuardManager: ...
    def global_weakref_manager(
        self,
        global_name: str,
        source,
        example_value,
        guard_manager_enum,
    ) -> GuardManager: ...
    def type_manager(
        self,
        source,
        example_value,
        guard_manager_enum,
    ) -> GuardManager: ...
    def getattr_manager(
        self,
        attr: str,
        source,
        example_value,
        guard_manager_enum,
    ) -> GuardManager: ...
    def lambda_manager(
        self,
        python_lambda,
        source,
        example_value,
        guard_manager_enum,
    ) -> GuardManager: ...

    # Leaf guards 方法，用于添加特定类型的守卫
    def add_lambda_guard(self, user_lambda, verbose_code_parts: list[str]) -> None: ...
    def add_id_match_guard(self, id_val, verbose_code_parts: list[str]) -> None: ...
    def add_equals_match_guard(
        self,
        equals_val,
        verbose_code_parts: list[str],
    ) -> None: ...
    def add_global_state_guard(self, verbose_code_parts: list[str]) -> None: ...

# 继承自 GuardManager 的根守卫管理器类，包含获取结尾 Lambda 守卫和添加结尾 Lambda 守卫的方法声明
class RootGuardManager(GuardManager):
    def get_epilogue_lambda_guards(self) -> list[LeafGuard]: ...
    def add_epilogue_lambda_guard(
        self,
        guard: LeafGuard,
        verbose_code_parts: list[str],
    ) -> None: ...

# 继承自 GuardManager 的字典守卫管理器类，包含获取键管理器和值管理器的方法声明
class DictGuardManager(GuardManager):
    def get_key_manager(
        self,
        index,
        source,
        example_value,
        guard_manager_enum,
    ) -> GuardManager: ...
    def get_value_manager(
        self,
        index,
        source,
        example_value,
        guard_manager_enum,
    ) -> GuardManager: ...

# 安装张量别名守卫函数声明，接受守卫管理器列表、张量名列表和详细代码部分列表作为参数
def install_tensor_aliasing_guard(
    guard_managers: list[GuardManager],
    tensor_names: list[str],
    verbose_code_parts: list[str],
): ...

# 安装无张量别名守卫函数声明，接受守卫管理器列表、张量名列表和详细代码部分列表作为参数
def install_no_tensor_aliasing_guard(
    guard_managers: list[GuardManager],
    tensor_names: list[str],
    verbose_code_parts: list[str],
): ...

# 张量守卫类，包含动态维度大小和步长列表的初始化声明，支持检查和详细检查方法
class TensorGuards:
    def __init__(
        self,
        *,
        dynamic_dims_sizes: list[torch.SymInt | None] | None = None,
        dynamic_dims_strides: list[torch.SymInt | None] | None = None,
    ) -> None: ...
    def check(self, *args) -> bool: ...
    def check_verbose(self, *args, tensor_check_names=None) -> bool | str: ...

# 断言张量大小和步长函数声明，接受张量项、大小和步长作为参数
def assert_size_stride(
    item: torch.Tensor,
    size: torch.types._size,
    stride: torch.types._size,
): ...
# 检查对象的标识符是否等于预期值，返回布尔类型结果
def check_obj_id(obj: object, expected: int) -> bool:
    ...

# 检查对象的类型标识符是否等于预期值，返回布尔类型结果
def check_type_id(obj: object, expected: int) -> bool:
    ...

# 返回字典的版本号，即字典的长度
def dict_version(d: dict[Any, Any]) -> int:
    ...
```