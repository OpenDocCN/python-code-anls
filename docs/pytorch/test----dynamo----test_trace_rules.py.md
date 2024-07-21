# `.\pytorch\test\dynamo\test_trace_rules.py`

```py
# Owner(s): ["module: dynamo"]

# 导入必要的模块和库
import dataclasses        # 用于数据类的装饰器，简化定义数据类的代码
import importlib         # 提供对 Python 动态加载模块的支持
import inspect           # 提供用于检查类和函数定义的工具
import math              # 提供数学函数库
import types             # 提供动态创建和操作 Python 类型的工具
import unittest          # 提供单元测试框架
import warnings          # 提供警告管理功能
from typing import Any, Dict, Set   # 引入类型提示相关的模块

import torch             # 引入 PyTorch 深度学习框架
import torch._dynamo.config as config          # 引入 PyTorch 内部的配置模块
import torch._dynamo.test_case                 # 引入 PyTorch 内部的测试用例模块
import torch._functorch.deprecated as deprecated_func  # 引入 PyTorch 内部的废弃功能模块
from torch._dynamo.trace_rules import (        # 从 trace_rules 模块中导入多个符号
    LEGACY_MOD_INLINELIST,                    # 引入旧模块的内联列表
    load_object,                              # 导入加载对象的函数
    manual_torch_name_rule_map,               # 引入手动规则映射
    MOD_INLINELIST,                           # 引入模块的内联列表
    torch_c_binding_in_graph_functions,       # 引入在图函数中使用 C 绑定的 Torch 函数
    torch_non_c_binding_in_graph_functions   # 引入在图函数中使用非 C 绑定的 Torch 函数
)
from torch._dynamo.utils import hashable, is_safe_constant, istype  # 从 utils 模块中导入多个实用函数
from torch._dynamo.variables import (         # 从 variables 模块中导入多个变量类
    TorchInGraphFunctionVariable,              # 引入 Torch 在图函数变量类
    UserFunctionVariable                      # 引入用户函数变量类
)

try:
    from .utils import create_dummy_module_and_function   # 尝试从当前包中导入创建虚拟模块和函数的工具函数
except ImportError:
    from utils import create_dummy_module_and_function   # 如果导入失败，则从全局中导入该工具函数

# 定义忽略的在图函数中使用 C 绑定的 Torch 函数名称集合
ignored_c_binding_in_graph_function_names = {
    # 下面的函数因为在 `trace_rules.manual_torch_name_rule_map` 中有手动规则定义，所以被忽略
    "torch._nested_tensor_from_mask",
    "torch._nested_from_padded",
    "torch.sparse_compressed_tensor",
    "torch.sparse_bsc_tensor",
    "torch.sparse_bsr_tensor",
    "torch.sparse_coo_tensor",
    "torch.sparse_csc_tensor",
    "torch.sparse_csr_tensor",
    "torch.cuda._get_device_properties",
    # 下面的函数通过在 `trace_rules.check` 中定义的规则被忽略
    "torch._functionalize_are_all_mutations_under_no_grad_or_inference_mode",
    "torch._cslt_sparse_mm_search",
    "torch._C._abort",
    "torch._C._mps_is_on_macos_or_newer",
    "torch._C._swap_tensor_impl",
    "torch._C._unsafe_reset_storage",
    "torch._dynamo.eval_frame.reset_code",
    "torch._C.autocast_decrement_nesting",
    "torch._C.autocast_increment_nesting",
    "torch._C.clear_autocast_cache",
    "torch._C.set_anomaly_enabled",
    "torch._C.set_autocast_cache_enabled",
    "torch._C.set_autocast_cpu_dtype",
    "torch._C.set_autocast_cpu_enabled",
    "torch._C.set_autocast_enabled",
    "torch._C.set_autocast_gpu_dtype",
    "torch._C.set_autocast_ipu_dtype",
    "torch._C.set_autocast_ipu_enabled",
    "torch._C.set_autocast_xla_dtype",
    "torch._C.set_autocast_xla_enabled",
    "torch.resize_as_",
    "torch.resize_as_sparse_",
    "torch._C._data_address",
    "torch._C._is_cow_tensor",
    "torch._lazy_clone",
    "torch._test_parallel_materialize",
    "torch._C._storage_address",
    "torch._C._pickle_save",
    "torch._validate_sparse_compressed_tensor_args",
    "torch._validate_sparse_csr_tensor_args",
    "torch._validate_sparse_bsr_tensor_args",
    "torch._validate_sparse_csc_tensor_args",
    "torch._validate_sparse_coo_tensor_args",
    "torch._validate_sparse_bsc_tensor_args",
    "torch._validate_compressed_sparse_indices",
}

# 如果 LLVM 被启用，添加另外一些在图函数中使用 C 绑定的 Torch 函数名称到忽略集合中
if torch._C._llvm_enabled():
    ignored_c_binding_in_graph_function_names |= {
        "torch._C._te.set_llvm_aot_workflow",
        "torch._C._te.set_llvm_target_cpu",
        "torch._C._te.set_llvm_target_attrs",
        "torch._C._te.set_llvm_target_triple",
    }
    }
`
    }
# Helper function to dump the torch name rule map generated based on
# the heuristic defined in gen_allowed_objs_and_ids.
def dump_allowed_torch_name_rule_map() -> None:
    # 生成允许对象及其标识符的映射，基于 gen_allowed_objs_and_ids 中定义的启发式规则
    m = gen_allowed_objs_and_ids(record=True, c_binding_only=False).name_rule_map
    # 遍历映射中的每一项，打印对象名和对应的变量名称
    for k, v in m.items():
        print(f'"{k}": {v.__name__},')


@dataclasses.dataclass
class AllowedObjects:
    """
    Track the objects, object id - name pairs, and name - dynamo wrapping rule pairs
    from the heuristic defined in `gen_allowed_objs_and_ids`.
    """
    # 对象标识符与名称的映射字典
    object_ids: Dict[int, str]
    # 在图中使用 C 绑定的函数集合
    c_binding_in_graph_functions: Set[Any]
    # 在图中使用非 C 绑定的函数集合
    non_c_binding_in_graph_functions: Set[Any]
    # 名称到 Torch 动态包装规则的映射字典
    name_rule_map: Dict[str, Any]


def gen_allowed_objs_and_ids(record=False, c_binding_only=True) -> AllowedObjects:
    """
    Walk torch.* and get the ids of all the stuff in it
    """
    # 忽略 torch.distributed 模块中的 UserWarning
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    # 存储 Torch 对象的标识符和名称的字典
    torch_object_ids = dict()
    # 在图中使用 C 绑定的函数集合
    c_binding_in_graph_functions = set()
    # 在图中使用非 C 绑定的函数集合
    non_c_binding_in_graph_functions = set()
    # 名称到 Torch 函数规则的映射字典
    torch_name_rule_map = dict()

    # 检查对象是否为特殊函数
    def is_special_functions(obj):
        return hashable(obj) and obj in {
            torch._C._cuda_isCurrentStreamCapturing,
            torch._C._graph_pool_handle,
        }

    # 根据启发式记录 Torch 中的图函数
    def heuristic_record_if_in_graph_function(obj, module, name):
        try:
            if hasattr(obj, "__wrapped__"):
                obj = obj.__wrapped__
        except Exception:
            pass
        # 如果对象是函数类型或特殊函数，则添加到对应的集合中
        if isinstance(
            obj,
            (
                types.FunctionType,
                types.BuiltinFunctionType,
                types.MethodDescriptorType,
                types.WrapperDescriptorType,
            ),
        ) or is_special_functions(obj):
            # 记录对象的名称和 TorchInGraphFunctionVariable 的关联
            torch_name_rule_map[
                f"{module.__name__}.{name}"
            ] = TorchInGraphFunctionVariable
            if c_binding_only:
                # 如果仅考虑 C 绑定的函数，则根据是否具有 __code__ 属性来划分
                if not hasattr(obj, "__code__"):
                    c_binding_in_graph_functions.add(obj)
            else:
                # 否则根据是否具有 __code__ 属性划分
                if hasattr(obj, "__code__"):
                    non_c_binding_in_graph_functions.add(obj)
                else:
                    c_binding_in_graph_functions.add(obj)
    # 定义一个函数 `_find_torch_objects`，用于查找并记录与 Torch 相关的对象
    def _find_torch_objects(module):
        # 如果模块名以指定列表中的任何字符串开头，则忽略该模块及其子模块
        if any(
            module.__name__.startswith(mod_name)
            for mod_name in config.allowed_functions_module_string_ignorelist
        ):
            return
        
        # 将模块对象的 ID 映射到其名称
        torch_object_ids[id(module)] = module.__name__
        
        # 遍历模块的属性
        for name, obj in list(module.__dict__.items()):
            # 如果对象的 ID 已经在 torch_object_ids 中，则跳过
            if id(obj) not in torch_object_ids:
                # 排除 HigherOrderOperator 类型，因为不希望将其添加到图中
                import torch._ops
                if isinstance(obj, torch._ops.HigherOrderOperator):
                    continue
                
                # 排除一些特定的函数和方法，不希望将它们添加到图中
                if obj in (
                    torch.func.grad,
                    deprecated_func.grad,
                    torch.func.vmap,
                    deprecated_func.vmap,
                    torch.nn.functional.triplet_margin_with_distance_loss,
                    torch.cond,
                ):
                    continue
                
                # 如果对象是模块类型，并且以 "torch." 开头且被允许，则记录其名称
                if isinstance(obj, types.ModuleType):
                    if obj.__name__.startswith("torch.") and _is_allowed_module_prefix(
                        obj
                    ):
                        torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                        # 递归调用 _find_torch_objects 处理子模块
                        _find_torch_objects(obj)
                
                # 如果对象的模块前缀是允许的，则记录其名称
                elif _is_allowed_module_prefix(obj):
                    if record:
                        # 如果需要记录，调用启发式记录函数
                        heuristic_record_if_in_graph_function(obj, module, name)
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                
                # 如果对象没有关联的模块且不是安全常量，则记录其名称
                elif inspect.getmodule(obj) is None and not is_safe_constant(obj):
                    if record:
                        # 如果需要记录，调用启发式记录函数
                        heuristic_record_if_in_graph_function(obj, module, name)
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"

    # 查找并记录与 Torch 相关的对象，起始点为 torch 模块
    _find_torch_objects(torch)
    # 查找并记录与 Torch 相关的对象，起始点为 math 模块
    _find_torch_objects(math)

    # 返回包含所有允许的对象及相关信息的 AllowedObjects 对象
    return AllowedObjects(
        torch_object_ids,  # 记录了所有 Torch 相关对象的 ID 到名称的映射
        c_binding_in_graph_functions,  # 记录了在图函数中使用的 C 绑定函数
        non_c_binding_in_graph_functions,  # 记录了在图函数中使用的非 C 绑定函数
        torch_name_rule_map,  # 记录了 Torch 名称规则映射
    )
class TraceRuleTests(torch._dynamo.test_case.TestCase):
    # 定义一个测试类 TraceRuleTests，继承自 torch._dynamo.test_case.TestCase

    def _check_set_equality(self, generated, used, rule_map, ignored_set):
        # 检查集合的相等性，用于验证生成的对象集合和使用的对象集合是否匹配
        x = generated - used
        y = used - generated
        msg1 = (
            f"New torch objects: {x} "
            f"were not added to `trace_rules.{rule_map}` or `test_trace_rules.{ignored_set}`. "
            "Refer the instruction in `torch/_dynamo/trace_rules.py` for more details."
        )
        msg2 = (
            f"Existing torch objects: {y} were removed. "
            f"Please remove them from `trace_rules.{rule_map}` or `test_trace_rules.{ignored_set}`. "
            "Refer the instruction in `torch/_dynamo/trace_rules.py` for more details."
        )
        self.assertTrue(len(x) == 0, msg1)
        self.assertTrue(len(y) == 0, msg2)
        # 断言生成的对象集合和使用的对象集合应为空集合，否则输出相应的消息提示

    # We are using python function and module string names for these inlinelist,
    # this unit test is to make sure the functions/modules can be correctly imported
    # or loaded in case there is typo in the strings.
    def test_skipfiles_inlinelist(self):
        # 测试跳过文件列表中的函数和模块是否能够正确导入或加载
        for m in LEGACY_MOD_INLINELIST.union(MOD_INLINELIST):
            self.assertTrue(
                isinstance(importlib.import_module(m), types.ModuleType),
                f"{m} from trace_rules.MOD_INLINELIST/LEGACY_MOD_INLINELIST is not a python module, please check and correct it.",
            )
            # 断言导入的模块是否为 types.ModuleType 类型，若不是则输出相应的错误信息

    @unittest.skip(
        "This test keeps getting broken and our disable infra is not handling well. see #120627"
    )
    def test_torch_name_rule_map_updated(self):
        # 测试 torch_name_rule_map 是否已更新
        # 根据 `allowed_functions.py` 中定义的启发式规则生成允许的对象
        objs = gen_allowed_objs_and_ids(record=True, c_binding_only=True)
        # 测试图形函数中的 C 绑定是否已更新到 torch_name_rule_map
        generated = objs.c_binding_in_graph_functions
        used = set()
        for x in (
            set(torch_c_binding_in_graph_functions.keys())
            | ignored_c_binding_in_graph_function_names
        ):
            obj = load_object(x)
            if obj is not None:
                used.add(obj)
        self._check_set_equality(
            generated,
            used,
            "torch_c_binding_in_graph_functions",
            "ignored_c_binding_in_graph_function_names",
        )
        # 对于非图形函数中的 C 绑定，仅测试它们是否能成功加载
        for f in torch_non_c_binding_in_graph_functions:
            self.assertTrue(
                isinstance(
                    load_object(f),
                    (
                        types.FunctionType,
                        types.BuiltinFunctionType,
                        types.MethodDescriptorType,
                        types.WrapperDescriptorType,
                    ),
                )
            )
            # 断言加载的对象类型是否为 types.FunctionType 等类型之一，以确保成功加载
    def test_force_inline_torch_function(self):
        # 定义一个内部函数 fn，用于根据输入 x 判断是否为 torch.Tensor 类型，并进行加一或减一操作
        def fn(x):
            if istype(x, torch.Tensor):  # 如果 x 是 torch.Tensor 类型
                return x + 1  # 返回 x + 1
            else:
                return x - 1  # 否则返回 x - 1

        _manual_torch_name_rule_map = manual_torch_name_rule_map.copy()
        # 强制内联 `torch._dynamo.utils.istype` 函数，通过设置跟踪规则。
        _manual_torch_name_rule_map["torch._dynamo.utils.istype"] = UserFunctionVariable

        _torch_name_rule_map = [
            _manual_torch_name_rule_map,  # 将手动设置的规则映射加入到总规则映射中
            torch_c_binding_in_graph_functions,  # 加入 torch C 绑定的图函数
            torch_non_c_binding_in_graph_functions,  # 加入 torch 非 C 绑定的图函数
        ]

        self.assertTrue(
            "torch._dynamo" not in torch._dynamo.trace_rules.LEGACY_MOD_INLINELIST
        )
        self.assertTrue("torch._dynamo" not in torch._dynamo.trace_rules.MOD_INLINELIST)

        with unittest.mock.patch(
            "torch._dynamo.trace_rules.torch_name_rule_map",  # 设置 torch_name_rule_map 的 patch
            _torch_name_rule_map,  # 使用上面定义的规则映射
        ), unittest.mock.patch(
            "torch._dynamo.trace_rules.get_torch_obj_rule_map",  # 设置 get_torch_obj_rule_map 的 patch
            torch._dynamo.trace_rules.get_torch_obj_rule_map.__wrapped__,  # 绕过 functools.lru_cache
        ):
            x = torch.rand(3)  # 生成一个 shape 为 (3,) 的随机 tensor x
            opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)  # 编译函数 fn 并优化
            ref = fn(x)  # 使用原始函数 fn 计算 ref
            res = opt_fn(x)  # 使用优化后的函数 opt_fn 计算 res
            self.assertEqual(ref, res)  # 断言 ref 和 res 相等

    def test_force_inline_custom_function(self):
        mod, func = create_dummy_module_and_function()  # 创建一个虚拟模块和函数

        def fn(x):
            return func(x)  # 调用虚拟模块中的函数 func

        _manual_torch_name_rule_map = manual_torch_name_rule_map.copy()
        # 强制内联 `mod.func` 函数，通过设置跟踪规则。
        _manual_torch_name_rule_map[
            f"{mod.__name__}.{func.__name__}"
        ] = UserFunctionVariable

        _torch_name_rule_map = [
            _manual_torch_name_rule_map,  # 将手动设置的规则映射加入到总规则映射中
            torch_c_binding_in_graph_functions,  # 加入 torch C 绑定的图函数
            torch_non_c_binding_in_graph_functions,  # 加入 torch 非 C 绑定的图函数
        ]

        with unittest.mock.patch(
            "torch._dynamo.trace_rules.torch_name_rule_map",  # 设置 torch_name_rule_map 的 patch
            _torch_name_rule_map,  # 使用上面定义的规则映射
        ), unittest.mock.patch(
            "torch._dynamo.trace_rules.get_torch_obj_rule_map",  # 设置 get_torch_obj_rule_map 的 patch
            torch._dynamo.trace_rules.get_torch_obj_rule_map.__wrapped__,  # 绕过 functools.lru_cache
        ):
            # 首先将模块添加到 SKIP_DIRS 中，以便默认情况下跳过它。
            torch._dynamo.trace_rules.add(mod.__name__)  # 将模块名称添加到跟踪规则中
            x = torch.rand(3)  # 生成一个 shape 为 (3,) 的随机 tensor x
            opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)  # 编译函数 fn 并优化
            ref = fn(x)  # 使用原始函数 fn 计算 ref
            res = opt_fn(x)  # 使用优化后的函数 opt_fn 计算 res
            self.assertEqual(ref, res)  # 断言 ref 和 res 相等
# 定义一个名为 TestModuleSurviveSkipFiles 的测试类，继承自 torch._dynamo.test_case.TestCase
class TestModuleSurviveSkipFiles(torch._dynamo.test_case.TestCase):
    
    # 标记该测试方法，如果 torch.distributed 不可用，则跳过执行，并提供相应的提示信息
    @unittest.skipIf(
        not torch.distributed.is_available(),
        "need to import MLP module from distributed",
    )
    # 定义名为 test_module_survive_skip_files 的测试方法
    def test_module_survive_skip_files(self):
        # 从 torch.testing._internal.common_fsdp 模块导入 MLP 类
        from torch.testing._internal.common_fsdp import MLP

        # 创建一个 MLP 类的实例，传入参数为 3
        model = MLP(3)
        # 生成一个形状为 (2, 3) 的随机张量 inp
        inp = torch.randn((2, 3))
        # 获取测试之前的 torch._dynamo.convert_frame.FRAME_COUNTER 的值
        frame_count_before = torch._dynamo.convert_frame.FRAME_COUNTER
        # 使用 "eager" 后端编译模型
        model.compile(backend="eager")
        # 将 inp 输入到模型中进行前向传播
        model(inp)
        # 获取测试之后的 torch._dynamo.convert_frame.FRAME_COUNTER 的值
        frame_count_after = torch._dynamo.convert_frame.FRAME_COUNTER
        # 断言 frame_count_after 应大于 frame_count_before，否则输出错误信息 "MLP did not survive skip files"
        self.assertTrue(
            frame_count_after > frame_count_before, "MLP did not survive skip files"
        )

# 如果该脚本被直接运行
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行所有测试
    run_tests()
```