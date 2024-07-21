# `.\pytorch\torch\testing\_internal\optests\generate_tests.py`

```py
# 忽略类型检查错误，这里可能是为了兼容性或其他原因
# 在此处导入必要的库和模块
import datetime  # 导入处理日期和时间的模块
import difflib  # 导入用于文本比较的模块
import functools  # 导入用于高阶函数操作的模块
import inspect  # 导入用于检查对象的模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入与操作系统交互的模块
import re  # 导入处理正则表达式的模块
import tempfile  # 导入处理临时文件和目录的模块
import threading  # 导入用于多线程编程的模块
import unittest  # 导入 Python 单元测试框架

import torch  # 导入 PyTorch 深度学习框架

import torch._dynamo  # 导入 PyTorch 内部动态图机制相关模块

import torch.utils._pytree as pytree  # 导入 PyTorch 工具相关模块
from torch._dynamo.utils import clone_input  # 从 PyTorch 内部动态图机制的工具中导入函数
from torch._library.custom_ops import CustomOpDef  # 导入自定义运算符定义类
from torch._subclasses.schema_check_mode import SchemaCheckMode  # 导入模式检查模块
from torch._utils_internal import get_file_path_2  # 导入内部工具中的文件路径函数
from torch.overrides import TorchFunctionMode  # 导入 Torch 函数模式
from torch.testing._internal.optests import (  # 导入内部测试相关函数
    aot_autograd_check,  # 导入 AOT 自动微分检查函数
    autograd_registration_check,  # 导入自动微分注册检查函数
    fake_check,  # 导入伪装检查函数
)


# 定义一个装饰器函数，用于标记不生成运算符检查测试的函数
def dontGenerateOpCheckTests(reason: str):
    def inner(fun):
        fun._torch_dont_generate_opcheck_tests = True
        return fun

    return inner


# 检查给定的张量是否为抽象张量
def is_abstract(tensor: torch.Tensor) -> bool:
    # 如果张量是元张量（即虚拟张量），返回 True
    if tensor.is_meta:
        return True
    # 如果张量是伪装张量，返回 True
    if torch._subclasses.fake_tensor.is_fake(tensor):
        return True
    # 否则返回 False
    return False


# 安全地执行模式检查
def safe_schema_check(
    op: torch._ops.OpOverload,  # 操作重载对象
    args: Tuple[Any, ...],  # 参数元组
    kwargs: Dict[str, Any],  # 关键字参数字典
    *,
    copy_inputs: bool = True,  # 是否复制输入，默认为 True
) -> Any:
    # 如果需要复制输入参数
    if copy_inputs:
        # 深度复制输入参数
        args, kwargs = deepcopy_tensors((args, kwargs))
    # 如果输入中任何一个是抽象张量，则返回 None
    if pytree.tree_any_only(torch.Tensor, is_abstract, (args, kwargs)):
        return None
    # 否则，使用模式检查模式执行操作
    with SchemaCheckMode():
        # 执行操作
        result = op(*args, **kwargs)
        return result


# 安全地执行自动微分注册检查
def safe_autograd_registration_check(
    op: torch._ops.OpOverload,  # 操作重载对象
    args: Tuple[Any, ...],  # 参数元组
    kwargs: Dict[str, Any],  # 关键字参数字典
    *,
    copy_inputs: bool = True,  # 是否复制输入，默认为 True
) -> None:
    # 如果输入中任何一个是抽象张量，则直接返回
    if pytree.tree_any_only(torch.Tensor, is_abstract, (args, kwargs)):
        return
    # 如果需要复制输入参数
    if copy_inputs:
        # 深度复制输入参数
        args, kwargs = deepcopy_tensors((args, kwargs))
    # 如果所有输入都不需要梯度，则直接返回
    if not pytree.tree_any_only(
        torch.Tensor, lambda x: x.requires_grad, (args, kwargs)
    ):
        return
    # 否则执行自动微分注册检查
    return autograd_registration_check(op, args, kwargs)


# 安全地执行伪装检查
def safe_fake_check(
    op: torch._ops.OpOverload,  # 操作重载对象
    args: Tuple[Any, ...],  # 参数元组
    kwargs: Dict[str, Any],  # 关键字参数字典
    *,
    copy_inputs: bool = True,  # 是否复制输入，默认为 True
) -> None:
    # 如果输入中任何一个是抽象张量，则返回 None
    if pytree.tree_any_only(torch.Tensor, is_abstract, (args, kwargs)):
        return None
    # 如果需要复制输入参数
    if copy_inputs:
        # 深度复制输入参数
        args, kwargs = deepcopy_tensors((args, kwargs))
    # 执行伪装检查
    return fake_check(op, args, kwargs)


# 安全地执行 AOT 自动微分检查
def safe_aot_autograd_check(
    op: torch._ops.OpOverload,  # 操作重载对象
    args: Tuple[Any, ...],  # 参数元组
    kwargs: Dict[str, Any],  # 关键字参数字典
    dynamic: bool,  # 是否为动态模式
    *,
    copy_inputs: bool = True,  # 是否复制输入，默认为 True
) -> Any:
    # NB: copy_inputs 对于 AOT 自动微分检查无效，它总是需要复制输入
    # 如果输入中任何一个是抽象张量，则返回 None
    if pytree.tree_any_only(torch.Tensor, is_abstract, (args, kwargs)):
        return None

    # 定义一个函数来处理输入参数的复制
    def func(*args, **kwargs):
        # 对输入参数中的张量进行复制
        args, kwargs = pytree.tree_map_only(torch.Tensor, torch.clone, (args, kwargs))
        # 执行操作
        return op(*args, **kwargs)

    # 返回操作的结果
    return func
    # 调用 aot_autograd_check 函数，该函数多次运行 func(*args, **kwargs)，并假设 func 不会修改其输入参数。
    # 返回 aot_autograd_check 函数的执行结果，其中 dynamic 和 check_gradients 参数默认为 "auto"。
    return aot_autograd_check(func, args, kwargs, dynamic, check_gradients="auto")
# 深度复制输入数据结构中的所有张量，使用 pytree.tree_map_only 函数进行映射
def deepcopy_tensors(inputs: Any) -> Any:
    return pytree.tree_map_only(torch.Tensor, clone_input, inputs)


# 所有测试工具的集合，以字典形式表示，每个条目包含一个测试工具名称和对应的函数引用
ALL_TEST_UTILS = {
    "test_schema": safe_schema_check,
    "test_autograd_registration": safe_autograd_registration_check,
    "test_faketensor": safe_fake_check,
    "test_aot_dispatch_static": functools.partial(
        safe_aot_autograd_check,
        dynamic=False,
    ),
    "test_aot_dispatch_dynamic": functools.partial(
        safe_aot_autograd_check,
        dynamic=True,
    ),
}

# Google 文档链接
GDOC = "https://docs.google.com/document/d/1Pj5HRZvdOq3xpFpbEjUZp2hBovhy7Wnxw14m6lF2154/edit"

# 默认测试工具名称列表，用于 generate_opcheck_tests 函数的默认参数
DEFAULT_TEST_UTILS = [
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
]

# 已弃用的默认测试工具名称列表，是 DEFAULT_TEST_UTILS 的扩展
DEPRECATED_DEFAULT_TEST_UTILS = DEFAULT_TEST_UTILS + [
    "test_aot_dispatch_static",
]


def generate_opcheck_tests(
    testcase: Any,
    namespaces: List[str],
    failures_dict_path: Optional[str] = None,
    additional_decorators: Dict[str, Callable] = None,
    test_utils: List[str] = DEFAULT_TEST_UTILS,
) -> None:
    """给定一个现有的 TestCase，利用现有的测试来生成自定义操作符的额外验证测试。

    对于 {TestCase 中的所有现有测试} x {所有测试工具}，
    我们将生成一个新的测试。新测试运行 TorchFunctionMode，
    拦截 ``op(*args, **kwargs)`` 调用，并调用 ``test_util(op, *args, **kwargs)``，
    其中 ``op`` 是一个操作符。

    我们支持的 test_util 在 ALL_TEST_UTILS 中，包括：
    - test_schema: 运行 SchemaCheckMode。
    - test_autograd_registration: 运行 autograd_registration_check。
    - test_faketensor: 运行 CrossRefFakeMode。
    - test_aot_dispatch_static: 运行 aot_autograd_check，检查在 eager 模式和使用 AOTAutograd 时输出是否相同。
    - test_aot_dispatch_dynamic: 与 aot_dispatch_static 相同，但使用动态形状而非静态形状运行 AOTAutograd。

    生成的测试名称为 ``{test_util}__{original_name}``。
    例如，如果有一个名为 ``test_cumsum`` 的方法，我们将生成一个 ``test_schema__test_cumsum``,
    ``test_faketensor__test_cumsum`` 等。

    更多详情请参阅 https://docs.google.com/document/d/1Pj5HRZvdOq3xpFpbEjUZp2hBovhy7Wnxw14m6lF2154/edit
    """
    Args:
        # 测试用例，将修改并生成额外的测试
        testcase: 要修改和生成额外测试的测试用例对象。
        # 我们只拦截这些命名空间中的自定义操作符调用
        namespaces: 仅拦截这些命名空间中的自定义操作符调用。
        # 失败字典路径，详见“validate_failures_dict_structure”以获取更多详情
        failures_dict_path: 失败字典的路径。
        # 测试工具列表，用于生成测试。例如：["test_schema", "test_faketensor"]
        test_utils: 要生成的测试工具列表。
    """
    # 如果没有提供额外的装饰器，则初始化为空字典
    if additional_decorators is None:
        additional_decorators = {}
    
    # 获取测试用例对象中以“test_”开头并且可调用的方法列表
    test_methods = [
        m
        for m in dir(testcase)
        if m.startswith("test_") and callable(getattr(testcase, m))
    ]
    
    # 如果失败字典路径未指定，则默认为与测试文件相同目录下的“failures_dict.json”
    if failures_dict_path is None:
        prev_frame = inspect.currentframe().f_back
        filename = inspect.getframeinfo(prev_frame)[0]
        failures_dict_path = get_file_path_2(
            os.path.dirname(filename), "failures_dict.json"
        )
    
    # 加载失败字典数据，根据需要创建文件
    failures_dict = FailuresDict.load(
        failures_dict_path, create_file=should_update_failures_dict()
    )
    
    # 验证失败字典的结构是否符合预期，针对指定的测试工具和测试用例
    validate_failures_dict_structure(failures_dict, test_utils, testcase)
    
    # 验证失败字典的格式是否正确，使用指定的失败字典路径
    validate_failures_dict_formatting(failures_dict_path)
    # 定义一个函数，用于构造测试方法，根据给定的属性、前缀和测试器
    def construct_method(attr, prefix, tester):
        # 获取测试类中指定属性的方法对象
        method = getattr(testcase, attr)
        # 检查方法是否被标记为不生成 OpCheck 测试，如果是，则直接返回
        if getattr(method, "_torch_dont_generate_opcheck_tests", False):
            return
        # 构建新的方法名，由前缀、双下划线和原方法名组成
        new_method_name = prefix + "__" + attr

        # 定义一个新方法，保留原方法的签名和功能
        @functools.wraps(method)
        def new_method(*args, **kwargs):
            # 进入 OpCheck 模式上下文，设置相关参数
            with OpCheckMode(
                namespaces,
                prefix,
                tester,
                failures_dict,
                f"{testcase.__name__}.{new_method_name}",
                failures_dict_path,
            ):
                # 调用原方法，并获取其返回结果
                result = method(*args, **kwargs)
            # 返回原方法的执行结果
            return result

        # 检查新方法是否有 pytestmark 标记
        if pytestmark := new_method.__dict__.get("pytestmark"):
            import pytest

            # 检查是否需要简化 parametrize 标记
            # 注意：需要将相关标记添加到 pytest.ini 文件中
            opcheck_only_one = False
            for mark in pytestmark:
                if isinstance(mark, pytest.Mark) and mark.name == "opcheck_only_one":
                    opcheck_only_one = True

            # 如果需要简化 parametrize 标记，则进行处理
            if opcheck_only_one:
                new_pytestmark = []
                for mark in pytestmark:
                    if isinstance(mark, pytest.Mark) and mark.name == "parametrize":
                        argnames, argvalues = mark.args
                        assert not mark.kwargs, "NYI"
                        # 特殊处理 device 参数，希望在所有设备上运行
                        if argnames != "device":
                            new_pytestmark.append(
                                pytest.mark.parametrize(
                                    argnames, (next(iter(argvalues)),)
                                )
                            )
                            continue
                    new_pytestmark.append(mark)
                # 更新新方法的 pytestmark 标记
                new_method.__dict__["pytestmark"] = new_pytestmark

        # 如果新方法名已存在于额外装饰器中，则应用这些装饰器
        if new_method_name in additional_decorators:
            for dec in additional_decorators[new_method_name]:
                new_method = dec(new_method)

        # 检查测试类中是否已存在同名方法，如果是则引发运行时错误
        if hasattr(testcase, new_method_name):
            raise RuntimeError(
                f"Tried to autogenerate {new_method_name} but {testcase} already "
                f"has method named {new_method_name}. Please rename the original "
                f"method on the TestCase."
            )
        # 将构造的新方法设置为测试类中的方法
        setattr(testcase, new_method_name, new_method)

    # 从 ALL_TEST_UTILS 中选择出给定测试工具名字的工具集合
    test_utils = {name: ALL_TEST_UTILS[name] for name in test_utils}
    # 遍历所有测试方法
    for attr in test_methods:
        # 对每个测试工具集合应用测试方法的构造函数
        for prefix, tester in test_utils.items():
            construct_method(attr, prefix, tester)

    # 生成标签测试，使用测试类、失败字典和额外装饰器
    generate_tag_tests(testcase, failures_dict, additional_decorators)
# 定义一个函数，用于生成标记测试用例的测试方法并将其添加到测试实例中
def generate_tag_tests(testcase, failures_dict, additional_decorators):
    # 定义生成单个测试方法的内部函数
    def generate_test(qualname, definitely_not_pt2_compliant, xfailed_tests):
        # 实际生成的测试方法
        def inner(self):
            try:
                # 查找指定名称的操作符
                op = torch._library.utils.lookup_op(qualname)
            except AttributeError as e:
                # 如果操作符在当前测试文件中无法导入，跳过该测试
                raise unittest.SkipTest(f"Can't import operator {qualname}") from e
            # 检查操作符是否被标记为 pt2_compliant
            op_marked_as_compliant = torch.Tag.pt2_compliant_tag in op.tags
            if not op_marked_as_compliant:
                return
            # 如果操作符被标记为 pt2_compliant，但是其有失败的 opcheck 测试
            if not definitely_not_pt2_compliant:
                return
            # 抛出断言错误，说明存在潜在的正确性问题需要修复
            raise AssertionError(
                f"op '{qualname}' was tagged with torch.Tag.pt2_compliant_tag "
                f"but it failed some of the generated opcheck tests "
                f"({xfailed_tests}). This may lead to silent correctness issues, "
                f"please fix this."
            )

        return inner

    # 遍历 failures_dict 中的数据项
    for qualname, test_dict in failures_dict.data.items():
        # 找出标记为 xfail 的测试用例，排除特定的测试用例
        xfailed_tests = [
            test
            for test, status_dict in test_dict.items()
            if "test_aot_dispatch_static" not in test
            and status_dict["status"] == "xfail"
        ]
        # 判断是否存在肯定不符合 pt2 标准的测试用例
        definitely_not_pt2_compliant = len(xfailed_tests) > 0
        # 生成测试方法
        generated = generate_test(qualname, definitely_not_pt2_compliant, xfailed_tests)

        # 根据 qualname 创建一个类似 mangled 的测试方法名称
        mangled_qualname = qualname.replace("::", "_").replace(".", "_")
        test_name = "test_pt2_compliant_tag_" + mangled_qualname

        # 如果 test_name 存在于 additional_decorators 中，则应用额外的装饰器
        if test_name in additional_decorators:
            for decorator in additional_decorators[test_name]:
                generated = decorator(generated)

        # 检查是否已经存在同名的测试方法，避免命名冲突
        if hasattr(testcase, test_name):
            raise RuntimeError(
                f"Tried to generate a test named {test_name}, but it exists "
                f"already. This could be because of a name collision (where "
                f"we generated two tests with the same name), or where we "
                f"generated a test with the same name as an existing test."
            )
        # 将生成的测试方法设置为 testcase 的属性
        setattr(testcase, test_name, generated)


# 定义测试选项的元组
TEST_OPTIONS = ("xfail", "skip", "xsuccess")


# 校验 failures_dict 文件格式的函数
def validate_failures_dict_formatting(failures_dict_path: str) -> None:
    # 读取 failures_dict 文件内容
    with open(failures_dict_path) as fp:
        actual = fp.read()
    # 加载 failures_dict 文件为 FailuresDict 对象
    failures_dict = FailuresDict.load(failures_dict_path)
    # 获取 failures_dict 对象的字符串表示
    expected = failures_dict._save(to_str=True)
    # 如果实际内容与预期内容相同，则格式正确，直接返回
    if actual == expected:
        return
    # 如果需要更新 failures_dict，则保存当前的 failures_dict
    if should_update_failures_dict():
        failures_dict = FailuresDict.load(failures_dict_path)
        failures_dict.save()
        return
    # 将预期内容按行分割为列表
    expected = expected.splitlines(1)
    # 将字符串 `actual` 按行分割为列表
    actual = actual.splitlines(1)
    # 使用 difflib 库生成 actual 和 expected 之间的差异信息
    diff = difflib.unified_diff(actual, expected)
    # 将差异信息列表转换为字符串形式
    diff = "".join(diff)
    # 抛出 RuntimeError 异常，包含详细的差异信息和建议
    raise RuntimeError(
        f"\n{diff}\n\nExpected the failures dict to be formatted "
        f"a certain way. Please see the above diff; you can correct "
        f"this either manually or by re-running the test with "
        f"PYTORCH_OPCHECK_ACCEPT=1"
    )
# 校验 failures_dict 结构的函数，接受三个参数：failure_dict 是一个 FailuresDict 类型的字典，test_utils 是一个字符串列表，testcase 是任意类型
def validate_failures_dict_structure(
    failure_dict: "FailuresDict", test_utils: List[str], testcase: Any
) -> None:
    """Validates the failures dict.

    The failure dict looks something like the following.
    It maps operator name (qualname) to a list of autogenerated tests.
    Each autogenerated test may have a check for the operator (if the operator is
    called by the test); the dictionary specifies if we should skip the check,
    or if we expect some check to fail.

    {
        "fbgemm::split_lengths": {
            "test_schema__test_split_lengths": {
                "comment": "you can put whatever you want into the comment section",
                "status": "xfail",
            },
            "test_schema__test_split_lengths_empty": {
                "comment": "",
                "status": "skip",
            },
        },
        "fbgemm::gather_lengths": {
            "test_schema__test_gather_lengths": {
                "comment": "",
                "status": "skip",
            },
        },
    }

    """
    # 将 failure_dict 参数重新赋值为其 data 属性
    failure_dict = failure_dict.data
    # 获取 failure_dict 中所有的操作符名称（qualname）
    qualnames = list(failure_dict.keys())
    # 遍历 failure_dict 中每个操作符的测试选项
    for test_to_option in failure_dict.values():
        # 获取每个操作符的测试名称列表
        test_names = list(test_to_option.keys())
        # 遍历每个测试名称及其对应的测试选项字典
        for test_name, test_dict in test_to_option.items():
            # 检查测试选项字典的键是否为 {'comment', 'status'}
            if set(test_dict.keys()) != set({"comment", "status"}):
                # 若不是，则抛出运行时错误
                raise RuntimeError(
                    "in failures_dict, expected sub-dict to have keys 'comment' and 'status'"
                )
            # 获取测试选项的状态
            test_option = test_dict["status"]
            # 检查测试选项状态是否在 TEST_OPTIONS 中
            if test_option not in TEST_OPTIONS:
                # 若不在，则抛出运行时错误
                raise RuntimeError(
                    f"In failures_dict, got status={test_option} but it needs to be in {TEST_OPTIONS}"
                )
            # 将测试名称分割成测试类和实际测试名称
            test_class, actual_test_name = test_name.split(".")
            # 检查 actual_test_name 是否以 test_utils 中任意字符串开头
            if not any(actual_test_name.startswith(test) for test in test_utils):
                # 若不是，则抛出运行时错误
                raise RuntimeError(
                    f"In failures_dict, test name '{test_name}' should begin with one of {test_utils}"
                )
            # 遍历 test_utils 列表中的每个测试名称
            for test in test_utils:
                # 如果 actual_test_name 不以当前 test 开头，则继续下一个循环
                if not actual_test_name.startswith(test):
                    continue
                # 去掉 pytest 参数化后缀
                base_test_name = actual_test_name[len(test) + 2 :]
                base_test_name = re.sub(r"\[.*\]", "", base_test_name)
                # 如果 testcase.__name__ 不等于 test_class，则继续下一个循环
                if testcase.__name__ != test_class:
                    continue
                # 如果 testcase 具有 base_test_name 属性，则继续下一个循环
                if hasattr(testcase, base_test_name):
                    continue
                # 若上述条件都不满足，则抛出运行时错误
                raise RuntimeError(
                    f"In failures dict, got test name '{test_name}'. We parsed this as "
                    f"running test '{test}' on '{base_test_name}', but "
                    f"{base_test_name} does not exist on the TestCase '{testcase.__name__}'. "
                    f"Maybe you need to change the test name?"
                )


def should_update_failures_dict() -> bool:
    # 定义一个函数，返回布尔值，用于确定是否应更新 failures_dict
    key = "PYTORCH_OPCHECK_ACCEPT"
    # 检查环境变量中是否存在指定的键，并且其对应的值是否为字符串 "1"
    return key in os.environ and os.environ[key] == "1"
# 设置一个线程本地变量，用于标识当前是否处于操作检查模式
_is_inside_opcheck_mode = threading.local()
# 将操作检查模式的值初始化为 False
_is_inside_opcheck_mode.value = False

# 定义一个函数，用于返回当前是否处于操作检查模式
def is_inside_opcheck_mode():
    return _is_inside_opcheck_mode.value

# OpCheckMode 类继承自 TorchFunctionMode，用于拦截运算符调用并运行相应的测试工具函数
class OpCheckMode(TorchFunctionMode):
    """
    For a given test, OpCheckMode intercepts calls to operators and runs
    test_util(op, args, kwargs) for each intercepted (op, args, kwargs).
    """

    def __init__(
        self,
        namespaces: List[str],         # 要拦截的运算符命名空间列表
        test_util_name: str,           # 测试工具函数的名称
        test_util: Callable,           # 实际的测试工具函数，其签名应为 (op, args, kwargs) -> None
        failures_dict: "FailuresDict", # 失败记录的字典
        test_name: str,                # 当前运行 OpCheckMode 的测试名称
        failures_dict_path: str,       # 失败记录字典的路径
    ):
        # 要拦截的运算符命名空间列表
        self.namespaces = namespaces
        # 测试工具函数，用于每个拦截的 (op, args, kwargs) 运行测试
        self.test_util = test_util
        # 测试工具函数的名称
        self.test_util_name = test_util_name
        # 当前运行 OpCheckMode 的测试名称
        self.test_name = test_name
        # 失败记录的字典，用于判断是否跳过测试或断言存在失败
        self.failures_dict = failures_dict
        # 失败记录字典的路径，用于改善错误消息
        self.failures_dict_path = failures_dict_path

        # OpCheckMode 抑制错误，将它们收集在这里，在退出时统一抛出
        # 字典，映射运算符的限定名称到 [(异常, 函数, 可能的参数, 可能的关键字参数)] 的列表
        self.seen_ops_to_errors = {}
    # 检查预期的失败情况
    for qualname in self.seen_ops_to_errors.keys():
        # 获取操作符在失败字典中的状态
        option = self.failures_dict.get_status(qualname, self.test_name)
        # 如果没有观察到任何错误
        if len(self.seen_ops_to_errors[qualname]) == 0:
            # 如果需要更新失败字典，则设置状态为 "xsuccess"
            if should_update_failures_dict():
                self.failures_dict.set_status(
                    qualname, self.test_name, "xsuccess", comment=""
                )
            else:
                # 如果操作符标记为 "xfail"，则引发 OpCheckError 异常
                if option == "xfail":
                    raise OpCheckError(
                        f"generate_opcheck_tests: Unexpected success for operator "
                        f"{qualname} on test {self.test_name}. This may mean that "
                        f"you have fixed this test failure. Please rerun the test with "
                        f"PYTORCH_OPCHECK_ACCEPT=1 to automatically update the test runner "
                        f"or manually remove the "
                        f"expected failure in the failure dict at "
                        f"{self.failures_dict_path}"
                        f"For more details, see "
                        f"{GDOC}"
                    )
            continue

    # 收集失败的操作符列表
    failed_ops = []
    for qualname in self.seen_ops_to_errors.keys():
        option = self.failures_dict.get_status(qualname, self.test_name)
        # 如果操作符状态不是 "xsuccess"，则继续下一个操作符
        if option != "xsuccess":
            continue
        # 如果观察到了错误，则将操作符添加到失败操作符列表中
        if len(self.seen_ops_to_errors[qualname]) > 0:
            failed_ops.append(qualname)

    # 如果没有失败的操作符，则返回
    if not failed_ops:
        return

    # 如果需要更新失败字典，则将所有失败操作符状态设置为 "xfail"
    if should_update_failures_dict():
        for op in failed_ops:
            self.failures_dict.set_status(op, self.test_name, "xfail")
        return

    # 如果不需要更新失败字典，则引发 OpCheckError 异常
    # 异常信息包括第一个错误的详细信息和建议的重现命令
    ex, op, args, kwargs = self.seen_ops_to_errors[failed_ops[0]][0]
    repro_command = generate_repro(
        self.test_util_name, op, args, kwargs, save_data=should_print_better_repro()
    )
    raise OpCheckError(
        f"Test generated by `generate_opcheck_tests`, {self.test_name}, "
        f"failed on operators {failed_ops}. This usually means that the "
        f"operators are not implemented correctly and may lead to silently "
        f"incorrect behavior. Set PYTORCH_OPCHECK_PRINT_BETTER_REPRO=1 for a standalone repro, "
        f"or please see "
        f"{GDOC} "
        f"for more recommendations. "
        f"To reproduce this problem locally, try to run the following:\n{repro_command}"
    ) from ex
    # 在进入上下文管理器时保存当前的操作检查模式状态和环境变量设置
    def __enter__(self, *args, **kwargs):
        self.prev_is_opcheck_mode = _is_inside_opcheck_mode.value
        self.prev_dynamo_disable = os.environ.get("TORCHDYNAMO_DISABLE", "")
        _is_inside_opcheck_mode.value = True  # 设置操作检查模式为True
        os.environ["TORCHDYNAMO_DISABLE"] = "1"  # 设置环境变量TORCHDYNAMO_DISABLE为"1"
        return super().__enter__(*args, **kwargs)  # 调用父类的__enter__方法并返回结果

    # 在退出上下文管理器时恢复先前保存的操作检查模式状态和环境变量设置
    def __exit__(self, *args, **kwargs):
        _is_inside_opcheck_mode.value = self.prev_is_opcheck_mode  # 恢复操作检查模式状态
        os.environ["TORCHDYNAMO_DISABLE"] = self.prev_dynamo_disable  # 恢复环境变量TORCHDYNAMO_DISABLE
        try:
            self.maybe_raise_errors_on_exit()  # 在退出时可能会触发错误处理
            if should_update_failures_dict():  # 如果需要更新失败字典
                self.failures_dict.save()  # 保存失败字典
        finally:
            result = super().__exit__(*args, **kwargs)  # 调用父类的__exit__方法并获取返回结果
        return result

    # 运行测试工具函数，捕获可能出现的异常并忽略处理
    def run_test_util(self, op, args, kwargs):
        try:
            self.test_util(op, args, kwargs, copy_inputs=False)  # 调用测试工具函数
        except torch._subclasses.fake_tensor.UnsupportedFakeTensorException:
            # 如果遇到UnsupportedFakeTensorException异常，则忽略处理
            # 可能是输入已经是FakeTensor或者处于torch.compile块中
            # 这种情况下，将其报告为失败会导致太多噪音，因此直接忽略
            pass
    # 实现了特殊方法 __torch_function__，用于拦截和处理对运算符的调用
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # 如果 kwargs 是 None，则将其设置为空字典
        kwargs = kwargs if kwargs else {}

        # 只拦截对操作符的调用
        if not isinstance(func, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
            return func(*args, **kwargs)

        # 如果正在进行 Torch 的跟踪、脚本编译或者动态编译，则直接调用原始函数
        if (
            torch.jit.is_tracing()
            or torch.jit.is_scripting()
            or torch._dynamo.is_compiling()
        ):
            return func(*args, **kwargs)

        # 对于 OpOverloadPacket，尝试解析唯一的重载，否则抛出异常
        if isinstance(func, torch._ops.OpOverloadPacket):
            func = resolve_unique_overload_or_throw(func)

        # 获取函数的全限定名，分割命名空间
        qualname = func.name()
        ns = qualname.split("::")[0]

        # 如果命名空间不在 self.namespaces 中，直接调用原始函数
        if ns not in self.namespaces:
            return func(*args, **kwargs)

        # 深度复制输入的张量参数和关键字参数
        args_c, kwargs_c = deepcopy_tensors((args, kwargs))

        # 调用原始函数并获取结果
        result = func(*args, **kwargs)

        # 根据 qualname 和 test_name 从 failures_dict 中获取执行状态选项
        option = self.failures_dict.get_status(qualname, self.test_name)

        # 如果状态是 "xsuccess" 或 "xfail"
        if option == "xsuccess" or option == "xfail":
            # 在执行期间抑制所有错误，稍后在 __exit__ 方法中抛出
            try:
                # 如果 qualname 不在 seen_ops_to_errors 中，将其添加进去
                if qualname not in self.seen_ops_to_errors:
                    self.seen_ops_to_errors[qualname] = []
                # 运行测试工具函数，传入深度复制的参数
                self.run_test_util(func, args_c, kwargs_c)
            except Exception as ex:
                # 如果需要更好的重现，将异常记录到 seen_ops_to_errors 中
                if should_print_better_repro():
                    self.seen_ops_to_errors[qualname].append((ex, func, args, kwargs))
                else:
                    self.seen_ops_to_errors[qualname].append((ex, func, None, None))
        # 如果状态是 "skip"，则什么也不做

        # 返回函数执行的结果
        return result
# 检查环境变量中是否设置了打印更好的重现命令的标志
def should_print_better_repro() -> bool:
    key = "PYTORCH_OPCHECK_PRINT_BETTER_REPRO"
    if key not in os.environ:
        return False
    value = os.environ[key]
    # 返回是否设置为字符串"1"或整数1，表示需要打印更好的重现命令
    return value == "1" or value == 1


def opcheck(
    op: Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket, CustomOpDef],
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    test_utils: Union[str, Sequence[str]] = DEFAULT_TEST_UTILS,
    raise_exception: bool = True,
) -> Dict[str, str]:
    """See torch.library.opcheck for docstring"""

    # 如果kwargs为None，设置为空字典
    if kwargs is None:
        kwargs = {}
    
    # 如果op是CustomOpDef类型，则将其转换为_opoverload
    if isinstance(op, CustomOpDef):
        op = op._opoverload
    
    # 如果op是OpOverloadPacket类型，则解析唯一的重载或抛出异常
    if isinstance(op, torch._ops.OpOverloadPacket):
        op = resolve_unique_overload_or_throw(op)
    
    # 如果op不是OpOverload类型，则抛出数值错误异常
    if not isinstance(op, torch._ops.OpOverload):
        raise ValueError(
            f"opcheck(op, ...): op must be instance of torch._ops.OpOverload, "
            f"e.g. torch.ops.aten.sin.default, got {type(op)}"
        )
    
    # 如果test_utils为"ALL"，将其设置为所有测试工具的键的元组
    if test_utils == "ALL":
        test_utils = tuple(ALL_TEST_UTILS.keys())
    
    # 如果test_utils是字符串，则转换为单元素元组
    if isinstance(test_utils, str):
        test_utils = (test_utils,)
    
    # 如果test_utils不是元组或列表，或者不是所有测试工具键的子集，则抛出数值错误异常
    if not isinstance(test_utils, (tuple, list)) or not set(test_utils).issubset(
        ALL_TEST_UTILS.keys()
    ):
        raise ValueError(
            f"opcheck(op, ..., test_utils={test_utils}), expected test_utils "
            f"to be subset of {tuple(ALL_TEST_UTILS.keys())} but it was not"
        )

    # 初始化结果字典
    results_dict = {}
    # 遍历所有测试工具
    for test_util in test_utils:
        # 获取测试工具对应的测试函数
        tester = ALL_TEST_UTILS[test_util]
        try:
            # 调用测试函数测试操作、参数和关键字参数
            tester(op, args, kwargs)
            # 如果测试通过，将结果记录为成功
            results_dict[test_util] = "SUCCESS"
        except Exception as ex:
            # 如果设置为抛出异常，抛出OpCheckError异常，否则记录异常信息
            if raise_exception:
                raise OpCheckError(
                    f"opcheck(op, ...): {test_util} failed with {ex} "
                    f"(scroll up for stack trace)"
                ) from ex
            results_dict[test_util] = ex
    # 返回所有测试工具的测试结果字典
    return results_dict


class OpCheckError(Exception):
    pass


def generate_repro(
    test: str,
    op: torch._ops.OpOverload,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    save_data: bool,
    dry_run: bool = False,
) -> str:
    # 如果需要保存数据
    if save_data:
        # 获取当前时间
        now = datetime.datetime.now()
        # 获取临时目录路径
        path = os.path.join(tempfile.gettempdir(), "pytorch_opcheck_safe_to_delete")
        # 计算当前时间的 Unix 时间戳，并转换为整数
        unix_timestamp = datetime.datetime.timestamp(now) * 100000
        # 构建保存文件的完整路径
        filepath = os.path.join(path, f"repro_{unix_timestamp}.pt")
        # 如果不是 dry_run 模式，则创建临时目录（如果不存在），并保存数据
        if not dry_run:
            os.makedirs(path, exist_ok=True)
            torch.save((args, kwargs), filepath)
        # 构建字符串，表示加载参数和关键字参数的语句
        args_kwargs = f'args, kwargs = torch.load("{filepath}")'
    else:
        # 如果不需要保存数据，构建一个字符串表示空的参数和关键字参数
        args_kwargs = (
            "# If you rerun your test with PYTORCH_OPCHECK_PRINT_BETTER_REPRO=1\n"
            "# we will fill them in same (args, kwargs) as in your test\n"
            "args = ()  # args to the operator\n"
            "kwargs = {}  # kwargs to the operator"
        )

    # 分割操作名称空间和操作名称
    ns, name = op._schema.name.split("::")
    # 获取操作的重载名称
    overload = op._overloadname

    # 构建复现脚本的命令
    repro_command = (
        f"# =========================================================\n"
        f"# BEGIN REPRO SCRIPT\n"
        f"# =========================================================\n"
        f"import torch\n"
        f"from torch.testing._internal.optests import opcheck\n"
        f"\n"
        f"# 确保已经通过导入或者 torch.ops.load_library(...) 加载了包含该操作的库\n"
        f"op = torch.ops.{ns}.{name}.{overload}\n"
        f"\n"
        f"{args_kwargs}\n"  # 加载参数和关键字参数的语句
        f'opcheck(op, args, kwargs, test_utils="{test}")\n'  # 调用 opcheck 函数进行操作测试
        f"# =========================================================\n"
        f"# END REPRO SCRIPT\n"
        f"# =========================================================\n"
    )
    return repro_command  # 返回复现脚本命令字符串
# 解析唯一的重载或抛出异常
def resolve_unique_overload_or_throw(
    op: torch._ops.OpOverloadPacket,
) -> torch._ops.OpOverload:
    # 获取操作符的所有模式
    all_schemas = torch._C._jit_get_schemas_for_operator(op._qualified_op_name)
    # 如果模式数量不为1，则抛出运行时错误
    if len(all_schemas) != 1:
        raise RuntimeError(
            f"opcheck can only test operators without overloads. "
            f"Got the following overloads for {op._qualified_op_name}: "
            f"{[schema.overload_name for schema in all_schemas]}"
        )

    # 获取第一个模式的重载名称
    overload_name = all_schemas[0].overload_name
    # 如果重载名称为空字符串，则返回默认的操作符重载
    if overload_name == "":
        return op.default
    # 否则，返回指定重载名称的操作符重载
    return getattr(op, overload_name)


# 定义用于 JSON 序列化的选项
DUMP_OPTIONS = {"indent": 2, "sort_keys": True}


# 定义包含失败数据的字典类型
FailuresDictData = Dict[str, Dict[str, Dict[str, str]]]


# 定义版本号和描述信息
VERSION = 1
DESCRIPTION = (
    f"This is a dict containing failures for tests autogenerated by "
    f"generate_opcheck_tests. "
    f"For more details, please see {GDOC}"
)


# 定义 FailuresDict 类，用于管理失败数据字典的读取和保存
class FailuresDict:
    def __init__(self, path: str, data: FailuresDictData):
        self.path = path
        self.data = data

    @staticmethod
    def load(path, *, create_file=False) -> "FailuresDict":
        # 如果指定创建文件且路径不存在，则创建一个新的空 FailuresDict 对象并保存
        if create_file and not os.path.exists(path):
            result = FailuresDict(path, {})
            FailuresDict.save()
            return result
        # 否则，从文件中加载数据
        with open(path) as fp:
            contents = fp.read()
            # 如果文件内容为空，则初始化一个新的数据结构
            if contents.strip() == "":
                dct = {
                    "_description": DESCRIPTION,
                    "data": {},
                    "_version": VERSION,
                }
            else:
                dct = json.loads(contents)
                # 确保数据结构包含必要的字段
                assert "data" in dct
                assert "_version" in dct and dct["_version"] == VERSION
        return FailuresDict(path, dct["data"])

    def _save(self, to_str=False) -> Optional[str]:
        # 准备要序列化的数据结构
        to_dump = {
            "_description": DESCRIPTION,
            "data": self.data,
            "_version": VERSION,
        }
        # 使用 JSON 序列化，并在末尾添加一个换行符以符合文件结尾的要求
        serialized = json.dumps(to_dump, **DUMP_OPTIONS) + "\n"
        if to_str:
            return serialized
        # 将序列化后的数据写入文件
        with open(self.path, "w") as fp:
            fp.write(serialized)
        return None

    def save(self) -> None:
        # 调用 _save 方法保存数据到文件
        return self._save()

    def get_status(self, qualname: str, test_name: str) -> str:
        # 获取指定测试的状态，如果不存在则返回默认状态 "xsuccess"
        if qualname not in self.data:
            return "xsuccess"
        dct = self.data[qualname]
        if test_name not in dct:
            return "xsuccess"
        return dct[test_name]["status"]

    def set_status(
        self,
        qualname: str,
        test_name: str,
        status: str,
        *,
        comment: Optional[str] = None,
    ):
        # 如果测试函数的完全限定名称不在数据字典中，就添加一个空字典作为值
        if qualname not in self.data:
            self.data[qualname] = {}
        
        # 取出当前测试函数的字典
        dct = self.data[qualname]
        
        # 如果测试名称不在当前测试函数字典中，就添加一个具有默认状态和空评论的字典项
        if test_name not in dct:
            dct[test_name] = {"status": None, "comment": ""}
        
        # 根据状态值来更新测试的状态和评论信息
        if status == "xsuccess":
            # 如果状态为"xsuccess"，则删除当前测试函数的这个测试名称的字典项
            del dct[test_name]
        else:
            # 否则更新测试名称的状态和评论信息
            dct[test_name]["status"] = status
            if comment is not None:
                dct[test_name]["comment"] = comment
```