# `.\pytorch\torch\testing\_internal\common_modules.py`

```py
# 忽略类型检查错误，用于类型检查工具mypy
# 导入PyTorch库
import torch
# 导入单元测试模块
import unittest
# 从标准库中导入深拷贝函数
from copy import deepcopy
# 导入枚举类型
from enum import Enum
# 导入装饰器相关函数
from functools import wraps, partial
# 导入迭代器相关函数
from itertools import chain, product
# 导入标准库的迭代器模块
import itertools
# 导入数学函数库
import math
# 导入PyTorch的函数式接口模块
import torch.nn.functional as F
# 导入PyTorch中的RNN相关函数
from torch.nn.utils.rnn import pack_padded_sequence
# 导入PyTorch的张量生成函数
from torch.testing import make_tensor
# 导入PyTorch CUDA测试相关模块
from torch.testing._internal.common_cuda import TEST_CUDNN
# 导入PyTorch数据类型相关模块
from torch.testing._internal.common_dtype import (
    floating_types, floating_and_complex_types_and, get_all_fp_dtypes)
# 导入PyTorch设备类型相关模块
from torch.testing._internal.common_device_type import (
    _TestParametrizer, _update_param_kwargs, toleranceOverride, tol,
    skipCUDAIfCudnnVersionLessThan, skipCUDAIfRocm, precisionOverride, skipMeta, skipMPS, skipCUDAVersionIn)
# 导入PyTorch方法调用的装饰器相关模块
from torch.testing._internal.common_methods_invocations import DecorateInfo
# 导入PyTorch的常用神经网络测试相关模块
from torch.testing._internal.common_nn import (
    cosineembeddingloss_reference, cross_entropy_loss_reference, ctcloss_reference,
    hingeembeddingloss_reference, huberloss_reference, kldivloss_reference,
    marginrankingloss_reference, multimarginloss_reference, multilabelmarginloss_reference,
    nllloss_reference, nlllossNd_reference, smoothl1loss_reference, softmarginloss_reference, get_reduction)
# 导入PyTorch的常用工具函数模块
from torch.testing._internal.common_utils import (
    freeze_rng_state, skipIfMps, GRADCHECK_NONDET_TOL, TEST_WITH_ROCM, IS_WINDOWS,
    skipIfTorchDynamo)
# 导入模块类型相关模块
from types import ModuleType
# 导入类型提示相关模块
from typing import List, Tuple, Type, Set, Dict
# 导入运算符模块
import operator

# 所有要测试的模块的命名空间列表
MODULE_NAMESPACES: List[ModuleType] = [
    torch.nn.modules,
    torch.ao.nn.qat.modules,
    torch.ao.nn.quantizable.modules,
    torch.ao.nn.quantized.modules,
    torch.ao.nn.quantized.modules,
]

# 不希望进行测试的模块集合
MODULES_TO_SKIP: Set[Type] = {
    torch.nn.Module,  # 抽象基类
    torch.nn.Container,  # 已弃用
    torch.nn.NLLLoss2d,  # 已弃用
    torch.ao.nn.quantized.MaxPool2d,  # 别名为 nn.MaxPool2d
    torch.ao.nn.quantized.MaxPool2d,  # 别名为 nn.MaxPool2d
}

# 所有要测试的模块类列表
MODULE_CLASSES: List[Type] = list(chain(*[
    [getattr(namespace, module_name) for module_name in namespace.__all__]  # type: ignore[attr-defined]
    for namespace in MODULE_NAMESPACES]))
# 过滤掉不需要测试的模块类
MODULE_CLASSES = [cls for cls in MODULE_CLASSES if cls not in MODULES_TO_SKIP]

# 模块类到常用名称的映射字典，有助于使测试名称更直观
# 例如：torch.nn.modules.linear.Linear -> "nn.Linear"
MODULE_CLASS_NAMES: Dict[Type, str] = {}
for namespace in MODULE_NAMESPACES:
    # 遍历命名空间中的所有模块名称，这些模块名称通常被指定为可导出的
    for module_name in namespace.__all__:  # type: ignore[attr-defined]
        # 获取命名空间中指定名称的模块类对象
        module_cls = getattr(namespace, module_name)
        # 提取命名空间的名称，并移除可能存在的'torch.'和'.modules'部分
        namespace_name = namespace.__name__.replace('torch.', '').replace('.modules', '')

        # 处理可能存在的别名，优先使用先前的名称
        if module_cls not in MODULE_CLASS_NAMES:
            # 将模块类对象及其对应的命名空间和模块名称添加到全局的模块类名称映射中
            MODULE_CLASS_NAMES[module_cls] = f'{namespace_name}.{module_name}'
# 定义一个枚举类型 TrainEvalMode，包含三个选项：train_only、eval_only、train_and_eval
TrainEvalMode = Enum('TrainEvalMode', ('train_only', 'eval_only', 'train_and_eval'))

# 定义一个类 modules，继承自 _TestParametrizer 类
class modules(_TestParametrizer):
    """ PROTOTYPE: Decorator for specifying a list of modules over which to run a test. """

    # 初始化方法，接受模块信息可迭代对象、允许的数据类型集合、训练评估模式和是否跳过动态的参数
    def __init__(self, module_info_iterable, allowed_dtypes=None,
                 train_eval_mode=TrainEvalMode.train_and_eval, skip_if_dynamo=True):
        # 将模块信息可迭代对象转换为列表
        self.module_info_list = list(module_info_iterable)
        # 如果指定了允许的数据类型集合，则转换为集合类型，否则为 None
        self.allowed_dtypes = set(allowed_dtypes) if allowed_dtypes is not None else None
        # 设置训练评估模式，默认为 TrainEvalMode.train_and_eval
        self.train_eval_mode = train_eval_mode
        # 是否跳过动态，默认为 True
        self.skip_if_dynamo = skip_if_dynamo

    # 内部方法，用于获取训练标志列表
    def _get_training_flags(self, module_info):
        # 初始化训练标志列表
        training_flags = []
        
        # 如果训练评估模式是 train_only 或 train_and_eval，则添加 True 到训练标志列表
        if (self.train_eval_mode == TrainEvalMode.train_only or
                self.train_eval_mode == TrainEvalMode.train_and_eval):
            training_flags.append(True)

        # 如果训练评估模式是 eval_only 或 train_and_eval，则添加 False 到训练标志列表
        if (self.train_eval_mode == TrainEvalMode.eval_only or
                self.train_eval_mode == TrainEvalMode.train_and_eval):
            training_flags.append(False)

        # 如果模块不区分训练和评估模式，则只保留一个训练标志
        if not module_info.train_and_eval_differ:
            training_flags = training_flags[:1]

        # 返回训练标志列表
        return training_flags
    # 定义一个参数化测试方法，接受三个参数：测试方法、泛型类和设备类
    def _parametrize_test(self, test, generic_cls, device_cls):
        # 如果设备类为 None，则抛出运行时错误，指示 @modules 装饰器只能在设备特定的上下文中使用
        if device_cls is None:
            raise RuntimeError('The @modules decorator is only intended to be used in a device-specific '
                               'context; use it with instantiate_device_type_tests() instead of '
                               'instantiate_parametrized_tests()')

        # 遍历存储的模块信息列表
        for module_info in self.module_info_list:
            # 获取特定设备类型支持的数据类型集合
            dtypes = set(module_info.supported_dtypes(device_cls.device_type))
            # 如果允许的数据类型不为 None，则取其交集
            if self.allowed_dtypes is not None:
                dtypes = dtypes.intersection(self.allowed_dtypes)

            # 获取模块的训练标志列表
            training_flags = self._get_training_flags(module_info)
            # 对每种训练标志和数据类型进行组合
            for (training, dtype) in product(training_flags, dtypes):
                # 构建测试名称；设备和数据类型部分在外部处理
                # 参见 [Note: device and dtype suffix placement]
                test_name = module_info.formatted_name
                if len(training_flags) > 1:
                    test_name += f"_{'train_mode' if training else 'eval_mode'}"

                # 构建传递给测试方法的参数 kwargs
                param_kwargs = {'module_info': module_info}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)
                _update_param_kwargs(param_kwargs, 'training', training)

                try:
                    # 定义测试方法的包装器，用于处理测试方法的执行
                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    # 如果设置了 self.skip_if_dynamo，并且未在 TorchDynamo 环境下运行，则使用 skipIfTorchDynamo 装饰器跳过测试
                    if self.skip_if_dynamo and not torch.testing._internal.common_utils.TEST_WITH_TORCHINDUCTOR:
                        test_wrapper = skipIfTorchDynamo("Policy: we don't run ModuleInfo tests w/ Dynamo")(test_wrapper)

                    # 定义装饰器函数，用于获取模块信息的装饰器
                    decorator_fn = partial(module_info.get_decorators, generic_cls.__name__,
                                           test.__name__, device_cls.device_type, dtype)

                    # 使用 yield 生成器返回测试方法的包装器、测试名称、参数 kwargs 和装饰器函数
                    yield (test_wrapper, test_name, param_kwargs, decorator_fn)
                except Exception as ex:
                    # 在重新抛出异常之前，提供调试用的错误消息
                    print(f"Failed to instantiate {test_name} for module {module_info.name}!")
                    raise ex
# 定义一个函数，用于获取模块类的通用名称
def get_module_common_name(module_cls):
    # 如果模块类存在于预定义的 MODULE_CLASS_NAMES 中，则返回其对应的通用名称
    if module_cls in MODULE_CLASS_NAMES:
        # 示例返回格式为 "nn.Linear"
        return MODULE_CLASS_NAMES[module_cls]
    else:
        # 否则返回模块类的名称
        return module_cls.__name__


# 定义一个类 FunctionInput，用于存储传递给函数的参数和关键字参数
class FunctionInput:
    """ Contains args and kwargs to pass as input to a function. """
    __slots__ = ['args', 'kwargs']

    def __init__(self, *args, **kwargs):
        # 初始化对象时，接收并存储传入的位置参数和关键字参数
        self.args = args
        self.kwargs = kwargs


# 定义一个类 ModuleInput，用于存储模块实例化和前向传播时的参数和描述信息
class ModuleInput:
    """ Contains args / kwargs for module instantiation + forward pass. """
    __slots__ = ['constructor_input', 'forward_input', 'desc', 'reference_fn']

    def __init__(self, constructor_input, forward_input=None, desc='', reference_fn=None):
        # 初始化对象时，接收构造函数的输入、前向传播的输入（可选）、描述信息和参考函数
        self.constructor_input = constructor_input  # 构造函数的输入参数
        self.forward_input = forward_input  # forward() 方法的输入参数
        self.desc = desc  # 此输入集合的描述信息
        self.reference_fn = reference_fn  # 带有特定签名的参考函数

        # 如果指定了 reference_fn，则创建其副本，以避免调用时产生意外的副作用
        if reference_fn is not None:

            @wraps(reference_fn)
            def copy_reference_fn(m, *args, **kwargs):
                # 使用 deepcopy 复制输入，以避免不必要的副作用
                args, kwargs = deepcopy(args), deepcopy(kwargs)

                # 注意，模块参数被传递进来以便使用
                return reference_fn(m, list(m.parameters()), *args, **kwargs)

            # 将副本函数赋值给 self.reference_fn
            self.reference_fn = copy_reference_fn


# 定义一个枚举类 ModuleErrorEnum，用于标识在测试模块时引发的错误类型
class ModuleErrorEnum(Enum):
    """ Enumerates when error is raised when testing modules. """
    CONSTRUCTION_ERROR = 0  # 模块构造时出错
    FORWARD_ERROR = 1  # 模块前向传播时出错


# 定义一个类 ErrorModuleInput，继承自 ModuleInput，用于标识在操作中引发错误的模块输入及相关信息
class ErrorModuleInput:
    """
    A ModuleInput that will cause the operation to throw an error plus information
    about the resulting error.
    """

    __slots__ = ["module_error_input", "error_on", "error_type", "error_regex"]

    def __init__(self,
                 module_error_input,
                 *,
                 error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
                 error_type=RuntimeError,
                 error_regex):
        # 初始化对象时，接收引起错误的模块输入、错误类型和错误正则表达式
        self.module_error_input = module_error_input  # 引起错误的模块输入
        self.error_on = error_on  # 错误发生的阶段（构造或前向传播）
        self.error_type = error_type  # 抛出的错误类型
        self.error_regex = error_regex  # 用于匹配错误消息的正则表达式


# 定义一个类 ModuleInfo，用于存储在测试中使用的模块信息
class ModuleInfo:
    """ Module information to be used in testing. """
    # 初始化函数，用于设置测试模块的各种参数和属性
    def __init__(self,
                 module_cls,  # 被测试模块的类对象
                 *,
                 module_inputs_func,  # 生成模块输入的函数
                 skips=(),  # 需要跳过的测试集
                 decorators=None,  # 应用于生成测试的额外装饰器
                 dtypes=floating_types(),  # 函数预期使用的数据类型
                 dtypesIfMPS=(torch.float16, torch.float32,),  # MPS（神经处理器）上函数预期使用的数据类型
                 supports_gradgrad=True,  # 是否支持二阶梯度
                 gradcheck_nondet_tol=0.0,  # 执行 gradcheck 时的非确定性容差
                 module_memformat_affects_out=False,  # 是否转换模块为通道最后格式会生成通道最后的输出
                 train_and_eval_differ=False,  # 模块在训练和评估时是否有不同的行为
                 module_error_inputs_func=None,  # 生成错误模块输入的函数
                 ):
        self.module_cls = module_cls  # 初始化被测试模块的类对象
        self.module_inputs_func = module_inputs_func  # 初始化生成模块输入的函数
        self.decorators = (*(decorators if decorators else []), *(skips if skips else []))  # 初始化装饰器列表，包括传入的装饰器和需要跳过的测试集
        self.dtypes = dtypes  # 初始化预期的数据类型
        self.dtypesIfMPS = dtypesIfMPS  # 初始化在MPS上预期的数据类型
        self.supports_gradgrad = supports_gradgrad  # 初始化是否支持二阶梯度
        self.gradcheck_nondet_tol = gradcheck_nondet_tol  # 初始化 gradcheck 的非确定性容差
        self.module_memformat_affects_out = module_memformat_affects_out  # 初始化模块内存格式是否影响输出格式
        self.train_and_eval_differ = train_and_eval_differ  # 初始化训练和评估时模块是否有不同行为
        self.module_error_inputs_func = module_error_inputs_func  # 初始化生成错误模块输入的函数
        self.is_lazy = issubclass(module_cls, torch.nn.modules.lazy.LazyModuleMixin)  # 检查模块是否为懒加载模块

    # 获取测试用例的装饰器列表
    def get_decorators(self, test_class, test_name, device, dtype, param_kwargs):
        result = []
        for decorator in self.decorators:
            if isinstance(decorator, DecorateInfo):  # 如果装饰器是 DecorateInfo 类型
                if decorator.is_active(test_class, test_name, device, dtype, param_kwargs):  # 如果装饰器在当前测试条件下是活跃的
                    result.extend(decorator.decorators)  # 将装饰器列表扩展到结果列表中
            else:
                result.append(decorator)  # 否则直接添加装饰器到结果列表中
        return result  # 返回装饰器列表作为结果

    # 根据设备类型返回支持的数据类型列表
    def supported_dtypes(self, device_type):
        if device_type == 'mps':  # 如果设备类型为 'mps'
            return self.dtypesIfMPS  # 返回 MPS 上预期的数据类型列表
        else:
            return self.dtypes  # 否则返回一般情况下预期的数据类型列表

    # 返回被测试模块的通用名称
    @property
    def name(self):
        return get_module_common_name(self.module_cls)

    # 返回被测试模块的格式化名称（将 '.' 替换为 '_'）
    @property
    def formatted_name(self):
        return self.name.replace('.', '_')
# 定义一个函数，生成部分张量（tensor），设定设备、数据类型、是否需要梯度等参数
def module_inputs_torch_nn_Linear(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数，固定设备、数据类型、是否需要梯度的参数，生成make_input函数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 定义模块输入列表，包含三个ModuleInput对象
    module_inputs = [
        ModuleInput(
            constructor_input=FunctionInput(10, 8),  # 线性层构造函数的输入参数
            forward_input=FunctionInput(input=make_input((4, 10))),  # 前向输入的参数
            reference_fn=lambda m, p, input: torch.mm(input, p[0].t()) + p[1].view(1, -1).expand(4, 8)  # 参考函数的定义
        ),
        ModuleInput(
            constructor_input=FunctionInput(10, 8, bias=False),  # 不带偏置的线性层构造函数的输入参数
            forward_input=FunctionInput(make_input((4, 10))),  # 前向输入的参数
            desc='no_bias',  # 描述信息
            reference_fn=lambda m, p, i: torch.mm(i, p[0].t())  # 参考函数的定义
        ),
        ModuleInput(
            constructor_input=FunctionInput(3, 5),  # 线性层构造函数的输入参数
            forward_input=FunctionInput(make_input(3)),  # 前向输入的参数
            desc='no_batch_dim',  # 描述信息
            reference_fn=lambda m, p, i: torch.mm(i.view(1, -1), p[0].t()).view(-1) + p[1]  # 参考函数的定义
        )
    ]

    return module_inputs  # 返回模块输入列表


# 定义一个函数，生成部分张量（tensor），设定设备、数据类型、是否需要梯度等参数
def module_inputs_torch_nn_Bilinear(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数，固定设备、数据类型、是否需要梯度的参数，生成make_input函数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 定义双线性函数的参考函数
    def bilinear_reference_fn(m, p, x1, x2, bias=True):
        # 使用 Einstein 求和约定计算双线性函数的结果
        result = torch.einsum('bn,anm,bm->ba', x1, p[0], x2)
        if bias:
            if x1.shape[0] == 1:
                result = result.view(-1) + p[1]  # 如果 x1 的行数为 1，则直接加上偏置参数 p[1]
            else:
                result = result + p[1].view(1, -1).expand(x1.shape[0], p[0].shape[0])  # 否则，扩展偏置参数并加到结果中
        return result  # 返回结果

    # 定义模块输入列表，包含三个ModuleInput对象
    module_inputs = [
        ModuleInput(
            constructor_input=FunctionInput(2, 3, 4),  # 双线性层构造函数的输入参数
            forward_input=FunctionInput(make_input((8, 2)), make_input((8, 3))),  # 前向输入的参数
            reference_fn=bilinear_reference_fn  # 参考函数的定义
        ),
        ModuleInput(
            constructor_input=FunctionInput(2, 3, 4, bias=False),  # 不带偏置的双线性层构造函数的输入参数
            forward_input=FunctionInput(make_input((8, 2)), make_input((8, 3))),  # 前向输入的参数
            desc='no_bias',  # 描述信息
            reference_fn=lambda m, p, x1, x2: bilinear_reference_fn(m, p, x1, x2, bias=False)  # 参考函数的定义
        ),
        ModuleInput(
            constructor_input=FunctionInput(2, 3, 4),  # 双线性层构造函数的输入参数
            forward_input=FunctionInput(make_input(2), make_input(3)),  # 前向输入的参数
            desc='no_batch_dim',  # 描述信息
            reference_fn=lambda m, p, x1, x2: bilinear_reference_fn(m, p, x1.view(1, -1), x2.view(1, -1))  # 参考函数的定义
        )
    ]

    return module_inputs  # 返回模块输入列表


# 定义一个函数，生成部分张量（tensor），设定设备、数据类型、是否需要梯度等参数
def module_inputs_torch_nn_KLDivLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数，固定设备、数据类型、是否需要梯度的参数，生成make_input函数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 定义不同情况下的参数列表
    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_batchmean', {'reduction': 'batchmean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('log_target', {'log_target': True})
    ]

    module_inputs = []  # 初始化模块输入列表为空
    # 对于每个测试案例中的描述和构造函数参数，执行以下操作：
    for desc, constructor_kwargs in cases:
        # 定义一个参考函数，使用给定的构造函数参数调用 Kullback-Leibler 散度损失函数
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return kldivloss_reference(i, t, **constructor_kwargs)

        # 创建大小为 (10, 10) 的输入并取其对数，作为输入数据
        input = make_input((10, 10)).log()
        # 如果设置了 'log_target' 参数为 True，则创建大小为 (10, 10) 的目标数据；否则对目标数据取对数
        target = make_input((10, 10)) if kwargs.get('log_target', False) else make_input((10, 10)).log()
        # 将模块的输入数据添加到列表中
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),  # 使用给定的构造函数参数作为构造输入
                forward_input=FunctionInput(input, target),  # 使用处理后的输入和目标数据作为前向输入
                desc=desc,  # 将当前测试案例的描述作为描述信息
                reference_fn=reference_fn  # 将定义的参考函数作为参考函数
            )
        )

        # 创建大小为 () 的标量输入并取其对数，作为输入数据
        scalar_input = make_input(()).log()
        # 如果设置了 'log_target' 参数为 True，则创建大小为 () 的目标数据；否则对目标数据取对数
        scalar_target = make_input(()) if kwargs.get('log_target', False) else make_input(()).log()
        # 将标量模块的输入数据添加到列表中
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),  # 使用给定的构造函数参数作为构造输入
                forward_input=FunctionInput(scalar_input, scalar_input),  # 使用处理后的标量输入作为前向输入
                desc='scalar_' + desc,  # 使用 'scalar_' 加上当前测试案例的描述作为描述信息
                reference_fn=reference_fn  # 将定义的参考函数作为参考函数
            )
        )

    # 返回构建好的模块输入列表
    return module_inputs
# 定义一个函数，用于生成指定形状的张量，并进行对数softmax处理，并根据参数决定是否需要梯度
def module_inputs_torch_nn_NLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 定义一个内部函数，用于生成指定形状的张量，设备和数据类型可选，默认不需要梯度
    def make_input(shape, device=device, dtype=dtype, requires_grad=requires_grad):
        return make_tensor(shape, device=device, dtype=dtype,
                           requires_grad=False).log_softmax(dim=1).requires_grad_(requires_grad)
    
    # 使用偏函数定义一个函数，用于生成指定设备和数据类型的张量，并设置不需要梯度
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # 定义一个包含不同情况的列表，每个元素是一个元组，包含一个字符串和一个字典作为参数
    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_none', {'reduction': 'none'}),
        ('ignore_index', {'ignore_index': 2}),
        ('weights', {'weight': make_weight(4).abs()}),
        ('weights_ignore_index', {'weight': make_weight(4).abs(), 'ignore_index': 2}),
        ('weights_ignore_index_neg', {'weight': make_weight(4).abs(), 'ignore_index': -1})
    ]

    # TODO: Uncomment when negative weights is supported.
    # 定义一个负权重的情况，暂时注释掉，因为不支持负权重
    # negative_weight = make_weight(10)
    # negative_weight[0] = -1
    # cases.append(('weights_negative', {'weight': negative_weight}))
    
    # 初始化模块输入为空列表
    module_inputs = []
    # 对于每个测试用例，从 cases 中获取描述和构造函数参数
    for desc, constructor_kwargs in cases:
        
        # 定义一个参考函数，用于计算负对数似然损失，使用给定的构造函数参数
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return nllloss_reference(i, t, **constructor_kwargs)

        # 将模块输入添加到列表中，包括构造函数的输入、前向输入、描述和参考函数
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((15, 4)),
                                                    torch.empty(15, device=device).uniform_().mul(4).floor().long()),
                        desc=desc,
                        reference_fn=reference_fn)
        )

        # 定义另一个参考函数，用于多维情况下的负对数似然损失，使用相同的构造函数参数
        def nd_reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return nlllossNd_reference(i, t, **constructor_kwargs)

        # 将另一个模块输入添加到列表中，包括构造函数的输入、多维前向输入、描述和参考函数
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(
                            make_input((2, 4, 5, 5)),
                            torch.empty(2, 5, 5, device=device).uniform_().mul(4).floor().long()),
                        desc=f"nd_{desc}",
                        reference_fn=nd_reference_fn)
        )

        # 将另一个模块输入添加到列表中，包括构造函数的输入、更高维度的前向输入、描述和相同的参考函数
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(
                            make_input((2, 4, 5, 5, 2, 2)),
                            torch.empty(2, 5, 5, 2, 2, device=device).uniform_().mul(4).floor().long()),
                        desc=f"higher_dim_{desc}",
                        reference_fn=nd_reference_fn)
        )

        # 将另一个模块输入添加到列表中，包括构造函数的输入、三维前向输入、描述和相同的参考函数
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(
                            make_input((2, 4, 5)),
                            torch.empty(2, 5, device=device).uniform_().mul(4).floor().long()),
                        desc=f"3d_{desc}",
                        reference_fn=nd_reference_fn)
        )

    # 返回所有模块输入的列表
    return module_inputs
# 定义一个函数 module_inputs_torch_nn_GaussianNLLLoss，接受多个参数并返回一个列表 module_inputs
def module_inputs_torch_nn_GaussianNLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用 functools.partial 创建 make_input 函数的特定版本，固定了部分参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 使用 functools.partial 创建 make_target 函数的特定版本，固定了部分参数
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # 定义一个列表 cases，包含多个元组，每个元组包含描述和构造函数参数的字典
    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
    ]

    # 初始化一个空列表 module_inputs
    module_inputs = []
    # 遍历 cases 列表
    for desc, constructor_kwargs in cases:
        # 向 module_inputs 列表添加一个 ModuleInput 对象，包含构造函数输入、前向输入、描述和参考函数
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input(3),
                                                    make_target(3),
                                                    make_input(1).abs()),
                        desc=desc,
                        reference_fn=no_batch_dim_reference_fn)
        )

    # 返回 module_inputs 列表作为函数的结果
    return module_inputs


# 定义一个函数 module_inputs_torch_nn_PoissonNLLLoss，接受多个参数并返回一个列表 module_inputs
def module_inputs_torch_nn_PoissonNLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用 functools.partial 创建 make_input 函数的特定版本，固定了部分参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 使用 functools.partial 创建 make_target 函数的特定版本，固定了部分参数
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # 定义一个列表 cases，包含多个元组，每个元组包含描述和构造函数参数的字典
    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('full', {'full': True}),
        ('no_log_input', {'log_input': False}),
        ('full_no_log_input', {'full': True, 'log_input': False}),
    ]

    # 定义一个函数 poissonnllloss_reference_fn，接受多个参数并返回计算结果
    def poissonnllloss_reference_fn(i, t, log_input=True, full=False, reduction='mean', eps=1e-8):
        # 如果 log_input 为真，则计算结果为 i 的指数减去 t 与 i 的乘积
        if log_input:
            result = i.exp() - t.mul(i)
        else:
            result = i - t.mul((i + eps).log())

        # 如果 full 为真，则根据条件向结果添加额外的数学运算
        if full:
            result += (t.mul(t.log()) - t + 0.5 * (2. * math.pi * t).log()).masked_fill(t <= 1, 0)

        # 根据 reduction 的值返回不同的结果
        if reduction == 'none':
            return result
        elif reduction == 'mean':
            return result.sum() / i.numel()
        else:
            return result.sum()

    # 初始化一个空列表 module_inputs
    module_inputs = []
    # 遍历 cases 列表
    for desc, constructor_kwargs in cases:
        # 定义一个内部函数 reference_fn，调用 poissonnllloss_reference_fn 函数，并传递适当的参数
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return poissonnllloss_reference_fn(i, t, **constructor_kwargs)

        # 获取 log_input 参数值，默认为 True
        log_input = constructor_kwargs.get('log_input', True)
        # 根据 log_input 的值选择不同的输入张量 input
        input = make_input((2, 3, 4, 5)) if log_input else make_input((2, 3, 4, 5)).abs().add(0.001)
        # 向 module_inputs 列表添加一个 ModuleInput 对象，包含构造函数输入、前向输入、描述和参考函数
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(input,
                                                    make_target((2, 3, 4, 5)).floor_().abs_()),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    # 返回 module_inputs 列表作为函数的结果
    return module_inputs
def module_inputs_torch_nn_MSELoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个局部函数 make_input，用于生成具有指定设备、数据类型和梯度属性的张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 创建一个局部函数 make_target，用于生成具有指定设备和数据类型的张量，但不需要梯度
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # 定义测试用例列表，每个测试用例包含一个描述字符串和构造函数的关键字参数字典
    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
    ]

    # 定义均方误差损失的参考函数，根据指定的减少(reduction)方式计算损失
    def mse_loss_reference_fn(m, p, i, t, reduction='mean'):
        if reduction == 'none':
            return (i - t).pow(2)  # 返回每个元素的平方差，未进行任何减少
        elif reduction == 'mean':
            return (i - t).pow(2).sum() / i.numel()  # 返回平均值，即平方差总和除以元素个数
        else:
            return (i - t).pow(2).sum()  # 返回平方差的总和

    # 初始化模块输入列表
    module_inputs = []
    # 遍历测试用例
    for desc, constructor_kwargs in cases:
        # 添加模块输入，包括构造函数输入、前向传播输入、描述和参考函数
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((2, 3, 4, 5)),
                                                    make_target((2, 3, 4, 5))),
                        desc=desc,
                        reference_fn=partial(mse_loss_reference_fn, **constructor_kwargs))
        )
        # 添加标量版本的模块输入
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input(()),
                                                    make_target(())),
                        desc=f'{desc}_scalar',
                        reference_fn=partial(mse_loss_reference_fn, **constructor_kwargs))
        )

    return module_inputs


def no_batch_dim_reference_fn(m, p, *args, **kwargs):
    """Reference function for modules supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.

    Currently it only supports modules which return a single Tensor as output.
    You can bind the following kwargs.
    Kwargs:
        batch_first[bool] : If True, all the Tensors in `args` while be unsqueezed at dim `0` .
                        and output will be squeezed at dim `0` else dim `1` for both.
        kwargs_to_batchify[dict] : Dictionary specifying the name of the argument and dimension to unsqueeze.
                               Useful if there are few arguments whose batch dimension are different
                               from the ones selected by `batch_first`.
        is_criterion[bool] : Specify if the module is a criterion and handle the reduction for output accordingly.
    """
    # 定义一个内部函数，用于获取并弹出指定键的值，如果键不存在则返回默认值
    def get_and_pop(key, default):
        v = kwargs.get(key, default)
        if key in kwargs:
            kwargs.pop(key)
        return v

    # 根据 batch_first 键的值确定批次维度的位置，如果为 True，则批次维度在最前面，否则在第二维
    batch_dim = 0 if get_and_pop('batch_first', True) else 1
    # 获取 kwargs_to_batchify 键的值，指定参数名称及其需要插入的维度
    kwargs_to_batchify = get_and_pop('kwargs_to_batchify', None)
    # 从参数中获取名为'is_criterion'的值，如果不存在则默认为False
    is_criterion = get_and_pop('is_criterion', False)

    # 如果kwargs_to_batchify不为None，则验证其类型为字典
    if kwargs_to_batchify is not None:
        assert isinstance(kwargs_to_batchify, dict)
        # 遍历kwargs中的键值对
        for k, v in kwargs.items():
            # 如果当前键k在kwargs_to_batchify中，并且对应的值v不为None
            if k in kwargs_to_batchify and v is not None:
                # 获取该键对应的批处理维度
                bdim = kwargs_to_batchify[k]
                # 将v在维度bdim上进行unsqueeze操作，并更新kwargs中的值
                kwargs[k] = v.unsqueeze(bdim)

    # 将args中的每个元素input在batch_dim维度上进行unsqueeze操作，并组成列表
    single_batch_input_args = [input.unsqueeze(batch_dim) for input in args]

    # 冻结随机数生成器的状态，并在此状态下调用函数m，传入单批输入参数和kwargs
    with freeze_rng_state():
        # 调用函数m，传入单批输入参数和kwargs，并在batch_dim维度上进行squeeze操作，得到output
        output = m(*single_batch_input_args, **kwargs).squeeze(batch_dim)

    # 如果is_criterion为True
    if is_criterion:
        # 获取函数m的减少(reduction)方式
        reduction = get_reduction(m)
        # 如果reduction为'none'，则返回output在第0维度上进行squeeze操作后的结果
        if reduction == 'none':
            return output.squeeze(0)
    
    # 返回output作为函数的输出
    return output
# MultiheadAttention 的参考函数，支持无批量维度输入

# 如果 batch_first 关键字参数存在，则确定批量维度为 0；否则为 1
batch_dim = 0 if kwargs.get('batch_first', True) else 1
# 如果 'batch_first' 在 kwargs 中，移除它
if 'batch_first' in kwargs:
    kwargs.pop('batch_first')
# 如果 'key_padding_mask' 在 kwargs 中且不为 None，则在第 0 维度上增加维度
if 'key_padding_mask' in kwargs and kwargs['key_padding_mask'] is not None:
    kwargs['key_padding_mask'] = kwargs['key_padding_mask'].unsqueeze(0)
# 将所有输入的无批量维度的参数 unsqueeze 到单批次输入参数中
single_batch_input_args = [input.unsqueeze(batch_dim) for input in args]
# 冻结随机数生成器状态，执行模块 m 的计算，并返回挤压后的输出以进行比较
with freeze_rng_state():
    output = m(*single_batch_input_args, **kwargs)
    # 返回第一个输出挤压掉批量维度，以及第二个输出挤压掉第 0 维度
    return (output[0].squeeze(batch_dim), output[1].squeeze(0))


# RNN 和 GRU 的参考函数，支持无批量维度输入

# 如果输入参数 args 的长度为 1，则将其解包为 inp，并且隐藏状态 h 设为 None
if len(args) == 1:
    inp, = args
    h = None
# 如果输入参数 args 的长度为 2，则将其解包为 inp 和 h，并在第 1 维度上增加隐藏状态 h 的维度
elif len(args) == 2:
    inp, h = args
    h = h.unsqueeze(1)

# 根据 batch_first 关键字参数确定批量维度为 0 或 1，并移除该关键字参数
batch_dim = 0 if kwargs['batch_first'] else 1
kwargs.pop('batch_first')
# 在输入参数 inp 上增加批量维度
inp = inp.unsqueeze(batch_dim)
# 将输入参数和隐藏状态组成单批次输入参数
single_batch_input_args = (inp, h)
# 冻结随机数生成器状态，执行模块 m 的计算，并返回挤压后的输出以进行比较
with freeze_rng_state():
    output = m(*single_batch_input_args, **kwargs)
    # 返回第一个输出挤压掉批量维度，以及第二个输出挤压掉第 1 维度
    return (output[0].squeeze(batch_dim), output[1].squeeze(1))


# LSTM 的参考函数，支持无批量维度输入

# 如果输入参数 args 的长度为 1，则将其解包为 inp，并且隐藏状态 h 设为 None
if len(args) == 1:
    inp, = args
    h = None
# 如果输入参数 args 的长度为 2，则将其解包为 inp 和 h，并在第 1 维度上增加隐藏状态 h 的维度
elif len(args) == 2:
    inp, h = args
    h = (h[0].unsqueeze(1), h[1].unsqueeze(1))

# 根据 batch_first 关键字参数确定批量维度为 0 或 1，并移除该关键字参数
batch_dim = 0 if kwargs['batch_first'] else 1
kwargs.pop('batch_first')
# 在输入参数 inp 上增加批量维度
inp = inp.unsqueeze(batch_dim)
# 将输入参数和隐藏状态组成单批次输入参数
single_batch_input_args = (inp, h)
# 冻结随机数生成器状态，执行模块 m 的计算，并返回挤压后的输出以进行比较
with freeze_rng_state():
    output = m(*single_batch_input_args, **kwargs)
    # 返回第一个输出挤压掉批量维度，以及第二个输出的第一个和第二个元素挤压掉第 1 维度
    return (output[0].squeeze(batch_dim), (output[1][0].squeeze(1), output[1][1].squeeze(1)))


# LSTMCell 的参考函数，支持无批量维度输入

# 将输入参数 args 解包为 inp 和隐藏状态元组 (h, c)
inp, (h, c) = args
# 将输入参数和隐藏状态组成单批次输入参数，所有参数在第 0 维度上增加维度
single_batch_input_args = (inp.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))
# 冻结随机数生成器状态，执行模块 m 的计算，并返回挤压后的输出以进行比较
with freeze_rng_state():
    output = m(*single_batch_input_args, **kwargs)
    # 返回第一个输出挤压掉第 0 维度，以及第二个输出的第一个和第二个元素挤压掉第 0 维度
    return (output[0].squeeze(0), output[1].squeeze(0))
# 生成回归标准输入的函数，接受一个生成输入函数作为参数
def generate_regression_criterion_inputs(make_input):
    # 返回一个列表，其中每个元素是一个 ModuleInput 对象
    return [
        ModuleInput(
            # 构造函数的输入，包含一个 reduction 参数
            constructor_input=FunctionInput(reduction=reduction),
            # 前向传播函数的输入，包含两个参数：一个元组和一个标量
            forward_input=FunctionInput(make_input((4, )), make_input(4,)),
            # 参考函数，部分应用于无批次维度的引用函数，设置为准则标志
            reference_fn=partial(no_batch_dim_reference_fn, is_criterion=True),
            # 描述字段，标记当前实例的类型
            desc=f'no_batch_dim_{reduction}'
        ) for reduction in ['none', 'mean', 'sum']]


# Torch 中用于 AvgPool1d 模块的输入生成函数
def module_inputs_torch_nn_AvgPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用了 make_tensor 函数的 make_input 函数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个列表，其中每个元素是一个 ModuleInput 对象
    return [
        ModuleInput(
            # 构造函数的输入，包含一个 kernel_size 参数
            constructor_input=FunctionInput(kernel_size=2),
            # 前向传播函数的输入，包含一个参数，尺寸为 (3, 6)
            forward_input=FunctionInput(make_input((3, 6))),
            # 描述字段，标记当前实例的类型
            desc='no_batch_dim',
            # 参考函数，用于无批次维度的引用函数
            reference_fn=no_batch_dim_reference_fn),
        ModuleInput(
            # 构造函数的输入，包含一个整数参数
            constructor_input=FunctionInput(2),
            # 前向传播函数的输入，包含一个参数，尺寸为 (2, 3, 6)
            forward_input=FunctionInput(make_input((2, 3, 6)))),
        ModuleInput(
            # 构造函数的输入，包含两个元组参数，尺寸分别为 (2,) 和 (2,)
            constructor_input=FunctionInput((2,), (2,)),
            # 前向传播函数的输入，包含一个参数，尺寸为 (2, 3, 6)
            forward_input=FunctionInput(make_input((2, 3, 6))),
            # 描述字段，标记当前实例的类型
            desc='stride'),
        ModuleInput(
            # 构造函数的输入，包含三个整数参数，尺寸分别为 2, 2, 1
            constructor_input=FunctionInput(2, 2, 1),
            # 前向传播函数的输入，包含一个参数，尺寸为 (2, 3, 6)
            forward_input=FunctionInput(make_input((2, 3, 6))),
            # 描述字段，标记当前实例的类型
            desc='stride_pad')]


# Torch 中用于 AvgPool2d 模块的输入生成函数
def module_inputs_torch_nn_AvgPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用了 make_tensor 函数的 make_input 函数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个列表，其中每个元素是一个 ModuleInput 对象
    return [
        ModuleInput(
            # 构造函数的输入，包含一个元组参数，尺寸为 (2, 2)
            constructor_input=FunctionInput((2, 2)),
            # 前向传播函数的输入，包含一个参数，尺寸为 (3, 6, 6)
            forward_input=FunctionInput(make_input((3, 6, 6))),
            # 描述字段，标记当前实例的类型
            desc='no_batch_dim',
            # 参考函数，用于无批次维度的引用函数
            reference_fn=no_batch_dim_reference_fn),
        ModuleInput(
            # 构造函数的输入，包含一个元组参数，尺寸为 (2, 2)
            constructor_input=FunctionInput((2, 2)),
            # 前向传播函数的输入，包含一个参数，尺寸为 (2, 3, 6, 6)
            forward_input=FunctionInput(make_input((2, 3, 6, 6)))),
        ModuleInput(
            # 构造函数的输入，包含两个元组参数，尺寸分别为 (2, 2) 和 (2, 2)
            constructor_input=FunctionInput((2, 2), (2, 2)),
            # 前向传播函数的输入，包含一个参数，尺寸为 (2, 3, 6, 6)
            forward_input=FunctionInput(make_input((2, 3, 6, 6))),
            # 描述字段，标记当前实例的类型
            desc='stride'),
        ModuleInput(
            # 构造函数的输入，包含三个元组参数，尺寸分别为 (2, 2), (2, 2), (1, 1)
            constructor_input=FunctionInput((2, 2), (2, 2), (1, 1)),
            # 前向传播函数的输入，包含一个参数，尺寸为 (2, 3, 6, 6)
            forward_input=FunctionInput(make_input((2, 3, 6, 6))),
            # 描述字段，标记当前实例的类型
            desc='stride_pad'),
        ModuleInput(
            # 构造函数的输入，包含两个元组参数和一个 divisor_override 参数，尺寸分别为 (2, 2), (2, 2), 1
            constructor_input=FunctionInput((2, 2), divisor_override=1),
            # 前向传播函数的输入，包含一个参数，尺寸为 (2, 3, 6, 6)
            forward_input=FunctionInput(make_input((2, 3, 6, 6))),
            # 描述字段，标记当前实例的类型
            desc='divisor'),
        ModuleInput(
            # 构造函数的输入，包含三个元组参数和一个 divisor_override 参数，尺寸分别为 (2, 2), (2, 2), (1, 1), 1
            constructor_input=FunctionInput((2, 2), (2, 2), (1, 1), divisor_override=1),
            # 前向传播函数的输入，包含一个参数，尺寸为 (2, 3, 6, 6)
            forward_input=FunctionInput(make_input((2, 3, 6, 6))),
            # 描述字段，标记当前实例的类型
            desc='divisor_stride_pad')]
# 定义一个函数，生成 Torch 中 AvgPool3d 模块的输入
def module_inputs_torch_nn_AvgPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数 partial，将 make_tensor 函数设定为生成张量的函数，传入设备、数据类型、梯度需求等参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

# 定义一个函数，生成 Torch 中 AdaptiveAvgPool1d 模块的输入
def module_inputs_torch_nn_AdaptiveAvgPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数 partial，将 make_tensor 函数设定为生成张量的函数，传入设备、数据类型、梯度需求等参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    
    # 返回一个列表，包含三个 ModuleInput 对象，每个对象描述不同的输入场景
    return [
        ModuleInput(
            constructor_input=FunctionInput(3,),
            forward_input=FunctionInput(make_input((1, 3, 5))),
            desc='single'),  # 描述：单个输入场景
        ModuleInput(
            constructor_input=FunctionInput(3,),
            forward_input=FunctionInput(make_input((3, 5))),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim'),  # 描述：无批量维度的参考函数
        ModuleInput(
            constructor_input=FunctionInput(1,),
            forward_input=FunctionInput(make_input((1, 3, 5))),
            desc='one_output')  # 描述：单输出场景
    ]

# 定义一个函数，生成 Torch 中 AdaptiveAvgPool2d 模块的输入
def module_inputs_torch_nn_AdaptiveAvgPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数 partial，将 make_tensor 函数设定为生成张量的函数，传入设备、数据类型、梯度需求等参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    
    # 返回一个列表，包含多个 ModuleInput 对象，每个对象描述不同的输入场景
    return [
        ModuleInput(
            constructor_input=FunctionInput(3,),
            forward_input=FunctionInput(make_input((1, 3, 5, 6))),
            desc='single'),  # 描述：单个输入场景
        ModuleInput(
            constructor_input=FunctionInput(3,),
            forward_input=FunctionInput(make_input((3, 5, 6))),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim'),  # 描述：无批量维度的参考函数
        ModuleInput(
            constructor_input=FunctionInput(1,),
            forward_input=FunctionInput(make_input((1, 3, 5, 6))),
            desc='single_1x1output'),  # 描述：单个 1x1 输出场景
        ModuleInput(
            constructor_input=FunctionInput((3, 4)),
            forward_input=FunctionInput(make_input((1, 3, 5, 6))),
            desc='tuple'),  # 描述：元组输入场景
        ModuleInput(
            constructor_input=FunctionInput((3, None)),
            forward_input=FunctionInput(make_input((1, 3, 5, 6))),
            desc='tuple_none')  # 描述：包含 None 的元组输入场景
    ]

# 定义一个函数，生成 Torch 中 AdaptiveAvgPool3d 模块的输入
def module_inputs_torch_nn_AdaptiveAvgPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数 partial，将 make_tensor 函数设定为生成张量的函数，传入设备、数据类型、梯度需求等参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 返回一个包含多个 ModuleInput 对象的列表，每个对象描述了一个模块的输入配置
    return [
        # 创建一个 ModuleInput 对象，使用构造函数输入为长度为 3 的 FunctionInput 对象，
        # 前向输入为包含元组 (2, 3, 5, 2, 7) 的 FunctionInput 对象，描述为 'single'
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 5, 2, 7))),
                    desc='single'),
        # 创建一个 ModuleInput 对象，使用构造函数输入为长度为 3 的 FunctionInput 对象，
        # 前向输入为包含元组 (3, 5, 2, 7) 的 FunctionInput 对象，参考函数为 no_batch_dim_reference_fn，描述为 'no_batch_dim'
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5, 2, 7))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        # 创建一个 ModuleInput 对象，使用构造函数输入为形状为 (3, 4, 5) 的 FunctionInput 对象，
        # 前向输入为包含元组 (2, 3, 5, 3, 7) 的 FunctionInput 对象，描述为 'tuple'
        ModuleInput(constructor_input=FunctionInput((3, 4, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 3, 7))),
                    desc='tuple'),
        # 创建一个 ModuleInput 对象，使用构造函数输入为形状为 (None, 4, 5) 的 FunctionInput 对象，
        # 前向输入为包含元组 (2, 3, 5, 3, 7) 的 FunctionInput 对象，描述为 'tuple_none'
        ModuleInput(constructor_input=FunctionInput((None, 4, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 3, 7))),
                    desc='tuple_none'),
        # 创建一个 ModuleInput 对象，使用构造函数输入为形状为 (3, 2, 2) 的 FunctionInput 对象，
        # 前向输入为包含元组 (1, 1, 3, 2, 6) 的 FunctionInput 对象，描述为 'last_dim'
        ModuleInput(constructor_input=FunctionInput((3, 2, 2)),
                    forward_input=FunctionInput(make_input((1, 1, 3, 2, 6))),
                    desc='last_dim')
    ]
def module_inputs_torch_nn_AdaptiveMaxPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用函数，用于生成张量，设定设备、数据类型、梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回包含多个 ModuleInput 对象的列表，每个对象表示不同的输入配置
    return [
        # 单个示例：构造输入是一个长度为 3 的 FunctionInput，前向输入是形状为 (1, 3, 5) 的张量
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((1, 3, 5))),
                    desc='single'),

        # 没有批量维度的示例：构造输入是一个长度为 3 的 FunctionInput，前向输入是形状为 (3, 5) 的张量
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')
    ]


def module_inputs_torch_nn_AdaptiveMaxPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用函数，用于生成张量，设定设备、数据类型、梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回包含多个 ModuleInput 对象的列表，每个对象表示不同的输入配置
    return [
        # 单个示例：构造输入是一个长度为 3 的 FunctionInput，前向输入是形状为 (1, 3, 5, 6) 的张量
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='single'),

        # 没有批量维度的示例：构造输入是一个长度为 3 的 FunctionInput，前向输入是形状为 (3, 5, 6) 的张量
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5, 6))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),

        # 元组形式的示例：构造输入是一个形状为 (3, 4) 的 FunctionInput，前向输入是形状为 (1, 3, 5, 6) 的张量
        ModuleInput(constructor_input=FunctionInput((3, 4)),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='tuple'),

        # 元组形式且具有 None 维度的示例：构造输入是一个形状为 (3, None) 的 FunctionInput，前向输入是形状为 (1, 3, 5, 6) 的张量
        ModuleInput(constructor_input=FunctionInput((3, None)),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='tuple_none')
    ]


def module_inputs_torch_nn_AdaptiveMaxPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用函数，用于生成张量，设定设备、数据类型、梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回包含多个 ModuleInput 对象的列表，每个对象表示不同的输入配置
    return [
        # 单个示例：构造输入是一个长度为 3 的 FunctionInput，前向输入是形状为 (2, 3, 5, 6, 7) 的张量
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 5, 6, 7))),
                    desc='single'),

        # 没有批量维度的示例：构造输入是一个长度为 3 的 FunctionInput，前向输入是形状为 (3, 5, 6, 7) 的张量
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5, 6, 7))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),

        # 元组形式的示例：构造输入是一个形状为 (3, 4, 5) 的 FunctionInput，前向输入是形状为 (2, 3, 5, 6, 7) 的张量
        ModuleInput(constructor_input=FunctionInput((3, 4, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 6, 7))),
                    desc='tuple'),

        # 元组形式且具有 None 维度的示例：构造输入是一个形状为 (3, None, 5) 的 FunctionInput，前向输入是形状为 (2, 3, 5, 6, 7) 的张量
        ModuleInput(constructor_input=FunctionInput((3, None, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 6, 7))),
                    desc='tuple_none'),

        # 单个非原子张量的示例：构造输入是一个长度为 3 的 FunctionInput，前向输入是形状为 (2, 3, 12, 9, 3) 的张量
        ModuleInput(constructor_input=FunctionInput(3),
                    forward_input=FunctionInput(make_input((2, 3, 12, 9, 3))),
                    desc='single_nonatomic'),

        # 元组形式且具有非原子张量的示例：构造输入是一个形状为 (3, 4, 5) 的 FunctionInput，前向输入是形状为 (2, 3, 6, 4, 10) 的张量
        ModuleInput(constructor_input=FunctionInput((3, 4, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 6, 4, 10))),
                    desc='tuple_nonatomic')
    ]
# 定义一个函数，用于生成 Torch 中 BatchNorm1d 模块的输入列表
def module_inputs_torch_nn_BatchNorm1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用的函数，用于生成张量，设定设备、数据类型、梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个包含不同输入配置的列表
    return [
        # 第一个输入：构造器参数是长度为 10 的函数输入，前向传播的输入是形状为 (4, 10) 的张量输入，描述为“affine”
        ModuleInput(constructor_input=FunctionInput(10,),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='affine'),

        # 第二个输入：构造器参数是长度为 5 的函数输入，前向传播的输入是形状为 (4, 5, 3) 的张量输入，描述为“3d_input”
        ModuleInput(constructor_input=FunctionInput(5,),
                    forward_input=FunctionInput(make_input((4, 5, 3))),
                    desc='3d_input'),

        # 第三个输入：构造器参数是长度为 10，epsilon 是 1e-3，没有 momentum，前向传播的输入是形状为 (4, 10) 的张量输入，描述为“affine_simple_average”
        ModuleInput(constructor_input=FunctionInput(10, 1e-3, None),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='affine_simple_average'),

        # 第四个输入：构造器参数是长度为 10，epsilon 是 1e-3，momentum 是 0.3，不进行 affine 操作，前向传播的输入是形状为 (4, 10) 的张量输入，描述为“not_affine”
        ModuleInput(constructor_input=FunctionInput(10, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='not_affine'),

        # 第五个输入：构造器参数是长度为 10，epsilon 是 1e-3，momentum 是 0.3，进行 affine 操作，但不进行统计追踪，前向传播的输入是形状为 (4, 10) 的张量输入，描述为“not_tracking_stats”
        ModuleInput(constructor_input=FunctionInput(10, 1e-3, 0.3, True, False),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='not_tracking_stats'),

        # 第六个输入：构造器参数是长度为 5，epsilon 是 1e-3，momentum 是 0.3，不进行 affine 操作，前向传播的输入是形状为 (4, 5, 3) 的张量输入，描述为“3d_input_not_affine”
        ModuleInput(constructor_input=FunctionInput(5, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((4, 5, 3))),
                    desc='3d_input_not_affine'),

        # 第七个输入：构造器参数是长度为 5，epsilon 是 1e-3，momentum 是 0.3，不进行 affine 操作，前向传播的输入是形状为 (0, 5, 9) 的张量输入，描述为“zero_batch”
        ModuleInput(constructor_input=FunctionInput(5, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((0, 5, 9))),
                    desc='zero_batch')
    ]


这段代码为一个函数，用于生成 Torch 中 BatchNorm1d 模块的不同输入配置，每个输入配置由构造器参数、前向传播的输入和描述组成。
    # 返回一个列表，包含多个 ModuleInput 对象
    return [
        # 创建 ModuleInput 对象，设置构造函数参数为一个包含 3 的元组，前向输入为指定形状的数据
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4)))),
        
        # 创建 ModuleInput 对象，设置构造函数参数为 (3, 1e-3, None)，前向输入为指定形状的数据，并附带描述信息
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, None),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='3d_simple_average'),
        
        # 创建 ModuleInput 对象，设置构造函数参数为 (3, 1e-3, 0.7)，前向输入为指定形状的数据，并附带描述信息
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.7),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='momentum'),
        
        # 创建 ModuleInput 对象，设置构造函数参数为 (3, 1e-3, 0.7, False)，前向输入为指定形状的数据，并附带描述信息
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.7, False),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='not_affine'),
        
        # 创建 ModuleInput 对象，设置构造函数参数为 (3, 1e-3, 0.7, True, False)，前向输入为指定形状的数据，并附带描述信息
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.7, True, False),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='not_tracking_stats'),
        
        # 创建 ModuleInput 对象，设置构造函数参数为 (5, 1e-3, 0.3, False)，前向输入为指定形状的数据，并附带描述信息
        ModuleInput(constructor_input=FunctionInput(5, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((0, 5, 2, 2, 2))),
                    desc='zero_batch')
    ]
# 定义一个函数，生成 Torch 中的卷积层（ConvNd）的输入模块列表
def module_inputs_torch_nn_ConvNd(module_info, device, dtype, requires_grad, training, **kwargs):
    # 从 kwargs 中获取参数 N
    N = kwargs['N']
    # 从 kwargs 中获取参数 lazy，如果不存在则默认为 False
    lazy = kwargs.get('lazy', False)
    # 从 kwargs 中获取参数 transposed，如果不存在则默认为 False
    transposed = kwargs.get('transposed', False)
    # 创建一个函数 make_input，部分应用了 make_tensor 函数，设定了设备、数据类型、梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 如果 transposed 为 True，则 conv_kwargs_list 包含一个空字典
    # 否则包含两个字典，第一个为空字典，第二个包含 'padding': 'same' 键值对
    conv_kwargs_list = [{}] if transposed else [{}, {'padding': 'same'}]
    # 定义卷积核大小、输入通道数、输出通道数
    kernel_size, C_in, C_out = 3, 4, 5
    # 定义没有批次维度的输入形状，元组包含 C_in 和一系列维度增加 3
    input_no_batch_shape = (C_in,) + tuple(i + 3 for i in range(N))
    # 定义包含批次维度的输入形状，元组包含 2 和没有批次维度的输入形状
    input_batch_shape = (2,) + input_no_batch_shape
    # 返回一个列表，列表元素是 ModuleInput 对象，由 itertools.product 生成
    return [
        ModuleInput(
            constructor_input=(FunctionInput(C_out, kernel_size, **conv_kwargs) if lazy else
                               FunctionInput(C_in, C_out, kernel_size, **conv_kwargs)),
            forward_input=FunctionInput(make_input(
                input_batch_shape if with_batch else input_no_batch_shape)),
            desc=('' if with_batch else 'no_batch_dim'),
            reference_fn=(None if with_batch else no_batch_dim_reference_fn)
        )
        for with_batch, conv_kwargs in itertools.product([True, False], conv_kwargs_list)
    ]


# 定义一个函数，生成 Torch 中的余弦嵌入损失（CosineEmbeddingLoss）的输入模块列表
def module_inputs_torch_nn_CosineEmbeddingLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个函数 make_input，部分应用了 make_tensor 函数，设定了设备、数据类型、梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 创建一个函数 make_target，部分应用了 make_tensor 函数，设定了设备和数据类型，但不需要梯度
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    
    # 定义一个包含不同参数组合的列表 cases，每个元素是一个元组，包含描述和构造函数的关键字参数字典
    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('margin', {'margin': 0.7})
    ]
    
    # 创建一个空列表 module_inputs 用于存储 ModuleInput 对象
    module_inputs = []
    # 遍历 cases 列表，每次迭代取出描述和构造函数的关键字参数字典
    for desc, constructor_kwargs in cases:
        # 定义一个 reference_fn 函数，接受多个参数，部分应用了 constructor_kwargs
        def reference_fn(m, p, i1, i2, t, constructor_kwargs=constructor_kwargs):
            return cosineembeddingloss_reference(i1, i2, t, **constructor_kwargs)
        
        # 将一个 ModuleInput 对象添加到 module_inputs 列表中
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((15, 10)), make_input((15, 10)),
                                            make_target((15,)).sign()),
                desc=desc,
                reference_fn=reference_fn
            )
        )
    
    # 返回存储 ModuleInput 对象的列表 module_inputs
    return module_inputs


# 定义一个函数，生成 Torch 中的 ELU 激活函数（ELU）的输入模块列表
def module_inputs_torch_nn_ELU(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个函数 make_input，部分应用了 make_tensor 函数，设定了设备、数据类型、梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 返回一个列表，包含四个 ModuleInput 对象
    return [
        # 第一个 ModuleInput 对象
        ModuleInput(
            # 构造函数输入，alpha 设为 2.0
            constructor_input=FunctionInput(alpha=2.),
            # forward 方法的输入，是一个包含三维张量的 FunctionInput 对象
            forward_input=FunctionInput(make_input((3, 2, 5))),
            # 引用函数，根据条件返回张量值
            reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2 * (i.exp() - 1))
        ),
        # 第二个 ModuleInput 对象
        ModuleInput(
            # 构造函数输入，alpha 设为 2.0
            constructor_input=FunctionInput(alpha=2.),
            # forward 方法的输入，是一个空的 FunctionInput 对象
            forward_input=FunctionInput(make_input(())),
            # 描述为标量
            desc='scalar'
        ),
        # 第三个 ModuleInput 对象
        ModuleInput(
            # 构造函数输入，使用默认值
            constructor_input=FunctionInput(),
            # forward 方法的输入，是一个包含一维张量的 FunctionInput 对象
            forward_input=FunctionInput(make_input((3,))),
            # 描述为没有批次维度
            desc='no_batch_dim',
            # 参考函数是 no_batch_dim_reference_fn
            reference_fn=no_batch_dim_reference_fn
        ),
        # 第四个 ModuleInput 对象
        ModuleInput(
            # 构造函数输入，alpha 设为 2.0
            constructor_input=FunctionInput(alpha=2.),
            # forward 方法的输入，是一个包含四维张量的 FunctionInput 对象
            forward_input=FunctionInput(make_input((2, 3, 2, 5))),
            # 描述为四维输入
            desc='4d_input'
        )
    ]
def module_inputs_torch_nn_CELU(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用函数，用于创建张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个由 ModuleInput 对象组成的列表
    return [
        # 创建 ModuleInput 对象，使用 FunctionInput 作为构造输入，alpha 设置为 2.0
        # forward_input 使用 make_input 创建一个形状为 (3, 2, 5) 的张量
        # reference_fn 定义为 lambda 表达式，对输入张量进行处理
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2. * ((.5 * i).exp() - 1))),
        
        # 创建 ModuleInput 对象，使用 FunctionInput 作为构造输入，alpha 设置为 2.0
        # forward_input 使用 make_input 创建一个标量张量
        # desc 设置为 'scalar'，用于描述这个 ModuleInput
        # reference_fn 定义为 lambda 表达式，对输入张量进行处理
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2. * ((.5 * i).exp() - 1)),
                    desc='scalar'),
        
        # 创建 ModuleInput 对象，使用 FunctionInput 作为构造输入，alpha 设置为 2.0
        # forward_input 使用 make_input 创建一个形状为 (3,) 的张量
        # desc 设置为 'no_batch_dim'，用于描述这个 ModuleInput
        # reference_fn 使用预定义的函数 no_batch_dim_reference_fn 处理输入张量
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((3,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn)]
    # 返回一个包含四个 ModuleInput 对象的列表
    return [
        # 创建 ModuleInput 对象，使用默认的 FunctionInput 作为构造器输入，
        # 并使用空元组构造 FunctionInput 作为 forward_input，描述为'scalar'
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
    
        # 创建 ModuleInput 对象，使用默认的 FunctionInput 作为构造器输入，
        # 并使用包含一个元素为4的元组构造 FunctionInput 作为 forward_input，
        # 同时设置 reference_fn 为 no_batch_dim_reference_fn，描述为'no_batch_dim'
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
    
        # 创建 ModuleInput 对象，使用默认的 FunctionInput 作为构造器输入，
        # 并使用包含元素为(2, 3, 4, 5)的元组构造 FunctionInput 作为 forward_input，
        # 描述为'channels_last_mem_format'
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    desc='channels_last_mem_format'),
    
        # 创建 ModuleInput 对象，使用默认的 FunctionInput 作为构造器输入，
        # 并使用包含元素为(2, 3, 3, 4, 5)的元组构造 FunctionInput 作为 forward_input，
        # 描述为'channels_last_3d_mem_format'
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 3, 4, 5))),
                    desc='channels_last_3d_mem_format')
    ]
# 创建一个局部函数 make_input，用于生成张量，根据给定的设备、数据类型、梯度需求等参数
make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

# 返回一个包含 ModuleInput 对象的列表，每个对象代表神经网络模块的输入配置
return [
    # 第一个 ModuleInput 对象，没有构造函数参数，前向输入是一个标量张量
    ModuleInput(constructor_input=FunctionInput(),
                forward_input=FunctionInput(make_input(())),
                desc='scalar'),
    
    # 第二个 ModuleInput 对象，没有构造函数参数，前向输入是一个大小为 4 的张量
    ModuleInput(constructor_input=FunctionInput(),
                forward_input=FunctionInput(make_input(4)),
                reference_fn=no_batch_dim_reference_fn,
                desc='no_batch_dim'),
    
    # 第三个 ModuleInput 对象，没有构造函数参数，前向输入是一个形状为 (2, 3, 4, 5) 的张量
    ModuleInput(constructor_input=FunctionInput(),
                forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                desc='channels_last_mem_format'),
    
    # 第四个 ModuleInput 对象，没有构造函数参数，前向输入是一个形状为 (2, 3, 3, 4, 5) 的张量
    ModuleInput(constructor_input=FunctionInput(),
                forward_input=FunctionInput(make_input((2, 3, 3, 4, 5))),
                desc='channels_last_3d_mem_format')
]



# 创建一个局部函数 make_input，用于生成张量，根据给定的设备、数据类型、梯度需求等参数
make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

# 返回一个包含 ModuleInput 对象的列表，每个对象代表神经网络模块的输入配置
return [
    # 第一个 ModuleInput 对象，没有构造函数参数，前向输入是一个形状为 (3, 2, 5) 的张量
    ModuleInput(constructor_input=FunctionInput(),
                forward_input=FunctionInput(make_input((3, 2, 5)))),
    
    # 第二个 ModuleInput 对象，没有构造函数参数，前向输入是一个大小为 4 的张量
    ModuleInput(constructor_input=FunctionInput(),
                forward_input=FunctionInput(make_input(4)),
                reference_fn=no_batch_dim_reference_fn,
                desc='no_batch_dim'),
    
    # 第三个 ModuleInput 对象，构造函数参数为 0.5，前向输入是一个形状为 (3, 2, 5) 的张量
    ModuleInput(constructor_input=FunctionInput(0.5),
                forward_input=FunctionInput(make_input((3, 2, 5))),
                desc='with_negval'),
    
    # 第四个 ModuleInput 对象，构造函数参数为 0.0，前向输入是一个形状为 (10, 10) 的张量
    ModuleInput(constructor_input=FunctionInput(0.0),
                forward_input=FunctionInput(make_input((10, 10))),
                desc='with_zero_negval'),
    
    # 第五个 ModuleInput 对象，构造函数参数为 0.5，前向输入是一个标量张量
    ModuleInput(constructor_input=FunctionInput(0.5),
                forward_input=FunctionInput(make_input(())),
                desc='with_negval_scalar')
]



# 创建一个局部函数 make_input，用于生成张量，根据给定的设备、数据类型、梯度需求等参数
make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
# 函数未完全定义，因此没有返回语句
    # 返回一个列表，包含多个 ModuleInput 对象，每个对象都包括构造输入、前向输入、参考函数和描述信息
    return [
        # 第一个 ModuleInput 对象，描述为标量
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(())),
            desc='scalar'
        ),
        # 第二个 ModuleInput 对象，描述为没有批次维度
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim'
        ),
        # 第三个 ModuleInput 对象，描述为一维数据
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((2, 3, 4))),
            reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
            desc='1d'
        ),
        # 第四个 ModuleInput 对象，描述为一维数据并带多个参数
        ModuleInput(
            constructor_input=FunctionInput(3),
            forward_input=FunctionInput(make_input((2, 3, 4))),
            reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
            desc='1d_multiparam'
        ),
        # 第五个 ModuleInput 对象，描述为二维数据
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((2, 3, 4, 5))),
            reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
            desc='2d'
        ),
        # 第六个 ModuleInput 对象，描述为二维数据并带多个参数
        ModuleInput(
            constructor_input=FunctionInput(3),
            forward_input=FunctionInput(make_input((2, 3, 4, 5))),
            reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
            desc='2d_multiparam'
        ),
        # 第七个 ModuleInput 对象，描述为三维数据
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((2, 3, 4, 5, 6))),
            reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
            desc='3d'
        ),
        # 第八个 ModuleInput 对象，描述为三维数据并带多个参数
        ModuleInput(
            constructor_input=FunctionInput(3),
            forward_input=FunctionInput(make_input((2, 3, 4, 5, 6))),
            reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
            desc='3d_multiparam'
        )
    ]
# 定义一个函数，生成用于 torch.nn.SELU 模块的输入
def module_inputs_torch_nn_SELU(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分函数，用于生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回模块输入列表
    return [
        # 第一个输入，构造函数输入为空，前向函数输入为形状为 (3, 2, 5) 的张量
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 2, 5)))),
        # 第二个输入，构造函数输入为空，前向函数输入为形状为 (4,) 的张量，参考函数为 no_batch_dim_reference_fn，描述为 'no_batch_dim'
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        # 第三个输入，构造函数输入为空，前向函数输入为标量张量，描述为 'scalar'
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar')]


# 定义一个函数，生成用于 torch.nn.SiLU 模块的输入
def module_inputs_torch_nn_SiLU(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分函数，用于生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回模块输入列表
    return [
        # 第一个输入，构造函数输入为空，前向函数输入为标量张量，参考函数为 lambda 函数，描述为 'scalar'
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, x, *_: x * torch.sigmoid(x),
                    desc='scalar'),
        # 第二个输入，构造函数输入为空，前向函数输入为形状为 (4,) 的张量，参考函数为 no_batch_dim_reference_fn，描述为 'no_batch_dim'
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        # 第三个输入，构造函数输入为空，前向函数输入为形状为 (5, 6, 7) 的张量，参考函数为 lambda 函数
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((5, 6, 7))),
                    reference_fn=lambda m, p, x, *_: x * torch.sigmoid(x))]


# 定义一个函数，生成用于 torch.nn.Softmax 模块的输入
def module_inputs_torch_nn_Softmax(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分函数，用于生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回模块输入列表
    return [
        # 第一个输入，构造函数输入为 1，前向函数输入为形状为 (10, 20) 的张量，参考函数为 lambda 函数
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((10, 20))),
                    reference_fn=lambda m, p, i: torch.exp(i).div(torch.exp(i).sum(1, True).expand(10, 20))),
        # 第二个输入，构造函数输入为 0，前向函数输入为标量张量，参考函数为 lambda 函数，描述为 'scalar'
        ModuleInput(constructor_input=FunctionInput(0),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: torch.exp(i).div(torch.exp(i).sum(0, True)),
                    desc='scalar'),
        # 第三个输入，构造函数输入为 -1，前向函数输入为形状为 (4, 5) 的张量，参考函数为 no_batch_dim_reference_fn，描述为 'no_batch_dim'
        ModuleInput(constructor_input=FunctionInput(-1),
                    forward_input=FunctionInput(make_input((4, 5))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


# 定义一个函数，生成用于 torch.nn.Softmax2d 模块的输入
def module_inputs_torch_nn_Softmax2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分函数，用于生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 返回一个列表，列表中包含两个 ModuleInput 对象
    return [
        # 第一个 ModuleInput 对象的构造参数和前向输入参数设置
        ModuleInput(
            constructor_input=FunctionInput(),  # 构造参数为空
            forward_input=FunctionInput(make_input((1, 3, 10, 20))),  # 前向输入是一个特定形状的张量
            reference_fn=lambda m, p, i: torch.exp(i).div(torch.exp(i).sum(1, False))  # 参考函数是对输入张量进行操作的匿名函数
        ),
        # 第二个 ModuleInput 对象的构造参数和前向输入参数设置
        ModuleInput(
            constructor_input=FunctionInput(),  # 构造参数为空
            forward_input=FunctionInput(make_input((3, 4, 5))),  # 前向输入是一个特定形状的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数是指定的函数 no_batch_dim_reference_fn
            desc='no_batch_dim'  # 对当前 ModuleInput 对象的描述信息
        )
    ]
# 定义一个函数，生成用于测试的输入模块列表，使用了torch.nn中的LogSoftmax模块
def module_inputs_torch_nn_LogSoftmax(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分函数make_input，用于创建张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个包含多个ModuleInput对象的列表，每个对象都包括构造输入、前向输入和参考函数
    return [
        # 第一个ModuleInput对象
        ModuleInput(
            constructor_input=FunctionInput(1),  # 构造输入为1
            forward_input=FunctionInput(make_input((10, 20))),  # 前向输入为一个10x20的张量
            reference_fn=lambda m, p, i: torch.exp(i).div_(torch.exp(i).sum(1, True).expand(10, 20)).log_()  # 参考函数计算LogSoftmax
        ),
        # 第二个ModuleInput对象
        ModuleInput(
            constructor_input=FunctionInput(1),  # 构造输入为1
            forward_input=FunctionInput(make_input((1, 3, 10, 20))),  # 前向输入为一个1x3x10x20的张量
            reference_fn=lambda m, p, i: torch.exp(i).div_(torch.exp(i).sum(1, False)).log_(),  # 参考函数计算LogSoftmax
            desc='multiparam'  # 描述信息，指明是多参数的情况
        ),
        # 第三个ModuleInput对象
        ModuleInput(
            constructor_input=FunctionInput(0),  # 构造输入为0
            forward_input=FunctionInput(make_input(())),  # 前向输入为一个标量
            reference_fn=lambda m, p, i: torch.exp(i).div_(torch.exp(i).sum(0, False)).log_(),  # 参考函数计算LogSoftmax
            desc='multiparam_scalar'  # 描述信息，指明是多参数标量的情况
        ),
        # 第四个ModuleInput对象
        ModuleInput(
            constructor_input=FunctionInput(-1),  # 构造输入为-1
            forward_input=FunctionInput(make_input((4, 5))),  # 前向输入为一个4x5的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数为预定义的无批量维度的参考函数
            desc='no_batch_dim'  # 描述信息，指明无批量维度的情况
        )
    ]


# 定义一个函数，生成用于测试的输入模块列表，使用了torch.nn中的Softmin模块
def module_inputs_torch_nn_Softmin(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分函数make_input，用于创建张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个包含多个ModuleInput对象的列表，每个对象都包括构造输入和前向输入
    return [
        # 第一个ModuleInput对象
        ModuleInput(
            constructor_input=FunctionInput(1),  # 构造输入为1
            forward_input=FunctionInput(make_input((10, 20)))  # 前向输入为一个10x20的张量
        ),
        # 第二个ModuleInput对象
        ModuleInput(
            constructor_input=FunctionInput(1),  # 构造输入为1
            forward_input=FunctionInput(make_input((2, 3, 5, 10))),  # 前向输入为一个2x3x5x10的张量
            desc='multidim'  # 描述信息，指明是多维度的情况
        ),
        # 第三个ModuleInput对象
        ModuleInput(
            constructor_input=FunctionInput(0),  # 构造输入为0
            forward_input=FunctionInput(make_input(())),  # 前向输入为一个标量
            desc='scalar'  # 描述信息，指明是标量的情况
        ),
        # 第四个ModuleInput对象
        ModuleInput(
            constructor_input=FunctionInput(-1),  # 构造输入为-1
            forward_input=FunctionInput(make_input((3, 4, 10))),  # 前向输入为一个3x4x10的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数为预定义的无批量维度的参考函数
            desc='no_batch_dim'  # 描述信息，指明无批量维度的情况
        )
    ]


# 定义一个函数，生成用于测试的输入模块列表，使用了torch.nn中的Softplus模块（未完整实现）
def module_inputs_torch_nn_Softplus(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分函数make_input，用于创建张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 返回一个包含多个 ModuleInput 对象的列表
    return [
        # 创建一个 ModuleInput 对象，使用默认构造函数输入，前向输入为一个形状为 (10, 20) 的张量
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((10, 20))),
            # 参考函数为对输入张量 i 进行计算，返回 torch.log(1 + torch.exp(i))
            reference_fn=lambda m, p, i: torch.log(1 + torch.exp(i))
        ),
        # 创建另一个 ModuleInput 对象，使用构造函数输入为 2，前向输入同样为形状为 (10, 20) 的张量
        ModuleInput(
            constructor_input=FunctionInput(2),
            forward_input=FunctionInput(make_input((10, 20))),
            # 给该 ModuleInput 对象添加描述 'beta'，参考函数为 1/2 * torch.log(1 + torch.exp(2 * i))
            reference_fn=lambda m, p, i: 1. / 2. * torch.log(1 + torch.exp(2 * i)),
            desc='beta'
        ),
        # 创建 ModuleInput 对象，构造函数输入为 (2, -100)，前向输入同样为形状为 (10, 20) 的张量
        ModuleInput(
            constructor_input=FunctionInput(2, -100),
            forward_input=FunctionInput(make_input((10, 20))),
            # 给该 ModuleInput 对象添加描述 'beta_threshold'，参考函数根据 i 的值进行条件判断后计算不同的结果
            reference_fn=(
                lambda m, p, i: ((i * 2) > -100).type_as(i) * i
                + ((i * 2) <= -100).type_as(i) * 1. / 2. * torch.log(1 + torch.exp(2 * i))
            ),
            desc='beta_threshold'
        ),
        # 创建 ModuleInput 对象，构造函数输入为 (2, -100)，前向输入为一个标量
        ModuleInput(
            constructor_input=FunctionInput(2, -100),
            forward_input=FunctionInput(make_input(())),
            # 给该 ModuleInput 对象添加描述 'beta_threshold_scalar'，参考函数同样根据 i 的值进行条件判断后计算不同的结果
            reference_fn=(
                lambda m, p, i: ((i * 2) > -100).type_as(i) * i
                + ((i * 2) <= -100).type_as(i) * 1. / 2. * torch.log(1 + torch.exp(2 * i))
            ),
            desc='beta_threshold_scalar'
        ),
        # 创建 ModuleInput 对象，使用默认构造函数输入，前向输入为一个形状为 (4,) 的张量
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),
            # 给该 ModuleInput 对象添加描述 'no_batch_dim'，参考函数为 no_batch_dim_reference_fn
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim'
        )
    ]
def module_inputs_torch_nn_Softshrink(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数创建一个make_input函数，方便后续生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个ModuleInput对象列表，每个对象包含构造器输入和前向传播输入
    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 2, 5)))),
        ModuleInput(constructor_input=FunctionInput(1,),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    desc='lambda'),
        ModuleInput(constructor_input=FunctionInput(1,),
                    forward_input=FunctionInput(make_input(())),
                    desc='lambda_scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Softsign(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数创建一个make_input函数，方便后续生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个ModuleInput对象列表，每个对象包含构造器输入、前向传播输入和参考函数（lambda表达式）
    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    reference_fn=lambda m, p, i: i.div(1 + torch.abs(i))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: i.div(1 + torch.abs(i)),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Tanh(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数创建一个make_input函数，方便后续生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个ModuleInput对象列表，每个对象包含构造器输入和前向传播输入
    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5)))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]



def module_inputs_torch_nn_Tanhshrink(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数创建一个make_input函数，方便后续生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 返回一个包含三个 ModuleInput 对象的列表
    return [
        # 创建 ModuleInput 对象，constructor_input 为空函数输入，forward_input 包含形状为 (2, 3, 4, 5) 的输入数据
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5)))),
        # 创建 ModuleInput 对象，constructor_input 为空函数输入，forward_input 包含空元组作为输入数据，描述为 'scalar'
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        # 创建 ModuleInput 对象，constructor_input 为空函数输入，forward_input 包含形状为 (4,) 的输入数据，
        # reference_fn 指定为 no_batch_dim_reference_fn，描述为 'no_batch_dim'
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')
    ]
def module_inputs_torch_nn_Threshold(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建局部函数 make_input，用于生成特定设备、数据类型、是否需要梯度的张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个列表，包含四个 ModuleInput 对象，每个对象都包括构造输入和前向输入
    return [
        ModuleInput(constructor_input=FunctionInput(2., 1.),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    desc='threshold_value'),             # 描述为阈值张量
        ModuleInput(constructor_input=FunctionInput(2., 10.),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    desc='large_value'),                # 描述为大张量
        ModuleInput(constructor_input=FunctionInput(2., 1.),
                    forward_input=FunctionInput(make_input(())),
                    desc='threshold_value_scalar'),    # 描述为阈值标量张量
        ModuleInput(constructor_input=FunctionInput(2., 1.),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')               # 描述为无批次维度的张量
    ]


def module_inputs_torch_nn_Mish(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建局部函数 make_input，用于生成特定设备、数据类型、是否需要梯度的张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个列表，包含三个 ModuleInput 对象，每个对象都包括构造输入和前向输入
    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((5, 6, 7))),
                    reference_fn=lambda m, p, i: i * torch.tanh(F.softplus(i))),  # 自定义引用函数
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: i * torch.tanh(F.softplus(i)),
                    desc='scalar'),                     # 描述为标量张量
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')                # 描述为无批次维度的张量
    ]


def module_inputs_torch_nn_L1Loss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建局部函数 make_input，用于生成特定设备、数据类型、是否需要梯度的张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个列表，包含两个 ModuleInput 对象，每个对象都包括构造输入和前向输入
    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4)),
                                                make_input((2, 3, 4))),
                    reference_fn=lambda m, p, i, t: 1. / i.numel() * sum((a - b).abs().sum()
                                                                         for a, b in zip(i, t))),  # 自定义引用函数
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(()), make_input(())),
                    reference_fn=lambda m, p, i, t: 1. / i.numel() * (i - t).abs().sum(),
                    desc='scalar')                      # 描述为标量张量
    ] + generate_regression_criterion_inputs(make_input)   # 合并额外生成的回归损失输入


def module_inputs_torch_nn_SmoothL1Loss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建局部函数 make_input，用于生成特定设备、数据类型、是否需要梯度的张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 定义包含多个测试用例的列表，每个元素是一个元组，包含描述和构造函数的参数字典
    cases: List[Tuple[str, dict]] = [
        ('', {}),  # 空参数字典的测试用例
        ('reduction_sum', {'reduction': 'sum'}),  # 包含'reduction'为'sum'的测试用例
        ('reduction_mean', {'reduction': 'mean'}),  # 包含'reduction'为'mean'的测试用例
        ('reduction_none', {'reduction': 'none'}),  # 包含'reduction'为'none'的测试用例
    ]

    # 初始化模块输入列表
    module_inputs = []
    
    # 遍历测试用例列表
    for desc, constructor_kwargs in cases:
        
        # 定义一个参考函数，使用指定的构造函数参数字典调用 smoothl1loss_reference 函数
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return smoothl1loss_reference(i, t, **constructor_kwargs)

        # 将每个测试用例的输入配置为 ModuleInput 对象，并添加到 module_inputs 列表中
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),  # 使用构造函数参数初始化的函数输入
                forward_input=FunctionInput(make_input((5, 10)),  # 使用指定形状初始化的前向输入
                                            make_input((5, 10))),  # 使用指定形状初始化的目标输入
                desc=desc,  # 描述字符串
                reference_fn=reference_fn  # 参考函数
            )
        )
        
        # 添加一个额外的 ModuleInput 对象，其构造函数参数为 scalar 版本的 constructor_kwargs
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),  # 使用构造函数参数初始化的函数输入
                forward_input=FunctionInput(make_input(()),  # 使用标量形状初始化的前向输入
                                            make_input(())),  # 使用标量形状初始化的目标输入
                desc=f'scalar_{desc}',  # 带有'scalar_'前缀的描述字符串
                reference_fn=reference_fn  # 参考函数
            )
        )

    # 返回填充好的 module_inputs 列表
    return module_inputs
# 定义一个函数，用于生成基于不同参数的张量输入
def module_inputs_torch_nn_BCELoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用的函数，用于生成具有特定设备、数据类型、梯度要求的张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 创建一个部分应用的函数，用于生成目标张量，不需要梯度
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    # 创建一个部分应用的函数，用于生成权重张量，不需要梯度
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # 定义测试用例，每个测试用例是一个元组，包含描述和构造参数的字典
    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('weights', {'weight': make_weight((10,))}),  # 添加权重参数的测试用例
    ]

    # 定义二元交叉熵损失函数的参考实现
    def bce_loss_reference_fn(m, p, i, t, reduction='mean', weight=None):
        # 计算二元交叉熵损失
        result = -(t * i.log() + (1 - t) * (1 - i).log())

        # 如果提供了权重，乘以权重
        if weight is not None:
            result = result * weight

        # 根据reduction参数返回不同的损失值
        if reduction == 'none':
            return result
        elif reduction == 'mean':
            return result.sum() / i.numel()
        else:
            return result.sum()

    # 存储模块输入的列表
    module_inputs = []
    
    # 对每个测试用例进行迭代，生成ModuleInput对象，并添加到module_inputs列表中
    for desc, constructor_kwargs in cases:
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),  # 使用构造参数创建FunctionInput对象
                forward_input=FunctionInput(  # 使用make_input和make_target生成前向输入
                    make_input((15, 10), low=1e-2, high=1 - 1e-2),
                    make_target((15, 10)).gt(0).to(dtype),
                ),
                desc=desc,  # 描述当前测试用例
                reference_fn=partial(bce_loss_reference_fn, **constructor_kwargs)  # 使用部分应用的损失函数作为参考函数
            )
        )

    # 创建一个标量权重张量
    scalar_weight = make_weight(())
    # 添加一个使用标量权重的ModuleInput对象到module_inputs列表中
    module_inputs.append(
        ModuleInput(
            constructor_input=FunctionInput(weight=scalar_weight),
            forward_input=FunctionInput(
                make_input((), low=1e-2, high=1 - 1e-2),
                make_target(()).gt(0).to(dtype),
            ),
            desc='scalar_weight',  # 描述为使用标量权重
            reference_fn=partial(bce_loss_reference_fn, weight=scalar_weight)  # 使用标量权重的损失函数作为参考函数
        )
    )

    # 返回所有生成的模块输入对象列表
    return module_inputs


def module_inputs_torch_nn_BCEWithLogitsLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用的函数，用于生成具有特定设备、数据类型、梯度要求的张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 创建一个部分应用的函数，用于生成目标张量，不需要梯度
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    # 创建一个部分应用的函数，用于生成权重张量，不需要梯度
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # 定义测试用例，每个测试用例是一个元组，包含描述和构造参数的字典
    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('weights', {'weight': make_weight((10,))}),  # 添加权重参数的测试用例
        ('scalar_weights', {'weight': make_weight(())})  # 添加标量权重参数的测试用例
    ]
    # 定义一个带有 BCEWithLogitsLoss 参考函数的方法，包括正样本权重 pos_weight 和相应的 SampleInputs
    def bce_withlogitsloss_reference_fn(m, p, i, t, reduction='mean', weight=None):
        # 计算输入张量 i 的逐元素负值并截断小于零的部分，得到一个新张量 max_val
        max_val = (-i).clamp(min=0)
        # 根据 BCEWithLogitsLoss 的计算公式计算损失结果 result
        result = (1 - t).mul_(i).add_(max_val).add_((-max_val).exp_().add_((-i - max_val).exp_()).log_())

        # 如果指定了权重 weight，则将结果 result 乘以权重
        if weight is not None:
            result = result * weight

        # 根据指定的 reduction 参数，计算最终的损失值
        if reduction == 'none':
            return result
        elif reduction == 'mean':
            return result.sum() / i.numel()  # 返回均值作为损失值
        else:
            return result.sum()  # 返回总和作为损失值

    # 初始化一个空列表 module_inputs，用于存储测试用例的输入
    module_inputs = []
    # 遍历测试用例 cases 中的每个元素，每个元素包含描述 desc 和构造函数参数 constructor_kwargs
    for desc, constructor_kwargs in cases:
        # 创建 ModuleInput 对象，包括构造函数输入 constructor_input 和前向传播输入 forward_input
        # 同时指定描述 desc 和参考函数 reference_fn（使用 bce_withlogitsloss_reference_fn 的偏函数）
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(
                    make_input((15, 10), low=1e-2, high=1 - 1e-2),  # 创建输入张量
                    make_target((15, 10)).gt(0).to(dtype)  # 创建目标张量并转换为指定的数据类型 dtype
                ),
                desc=desc,
                reference_fn=partial(bce_withlogitsloss_reference_fn, **constructor_kwargs)  # 使用偏函数绑定参数
            )
        )

    # 返回存储了所有测试用例输入的 module_inputs 列表
    return module_inputs
# 定义一个函数，生成指定设备和数据类型的张量，并设置是否需要梯度
def module_inputs_torch_nn_CrossEntropyLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用 partial 函数生成一个便捷函数 make_input，用于创建张量输入
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 使用 partial 函数生成一个便捷函数 make_target，用于创建长整型张量目标
    make_target = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)
    # 使用 partial 函数生成一个便捷函数 make_weight，用于创建不需要梯度的指定数据类型的张量

    # 定义损失函数的归约方式列表
    reductions: List[str] = ['mean', 'sum', 'none']
    # 定义损失函数的测试用例，每个测试用例是一个元组，包含一个标识字符串和一个参数字典
    cases: List[Tuple[str, dict]] = [
        ('', {}),  # 空参数的测试用例
        ('weights', {'weight': make_weight((3,))}),  # 带权重参数的测试用例，权重为一个形状为 (3,) 的张量
        ('ignore_index', {'ignore_index': 1}),  # 忽略索引为 1 的测试用例
        ('label_smoothing', {'label_smoothing': 0.15}),  # 标签平滑参数为 0.15 的测试用例
        ('ignore_index_label_smoothing', {'ignore_index': 1, 'label_smoothing': 0.15})  # 同时包含忽略索引和标签平滑参数的测试用例
    ]

    # 初始化模块输入列表
    module_inputs = []
    # 返回模块输入列表
    return module_inputs



# 定义另一个函数，生成指定设备和数据类型的张量，并设置是否需要梯度
def module_inputs_torch_nn_CTCLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用 partial 函数生成一个便捷函数 make_input，用于创建张量输入
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 使用 partial 函数生成一个便捷函数 make_target，用于创建不需要梯度的张量目标

    # 定义 CTCLoss 的测试用例，每个测试用例是一个元组，包含一个标识字符串和一个参数字典
    cases: List[Tuple[str, dict]] = [
        ('', {}),  # 空参数的测试用例
        ('reduction_sum', {'reduction': 'sum'}),  # 归约方式为 sum 的测试用例
        ('reduction_mean', {'reduction': 'mean'}),  # 归约方式为 mean 的测试用例
        ('reduction_none', {'reduction': 'none'}),  # 归约方式为 none 的测试用例
        ('blank', {'blank': 14})  # 空白标记为 14 的测试用例
    ]
    
    # 定义目标数据类型列表，包括 torch.int 和 torch.long
    target_dtypes = [torch.int, torch.long]

    # 初始化模块输入列表
    module_inputs = []
    # 对于每个目标数据类型和测试用例，迭代处理
    for target_dtype, (desc, constructor_kwargs) in product(target_dtypes, cases):
        # 定义一个参考函数，用于调用参考实现计算损失
        def reference_fn(m, p, i, t, il, tl, constructor_kwargs=constructor_kwargs):
            return ctcloss_reference(i, t, il, tl, **constructor_kwargs)

        # 获取构造函数参数中的 blank 值，默认为 0
        blank = constructor_kwargs.get('blank', 0)
        # 根据 blank 值设置 low 和 high 的初始值
        low = 0 if blank == 14 else 1
        high = 14 if blank == 14 else 15

        # 向 module_inputs 列表添加 ModuleInput 对象，每个对象包含构造输入、前向输入、描述和参考函数
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                # 设置前向输入，包括 log_softmax 处理后的输入、制作的目标数据、固定维度的长度元组
                forward_input=FunctionInput(make_input((50, 3, 15)).log_softmax(2),
                                            make_target((3, 30), dtype=target_dtype, low=low, high=high),
                                            (50, 50, 50), (30, 25, 20)),
                desc=f'{desc}_lengths_intlists',  # 设置描述，表明使用整数列表长度的情况
                reference_fn=reference_fn)  # 设置参考函数
        )
        # 向 module_inputs 列表添加 ModuleInput 对象，每个对象与上述相似，但是前向输入中的固定长度使用了设备上的张量
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((50, 3, 15)).log_softmax(2),
                                            make_target((3, 30), dtype=target_dtype, low=low, high=high),
                                            torch.tensor((50, 50, 50), device=device),
                                            torch.tensor((30, 25, 20), device=device)),
                desc=f'{desc}_lengths_tensors',  # 设置描述，表明使用张量长度的情况
                reference_fn=reference_fn)  # 设置参考函数
        )
        # 向 module_inputs 列表添加 ModuleInput 对象，每个对象与上述相似，但是目标数据的维度为一维，并使用整数列表长度
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((50, 3, 15)).log_softmax(2),
                                            make_target((30 + 25 + 20,), dtype=target_dtype, low=low, high=high),
                                            (50, 50, 50), (30, 25, 20)),
                desc=f'{desc}_1d_target_lengths_intlists',  # 设置描述，表明目标数据为一维整数列表长度的情况
                reference_fn=reference_fn)  # 设置参考函数
        )
        # 向 module_inputs 列表添加 ModuleInput 对象，每个对象与上述相似，但是目标数据的维度为一维，并使用设备上的张量长度
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((50, 3, 15)).log_softmax(2),
                                            make_target((30 + 25 + 20,), dtype=target_dtype, low=low, high=high),
                                            torch.tensor((50, 50, 50), device=device),
                                            torch.tensor((30, 25, 20), device=device)),
                desc=f'{desc}_1d_target_lengths_tensors',  # 设置描述，表明目标数据为一维张量长度的情况
                reference_fn=reference_fn)  # 设置参考函数
        )

    # 返回构造好的 module_inputs 列表
    return module_inputs
# 构造函数，生成一组 ModuleInput 对象的列表，用于 nn.GroupNorm 模块的测试输入
def module_inputs_torch_nn_GroupNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    # 部分应用 make_tensor 函数，固定了部分参数，以便在构造输入时使用
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回 ModuleInput 对象的列表，每个对象描述了不同的测试用例
    return [
        ModuleInput(
            constructor_input=FunctionInput(3, 6, 1e-3),  # 构造函数输入，用于初始化 nn.GroupNorm 模块
            forward_input=FunctionInput(make_input((4, 6, 5))),  # 模块前向输入，传递给 nn.GroupNorm 模块的输入
            desc='1d_affine'),  # 描述该测试用例的简短标识符
        ModuleInput(
            constructor_input=FunctionInput(3, 12, 1e-3),
            forward_input=FunctionInput(make_input((4, 12))),
            desc='1d_affine_GN'),
        ModuleInput(
            constructor_input=FunctionInput(1, 6, 1e-3),
            forward_input=FunctionInput(make_input((150, 6))),
            desc='1d_affine_large_batch'),
        ModuleInput(
            constructor_input=FunctionInput(5, 5, 1e-3, False),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_no_affine_IN'),
        ModuleInput(
            constructor_input=FunctionInput(1, 10, 1e-3, False),
            forward_input=FunctionInput(make_input((4, 10))),
            desc='1d_no_affine_LN'),
        ModuleInput(
            constructor_input=FunctionInput(3, 6, 1e-3),
            forward_input=FunctionInput(make_input((4, 6, 2, 3))),
            desc='2d_affine'),
        ModuleInput(
            constructor_input=FunctionInput(3, 6, 1e-3),
            forward_input=FunctionInput(make_input((4, 6, 28, 28))),
            desc='2d_affine_large_feature'),
        ModuleInput(
            constructor_input=FunctionInput(3, 51, 1e-5, False),
            forward_input=FunctionInput(make_input((2, 51, 28, 28))),
            desc='2d_no_affine_large_feature'),
        ModuleInput(
            constructor_input=FunctionInput(3, 3, 1e-3, False),
            forward_input=FunctionInput(make_input((4, 3, 2, 3))),
            desc='2d_no_affine_IN'),
        ModuleInput(
            constructor_input=FunctionInput(1, 3, 1e-3, False),
            forward_input=FunctionInput(make_input((4, 3, 2, 3))),
            desc='2d_no_affine_LN'),
    ]


# 构造函数，生成一组 ModuleInput 对象的列表，用于 nn.Hardshrink 模块的测试输入
def module_inputs_torch_nn_Hardshrink(module_info, device, dtype, requires_grad, training, **kwargs):
    # 部分应用 make_tensor 函数，固定了部分参数，以便在构造输入时使用
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回 ModuleInput 对象的列表，每个对象描述了不同的测试用例
    return [
        ModuleInput(
            constructor_input=FunctionInput(2.),  # 构造函数输入，用于初始化 nn.Hardshrink 模块
            forward_input=FunctionInput(make_input((4, 3, 2, 4))),  # 模块前向输入，传递给 nn.Hardshrink 模块的输入
        ),
        ModuleInput(
            constructor_input=FunctionInput(2.),
            forward_input=FunctionInput(make_input(())),  # 模块前向输入，传递给 nn.Hardshrink 模块的输入
            desc='scalar',  # 描述该测试用例的简短标识符
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),  # 模块前向输入，传递给 nn.Hardshrink 模块的输入
            reference_fn=no_batch_dim_reference_fn,  # 参考函数，用于无批次维度情况下的比较
            desc='no_batch_dim',  # 描述该测试用例的简短标识符
        )
    ]


# 构造函数，生成一组 ModuleInput 对象的列表，用于 nn.Hardswish 模块的测试输入
def module_inputs_torch_nn_Hardswish(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个局部函数 make_input，使用 partial 函数创建一个预定义参数的版本
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个包含两个 ModuleInput 对象的列表
    return [
        # 第一个 ModuleInput 对象，使用 FunctionInput 作为构造器输入，forward_input 是使用 make_input 创建的张量
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim',  # 描述信息指示这是一个没有批次维度的输入
        ),
        # 第二个 ModuleInput 对象，同样使用 FunctionInput 作为构造器输入，forward_input 是使用 make_input 创建的四维张量
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((2, 3, 2, 5))),
            desc='4d_input'  # 描述信息指示这是一个四维输入
        )
    ]
# 定义一个函数，生成特定类型的输入张量的部分函数，用于创建具有指定设备、数据类型和梯度属性的张量
def module_inputs_torch_nn_Hardtanh(module_info, device, dtype, requires_grad, training, **kwargs):
    # 偏函数，用于创建张量，固定设备、数据类型和梯度属性
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个列表，包含不同的 ModuleInput 对象
    return [
        ModuleInput(
            constructor_input=FunctionInput(),  # 构造函数输入为空
            forward_input=FunctionInput(make_input((3, 2, 5))),  # 前向输入为形状为 (3, 2, 5) 的张量
            reference_fn=lambda m, p, i: i.clamp(-1, 1),  # 参考函数用于将输入张量限制在 [-1, 1] 范围内
        ),
        ModuleInput(
            constructor_input=FunctionInput(),  # 构造函数输入为空
            forward_input=FunctionInput(make_input(())),  # 前向输入为标量张量
            reference_fn=lambda m, p, i: i.clamp(-1, 1),  # 参考函数用于将输入张量限制在 [-1, 1] 范围内
            desc='scalar',  # 描述为标量
        ),
        ModuleInput(
            constructor_input=FunctionInput(),  # 构造函数输入为空
            forward_input=FunctionInput(make_input(4)),  # 前向输入为形状为 (4,) 的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数为 no_batch_dim_reference_fn
            desc='no_batch_dim',  # 描述为无批次维度
        )
    ]


# 定义一个函数，生成特定类型的输入张量和目标张量的部分函数，用于创建具有指定设备、数据类型和梯度属性的张量
def module_inputs_torch_nn_HingeEmbeddingLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 偏函数，用于创建输入张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 偏函数，用于创建目标张量
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # 不同测试用例的列表，每个测试用例是描述和构造函数参数字典的元组
    cases: List[Tuple[str, dict]] = [
        ('', {}),  # 空字典的测试用例
        ('reduction_sum', {'reduction': 'sum'}),  # reduction 参数为 sum 的测试用例
        ('reduction_mean', {'reduction': 'mean'}),  # reduction 参数为 mean 的测试用例
        ('reduction_none', {'reduction': 'none'}),  # reduction 参数为 none 的测试用例
        ('margin', {'margin': 0.5})  # margin 参数为 0.5 的测试用例
    ]

    module_inputs = []  # 初始化 ModuleInput 对象列表
    # 遍历所有测试用例
    for desc, constructor_kwargs in cases:
        # 定义参考函数，接受模块、参数、输入张量、目标张量和构造函数参数
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return hingeembeddingloss_reference(i, t, **constructor_kwargs)  # 调用 hingeembeddingloss_reference 函数

        # 添加一个 ModuleInput 对象到列表，包括构造函数输入、前向输入、描述和参考函数
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((10,)),  # 创建形状为 (10,) 的输入张量
                                                    make_target((10,)).gt(0).to(dtype).mul_(2).sub_(1)),  # 创建目标张量并进行操作
                        desc=desc,  # 描述为当前测试用例的描述
                        reference_fn=reference_fn)  # 参考函数为定义的 reference_fn
        )
        # 添加另一个 ModuleInput 对象到列表，与上一个对象类似，但前向输入是标量张量
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input(()),
                                                    make_target(()).gt(0).to(dtype).mul_(2).sub_(1)),  # 创建标量形式的前向输入
                        desc=f'scalar_{desc}',  # 描述为标量形式的当前测试用例描述
                        reference_fn=reference_fn)  # 参考函数为定义的 reference_fn
        )

    return module_inputs  # 返回 ModuleInput 对象列表


# 定义一个函数，生成特定类型的输入张量的部分函数，用于创建具有指定设备、数据类型和梯度属性的张量
def module_inputs_torch_nn_HuberLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 偏函数，用于创建张量，固定设备、数据类型和梯度属性
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 不同测试用例的列表，每个测试用例是描述和构造函数参数字典的元组
    cases: List[Tuple[str, dict]] = [
        ('', {}),  # 空字典的测试用例
        ('reduction_sum', {'reduction': 'sum'}),  # reduction 参数为 sum 的测试用例
        ('reduction_mean', {'reduction': 'mean'}),  # reduction 参数为 mean 的测试用例
        ('reduction_none', {'reduction': 'none'}),  # reduction 参数为 none 的测试用例
    ]

    module_inputs = []  # 初始化 ModuleInput 对象列表
    # 对于每个测试用例 cases 中的描述和构造函数参数
    for desc, constructor_kwargs in cases:
        # 定义一个参考函数 reference_fn，该函数调用 huberloss_reference 函数，传入参数 i 和 t，
        # 同时将构造函数的关键字参数 constructor_kwargs 作为默认参数传入
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return huberloss_reference(i, t, **constructor_kwargs)

        # 将以下内容添加到 module_inputs 列表中：
        # - constructor_input: 使用构造函数的关键字参数创建一个 FunctionInput 对象
        # - forward_input: 使用 make_input((5, 10)) 创建一个输入数据的 FunctionInput 对象
        # - desc: 当前测试用例的描述
        # - reference_fn: 上面定义的 reference_fn 函数
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((5, 10)),
                                                    make_input((5, 10))),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    # 返回包含所有 module_inputs 的列表
    return module_inputs
# 定义函数 module_inputs_torch_nn_InstanceNormNd，接受多个参数和关键字参数
def module_inputs_torch_nn_InstanceNormNd(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建 make_input 函数的偏函数，指定了设备、数据类型和梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 从 kwargs 中获取 lazy 参数，默认为 False
    lazy = kwargs.get('lazy', False)
    # 从 kwargs 中获取 N 参数
    N = kwargs['N']
    # 设置 num_features, eps, momentum, affine, track_running_stats 的初始值
    num_features, eps, momentum, affine, track_running_stats = 3, 1e-3, 0.3, False, True
    # 根据不同的 N 值选择对应的 input_no_batch_shape
    input_no_batch_shape_dict = {1: (3, 15), 2: (3, 6, 6), 3: (3, 4, 4, 4)}
    input_no_batch_shape = input_no_batch_shape_dict[N]
    # 组合得到 input_batch_shape，添加了一个维度 4
    input_batch_shape = (4,) + input_no_batch_shape

    # 返回包含四个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum) if lazy else FunctionInput(num_features, eps, momentum)
            ),
            forward_input=FunctionInput(make_input(input_batch_shape))),
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum, affine, track_running_stats) if lazy else
                FunctionInput(num_features, eps, momentum, affine, track_running_stats)
            ),
            forward_input=FunctionInput(make_input(input_batch_shape)),
            desc='tracking_stats'),
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum) if lazy else FunctionInput(num_features, eps, momentum)
            ),
            forward_input=FunctionInput(make_input(input_no_batch_shape)),
            reference_fn=no_batch_dim_reference_fn,
            desc='tracking_stats_no_batch_dim'),
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum, affine, track_running_stats) if lazy else
                FunctionInput(num_features, eps, momentum, affine, track_running_stats)
            ),
            forward_input=FunctionInput(make_input(input_no_batch_shape)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim')
    ]
    # 返回一个包含多个 ModuleInput 对象的列表，每个对象代表一个模块输入的描述
    return [
        # 第一个 ModuleInput 对象
        ModuleInput(
            # 使用 FunctionInput 构造器创建输入，参数是一个长度为 5 的数组和一个很小的数
            constructor_input=FunctionInput([5], 1e-3),
            # 使用 make_input 函数创建输入，参数是一个形状为 (4, 5, 5) 的输入数据
            forward_input=FunctionInput(make_input((4, 5, 5))),
            # 描述信息为 '1d_elementwise_affine'
            desc='1d_elementwise_affine'
        ),
        # 第二个 ModuleInput 对象，与第一个类似，但输入数据的形状更大
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((128, 5, 5))),
            desc='1d_elementwise_affine_large_batch'
        ),
        # 第三个 ModuleInput 对象，构造器输入包含一个额外的布尔参数
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3, False),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_no_elementwise_affine'
        ),
        # 第四个 ModuleInput 对象，输入数据是一个形状为 (4, 2, 2, 5) 的数据
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_elementwise_affine'
        ),
        # 第五个 ModuleInput 对象，构造器输入包含额外的布尔参数
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3, False),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_no_elementwise_affine'
        ),
        # 第六个 ModuleInput 对象，输入数据是一个形状为 (0, 5) 的空数据
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((0, 5))),
            desc='1d_empty_elementwise_affine'
        ),
        # 第七个 ModuleInput 对象，构造器输入包含额外的布尔参数，并且不包含偏置
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3, elementwise_affine=True, bias=False),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_elementwise_affine_no_bias'
        ),
    ]
# 定义一个函数，用于生成 Torch 的神经网络模块的输入配置
def module_inputs_torch_nn_RMSNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用的函数，用于生成张量，并指定设备、数据类型、梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 定义一个参考函数 rms_norm_reference_fn，用于计算 RMS 标准化
    def rms_norm_reference_fn(m, p, i):
        # 获取 eps 值，若未指定则使用输入张量数据类型的机器精度 eps
        eps = m.eps
        if eps is None:
            eps = torch.finfo(i.dtype).eps
        # 获取输入张量的维度数
        ndim = i.ndim
        # 获取标准化形状
        normalized_shape = m.normalized_shape
        # 获取权重
        weight = m.weight
        # 计算 dims 列表，用于指定平均计算的维度
        dims = [ndim - i - 1 for i in range(len(normalized_shape))]
        # 计算 RMS 标准化的结果
        result = i * torch.rsqrt(i.pow(2).mean(dim=dims, keepdim=True) + m.eps)
        # 若存在权重，则将结果乘以权重
        if weight is not None:
            result *= weight
        # 返回计算结果
        return result

    # 返回一个包含多个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((128, 5, 5))),
            desc='1d_elementwise_affine_large_batch',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3, False),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_no_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3, False),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_no_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((0, 5))),
            desc='1d_empty_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
    ]


# 定义一个函数，用于生成 Torch 的神经网络模块的输入配置
def module_inputs_torch_nn_LocalResponseNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用的函数，用于生成张量，并指定设备、数据类型、梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个包含多个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput(3,),
            forward_input=FunctionInput(make_input((1, 5, 7))),
            desc='1d'),
        ModuleInput(
            constructor_input=FunctionInput(2,),
            forward_input=FunctionInput(make_input((1, 5, 7, 7))),
            desc='2d_uneven_pad'),
        ModuleInput(
            constructor_input=FunctionInput(1, 1., 0.5, 2.),
            forward_input=FunctionInput(make_input((1, 5, 7, 7, 7))),
            desc='3d_custom_params'),
    ]


# 定义一个函数，用于生成 Torch 的神经网络模块的输入配置
def module_inputs_torch_nn_LPPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数 partial 创建一个新的函数 make_input，固定了部分参数：设备、数据类型、是否需要梯度
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个包含三个 ModuleInput 对象的列表
    return [
        # 第一个 ModuleInput 对象
        ModuleInput(
            # 使用 FunctionInput 创建构造器的输入，传入两个参数 1.5 和 2
            constructor_input=FunctionInput(1.5, 2),
            # 使用 make_input 函数创建前向传播的输入，传入一个形状为 (1, 3, 7) 的张量
            forward_input=FunctionInput(make_input((1, 3, 7))),
            # 描述为 'norm'
            desc='norm'),
        
        # 第二个 ModuleInput 对象
        ModuleInput(
            # 使用 FunctionInput 创建构造器的输入，传入三个参数 2, 2, 3
            constructor_input=FunctionInput(2, 2, 3),
            # 使用 make_input 函数创建前向传播的输入，传入一个形状为 (1, 3, 7) 的张量
            forward_input=FunctionInput(make_input((1, 3, 7)))),
        
        # 第三个 ModuleInput 对象
        ModuleInput(
            # 使用 FunctionInput 创建构造器的输入，传入两个参数 2, 2, 3
            constructor_input=FunctionInput(2, 2, 3),
            # 使用 make_input 函数创建前向传播的输入，传入一个形状为 (3, 7) 的张量
            forward_input=FunctionInput(make_input((3, 7))),
            # 设置 reference_fn 为 no_batch_dim_reference_fn
            reference_fn=no_batch_dim_reference_fn,
            # 描述为 'no_batch_dim'
            desc='no_batch_dim'),
    ]
def module_inputs_torch_nn_LPPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 部分函数make_tensor的偏函数，固定了设备、数据类型、梯度需求参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回包含三个ModuleInput对象的列表，每个对象描述了不同的模块输入
    return [
        ModuleInput(
            # 构造函数的输入描述，示例中为(2, 2, 2)
            constructor_input=FunctionInput(2, 2, 2),
            # 前向传播函数的输入描述，示例中为make_input((1, 3, 7, 7))
            forward_input=FunctionInput(make_input((1, 3, 7, 7)))),
        ModuleInput(
            # 构造函数的输入描述，示例中为(2, 2, 2)
            constructor_input=FunctionInput(2, 2, 2),
            # 前向传播函数的输入描述，示例中为make_input((3, 7, 7))
            forward_input=FunctionInput(make_input((3, 7, 7))),
            # 引用函数的参考，示例中为no_batch_dim_reference_fn
            reference_fn=no_batch_dim_reference_fn,
            # 描述字段，示例中为'no_batch_dim'
            desc='no_batch_dim'),
        ModuleInput(
            # 构造函数的输入描述，示例中为(1.5, 2)
            constructor_input=FunctionInput(1.5, 2),
            # 前向传播函数的输入描述，示例中为make_input((1, 3, 7, 7))
            forward_input=FunctionInput(make_input((1, 3, 7, 7))),
            # 描述字段，示例中为'norm'
            desc='norm'),
    ]


def module_inputs_torch_nn_LPPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 部分函数make_tensor的偏函数，固定了设备、数据类型、梯度需求参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回包含三个ModuleInput对象的列表，每个对象描述了不同的模块输入
    return [
        ModuleInput(
            # 构造函数的输入描述，示例中为(2, 2, 2)
            constructor_input=FunctionInput(2, 2, 2),
            # 前向传播函数的输入描述，示例中为make_input((1, 3, 7, 7, 7))
            forward_input=FunctionInput(make_input((1, 3, 7, 7, 7)))),
        ModuleInput(
            # 构造函数的输入描述，示例中为(2, 2, 2)
            constructor_input=FunctionInput(2, 2, 2),
            # 前向传播函数的输入描述，示例中为make_input((3, 7, 7, 7))
            forward_input=FunctionInput(make_input((3, 7, 7, 7))),
            # 引用函数的参考，示例中为no_batch_dim_reference_fn
            reference_fn=no_batch_dim_reference_fn,
            # 描述字段，示例中为'no_batch_dim'
            desc='no_batch_dim'),
        ModuleInput(
            # 构造函数的输入描述，示例中为(1.5, 2)
            constructor_input=FunctionInput(1.5, 2),
            # 前向传播函数的输入描述，示例中为make_input((1, 3, 7, 7, 7))
            forward_input=FunctionInput(make_input((1, 3, 7, 7, 7))),
            # 描述字段，示例中为'norm'
            desc='norm'),
    ]


def module_inputs_torch_nn_MaxPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 部分函数make_tensor的偏函数，固定了设备、数据类型、梯度需求参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回包含三个ModuleInput对象的列表，每个对象描述了不同的模块输入
    return [
        ModuleInput(
            # 构造函数的输入描述，示例中为(4)
            constructor_input=FunctionInput(4),
            # 前向传播函数的输入描述，示例中为make_input((2, 10, 4))
            forward_input=FunctionInput(make_input((2, 10, 4))),
            # 描述字段，示例中为'3d_input'
            desc='3d_input'),
        ModuleInput(
            # 构造函数的输入描述，示例中为(4, 4)
            constructor_input=FunctionInput(4, 4),
            # 前向传播函数的输入描述，示例中为make_input((2, 10, 4))
            forward_input=FunctionInput(make_input((2, 10, 4))),
            # 描述字段，示例中为'stride'
            desc='stride'),
        ModuleInput(
            # 构造函数的输入描述，示例中为(4, return_indices=True)
            constructor_input=FunctionInput(4, return_indices=True),
            # 前向传播函数的输入描述，示例中为make_input((2, 10, 4))
            forward_input=FunctionInput(make_input((2, 10, 4))),
            # 描述字段，示例中为'return_indices'
            desc='return_indices'),
    ]


def module_inputs_torch_nn_MaxPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 部分函数make_tensor的偏函数，固定了设备、数据类型、梯度需求参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 返回一个列表，包含三个 ModuleInput 对象
    return [
        # 第一个 ModuleInput 对象，使用指定的参数构造和前向输入
        ModuleInput(
            constructor_input=FunctionInput((3, 3), (2, 2), (1, 1)),  # 构造函数输入参数：kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            forward_input=FunctionInput(make_input((3, 7, 7))),  # 前向输入参数：3D 输入数据 (3, 7, 7)
            desc='3d_input'),  # 描述信息：3D 输入
        # 第二个 ModuleInput 对象，使用指定的参数构造和前向输入
        ModuleInput(
            constructor_input=FunctionInput((3, 3), (2, 2), (1, 1)),  # 构造函数输入参数：kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            forward_input=FunctionInput(make_input((1, 3, 7, 7))),  # 前向输入参数：4D 输入数据 (1, 3, 7, 7)
            desc='4d_input'),  # 描述信息：4D 输入
        # 第三个 ModuleInput 对象，使用指定的参数构造和前向输入，同时设置 return_indices=True
        ModuleInput(
            constructor_input=FunctionInput((3, 3), (2, 2), (1, 1), return_indices=True),  # 构造函数输入参数：kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)，同时返回索引
            forward_input=FunctionInput(make_input((1, 3, 7, 7))),  # 前向输入参数：4D 输入数据 (1, 3, 7, 7)
            desc='return_indices'),  # 描述信息：返回索引
    ]
# 定义一个函数，生成指定设备、数据类型、梯度需求的张量的部分函数
def module_inputs_torch_nn_MaxPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 偏函数，用于生成张量，指定设备、数据类型、是否需要梯度
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个列表，包含多个 ModuleInput 对象，每个对象表示一个输入
    return [
        ModuleInput(
            constructor_input=FunctionInput((2, 2, 2)),  # 构造函数的输入，是一个形状为 (2, 2, 2) 的张量
            forward_input=FunctionInput(make_input((2, 3, 5, 5, 5)))),  # 前向输入，是一个形状为 (2, 3, 5, 5, 5) 的张量
        ModuleInput(
            constructor_input=FunctionInput(2, (2, 2, 2)),  # 构造函数的输入，包含一个数值为 2 和形状为 (2, 2, 2) 的元组
            forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),  # 前向输入，是一个形状为 (2, 3, 5, 5, 5) 的张量
            desc='stride'),  # 描述信息，指定为 'stride'
        ModuleInput(
            constructor_input=FunctionInput(2, 2, (1, 1, 1)),  # 构造函数的输入，包含两个数值 2 和一个形状为 (1, 1, 1) 的元组
            forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),  # 前向输入，是一个形状为 (2, 3, 5, 5, 5) 的张量
            desc='stride_padding'),  # 描述信息，指定为 'stride_padding'
        ModuleInput(
            constructor_input=FunctionInput(2, 2, (1, 1, 1), return_indices=True),  # 构造函数的输入，包含两个数值 2、一个形状为 (1, 1, 1) 的元组和 return_indices=True
            forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),  # 前向输入，是一个形状为 (2, 3, 5, 5, 5) 的张量
            desc='return_indices'),  # 描述信息，指定为 'return_indices'
    ]


# 定义一个函数，生成指定设备、数据类型、梯度需求的张量的部分函数
def module_inputs_torch_nn_FractionalMaxPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 偏函数，用于生成张量，指定设备、数据类型、是否需要梯度
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 定义一个内部函数，生成随机样本张量
    def make_random_samples():
        return torch.empty((1, 3, 2), dtype=torch.double, device=device).uniform_()

    # 返回一个列表，包含多个 ModuleInput 对象，每个对象表示一个输入
    return [
        ModuleInput(
            constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples()),  # 构造函数的输入，包含数值 2、output_ratio=0.5 和 _random_samples=make_random_samples() 的 FunctionInput 对象
            forward_input=FunctionInput(make_input((1, 3, 5, 7))),  # 前向输入，是一个形状为 (1, 3, 5, 7) 的张量
            desc='ratio'),  # 描述信息，指定为 'ratio'
        ModuleInput(
            constructor_input=FunctionInput((2, 3), output_size=(4, 3), _random_samples=make_random_samples()),  # 构造函数的输入，包含元组 (2, 3)、output_size=(4, 3) 和 _random_samples=make_random_samples() 的 FunctionInput 对象
            forward_input=FunctionInput(make_input((1, 3, 7, 6))),  # 前向输入，是一个形状为 (1, 3, 7, 6) 的张量
            desc='size'),  # 描述信息，指定为 'size'
        ModuleInput(
            constructor_input=FunctionInput(
                2, output_ratio=0.5, _random_samples=make_random_samples(), return_indices=True
            ),  # 构造函数的输入，包含数值 2、output_ratio=0.5、_random_samples=make_random_samples() 和 return_indices=True 的 FunctionInput 对象
            forward_input=FunctionInput(make_input((1, 3, 5, 7))),  # 前向输入，是一个形状为 (1, 3, 5, 7) 的张量
            desc='ratio_return_indices'),  # 描述信息，指定为 'ratio_return_indices'
        ModuleInput(
            constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples()),  # 构造函数的输入，包含数值 2、output_ratio=0.5 和 _random_samples=make_random_samples() 的 FunctionInput 对象
            forward_input=FunctionInput(make_input((3, 5, 7))),  # 前向输入，是一个形状为 (3, 5, 7) 的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数，指定为 no_batch_dim_reference_fn
            desc='ratio_no_batch_dim'),  # 描述信息，指定为 'ratio_no_batch_dim'
        ModuleInput(
            constructor_input=FunctionInput((2, 3), output_size=(4, 3), _random_samples=make_random_samples()),  # 构造函数的输入，包含元组 (2, 3)、output_size=(4, 3) 和 _random_samples=make_random_samples() 的 FunctionInput 对象
            forward_input=FunctionInput(make_input((3, 7, 6))),  # 前向输入，是一个形状为 (3, 7, 6) 的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数，指定为 no_batch_dim_reference_fn
            desc='size_no_batch_dim'),  # 描述信息，指定为 'size_no_batch_dim'
    ]


# 定义一个函数，生成指定设备、数据类型、梯度需求的张量的部分函数
def module_inputs_torch_nn_FractionalMaxPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 偏函数，用于生成张量，指定设备、数据类型、是否需要梯度
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 定义一个内部函数，生成随机样本张量
    def make_random_samples():
        return torch.empty((2, 4, 3), dtype=torch.double, device=device).uniform_()
    返回一个包含多个 ModuleInput 对象的列表，每个对象描述了模块的输入配置和描述信息

    return [
        # 创建 ModuleInput 对象，指定构造函数的输入参数和描述信息为 'ratio'
        ModuleInput(
            constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((2, 4, 5, 5, 5))),
            desc='ratio'),
        # 创建 ModuleInput 对象，指定构造函数的输入参数和描述信息为 'size'
        ModuleInput(
            constructor_input=FunctionInput((2, 2, 2), output_size=(4, 4, 4), _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((2, 4, 7, 7, 7))),
            desc='size'),
        # 创建 ModuleInput 对象，指定构造函数的输入参数和描述信息为 'asymsize'
        ModuleInput(
            constructor_input=FunctionInput((4, 2, 3), output_size=(10, 3, 2), _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((2, 4, 16, 7, 5))),
            desc='asymsize'),
        # 创建 ModuleInput 对象，指定构造函数的输入参数和描述信息为 'ratio_return_indices'
        ModuleInput(
            constructor_input=FunctionInput(
                2, output_ratio=0.5, _random_samples=make_random_samples(), return_indices=True
            ),
            forward_input=FunctionInput(make_input((2, 4, 5, 5, 5))),
            desc='ratio_return_indices'),
        # 创建 ModuleInput 对象，指定构造函数的输入参数和描述信息为 'ratio_no_batch_dim'，并指定参考函数
        ModuleInput(
            constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((4, 5, 5, 5))),
            reference_fn=no_batch_dim_reference_fn,
            desc='ratio_no_batch_dim'),
        # 创建 ModuleInput 对象，指定构造函数的输入参数和描述信息为 'size_no_batch_dim'，并指定参考函数
        ModuleInput(
            constructor_input=FunctionInput((2, 2, 2), output_size=(4, 4, 4), _random_samples=make_random_samples()),
            forward_input=FunctionInput(make_input((4, 7, 7, 7))),
            reference_fn=no_batch_dim_reference_fn,
            desc='size_no_batch_dim'),
    ]
def module_inputs_torch_nn_Sigmoid(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数 partial 设置 make_input 函数的默认参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个包含多个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput(),  # 构造函数的输入为一个空的 FunctionInput 对象
            forward_input=FunctionInput(make_input(())),  # 前向输入为调用 make_input 创建的标量张量
            desc='scalar'  # 描述当前 ModuleInput 对象的名称为 'scalar'
        ),
        ModuleInput(
            constructor_input=FunctionInput(),  # 构造函数的输入为一个空的 FunctionInput 对象
            forward_input=FunctionInput(make_input(4)),  # 前向输入为调用 make_input 创建的大小为 4 的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数为 no_batch_dim_reference_fn
            desc='no_batch_dim',  # 描述当前 ModuleInput 对象的名称为 'no_batch_dim'
        ),
        ModuleInput(
            constructor_input=FunctionInput(),  # 构造函数的输入为一个空的 FunctionInput 对象
            forward_input=FunctionInput(make_input((2, 3, 4, 5))),  # 前向输入为调用 make_input 创建的 4 维张量
            desc='channels_last_mem_format'  # 描述当前 ModuleInput 对象的名称为 'channels_last_mem_format'
        ),
        ModuleInput(
            constructor_input=FunctionInput(),  # 构造函数的输入为一个空的 FunctionInput 对象
            forward_input=FunctionInput(make_input((2, 3, 3, 4, 5))),  # 前向输入为调用 make_input 创建的 5 维张量
            desc='channels_last_3d_mem_format'  # 描述当前 ModuleInput 对象的名称为 'channels_last_3d_mem_format'
        )
    ]


def module_inputs_torch_nn_LogSigmoid(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数 partial 设置 make_input 函数的默认参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个包含多个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput(),  # 构造函数的输入为一个空的 FunctionInput 对象
            forward_input=FunctionInput(make_input(())),  # 前向输入为调用 make_input 创建的标量张量
            reference_fn=lambda m, p, i: i.sigmoid().log(),  # 自定义的参考函数，对输入进行 sigmoid 和 log 操作
            desc='scalar'  # 描述当前 ModuleInput 对象的名称为 'scalar'
        ),
        ModuleInput(
            constructor_input=FunctionInput(),  # 构造函数的输入为一个空的 FunctionInput 对象
            forward_input=FunctionInput(make_input((2, 3, 4))),  # 前向输入为调用 make_input 创建的大小为 (2, 3, 4) 的张量
            reference_fn=lambda m, p, i: i.sigmoid().log(),  # 自定义的参考函数，对输入进行 sigmoid 和 log 操作
        ),
        ModuleInput(
            constructor_input=FunctionInput(),  # 构造函数的输入为一个空的 FunctionInput 对象
            forward_input=FunctionInput(make_input(4)),  # 前向输入为调用 make_input 创建的大小为 4 的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数为 no_batch_dim_reference_fn
            desc='no_batch_dim',  # 描述当前 ModuleInput 对象的名称为 'no_batch_dim'
        ),
    ]


def module_inputs_torch_nn_MarginRankingLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数 partial 设置 make_input 和 make_target 函数的默认参数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)

    # 定义一个包含多个元组的列表
    cases: List[Tuple[str, dict]] = [
        ('', {}),  # 空字符串和空字典的元组
        ('reduction_sum', {'reduction': 'sum'}),  # 字符串 'reduction_sum' 和包含 'reduction' 键的字典
        ('reduction_mean', {'reduction': 'mean'}),  # 字符串 'reduction_mean' 和包含 'reduction' 键的字典
        ('reduction_none', {'reduction': 'none'}),  # 字符串 'reduction_none' 和包含 'reduction' 键的字典
        ('margin', {'margin': 0.5})  # 字符串 'margin' 和包含 'margin' 键的字典
    ]

    module_inputs = []  # 初始化一个空列表，用于存储 ModuleInput 对象

    # 遍历 cases 列表中的元组
    for desc, constructor_kwargs in cases:
        # 定义一个包含多个参数的自定义参考函数 reference_fn
        def reference_fn(m, p, i1, i2, t, constructor_kwargs=constructor_kwargs):
            return marginrankingloss_reference(i1, i2, t, **constructor_kwargs)  # 调用 marginrankingloss_reference 函数

        # 将新的 ModuleInput 对象添加到 module_inputs 列表中
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),  # 根据 constructor_kwargs 创建构造函数输入
                        forward_input=FunctionInput(make_input((50,)), make_input((50,)),  # 前向输入为调用 make_input 创建的张量
                                                    make_target((50,)).sign()),  # 调用 make_target 创建的张量的符号
                        desc=desc,  # 描述当前 ModuleInput 对象的名称为 desc
                        reference_fn=reference_fn)  # 设置当前 ModuleInput 对象的参考函数为 reference_fn
        )
    # 返回函数的输入模块
    return module_inputs
def module_inputs_torch_nn_MultiLabelMarginLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 部分函数定义，用于创建输入张量和目标张量，设定设备、数据类型和梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)

    # 不同情况下的测试用例列表
    cases: List[Tuple[str, dict]] = [
        ('', {}),  # 空情况，使用默认参数
        ('reduction_sum', {'reduction': 'sum'}),  # 指定 reduction 参数为 sum
        ('reduction_mean', {'reduction': 'mean'}),  # 指定 reduction 参数为 mean
        ('reduction_none', {'reduction': 'none'}),  # 指定 reduction 参数为 none
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        # 定义参考函数，调用自定义的 multilabelmarginloss_reference 函数
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return multilabelmarginloss_reference(i, t, **constructor_kwargs)

        # 构建 ModuleInput 对象，用于描述每个测试用例
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((10,)),
                                                    make_target((10), low=0, high=10)),
                        desc=f'1d_{desc}',  # 描述，包括描述符号和描述
                        reference_fn=reference_fn)
        )

        # 构建 ModuleInput 对象，用于描述每个测试用例的另一种情况
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((5, 10)),
                                                    make_target((5, 10), low=0, high=10)),
                        desc=desc,  # 描述符号
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_MultiMarginLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 部分函数定义，用于创建输入张量、目标张量和权重张量，设定设备、数据类型和梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # 不同情况下的测试用例列表
    cases: List[Tuple[str, dict]] = [
        ('', {}),  # 空情况，使用默认参数
        ('reduction_sum', {'reduction': 'sum'}),  # 指定 reduction 参数为 sum
        ('reduction_mean', {'reduction': 'mean'}),  # 指定 reduction 参数为 mean
        ('reduction_none', {'reduction': 'none'}),  # 指定 reduction 参数为 none
        ('p', {'p': 2}),  # 指定 p 参数为 2
        ('margin', {'margin': 0.5}),  # 指定 margin 参数为 0.5
        ('weights', {'weight': make_weight(10)})  # 指定 weight 参数为 10 个元素的张量
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        # 定义参考函数，调用自定义的 multimarginloss_reference 函数
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return multimarginloss_reference(i, t, **constructor_kwargs)

        # 构建 ModuleInput 对象，用于描述每个测试用例
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((5, 10)),
                                                    make_target((5), low=0, high=10)),
                        desc=desc,  # 描述符号
                        reference_fn=reference_fn)
        )

    return module_inputs
    # 使用偏函数 partial 创建 make_input 函数，指定默认的设备、数据类型、梯度计算需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 使用偏函数 partial 创建 make_target 函数，指定默认的设备、数据类型为 long，不需要梯度计算
    make_target = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)
    # 使用偏函数 partial 创建 make_weight 函数，指定默认的设备、数据类型为 dtype，不需要梯度计算
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # 定义一个包含不同测试用例的列表 cases，每个元素是一个元组，包含描述字符串和构造函数的参数字典
    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('weight', {'weight': make_weight(10)}),
    ]

    # 定义一个函数 multilabelsoftmargin_loss_reference_fn，计算多标签软边界损失
    def multilabelsoftmargin_loss_reference_fn(m, p, i, t, reduction='mean', weight=None):
        # 计算损失结果，根据标签 t 和预测值 i 的 sigmoid 函数计算得到的交叉熵
        result = t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()
        # 如果提供了权重 weight，则将结果乘以权重
        if weight is not None:
            result *= weight
        # 对最后一个维度求和，然后除以张量 i 最后一个维度的大小，得到平均损失
        result = (-result).sum(i.dim() - 1) / i.size(-1)

        # 根据 reduction 参数进行不同的损失值计算
        if reduction == 'none':
            return result  # 返回未约简的损失结果
        elif reduction == 'mean':
            return result.mean()  # 返回平均损失结果
        else:
            return result.sum()  # 返回总和损失结果

    # 定义一个空列表 module_inputs，用于存储各种输入情况下的模块输入对象
    module_inputs = []
    # 遍历 cases 列表，每个元素构造一个 ModuleInput 对象并添加到 module_inputs 中
    for desc, constructor_kwargs in cases:
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),  # 使用构造函数参数构造构造器输入
                forward_input=FunctionInput(
                    make_input((5, 10)),  # 调用 make_input 函数创建输入张量
                    make_target((5, 10), low=0, high=2)  # 调用 make_target 函数创建目标张量
                ),
                desc=desc,  # 描述字符串
                reference_fn=partial(multilabelsoftmargin_loss_reference_fn, **constructor_kwargs)  # 多标签软边界损失函数的偏函数
            )
        )

    return module_inputs  # 返回构造好的 module_inputs 列表作为模块输入
# 定义一个函数，用于生成张量，并部分固定参数（device, dtype, requires_grad）
def module_inputs_torch_nn_SoftMarginLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    # 部分应用函数make_tensor，生成用于输入的张量函数make_input
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 生成用于目标张量的函数make_target，不需要梯度
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # 定义多个测试用例，每个用例是一个元组，包含一个描述字符串和构造函数的关键字参数
    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
    ]

    module_inputs = []
    # 遍历测试用例
    for desc, constructor_kwargs in cases:
        # 定义一个参考函数，调用softmarginloss_reference函数，使用测试用例中的构造函数关键字参数
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return softmarginloss_reference(i, t, **constructor_kwargs)

        # 构造ModuleInput对象，包括构造函数的输入、前向输入、描述和参考函数
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((5, 5)), make_target((5, 5)).sign()),
                desc=desc,
                reference_fn=reference_fn
            )
        )

    # 返回所有构造的ModuleInput对象列表
    return module_inputs


# 定义一个函数，生成TransformerEncoder模块的输入样本列表
def module_inputs_torch_nn_TransformerEncoder(module_info, device, dtype, requires_grad, training, **kwargs):
    # 调用module_inputs_torch_nn_TransformerEncoderLayer生成TransformerEncoderLayer模块的输入样本列表
    samples = []
    for layer_module_input in module_inputs_torch_nn_TransformerEncoderLayer(
            None, device, dtype, requires_grad, training):
        # 从模块输入中获取构造函数的参数和关键字参数
        l_args, l_kwargs = (layer_module_input.constructor_input.args,
                            layer_module_input.constructor_input.kwargs)
        # 设置关键字参数中的设备和数据类型
        l_kwargs['device'] = device
        l_kwargs['dtype'] = dtype
        # 构造TransformerEncoderLayer对象，传递给TransformerEncoder使用
        encoder_layer = torch.nn.TransformerEncoderLayer(*l_args, **l_kwargs)
        num_layers = 2
        # 注意：TransformerEncoderLayer接受"src_mask"参数，而TransformerEncoder接受"mask"参数；对关键字参数进行重命名
        forward_input = layer_module_input.forward_input
        if 'src_mask' in forward_input.kwargs:
            forward_input.kwargs['mask'] = forward_input.kwargs['src_mask']
            del forward_input.kwargs['src_mask']
        # 构造ModuleInput对象，包括构造函数的输入、前向输入和描述
        samples.append(ModuleInput(
            constructor_input=FunctionInput(encoder_layer, num_layers),
            forward_input=forward_input,
            desc=layer_module_input.desc
        ))
    # 返回所有构造的ModuleInput对象列表
    return samples

# 定义一个函数，生成TransformerEncoderLayer模块的输入样本列表
def module_inputs_torch_nn_TransformerEncoderLayer(module_info, device, dtype, requires_grad, training, **kwargs):
    # 部分应用函数make_tensor，生成用于输入的张量函数make_input
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = [
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 16, 0.0),  # 创建 ModuleInput 对象，传入构造函数的参数
            forward_input=FunctionInput(                      # 创建 FunctionInput 对象，作为 ModuleInput 的前向输入
                make_input((2, 3, 4))                         # 调用 make_input 函数生成输入数据
            ),
            desc='relu_activation'                            # 描述当前样本的字符串标识
        ),
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 8, 0.0, F.gelu),  # 创建 ModuleInput 对象，传入构造函数的参数和激活函数
            forward_input=FunctionInput(                            # 创建 FunctionInput 对象，作为 ModuleInput 的前向输入
                make_input((2, 3, 4))                               # 调用 make_input 函数生成输入数据
            ),
            desc='gelu_activation'                                  # 描述当前样本的字符串标识
        ),
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 8, 0.0, bias=False),  # 创建 ModuleInput 对象，传入构造函数的参数并禁用偏置
            forward_input=FunctionInput(                                # 创建 FunctionInput 对象，作为 ModuleInput 的前向输入
                make_input((2, 3, 4))                                   # 调用 make_input 函数生成输入数据
            ),
            desc='no_bias'                                              # 描述当前样本的字符串标识
        ),
    ]

    # Samples below are for validating the no-batch-dim support.
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))  # 定义关键填充掩码
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3)))  # 定义注意力掩码
    for src_mask, src_key_padding_mask, norm_first, batch_first, bias in \
            itertools.product(attn_masks, key_padding_masks, (True, False), (True, False), (True, False)):
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,  # 创建 ModuleInput 对象，传入构造函数的参数
                                                dropout=0.0, batch_first=batch_first,
                                                norm_first=norm_first, bias=bias),
                forward_input=FunctionInput(                                          # 创建 FunctionInput 对象，作为 ModuleInput 的前向输入
                    make_input((3, 4)), src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
                ),
                reference_fn=partial(no_batch_dim_reference_fn,                      # 设置参考函数 partial(no_batch_dim_reference_fn)
                                     batch_first=batch_first, kwargs_to_batchify={'src_key_padding_mask': 0}),
                desc=f'no_batch_dim_batch_first_{batch_first}'                       # 描述当前样本的字符串标识
            ))

    # Samples below where we pass reference_fn are for validating the fast path,
    # since the fast path requires no_grad mode, we run the fast path in .eval()
    # and no_grad() in the reference_fn and verify that against the results in train mode.
    def fast_path_reference_fn(module, parameters, *args, **kwargs):
        assert module.training                              # 断言模型当前为训练模式
        module.train(False)                                 # 将模型设为评估模式（eval mode）
        with torch.no_grad():                               # 使用 torch.no_grad() 禁用梯度计算
            output = module(*args, **kwargs)                # 在评估模式下运行模型
        module.train(True)                                  # 恢复模型为训练模式
        return output                                       # 返回模型输出
    if training:
        # 如果处于训练模式，执行以下操作：
        for norm_first, bias in itertools.product((True, False), (True, False)):
            # 使用 itertools.product 生成 (True, False) 的组合，分别赋给 norm_first 和 bias
            samples.append(
                # 将以下内容添加到 samples 列表中：
                ModuleInput(
                    constructor_input=FunctionInput(
                        4, 2, 8, dropout=0.0, batch_first=True, norm_first=norm_first, bias=bias
                    ),
                    # 构造 ModuleInput 对象，使用 FunctionInput 作为构造器的输入参数
                    forward_input=FunctionInput(
                        make_input((2, 3, 4)),
                    ),
                    # 设置 reference_fn 为 fast_path_reference_fn，当 bias=True；否则为 None
                    reference_fn=fast_path_reference_fn if bias else None,
                    # 设置描述，说明 fastpath 在 bias=False 时不运行
                    desc=f'fastpath_{bias}_norm_first_{norm_first}'
                )
            )

    # 返回 samples 列表作为函数结果
    return samples
def module_inputs_torch_nn_TransformerDecoderLayer(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用函数，用于创建张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 定义输入样本列表
    samples = [
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 16, 0.0),  # 构造函数的输入参数
            forward_input=FunctionInput(
                make_input((2, 3, 4)), make_input((2, 3, 4))  # 前向输入的张量数据
            ),
            desc='relu_activation'  # 模块输入的描述信息
        ),
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 8, 0.0, F.gelu),  # 使用 gelu 激活函数的构造函数输入参数
            forward_input=FunctionInput(
                make_input((2, 3, 4)), make_input((2, 3, 4))  # 前向输入的张量数据
            ),
            desc='gelu_activation'  # 模块输入的描述信息
        ),
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 8, 0.0, bias=False),  # 不包含偏置项的构造函数输入参数
            forward_input=FunctionInput(
                make_input((2, 3, 4)), make_input((2, 3, 4))  # 前向输入的张量数据
            ),
            desc='no_bias'  # 模块输入的描述信息
        ),
    ]

    # 定义键填充掩码列表，第一个元素为None，第二个元素为 torch 张量
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))

    # 定义注意力掩码列表，第一个元素为None，第二个元素为 torch 张量，复制成 (3, 3) 的形状
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3)))
    for tgt_mask, tgt_key_padding_mask, norm_first, bias, batch_first in \
            itertools.product(attn_masks, key_padding_masks, (True, False), (True, False), (True, False)):
        # 使用 itertools.product 生成所有可能的参数组合，遍历每个组合
        # tgt_mask: 目标掩码
        # tgt_key_padding_mask: 目标关键填充掩码
        # norm_first: 是否先进行归一化
        # bias: 是否使用偏置
        # batch_first: 是否批量优先

        # 使用相同的掩码作为目标和记忆的掩码
        memory_mask = tgt_mask
        memory_key_padding_mask = tgt_key_padding_mask

        # 创建 ModuleInput 对象并添加到 samples 列表中
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                dropout=0.0, batch_first=batch_first,
                                                norm_first=norm_first, bias=bias),
                forward_input=FunctionInput(
                    make_input((3, 4)), make_input((3, 4)), tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask
                ),
                reference_fn=partial(no_batch_dim_reference_fn,
                                     batch_first=batch_first,
                                     kwargs_to_batchify={'tgt_key_padding_mask': 0, 'memory_key_padding_mask': 0}),
                desc=f'no_batch_dim_batch_first_{batch_first}'
            ))

        # 创建输入数据 src 和 tgt
        src, tgt = make_input((2, 3, 4)), make_input((2, 3, 4))
        
        # 如果不是批量优先，对 src 和 tgt 进行转置
        if not batch_first:
            src, tgt = src.transpose(0, 1), tgt.transpose(0, 1)
        
        # 如果存在目标关键填充掩码
        if tgt_key_padding_mask is not None:
            # 扩展目标关键填充掩码
            memory_key_padding_mask, tgt_key_padding_mask = (tgt_key_padding_mask.expand(2, 3),) * 2
        
        # 创建 ModuleInput 对象并添加到 samples 列表中
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                dropout=0.0, batch_first=batch_first,
                                                norm_first=norm_first, bias=bias),
                forward_input=FunctionInput(
                    src, tgt, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask
                ),
                desc=f'norm_first_{norm_first}_batch_first_{batch_first}_bias_{bias}'
            ))

    # 返回生成的样本列表
    return samples
# 定义一个函数 module_inputs_torch_nn_Transformer，接受多个参数，返回一个样本列表
def module_inputs_torch_nn_Transformer(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数 make_tensor 创建一个局部函数 make_input，用于创建张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 初始化空列表 samples，用于存储样本数据
    samples = []
    # 下面的样本是为了验证不支持批次维度的情况。
    # 定义 key_padding_masks，包括 None 和一个 Torch 张量作为元组的两个元素
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))
    # 定义 attn_masks，包括 None 和一个扩展后的 Torch 张量作为元组的两个元素
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3)))
    # 使用 itertools.product 生成参数的笛卡尔积，用于迭代生成样本
    for mask, key_padding_mask, norm_first, bias, batch_first in \
            itertools.product(attn_masks, key_padding_masks, (True, False), (True, False), (True, False)):
        # 对于每一对 mask，使用相同的 mask 分别赋值给 src_mask 和 tgt_mask
        src_mask, tgt_mask = (mask,) * 2
        # 对于每一对 key_padding_mask，使用相同的 key_padding_mask 分别赋值给 src_key_padding_mask 和 tgt_key_padding_mask
        src_key_padding_mask, tgt_key_padding_mask = (key_padding_mask,) * 2
        # 创建 ModuleInput 对象并添加到 samples 列表中，每个对象包括 constructor_input、forward_input、reference_fn 和 desc
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                num_encoder_layers=1, num_decoder_layers=1,
                                                dropout=0.0, batch_first=batch_first, norm_first=norm_first, bias=bias),
                forward_input=FunctionInput(
                    make_input((3, 4)), make_input((3, 4)), tgt_mask=tgt_mask, src_mask=src_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, src_key_padding_mask=src_key_padding_mask
                ),
                reference_fn=partial(no_batch_dim_reference_fn,
                                     batch_first=batch_first,
                                     kwargs_to_batchify={'tgt_key_padding_mask': 0, 'src_key_padding_mask': 0}),
                desc=f'no_batch_dim_batch_first_{batch_first}'
            ))
        
        # 创建 src 和 tgt 张量，并根据 batch_first 条件进行转置
        src, tgt = make_input((2, 3, 4)), make_input((2, 3, 4))
        if not batch_first:
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
        # 如果 key_padding_mask 不为 None，则扩展它以匹配维度
        if key_padding_mask is not None:
            src_key_padding_mask, tgt_key_padding_mask = (key_padding_mask.expand(2, 3),) * 2
        
        # 创建另一个 ModuleInput 对象并添加到 samples 列表中，每个对象包括 constructor_input 和 forward_input
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                num_encoder_layers=1, num_decoder_layers=1,
                                                dropout=0.0, batch_first=batch_first, norm_first=norm_first, bias=bias),
                forward_input=FunctionInput(
                    src, tgt, tgt_mask=tgt_mask, src_mask=src_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, src_key_padding_mask=src_key_padding_mask
                ),
            ))
    # 返回生成的样本列表
    return samples


# 定义一个函数 module_inputs_torch_nn_Embedding，接受多个参数，返回一个空的张量（long 类型）
def module_inputs_torch_nn_Embedding(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数 partial 创建一个局部函数 make_empty，用于创建空的长整型张量
    make_empty = partial(torch.empty, device=device, dtype=torch.long, requires_grad=False)
    return [
        # 返回一个包含两个 ModuleInput 对象的列表

        ModuleInput(
            # 第一个 ModuleInput 对象的构造函数参数
            constructor_input=FunctionInput(num_embeddings=4, embedding_dim=3),
            # 第一个 ModuleInput 对象的 forward 方法参数
            forward_input=FunctionInput(make_empty(2, 3).random_(4))
        ),

        ModuleInput(
            # 第二个 ModuleInput 对象的构造函数参数
            constructor_input=FunctionInput(num_embeddings=4, embedding_dim=3),
            # 第二个 ModuleInput 对象的 forward 方法参数，包含以下操作：
            # 1. 生成一个 1x512 的空 Tensor
            # 2. 用数字 4 随机填充这个 Tensor
            # 3. 将这个 Tensor 扩展为一个 7x512 的 Tensor
            forward_input=FunctionInput(make_empty(1, 512).random_(4).expand(7, 512)),
            # 第二个 ModuleInput 对象的描述信息
            desc='discontiguous'
        ),
    ]
# 定义一个函数，生成用于测试多头注意力模块的输入样本
def module_inputs_torch_nn_MultiheadAttention(module_info, device, dtype, requires_grad, training, **kwargs):
    # 定义一个偏函数，用于生成张量，指定设备、数据类型和梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 初始化一个空列表，用于存储输入样本
    samples = []
    # 定义布尔类型的取值范围
    bool_vals = (True, False)
    # 定义 key_padding_masks 和 attn_masks 的取值范围
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3, 3)))
    # 生成所有可能的组合，用于构造输入样本
    products = itertools.product(bool_vals, bool_vals, bool_vals, key_padding_masks, attn_masks)
    # 遍历每个组合
    for bias, add_bias_kv, add_zero_attn, key_padding_mask, attn_mask in products:
        # 添加一个 ModuleInput 对象到 samples 列表，用于测试多头注意力模块
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(embed_dim=3, num_heads=3, batch_first=True,
                                                bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn),
                forward_input=FunctionInput(make_input((3, 3)), make_input((3, 3)), make_input((3, 3)),
                                            key_padding_mask=key_padding_mask, attn_mask=attn_mask),
                reference_fn=no_batch_dim_reference_mha,
            )
        )
        # 添加另一个 ModuleInput 对象到 samples 列表，用于测试多头注意力模块（batch_first=False）
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(embed_dim=3, num_heads=3, batch_first=False,
                                                bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn),
                forward_input=FunctionInput(make_input((3, 3)), make_input((3, 3)), make_input((3, 3)),
                                            key_padding_mask=key_padding_mask, attn_mask=attn_mask),
                reference_fn=partial(no_batch_dim_reference_mha, batch_first=False),
            )
        )

    # 返回生成的样本列表
    return samples


# 定义一个函数，生成用于测试 RNN/GRU 单元的输入样本
def module_inputs_torch_nn_RNN_GRU_Cell(module_info, device, dtype, requires_grad, training, **kwargs):
    # 定义一个偏函数，用于生成张量，指定设备、数据类型和梯度需求
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 初始化一个包含两个 ModuleInput 对象的列表，用于测试 RNN/GRU 单元
    samples = [
        ModuleInput(
            constructor_input=FunctionInput(5, 10),
            forward_input=FunctionInput(make_input(5), make_input(10)),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput(5, 10, bias=True),
            forward_input=FunctionInput(make_input(5), make_input(10)),
            reference_fn=no_batch_dim_reference_fn,
        )
    ]

    # 检查 kwargs 中是否有 'is_rnn' 参数，若有则获取其值，默认为 False
    is_rnn = kwargs.get('is_rnn', False)
    # 如果是 RNN 模型
    if is_rnn:
        # RNN 模型也支持 `nonlinearity` 参数。
        # 默认是 `tanh`，这里检查是否设定为 `relu`
        # 创建一个 ModuleInput 对象，并添加到 samples 列表中
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(5, 10, bias=True, nonlinearity='relu'),
                # 通过 make_input 函数生成 forward_input 的输入
                forward_input=FunctionInput(make_input(5), make_input(10)),
                # 使用 no_batch_dim_reference_fn 作为 reference_fn
                reference_fn=no_batch_dim_reference_fn,
            )
        )

    # 返回构建好的 samples 列表
    return samples
def module_inputs_torch_nn_LSTMCell(module_info, device, dtype, requires_grad, training, **kwargs):
    # 当前所有样本用于验证不支持批次维度的情况。
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 定义一些样本，每个样本包含构造器输入、前向输入和参考函数
    samples = (
        ModuleInput(
            constructor_input=FunctionInput(5, 10),
            forward_input=FunctionInput(make_input(5), (make_input(10), make_input(10))),
            reference_fn=no_batch_dim_reference_lstmcell,
        ),
        ModuleInput(
            constructor_input=FunctionInput(5, 10, bias=True),
            forward_input=FunctionInput(make_input(5), (make_input(10), make_input(10))),
            reference_fn=no_batch_dim_reference_lstmcell,
        ),
    )

    return samples

def make_packed_sequence(inp, batch_sizes):
    required_grad = inp.requires_grad
    inp.requires_grad_(False)  # 用户无法访问inp，因此无法获取其梯度
    # 将输入序列打包成PackedSequence对象
    seq = pack_padded_sequence(inp, batch_sizes)
    seq.data.requires_grad_(required_grad)
    return seq


def module_inputs_torch_nn_RNN_GRU(module_info, device, dtype, requires_grad, training, with_packed_sequence=False, **kwargs):
    # 当前所有样本用于验证不支持批次维度的情况。
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    is_rnn = kwargs['is_rnn']
    nonlinearity = ('relu', 'tanh')
    bias = (False, True)
    batch_first = (False, True)
    bidirectional = (False, True)

    samples = []
    if is_rnn:
        # 使用product函数生成不同参数组合的笛卡尔积
        prod_gen = product(nonlinearity, bias, batch_first, bidirectional)
    else:
        prod_gen = product(bias, batch_first, bidirectional)
    for args in prod_gen:
        # 根据是否为 RNN 设置不同的参数解构
        if is_rnn:
            nl, b, b_f, bidir = args  # 如果是 RNN，解构参数为非线性激活函数、偏置、批处理优先、双向标志
        else:
            b, b_f, bidir = args  # 如果不是 RNN，解构参数为偏置、批处理优先、双向标志

        # 构造构造函数所需的参数字典和隐藏层参数字典
        cons_args = {'input_size': 2, 'hidden_size': 2, 'num_layers': 2,
                     'batch_first': b_f, 'bias': b, 'bidirectional': bidir}
        cons_args_hidden = {'input_size': 2, 'hidden_size': 3, 'num_layers': 2,
                            'batch_first': b_f, 'bias': b, 'bidirectional': bidir}

        # 如果是 RNN，设置非线性激活函数参数
        if is_rnn:
            cons_args['nonlinearity'] = nl
            cons_args_hidden['nonlinearity'] = nl

        # 向样本列表添加模块输入
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(**cons_args),  # 构造函数输入
                forward_input=FunctionInput(make_input((3, 2))),  # 前向输入
                reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f),  # 参考函数
            )
        )
        # 向样本列表添加隐藏层参数的模块输入
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(**cons_args_hidden),  # 构造函数输入
                forward_input=FunctionInput(make_input((3, 2)), make_input((4 if bidir else 2, 3))),  # 前向输入
                reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f),  # 参考函数
            )
        )

        # 如果需要使用 packed sequence，则添加相应的模块输入
        if with_packed_sequence:
            samples.append(
                ModuleInput(
                    constructor_input=FunctionInput(**cons_args),  # 构造函数输入
                    forward_input=FunctionInput(make_packed_sequence(make_input((5, 2, 2)), torch.tensor([5, 3]))),  # 前向输入
                    reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f),  # 参考函数
                )
            )
            samples.append(
                ModuleInput(
                    constructor_input=FunctionInput(**cons_args),  # 构造函数输入
                    forward_input=FunctionInput(make_packed_sequence(make_input((5, 5, 2)), torch.tensor([5, 3, 3, 2, 2]))),  # 前向输入
                    reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f),  # 参考函数
                )
            )

    return samples
# 定义一个函数，生成用于测试的模块输入样本，针对 torch.nn.LSTM 模块
def module_inputs_torch_nn_LSTM(module_info, device, dtype, requires_grad, training, **kwargs):
    # 定义一个偏函数，用于创建具有特定设备、数据类型和梯度要求的张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 定义不同的参数组合
    bias = (False, True)
    batch_first = (False, True)
    bidirectional = (False, True)
    proj_sizes = (0, 2)

    # 使用 itertools.product 生成所有参数组合的迭代器
    prod_gen = product(bias, batch_first, bidirectional, proj_sizes)

    # 初始化空列表，用于存储样本
    samples = []

    # 遍历所有参数组合
    for args in prod_gen:
        b, b_f, bidir, proj_size = args
        hidden_size = 3
        # 构造参数字典
        cons_args = {'input_size': 2, 'hidden_size': hidden_size, 'num_layers': 2, 'proj_size': proj_size,
                     'batch_first': b_f, 'bias': b, 'bidirectional': bidir}
        cons_args_hidden = {'input_size': 2, 'hidden_size': hidden_size, 'num_layers': 2, 'proj_size': proj_size,
                            'batch_first': b_f, 'bias': b, 'bidirectional': bidir}

        # 创建 ModuleInput 对象并添加到样本列表
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(**cons_args),
                forward_input=FunctionInput(make_input((2, 2))),
                reference_fn=partial(no_batch_dim_reference_lstm, batch_first=b_f),
            )
        )

        # 计算隐藏状态输出维度并生成隐藏状态张量
        h_out = proj_size if proj_size > 0 else hidden_size
        hx = (make_input((4 if bidir else 2, h_out)), make_input((4 if bidir else 2, hidden_size)))
        # 创建 ModuleInput 对象并添加到样本列表
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(**cons_args_hidden),
                forward_input=FunctionInput(make_input((3, 2)), hx),
                reference_fn=partial(no_batch_dim_reference_lstm, batch_first=b_f),
            )
        )

    # 返回生成的样本列表
    return samples


# 定义一个函数，生成用于测试的模块输入样本，针对 torch.nn.ReflectionPad1d 模块
def module_inputs_torch_nn_ReflectionPad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 定义一个偏函数，用于创建具有特定设备、数据类型和梯度要求的张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回两个 ModuleInput 对象的列表，用于测试 ReflectionPad1d
    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((2, 3))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2)),
            forward_input=FunctionInput(make_input((2, 3, 4))),
        ),
    ]


# 定义一个函数，生成用于测试的模块输入样本，针对 torch.nn.ReflectionPad2d 模块
def module_inputs_torch_nn_ReflectionPad2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 定义一个偏函数，用于创建具有特定设备、数据类型和梯度要求的张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回两个 ModuleInput 对象的列表，用于测试 ReflectionPad2d
    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((3, 4, 5))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4)),
            forward_input=FunctionInput(make_input((3, 4, 5, 6))),
        ),
    ]


# 定义一个函数，生成用于测试的模块输入样本，针对 torch.nn.ReflectionPad3d 模块
def module_inputs_torch_nn_ReflectionPad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 该函数尚未实现
    # 使用 partial 函数生成一个新的函数 make_input，固定了部分参数（device, dtype, requires_grad）
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    
    # 返回一个包含两个 ModuleInput 对象的列表
    return [
        # 第一个 ModuleInput 对象，构造函数输入为 FunctionInput(1)，前向输入为 make_input((2, 3, 4, 5))，参考函数为 no_batch_dim_reference_fn
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((2, 3, 4, 5))),
            reference_fn=no_batch_dim_reference_fn
        ),
        # 第二个 ModuleInput 对象，构造函数输入为 FunctionInput((1, 2, 1, 2, 1, 2))，前向输入为 make_input((3, 3, 3, 3, 3))
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 1, 2, 1, 2)),
            forward_input=FunctionInput(make_input((3, 3, 3, 3, 3))),
        ),
    ]
# 为 torch.nn.ReplicationPad1d 模块生成输入数据
def module_inputs_torch_nn_ReplicationPad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 部分函数 make_tensor 的部分参数已经预设为固定值
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回包含两个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput(1),  # 构造函数的输入是整数 1
            forward_input=FunctionInput(make_input((3, 4))),  # 前向传播的输入是形状为 (3, 4) 的张量
            reference_fn=no_batch_dim_reference_fn  # 参考函数为 no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2)),  # 构造函数的输入是元组 (1, 2)
            forward_input=FunctionInput(make_input((3, 4, 5))),  # 前向传播的输入是形状为 (3, 4, 5) 的张量
        ),
    ]

# 为 torch.nn.ReplicationPad2d 模块生成输入数据
def module_inputs_torch_nn_ReplicationPad2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回包含两个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput(1),  # 构造函数的输入是整数 1
            forward_input=FunctionInput(make_input((3, 4, 5))),  # 前向传播的输入是形状为 (3, 4, 5) 的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数为 no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4)),  # 构造函数的输入是元组 (1, 2, 3, 4)
            forward_input=FunctionInput(make_input((3, 4, 5, 6))),  # 前向传播的输入是形状为 (3, 4, 5, 6) 的张量
        ),
    ]

# 为 torch.nn.ReplicationPad3d 模块生成输入数据
def module_inputs_torch_nn_ReplicationPad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回包含两个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput(1),  # 构造函数的输入是整数 1
            forward_input=FunctionInput(make_input((3, 4, 5, 6))),  # 前向传播的输入是形状为 (3, 4, 5, 6) 的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数为 no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4, 5, 6)),  # 构造函数的输入是元组 (1, 2, 3, 4, 5, 6)
            forward_input=FunctionInput(make_input((3, 4, 5, 6, 7))),  # 前向传播的输入是形状为 (3, 4, 5, 6, 7) 的张量
        ),
    ]

# 为 torch.nn.ZeroPad1d 模块生成输入数据
def module_inputs_torch_nn_ZeroPad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回包含两个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput(1),  # 构造函数的输入是整数 1
            forward_input=FunctionInput(make_input((3, 4))),  # 前向传播的输入是形状为 (3, 4) 的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数为 no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2)),  # 构造函数的输入是元组 (1, 2)
            forward_input=FunctionInput(make_input((3, 4, 5))),  # 前向传播的输入是形状为 (3, 4, 5) 的张量
        ),
    ]

# 为 torch.nn.ZeroPad2d 模块生成输入数据
def module_inputs_torch_nn_ZeroPad2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回包含两个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput(1),  # 构造函数的输入是整数 1
            forward_input=FunctionInput(make_input((1, 2, 3))),  # 前向传播的输入是形状为 (1, 2, 3) 的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数为 no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4)),  # 构造函数的输入是元组 (1, 2, 3, 4)
            forward_input=FunctionInput(make_input((1, 2, 3, 4))),  # 前向传播的输入是形状为 (1, 2, 3, 4) 的张量
        ),
    ]

# 为 torch.nn.ZeroPad3d 模块生成输入数据
def module_inputs_torch_nn_ZeroPad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用偏函数 partial 创建一个 make_input 函数，固定了部分参数（device, dtype, requires_grad）
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个包含两个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput(1),  # 第一个 ModuleInput 的构造函数参数是 FunctionInput(1)
            forward_input=FunctionInput(make_input((3, 4, 5, 6))),  # 第一个 ModuleInput 的前向输入是 make_input((3, 4, 5, 6))
            reference_fn=no_batch_dim_reference_fn,  # 第一个 ModuleInput 的参考函数是 no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4, 5, 6)),  # 第二个 ModuleInput 的构造函数参数是 FunctionInput((1, 2, 3, 4, 5, 6))
            forward_input=FunctionInput(make_input((1, 2, 3, 4, 5))),  # 第二个 ModuleInput 的前向输入是 make_input((1, 2, 3, 4, 5))
        ),
    ]
# 定义一个函数，生成用于 Torch 模块输入的数据结构列表
def module_inputs_torch_nn_ConstantPad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用函数，用于生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个包含两个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput(1, 2),  # 构造器输入：1, 2
            forward_input=FunctionInput(make_input((3, 4))),  # 前向输入：生成一个形状为 (3, 4) 的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数：no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2), 3),  # 构造器输入：(1, 2), 3
            forward_input=FunctionInput(make_input((3, 4, 5))),  # 前向输入：生成一个形状为 (3, 4, 5) 的张量
        ),
    ]

# 定义一个函数，生成用于 Torch 模块输入的数据结构列表
def module_inputs_torch_nn_ConstantPad2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用函数，用于生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个包含两个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput(1, 3),  # 构造器输入：1, 3
            forward_input=FunctionInput(make_input((3, 4, 5))),  # 前向输入：生成一个形状为 (3, 4, 5) 的张量
            reference_fn=no_batch_dim_reference_fn  # 参考函数：no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4), 5),  # 构造器输入：(1, 2, 3, 4), 5
            forward_input=FunctionInput(make_input((1, 2, 3, 4))),  # 前向输入：生成一个形状为 (1, 2, 3, 4) 的张量
        ),
    ]

# 定义一个函数，生成用于 Torch 模块输入的数据结构列表
def module_inputs_torch_nn_ConstantPad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用函数，用于生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个包含两个 ModuleInput 对象的列表
    return [
        ModuleInput(
            constructor_input=FunctionInput(1, 3),  # 构造器输入：1, 3
            forward_input=FunctionInput(make_input((3, 4, 5, 6))),  # 前向输入：生成一个形状为 (3, 4, 5, 6) 的张量
            reference_fn=no_batch_dim_reference_fn,  # 参考函数：no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 3, 4, 5, 6), 7),  # 构造器输入：(1, 2, 3, 4, 5, 6), 7
            forward_input=FunctionInput(make_input((1, 2, 1, 2, 1))),  # 前向输入：生成一个形状为 (1, 2, 1, 2, 1) 的张量
        ),
    ]

# 定义一个函数，生成用于 Torch 模块输入的数据结构列表
def module_inputs_torch_nn_CircularPad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个部分应用函数，用于生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 定义一个环形填充的参考函数
    def padding1d_circular_ref(inp, pad):
        r""" input:
                [[[0., 1., 2.],
                  [3., 4., 5.]]]
                pad: (1, 2)
                output:
                    [[[2., 0., 1., 2., 0., 1.],
                      [5., 3., 4., 5., 3., 4.]]]
            """
        # 返回经过环形填充后的张量
        return torch.cat([inp[:, :, -pad[0]:], inp, inp[:, :, :pad[1]]], dim=2)
    # 返回一个包含多个 ModuleInput 对象的列表，每个对象描述了一个模块的输入情况
    return [
        # 第一个 ModuleInput 对象，描述了第一个模块的输入情况
        ModuleInput(
            # 该模块的构造函数输入，是一个 FunctionInput 对象，参数为 1
            constructor_input=FunctionInput(1),
            # 该模块前向传播函数的输入，是一个 FunctionInput 对象，生成一个形状为 (3, 4) 的输入数据
            forward_input=FunctionInput(make_input((3, 4))),
            # 参考函数，指定为 no_batch_dim_reference_fn
            reference_fn=no_batch_dim_reference_fn
        ),
        # 第二个 ModuleInput 对象，描述了第二个模块的输入情况
        ModuleInput(
            # 该模块的构造函数输入，是一个 FunctionInput 对象，参数为 (1, 2)
            constructor_input=FunctionInput((1, 2)),
            # 该模块前向传播函数的输入，是一个 FunctionInput 对象，生成一个形状为 (1, 2, 3) 的输入数据
            forward_input=FunctionInput(make_input((1, 2, 3))),
            # 参考函数，使用 lambda 表达式定义，根据输入参数调用 padding1d_circular_ref 函数
            reference_fn=lambda m, p, i: padding1d_circular_ref(i, m.padding),
        ),
        # 第三个 ModuleInput 对象，描述了第三个模块的输入情况
        ModuleInput(
            # 该模块的构造函数输入，是一个 FunctionInput 对象，参数为 (3, 1)
            constructor_input=FunctionInput((3, 1)),
            # 该模块前向传播函数的输入，是一个 FunctionInput 对象，生成一个形状为 (1, 2, 3) 的输入数据
            forward_input=FunctionInput(make_input((1, 2, 3))),
            # 参考函数，使用 lambda 表达式定义，根据输入参数调用 padding1d_circular_ref 函数
            reference_fn=lambda m, p, i: padding1d_circular_ref(i, m.padding),
        ),
        # 第四个 ModuleInput 对象，描述了第四个模块的输入情况
        ModuleInput(
            # 该模块的构造函数输入，是一个 FunctionInput 对象，参数为 (3, 3)
            constructor_input=FunctionInput((3, 3)),
            # 该模块前向传播函数的输入，是一个 FunctionInput 对象，生成一个形状为 (1, 2, 3) 的输入数据
            forward_input=FunctionInput(make_input((1, 2, 3))),
            # 参考函数，使用 lambda 表达式定义，根据输入参数调用 padding1d_circular_ref 函数
            reference_fn=lambda m, p, i: padding1d_circular_ref(i, m.padding),
        ),
    ]
# 定义一个函数，用于生成 Torch 张量，进行填充操作
def module_inputs_torch_nn_CircularPad2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 使用 functools.partial 函数固定参数，生成特定设备、数据类型和梯度属性的张量生成函数
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 定义一个二维环形填充函数，接受输入张量和填充元组
    def padding2d_circular_ref(inp, pad):
        r"""input:
                [[[[0., 1., 2],
                   [3., 4., 5.]]]]
                pad: (1, 2, 2, 1)
        output:
            [[[[2., 0., 1., 2., 0., 1.],
               [5., 3., 4., 5., 3., 4.],
               [2., 0., 1., 2., 0., 1.],
               [5., 3., 4., 5., 3., 4.],
               [2., 0., 1., 2., 0., 1.]]]]
        """
        # 在第二维度上进行环形填充操作
        inp = torch.cat([inp[:, :, -pad[2]:], inp, inp[:, :, :pad[3]]], dim=2)
        # 在第三维度上进行环形填充操作
        return torch.cat([inp[:, :, :, -pad[0]:], inp, inp[:, :, :, :pad[1]]], dim=3)

    # 返回一个列表，其中包含四个 ModuleInput 对象
    return [
        ModuleInput(
            constructor_input=FunctionInput(1),  # 构造函数的输入为标量 1
            forward_input=FunctionInput(make_input((3, 4, 5))),  # 前向传播函数的输入为形状为 (3, 4, 5) 的张量
            reference_fn=no_batch_dim_reference_fn,  # 引用函数指定为 no_batch_dim_reference_fn
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 2, 1)),  # 构造函数的输入为形状为 (1, 2, 2, 1) 的张量
            forward_input=FunctionInput(make_input((1, 1, 2, 3))),  # 前向传播函数的输入为形状为 (1, 1, 2, 3) 的张量
            reference_fn=lambda m, p, i: padding2d_circular_ref(i, m.padding),  # 引用函数使用 padding2d_circular_ref 函数
        ),
        ModuleInput(
            constructor_input=FunctionInput((2, 3, 2, 2)),  # 构造函数的输入为形状为 (2, 3, 2, 2) 的张量
            forward_input=FunctionInput(make_input((1, 1, 2, 3))),  # 前向传播函数的输入为形状为 (1, 1, 2, 3) 的张量
            reference_fn=lambda m, p, i: padding2d_circular_ref(i, m.padding),  # 引用函数使用 padding2d_circular_ref 函数
        ),
        ModuleInput(
            constructor_input=FunctionInput((3, 3, 3, 1)),  # 构造函数的输入为形状为 (3, 3, 3, 1) 的张量
            forward_input=FunctionInput(make_input((1, 1, 3, 3))),  # 前向传播函数的输入为形状为 (1, 1, 3, 3) 的张量
            reference_fn=lambda m, p, i: padding2d_circular_ref(i, m.padding),  # 引用函数使用 padding2d_circular_ref 函数
        ),
    ]

def module_inputs_torch_nn_CircularPad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 在这里继续实现 CircularPad3d 的函数定义
    def padding3d_circular_ref(inp, pad):
        r"""input:
                [[[[[ 0.,  1.,  2.],
                    [ 3.,  4.,  5.]],
                   [[ 6.,  7.,  8.],
                    [ 9., 10., 11.]]]]]
            pad: (1, 2, 2, 1, 1, 2)
            output: [[[[[ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.]],

                       [[ 2.,  0.,  1.,  2.,  0.,  1.],
                        [ 5.,  3.,  4.,  5.,  3.,  4.],
                        [ 2.,  0.,  1.,  2.,  0.,  1.],
                        [ 5.,  3.,  4.,  5.,  3.,  4.],
                        [ 2.,  0.,  1.,  2.,  0.,  1.]],

                       [[ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.]],

                       [[ 2.,  0.,  1.,  2.,  0.,  1.],
                        [ 5.,  3.,  4.,  5.,  3.,  4.],
                        [ 2.,  0.,  1.,  2.,  0.,  1.],
                        [ 5.,  3.,  4.,  5.,  3.,  4.],
                        [ 2.,  0.,  1.,  2.,  0.,  1.]],

                       [[ 8.,  6.,  7.,  8.,  ```
                        [ 6.,  7.],
                        [11.,  9.],
                        [10., 11.],
                        [ 9., 10.],
                        [ 8.,  6.]],

                       [[ 7.,  8.],
                        [ 6.,  7.],
                        [ 8.,  6.],
                        [ 7.,  8.],
                        [ 6.,  7.]],

                       [[ 2.,  0.],
                        [ 1.,  2.],
                        [ 0.,  1.],
                        [ 2.,  0.],
                        [ 1.,  2.]],

                       [[ 5.,  3.],
                        [ 4.,  5.],
                        [ 3.,  4.],
                        [ 5.,  3.],
                        [ 4.,  5.]],

                       [[ 2.,  0.],
                        [ 1.,  2.],
                        [ 0.,  1.],
                        [ 2.,  0.],
                        [ 1.,  2.]]]]]
        """
        # 在第三维度上进行循环填充，负向填充pad[4]个元素，正向填充pad[5]个元素
        inp = torch.cat([inp[:, :, -pad[4]:], inp, inp[:, :, :pad[5]]], dim=2)
        # 在第四维度上进行循环填充，负向填充pad[2]个元素，正向填充pad[3]个元素
        inp = torch.cat([inp[:, :, :, -pad[2]:], inp, inp[:, :, :, :pad[3]]], dim=3)
        # 在第五维度上进行循环填充，负向填充pad[0]个元素，正向填充pad[1]个元素
        return torch.cat([inp[:, :, :, :, -pad[0]:], inp, inp[:, :, :, :, :pad[1]]], dim=4)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1),
            forward_input=FunctionInput(make_input((3, 4, 5, 6))),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput((1, 2, 1, 2, 1, 2)),
            forward_input=FunctionInput(make_input((1, 1, 2, 2, 3))),
            reference_fn=lambda m, p, i: padding3d_circular_ref(i, m.padding)
        ),
        ModuleInput(
            constructor_input=FunctionInput((3, 2, 2, 1, 1, 2)),
            forward_input=FunctionInput(make_input((1, 1, 2, 2, 3))),
            reference_fn=lambda m, p, i: padding3d_circular_ref(i, m.padding)
        ),
        ModuleInput(
            constructor_input=FunctionInput((3, 3, 2, 1, 2, 2)),
            forward_input=FunctionInput(make_input((1, 1, 2, 2, 3))),
            reference_fn=lambda m, p, i: padding3d_circular_ref(i, m.padding)
        ),
    ]


注释：
# 用于装饰 RNN、GRU 和 LSTM 模块信息的装饰器列表
rnn_gru_lstm_module_info_decorators = (
    # 对于 cuDNN 和 MIOpen 的共享问题，定义装饰信息对象
    DecorateInfo(
        unittest.expectedFailure, "TestModule", "test_grad",
        active_if=(TEST_CUDNN and not TEST_WITH_ROCM), device_type='cuda'
    ),
    # 对于 cuDNN 的双向梯度问题，定义装饰信息对象
    DecorateInfo(
        unittest.expectedFailure, "TestModule", "test_gradgrad",
        active_if=(TEST_CUDNN and not TEST_WITH_ROCM), device_type='cuda'
    ),
    # 指出 cuDNN GRU 不接受非连续的 hx 张量
    DecorateInfo(
        unittest.expectedFailure, "TestModule", "test_non_contiguous_tensors",
        active_if=(TEST_CUDNN and not TEST_WITH_ROCM), device_type='cuda'
    ),
    # 指出 MIOPEN GRU 不接受非连续的 hx 张量（仅针对 float，由于测试在 ROCM 环境下运行）
    DecorateInfo(
        unittest.expectedFailure, "TestModule", "test_non_contiguous_tensors",
        active_if=(TEST_CUDNN and TEST_WITH_ROCM), dtypes=(torch.float,), device_type='cuda'
    ),
    # 跳过 CUDA 版本为 11.7 的测试，适用于 CUDA 设备的扩展权重模块测试
    DecorateInfo(
        skipCUDAVersionIn([(11, 7)]), "TestExpandedWeightModule", "test_module",
        device_type='cuda'
    ),
    # 跳过 CUDA 版本为 11.7 的测试，适用于 CUDA 设备的 RNN 分解模块测试
    DecorateInfo(
        skipCUDAVersionIn([(11, 7)]), "TestDecomp", "test_rnn_decomp_module",
        device_type='cuda'
    )
)

# 模块错误输入函数的开始

def module_error_inputs_torch_nn_RNN_GRU_Cell(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建输入张量的辅助函数，根据给定参数生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 定义一个列表 `samples`，其中包含了多个 `ErrorModuleInput` 对象
    samples = [
        # 创建第一个 `ErrorModuleInput` 对象，传入一个 `ModuleInput` 对象和错误相关的参数
        ErrorModuleInput(
            # 创建 `ModuleInput` 对象，传入构造函数和前向输入的参数
            ModuleInput(
                constructor_input=FunctionInput(10, 20),  # 构造函数的输入参数
                forward_input=FunctionInput(make_input(3, 11), make_input(3, 20)),  # 前向输入的参数
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,  # 指定错误类型为前向错误
            error_type=RuntimeError,  # 指定错误的类型为 RuntimeError
            error_regex="input has inconsistent input_size: got 11 expected 10"  # 错误信息的正则表达式
        ),
        # 创建第二个 `ErrorModuleInput` 对象，传入一个 `ModuleInput` 对象和错误相关的参数，依此类推
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 10), make_input(3, 21)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="hidden0 has inconsistent hidden_size: got 21, expected 20"
        ),
        # 创建第三个 `ErrorModuleInput` 对象
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 10), make_input(5, 20)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="Input batch size 3 doesn't match hidden0 batch size 5"
        ),
        # 创建第四个 `ErrorModuleInput` 对象
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 10), make_input(3, 1, 1, 20)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex="Expected hidden to be 1D or 2D, got 4D instead"
        ),
        # 创建第五个 `ErrorModuleInput` 对象
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20, 'relu'),
                forward_input=FunctionInput(make_input(3, 10), make_input(3, 21)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="hidden0 has inconsistent hidden_size: got 21, expected 20"
        ),
        # 创建第六个 `ErrorModuleInput` 对象
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20, 'tanh'),
                forward_input=FunctionInput(make_input(3, 10), make_input(3, 21)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="hidden0 has inconsistent hidden_size: got 21, expected 20"
        ),
    ]
    
    # 返回构建好的 `samples` 列表
    return samples
# 定义一个函数，生成用于测试 Torch 的 LSTMCell 模块的错误输入样本列表
def module_error_inputs_torch_nn_LSTMCell(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建一个偏函数 make_input，用于生成张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 定义一组错误输入样本列表
    samples = [
        ErrorModuleInput(
            # 定义模块输入，包括构造器输入和前向输入
            ModuleInput(
                constructor_input=FunctionInput(10, 20),  # 构造器输入为 (10, 20)
                forward_input=FunctionInput(make_input(3, 11), (make_input(3, 20), make_input(3, 20))),  # 前向输入包括一个 (3, 11) 和两个 (3, 20) 的张量
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,  # 错误类型为前向计算错误
            error_type=RuntimeError,  # 错误的具体类型为 RuntimeError
            error_regex="input has inconsistent input_size: got 11 expected 10"  # 错误信息正则表达式描述了输入大小不一致的问题
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 10), (make_input(3, 21), make_input(3, 21))),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="hidden0 has inconsistent hidden_size: got 21, expected 20"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 10), (make_input(5, 20), make_input(5, 20))),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=RuntimeError,
            error_regex="Input batch size 3 doesn't match hidden0 batch size 5"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, 20),
                forward_input=FunctionInput(make_input(3, 10), (make_input(3, 1, 1, 20), make_input(3, 1, 1, 20))),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex="Expected hx\\[0\\] to be 1D or 2D, got 4D instead"
        ),
    ]
    return samples  # 返回错误输入样本列表


# 定义一个函数，生成用于测试 Torch 的 RNN 和 GRU 模块的错误输入样本列表
def module_error_inputs_torch_nn_RNN_GRU(module_info, device, dtype, requires_grad, training, **kwargs):
    # 定义一组错误输入样本列表
    samples = [
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(10, 0, 1)),  # 构造器输入为 (10, 0, 1)
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,  # 错误类型为构造错误
            error_type=ValueError,  # 错误的具体类型为 ValueError
            error_regex="hidden_size must be greater than zero"  # 错误信息描述了隐藏单元大小必须大于零的问题
        ),
        ErrorModuleInput(
            ModuleInput(constructor_input=FunctionInput(10, 10, 0)),  # 构造器输入为 (10, 10, 0)
            error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
            error_type=ValueError,
            error_regex="num_layers must be greater than zero"  # 错误信息描述了层数必须大于零的问题
        ),
    ]
    return samples  # 返回错误输入样本列表


# 定义一个函数，生成用于测试 Torch 的 Pad1d 模块的错误输入样本列表
def module_error_inputs_torch_nn_Pad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    # 获取是否为常数填充的参数
    is_constant = kwargs.get('is_constant', False)
    # 返回一个包含单个 ErrorModuleInput 对象的列表
    return [
        # 创建一个 ErrorModuleInput 对象，并传入以下参数：
        ErrorModuleInput(
            # 根据条件选择不同的构造函数输入：
            ModuleInput(
                constructor_input=FunctionInput(1, 3) if is_constant else FunctionInput(3),
                # 设置前向输入为一个特定的函数输入对象
                forward_input=FunctionInput(make_input((2, 3, 4, 5))),
            ),
            # 指定错误类型为 FORWARD_ERROR
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            # 指定错误的类型为 ValueError
            error_type=ValueError,
            # 指定错误的正则表达式消息
            error_regex=r"expected 2D or 3D input \(got 4D input\)",
        ),
    ]
def module_error_inputs_torch_nn_Pad2d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建部分应用函数make_input，用于后续创建张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 获取关键字参数kwargs中的'is_constant'，默认为False
    is_constant = kwargs.get('is_constant', False)

    # 返回一个包含ErrorModuleInput对象的列表，每个对象描述一个错误输入情况
    return [
        ErrorModuleInput(
            # 定义ModuleInput对象，描述模块输入的构造方式和前向传播时的输入
            ModuleInput(
                # 如果is_constant为True，则使用1, 3作为构造输入；否则使用3
                constructor_input=FunctionInput(1, 3) if is_constant else FunctionInput(3),
                # 调用make_input函数创建一个形状为(2, 3)的张量作为前向输入
                forward_input=FunctionInput(make_input((2, 3))),
            ),
            # 指定错误类型为前向传播错误
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            # 指定错误类型为ValueError
            error_type=ValueError,
            # 指定错误信息的正则表达式，描述期望3D或4D输入但得到了2D输入的情况
            error_regex=r"expected 3D or 4D input \(got 2D input\)",
        ),
    ]

def module_error_inputs_torch_nn_Pad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    # 创建部分应用函数make_input，用于后续创建张量
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # 获取关键字参数kwargs中的'is_constant'，默认为False
    is_constant = kwargs.get('is_constant', False)

    # 返回一个包含ErrorModuleInput对象的列表，每个对象描述一个错误输入情况
    return [
        ErrorModuleInput(
            # 定义ModuleInput对象，描述模块输入的构造方式和前向传播时的输入
            ModuleInput(
                # 如果is_constant为True，则使用1, 3作为构造输入；否则使用3
                constructor_input=FunctionInput(1, 3) if is_constant else FunctionInput(3),
                # 调用make_input函数创建一个形状为(2, 3)的张量作为前向输入
                forward_input=FunctionInput(make_input((2, 3))),
            ),
            # 指定错误类型为前向传播错误
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            # 指定错误类型为ValueError
            error_type=ValueError,
            # 指定错误信息的正则表达式，描述期望4D或5D输入但得到了2D输入的情况
            error_regex=r"expected 4D or 5D input \(got 2D input\)",
        ),
    ]

# 以字母顺序排列的ModuleInfo条目数据库
module_db: List[ModuleInfo] = [
    ModuleInfo(torch.nn.AdaptiveAvgPool1d,
               # 指定模块输入函数为module_inputs_torch_nn_AdaptiveAvgPool1d
               module_inputs_func=module_inputs_torch_nn_AdaptiveAvgPool1d,
               # 跳过指定的装饰信息（在MPS后端上如果输入/输出大小不可被整除时会失败）
               skips=(
                   DecorateInfo(skipMPS),
               )
               ),
    ModuleInfo(torch.nn.AdaptiveAvgPool2d,
               # 指定梯度检查的非确定性容差为GRADCHECK_NONDET_TOL
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 指定模块输入函数为module_inputs_torch_nn_AdaptiveAvgPool2d
               module_inputs_func=module_inputs_torch_nn_AdaptiveAvgPool2d,
               # 跳过指定的装饰信息列表
               skips=(
                   # 如果在MPS后端上输入/输出大小不可被整除时会失败
                   DecorateInfo(skipMPS),
                   # 如果输出大小为1x1，在反向检查时会失败
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                   ),
               )
               ),
    ModuleInfo(torch.nn.AdaptiveAvgPool3d,
               # 指定梯度检查的非确定性容差为GRADCHECK_NONDET_TOL
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 指定模块输入函数为module_inputs_torch_nn_AdaptiveAvgPool3d
               module_inputs_func=module_inputs_torch_nn_AdaptiveAvgPool3d,
               # 跳过指定的装饰信息（在MPS后端上不支持）
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   DecorateInfo(skipMPS),
               )
               ),
    ModuleInfo(torch.nn.AdaptiveMaxPool1d,
               # 指定模块输入函数为module_inputs_torch_nn_AdaptiveMaxPool1d
               module_inputs_func=module_inputs_torch_nn_AdaptiveMaxPool1d,
               ),
]
    ModuleInfo(torch.nn.AdaptiveMaxPool2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_inputs_func=module_inputs_torch_nn_AdaptiveMaxPool2d,
               ),


    # 创建 ModuleInfo 对象，配置 AdaptiveMaxPool2d 模块的信息
    ModuleInfo(torch.nn.AdaptiveMaxPool2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_inputs_func=module_inputs_torch_nn_AdaptiveMaxPool2d,
               ),



    ModuleInfo(torch.nn.AdaptiveMaxPool3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_inputs_func=module_inputs_torch_nn_AdaptiveMaxPool3d,
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # not supported on MPS backend
                   DecorateInfo(skipMPS),)
               ),


    # 创建 ModuleInfo 对象，配置 AdaptiveMaxPool3d 模块的信息
    ModuleInfo(torch.nn.AdaptiveMaxPool3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_inputs_func=module_inputs_torch_nn_AdaptiveMaxPool3d,
               skips=(
                   # 标记为跳过的测试：不支持在 MPS 后端上运行
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 标记为跳过的测试：不支持在 MPS 后端上运行
                   DecorateInfo(skipMPS),)
               ),



    ModuleInfo(torch.nn.AvgPool1d,
               module_inputs_func=module_inputs_torch_nn_AvgPool1d,
               ),


    # 创建 ModuleInfo 对象，配置 AvgPool1d 模块的信息
    ModuleInfo(torch.nn.AvgPool1d,
               module_inputs_func=module_inputs_torch_nn_AvgPool1d,
               ),



    ModuleInfo(torch.nn.AvgPool2d,
               module_inputs_func=module_inputs_torch_nn_AvgPool2d,
               skips=(
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                       device_type='cuda',
                   ),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float16]),),
               ),


    # 创建 ModuleInfo 对象，配置 AvgPool2d 模块的信息
    ModuleInfo(torch.nn.AvgPool2d,
               module_inputs_func=module_inputs_torch_nn_AvgPool2d,
               skips=(
                   # 标记为预期失败的测试：在 CUDA 上，channels_last 反向传播与 channels_first 反向传播之间的差异过大
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                       device_type='cuda',
                   ),
                   # 标记为跳过的测试：不支持在 MPS 后端上运行，特定数据类型为 torch.float16
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float16]),),
               ),



    ModuleInfo(torch.nn.AvgPool3d,
               module_inputs_func=module_inputs_torch_nn_AvgPool3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   DecorateInfo(skipMPS),)
               ),


    # 创建 ModuleInfo 对象，配置 AvgPool3d 模块的信息
    ModuleInfo(torch.nn.AvgPool3d,
               module_inputs_func=module_inputs_torch_nn_AvgPool3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   # 标记为跳过的测试：不支持 channels_last 格式，因为不接受 4D 输入
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 标记为跳过的测试：不支持在 MPS 后端上运行
                   DecorateInfo(skipMPS),)
               ),
    # 定义 ModuleInfo 对象，描述 torch.nn.BatchNorm1d 模块
    ModuleInfo(torch.nn.BatchNorm1d,
               train_and_eval_differ=True,  # 指示训练和评估模式的差异
               module_inputs_func=module_inputs_torch_nn_BatchNorm1d,  # 设置模块输入的功能
               skips=(
                   # 跳过测试，因为在 MPS 后端上失败，问题正在调查中
                   # 参见 https://github.com/pytorch/pytorch/issues/100914
                   DecorateInfo(skipMPS),
                   # 跟踪测试失败，而不是在 test_aotdispatch.py 列表中，因为评估模式通过了
                   # RuntimeError: tried to get Double out of SymInt
                   DecorateInfo(
                       unittest.expectedFailure, 'TestEagerFusionModuleInfo',
                       'test_aot_autograd_symbolic_module_exhaustive',
                       active_if=operator.itemgetter('training')
                   ),
                   # 报告测试失败，错误为 torch._subclasses.fake_tensor.DataDependentOutputException: aten._local_scalar_dense.default
                   DecorateInfo(
                       unittest.expectedFailure, 'TestEagerFusionModuleInfo',
                       'test_aot_autograd_module_exhaustive',
                       active_if=operator.itemgetter('training')
                   ))
               ),
    # 定义 ModuleInfo 对象，描述 torch.nn.BatchNorm2d 模块
    ModuleInfo(torch.nn.BatchNorm2d,
               train_and_eval_differ=True,  # 指示训练和评估模式的差异
               module_inputs_func=module_inputs_torch_nn_BatchNorm2d,  # 设置模块输入的功能
               skips=(
                   # 跳过测试，因为在 MPS 后端上失败，问题正在调查中
                   # 参见 https://github.com/pytorch/pytorch/issues/100914
                   DecorateInfo(skipMPS),
                   # 跟踪测试失败，而不是在 test_aotdispatch.py 列表中，因为评估模式通过了
                   # RuntimeError: tried to get Double out of SymInt
                   DecorateInfo(
                       unittest.expectedFailure, 'TestEagerFusionModuleInfo',
                       'test_aot_autograd_symbolic_module_exhaustive',
                       active_if=operator.itemgetter('training')
                   ),
                   # 报告测试失败，错误为 torch._subclasses.fake_tensor.DataDependentOutputException: aten._local_scalar_dense.default
                   DecorateInfo(
                       unittest.expectedFailure, 'TestEagerFusionModuleInfo',
                       'test_aot_autograd_module_exhaustive',
                       active_if=operator.itemgetter('training')
                   ),)
               ),
    # 定义 ModuleInfo 对象，包含 torch.nn.BatchNorm3d 模块信息
    ModuleInfo(torch.nn.BatchNorm3d,
               # 设置 train_and_eval_differ 参数为 True
               train_and_eval_differ=True,
               # 指定 module_inputs_func 为 module_inputs_torch_nn_BatchNorm3d 函数
               module_inputs_func=module_inputs_torch_nn_BatchNorm3d,
               # 定义 skips 元组，包含多个 DecorateInfo 对象，用于跳过测试
               skips=(
                   # 跳过在 MPS 后端不支持的测试
                   DecorateInfo(skipMPS),
                   # 在测试 test_aotdispatch.py 中，遇到 RuntimeError: tried to get Double out of SymInt 错误时的跳过信息
                   DecorateInfo(
                       unittest.expectedFailure, 'TestEagerFusionModuleInfo',
                       'test_aot_autograd_symbolic_module_exhaustive',
                       active_if=operator.itemgetter('training')
                   ),
                   # 在测试 test_aot_autograd_module_exhaustive 中，针对 'training' 情况下的跳过信息
                   DecorateInfo(
                       unittest.expectedFailure, 'TestEagerFusionModuleInfo',
                       'test_aot_autograd_module_exhaustive',
                       active_if=operator.itemgetter('training')
                   ),
               ),
               ),

    # 定义 ModuleInfo 对象，包含 torch.nn.CELU 模块信息
    ModuleInfo(torch.nn.CELU,
               # 指定 module_inputs_func 为 module_inputs_torch_nn_CELU 函数
               module_inputs_func=module_inputs_torch_nn_CELU,
               # 对于 'mps' 设备和 torch.float16 类型的情况，预期的测试失败信息
               skips=(
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_check_inplace',
                                device_type='mps', dtypes=[torch.float16]),
               ),
               ),

    # 定义 ModuleInfo 对象，包含 torch.nn.Conv1d 模块信息
    ModuleInfo(torch.nn.Conv1d,
               # 指定 module_inputs_func 为 module_inputs_torch_nn_ConvNd 函数，其中 N=1，lazy=False
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=1, lazy=False),
               # 设置 gradcheck_nondet_tol 参数为 GRADCHECK_NONDET_TOL 的值
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 设置 module_memformat_affects_out 参数为 True
               module_memformat_affects_out=True,
               # 定义 skips 元组，包含多个 DecorateInfo 对象，用于跳过测试
               skips=(
                   # 对于 CUDA，需要 cudnn >= 7603 支持 channels_last，否则跳过测试
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # 在 ROCM 平台下，float32 类型存在问题 #70125，跳过测试
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # 在 MPS 平台下，针对 'mps' 设备和 torch.float16 类型的情况，跳过测试
                   DecorateInfo(skipIfMps, "TestModule", "test_memory_format",
                                device_type='mps', dtypes=[torch.float16]),
                   DecorateInfo(skipIfMps, "TestModule", "test_non_contiguous_tensors",
                                device_type='mps', dtypes=[torch.float16]),
               ),
               # 定义 decorators 元组，包含 DecorateInfo 对象，设置精度修正
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    # 定义 ModuleInfo 对象，用于存储 torch.nn.Conv2d 模块相关信息及测试配置
    ModuleInfo(torch.nn.Conv2d,
               # 定义模块输入函数，使用 torch.nn.ConvNd 模块的部分参数 N=2，lazy=False
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=2, lazy=False),
               # 梯度检查时的非确定性容差值
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 内存格式是否影响输出
               module_memformat_affects_out=True,
               # 跳过的测试用例列表
               skips=(
                   # 当 CUDA 版本小于 7603 时，跳过 channels_last 的支持测试
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # 在 ROCM 上由于 float32 问题 #70125 跳过测试
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # 对于 CUDA 设备和 float64 类型，跳过该测试，详细见 issue #80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format",
                                device_type='cuda', dtypes=[torch.float64]),
                   # 在 MPS 后端上，由于 channels last 测试失败，跳过该测试
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format",
                                device_type='mps', dtypes=[torch.float32]),
                   # 在 MPS 上，由于数据类型为 torch.float16，跳过测试，详细见 #119108
                   DecorateInfo(skipIfMps, "TestModule", "test_memory_format",
                                device_type='mps', dtypes=[torch.float16]),
                   # 在 MPS 上，由于数据类型为 torch.float16，跳过测试
                   DecorateInfo(skipIfMps, "TestModule", "test_non_contiguous_tensors",
                                device_type='mps', dtypes=[torch.float16]),
               ),
               # 装饰器列表，用于覆盖测试精度
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.Conv3d,
               # 定义模块输入函数，部分应用于 torch.nn.ConvNd 模块的输入处理，设置 N=3，非延迟处理
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=3, lazy=False),
               # 梯度检查非确定性容差设置为 GRADCHECK_NONDET_TOL
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 模块内存格式影响输出设置为 True
               module_memformat_affects_out=True,
               # 跳过条件元组包含以下内容：
               skips=(
                   # 当 CUDA 版本低于 8005 时跳过，要求支持 channels_last
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=8005), 'TestModule', 'test_memory_format'),
                   # 在 ROCM 上对于 float32 存在问题 #70125，跳过
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # MPS 后端不支持 Conv3d，跳过
                   DecorateInfo(skipMPS),
                   # 此项之前错误地被跳过，需要进一步调查，参见 issue #80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),
               ),
               # 修饰器包含以下内容：
               decorators=(
                   # 设置精度覆盖，针对 torch.float32
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.ConvTranspose1d,
               # 定义模块输入函数，部分应用于 torch.nn.ConvNd 模块的输入处理，设置 N=1，非延迟处理，转置为 True
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=1, lazy=False, transposed=True),
               # 梯度检查非确定性容差设置为 GRADCHECK_NONDET_TOL
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 模块内存格式影响输出设置为 True
               module_memformat_affects_out=True,
               # 数据类型包括浮点数和复数类型以及 torch.chalf
               dtypes=floating_and_complex_types_and(torch.chalf),
               # 跳过条件元组包含以下内容：
               skips=(
                   # 当 CUDA 版本低于 7603 时跳过，要求支持 channels_last
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # 在 ROCM 上对于 float32 存在问题 #70125，跳过
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # 在 CPU 上未实现 chalf 类型，跳过
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_cpu_gpu_parity',
                                dtypes=(torch.chalf,), device_type='cuda'),
                   # 参见 #119108：MPSNDArrayConvolutionA14.mm:3976: failed assertion `destination datatype must be fp32'
                   # 由于致命的 Python 错误而无法使用 xfail
                   DecorateInfo(skipIfMps, "TestModule", "test_memory_format",
                                device_type='mps', dtypes=[torch.float16]),
                   DecorateInfo(skipIfMps, "TestModule", "test_non_contiguous_tensors",
                                device_type='mps', dtypes=[torch.float16]),),
               # 修饰器包含以下内容：
               decorators=(
                   # 设置精度覆盖，针对 torch.float32
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
                   # 设置精度覆盖，针对 torch.chalf
                   DecorateInfo(precisionOverride({torch.chalf: 5e-03}), 'TestModule', 'test_memory_format'),
               )),
    # 定义 ModuleInfo 对象，包含 torch.nn.ConvTranspose2d 模块及其相关配置信息
    ModuleInfo(torch.nn.ConvTranspose2d,
               # 指定模块输入函数为 module_inputs_torch_nn_ConvNd 的部分应用，设定参数 N=2, lazy=False, transposed=True
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=2, lazy=False, transposed=True),
               # 梯度检查非确定性容忍度设为 GRADCHECK_NONDET_TOL
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 内存格式影响输出结果
               module_memformat_affects_out=True,
               # 指定浮点数及复数数据类型以及 torch.chalf
               dtypes=floating_and_complex_types_and(torch.chalf),
               # 跳过特定测试条件的元组
               skips=(
                   # 当 CUDA 版本低于 7603 时，跳过 channels_last 支持测试
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # 在 ROCM 上对于 float32 存在问题，跳过相关测试
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # 在反向检查时因为 ViewAsRealBackward 对梯度进行连续性处理，跳过相关测试
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_memory_format',
                                dtypes=(torch.complex32, torch.complex64, torch.complex128)),
                   # CUDA 设备上存在问题，需要调查
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='cuda',
                                dtypes=[torch.float64, torch.complex128]),
                   # 在 MPS 后端上 channels last 测试失败
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format",
                                device_type='mps', dtypes=[torch.float32]),
                   # 在 CPU 上对于 chalf 类型尚未实现
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_cpu_gpu_parity',
                                dtypes=(torch.chalf,), device_type='cuda'),
                   # MPSNDArrayConvolutionA14.mm:3976 错误，跳过相关测试
                   DecorateInfo(skipIfMps, "TestModule", "test_memory_format",
                                device_type='mps', dtypes=[torch.float16]),
                   DecorateInfo(skipIfMps, "TestModule", "test_non_contiguous_tensors",
                                device_type='mps', dtypes=[torch.float16]),
               ),
               # 修饰器列表，覆盖特定精度要求
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
                   DecorateInfo(precisionOverride({torch.chalf: 5e-03}), 'TestModule', 'test_memory_format'),
               )),
    # 定义 ModuleInfo 实例，用于 torch.nn.ConvTranspose3d 模块
    ModuleInfo(torch.nn.ConvTranspose3d,
               # 设置模块输入函数为特定的模块输入处理函数，传入部分参数
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=3, lazy=False, transposed=True),
               # 指定支持的数据类型包括浮点数和复数类型以及 torch.chalf
               dtypes=floating_and_complex_types_and(torch.chalf),
               # 梯度检查时的非确定性容忍度设定为预定义的常量 GRADCHECK_NONDET_TOL
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 模块的内存格式是否影响输出结果
               module_memformat_affects_out=True,
               # 跳过的测试用例集合，包括多个 DecorateInfo 对象
               skips=(
                   # 如果 CUDA 版本低于 8005，则跳过此测试（对 channels_last 的支持）
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=8005), 'TestModule', 'test_memory_format'),
                   # 在 ROCM 平台上，对于 float32 存在问题，因此跳过此测试
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # 不支持 MPS 后端的 ConvTranspose3d 测试，因此跳过
                   DecorateInfo(skipMPS),
                   # 此问题之前被错误跳过，需要进一步调查
                   # 参见 https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),
                   # 仅在 ROCm 平台上失败的测试用例集合
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='cuda',
                                dtypes=[torch.complex32, torch.complex64], active_if=TEST_WITH_ROCM),
                   # 在 CPU 上不支持 chalf 类型的测试用例，因此跳过
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_cpu_gpu_parity',
                                dtypes=(torch.chalf,), device_type='cuda'),
               ),
               # 修饰器集合，用于精度覆盖测试
               decorators=(
                   # 覆盖 float32 精度至 1e-04
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
                   # 覆盖 complex64 精度至 1e-04
                   DecorateInfo(precisionOverride({torch.complex64: 1e-04}), 'TestModule', 'test_cpu_gpu_parity'),
                   # 覆盖 chalf 精度至 5e-03
                   DecorateInfo(precisionOverride({torch.chalf: 5e-03}), 'TestModule', 'test_memory_format'),
               )),
    # 定义 ModuleInfo 实例，用于 torch.nn.CosineEmbeddingLoss 模块
    ModuleInfo(torch.nn.CosineEmbeddingLoss,
               # 设置模块输入函数为特定的模块输入处理函数
               module_inputs_func=module_inputs_torch_nn_CosineEmbeddingLoss,
               # 跳过的测试用例集合，仅包含一个 DecorateInfo 对象
               skips=(
                   # 不支持 loss functions 的 channels_last 格式
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 定义 ModuleInfo 实例，用于 torch.nn.ELU 模块
    ModuleInfo(torch.nn.ELU,
               # 设置模块输入函数为特定的模块输入处理函数
               module_inputs_func=module_inputs_torch_nn_ELU,
               # 跳过的测试用例集合，仅包含一个 DecorateInfo 对象
               skips=(
                   # 对于 MPS 设备类型和 float16 数据类型的测试用例，预期失败
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_check_inplace',
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    # 创建 ModuleInfo 对象，用于 torch.nn.FractionalMaxPool2d 模块，指定输入函数和梯度检查的非确定性容差
    ModuleInfo(torch.nn.FractionalMaxPool2d,
               module_inputs_func=module_inputs_torch_nn_FractionalMaxPool2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   # MPS 后端不支持该功能，因此跳过测试
                   DecorateInfo(skipMPS),
                   # 装饰信息表明该测试被跳过，用于 'TestModule' 模块的 'test_memory_format' 测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 创建 ModuleInfo 对象，用于 torch.nn.FractionalMaxPool3d 模块，指定输入函数和梯度检查的非确定性容差
    ModuleInfo(torch.nn.FractionalMaxPool3d,
               module_inputs_func=module_inputs_torch_nn_FractionalMaxPool3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   # MPS 后端不支持该功能，因此跳过测试
                   DecorateInfo(skipMPS),
                   # 装饰信息表明该测试被跳过，用于 'TestModule' 模块的 'test_memory_format' 测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 创建 ModuleInfo 对象，用于 torch.nn.L1Loss 模块，指定输入函数
               没有通道最后的支持用于损失函数
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 创建 ModuleInfo 对象，用于 torch.nn.SmoothL1Loss 模块，指定输入函数
               没有通道最后的支持用于损失函数
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 查看问题 #119108：输入类型 'tensor<f32>' 和 'tensor<15x10xf16>' 不兼容广播
                   DecorateInfo(skipIfMps, 'TestModule', 'test_non_contiguous_tensors', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.LazyConv1d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=1, lazy=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # channels_last support on cuda requires cudnn >= 7603
                   # 如果使用 CUDA，需要 cudnn 版本 >= 7603 才支持 channels_last
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   
                   # Failure on ROCM for float32 issue #70125
                   # ROCM 环境下，对于 float32 存在问题 #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # Lazy 模块目前在 meta 设备上与 ModuleInfo 测试不兼容
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
                   
                   # See #119108: MPSNDArrayConvolutionA14.mm:3976: failed assertion `destination datatype must be fp32'
                   # xfail does not work due to Fatal Python error: Aborted
                   # xfail 由于严重错误导致无法使用
                   DecorateInfo(skipIfMps, "TestModule", "test_memory_format",
                                device_type='mps', dtypes=[torch.float16]),
                   
                   # MPS 环境下的另一个跳过条件，用于 test_non_contiguous_tensors 测试
                   # MPS environment 中另一个跳过条件，用于 test_non_contiguous_tensors 测试
                   DecorateInfo(skipIfMps, "TestModule", "test_non_contiguous_tensors",
                                device_type='mps', dtypes=[torch.float16]),
               ),
               decorators=(
                   # Override precision for float32 to 1e-04 for test_memory_format in TestModule
                   # 为 TestModule 中的 test_memory_format 测试，将 float32 的精度覆盖为 1e-04
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    # 创建 ModuleInfo 对象，指定 LazyConv2d 模块，同时配置其它参数和修饰器
    ModuleInfo(torch.nn.LazyConv2d,
               # 配置模块输入函数为 torch_nn_ConvNd 的部分应用，设置 N=2，并启用 lazy 模式
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=2, lazy=True),
               # 梯度检查非确定性容差设为预定义的值
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 模块内存格式影响输出结果
               module_memformat_affects_out=True,
               # 跳过以下测试用例
               skips=(
                   # 在 CUDA 上要求 cudnn 版本 >= 7603 才支持 channels_last
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # ROCM 平台 float32 的问题，issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # Lazy 模块目前与元设备上的 ModuleInfo 测试不兼容，参见 issue #70505
                   DecorateInfo(skipMeta),
                   # 此项之前错误地被跳过，需要进一步调查，参见 issue #80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format",
                                device_type='cuda', dtypes=[torch.float64]),
                   # MPS 后端上 channels last 测试失败
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format",
                                device_type='mps', dtypes=[torch.float32]),
                   # 参见 #119108: MPSNDArrayConvolutionA14.mm:3976: failed assertion `destination datatype must be fp32'
                   # xfail 由于严重 Python 错误而无法正常工作
                   DecorateInfo(skipIfMps, "TestModule", "test_memory_format",
                                device_type='mps', dtypes=[torch.float16]),
                   # MPS 上跳过非连续张量测试
                   DecorateInfo(skipIfMps, "TestModule", "test_non_contiguous_tensors",
                                device_type='mps', dtypes=[torch.float16]),
               ),
               # 设置精度覆盖器，针对 float32 的测试用例设定精度为 1e-04
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.LazyConv3d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=3, lazy=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # 检查 CUDA 版本是否大于等于 8005，以支持 channels_last
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=8005), 'TestModule', 'test_memory_format'),
                   # 在 ROCM 平台上，对于 float32 存在问题 #70125，因此跳过测试
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # Lazy 模块在 meta 设备上无法与 ModuleInfo 测试很好地配合，参见 issue #70505
                   DecorateInfo(skipMeta),
                   # LazyConv3d 在 MPS 后端不受支持
                   DecorateInfo(skipMPS),
                   # 此测试之前被错误地跳过，需要进一步调查，参见 issue #80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),
               ),
               decorators=(
                   # 为 torch.float32 设置精度覆盖值
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.LazyConvTranspose1d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=1, lazy=True, transposed=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # 检查 CUDA 版本是否大于等于 7603，以支持 channels_last
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # 在 ROCM 平台上，对于 float32 存在问题 #70125，因此跳过测试
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # Lazy 模块在 meta 设备上无法与 ModuleInfo 测试很好地配合，参见 issue #70505
                   DecorateInfo(skipMeta),
                   # MPSNDArrayConvolutionA14.mm:3976: failed assertion `destination datatype must be fp32'，因此跳过测试
                   DecorateInfo(skipIfMps, "TestModule", "test_memory_format",
                                device_type='mps', dtypes=[torch.float16]),
                   # 在 MPS 后端，对于 float16 的非连续张量也需要跳过测试
                   DecorateInfo(skipIfMps, "TestModule", "test_non_contiguous_tensors",
                                device_type='mps', dtypes=[torch.float16]),
               ),
               decorators=(
                   # 为 torch.float32 设置精度覆盖值
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    # 使用 ModuleInfo 初始化一个 LazyConvTranspose2d 模块信息对象，设置相关参数和选项
    ModuleInfo(torch.nn.LazyConvTranspose2d,
               
               # 使用部分函数应用设置 module_inputs_func 参数，传入特定参数 N=2, lazy=True, transposed=True
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=2, lazy=True, transposed=True),
               
               # 设置梯度检查时的非确定性容差值为预定义的 GRADCHECK_NONDET_TOL
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               
               # 指定模块的内存格式是否影响输出结果
               module_memformat_affects_out=True,
               
               # 设置需要跳过的测试项元组 skips
               skips=(
                   # 对于 CUDA，仅在 cudnn 版本大于等于 7603 时支持 channels_last
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   
                   # 在 ROCM 平台上，对于 float32 存在问题，故跳过此测试
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   
                   # Lazy 模块在 meta 设备上不与 ModuleInfo 测试兼容，详见 GitHub issue 70505
                   DecorateInfo(skipMeta),
                   
                   # 在 CUDA 设备上，对于 float64 存在问题，故跳过此测试，详见 GitHub issue 80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='cuda',
                                dtypes=[torch.float64]),
                   
                   # 在 MPS 后端，channels last 测试失败
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format",
                                device_type='mps', dtypes=[torch.float32]),
                   
                   # MPS 后端相关问题，跳过对于 float16 的测试
                   DecorateInfo(skipIfMps, "TestModule", "test_memory_format",
                                device_type='mps', dtypes=[torch.float16]),
                   
                   # MPS 后端相关问题，跳过非连续张量的测试
                   DecorateInfo(skipIfMps, "TestModule", "test_non_contiguous_tensors",
                                device_type='mps', dtypes=[torch.float16]),
               ),
               
               # 设置修饰器 decorators，这里使用 precisionOverride 重写 float32 的精度值
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    # 定义 ModuleInfo 对象，用于 LazyConvTranspose3d 模块的测试配置
    ModuleInfo(torch.nn.LazyConvTranspose3d,
               # 设置模块输入函数为 torch_nn_ConvNd 的部分应用，N=3，lazy=True，transposed=True
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=3, lazy=True, transposed=True),
               # 梯度检查的非确定性容忍度
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 模块内存格式影响输出
               module_memformat_affects_out=True,
               # 跳过的测试情况
               skips=(
                   # 当 CUDA 上的 cuDNN 版本小于 8005 时跳过，支持 channels_last
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=8005), 'TestModule', 'test_memory_format'),
                   # 在 ROCM 上存在 float32 问题，跳过
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # Lazy 模块目前与元设备上的 ModuleInfo 测试不兼容
                   DecorateInfo(skipMeta),
                   # LazyConvTranspose3d 在 MPS 后端不受支持
                   DecorateInfo(skipMPS),
                   # 此项之前错误地被跳过，需要进一步调查
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),
               ),
               # 修饰器配置
               decorators=(
                   # 对于 torch.float32 类型的测试，设置精度覆盖
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    # 定义 ModuleInfo 对象，用于 torch.nn.Linear 模块的测试配置
    ModuleInfo(torch.nn.Linear,
               # 设置模块输入函数为 module_inputs_torch_nn_Linear
               module_inputs_func=module_inputs_torch_nn_Linear,
               # 跳过的测试情况
               skips=(
                   # 当前不支持 Linear 的 channels_last
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 定义 ModuleInfo 对象，用于 torch.nn.Bilinear 模块的测试配置
    ModuleInfo(torch.nn.Bilinear,
               # 设置模块输入函数为 module_inputs_torch_nn_Bilinear
               module_inputs_func=module_inputs_torch_nn_Bilinear,
               # 修饰器配置
               decorators=[
                   # 设置容差覆盖，针对不同设备和数据类型
                   DecorateInfo(
                       toleranceOverride({
                           torch.float32: tol(atol=1e-4, rtol=1e-4),
                           torch.float64: tol(atol=1e-4, rtol=1e-4)}),
                       'TestModule', 'test_forward', device_type='cpu'),
               ],
               # 跳过的测试情况
               skips=(
                   # 当前不支持 Bilinear 的 channels_last
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 见问题 #119108：容差问题
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    # 定义 ModuleInfo 对象，用于 torch.nn.LPPool1d 模块的测试配置
    ModuleInfo(torch.nn.LPPool1d,
               # 设置模块输入函数为 module_inputs_torch_nn_LPPool1d
               module_inputs_func=module_inputs_torch_nn_LPPool1d,
               # 跳过的测试情况
               skips=(
                   # 跳过梯度测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_grad'),
                   # 跳过二阶梯度测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),)
               ),
    # 定义 ModuleInfo 对象，包含 torch.nn.LPPool2d 模块信息及相关配置
    ModuleInfo(torch.nn.LPPool2d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_LPPool2d,
               # 定义跳过的测试信息列表
               skips=(
                   # 装饰器信息：跳过 'TestModule' 下的 'test_grad' 测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_grad'),
                   # 装饰器信息：跳过 'TestModule' 下的 'test_gradgrad' 测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),
                   # 装饰器信息：预期测试失败，在 MPS 设备上的 'test_memory_format' 测试
                   # 参考：https://github.com/pytorch/pytorch/issues/107214
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                       device_type='mps',
                   ),
               ),
    # 定义 ModuleInfo 对象，包含 torch.nn.LPPool3d 模块信息及相关配置
    ModuleInfo(torch.nn.LPPool3d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_LPPool3d,
               # 定义跳过的测试信息列表
               skips=(
                   # 装饰器信息：跳过 'TestModule' 下的 'test_grad' 测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_grad'),
                   # 装饰器信息：跳过 'TestModule' 下的 'test_gradgrad' 测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),
                   # 装饰器信息：跳过 'TestModule' 下的 'test_memory_format' 测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 装饰器信息：根据条件跳过测试
                   DecorateInfo(skipIfMps),
               ),
    # 定义 ModuleInfo 对象，包含 torch.nn.MaxPool1d 模块信息及相关配置
    ModuleInfo(torch.nn.MaxPool1d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_MaxPool1d,
               ),
    # 定义 ModuleInfo 对象，包含 torch.nn.MaxPool2d 模块信息及相关配置
    ModuleInfo(torch.nn.MaxPool2d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_MaxPool2d,
               ),
    # 定义 ModuleInfo 对象，包含 torch.nn.MaxPool3d 模块信息及相关配置
    ModuleInfo(torch.nn.MaxPool3d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_MaxPool3d,
               # 定义梯度检查的非确定性容差值
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 定义跳过的测试信息列表
               skips=(
                   # 装饰器信息：在 MPS 后端上不支持的测试
                   DecorateInfo(skipMPS),
               ),
    # 定义 ModuleInfo 对象，包含 torch.nn.KLDivLoss 模块信息及相关配置
    ModuleInfo(torch.nn.KLDivLoss,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_KLDivLoss,
               # 定义跳过的测试信息列表
               skips=(
                   # 装饰器信息：损失函数不支持 channels_last 格式
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 装饰器信息：参考 https://github.com/pytorch/pytorch/issues/115588
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_cpu_gpu_parity'),
                   # 装饰器信息：跳过 'TestModule' 下的 'test_grad' 测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_grad'),
                   # 装饰器信息：跳过 'TestModule' 下的 'test_gradgrad' 测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),
               ),
    # 创建 ModuleInfo 对象，表示 torch.nn.MSELoss 模块信息
    ModuleInfo(torch.nn.MSELoss,
               # 指定输入模块的函数为 module_inputs_torch_nn_MSELoss
               module_inputs_func=module_inputs_torch_nn_MSELoss,
               # 定义跳过的测试用例列表
               skips=(
                   # 没有对 channels_last 的支持用于损失函数。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 见 #119108: 输入类型 'tensor<f32>' 和 'tensor<15x10xf16>' 不兼容广播
                   DecorateInfo(skipIfMps, 'TestModule', 'test_non_contiguous_tensors', dtypes=[torch.float16]),
                   # 见 #119108: 公差问题
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    # 创建 ModuleInfo 对象，表示 torch.nn.MarginRankingLoss 模块信息
    ModuleInfo(torch.nn.MarginRankingLoss,
               # 指定输入模块的函数为 module_inputs_torch_nn_MarginRankingLoss
               module_inputs_func=module_inputs_torch_nn_MarginRankingLoss,
               # 定义跳过的测试用例列表
               skips=(
                   # 没有对 channels_last 的支持用于损失函数。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 创建 ModuleInfo 对象，表示 torch.nn.MultiLabelMarginLoss 模块信息
    ModuleInfo(torch.nn.MultiLabelMarginLoss,
               # 指定输入模块的函数为 module_inputs_torch_nn_MultiLabelMarginLoss
               module_inputs_func=module_inputs_torch_nn_MultiLabelMarginLoss,
               # 定义跳过的测试用例列表
               skips=(
                   # 没有对 channels_last 的支持用于损失函数。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 'aten::multilabel_margin_loss_forward' 目前未在 MPS 设备上实现。
                   DecorateInfo(skipIfMps, 'TestModule'),
                   # 'aten::multilabel_margin_loss_backward' 的导数未实现。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),)
               ),
    # 创建 ModuleInfo 对象，表示 torch.nn.MultiMarginLoss 模块信息
    ModuleInfo(torch.nn.MultiMarginLoss,
               # 指定输入模块的函数为 module_inputs_torch_nn_MultiMarginLoss
               module_inputs_func=module_inputs_torch_nn_MultiMarginLoss,
               # 定义跳过的测试用例列表
               skips=(
                   # 没有对 channels_last 的支持用于损失函数。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 'aten::multi_margin_loss' 目前未在 MPS 设备上实现。
                   DecorateInfo(skipIfMps, 'TestModule'),
                   # RuntimeError: 'aten::multi_margin_loss_backward' 的导数未实现。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),)
               ),
    # 创建 ModuleInfo 对象，表示 torch.nn.SoftMarginLoss 模块信息
    ModuleInfo(torch.nn.SoftMarginLoss,
               # 指定输入模块的函数为 module_inputs_torch_nn_SoftMarginLoss
               module_inputs_func=module_inputs_torch_nn_SoftMarginLoss,
               # 定义跳过的测试用例列表
               skips=(
                   # 没有对 channels_last 的支持用于损失函数。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 见 #119108: 公差问题
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.MultiLabelSoftMarginLoss,
               module_inputs_func=module_inputs_torch_nn_MultiLabelSoftMarginLoss,
               skips=(
                   # 不支持 channels_last 的损失函数。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.NLLLoss,
               module_inputs_func=module_inputs_torch_nn_NLLLoss,
               skips=(
                   # 不支持 channels_last 的损失函数。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 见 issue #119108：容差问题
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.GaussianNLLLoss,
               module_inputs_func=module_inputs_torch_nn_GaussianNLLLoss,
               skips=(
                   # 不支持 channels_last 的损失函数。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)),
    ModuleInfo(torch.nn.PoissonNLLLoss,
               module_inputs_func=module_inputs_torch_nn_PoissonNLLLoss,
               skips=(
                   # 不支持 channels_last 的损失函数。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)),
    ModuleInfo(torch.nn.HingeEmbeddingLoss,
               module_inputs_func=module_inputs_torch_nn_HingeEmbeddingLoss,
               skips=(
                   # 不支持 channels_last 的损失函数。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.HuberLoss,
               module_inputs_func=module_inputs_torch_nn_HuberLoss,
               skips=(
                   # 不支持 channels_last 的损失函数。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 见 issue #119108：输出数据类型似乎不正确
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.BCELoss,
               module_inputs_func=module_inputs_torch_nn_BCELoss,
               skips=(
                   # 不支持 channels_last 的损失函数。
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 错误：输入类型 'tensor<f32>' 和 'tensor<15x10xf16>' 不兼容广播
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.BCEWithLogitsLoss,
               module_inputs_func=module_inputs_torch_nn_BCEWithLogitsLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # see #119108: tolerance issue
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.CrossEntropyLoss,
               module_inputs_func=module_inputs_torch_nn_CrossEntropyLoss,
               dtypes=get_all_fp_dtypes(include_half=True, include_bfloat16=False),
               decorators=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_memory_format'),
                   # Define a tolerance override for float16 dtype.
                   DecorateInfo(toleranceOverride({torch.float16: tol(atol=3e-2, rtol=1e-3)}), "TestModule",
                                "test_forward", dtypes=[torch.float16], device_type='cpu'),
                   # Expect failure in CPU-GPU parity test for float16 dtype on CUDA.
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_cpu_gpu_parity", dtypes=[torch.float16],
                                device_type='cuda'),),
               ),
    ModuleInfo(torch.nn.CTCLoss,
               module_inputs_func=module_inputs_torch_nn_CTCLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # Skip test if running on MPS device.
                   DecorateInfo(skipIfMps, 'TestModule'),
                   # Skip gradient tests due to unimplemented derivative.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_grad'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad'),
                   # Issue with non-contiguous tensors, skip this test.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_non_contiguous_tensors'),)
               ),
    ModuleInfo(torch.nn.GELU,
               module_inputs_func=module_inputs_torch_nn_GELU,
               skips=(
                   # See #119108: tolerance issue
                   # Expect failure in forward test for float16 on MPS device.
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward",
                                device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.GLU,
               module_inputs_func=module_inputs_torch_nn_GLU,
               ),


注释：
    # 定义 ModuleInfo 实例，用于 torch.nn.GroupNorm 模块
    ModuleInfo(torch.nn.GroupNorm,
               # 指定模块输入函数为 module_inputs_torch_nn_GroupNorm
               module_inputs_func=module_inputs_torch_nn_GroupNorm,
               # 获取所有浮点类型数据类型，包括 bfloat16 和 half
               dtypes=get_all_fp_dtypes(include_bfloat16=True, include_half=True),
               # 跳过以下测试条件
               skips=(
                   # 跳过测试，链接到 GitHub 上的 issue 跟踪页面
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_cpu_gpu_parity'),
                   # 对于 CPU 设备，设置特定的测试容忍度
                   DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-4, rtol=1e-4)}),
                                'TestModule', 'test_memory_format', device_type='cpu'),
                   # 当前不支持 GroupNorm 的 channels_last 特性
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format', device_type='cuda'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format', device_type='mps'),
                   # 当使用 ROCm 时，跳过梯度测试
                   DecorateInfo(unittest.skip("Skipped!"), "TestModule", "test_grad",
                                active_if=TEST_WITH_ROCM, device_type='cuda'),
               )
               ),
    # 定义 ModuleInfo 实例，用于 torch.nn.Hardshrink 模块
    ModuleInfo(torch.nn.Hardshrink,
               # 指定模块输入函数为 module_inputs_torch_nn_Hardshrink
               module_inputs_func=module_inputs_torch_nn_Hardshrink,
               # 跳过 MPS 后端的测试
               skips=(
                   DecorateInfo(skipMPS),),
               ),
    # 定义 ModuleInfo 实例，用于 torch.nn.Hardswish 模块
    ModuleInfo(torch.nn.Hardswish,
               # 指定模块输入函数为 module_inputs_torch_nn_Hardswish
               module_inputs_func=module_inputs_torch_nn_Hardswish,
               # 不支持 MPS 后端的反向传播检查
               skips=(
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                       device_type='mps',
                   ),),
               # 不支持 gradgrad 计算
               supports_gradgrad=False),
    # 定义 ModuleInfo 实例，用于 torch.nn.Hardtanh 模块
    ModuleInfo(torch.nn.Hardtanh,
               # 指定模块输入函数为 module_inputs_torch_nn_Hardtanh
               module_inputs_func=module_inputs_torch_nn_Hardtanh,
               ),
    # 定义 ModuleInfo 实例，用于 torch.nn.InstanceNorm1d 模块
    ModuleInfo(torch.nn.InstanceNorm1d,
               # 指定模块输入函数为 module_inputs_torch_nn_InstanceNormNd，其中 N=1
               module_inputs_func=partial(module_inputs_torch_nn_InstanceNormNd, N=1),
               # 训练和评估过程中行为不同
               train_and_eval_differ=True,
               # 不支持 channels_last 特性的 InstanceNorm1d
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 定义 ModuleInfo 实例，用于 torch.nn.InstanceNorm2d 模块
    ModuleInfo(torch.nn.InstanceNorm2d,
               # 指定模块输入函数为 module_inputs_torch_nn_InstanceNormNd，其中 N=2
               module_inputs_func=partial(module_inputs_torch_nn_InstanceNormNd, N=2),
               # 训练和评估过程中行为不同
               train_and_eval_differ=True,
               # 不支持 channels_last 特性的 InstanceNorm2d
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 定义一个 ModuleInfo 实例，包含 torch.nn.InstanceNorm3d 模块信息及其配置
    ModuleInfo(torch.nn.InstanceNorm3d,
               # 部分函数应用，配置输入函数为 module_inputs_torch_nn_InstanceNormNd，N=3
               module_inputs_func=partial(module_inputs_torch_nn_InstanceNormNd, N=3),
               # 训练与评估有区别
               train_and_eval_differ=True,
               skips=(
                   # 在 MPS 后端不受支持
                   DecorateInfo(skipMPS),
                   # InstanceNorm3d 目前不支持 channels_last
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 定义一个 ModuleInfo 实例，包含 torch.nn.LocalResponseNorm 模块信息及其配置
    ModuleInfo(torch.nn.LocalResponseNorm,
               # 配置输入函数为 module_inputs_torch_nn_LocalResponseNorm
               module_inputs_func=module_inputs_torch_nn_LocalResponseNorm,
               skips=(
                   # 使用不支持 MPS 后端的 avg_pool3d
                   DecorateInfo(skipMPS),)
               ),
    # 定义一个 ModuleInfo 实例，包含 torch.nn.LayerNorm 模块信息及其配置
    ModuleInfo(torch.nn.LayerNorm,
               # 配置输入函数为 module_inputs_torch_nn_LayerNorm
               module_inputs_func=module_inputs_torch_nn_LayerNorm,
               skips=(
                   # LayerNorm 目前不支持 channels_last
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 定义一个 ModuleInfo 实例，包含 torch.nn.RMSNorm 模块信息及其配置
    ModuleInfo(torch.nn.RMSNorm,
               # 配置输入函数为 module_inputs_torch_nn_RMSNorm
               module_inputs_func=module_inputs_torch_nn_RMSNorm,
               ),
    # 定义一个 ModuleInfo 实例，包含 torch.nn.TransformerEncoder 模块信息及其配置
    # TransformerEncoder 使用与 TransformerEncoderLayer 相同的输入
    ModuleInfo(torch.nn.TransformerEncoder,
               # 训练与评估有区别
               train_and_eval_differ=True,
               # 配置输入函数为 module_inputs_torch_nn_TransformerEncoder
               module_inputs_func=module_inputs_torch_nn_TransformerEncoder,
               decorators=[
                   # 不支持 SDPA 反向导数
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad',
                                device_type='cpu'),
               ],
               skips=(
                   # TransformerEncoderLayer 目前不支持 channels_last
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 不直接支持 device / dtype kwargs，因为它只是 TransformerEncoderLayers 的容器
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_factory_kwargs'),)
               ),
    # 定义 ModuleInfo 对象，描述 torch.nn.TransformerEncoderLayer 模块
    ModuleInfo(torch.nn.TransformerEncoderLayer,
               # 声明 train_and_eval_differ 参数为 True
               train_and_eval_differ=True,
               # 指定 module_inputs_func 参数为 module_inputs_torch_nn_TransformerEncoderLayer 函数
               module_inputs_func=module_inputs_torch_nn_TransformerEncoderLayer,
               # 添加装饰器信息列表
               decorators=[
                   # 添加 DecorateInfo 对象，设置特定于 torch.float32 类型的容差修正
                   DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-4, rtol=1e-4)}),
                                'TestModule', 'test_non_contiguous_tensors',
                                device_type='cpu', active_if=IS_WINDOWS),
                   # 添加 DecorateInfo 对象，跳过未实现的测试函数，用于 SDPA 后向导数
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad',
                                device_type='cpu'),
               ],
               # 设置 skips 元组，包含 DecorateInfo 对象，跳过不支持的测试用例
               skips=(
                   # 添加 DecorateInfo 对象，跳过未实现的测试函数，无通道优先支持
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 定义 ModuleInfo 对象，描述 torch.nn.TransformerDecoderLayer 模块
    ModuleInfo(torch.nn.TransformerDecoderLayer,
               # 指定 module_inputs_func 参数为 module_inputs_torch_nn_TransformerDecoderLayer 函数
               module_inputs_func=module_inputs_torch_nn_TransformerDecoderLayer,
               # 添加装饰器信息列表
               decorators=[
                   # 添加 DecorateInfo 对象，跳过未实现的测试函数，用于 SDPA 后向导数
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad',
                                device_type='cpu'),
               ],
               # 设置 skips 元组，包含 DecorateInfo 对象，跳过不支持的测试用例
               skips=(
                   # 添加 DecorateInfo 对象，跳过未实现的测试函数，无通道优先支持
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 定义 ModuleInfo 对象，描述 torch.nn.Transformer 模块
    ModuleInfo(torch.nn.Transformer,
               # 指定 module_inputs_func 参数为 module_inputs_torch_nn_Transformer 函数
               module_inputs_func=module_inputs_torch_nn_Transformer,
               # 添加装饰器信息列表
               decorators=[
                   # 添加 DecorateInfo 对象，跳过未实现的测试函数，用于 SDPA 后向导数
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_gradgrad',
                                device_type='cpu'),
               ],
               # 设置 skips 元组，包含 DecorateInfo 对象，跳过不支持的测试用例
               skips=(
                   # 添加 DecorateInfo 对象，跳过未实现的测试函数，无通道优先支持
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 定义 ModuleInfo 对象，描述 torch.nn.MultiheadAttention 模块
    ModuleInfo(torch.nn.MultiheadAttention,
               # 声明 train_and_eval_differ 参数为 True
               train_and_eval_differ=True,
               # 指定 module_inputs_func 参数为 module_inputs_torch_nn_MultiheadAttention 函数
               module_inputs_func=module_inputs_torch_nn_MultiheadAttention,
               # 设置 skips 元组，包含 DecorateInfo 对象，跳过不支持的测试用例
               skips=(
                   # 添加 DecorateInfo 对象，跳过未实现的测试函数，无通道优先支持
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    # 定义 ModuleInfo 对象，描述 torch.nn.Embedding 模块
    ModuleInfo(torch.nn.Embedding,
               # 指定 module_inputs_func 参数为 module_inputs_torch_nn_Embedding 函数
               module_inputs_func=module_inputs_torch_nn_Embedding,
               # 添加装饰器信息列表
               decorators=[
                   # 添加 DecorateInfo 对象，设置特定于 torch.float32 类型的容差修正
                   DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-4, rtol=1e-4)}),
                                'TestModule', 'test_non_contiguous_tensors',
                                device_type='mps')],
               # 设置 skips 元组，包含 DecorateInfo 对象，跳过不支持的测试用例
               skips=(
                   # 添加 DecorateInfo 对象，跳过未实现的测试函数，无通道优先支持
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.ReLU,
               module_inputs_func=module_inputs_torch_nn_ReLU,
               skips=(
                   # 在 MPS 上的反向检查失败
                   # 参见 https://github.com/pytorch/pytorch/issues/107214
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                       device_type='mps',
                   ),)
               ),
    ModuleInfo(torch.nn.LeakyReLU,
               module_inputs_func=module_inputs_torch_nn_LeakyReLU,
               ),
    ModuleInfo(torch.nn.ReLU6,
               module_inputs_func=module_inputs_torch_nn_ReLU6,
               skips=(
                   # 在 MPS 后端上测试失败，正在调查中
                   # 参见 https://github.com/pytorch/pytorch/issues/100914
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.PReLU,
               module_inputs_func=module_inputs_torch_nn_PReLU,
               skips=(
                   # 在 MPS 后端上测试失败，正在调查中
                   # 参见 https://github.com/pytorch/pytorch/issues/100914
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.RNNCell,
               module_inputs_func=partial(module_inputs_torch_nn_RNN_GRU_Cell, is_rnn=True),
               module_error_inputs_func=module_error_inputs_torch_nn_RNN_GRU_Cell,
               ),
    ModuleInfo(torch.nn.GRUCell,
               module_inputs_func=module_inputs_torch_nn_RNN_GRU_Cell,
               module_error_inputs_func=module_error_inputs_torch_nn_RNN_GRU_Cell,
               ),
    ModuleInfo(torch.nn.LSTMCell,
               module_inputs_func=module_inputs_torch_nn_LSTMCell,
               module_error_inputs_func=module_error_inputs_torch_nn_LSTMCell,
               ),
    ModuleInfo(torch.nn.Sigmoid,
               module_inputs_func=module_inputs_torch_nn_Sigmoid,
               skips=(
                   # 在 MPS 上的反向检查失败
                   # 参见 https://github.com/pytorch/pytorch/issues/107214
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                       device_type='mps',
                   ),)
               ),
    ModuleInfo(torch.nn.LogSigmoid,
               module_inputs_func=module_inputs_torch_nn_LogSigmoid,
               skips=(
                   # 查看问题 #119108：容差问题
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward", device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.SiLU,
               module_inputs_func=module_inputs_torch_nn_SiLU,
               ),
    ModuleInfo(torch.nn.Softmax,
               module_inputs_func=module_inputs_torch_nn_Softmax,
               ),
    ModuleInfo(torch.nn.Softmax2d,
               module_inputs_func=module_inputs_torch_nn_Softmax2d,
               skips=(
                   # 当前不支持 Softmax2d 的 channels last 格式
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 见问题 #119108：容差问题
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward", device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.LogSoftmax,
               module_inputs_func=module_inputs_torch_nn_LogSoftmax,
               skips=(
                   # 当前不支持 LogSoftmax 的 channels last 格式
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # 见问题 #119108：出现 inf 和 nan 错误
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_forward", device_type='mps', dtypes=[torch.float16]),)
               ),
    ModuleInfo(torch.nn.Softmin,
               module_inputs_func=module_inputs_torch_nn_Softmin,
               skips=(
                   # 当前不支持 Softmin 的 channels last 格式
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.Softplus,
               module_inputs_func=module_inputs_torch_nn_Softplus,
               skips=(
                   # 在 MPS 后端上测试失败，正在调查中。
                   # 参见 https://github.com/pytorch/pytorch/issues/100914
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.Softshrink,
               module_inputs_func=module_inputs_torch_nn_Softshrink,
               skips=(
                   # 在 MPS 后端上不支持
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.Softsign,
               module_inputs_func=module_inputs_torch_nn_Softsign,
               ),
    ModuleInfo(torch.nn.Tanh,
               module_inputs_func=module_inputs_torch_nn_Tanh,
               skips=(
                   # 在 MPS 上的反向检查失败
                   # 参见 https://github.com/pytorch/pytorch/issues/107214
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                       device_type='mps',
                   ),)
               ),
    ModuleInfo(torch.nn.Tanhshrink,
               module_inputs_func=module_inputs_torch_nn_Tanhshrink,
               skips=(
                   # 在 MPS 上的反向检查失败
                   # 参见 https://github.com/pytorch/pytorch/issues/107214
                   DecorateInfo(
                       unittest.expectedFailure,
                       'TestModule',
                       'test_memory_format',
                       active_if=operator.itemgetter('training'),
                       device_type='mps',
                   ),)
               ),
    ModuleInfo(torch.nn.Threshold,
               module_inputs_func=module_inputs_torch_nn_Threshold,
               skips=(
                   # 在 MPS 后端上测试失败，正在调查中
                   # 参见 https://github.com/pytorch/pytorch/issues/100914
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.Mish,
               module_inputs_func=module_inputs_torch_nn_Mish,
               skips=(
                   # 不支持 MPS 后端
                   DecorateInfo(skipMPS),)
               ),
    ModuleInfo(torch.nn.RNN,
               train_and_eval_differ=True,
               module_inputs_func=partial(module_inputs_torch_nn_RNN_GRU, is_rnn=True),
               module_error_inputs_func=module_error_inputs_torch_nn_RNN_GRU,
               decorators=rnn_gru_lstm_module_info_decorators
               ),
    ModuleInfo(torch.nn.GRU,
               train_and_eval_differ=True,
               module_inputs_func=partial(module_inputs_torch_nn_RNN_GRU, is_rnn=False),
               module_error_inputs_func=module_error_inputs_torch_nn_RNN_GRU,
               decorators=rnn_gru_lstm_module_info_decorators),
    ModuleInfo(torch.nn.LSTM,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_LSTM,
               module_error_inputs_func=module_error_inputs_torch_nn_RNN_GRU,
               skips=(
                   # 带有投影的 LSTM 目前不支持 MPS
                   DecorateInfo(skipMPS),),
               decorators=rnn_gru_lstm_module_info_decorators),
    ModuleInfo(torch.nn.ReflectionPad1d,
               module_inputs_func=module_inputs_torch_nn_ReflectionPad1d,
               ),
    ModuleInfo(torch.nn.ReflectionPad2d,
               module_inputs_func=module_inputs_torch_nn_ReflectionPad2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   # 被跳过的单元测试，因为在 CUDA 设备和 MPS 上运行
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='cuda'),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='mps'),)
               ),
    # 创建 ModuleInfo 对象，指定模块为 torch.nn.ReflectionPad3d
    ModuleInfo(torch.nn.ReflectionPad3d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_ReflectionPad3d,
               # 梯度检查的非确定性容差
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 跳过的装饰器信息元组
               skips=(
                   # 使用 unittest.skip 装饰器跳过 CUDA 设备上的测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='cuda'),
                   # 使用 unittest.skip 装饰器跳过 MPS 设备上的测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='mps'),)
               ),
    # 创建 ModuleInfo 对象，指定模块为 torch.nn.ReplicationPad1d
    ModuleInfo(torch.nn.ReplicationPad1d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_ReplicationPad1d,
               ),
    # 创建 ModuleInfo 对象，指定模块为 torch.nn.ReplicationPad2d
    ModuleInfo(torch.nn.ReplicationPad2d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_ReplicationPad2d,
               # 梯度检查的非确定性容差
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 跳过的装饰器信息元组
               skips=(
                   # 使用 unittest.skip 装饰器跳过 CUDA 设备上的测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='cuda'),
                   # 使用 unittest.skip 装饰器跳过 MPS 设备上的测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='mps'),)
               ),
    # 创建 ModuleInfo 对象，指定模块为 torch.nn.ReplicationPad3d
    ModuleInfo(torch.nn.ReplicationPad3d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_ReplicationPad3d,
               # 梯度检查的非确定性容差
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               # 跳过的装饰器信息元组
               skips=(
                   # 使用 unittest.skip 装饰器跳过 CUDA 设备上的测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='cuda'),
                   # 使用 unittest.skip 装饰器跳过 MPS 设备上的测试
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format',
                                device_type='mps'),)
               ),
    # 创建 ModuleInfo 对象，指定模块为 torch.nn.SELU
    ModuleInfo(torch.nn.SELU,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_SELU,
               # 跳过的装饰器信息元组
               skips=(
                   # 在 MPS 后端上失败的测试，正在调查
                   # 参见 https://github.com/pytorch/pytorch/issues/100914
                   DecorateInfo(skipMPS),)
               ),
    # 创建 ModuleInfo 对象，指定模块为 torch.nn.ZeroPad1d
    ModuleInfo(torch.nn.ZeroPad1d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_ZeroPad1d,
               ),
    # 创建 ModuleInfo 对象，指定模块为 torch.nn.ZeroPad2d
    ModuleInfo(torch.nn.ZeroPad2d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_ZeroPad2d,
               # 跳过的装饰器信息元组
               skips=(
                   # 在 MPS 后端上的通道最后测试失败
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='mps'),)
               ),
    # 创建 ModuleInfo 对象，指定模块为 torch.nn.ZeroPad3d
    ModuleInfo(torch.nn.ZeroPad3d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_ZeroPad3d,
               # 跳过的装饰器信息元组
               skips=(
                   # 在 MPS 后端上的通道最后测试失败
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='mps'),)
               ),
    # 创建 ModuleInfo 对象，指定模块为 torch.nn.CircularPad1d
    ModuleInfo(torch.nn.CircularPad1d,
               # 指定输入模块函数
               module_inputs_func=module_inputs_torch_nn_CircularPad1d,
               # 指定错误输入模块函数
               module_error_inputs_func=module_error_inputs_torch_nn_Pad1d,
               ),
    ModuleInfo(torch.nn.CircularPad2d,
               module_inputs_func=module_inputs_torch_nn_CircularPad2d,
               module_error_inputs_func=module_error_inputs_torch_nn_Pad2d,
               ),
    # 定义一个 ModuleInfo 对象，描述了 torch.nn.CircularPad2d 模块的信息，
    # 包括输入数据生成函数和错误输入数据生成函数

    ModuleInfo(torch.nn.CircularPad3d,
               module_inputs_func=module_inputs_torch_nn_CircularPad3d,
               module_error_inputs_func=module_error_inputs_torch_nn_Pad3d,
               skips=(
                   # 在 MPS 后端上以 channels last 测试失败
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),)
               ),
    # 定义另一个 ModuleInfo 对象，描述了 torch.nn.CircularPad3d 模块的信息，
    # 包括输入数据生成函数和错误输入数据生成函数，并指定在特定条件下跳过测试

    ModuleInfo(torch.nn.ConstantPad1d,
               module_inputs_func=module_inputs_torch_nn_ConstantPad1d,
               ),
    # 定义 ModuleInfo 对象，描述了 torch.nn.ConstantPad1d 模块的信息，
    # 包括输入数据生成函数

    ModuleInfo(torch.nn.ConstantPad2d,
               module_inputs_func=module_inputs_torch_nn_ConstantPad2d,
               skips=(
                   # 在 MPS 后端上以 channels last 测试失败
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='mps'),)
               ),
    # 定义 ModuleInfo 对象，描述了 torch.nn.ConstantPad2d 模块的信息，
    # 包括输入数据生成函数，并指定在特定条件下跳过测试

    ModuleInfo(torch.nn.ConstantPad3d,
               module_inputs_func=module_inputs_torch_nn_ConstantPad3d,
               skips=(
                   # 在 MPS 后端上以 channels last 测试失败
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='mps'),)
               )
    # 定义 ModuleInfo 对象，描述了 torch.nn.ConstantPad3d 模块的信息，
    # 包括输入数据生成函数，并指定在特定条件下跳过测试
# 定义一个空列表，用于存储过滤后的结果
result = []

# 对于列表中的每个元素，检查是否为偶数，如果是则添加到结果列表中
for item in items:
    if item % 2 == 0:
        result.append(item)

# 返回过滤后的结果列表
return result
```