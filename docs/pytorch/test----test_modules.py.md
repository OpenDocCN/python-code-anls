# `.\pytorch\test\test_modules.py`

```
# Owner(s): ["module: nn"]

# 从标准库中导入必要的模块和函数
from itertools import chain, product
from inspect import signature, isgenerator
from copy import deepcopy
import tempfile
from operator import methodcaller

# 导入 PyTorch 库
import torch

# 导入 PyTorch 内部测试相关的模块和函数
from torch._subclasses.meta_utils import assert_metadata_eq
from torch.testing._internal.common_cuda import with_tf32_off
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyCPU, onlyCUDA, toleranceOverride, tol, skipMeta)
from torch.testing._internal.common_modules import module_db, modules, ModuleErrorEnum, TrainEvalMode
from torch.testing._internal.common_utils import (
    TestCase, run_tests, freeze_rng_state, mock_wrapper, get_tensors_from, gradcheck,
    gradgradcheck, parametrize, wrapSwapTensorsTest)

# 导入 unittest.mock 库中的 patch 和 call 函数
from unittest.mock import patch, call

# 定义一个测试类 TestModule，继承自 TestCase
class TestModule(TestCase):
    # 开启 CUDA 内存泄漏检查
    _do_cuda_memory_leak_check = True
    # 开启使用非默认 CUDA 流进行测试
    _do_cuda_non_default_stream = True
    # 定义测试精度和相对容差
    precision = 1e-5
    rel_tol = 1e-5

    # 定义一个私有方法，用于验证模块的参数和缓冲区的设备和数据类型
    def _assert_module_parameters_and_buffer_are(self, module, device, dtype):
        # 检查创建的参数和缓冲区的设备位置和数据类型
        # 只验证浮点数数据类型，因为这是 kwarg 或方法如 `float()` 的适用范围。
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # 定义内部函数 _check_module，用于验证每个项目的设备和数据类型
        def _check_module(items, name, device=device, dtype=dtype):
            for item_name, item in items:
                self.assertEqual(
                    item.device, device,
                    f'{name} {item_name} is on device {item.device} instead of the expected device {device}')
                if item.dtype.is_floating_point:
                    self.assertEqual(
                        item.dtype, dtype,
                        f'{name} {item_name} is of dtype {item.dtype} instead of the expected dtype {dtype}')

        # 验证模块的所有参数的设备和数据类型
        _check_module(module.named_parameters(), "Parameter")
        # 验证模块的所有缓冲区的设备和数据类型
        _check_module(module.named_buffers(), "Buffer")

    # 使用预定义的模块数据库进行测试模块
    @modules(module_db)
    # 测试模型的前向传播功能
    def test_forward(self, device, dtype, module_info, training):
        # 从模块信息中获取模块类
        module_cls = module_info.module_cls
        # 使用模块信息的函数获取模块输入数据
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        # 定义一个映射，根据数据类型选择对应的方法调用器
        dtype_to_method_caller = {
            torch.float32: methodcaller("float"),
            torch.float64: methodcaller("double"),
        }
        # 遍历模块输入数据
        for module_input in module_inputs:
            # 如果当前模块输入没有前向传播的输入数据，则跳过
            if module_input.forward_input is None:
                continue

            # 冻结随机数生成器状态
            with freeze_rng_state():
                # === 实例化模块。 ===
                # 从构造函数输入参数和关键字参数中获取参数和关键字
                args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
                # 使用获取的参数和关键字实例化模块对象
                m = module_cls(*args, **kwargs)
                # 将模块对象移动到指定设备上，并转换为指定数据类型
                m.to(device).to(dtype)
                # 设置模块的训练模式
                m.train(training)

                # === 执行前向传播。 ===
                # 从前向传播输入参数和关键字中获取参数和关键字
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                # 执行模块的前向传播，并获取输出结果
                outputs = m(*args, **kwargs)

                # === 如果指定了参考函数，则将输出与参考输出进行比较。 ===
                # TODO: 处理精度问题
                reference_fn = module_input.reference_fn
                if reference_fn is not None:
                    # 使用参考函数计算模块的参考输出
                    ref_outputs = reference_fn(m, *args, **kwargs)
                    # 使用断言检查模块输出与参考输出是否相等
                    self.assertEqual(outputs, ref_outputs)

                # === 使用方法调用器并验证参数和缓冲区。 ===
                # 如果当前数据类型在映射中，则调用对应的方法
                if dtype in dtype_to_method_caller:
                    # 调用对应数据类型的方法调用器对模块进行操作
                    dtype_to_method_caller[dtype](m)
                    # 再次执行模块的前向传播以验证参数和缓冲区
                    m(*args, **kwargs)
                    # 使用断言验证模块的参数和缓冲区是否符合预期
                    self._assert_module_parameters_and_buffer_are(m, device, dtype)
    # 定义测试函数，用于测试多设备间的数据传输功能
    def test_multiple_device_transfer(self, device, dtype, module_info, training):
        # 从 module_info 中获取模块类
        module_cls = module_info.module_cls
        # 使用 module_info 中的函数获取在指定设备上和 CPU 上的模块输入
        module_inputs_device = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                              requires_grad=False, training=training)
        module_inputs_cpu = module_info.module_inputs_func(module_info, device="cpu", dtype=dtype,
                                                           requires_grad=False, training=training)
        # 遍历设备和 CPU 上的模块输入
        for module_input_device, module_input_cpu in zip(module_inputs_device, module_inputs_cpu):
            # 如果设备上的 forward_input 为空，则跳过
            if module_input_device.forward_input is None:
                continue

            with freeze_rng_state():
                # === 实例化模块 ===
                args, kwargs = module_input_device.constructor_input.args, module_input_device.constructor_input.kwargs
                m = module_cls(*args, **kwargs)
                # 将模块移动到指定的设备和数据类型上
                m.to(device).to(dtype)
                # 设置模块的训练状态
                m.train(training)

                # === 在 GPU 上进行前向传播 ===
                input_device_args = module_input_device.forward_input.args
                input_device_kwargs = module_input_device.forward_input.kwargs
                m(*input_device_args, **input_device_kwargs)
                # 断言模块的参数和缓冲区在指定设备和数据类型上
                self._assert_module_parameters_and_buffer_are(m, device, dtype)

                # === 移动到 CPU 上 ===
                input_cpu_args = module_input_cpu.forward_input.args
                input_cpu_kwargs = module_input_cpu.forward_input.kwargs
                m.cpu()
                m(*input_cpu_args, **input_cpu_kwargs)
                # 断言模块的参数和缓冲区在 CPU 上
                self._assert_module_parameters_and_buffer_are(m, "cpu", dtype)

                # === 返回到 GPU 并进行前向传播 ===
                m.cuda()
                m(*input_device_args, **input_device_kwargs)
                # 断言模块的参数和缓冲区在指定设备和数据类型上
                self._assert_module_parameters_and_buffer_are(m, device, dtype)

                # 如果有多个 GPU，则测试跨 GPU 的数据传输是否正常工作
                if torch.cuda.device_count() >= 2:
                    # === 测试跨 GPU 的数据传输 ===
                    def _to_device1(objs):
                        if isinstance(objs, (tuple, list)):
                            return type(objs)(_to_device1(item) for item in objs)
                        elif isinstance(objs, dict):
                            return {name: _to_device1(item) for name, item in objs.items()}
                        elif isinstance(objs, torch.Tensor):
                            return objs.cuda(1)
                        else:
                            return objs
                    # 将设备上的输入数据移动到第一个 GPU 上
                    input_device_1_args = _to_device1(input_device_args)
                    input_device_1_kwargs = _to_device1(input_device_kwargs)

                    m.cuda(1)
                    # 使用第一个 GPU 运行模块
                    with torch.cuda.device(1):
                        m(*input_device_1_args, **input_device_1_kwargs)
                    # 断言模块的参数和缓冲区在指定的第一个 GPU 上
                    self._assert_module_parameters_and_buffer_are(m, torch.device("cuda:1"), dtype)

    @modules(module_db)
    # 测试模块能否通过 repr 和 str 方法表示且不出错
    def test_repr(self, device, dtype, module_info, training):
        # 获取模块类
        module_cls = module_info.module_cls
        # 获取模块输入信息
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        # 遍历模块的每个输入
        for module_input in module_inputs:
            # 获取构造函数的参数和关键字参数
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            # 创建模块实例
            m = module_cls(*args, **kwargs)
            # 将模块移到指定设备并设置数据类型
            m.to(device).to(dtype)
            # 设置模块的训练状态
            m.train(training)

            # 检查 repr 和 str 方法是否能正常调用且不抛出错误
            m.__repr__()
            str(m)

    # 使用装饰器 modules(module_db) 标记的测试函数
    @modules(module_db)
    # 测试模块能否被正确序列化和反序列化
    def test_save_load(self, device, dtype, module_info, training):
        # 获取模块类
        module_cls = module_info.module_cls
        # 获取模块输入信息
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        # 遍历模块的每个输入
        for module_input in module_inputs:
            # 如果 forward_input 为 None，则跳过
            if module_input.forward_input is None:
                continue

            # 获取构造函数的参数和关键字参数
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            # 使用 freeze_rng_state() 上下文管理器
            with freeze_rng_state():
                # 实例化模块
                args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
                m = module_cls(*args, **kwargs)
                # 将模块移到指定设备并设置数据类型
                m.to(device).to(dtype)
                # 设置模块的训练状态
                m.train(training)
                # 获取模块的状态字典
                sd = m.state_dict()

                # 执行前向传播
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                output = m(*args, **kwargs)

                # 检查保存和加载后的模块是否产生相同的输出
                with tempfile.TemporaryFile() as f:
                    torch.save(m, f)
                    f.seek(0)
                    m_copy = torch.load(f)
                    output_from_copy = m_copy(*args, **kwargs)
                    self.assertEqual(output, output_from_copy)

                # 检查保存和加载后的状态字典是否相同（包括仅加载权重的情况）
                with tempfile.TemporaryFile() as f:
                    torch.save(sd, f)
                    f.seek(0)
                    sd_copy = torch.load(f)
                    self.assertEqual(sd_copy, sd)
                    del sd_copy
                    f.seek(0)
                    sd_copy_wo = torch.load(f, weights_only=True)
                    self.assertEqual(sd_copy_wo, sd)

    # 使用 skipMeta 装饰器跳过的测试函数
    @skipMeta
    @modules([module_info for module_info in module_db
              if 'inplace' in signature(module_info.module_cls).parameters])
    # 定义测试函数，用于验证模块的就地操作是否与非就地操作给出相同的结果
    def test_check_inplace(self, device, dtype, module_info, training):
        # 获取模块类和模块输入信息的函数
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=True, training=training)
        # 遍历每个模块输入
        for module_input in module_inputs:
            # 如果模块的前向输入为空，则跳过
            if module_input.forward_input is None:
                continue

            # === 实例化模块 ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            # 创建非就地模块对象并设置设备和数据类型
            m_op = module_cls(*args, **kwargs, inplace=False)
            m_op.to(device).to(dtype)
            m_op.train(training)
            # 创建就地模块对象并设置设备和数据类型
            m_inplace = module_cls(*args, **kwargs, inplace=True)
            m_inplace.to(device).to(dtype)
            m_inplace.train(training)

            # === 就地模块仅支持在第一个参数上进行就地操作 ===
            input_args, input_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs

            # === 禁止第一个输入参数出现在 input_kwargs 中 ===
            forward_sig = signature(m_op).parameters
            self.assertGreaterEqual(len(forward_sig), 1)
            first_param_name = next(iter(forward_sig.items()))
            self.assertNotIn(first_param_name, input_kwargs)

            # === 非就地操作不会修改原始张量 ===
            self.assertGreaterEqual(len(input_args), 1)
            input_version = input_args[0]._version
            with freeze_rng_state():
                output_op = m_op(*input_args, **input_kwargs)
            self.assertEqual(input_args[0]._version, input_version)

            # === 检查就地操作是否给出相同的结果 ===
            input_arg_copy = deepcopy(input_args)
            input_arg_clone = tuple(i.clone() for i in input_arg_copy)
            input_clone_version = input_arg_clone[0]._version
            with freeze_rng_state():
                output_ip = m_inplace(*input_arg_clone, **input_kwargs)
            self.assertGreater(input_arg_clone[0]._version, input_clone_version)
            self.assertEqual(output_op, output_ip)

            # === 检查梯度是否相同 ===
            grad = output_op.data.clone().normal_()
            output_op.backward(grad)
            output_ip.backward(grad)
            self.assertEqual(input_args[0].grad, input_arg_copy[0].grad)
    # 递归遍历 obj 对象及其嵌套结构，应用 func 函数，并返回处理后的对象
    def _traverse_obj(self, obj, func):
        # 如果 obj 是元组或列表，则对每个元素递归调用 _traverse_obj，并返回相同类型的对象
        if isinstance(obj, (tuple, list)):
            return type(obj)(self._traverse_obj(o, func) for o in obj)
        # 如果 obj 是生成器，则将其转换为元组，并对每个元素递归调用 _traverse_obj
        elif isgenerator(obj):
            return tuple(self._traverse_obj(o, func) for o in obj)
        # 如果 obj 是字典，则对每个键值对递归调用 _traverse_obj，并返回处理后的字典
        elif isinstance(obj, dict):
            return {name: self._traverse_obj(o, func) for name, o in obj.items()}
        # 如果 obj 是 torch.Tensor 或 torch.nn.Parameter，则应用 func 函数
        elif isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
            return func(obj)
        # 对于其他类型的 obj，直接返回其本身
        else:
            return obj

    # 递归遍历 obj 对象及其嵌套结构，保留其梯度信息
    def _retain_grad(self, obj):
        # 内部函数 inner_retain_grad 用于检查是否需要保留梯度
        def inner_retain_grad(obj):
            if obj.requires_grad:
                obj.retain_grad()
        # 对 obj 对象及其嵌套结构调用 _traverse_obj，并应用 inner_retain_grad
        self._traverse_obj(obj, inner_retain_grad)

    # 递归遍历 obj 对象及其嵌套结构，获取梯度信息
    def _get_grads(self, obj):
        # 内部函数 inner_get_grad 用于获取梯度信息
        def inner_get_grad(obj):
            if obj.requires_grad:
                return obj.grad
        # 对 obj 对象及其嵌套结构调用 _traverse_obj，并应用 inner_get_grad
        return self._traverse_obj(obj, inner_get_grad)

    # 递归遍历 obj 对象及其嵌套结构，将梯度信息置零
    def _zero_grad(self, obj):
        # 内部函数 inner_zero_grad 用于将梯度置零
        def inner_zero_grad(obj):
            if obj.grad is not None:
                obj.grad = None
        # 对 obj 对象及其嵌套结构调用 _traverse_obj，并应用 inner_zero_grad
        self._traverse_obj(obj, inner_zero_grad)

    # 使用给定的设备、数据类型、模块信息和训练标志来测试梯度
    @modules(module_db)
    @modules(module_db, allowed_dtypes=[torch.double])
    def test_grad(self, device, dtype, module_info, training):
        # 调用 _test_gradients_helper 方法来辅助测试梯度
        self._test_gradients_helper(device, dtype, module_info, training, gradcheck)

    # 使用给定的设备、数据类型、模块信息和训练标志来测试二阶梯度
    @modules([m for m in module_db if m.supports_gradgrad],
             allowed_dtypes=[torch.double])
    def test_gradgrad(self, device, dtype, module_info, training):
        # 调用 _test_gradients_helper 方法来辅助测试二阶梯度
        self._test_gradients_helper(device, dtype, module_info, training, gradgradcheck)

    # 仅限于 CUDA 设备的测试装饰器
    @onlyCUDA
    # 关闭 TF32 模式以全精度计算
    @with_tf32_off  # Turn off TF32 to compute at full precision https://github.com/pytorch/pytorch/issues/86798
    # 容差覆盖装饰器，定义不同数据类型的容差值
    @toleranceOverride({torch.float32: tol(5e-2, 0),
                        torch.float64: tol(4e-4, 0)})
    @modules(module_db)
    @with_tf32_off
    @modules(module_db)
    # 测试训练模式和评估模式对每个模块的影响是否不同。用于验证 ModuleInfo 条目的标志是否正确。
    @modules(module_db, train_eval_mode=TrainEvalMode.train_only)
    # 定义测试方法，检查训练和评估模式是否不同
    def test_if_train_and_eval_modes_differ(self, device, dtype, module_info, training):
        # 获取模块的类对象
        module_cls = module_info.module_cls
        # 获取模块输入的函数，并传入设备、数据类型等参数
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)

        # 遍历模块的输入
        for module_input in module_inputs:
            # 如果 forward_input 为空，则跳过
            if module_input.forward_input is None:
                continue

            # === 实例化模块。===
            # 获取构造模块实例的参数和关键字参数
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            # 根据参数和关键字参数创建模块实例
            m = module_cls(*args, **kwargs)
            # 将模块实例移动到指定设备和数据类型
            m.to(device).to(dtype)
            # 设置模块的训练模式
            m.train(training)

            # 删除模块的 'training' 属性，查看 forward 方法是否仍然可用。
            delattr(m, 'training')

            # === 执行前向传播。===
            try:
                # 获取前向传播的参数和关键字参数
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                # 调用模块的前向传播方法
                m(*args, **kwargs)
            except AttributeError as e:
                # 如果捕获到属性错误，并且错误信息包含 "'training'" 字符串
                if "'training'" in str(e):
                    # 断言模块信息中 train_and_eval_differ 为真，
                    # 否则输出相应的错误信息提示
                    self.assertTrue(module_info.train_and_eval_differ,
                                    f"The ModuleInfo entry for {module_info.name} has "
                                    "train_and_eval_differ=False, but the training mode was found to "
                                    "affect the forward pass. Consider setting train_and_eval_differ=True "
                                    "for this ModuleInfo entry.")
                else:
                    # 如果不是以上情况，抛出原始异常
                    raise e


    @onlyCPU
    @modules(module_db)
    # 定义测试方法，用于初始化设备上下文
    def test_device_ctx_init(self, device, dtype, module_info, training):
        # 从 module_info 中获取模块类
        module_cls = module_info.module_cls
        # 使用 module_info 的模块输入函数获取模块输入
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        # 在 'meta' 设备上创建模块输入的元数据版本
        with torch.device('meta'):
            module_inputs_meta = module_info.module_inputs_func(module_info, device=None, dtype=dtype,
                                                                requires_grad=False, training=training)

        # 遍历模块输入和元数据版本的模块输入
        for module_input, module_input_meta in zip(module_inputs, module_inputs_meta):
            # 从模块输入中获取构造函数的参数和关键字参数
            c_args, c_kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            # 从元数据版本的模块输入中获取构造函数的参数和关键字参数
            c_args_meta, c_kwargs_meta = module_input_meta.constructor_input.args, module_input_meta.constructor_input.kwargs

            # 使用非 'meta' 设备创建模块实例
            m_cpu = module_cls(*c_args, **c_kwargs)

            # 在 'meta' 设备上创建元数据版本的模块实例
            with torch.device('meta'):
                m = module_cls(*c_args_meta, **c_kwargs_meta)

            # 比较模块参数和缓冲区，确保元数据和非 'meta' 设备的模块一致性
            for (p_meta, p_cpu) in chain(zip(m.parameters(), m_cpu.parameters()),
                                         zip(m.buffers(), m_cpu.buffers())):
                # 如果参数是延迟加载的，则跳过
                if torch.nn.parameter.is_lazy(p_meta):
                    continue
                # 断言元数据参数应标记为元数据
                self.assertTrue(p_meta.is_meta)
                # 使用 assert_metadata_eq 函数比较元数据和非 'meta' 设备的模块参数
                assert_metadata_eq(self.assertEqual, p_meta, p_cpu)


    # 为每个具有模块错误输入函数的模块执行错误测试
    @modules([module for module in module_db if module.module_error_inputs_func is not None])
    def test_errors(self, device, dtype, module_info, training):
        # 从 module_info 中获取模块类
        module_cls = module_info.module_cls
        # 使用 module_info 的模块错误输入函数获取错误输入
        error_inputs = module_info.module_error_inputs_func(module_info, device=device, dtype=dtype,
                                                            requires_grad=False, training=training)
        # 遍历每个错误输入
        for error_input in error_inputs:
            # 从错误输入中获取模块错误输入
            module_input = error_input.module_error_input
            # 从模块错误输入中获取构造函数的参数和关键字参数
            c_args, c_kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            # 根据错误类型执行不同的测试
            if error_input.error_on == ModuleErrorEnum.CONSTRUCTION_ERROR:
                # 测试模块在构造阶段是否引发了预期的异常
                with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                    m = module_cls(*c_args, **c_kwargs)
            elif error_input.error_on == ModuleErrorEnum.FORWARD_ERROR:
                # 创建模块实例
                m = module_cls(*c_args, **c_kwargs)
                # 从模块错误输入中获取前向输入的参数和关键字参数
                fw_args, fw_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                # 测试模块在前向传播阶段是否引发了预期的异常
                with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                    m(*fw_args, **fw_kwargs)
            else:
                # 抛出未实现的错误类型异常
                raise NotImplementedError(f"Unknown error type {error_input.error_on}")

    # 仅在 float32 类型上运行此测试，因为测试循环遍历所有数据类型
    @modules([module for module in module_db if not module.is_lazy], allowed_dtypes=[torch.float32])
    @parametrize('swap', [True, False])
    @parametrize('set_grad', [True, False])
    @wrapSwapTensorsTest()
    # 使用装饰器modules从module_db中选择非惰性模块，并指定允许的数据类型为torch.float32
    @modules([module for module in module_db if not module.is_lazy], allowed_dtypes=[torch.float32])
    # 参数化测试，参数为'swap'，取值为True和False
    @parametrize('swap', [True, False])
    # 使用wrapSwapTensorsTest装饰器封装测试方法
    @wrapSwapTensorsTest()
    # 定义测试方法test_to_empty，接受device, dtype, module_info, swap, training作为参数
    def test_to_empty(self, device, dtype, module_info, swap, training):
        # 从module_info中获取模块类
        module_cls = module_info.module_cls

        # 使用torch.device("meta")创建上下文，准备模块输入
        with torch.device("meta"):
            # 调用module_info.module_inputs_func生成模块输入，设备为None，数据类型为dtype，
            # requires_grad为False，训练模式为training
            module_inputs = module_info.module_inputs_func(module_info, device=None, dtype=dtype,
                                                           requires_grad=False, training=training)

        # 设置torch.__future__.set_swap_module_params_on_conversion(swap)
        torch.__future__.set_swap_module_params_on_conversion(swap)
        # 将设备转换为torch.device对象
        device_ = torch.device(device)

        # 遍历模块输入
        for module_input in module_inputs:
            # 获取构造器输入参数和关键字参数
            c_args, c_kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            # 使用torch.device("meta")创建上下文，实例化模块m
            with torch.device("meta"):
                m = module_cls(*c_args, **c_kwargs)

            # 记录操作前的参数id和_cdata
            p_ids_before = [id(p) for p in m.parameters()]
            p_cdatas_before = [p._cdata for p in m.parameters()]

            # 调用m.to_empty方法，将模块参数置为空
            m.to_empty(device=device_)

            # 断言所有参数均为torch.nn.Parameter类型
            self.assertTrue(all(isinstance(p, torch.nn.Parameter) for p in m.parameters()))
            # 断言所有参数的设备为device_
            self.assertTrue(all(p.device == device_ for p in m.parameters()))
            # 断言所有参数的数据类型为dtype
            self.assertTrue(all(p.dtype == dtype for p in m.parameters()))

            # 记录操作后的参数id和_cdata
            p_ids_after = [id(p) for p in m.parameters()]
            p_cdatas_after = [p._cdata for p in m.parameters()]

            if swap:
                # 如果swap为True，断言参数的id相同但_cdata不同，表示THPVariable的_cdata被交换
                self.assertTrue(all(a == b for a, b in zip(p_ids_before, p_ids_after)))
                self.assertTrue(all(a != b for a, b in zip(p_cdatas_before, p_cdatas_after)))
            else:
                # 如果swap为False，断言参数的id和_cdata均不同，表示创建了新的参数并分配给了模块
                self.assertTrue(all(a != b for a, b in zip(p_ids_before, p_ids_after)))
                self.assertTrue(all(a != b for a, b in zip(p_cdatas_before, p_cdatas_after)))
# 使用指定的 TestModule 实例化设备类型的测试，并将其添加到全局变量中
instantiate_device_type_tests(TestModule, globals(), allow_mps=True)

# 如果当前脚本被作为主程序运行，则执行测试函数
if __name__ == '__main__':
    run_tests()
```