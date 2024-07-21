# `.\pytorch\test\ao\sparsity\test_sparsity_utils.py`

```py
# Owner(s): ["module: unknown"]

# 导入必要的库
import logging

import torch
from torch.ao.pruning.sparsifier.utils import (
    fqn_to_module,
    get_arg_info_from_tensor_fqn,
    module_to_fqn,
)

# 导入用于测试的模型和工具类
from torch.testing._internal.common_quantization import (
    ConvBnReLUModel,
    ConvModel,
    FunctionalLinear,
    LinearAddModel,
    ManualEmbeddingBagLinear,
    SingleLayerLinearModel,
    TwoLayerLinearModel,
)
from torch.testing._internal.common_utils import TestCase

# 配置日志格式和日志级别
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# 定义模型列表，包含不同类型的模型类
model_list = [
    ConvModel,
    SingleLayerLinearModel,
    TwoLayerLinearModel,
    LinearAddModel,
    ConvBnReLUModel,
    ManualEmbeddingBagLinear,
    FunctionalLinear,
]

# 定义测试类，继承自TestCase
class TestSparsityUtilFunctions(TestCase):

    # 测试模块转全限定名函数 module_to_fqn
    def test_module_to_fqn(self):
        """
        Tests that module_to_fqn works as expected when compared to known good
        module.get_submodule(fqn) function
        """
        for model_class in model_list:
            model = model_class()
            # 获取模型中所有命名模块的列表，包括模型本身
            list_of_modules = [m for _, m in model.named_modules()] + [model]
            for module in list_of_modules:
                # 将模块转换为全限定名
                fqn = module_to_fqn(model, module)
                # 使用全限定名获取子模块并进行比较
                check_module = model.get_submodule(fqn)
                self.assertEqual(module, check_module)

    # 测试模块转全限定名函数 module_to_fqn，当传入无效的全限定名时返回None
    def test_module_to_fqn_fail(self):
        """
        Tests that module_to_fqn returns None when an fqn that doesn't
        correspond to a path to a node/tensor is given
        """
        for model_class in model_list:
            model = model_class()
            # 尝试将无关模块转换为全限定名
            fqn = module_to_fqn(model, torch.nn.Linear(3, 3))
            self.assertEqual(fqn, None)

    # 测试模块转全限定名函数 module_to_fqn，当模型和目标模块相同时返回空字符串''
    def test_module_to_fqn_root(self):
        """
        Tests that module_to_fqn returns '' when model and target module are the same
        """
        for model_class in model_list:
            model = model_class()
            # 将模型本身转换为全限定名
            fqn = module_to_fqn(model, model)
            self.assertEqual(fqn, "")

    # 测试全限定名转模块函数 fqn_to_module，确保其为 module_to_fqn 的逆操作
    def test_fqn_to_module(self):
        """
        Tests that fqn_to_module operates as inverse
        of module_to_fqn
        """
        for model_class in model_list:
            model = model_class()
            # 获取模型中所有命名模块的列表，包括模型本身
            list_of_modules = [m for _, m in model.named_modules()] + [model]
            for module in list_of_modules:
                # 将模块转换为全限定名
                fqn = module_to_fqn(model, module)
                # 将全限定名转换回模块，并进行比较
                check_module = fqn_to_module(model, fqn)
                self.assertEqual(module, check_module)

    # 测试全限定名转模块函数 fqn_to_module，当传入无效的全限定名时返回None
    def test_fqn_to_module_fail(self):
        """
        Tests that fqn_to_module returns None when it tries to
        find an fqn of a module outside the model
        """
        for model_class in model_list:
            model = model_class()
            # 尝试获取模型外部的模块
            fqn = "foo.bar.baz"
            check_module = fqn_to_module(model, fqn)
            self.assertEqual(check_module, None)
    # 测试 fqn_to_module 函数对张量有效，同时也适用于模型的所有参数。这通过识别带有张量的模块来测试。
    # 在此过程中，使用 module_to_fqn 函数来生成模块的完全限定名（FQN），然后将张量名与之结合成张量的完全限定名（tensor_fqn）。
    def test_fqn_to_module_for_tensors(self):
        """
        Tests that fqn_to_module works for tensors, actually all parameters
        of the model. This is tested by identifying a module with a tensor,
        and generating the tensor_fqn using module_to_fqn on the module +
        the name of the tensor.
        """
        # 遍历模型列表中的每个模型类
        for model_class in model_list:
            # 创建模型实例
            model = model_class()
            # 获取模型及其所有子模块的列表
            list_of_modules = [m for _, m in model.named_modules()] + [model]
            # 遍历模块列表中的每个模块
            for module in list_of_modules:
                # 使用 module_to_fqn 函数生成模块的完全限定名（module_fqn）
                module_fqn = module_to_fqn(model, module)
                # 遍历模块中的每个命名参数（张量）
                for tensor_name, tensor in module.named_parameters(recurse=False):
                    # 构建张量的完全限定名（tensor_fqn），处理张量在根模块上的情况
                    tensor_fqn = (
                        module_fqn + ("." if module_fqn != "" else "") + tensor_name
                    )
                    # 使用 fqn_to_module 函数检查张量的完全限定名对应的模块
                    check_tensor = fqn_to_module(model, tensor_fqn)
                    # 断言张量与检查得到的张量模块相等
                    self.assertEqual(tensor, check_tensor)

    # 测试 get_arg_info_from_tensor_fqn 函数对模型所有参数的工作情况。生成张量的完全限定名与 test_fqn_to_module_for_tensors 中的方式相同，
    # 然后将其与已知的父模块及张量名一起与模块的完全限定名（module_fqn）进行比较。
    def test_get_arg_info_from_tensor_fqn(self):
        """
        Tests that get_arg_info_from_tensor_fqn works for all parameters of the model.
        Generates a tensor_fqn in the same way as test_fqn_to_module_for_tensors and
        then compares with known (parent) module and tensor_name as well as module_fqn
        from module_to_fqn.
        """
        # 遍历模型列表中的每个模型类
        for model_class in model_list:
            # 创建模型实例
            model = model_class()
            # 获取模型及其所有子模块的列表
            list_of_modules = [m for _, m in model.named_modules()] + [model]
            # 遍历模块列表中的每个模块
            for module in list_of_modules:
                # 使用 module_to_fqn 函数生成模块的完全限定名（module_fqn）
                module_fqn = module_to_fqn(model, module)
                # 遍历模块中的每个命名参数（张量）
                for tensor_name, tensor in module.named_parameters(recurse=False):
                    # 构建张量的完全限定名（tensor_fqn）
                    tensor_fqn = (
                        module_fqn + ("." if module_fqn != "" else "") + tensor_name
                    )
                    # 使用 get_arg_info_from_tensor_fqn 函数获取张量完全限定名的参数信息（arg_info）
                    arg_info = get_arg_info_from_tensor_fqn(model, tensor_fqn)
                    # 断言获取的参数信息与预期相符
                    self.assertEqual(arg_info["module"], module)
                    self.assertEqual(arg_info["module_fqn"], module_fqn)
                    self.assertEqual(arg_info["tensor_name"], tensor_name)
                    self.assertEqual(arg_info["tensor_fqn"], tensor_fqn)

    # 测试 get_arg_info_from_tensor_fqn 函数对无效张量完全限定名输入的工作情况。预期输出中，模块应为 None。
    def test_get_arg_info_from_tensor_fqn_fail(self):
        """
        Tests that get_arg_info_from_tensor_fqn works as expected for invalid tensor_fqn
        inputs. The string outputs still work but the output module is expected to be None.
        """
        # 遍历模型列表中的每个模型类
        for model_class in model_list:
            # 创建模型实例
            model = model_class()
            # 设置一个无效的张量完全限定名（tensor_fqn）
            tensor_fqn = "foo.bar.baz"
            # 使用 get_arg_info_from_tensor_fqn 函数获取无效张量完全限定名的参数信息（arg_info）
            arg_info = get_arg_info_from_tensor_fqn(model, tensor_fqn)
            # 断言获取的参数信息与预期相符
            self.assertEqual(arg_info["module"], None)
            self.assertEqual(arg_info["module_fqn"], "foo.bar")
            self.assertEqual(arg_info["tensor_name"], "baz")
            self.assertEqual(arg_info["tensor_fqn"], "foo.bar.baz")
```