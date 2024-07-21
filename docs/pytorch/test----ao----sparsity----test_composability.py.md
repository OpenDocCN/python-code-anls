# `.\pytorch\test\ao\sparsity\test_composability.py`

```py
# Owner(s): ["module: unknown"]

# 导入必要的库
import logging

import torch
import torch.ao.quantization as tq
from torch import nn
from torch.ao import pruning
from torch.ao.pruning import fqn_to_module
from torch.ao.quantization.quantize_fx import (
    convert_fx,
    convert_to_reference_fx,
    prepare_fx,
    prepare_qat_fx,
)
from torch.testing._internal.common_utils import TestCase

# 设置日志格式和日志级别
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# 定义稀疏化的默认配置
sparse_defaults = {
    "sparsity_level": 0.8,
    "sparse_block_shape": (1, 4),
    "zeros_per_block": 4,
}

# 定义函数：获取模型、稀疏化器和稀疏化配置
def _get_model_and_sparsifier_and_sparse_config(qconfig=None):
    # 定义一个包含量化和反量化操作的神经网络模型
    model = nn.Sequential(
        nn.Linear(4, 4),  # 0
        nn.ReLU(),
        nn.Linear(4, 4),  # 2
        nn.ReLU(),
        tq.QuantStub(),
        nn.Linear(4, 4),  # 5
        nn.ReLU(),
        tq.DeQuantStub(),
    )
    # 如果提供了量化配置，则将配置应用于相应的量化和反量化模块
    if qconfig:
        model[4].qconfig = qconfig
        model[5].qconfig = qconfig

    # 创建一个权重规范稀疏化器
    sparsifier = pruning.WeightNormSparsifier(**sparse_defaults)

    # 定义稀疏化配置列表，包含稀疏张量的详细信息
    sparse_config = [
        {
            "tensor_fqn": "5.weight",
            "sparsity_level": 0.7,
            "sparse_block_shape": (1, 4),
            "zeros_per_block": 4,
        },
        {"tensor_fqn": "0.weight"},
    ]
    return model, sparsifier, sparse_config


# 定义函数：执行压缩掩码计算和转换操作
def _squash_mask_calibrate_and_convert(model, sparsifier, input):
    # 执行稀疏化器的步骤，准备压缩掩码
    sparsifier.step()
    sparsifier.squash_mask()
    # 使用输入数据推断模型
    model(input)
    # 将模型转换为量化模型（inplace=True表示在原地操作）
    tq.convert(model, inplace=True)


# 定义函数：计算张量的稀疏度
def _calculate_sparsity(tensor):
    return ((tensor == 0).sum() / tensor.numel()).item()


# 这一系列测试用例用于验证稀疏性和量化的组合目标。特别是，验证在不同顺序下执行量化和稀疏模型操作
# 是否会引发问题。
class TestComposability(TestCase):
    # 这个测试用例验证在执行稀疏准备之前执行量化准备是否会导致问题，并验证是否插入了正确的观察器，
    # 以及量化模型是否按预期工作。
    # 定义一个测试方法，用于测试在稀疏化准备之前是否进行了量化准备
    def test_q_prep_before_s_prep(self):
        (
            mod,  # 获取模型
            sparsifier,  # 获取稀疏化器
            sparse_config,  # 获取稀疏配置
        ) = _get_model_and_sparsifier_and_sparse_config(
            tq.get_default_qconfig("fbgemm")  # 获取默认的量化配置
        )

        tq.prepare(mod, inplace=True)  # 在模型上进行量化准备，使用原地操作
        sparsifier.prepare(mod, config=sparse_config)  # 在模型上进行稀疏化准备，使用给定的稀疏配置

        # 检查是否正确地向模块添加了参数化
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))
        # 检查是否正确地插入了观察器
        self.assertTrue(hasattr(mod[5], "activation_post_process"))

        _squash_mask_calibrate_and_convert(mod, sparsifier, torch.randn(1, 4, 4, 4))

        # 检查最终的模块是否是预期的量化模块，并且模型能够成功运行
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

    # 此测试检查在进行量化准备之前是否进行稀疏化准备会导致任何问题。
    # 特别是，以前的量化流程无法匹配稀疏化准备后的模块名称（添加参数化会更改模块类名称），
    # 这将导致这些参数化模块无法被量化。此测试验证了针对此问题的修复是否成功。
    def test_s_prep_before_q_prep(self):
        (
            mod,  # 获取模型
            sparsifier,  # 获取稀疏化器
            sparse_config,  # 获取稀疏配置
        ) = _get_model_and_sparsifier_and_sparse_config(
            tq.get_default_qconfig("fbgemm")  # 获取默认的量化配置
        )

        sparsifier.prepare(mod, config=sparse_config)  # 在模型上进行稀疏化准备，使用给定的稀疏配置
        tq.prepare(mod, inplace=True)  # 在模型上进行量化准备，使用原地操作

        # 检查是否正确地向模块添加了参数化，并且在准备过程中没有丢失任何参数化
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))

        # 检查是否正确地插入了观察器，并且匹配成功
        self.assertTrue(hasattr(mod[5], "activation_post_process"))

        _squash_mask_calibrate_and_convert(mod, sparsifier, torch.randn(1, 4, 4, 4))

        # 检查最终的模块是否是预期的量化模块，并且模型能够成功运行
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

    # 如果稀疏化的模块尚未经历最终的压缩掩码操作，那么可能发生前述 test_s_prep_before_q_prep 中提到的问题。
    # 此测试验证了转换流程的修复避免了这个问题，并且生成的量化模块使用了稀疏版本的权重值。
    # 定义测试方法，用于验证在没有压缩掩码的情况下进行转换
    def test_convert_without_squash_mask(self):
        # 调用辅助函数获取模型、稀疏器和稀疏配置
        (
            mod,
            sparsifier,
            sparse_config,
        ) = _get_model_and_sparsifier_and_sparse_config(
            tq.get_default_qconfig("fbgemm")
        )

        # 使用稀疏器准备模型，应用稀疏配置
        sparsifier.prepare(mod, config=sparse_config)
        # 使用量化器准备模型，直接修改原对象
        tq.prepare(mod, inplace=True)

        # 检查是否正确为模块添加了参数化，并且在准备过程中没有丢失
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))

        # 检查是否成功插入了正确的观察器，并且匹配操作成功完成
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        # 稀疏器进行一步稀疏化操作
        sparsifier.step()
        # 计算第5个模块权重的稀疏度水平
        sparsity_level = _calculate_sparsity(mod[5].weight)
        # 模拟输入数据，执行模型
        mod(torch.randn(1, 4, 4, 4))
        # 转换模型为量化模型，直接修改原对象
        tq.convert(mod, inplace=True)

        # 检查最终模块是否为预期的量化模块，并且模型运行正常
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

        # 检查模块是否确实被稀疏化
        cur_sparsity = _calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])

    # 这个测试检查在融合之前进行稀疏化是否会导致任何问题。担心的是稀疏器与需要稀疏化的模块之间的链接会被中断。
    def test_s_prep_before_fusion(self):
        # 调用辅助函数获取模型、稀疏器和稀疏配置
        (
            mod,
            sparsifier,
            sparse_config,
        ) = _get_model_and_sparsifier_and_sparse_config(
            tq.get_default_qconfig("fbgemm")
        )
        # 使用稀疏器准备模型，应用稀疏配置
        sparsifier.prepare(mod, config=sparse_config)
        # 融合指定的模块，原地修改模型
        tq.fuse_modules(mod, [["5", "6"]], inplace=True)
        # 设置第5个模块的量化配置为默认的fbgemm配置
        mod[5].qconfig = tq.get_default_qconfig("fbgemm")
        # 使用量化器准备模型，直接修改原对象
        tq.prepare(mod, inplace=True)

        # 检查是否正确为模块添加了参数化，并且在准备或融合过程中没有丢失
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5][0], "parametrizations"))

        # 检查是否成功插入了正确的观察器，并且匹配操作成功完成
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        # 调用内部函数，执行压缩掩码校准和转换操作，模拟输入数据
        _squash_mask_calibrate_and_convert(mod, sparsifier, torch.randn(1, 4, 4, 4))

        # 检查最终模块是否为预期的量化ReLU模块，并且模型运行正常
        self.assertTrue(isinstance(mod[5], torch.ao.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))
    # 测试在执行稀疏准备之前进行融合是否会引发问题。
    # 主要的担忧是融合可能会破坏稀疏配置中模块的链接。
    def test_fusion_before_s_prep(self):
        # 调用辅助函数获取模型、稀疏器和稀疏配置
        (
            mod,
            sparsifier,
            _,
        ) = _get_model_and_sparsifier_and_sparse_config(
            tq.get_default_qconfig("fbgemm")
        )
        # 在模型中进行模块融合，此操作会在原地修改模型
        tq.fuse_modules(mod, [["5", "6"]], inplace=True)
    
        # 如果正确设置全限定名（fully qualified name），尽管融合操作破坏了稀疏配置，但稀疏器仍能正常工作
        sparse_config = [
            {
                "tensor_fqn": "5.0.weight",
                "sparsity_level": 0.7,
                "sparse_block_shape": (1, 4),
                "zeros_per_block": 4,
            },
            {"tensor_fqn": "0.weight"},
        ]
    
        # 使用稀疏器准备模型，传入稀疏配置
        sparsifier.prepare(mod, config=sparse_config)
        # 设置模型的量化配置
        mod[5].qconfig = tq.get_default_qconfig("fbgemm")
        # 对模型进行量化准备，原地修改模型
        tq.prepare(mod, inplace=True)
    
        # 检查正确的模块是否已添加参数化，并确保在准备过程中没有丢失模块
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5][0], "parametrizations"))
    
        # 检查是否成功插入了正确的观察器，并且匹配成功
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        # 稀疏器执行稀疏化的一步操作
        sparsifier.step()
        # 计算当前稀疏度水平
        sparsity_level = _calculate_sparsity(mod[5][0].weight)
        # 使用模型进行前向传播计算
        mod(torch.randn(1, 4, 4, 4))
        # 将模型转换为QAT格式，原地修改模型
        tq.convert(mod, inplace=True)
    
        # 检查最终模块是否是预期的量化ReLU线性模块，并且模型能够正常运行
        self.assertTrue(isinstance(mod[5], torch.ao.nn.intrinsic.quantized.LinearReLU))
        # 检查模型输出的形状是否符合预期
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))
    
        # 检查模块是否实际上已经稀疏化
        cur_sparsity = _calculate_sparsity(mod[5]._weight_bias()[0])
        # 断言当前稀疏度大于等于预期的稀疏度水平
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        # 断言稀疏度水平大于等于稀疏配置中的设定值
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        # 断言当前稀疏度大于等于稀疏配置中的设定值
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])
    
    # 测试在执行QAT准备之前执行稀疏准备是否会引发问题。
    # 主要的担忧是QAT准备阶段可能无法识别参数化模块，并且QAT转换步骤可能会移除模块的参数化。
    # 定义测试方法，用于测试在稀疏准备之前进行量化感知训练准备是否会导致问题
    def test_qat_prep_before_s_prep(self):
        # 从 _get_model_and_sparsifier_and_sparse_config 函数中获取模型、稀疏化器和稀疏配置
        mod, sparsifier, _ = _get_model_and_sparsifier_and_sparse_config(
            tq.get_default_qat_qconfig("fbgemm")
        )
        # 在模型上进行量化感知训练准备，直接修改原模型
        tq.prepare_qat(mod, inplace=True)
    
        # 需要在新模块上设置稀疏配置
        sparse_config = [
            {
                "tensor_fqn": "5.weight",
                "sparsity_level": 0.7,
                "sparse_block_shape": (1, 4),
                "zeros_per_block": 4,
            },
            {"tensor_fqn": "0.weight"},
        ]
        # 使用稀疏化器对模型进行稀疏准备，配置为 sparse_config
        sparsifier.prepare(mod, config=sparse_config)
    
        # 检查是否正确添加了参数化到正确模块，并且在量化感知训练准备期间没有丢失
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))
    
        # 检查是否成功插入了正确的观察者，并且匹配成功
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        self.assertTrue(isinstance(mod[5], torch.ao.nn.qat.Linear))
    
        # 使用 _squash_mask_calibrate_and_convert 函数压缩、校准和转换模型
        _squash_mask_calibrate_and_convert(mod, sparsifier, torch.randn(1, 4, 4, 4))
    
        # 检查最终模块是否为预期的量化模块，并且模型运行正常
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))
    
        # 检查模块是否实际上被稀疏化
        cur_sparsity = _calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])
# 检查模型是否具有激活后处理模块
def _module_has_activation_post_process(model, fqn_of_module):
    # 遍历模型的计算图中的每个节点
    for node in model.graph.nodes:
        # 查找名称中包含 "activation_post_process" 的节点
        if "activation_post_process" in node.name:
            # 如果找到了激活后处理节点，检查其参数列表第一个参数是否为目标模块的全限定名
            if node.args[0].target == fqn_of_module:
                # 如果匹配到目标模块，返回True
                return True
    # 如果未找到匹配的激活后处理模块，返回False
    return False


# 测试类，用于验证量化和稀疏化流程的组合性
class TestFxComposability(TestCase):
    r"""This series of tests checks that various steps of the quantization and sparsity flow
    compose cleanly despite variation in sequencing.
    """
    # 定义一个测试方法，用于验证 prepare_fx -> sparse prepare -> convert_fx 的顺序是否能够无问题地组合，并且最终结果在不调用 squash mask 的情况下实现稀疏化。
    # 同时也测试了 prepare_fx 过程中自动融合的功能。
    def test_q_prep_fx_before_s_prep(self):
        r"""
        This test checks that the ordering of prepare_fx -> sparse prepare -> convert_fx
        compose cleanly without issue and that the final result is sparsified without
        having to call squash mask between sparse prepare and convert_fx. This also tests the
        automatic fusion that occurs during prepare_fx.
        """
        # 调用辅助函数 _get_model_and_sparsifier_and_sparse_config() 获取模型、稀疏化器及稀疏配置信息
        (
            mod,
            sparsifier,
            _,
        ) = _get_model_and_sparsifier_and_sparse_config()

        # 创建一个示例张量
        example = torch.randn(1, 4, 4, 4)
        # 获取默认的量化配置 "fbgemm"
        qconfig = tq.get_default_qconfig("fbgemm")
        # 设置量化配置映射，将模块 "4" 和 "5" 的名称与 qconfig 关联起来
        qconfig_mapping = (
            tq.QConfigMapping()
            .set_module_name("4", qconfig)
            .set_module_name("5", qconfig)
        )

        # 对模型进行准备，应用量化配置映射，并传入示例张量
        mod = prepare_fx(mod, qconfig_mapping, (example,))

        # 自动融合可能导致功能出现问题，但如果正确使用完全限定名称（fqn），它仍然可以工作
        # 在 sparse prepare 和 convert_fx 之间不需要调用 squash mask
        # 准备稀疏化配置
        sparse_config = [
            {
                "tensor_fqn": "5.0.weight",
                "sparsity_level": 0.7,
                "sparse_block_shape": (1, 4),
                "zeros_per_block": 4,
            },
            {"tensor_fqn": "0.0.weight"},
        ]
        # 使用稀疏化器 sparsifier 对模型进行稀疏准备，传入稀疏配置
        sparsifier.prepare(mod, config=sparse_config)

        # 检查正确的模块是否已添加了参数化，并且在 prepare 过程中没有丢失
        self.assertTrue(hasattr(fqn_to_module(mod, "0.0"), "parametrizations"))
        self.assertTrue(hasattr(fqn_to_module(mod, "5.0"), "parametrizations"))

        # 检查是否成功插入了正确的观察器，并且匹配操作已成功完成
        self.assertTrue(_module_has_activation_post_process(mod, "5"))
        
        # 执行稀疏化的下一步操作
        sparsifier.step()
        
        # 计算当前稀疏度水平
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, "5.0.weight"))
        
        # 在示例张量上执行模型
        mod(example)
        
        # 将模型转换为最终形式
        mod = convert_fx(mod)

        # 检查最终的模块是否是预期的量化模块，并且模型能够正常运行
        self.assertTrue(
            isinstance(
                fqn_to_module(mod, "5"), torch.ao.nn.intrinsic.quantized.LinearReLU
            )
        )
        # 检查模型输出的形状是否符合预期
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))

        # 检查模型是否确实被稀疏化
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, "5")._weight_bias()[0])
        # 断言当前稀疏度大于等于预期稀疏度水平
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        # 断言预期稀疏度水平大于等于稀疏配置中的设定值
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        # 断言当前稀疏度大于等于稀疏配置中的设定值
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])
    # 定义测试函数 test_q_prep_fx_s_prep_ref_conv，用于验证以下顺序的操作能够顺利组合，而且在调用 convert_to_reference_fx 之前就进行了稀疏化处理：prepare_fx -> sparse prepare -> convert_to_reference_fx
    def test_q_prep_fx_s_prep_ref_conv(self):
        r"""
        This checks that the ordering: prepare_fx -> sparse prepare -> convert_to_reference_fx
        compose cleanly without issue and that the final result is sparsified without
        having to call squash mask before convert_to_reference_fx.
        """
        # 从辅助函数 _get_model_and_sparsifier_and_sparse_config() 中获取模型 mod、稀疏化器 sparsifier，以及其他需要的配置信息
        (mod, sparsifier, _) = _get_model_and_sparsifier_and_sparse_config()

        # 创建一个例子张量 example，形状为 (1, 4, 4, 4)
        example = torch.randn(1, 4, 4, 4)
        # 获取默认的量化配置 "fbgemm"
        qconfig = tq.get_default_qconfig("fbgemm")
        # 创建一个 QConfigMapping 对象，设置模块 "4" 和 "5" 的量化配置为 qconfig
        qconfig_mapping = (
            tq.QConfigMapping()
            .set_module_name("4", qconfig)
            .set_module_name("5", qconfig)
        )

        # 使用 prepare_fx 函数对模型 mod 进行准备，使用 qconfig_mapping 和 example 作为参数
        mod = prepare_fx(mod, qconfig_mapping, (example,))

        # 定义一个稀疏配置 sparse_config 列表，包含两个字典元素，用于指定需要稀疏化的张量和相应的配置
        sparse_config = [
            {
                "tensor_fqn": "5.0.weight",
                "sparsity_level": 0.7,
                "sparse_block_shape": (1, 4),
                "zeros_per_block": 4,
            },
            {"tensor_fqn": "0.0.weight"},
        ]
        # 使用 sparsifier 的 prepare 方法，对模型 mod 进行稀疏化处理，传入 sparse_config 作为配置参数
        sparsifier.prepare(mod, config=sparse_config)

        # 检查模块 "0.0" 和 "5.0" 是否成功添加了 parametrizations 属性
        self.assertTrue(hasattr(fqn_to_module(mod, "0.0"), "parametrizations"))
        self.assertTrue(hasattr(fqn_to_module(mod, "5.0"), "parametrizations"))

        # 检查是否成功插入了正确的观察器，并且匹配操作成功进行
        self.assertTrue(_module_has_activation_post_process(mod, "5"))
        
        # 执行稀疏化器的 step 方法，推进稀疏化过程
        sparsifier.step()
        # 计算 "5.0.weight" 张量的稀疏度
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, "5.0.weight"))
        # 使用 example 执行模型 mod
        mod(example)
        # 将模型转换为参考引用模型，即调用 convert_to_reference_fx 函数
        mod = convert_to_reference_fx(mod)

        # 检查最终的模块是否为预期的量化模块类型，并且模型能够正常运行
        self.assertTrue(
            isinstance(fqn_to_module(mod, "5"), torch.ao.nn.intrinsic.LinearReLU)
        )
        # 检查模型输出形状是否符合预期
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))
        # 检查模块 "5.0" 是否被成功转换为参考引用模型类型
        self.assertTrue(
            isinstance(
                fqn_to_module(mod, "5.0"), torch.ao.nn.quantized.reference.Linear
            )
        )

        # 检查模型是否实际上被成功稀疏化
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, "5.0.weight"))
        # 检查当前稀疏度是否大于等于之前计算的稀疏度
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        # 检查稀疏度是否大于等于 sparse_config 中指定的稀疏度
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        # 检查当前稀疏度是否大于等于 sparse_config 中指定的稀疏度
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])
    def test_s_prep_before_q_prep_fx(self):
        r"""
        This test checks that the ordering of sparse prepare -> prepare_fx -> convert_fx
        compose cleanly without issue and that the final result is sparsified without
        having to call squash mask before convert_fx.
        """
        # 获取模型、稀疏化器和稀疏配置
        (mod,
         sparsifier,
         sparse_config) = _get_model_and_sparsifier_and_sparse_config()
        
        # 对模型进行稀疏化准备操作
        sparsifier.prepare(mod, config=sparse_config)

        # 创建一个示例输入张量
        example = torch.randn(1, 4, 4, 4)
        
        # 获取默认的量化配置
        qconfig = tq.get_default_qconfig("fbgemm")
        
        # 配置量化映射
        qconfig_mapping = (
            tq.QConfigMapping()
            .set_module_name("4", qconfig)
            .set_module_name("5", qconfig)
        )
        
        # 使用量化映射对模型进行准备
        mod = prepare_fx(mod, qconfig_mapping, (example,))

        # 检查正确的模块是否已添加参数化，并且在准备过程中没有丢失
        self.assertTrue(hasattr(fqn_to_module(mod, "0.0"), "parametrizations"))
        self.assertTrue(hasattr(fqn_to_module(mod, "5.0"), "parametrizations"))

        # 检查正确的观察器是否已插入，并且匹配成功
        self.assertTrue(_module_has_activation_post_process(mod, "5"))
        
        # 执行稀疏化器的步骤
        sparsifier.step()
        
        # 计算稀疏性水平
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, "5.0.weight"))
        
        # 使用示例输入张量对模型进行前向传播
        mod(example)
        
        # 将模型转换为量化后的模型
        mod = convert_fx(mod)

        # 检查最终的模块是否为预期的量化模块，并且模型能够正常运行
        self.assertTrue(
            isinstance(
                fqn_to_module(mod, "5"), torch.ao.nn.intrinsic.quantized.LinearReLU
            )
        )
        
        # 检查模型输出的形状是否正确
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))

        # 检查模型是否真正稀疏化
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, "5")._weight_bias()[0])
        
        # 断言当前稀疏性水平大于等于预期的稀疏性水平
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        
        # 断言预期的稀疏性水平大于等于稀疏配置中的稀疏性水平
        self.assertGreaterAlmostEqual(sparsity_level, sparse_config[0]["sparsity_level"])
        
        # 断言当前稀疏性水平大于等于稀疏配置中的稀疏性水平
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])
    def test_s_prep_before_qat_prep_fx(self):
        r"""
        This test checks that the ordering of sparse prepare -> prepare_qat_fx -> convert_fx
        compose cleanly without issue and that the final result is sparsified without
        having to call squash mask before convert_fx.
        """
        # 获取模型、稀疏化器和稀疏配置
        (
            mod,
            sparsifier,
            sparse_config,
        ) = _get_model_and_sparsifier_and_sparse_config()
        
        # 对模型进行稀疏化准备
        sparsifier.prepare(mod, config=sparse_config)

        # 创建示例输入
        example = torch.randn(1, 4, 4, 4)
        
        # 获取默认的量化训练QConfig
        qconfig = tq.get_default_qat_qconfig("fbgemm")
        
        # 设置模块名称到QConfig的映射
        qconfig_mapping = (
            tq.QConfigMapping()
            .set_module_name("4", qconfig)
            .set_module_name("5", qconfig)
        )
        
        # 准备量化训练FX模型
        mod = prepare_qat_fx(mod, qconfig_mapping, (example,))

        # 检查正确的模块是否添加了参数化，并且在准备过程中没有丢失
        self.assertTrue(hasattr(fqn_to_module(mod, "0.0"), "parametrizations"))
        self.assertTrue(hasattr(fqn_to_module(mod, "5"), "parametrizations"))
        self.assertTrue(
            isinstance(fqn_to_module(mod, "5"), torch.ao.nn.intrinsic.qat.LinearReLU)
        )

        # 检查是否成功插入了正确的观察器，并且匹配成功
        self.assertTrue(_module_has_activation_post_process(mod, "5"))
        
        # 执行稀疏化的下一步操作
        sparsifier.step()
        
        # 计算当前稀疏度水平
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, "5.weight"))
        
        # 运行模型
        mod(example)
        
        # 转换为量化后模型
        mod = convert_fx(mod)

        # 检查最终模块是否是预期的量化模块，并且模型能够运行
        self.assertTrue(
            isinstance(
                fqn_to_module(mod, "5"), torch.ao.nn.intrinsic.quantized.LinearReLU
            )
        )
        
        # 检查模型输出形状是否正确
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))

        # 检查模型是否真正被稀疏化
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, "5")._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])
    def test_s_prep_q_prep_fx_ref(self):
        r"""
        This checks that the ordering: sparse prepare -> prepare_fx -> convert_to_reference_fx
        compose cleanly without issue and that the final result is sparsified without
        having to call squash mask before convert_to_reference_fx.
        """
        (
            mod,
            sparsifier,
            sparse_config,
        ) = _get_model_and_sparsifier_and_sparse_config()
        
        # 获得模型、稀疏化器和稀疏配置

        sparsifier.prepare(mod, config=sparse_config)
        # 使用稀疏配置对模型进行稀疏化准备

        example = torch.randn(1, 4, 4, 4)
        qconfig = tq.get_default_qconfig("fbgemm")
        qconfig_mapping = (
            tq.QConfigMapping()
            .set_module_name("4", qconfig)
            .set_module_name("5", qconfig)
        )
        mod = prepare_fx(mod, qconfig_mapping, (example,))
        
        # 使用指定的量化配置对模型进行准备

        # check that correct modules had parametrizations added and
        # that none were lost during prepare
        self.assertTrue(hasattr(fqn_to_module(mod, "0.0"), "parametrizations"))
        self.assertTrue(hasattr(fqn_to_module(mod, "5.0"), "parametrizations"))
        
        # 检查是否正确地为模块添加了参数化，并且在准备过程中没有丢失任何模块

        # check that correct observers were inserted and that matching
        # occurred successfully
        self.assertTrue(_module_has_activation_post_process(mod, "5"))
        
        # 检查是否正确地插入了观察器，并且匹配操作成功

        sparsifier.step()
        # 稀疏化器进行一步操作

        sparsity_level = _calculate_sparsity(fqn_to_module(mod, "5.0.weight"))
        # 计算当前稀疏水平

        mod(example)
        mod = convert_to_reference_fx(mod)
        # 将模型转换为参考量化模型

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(
            isinstance(fqn_to_module(mod, "5"), torch.ao.nn.intrinsic.LinearReLU)
        )
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))
        self.assertTrue(
            isinstance(
                fqn_to_module(mod, "5.0"), torch.ao.nn.quantized.reference.Linear
            )
        )
        
        # 检查最终模块是否符合预期的量化模块，并且模型能够运行

        # check that module was actually sparsified
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, "5.0.weight"))
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])
        
        # 检查模块是否实际上已经稀疏化
```