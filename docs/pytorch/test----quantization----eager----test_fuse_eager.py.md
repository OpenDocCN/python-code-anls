# `.\pytorch\test\quantization\eager\test_fuse_eager.py`

```py
# Owner(s): ["oncall: quantization"]

# 导入必要的模块和库
import copy  # 导入深拷贝函数
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
import torch.ao.nn.quantized as nnq  # 导入量化后的神经网络模块
import torch.ao.nn.intrinsic as nni  # 导入内部操作的神经网络模块
import torch.ao.nn.intrinsic.quantized as nniq  # 导入量化内部操作的神经网络模块
import torch.ao.nn.intrinsic.qat as nniqat  # 导入量化训练的神经网络模块
from torch.ao.quantization import (  # 导入量化相关函数和类
    quantize,
    prepare,
    convert,
    prepare_qat,
    quantize_qat,
    fuse_modules,
    fuse_modules_qat,
    QConfig,
    default_qconfig,
    default_qat_qconfig,
)

from torch.testing._internal.common_quantization import (  # 导入量化测试相关函数和类
    QuantizationTestCase,
    ModelForFusion,
    ModelWithSequentialFusion,
    ModelForLinearBNFusion,
    ModelForFusionWithBias,
    ModelForConvTransposeBNFusion,
    SingleLayerLinearModel,
    test_only_eval_fn,
    test_only_train_fn,
    skipIfNoFBGEMM,
)

from torch.testing._internal.common_quantized import (  # 导入量化测试相关函数
    override_quantized_engine,
    supported_qengines,
)

# 使用 FBGEMM 引擎时跳过测试
@skipIfNoFBGEMM
class TestFuseEager(QuantizationTestCase):
    # 测试类，继承自 QuantizationTestCase
    # 定义测试方法，用于测试融合模块的训练过程
    def test_fuse_module_train(self):
        # 创建一个使用默认量化配置进行训练的模型
        model = ModelForFusion(default_qat_qconfig).train()
        # 逐步测试模块融合过程
        model = fuse_modules_qat(model, ['conv1', 'bn1', 'relu1'])
        model = fuse_modules_qat(model, ['sub1.conv', 'sub1.bn'])
        
        # 断言第一层融合的结果类型为 ConvBnReLU2d 类型
        self.assertEqual(type(model.conv1), nni.ConvBnReLU2d,
                         msg="Fused Conv + BN + Relu first layer")
        # 断言第一层融合的结果中跳过了 BN 层，因此为 Identity 类型
        self.assertEqual(type(model.bn1), torch.nn.Identity,
                         msg="Fused Conv + BN + Relu (skipped BN)")
        # 断言第一层融合的结果中跳过了 Relu 层，因此为 Identity 类型
        self.assertEqual(type(model.relu1), torch.nn.Identity,
                         msg="Fused Conv + BN + Relu (skipped Relu)")

        # 断言子模块 sub1 的融合结果类型为 ConvBn2d
        self.assertEqual(type(model.sub1.conv), nni.ConvBn2d,
                         msg="Fused submodule Conv + BN")
        # 断言子模块 sub1 的融合结果中跳过了 BN 层，因此为 Identity 类型
        self.assertEqual(type(model.sub1.bn), torch.nn.Identity,
                         msg="Fused submodule Conv + BN (skipped BN)")
        # 断言子模块 sub2 的卷积层没有融合，因此类型为 Conv2d
        self.assertEqual(type(model.sub2.conv), torch.nn.Conv2d,
                         msg="Non-fused submodule Conv")
        # 断言子模块 sub2 的激活函数没有融合，因此类型为 ReLU
        self.assertEqual(type(model.sub2.relu), torch.nn.ReLU,
                         msg="Non-fused submodule ReLU")

        # 准备量化感知训练 (QAT) 模型
        model = prepare_qat(model)
        # 检查模型的观察者
        self.checkObservers(model)

        # 定义检查 QAT 模型的方法
        def checkQAT(model):
            self.assertEqual(type(model.conv1), nniqat.ConvBnReLU2d)
            self.assertEqual(type(model.bn1), nn.Identity)
            self.assertEqual(type(model.relu1), nn.Identity)
            self.assertEqual(type(model.sub1.conv), nniqat.ConvBn2d)
            self.assertEqual(type(model.sub1.bn), nn.Identity)
            self.assertEqual(type(model.sub2.conv), nn.Conv2d)
            self.assertEqual(type(model.sub2.relu), nn.ReLU)

        # 调用检查 QAT 模型的方法
        checkQAT(model)
        # 使用训练数据进行测试
        test_only_train_fn(model, self.img_data_1d_train)
        # 将模型转换为量化模型
        model = convert(model)

        # 定义检查量化模型的方法
        def checkQuantized(model):
            self.assertEqual(type(model.conv1), nniq.ConvReLU2d)
            self.assertEqual(type(model.bn1), nn.Identity)
            self.assertEqual(type(model.relu1), nn.Identity)
            self.assertEqual(type(model.sub1.conv), nnq.Conv2d)
            self.assertEqual(type(model.sub1.bn), nn.Identity)
            self.assertEqual(type(model.sub2.conv), nn.Conv2d)
            self.assertEqual(type(model.sub2.relu), nn.ReLU)
            # 使用评估数据进行测试
            test_only_eval_fn(model, self.img_data_1d)
            # 检查模型是否没有配置量化信息
            self.checkNoQconfig(model)

        # 当运行到特定的 RuntimeError 时，断言期望的异常信息
        with self.assertRaisesRegex(RuntimeError, "Could not run 'aten::native_batch_norm' with arguments from the 'QuantizedCPU'"):
            checkQuantized(model)

        # 重新创建一个使用默认量化配置进行训练的模型
        model = ModelForFusion(default_qat_qconfig).train()
        # 使用多组模块进行融合
        model = fuse_modules_qat(
            model,
            [['conv1', 'bn1', 'relu1'],
             ['sub1.conv', 'sub1.bn']])
        # 对融合后的模型进行量化感知训练
        model = quantize_qat(model, test_only_train_fn, [self.img_data_1d_train])
        # 当运行到特定的 RuntimeError 时，断言期望的异常信息
        with self.assertRaisesRegex(RuntimeError, "Could not run 'aten::native_batch_norm' with arguments from the 'QuantizedCPU'"):
            checkQuantized(model)
    # 定义一个测试方法，用于验证序列模型的融合效果
    def test_fusion_sequential_model_eval(self):
        # 遍历支持的量化引擎列表
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖默认引擎设置
            with override_quantized_engine(qengine):
                # 创建一个包含序列融合的模型，并将其设置为评估模式
                model = ModelWithSequentialFusion().eval()
                # 将模型转换为浮点数格式
                model.to(torch.float)
                # 融合模块：将指定的模块序列进行融合
                fuse_modules(
                    model,
                    [['conv1', 'relu1'],
                     ['features.0.0', 'features.0.1', 'features.0.2'],
                     ['features.1.0', 'features.1.1', 'features.1.2'],
                     ['features.2.0', 'features.2.1', 'features.2.2'],
                     ['classifier.0', 'classifier.1']],
                    inplace=True)
                # 断言检查融合后的模块类型是否符合预期
                self.assertEqual(type(model.conv1), nni.ConvReLU2d,
                                 msg="Fused Conv + Relu: nni.ConvReLU2d")
                self.assertEqual(type(model.conv1[0]), nn.Conv2d,
                                 msg="Fused Conv + Relu: Conv2d")
                self.assertEqual(type(model.conv1[1]), nn.ReLU,
                                 msg="Fused Conv + Relu: Relu")
                self.assertEqual(type(model.relu1), nn.Identity,
                                 msg="Fused Conv + Relu: Identity")
                # 遍历特征提取部分的每个子模块，检查其类型
                for i in range(3):
                    self.assertEqual(type(model.features[i][0]), nni.ConvReLU2d,
                                     msg="Fused submodule Conv + folded BN")
                    self.assertEqual(type(model.features[i][1]), nn.Identity,
                                     msg="Fused submodule (skipped BN)")
                    self.assertEqual(type(model.features[i][2]), nn.Identity,
                                     msg="Non-fused submodule Conv")
                # 检查分类器部分的量化后模块类型
                self.assertEqual(type(model.classifier[0]), nni.LinearReLU)
                self.assertEqual(type(model.classifier[1]), nn.Identity)
                # 配置模型的量化配置为当前引擎的默认量化配置
                model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                # 准备模型以进行量化
                prepare(model, inplace=True)
                # 检查模型的观察器状态
                self.checkObservers(model)
                # 对模型进行推理，以便观察其量化效果
                model(self.img_data_2d[0][0])
                # 将模型转换为量化表示
                convert(model, inplace=True)
                # 再次对模型进行推理，确认量化后的效果
                model(self.img_data_2d[1][0])
                # 使用序列量化后的检查方法验证模型
                self.checkModelWithSequentialQuantized(model)

    # 验证包含序列量化的模型的各个部分是否符合预期类型
    def checkModelWithSequentialQuantized(self, model):
        self.assertEqual(type(model.conv1), nniq.ConvReLU2d)
        self.assertEqual(type(model.relu1), nn.Identity)
        # 遍历特征提取部分的每个子模块，检查其量化后的类型
        for i in range(3):
            self.assertEqual(type(model.features[i][0]), nniq.ConvReLU2d)
            self.assertEqual(type(model.features[i][1]), nn.Identity)
            self.assertEqual(type(model.features[i][2]), nn.Identity)
        # 检查分类器部分的量化后模块类型
        self.assertEqual(type(model.classifier[0]), nniq.LinearReLU)
        self.assertEqual(type(model.classifier[1]), nn.Identity)
    def test_fusion_conv_with_bias(self):
        # 遍历支持的量化引擎
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖量化引擎设置
            with override_quantized_engine(qengine):
                # 创建原始模型并设置为训练模式
                model_orig = ModelForFusionWithBias().train()

                # 创建参考模型
                model_ref = copy.deepcopy(model_orig)
                # 对未融合的模型进行推理
                out_ref = model_ref(self.img_data_2d[0][0])

                # 融合模型
                model_orig.qconfig = QConfig(activation=torch.nn.Identity,
                                             weight=torch.nn.Identity)
                model = fuse_modules_qat(
                    model_orig,
                    [["conv1", "bn1", "relu1"],
                     ["conv2", "bn2"]])
                # 准备量化感知训练模型，不在原地修改
                prep_model = prepare_qat(model, inplace=False)
                # 对融合后但未加入观察器的模型进行推理
                out_fused = prep_model(self.img_data_2d[0][0])

                # 断言两个输出结果相等
                self.assertEqual(out_ref, out_fused)

                # 检查 BatchNorm 层是否正确
                def checkBN(bn_ref, bn):
                    self.assertEqual(bn_ref.weight, bn.weight)
                    self.assertEqual(bn_ref.bias, bn.bias)
                    self.assertEqual(bn_ref.running_mean, bn.running_mean)
                    self.assertEqual(bn_ref.running_var, bn.running_var)

                # 检查每个融合的模块的 BatchNorm 层
                checkBN(model_ref.bn1, prep_model.conv1.bn)
                checkBN(model_ref.bn2, prep_model.conv2.bn)

                # 使用默认的量化配置设置模型的量化配置
                model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                # 在原地准备量化感知训练模型
                prepare_qat(model, inplace=True)

                # 对模型进行推理
                model(self.img_data_2d[0][0])

                # 检查量化感知训练模型的结构
                def checkQAT(model):
                    self.assertEqual(type(model.conv1), nniqat.ConvBnReLU2d)
                    self.assertEqual(type(model.bn1), nn.Identity)
                    self.assertEqual(type(model.relu1), nn.Identity)
                    self.assertEqual(type(model.conv2), nniqat.ConvBn2d)
                    self.assertEqual(type(model.bn2), nn.Identity)

                # 检查模型结构是否符合预期
                checkQAT(model)


    def test_fusion_linear_bn_eval(self):
        # 创建线性 BatchNorm 融合模型并设置为训练模式
        model = ModelForLinearBNFusion().train()
        inp1 = torch.randn(8, 20)
        inp2 = torch.randn(8, 20)

        # 将输入数据传入模型以更新均值和方差
        model(inp1)
        # 将模型设置为评估模式
        model.eval()
        # 保存评估模式下的输出
        golden = model(inp2)

        # 融合线性层和 BatchNorm 层
        model = fuse_modules(model, [["fc", "bn"]])
        # 断言 BatchNorm 层已替换为 nn.Identity，且输出结果与评估模式下的输出一致
        self.assertEqual(type(model.bn), nn.Identity)
        self.assertEqual(golden, model(inp2))
    # 定义测试方法，验证融合卷积转置和批量归一化的模型行为
    def test_fusion_convtranspose_bn_eval(self):
        # 创建一个用于融合卷积转置和批量归一化的模型，并设置为训练模式
        model = ModelForConvTransposeBNFusion().train()
        # 生成两个输入张量，形状为[8, 3, 16]
        inp1 = torch.randn(8, 3, 16)
        inp2 = torch.randn(8, 3, 16)

        # 让模型处理inp1，使得批量归一化的运行均值和方差被更新
        model(inp1)
        # 将模型切换为评估模式
        model.eval()
        # 保存评估模式下对inp2的计算结果
        golden = model(inp2)

        # 融合模型的指定层（卷积层和对应的批量归一化层）
        model = fuse_modules(model, [["conv1", "bn1"], ["conv2", "bn2"], ["conv3", "bn3"]])
        # 验证融合后的模型中各层是否为身份映射
        self.assertEqual(type(model.bn1), nn.Identity)
        self.assertEqual(type(model.bn2), nn.Identity)
        self.assertEqual(type(model.bn3), nn.Identity)

        # 验证融合后模型对inp2的计算结果与之前golden的计算结果是否一致
        self.assertEqual(golden, model(inp2))

    # 定义测试方法，验证自定义融合函数的行为
    def test_fuse_function_customization(self):
        # 创建一个单层线性模型，并设置为训练模式，然后切换为评估模式
        dummy_model = SingleLayerLinearModel().train()
        dummy_model.eval()

        # 定义一个自定义的融合函数
        def custom_fuse_func(module, is_qat, add_fuser_mapping):
            return [torch.nn.Identity()]

        # 使用自定义融合函数融合指定层（fc1）
        dummy_model = fuse_modules(dummy_model, [["fc1"]], fuser_func=custom_fuse_func)
        # 验证融合后的模型中fc1层是否为身份映射
        self.assertEqual(type(dummy_model.fc1), nn.Identity)
    def test_forward_hooks_preserved(self):
        r"""Test case that checks whether forward pre hooks of the first module and
        post forward hooks of the last module in modules list passed to fusion function preserved.
        (e.g. before fusion: [nn.Conv2d (with pre forward hooks), nn.BatchNorm2d, nn.ReLU (with post forward hooks)]
        after fusion: [nni.ConvBnReLU2d (with pre and post hooks), nn.Identity, nn.Identity])
        """
        # 创建一个 QAT 模式下的测试模型
        model = ModelForFusion(default_qat_qconfig).train()

        # 计数器，用于统计前向预处理和前向处理的调用次数
        counter = {
            'pre_forwards': 0,
            'forwards': 0,
        }
        # 标志变量，用于标记是否已经进行了模块融合
        fused = False

        # 前向预处理钩子函数，用于检查第一个模块的前向预处理是否保留
        def fw_pre_hook(fused_module_class, h_module, input):
            if fused:
                self.assertEqual(type(h_module), fused_module_class,
                                 "After fusion owner of the first module's forward pre hook is not a fused module")
            counter['pre_forwards'] += 1

        # 前向处理钩子函数，用于检查最后一个模块的前向处理是否保留
        def fw_hook(fused_module_class, h_module, input, output):
            if fused:
                self.assertEqual(type(h_module), fused_module_class,
                                 "After fusion owner of the last module's forward hook is not a fused module")
            counter['forwards'] += 1

        # 注册两个前向预处理钩子和两个前向处理钩子，期望每次推断时计数器增加两次
        model.conv1.register_forward_pre_hook(lambda *args: fw_pre_hook(nni.ConvBnReLU2d, *args))
        model.sub1.conv.register_forward_pre_hook(lambda *args: fw_pre_hook(nni.ConvBn2d, *args))
        model.relu1.register_forward_hook(lambda *args: fw_hook(nni.ConvBnReLU2d, *args))
        model.sub1.bn.register_forward_hook(lambda *args: fw_hook(nni.ConvBn2d, *args))

        # 使用测试数据进行仅评估功能的测试
        test_only_eval_fn(model, self.img_data_1d)
        self.assertEqual(counter['pre_forwards'], 2 * len(self.img_data_1d))
        self.assertEqual(counter['forwards'], 2 * len(self.img_data_1d))

        # 对模型进行模块融合
        model = fuse_modules_qat(model, ['conv1', 'bn1', 'relu1'])
        model = fuse_modules_qat(model, ['sub1.conv', 'sub1.bn'])

        # 标记模块已融合
        fused = True
        # 记录融合前的前向预处理和前向处理计数
        before_fusion_pre_count = counter['pre_forwards']
        before_fusion_post_count = counter['forwards']
        # 使用测试数据进行仅评估功能的测试
        test_only_eval_fn(model, self.img_data_1d)
        # 检查融合后前向预处理和前向处理计数的增加量是否符合预期
        self.assertEqual(counter['pre_forwards'] - before_fusion_pre_count, 2 * len(self.img_data_1d))
        self.assertEqual(counter['forwards'] - before_fusion_post_count, 2 * len(self.img_data_1d))
    def test_fuse_modules_with_nested_hooks(self):
        r"""Test case that checks whether a nested module with sub-sub modules registered with hooks
        can be safely fused. Safeguard for issues similar to https://github.com/pytorch/pytorch/issues/105063
        in the future.
        """
        # 定义一个自定义的钩子函数 myhook，接受任意参数并返回空字符串
        def myhook(*x):
            return ""

        # 遍历支持的量化引擎
        for qengine in supported_qengines:
            # 使用当前量化引擎进行上下文覆盖
            with override_quantized_engine(qengine):
                # 创建并评估一个带有序列化融合的模型对象
                model = ModelWithSequentialFusion().eval()

                # 遍历模型的所有子模块
                for sub_model in model.modules():
                    # 如果子模块是 nn.Sequential 类型
                    if isinstance(sub_model, nn.Sequential):
                        # 遍历序列模块中的每一层
                        for layer in sub_model:
                            # 如果层具有 'register_forward_hook' 属性
                            if hasattr(layer, 'register_forward_hook'):
                                # 注册自定义的前向钩子函数 myhook
                                layer.register_forward_hook(myhook)

                # 融合模型的指定模块路径 [['features.0.0', 'features.0.1', 'features.0.2']]，并原地修改模型
                fuse_modules(model, [['features.0.0', 'features.0.1', 'features.0.2']], inplace=True)

                # 断言融合后的第一个子模块的第一个层的类型为 nni.ConvReLU2d
                self.assertEqual(
                    type(model.features[0][0]),
                    nni.ConvReLU2d,
                    msg="Fused submodule Conv + folded BN"
                )

                # 断言融合后的第一个子模块的第二个层的类型为 nn.Identity
                self.assertEqual(
                    type(model.features[0][1]),
                    nn.Identity,
                    msg="Fused submodule (skipped BN)"
                )

                # 断言未融合的第一个子模块的第三个层的类型仍为 nn.Identity
                self.assertEqual(
                    type(model.features[0][2]),
                    nn.Identity,
                    msg="Non-fused submodule Conv"
                )
# 如果当前脚本被直接运行（而非被导入为模块），则抛出运行时错误。
if __name__ == '__main__':
    # 抛出运行时异常，提示用户不应直接运行此测试文件，而是应该使用指定的方式运行。
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_quantization.py TESTNAME\n\n"
        "instead."
    )
```