# `.\pytorch\test\quantization\eager\test_model_numerics.py`

```
# Owner(s): ["oncall: quantization"]

import torch  # 导入PyTorch库

from torch.testing._internal.common_quantization import (  # 导入测试所需的量化模块和类
    QuantizationTestCase,
    ModelMultipleOps,
    ModelMultipleOpsNoAvgPool,
)
from torch.testing._internal.common_quantized import (  # 导入测试所需的量化模块和函数
    override_quantized_engine,
    supported_qengines,
)

class TestModelNumericsEager(QuantizationTestCase):  # 定义测试类，继承自QuantizationTestCase类
    def test_float_quant_compare_per_tensor(self):  # 定义测试方法，用于比较基于张量的浮点量化
        for qengine in supported_qengines:  # 遍历支持的量化引擎列表
            with override_quantized_engine(qengine):  # 使用当前量化引擎进行上下文管理
                torch.manual_seed(42)  # 设置随机种子
                my_model = ModelMultipleOps().to(torch.float32)  # 创建多操作模型，并转换为float32类型
                my_model.eval()  # 设置模型为评估模式
                calib_data = torch.rand(1024, 3, 15, 15, dtype=torch.float32)  # 创建用于校准的随机数据
                eval_data = torch.rand(1, 3, 15, 15, dtype=torch.float32)  # 创建用于评估的随机数据
                out_ref = my_model(eval_data)  # 使用浮点模型评估数据并获取输出
                qModel = torch.ao.quantization.QuantWrapper(my_model)  # 将模型封装为量化模型
                qModel.eval()  # 设置量化模型为评估模式
                qModel.qconfig = torch.ao.quantization.default_qconfig  # 设置默认的量化配置
                torch.ao.quantization.fuse_modules(qModel.module, [['conv1', 'bn1', 'relu1']], inplace=True)  # 融合模块
                torch.ao.quantization.prepare(qModel, inplace=True)  # 准备模型进行量化
                qModel(calib_data)  # 使用校准数据对模型进行量化
                torch.ao.quantization.convert(qModel, inplace=True)  # 将模型转换为量化模型
                out_q = qModel(eval_data)  # 使用量化模型评估数据并获取输出
                SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))  # 计算信噪比的分贝值
                # 量化模型输出的数值应与浮点模型输出接近
                # 设置目标信噪比为30 dB，以便相对误差在期望输出以下1e-3
                self.assertGreater(SQNRdB, 30, msg='Quantized model numerics diverge from float, expect SQNR > 30 dB')

    def test_float_quant_compare_per_channel(self):  # 定义测试方法，用于比较基于通道的浮点量化
        # Test for per-channel Quant
        torch.manual_seed(67)  # 设置随机种子
        my_model = ModelMultipleOps().to(torch.float32)  # 创建多操作模型，并转换为float32类型
        my_model.eval()  # 设置模型为评估模式
        calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)  # 创建用于校准的随机数据
        eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)  # 创建用于评估的随机数据
        out_ref = my_model(eval_data)  # 使用浮点模型评估数据并获取输出
        q_model = torch.ao.quantization.QuantWrapper(my_model)  # 将模型封装为量化模型
        q_model.eval()  # 设置量化模型为评估模式
        q_model.qconfig = torch.ao.quantization.default_per_channel_qconfig  # 设置默认的通道量化配置
        torch.ao.quantization.fuse_modules(q_model.module, [['conv1', 'bn1', 'relu1']], inplace=True)  # 融合模块
        torch.ao.quantization.prepare(q_model)  # 准备模型进行量化
        q_model(calib_data)  # 使用校准数据对模型进行量化
        torch.ao.quantization.convert(q_model)  # 将模型转换为量化模型
        out_q = q_model(eval_data)  # 使用量化模型评估数据并获取输出
        SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))  # 计算信噪比的分贝值
        # 量化模型输出的数值应与浮点模型输出接近
        # 设置目标信噪比为35 dB
        self.assertGreater(SQNRdB, 35, msg='Quantized model numerics diverge from float, expect SQNR > 35 dB')
    # 定义测试函数，用于比较假量化和真量化之间的数值差异
    def test_fake_quant_true_quant_compare(self):
        # 遍历支持的量化引擎
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖量化设置
            with override_quantized_engine(qengine):
                # 设置随机种子以确保可复现性
                torch.manual_seed(67)
                # 创建一个模型实例，不包含平均池化层，并转换为单精度浮点数
                my_model = ModelMultipleOpsNoAvgPool().to(torch.float32)
                # 生成用于校准的随机数据，形状为 (2048, 3, 15, 15)，数据类型为单精度浮点数
                calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
                # 生成用于评估的随机数据，形状为 (10, 3, 15, 15)，数据类型为单精度浮点数
                eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
                # 将模型设置为评估模式
                my_model.eval()
                # 使用 QuantWrapper 包装模型，进行量化模型的创建
                fq_model = torch.ao.quantization.QuantWrapper(my_model)
                # 设置量化模型为训练模式
                fq_model.train()
                # 设置量化配置为默认的量化训练配置
                fq_model.qconfig = torch.ao.quantization.default_qat_qconfig
                # 融合指定的模块（conv1, bn1, relu1）以加速量化训练
                torch.ao.quantization.fuse_modules_qat(fq_model.module, [['conv1', 'bn1', 'relu1']], inplace=True)
                # 准备模型以进行量化训练
                torch.ao.quantization.prepare_qat(fq_model)
                # 将模型设置为评估模式
                fq_model.eval()
                # 禁用假量化观察器
                fq_model.apply(torch.ao.quantization.disable_fake_quant)
                # 冻结批归一化统计量
                fq_model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
                # 在校准数据上运行量化模型
                fq_model(calib_data)
                # 启用假量化
                fq_model.apply(torch.ao.quantization.enable_fake_quant)
                # 禁用观察器
                fq_model.apply(torch.ao.quantization.disable_observer)
                # 使用假量化模型对评估数据进行推理
                out_fq = fq_model(eval_data)
                # 计算信噪比（SQNR），单位为分贝（dB）
                SQNRdB = 20 * torch.log10(torch.norm(out_ref) / (torch.norm(out_ref - out_fq) + 1e-10))
                # 断言：量化模型的输出与浮点模型的输出应该在数值上接近
                self.assertGreater(SQNRdB, 35, msg='Quantized model numerics diverge from float, expect SQNR > 35 dB')
                # 将量化模型转换为真量化模型
                torch.ao.quantization.convert(fq_model)
                # 使用真量化模型对评估数据进行推理
                out_q = fq_model(eval_data)
                # 计算假量化和真量化之间的信噪比（SQNR），单位为分贝（dB）
                SQNRdB = 20 * torch.log10(torch.norm(out_fq) / (torch.norm(out_fq - out_q) + 1e-10))
                # 断言：假量化和真量化的数值应该接近
                self.assertGreater(SQNRdB, 60, msg='Fake quant and true quant numerics diverge, expect SQNR > 60 dB')

    # 用于比较仅权重量化模型数值和仅激活量化模型数值与浮点数之间的差异的测试
    # 定义一个测试函数，用于测试仅权重或仅激活量化的情况下的假量化功能
    def test_weight_only_activation_only_fakequant(self):
        # 遍历支持的量化引擎列表
        for qengine in supported_qengines:
            # 使用当前量化引擎覆盖默认引擎设置
            with override_quantized_engine(qengine):
                # 设置随机种子为67
                torch.manual_seed(67)
                # 创建用于校准的随机数据，维度为 (2048, 3, 15, 15)，数据类型为 float32
                calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
                # 创建用于评估的随机数据，维度为 (10, 3, 15, 15)，数据类型为 float32
                eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
                # 定义量化配置集合，包括默认的仅权重量化配置和仅激活量化配置
                qconfigset = {torch.ao.quantization.default_weight_only_qconfig,
                              torch.ao.quantization.default_activation_only_qconfig}
                # 定义信噪比（Signal-to-Quantization-Noise Ratio）的目标值，用于评估量化后模型的数值稳定性
                SQNRTarget = [35, 45]
                # 遍历量化配置集合的索引和配置
                for idx, qconfig in enumerate(qconfigset):
                    # 创建一个未平均池化的多操作模型实例，并转换为 float32 数据类型
                    my_model = ModelMultipleOpsNoAvgPool().to(torch.float32)
                    # 设置模型为评估模式
                    my_model.eval()
                    # 对评估数据进行前向传播，获取参考输出
                    out_ref = my_model(eval_data)
                    # 使用 QuantWrapper 封装模型，使其支持量化训练
                    fq_model = torch.ao.quantization.QuantWrapper(my_model)
                    # 将封装后的模型设置为训练模式
                    fq_model.train()
                    # 配置模型的量化配置为当前配置
                    fq_model.qconfig = qconfig
                    # 使用量化模型API将模型的特定模块融合（例如：conv1、bn1、relu1）
                    torch.ao.quantization.fuse_modules_qat(fq_model.module, [['conv1', 'bn1', 'relu1']], inplace=True)
                    # 准备模型以进行量化训练
                    torch.ao.quantization.prepare_qat(fq_model)
                    # 将模型设置为评估模式
                    fq_model.eval()
                    # 禁用假量化操作
                    fq_model.apply(torch.ao.quantization.disable_fake_quant)
                    # 冻结批标准化统计信息
                    fq_model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
                    # 使用校准数据进行模型的量化训练
                    fq_model(calib_data)
                    # 启用假量化操作
                    fq_model.apply(torch.ao.quantization.enable_fake_quant)
                    # 禁用观察者
                    fq_model.apply(torch.ao.quantization.disable_observer)
                    # 对评估数据进行前向传播，获取量化后的输出
                    out_fq = fq_model(eval_data)
                    # 计算信噪比（SQNR），以评估量化模型的数值稳定性
                    SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_fq))
                    # 使用断言检查信噪比是否大于预设的目标值，若小于则输出错误消息
                    self.assertGreater(SQNRdB, SQNRTarget[idx], msg='Quantized model numerics diverge from float')
# 如果脚本被直接运行而非作为模块被导入，抛出运行时错误并显示以下消息：
# "This test file is not meant to be run directly, use:"
# 换行后提示如何正确使用此测试文件的命令：
# "\tpython test/test_quantization.py TESTNAME\n\n"
# "instead."
if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
```