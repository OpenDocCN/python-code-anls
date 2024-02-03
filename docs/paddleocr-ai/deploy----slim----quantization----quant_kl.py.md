# `.\PaddleOCR\deploy\slim\quantization\quant_kl.py`

```py
# 版权声明
# 该代码版权归 PaddlePaddle 作者所有
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用该文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

# 获取当前文件所在目录
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径中
sys.path.append(__dir__)
# 将当前目录的上级目录添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(__dir__, '..', '..', '..')))
# 将当前目录的上上级目录添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(__dir__, '..', '..', '..', 'tools')))

# 导入其他必要的库
import yaml
import paddle
import paddle.distributed as dist

# 设置随机种子
paddle.seed(2)

# 导入自定义模块
from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
import tools.program as program
import paddleslim
from paddleslim.dygraph.quant import QAT
import numpy as np

# 获取当前进程的世界大小
dist.get_world_size()

# 定义 PACT 类，继承自 paddle.nn.Layer
class PACT(paddle.nn.Layer):
    def __init__(self):
        super(PACT, self).__init__()
        # 定义 alpha 参数的属性
        alpha_attr = paddle.ParamAttr(
            name=self.full_name() + ".pact",
            initializer=paddle.nn.initializer.Constant(value=20),
            learning_rate=1.0,
            regularizer=paddle.regularizer.L2Decay(2e-5))

        # 创建 alpha 参数
        self.alpha = self.create_parameter(
            shape=[1], attr=alpha_attr, dtype='float32')
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 计算左侧激活函数的输出，使用 ReLU 函数
        out_left = paddle.nn.functional.relu(x - self.alpha)
        # 计算右侧激活函数的输出，使用 ReLU 函数
        out_right = paddle.nn.functional.relu(-self.alpha - x)
        # 更新输入 x，实现残差连接
        x = x - out_left + out_right
        # 返回更新后的 x
        return x
quant_config = {
    # 权重预处理类型，默认为None，不执行任何预处理
    'weight_preprocess_type': None,
    # 激活预处理类型，默认为None，不执行任何预处理
    'activation_preprocess_type': None,
    # 权重量化类型，默认为'channel_wise_abs_max'
    'weight_quantize_type': 'channel_wise_abs_max',
    # 激活量化类型，默认为'moving_average_abs_max'
    'activation_quantize_type': 'moving_average_abs_max',
    # 权重量化位数，默认为8
    'weight_bits': 8,
    # 激活量化位数，默认为8
    'activation_bits': 8,
    # 量化后的数据类型，如'uint8'、'int8'等，默认为'int8'
    'dtype': 'int8',
    # 'range_abs_max'量化的窗口大小，默认为10000
    'window_size': 10000,
    # 移动平均的衰减系数，默认为0.9
    'moving_rate': 0.9,
    # 对于动态图量化，将对quantizable_layer_type中类型的层进行量化
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}

# 生成样本数据的生成器
def sample_generator(loader):
    def __reader__():
        for indx, data in enumerate(loader):
            images = np.array(data[0])
            yield images

    return __reader__

# 生成样本数据的生成器，用于LayoutXLM-SER模型
def sample_generator_layoutxlm_ser(loader):
    def __reader__():
        for indx, data in enumerate(loader):
            input_ids = np.array(data[0])
            bbox = np.array(data[1])
            attention_mask = np.array(data[2])
            token_type_ids = np.array(data[3])
            images = np.array(data[4])
            yield [input_ids, bbox, attention_mask, token_type_ids, images]

    return __reader__

# 主函数
def main(config, device, logger, vdl_writer):
    # 初始化分布式环境
    if config['Global']['distributed']:
        dist.init_parallel_env()

    global_config = config['Global']

    # 构建数据加载器
    set_signal_handlers()
    config['Train']['loader']['num_workers'] = 0
    # 检查模型类型是否为 'kie'，且后端模型为 'LayoutXLMForSer'
    is_layoutxlm_ser =  config['Architecture']['model_type'] =='kie' and config['Architecture']['Backbone']['name'] == 'LayoutXLMForSer'
    # 构建训练数据加载器
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    # 如果需要进行评估
    if config['Eval']:
        # 设置评估数据加载器的工作线程数为 0
        config['Eval']['loader']['num_workers'] = 0
        # 构建评估数据加载器
        valid_dataloader = build_dataloader(config, 'Eval', device, logger)
        # 如果是 LayoutXLMForSer 模型，则将训练数据加载器替换为评估数据加载器
        if is_layoutxlm_ser:
            train_dataloader = valid_dataloader
    else:
        valid_dataloader = None

    # 启用静态图模式
    paddle.enable_static()
    # 创建 Executor 对象
    exe = paddle.static.Executor(device)

    # 检查全局配置中是否包含 'inference_model' 键
    if 'inference_model' in global_config.keys():  # , 'inference_model'):
        inference_model_dir = global_config['inference_model']
    else:
        # 如果不包含，则使用预训练模型的目录作为推理模型目录
        inference_model_dir = os.path.dirname(global_config['pretrained_model'])
        # 检查推理模型文件是否存在
        if  not (os.path.exists(os.path.join(inference_model_dir, "inference.pdmodel")) and \
            os.path.exists(os.path.join(inference_model_dir, "inference.pdiparams")) ):
            # 如果不存在，则抛出数值错误
            raise ValueError(
                "Please set inference model dir in Global.inference_model or Global.pretrained_model for post-quantazition"
            )
    
    # 根据模型类型选择样本生成器
    if is_layoutxlm_ser:
        generator = sample_generator_layoutxlm_ser(train_dataloader)
    else:
        generator = sample_generator(train_dataloader)

    # 对模型进行后量化
    paddleslim.quant.quant_post_static(
        executor=exe,
        model_dir=inference_model_dir,
        model_filename='inference.pdmodel',
        params_filename='inference.pdiparams',
        quantize_model_path=global_config['save_inference_dir'],
        sample_generator=generator,
        save_model_filename='inference.pdmodel',
        save_params_filename='inference.pdiparams',
        batch_size=1,
        batch_nums=None)
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 调用 program 模块的 preprocess 函数，传入 is_train=True 参数，获取 config, device, logger, vdl_writer 四个变量
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    # 调用 main 函数，传入 config, device, logger, vdl_writer 四个变量作为参数
    main(config, device, logger, vdl_writer)
```