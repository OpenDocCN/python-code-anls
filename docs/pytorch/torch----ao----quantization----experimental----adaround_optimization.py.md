# `.\pytorch\torch\ao\quantization\experimental\adaround_optimization.py`

```py
# mypy: allow-untyped-defs
# 引入必要的库和模块
import copy  # 引入copy模块，用于对象深拷贝
from typing import Any, Callable, List, Optional, Tuple, Type, Union  # 引入类型提示相关的模块

import torch  # 引入PyTorch库
from torch.ao.quantization.experimental.adaround_fake_quantize import (
    AdaroundFakeQuantizer,  # 导入AdaroundFakeQuantizer类
)
from torch.ao.quantization.experimental.adaround_loss import AdaptiveRoundingLoss  # 导入AdaptiveRoundingLoss类
from torch.ao.quantization.observer import MinMaxObserver  # 导入MinMaxObserver类
from torch.nn import functional as F  # 导入torch.nn.functional并简写为F
from torch.nn.parallel import DataParallel  # 导入DataParallel类
from torch.utils.data import DataLoader, TensorDataset  # 导入DataLoader和TensorDataset类


class AdaptiveRoundingOptimizer:
    def __init__(
        self,
        model: Union[torch.nn.Module, torch.nn.DataParallel],  # 接收类型为torch.nn.Module或torch.nn.DataParallel的model参数
        callback: Callable[  # 回调函数，接收参数为model、任意类型的数据和可选的module对象
            [
                Union[torch.nn.Module, torch.nn.DataParallel],
                Any,
                Optional[torch.nn.Module],
            ],
            None,  # 返回值为None
        ],
        forward_hook_wrapper: Callable[[List[torch.Tensor]], Callable],  # 接收类型为List[torch.Tensor]的列表，返回值为Callable对象的函数
        data: Any,  # 接收任意类型的数据作为输入
        observer: Type[torch.ao.quantization.observer.ObserverBase] = MinMaxObserver,  # 观察器类型，默认为MinMaxObserver
        max_iter=10000,  # 最大迭代次数，默认为10000
        dtype: torch.dtype = torch.qint8,  # 数据类型，默认为torch.qint8
        quant_min=-128,  # 量化的最小值，默认为-128
        quant_max=127,  # 量化的最大值，默认为127
        qscheme: torch.qscheme = torch.per_tensor_symmetric,  # 量化方案，默认为torch.per_tensor_symmetric
        batch_size: int = 256,  # 批处理大小，默认为256
        feed_forward_wrapper: Optional[torch.nn.Module] = None,  # 可选的前向包装器，默认为None
    ):
        if torch.cuda.is_available():  # 检查是否支持CUDA
            self.model = model.cuda()  # 将模型移动到CUDA设备
            if torch.cuda.device_count() > 1:  # 如果存在多个CUDA设备
                self.model = torch.nn.DataParallel(model)  # 使用DataParallel在多个GPU上运行模型
        else:
            self.model = model  # 如果不支持CUDA，则使用CPU上的模型
        self.q_model = copy.deepcopy(self.model)  # 创建模型的深拷贝
        self.device = torch.device("cuda") if torch.cuda.is_available() else None  # 设置设备为CUDA或None
        self.callback = callback  # 回调函数，用于模型量化过程中调用
        self.forward_hook_wrapper = forward_hook_wrapper  # 前向钩子包装器函数，用于修改前向传播行为
        self.data = data  # 输入数据，用于模型量化
        self.batch_size = min(batch_size, len(data))  # 计算批处理大小，不超过数据长度
        self.max_iter = max_iter  # 最大迭代次数，控制量化优化的迭代次数
        self.adaptive_round_loss_fn = AdaptiveRoundingLoss(  # 自适应舍入损失函数
            max_iter=self.max_iter, warm_start=0.2  # 设置最大迭代次数和温暖启动比例
        )
        self.dtype = dtype  # 数据类型，用于模型量化
        self.observer = observer  # 观察器类型，用于模型量化过程中观察参数
        self.quant_min = quant_min  # 量化的最小值
        self.quant_max = quant_max  # 量化的最大值
        self.qscheme = qscheme  # 量化方案
        self.feed_forward_wrapper = feed_forward_wrapper  # 前向传播包装器，用于包装前向传播函数
    # 定义一个方法 `run_adaround`，返回类型为 `torch.nn.Module`
    def run_adaround(self) -> torch.nn.Module:
        # 初始化一个空列表，用于存储元组 (层名称, 原始模块, 量化模块)
        layer_list: List[Tuple[str, torch.nn.Module, torch.nn.Module]] = []
        # 遍历模型的所有命名模块和量化模块，zip函数将两个迭代器打包成元组
        for (name, module), q_module in zip(
            self.model.named_modules(), self.q_model.modules()
        ):
            # 如果模块是 ReLU 激活函数，禁用所有的原地操作
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
            # 如果量化模块是 ReLU 激活函数，禁用所有的原地操作
            if isinstance(q_module, torch.nn.ReLU):
                q_module.inplace = False
            # 如果模块是 Conv1d 或者 Linear 层，将模块和量化模块加入到列表中
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
                layer_list.append((name, module, q_module))
        # 打印层的总数
        print(f"Total number of layers : {len(layer_list)}")  # noqa: G004

        # 遍历层列表中的每个元组，打印信息并调用 optimize_adaptive_rounding 方法
        for name, module, q_module in layer_list:
            print(
                f"Kick start adaptive rounding on {name} module {module}"  # noqa: G004
            )
            self.optimize_adaptive_rounding(
                module,
                q_module,
                None,
            )

        # 返回量化模型的 module 属性，如果 self.q_model 是 DataParallel 则返回 self.q_model.module，否则返回 self.q_model
        return (
            self.q_model.module
            if isinstance(self.q_model, DataParallel)
            else self.q_model
        )

    # 定义一个方法 `get_data_inp_out`，接受模块、量化模块和数据列表作为参数，返回三个列表的元组
    def get_data_inp_out(
        self, module: torch.nn.Module, q_module: torch.nn.Module, data: List[Any]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        # 初始化四个列表，用于存储不同类型的张量
        fp_out: List[torch.Tensor] = []
        q_input: List[torch.Tensor] = []
        fp_input: List[torch.Tensor] = []
        fp32_fetcher: List[torch.Tensor] = []
        quant_fetcher: List[torch.Tensor] = []
        # 注册 forward hook，用于获取模型输出的张量
        handler1 = module.register_forward_hook(self.forward_hook_wrapper(fp32_fetcher))
        handler2 = q_module.register_forward_hook(
            self.forward_hook_wrapper(quant_fetcher)
        )
        # 如果 CUDA 可用，将模型和量化模型移动到 GPU
        if torch.cuda.is_available():
            # 以防止模型不断下降到 CPU，需要持续移动模型
            self.model = self.model.cuda()
            self.q_model = self.q_model.cuda()
        # 遍历数据列表，对每个数据应用模型和量化模型的回调函数
        for data_ in data:
            with torch.no_grad():
                # 调用回调函数，计算前向传播并将结果存储到相应列表中
                self.callback(self.model, data_, self.feed_forward_wrapper)
                self.callback(self.q_model, data_, self.feed_forward_wrapper)
            # 获取 fp32_fetcher 和 quant_fetcher 中的张量，并添加到对应列表中
            fp32_output = fp32_fetcher[1]
            quant_input = quant_fetcher[0]
            fp_out.append(fp32_output)
            q_input.append(quant_input)
            fp_input.append(fp32_fetcher[0])
        # 移除 forward hook
        handler1.remove()
        handler2.remove()
        # 返回量化输入、fp_out、fp_input 三个列表的元组
        return q_input, fp_out, fp_input

    # 使用 torch.no_grad 装饰器，禁用 PyTorch 自动求导功能
    @torch.no_grad()
    def feed_forward(self, x, weight, module):
        # 根据 module 的类型进行不同的前向传播计算
        if isinstance(module, torch.nn.Conv1d):
            # 如果是 Conv1d 模块，则使用 conv1d 函数进行卷积操作
            out = torch.nn.functional.conv1d(
                x,
                weight,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
        elif isinstance(module, torch.nn.Linear):
            # 如果是 Linear 模块，则使用 linear 函数进行线性操作
            out = torch.nn.functional.linear(
                x,
                weight,
                bias=module.bias,
            )
        else:
            # 如果是其他类型的模块，则抛出未实现错误
            raise NotImplementedError
        return out

    def _compute_and_display_local_losses(
        self,
        ada_quantizer: AdaroundFakeQuantizer,
        q_module: torch.nn.Module,
        q_inp: torch.Tensor,
        fp_out: torch.Tensor,
    ):
        with torch.no_grad():
            # 关闭自动求导
            ada_quantizer.use_soft_rounding = False
            # 对量化模块的权重进行硬量化
            q_w_hard_round = ada_quantizer(q_module.weight)
            # 使用 feed_forward 方法进行前向传播计算（硬量化）
            out_hard_quant = self.feed_forward(q_inp, q_w_hard_round, q_module)
            # 开启软量化
            ada_quantizer.use_soft_rounding = True
            # 对量化模块的权重进行软量化
            q_w_soft_round = ada_quantizer(q_module.weight)
            # 使用 feed_forward 方法进行前向传播计算（软量化）
            out_soft_quant = self.feed_forward(q_inp, q_w_soft_round, q_module)
            # 计算软量化损失
            soft_quant_loss = F.mse_loss(out_soft_quant, fp_out)
            # 计算硬量化损失
            hard_quant_loss = F.mse_loss(out_hard_quant, fp_out)
            # 打印软量化损失和硬量化损失
            print(
                f"soft quant loss: {soft_quant_loss.item()} hard quant loss: {hard_quant_loss.item()}"  # noqa: G004
            )

    def optimize_adaptive_rounding(
        self,
        module: torch.nn.Module,
        q_module: torch.nn.Module,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        # 该方法用于优化自适应舍入
```