# `.\pytorch\torch\distributed\optim\functional_adamw.py`

```
# mypy: allow-untyped-defs
from typing import Dict, List, Optional, Tuple  # 导入必要的类型定义

import torch  # 导入PyTorch库
import torch.optim._functional as F  # 导入PyTorch中的优化函数模块
from torch import Tensor  # 导入Tensor类型


__all__: List[str] = []  # 初始化一个空列表，用于存放模块中所有公开的名称


# 定义一个 TorchScript 兼容的 Functional AdamW 优化器类
# 在这个优化器中，我们以一种函数式的方式使用它。
# 不使用 `param.grad` 来更新参数，而是明确允许分布式优化器将梯度传递给 `step` 函数。
# 这样做可以将梯度和参数分离，允许多线程训练器在更新参数时不会在同一 .grad 上积累数据痕迹。
# 注意：这应该只由分布式优化器内部使用，而不是暴露给用户使用。
@torch.jit.script
class _FunctionalAdamW:
    def __init__(
        self,
        params: List[Tensor],  # 参数列表，包含待优化的张量
        lr: float = 1e-3,  # 学习率，默认为 0.001
        betas: Tuple[float, float] = (0.9, 0.999),  # beta 参数，默认为 (0.9, 0.999)
        eps: float = 1e-8,  # epsilon 参数，默认为 1e-8
        weight_decay: float = 1e-2,  # 权重衰减，默认为 0.01
        amsgrad: bool = False,  # 是否使用 AMSGrad 算法，默认为 False
        maximize: bool = False,  # 是否最大化优化目标，默认为 False
        foreach: bool = False,  # 是否为每个参数应用优化，默认为 False
        fused: bool = False,  # 是否启用融合优化，默认为 False
        _allow_empty_param_list: bool = False,  # 是否允许空参数列表，默认为 False
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")  # 如果学习率小于等于 0，抛出异常
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")  # 如果 epsilon 小于等于 0，抛出异常
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")  # 如果 beta1 不在 [0, 1) 范围内，抛出异常
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")  # 如果 beta2 不在 [0, 1) 范围内，抛出异常
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")  # 如果权重衰减小于等于 0，抛出异常

        # 设置默认参数字典
        self.defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
        }
        self.amsgrad = amsgrad  # 是否使用 AMSGrad 算法
        self.maximize = maximize  # 是否最大化优化目标
        self.foreach = foreach  # 是否为每个参数应用优化
        self.fused = fused  # 是否启用融合优化
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})  # 初始化状态字典

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")  # 如果参数列表为空且不允许空列表，抛出异常

        # 注意：我们只有一个参数组，并且不允许用户添加额外的参数组，因为这不是常见的使用情况。
        self.param_group = {"params": params}  # 设置参数组
    python
        def step_param(self, param: Tensor, grad: Optional[Tensor]):
            # 初始化空列表，用于存储有梯度的参数、梯度值、指数移动平均和平方梯度平均值等状态
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps: List[Tensor] = []
            
            # 检查参数是否为复数类型
            has_complex = torch.is_complex(param)
            
            # 如果梯度不为None，将参数和对应的梯度添加到相应的列表中
            if grad is not None:
                params_with_grad.append(param)
                grads.append(grad)
            
            # 懒惰状态初始化
            if param not in self.state:
                # 如果参数不在状态字典中，初始化其状态为一个空字典
                self.state[param] = {}
                state = self.state[param]
                
                # 设置初始步数为0
                state["step"] = torch.tensor(0.0)
                
                # 初始化指数移动平均值（exp_avg）为零张量，保留参数的内存格式
                state["exp_avg"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                
                # 初始化指数移动平方梯度平均值（exp_avg_sq）为零张量，保留参数的内存格式
                state["exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                
                # 如果开启了AMSGrad，初始化最大指数移动平方梯度平均值为零张量，保留参数的内存格式
                if self.amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
    
            # 获取参数对应的状态字典
            state = self.state[param]
    
            # 将参数的指数移动平均值和平方梯度平均值添加到对应列表中
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
    
            # 如果使用AMSGrad，将最大指数移动平方梯度平均值添加到列表中
            if self.amsgrad:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
    
            # 将步数张量添加到状态步数列表中
            state_steps.append(state["step"])
    
            # 在无梯度更新的情况下，执行AdamW优化器操作
            with torch.no_grad():
                F.adamw(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=self.amsgrad,
                    maximize    maximize=self.maximize,
                    beta1=self.defaults["beta1"],
                    beta2=self.defaults["beta2"],
                    lr=self.defaults["lr"],
                    weight_decay=self.defaults["weight_decay"],
                    eps=self.defaults["eps"],
                    foreach=self.foreach,
                    fused=self.fused,
                    grad_scale=None,
                    found_inf=None,
                    has_complex=has_complex,
                )
    def step(self, gradients: List[Optional[Tensor]]):
        # 获取当前优化器参数组的参数列表
        params = self.param_group["params"]
        # 用来存储具有梯度的参数
        params_with_grad = []
        # 存储梯度
        grads = []
        # 存储指数移动平均值
        exp_avgs = []
        # 存储指数移动平方平均值
        exp_avg_sqs = []
        # 存储最大的指数移动平方平均值
        max_exp_avg_sqs = []
        # 存储状态步数
        state_steps: List[Tensor] = []

        # 检查参数和梯度的长度是否匹配，如果不匹配则抛出错误
        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        # 是否存在复数类型的参数
        has_complex = False
        # 遍历参数和梯度
        for param, gradient in zip(self.param_group["params"], gradients):
            # 如果梯度不为None
            if gradient is not None:
                # 检查参数是否为复数类型
                has_complex |= torch.is_complex(param)
                # 将具有梯度的参数加入列表
                params_with_grad.append(param)
                # 将梯度加入列表
                grads.append(gradient)
                # 懒惰地初始化状态
                if param not in self.state:
                    # 如果状态中不存在该参数，则初始化
                    self.state[param] = {}
                    state = self.state[param]
                    # 初始化步数为0的张量
                    state["step"] = torch.tensor(0.0)
                    # 指数移动平均值的初始化
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # 指数移动平方平均值的初始化
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # 如果使用amsgrad，则初始化最大的指数移动平方平均值
                    if self.amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                # 获取当前参数的状态
                state = self.state[param]
                # 将指数移动平均值加入列表
                exp_avgs.append(state["exp_avg"])
                # 将指数移动平方平均值加入列表
                exp_avg_sqs.append(state["exp_avg_sq"])

                # 如果使用amsgrad，则将最大的指数移动平方平均值加入列表
                if self.amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                # 将状态步数加入列表
                state_steps.append(state["step"])

        # 进行无梯度操作的上下文管理
        with torch.no_grad():
            # 调用F.adamw函数执行优化步骤
            F.adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=self.amsgrad,
                maximize=self.maximize,
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                eps=self.defaults["eps"],
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
                has_complex=has_complex,
            )
```