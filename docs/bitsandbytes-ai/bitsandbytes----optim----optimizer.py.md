# `bitsandbytes\bitsandbytes\optim\optimizer.py`

```py
# 导入必要的模块
from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain

import torch

import bitsandbytes.functional as F

# 定义一个模拟参数类，用于初始化参数
class MockArgs:
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])

# 定义全局优化管理器类
class GlobalOptimManager:
    _instance = None

    def __init__(self):
        # 避免直接实例化该类，需要通过 get_instance() 方法获取实例
        raise RuntimeError("Call get_instance() instead")

    # 初始化方法，初始化一些属性
    def initialize(self):
        self.pid2config = {}
        self.index2config = {}
        self.optimizer = None
        self.uses_config_override = False
        self.module_weight_config_triple = []

    # 类方法，获取全局唯一实例
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    # 注册参数方法，将参数组注册到管理器中
    def register_parameters(self, params):
        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for group_index, group in enumerate(param_groups):
            for p_index, p in enumerate(group["params"]):
                if id(p) in self.pid2config:
                    self.index2config[(group_index, p_index)] = self.pid2config[id(p)]

    # 覆盖配置方法，用于覆盖参数的配置
    def override_config(self, parameters, key=None, value=None, key_value_dict=None):
    ):
        """
        Overrides initial optimizer config for specific parameters.

        The key-values of the optimizer config for the input parameters are overridden
        This can be both, optimizer parameters like "betas", or "lr" or it can be
        8-bit specific parameters like "optim_bits", "percentile_clipping".

        Parameters
        ----------
        parameters : torch.Tensor or list(torch.Tensors)
            The input parameters.
        key : str
            The hyperparamter to override.
        value : object
            The value for the hyperparamters.
        key_value_dict : dict
            A dictionary with multiple key-values to override.
        """
        # 标记使用了配置覆盖
        self.uses_config_override = True
        # 如果参数是 torch.nn.Parameter 类型，则转换为列表
        if isinstance(parameters, torch.nn.Parameter):
            parameters = [parameters]
        # 如果参数是 torch.Tensor 类型，则转换为列表
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        # 如果 key 和 value 都不为空，则将其添加到 key_value_dict 中
        if key is not None and value is not None:
            assert key_value_dict is None
            key_value_dict = {key: value}

        # 如果 key_value_dict 不为空，则遍历参数列表，更新配置
        if key_value_dict is not None:
            for p in parameters:
                if id(p) in self.pid2config:
                    self.pid2config[id(p)].update(key_value_dict)
                else:
                    self.pid2config[id(p)] = key_value_dict

    # 注册模块覆盖
    def register_module_override(self, module, param_name, config):
        self.module_weight_config_triple.append((module, param_name, config))
# 定义一个名为Optimizer8bit的类，继承自torch.optim.Optimizer类
class Optimizer8bit(torch.optim.Optimizer):
    # 初始化函数，接受参数params、defaults、optim_bits和is_paged，默认optim_bits为32，is_paged为False
    def __init__(self, params, defaults, optim_bits=32, is_paged=False):
        # 调用父类的初始化函数
        super().__init__(params, defaults)
        # 初始化initialized为False
        self.initialized = False
        # 初始化name2qmap为空字典
        self.name2qmap = {}
        # 初始化is_paged为传入的is_paged参数
        self.is_paged = is_paged
        # 获取全局的页面管理器实例
        self.page_mng = F.GlobalPageManager.get_instance()

        # 获取全局的优化管理器实例
        self.mng = GlobalOptimManager.get_instance()
        # 初始化不可转换为张量的键的集合
        self.non_castable_tensor_keys = {
                "qmap1",
                "qmap2",
                "max1",
                "max2",
                "new_max1",
                "new_max2",
                "state1",
                "state2",
                "gnorm_vec",
                "absmax1",
                "absmax2",
                "unorm_vec",
        }

        # 如果optim_bits为8，则调用fill_qmap函数
        if optim_bits == 8:
            self.fill_qmap()

    # 填充name2qmap字典的函数
    def fill_qmap(self):
        # 在name2qmap字典中添加"dynamic"键，并创建一个有符号的动态映射
        self.name2qmap["dynamic"] = F.create_dynamic_map(signed=True)
        # 在name2qmap字典中添加"udynamic"键，并创建一个无符号的动态映射
        self.name2qmap["udynamic"] = F.create_dynamic_map(signed=False)

    # 设置状态的函数
    def __setstate__(self, state):
        # 调用父类的设置状态函数
        super().__setstate__(state)

    # 将参数转移到GPU的函数
    def to_gpu(self):
        # 遍历参数组
        for gindex, group in enumerate(self.param_groups):
            # 遍历参数组中的参数
            for pindex, p in enumerate(group["params"]):
                # 如果参数p在状态中
                if p in self.state:
                    # 获取参数p的值
                    values = self.state[p]
                    # 遍历参数p的值
                    for k, v in values.items():
                        # 如果值v是torch.Tensor类型
                        if isinstance(v, torch.Tensor):
                            # 获取v的is_paged属性，如果没有则默认为False
                            is_paged = getattr(v, 'is_paged', False)
                            # 如果v不是分页的
                            if not is_paged:
                                # 将v转移到p所在设备上
                                self.state[p][k] = v.to(p.device)
    # 检查是否存在参数覆盖
    def check_overrides(self):
        # 遍历每个模块、属性、配置的三元组
        for module, attr, config in self.mng.module_weight_config_triple:
            # 获取属性对应的模块
            pmodule = getattr(module, attr)
            # 断言模块不为空
            assert pmodule is not None
            # 断言模块是 torch.Tensor 或 torch.Parameter 类型
            assert isinstance(pmodule, torch.Tensor) or isinstance(
                pmodule, torch.Parameter
            )
            # 初始化 found 变量为 False
            found = False
            # 遍历参数组
            for gindex, group in enumerate(self.param_groups):
                if found:
                    break
                # 遍历参数组中的参数
                for pindex, p in enumerate(group["params"]):
                    if found:
                        break
                    # 如果找到匹配的参数
                    if id(p) == id(pmodule):
                        # 找到匹配的参数
                        # 初始化覆盖配置
                        self.mng.pid2config[id(p)] = config
                        self.mng.index2config[
                            (gindex, pindex)
                        ] = self.mng.pid2config[id(p)]
                        found = True

    # 禁用梯度计算
    @torch.no_grad()
    # 执行单个优化步骤
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # 初始化损失值
        loss = None
        # 如果有闭包函数，则重新计算模型并返回损失值
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 初始化溢出列表
        overflows = []

        # 如果未初始化，则检查覆盖项，将模型转移到 GPU（用于 fairseq 纯 fp16 训练），并标记为已初始化
        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        # 遍历参数组
        for gindex, group in enumerate(self.param_groups):
            # 遍历参数
            for pindex, p in enumerate(group["params"]):
                # 如果参数梯度为 None，则跳过
                if p.grad is None:
                    continue
                # 获取参数状态
                state = self.state[p]
                # 如果状态为空，则初始化状态
                if len(state) == 0:
                    self.init_state(group, p, gindex, pindex)

                # 预取状态
                self.prefetch_state(p)
                # 更新步骤
                self.update_step(group, p, gindex, pindex)
                # 同步 CUDA 设备
                torch.cuda.synchronize()
        # 如果是分页操作
        if self.is_paged:
            # 所有分页操作都是异步的，需要同步以确保所有张量处于正确状态
            torch.cuda.synchronize()

        # 返回损失值
        return loss

    # 获取配置信息
    def get_config(self, gindex, pindex, group):
        # 初始化配置字典
        config = {}
        # 设置配置参数
        config["betas"] = group["betas"]
        config["eps"] = group["eps"]
        config["weight_decay"] = group["weight_decay"]
        config["lr"] = group["lr"]
        config["optim_bits"] = self.args.optim_bits
        config["min_8bit_size"] = self.args.min_8bit_size
        config["percentile_clipping"] = self.args.percentile_clipping
        config["block_wise"] = self.args.block_wise
        config["max_unorm"] = self.args.max_unorm
        config["skip_zeros"] = self.args.skip_zeros

        # 如果索引在管理器中存在，则更新配置
        if (gindex, pindex) in self.mng.index2config:
            config.update(self.mng.index2config[(gindex, pindex)])
        # 返回配置信息
        return config
    # 初始化状态的方法，需要被子类重写
    def init_state(self, group, p, gindex, pindex):
        raise NotImplementedError("init_state method needs to be overridden")

    # 更新步骤的方法，需要被子类重写
    def update_step(self, group, p, gindex, pindex):
        raise NotImplementedError(
            "The update_step method needs to be overridden"
        )

    # 获取状态缓冲区的方法，根据是否分页和张量大小来返回不同的结果
    def get_state_buffer(self, p, dtype=torch.float32):
        if not self.is_paged or p.numel() < 1e5:
            return torch.zeros_like(p, dtype=dtype, device=p.device)
        else:
            # 大于1 MB时，使用分页的方法获取缓冲区
            buff = F.get_paged(*p.shape, dtype=dtype, device=p.device)
            # 将缓冲区填充为0
            F.fill(buff, 0)
            # 将缓冲区添加到分页管理器的分页张量列表中
            self.page_mng.paged_tensors.append(buff)
            return buff

    # 预取状态的方法，根据是否分页来预取张量
    def prefetch_state(self, p):
        if self.is_paged:
            # 获取状态字典中的状态1
            state = self.state[p]
            s1 = state['state1']
            # 检查状态1是否分页
            is_paged = getattr(s1, 'is_paged', False)
            if is_paged:
                # 预取状态1的张量
                F.prefetch_tensor(state['state1'])
                # 如果状态字典中有状态2，则预取状态2的张量
                if 'state2' in state:
                    F.prefetch_tensor(state['state2'])
# 定义一个名为Optimizer2State的类，继承自Optimizer8bit类
class Optimizer2State(Optimizer8bit):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        optimizer_name,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        max_unorm=0.0,
        skip_zeros=False,
        is_paged=False
    ):
        # 检查学习率是否大于等于0
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 检查epsilon值是否大于等于0
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        # 如果betas是字符串，则解析为列表
        if isinstance(betas, str):
            # 格式：'(beta1, beta2)'
            betas = betas.replace("(", "").replace(")", "").strip().split(",")
            betas = [float(b) for b in betas]
        # 检查betas列表中的值是否在0到1之间
        for i in range(len(betas)):
            if not 0.0 <= betas[i] < 1.0:
                raise ValueError(
                    f"Invalid beta parameter at index {i}: {betas[i]}"
                )
        # 检查权重衰减值是否大于等于0
        if not 0.0 <= weight_decay:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}"
            )
        # 创建包含默认参数的字典
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # 调用父类的初始化函数
        super().__init__(params, defaults, optim_bits, is_paged)

        # 如果args为None，则创建一个空字典
        if args is None:
            args = {}
            # 设置args字典的各个参数值
            args["optim_bits"] = optim_bits
            args["percentile_clipping"] = 100
            args["min_8bit_size"] = min_8bit_size
            args["percentile_clipping"] = percentile_clipping
            args["block_wise"] = block_wise
            args["max_unorm"] = max_unorm
            args["skip_zeros"] = skip_zeros

            # 使用MockArgs类创建self.args对象
            self.args = MockArgs(args)
        else:
            # 否则直接使用传入的args参数
            self.args = args

        # 设置optimizer_name属性
        self.optimizer_name = optimizer_name

    # 装饰器，指示下面的函数不会计算梯度
    @torch.no_grad()
    # 装饰器，指示下面的函数不会计算梯度
    @torch.no_grad()
# 定义一个名为Optimizer1State的类，继承自Optimizer8bit类
class Optimizer1State(Optimizer8bit):
    # 初始化函数，接受一系列参数来配置优化器
    def __init__(
        self,
        optimizer_name,
        params,
        lr=1e-3,
        betas=(0.9, 0.0),
        eps=1e-8,
        weight_decay=0.0,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        max_unorm=0.0,
        skip_zeros=False,
        is_paged=False
    ):
        # 检查学习率是否合法
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 检查 epsilon 值是否合法
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        # 检查 beta 参数是否合法
        for i in range(len(betas)):
            if not 0.0 <= betas[i] < 1.0:
                raise ValueError(
                    f"Invalid beta parameter at index {i}: {betas[i]}"
                )
        # 检查权重衰减值是否合法
        if not 0.0 <= weight_decay:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}"
            )
        # 创建默认参数字典
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # 调用父类的初始化函数
        super().__init__(params, defaults, optim_bits, is_paged)
    
        # 如果参数为空，则设置默认参数
        if args is None:
            args = {}
            args["optim_bits"] = optim_bits
            args["percentile_clipping"] = 100
            args["min_8bit_size"] = min_8bit_size
            args["percentile_clipping"] = percentile_clipping
            args["block_wise"] = block_wise
            args["max_unorm"] = max_unorm
            args["skip_zeros"] = skip_zeros
    
            # 使用 MockArgs 类创建参数对象
            self.args = MockArgs(args)
        else:
            self.args = args
    
        # 设置优化器名称
        self.optimizer_name = optimizer_name
    
    @torch.no_grad()
    # 初始化参数的状态信息
    def init_state(self, group, p, gindex, pindex):
        # 获取指定组、参数在组内的索引的配置信息
        config = self.get_config(gindex, pindex, group)

        # 根据配置信息中的优化器位数选择数据类型
        if config["optim_bits"] == 32:
            dtype = torch.float32
        elif config["optim_bits"] == 8:
            dtype = torch.uint8
        else:
            raise NotImplementedError(
                f'Amount of optimizer bits not supported: {config["optim_bits"]}'
            )

        # 如果参数元素数量小于配置信息中的最小8位大小，则强制使用32位浮点数数据类型
        if p.numel() < config["min_8bit_size"]:
            dtype = torch.float32

        # 获取参数的状态信息
        state = self.state[p]
        state["step"] = 0

        # 根据数据类型不同进行不同的处理
        if dtype == torch.float32 or (
            dtype == torch.uint8 and p.numel() < 4096
        ):
            state["state1"] = self.get_state_buffer(p, dtype=torch.float32)
        elif dtype == torch.uint8:
            if state["step"] == 0:
                if "dynamic" not in self.name2qmap:
                    self.fill_qmap()
                self.name2qmap["dynamic"] = self.name2qmap["dynamic"].to(
                    p.device
                )

            state["state1"] = self.get_state_buffer(p, dtype=torch.uint8)
            state["qmap1"] = self.name2qmap["dynamic"]

            # 根据配置信息中的块状处理标志进行不同的处理
            if config["block_wise"]:
                n = p.numel()
                blocks = n // 2048
                blocks += 1 if n % 2048 > 0 else 0

                state["absmax1"] = torch.zeros(
                    (blocks,), dtype=torch.float32, device=p.device
                )
            else:
                state["max1"] = torch.zeros(
                    (1,), dtype=torch.float32, device=p.device
                )
                state["new_max1"] = torch.zeros(
                    (1,), dtype=torch.float32, device=p.device
                )

        # 如果配置信息中的百分位剪裁小于100，则初始化梯度范数向量
        if config["percentile_clipping"] < 100:
            state["gnorm_vec"] = torch.zeros((100,), device=p.device)

        # 如果配置信息中的最大非规范值大于0，则初始化非规范值向量
        if config["max_unorm"] > 0.0:
            state["unorm_vec"] = torch.zeros((1,), device=p.device)

    # 禁用梯度计算
    @torch.no_grad()
```