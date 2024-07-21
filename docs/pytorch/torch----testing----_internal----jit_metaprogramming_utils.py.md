# `.\pytorch\torch\testing\_internal\jit_metaprogramming_utils.py`

```py
# 忽略类型检查错误
# Torch
# 从 torch.jit.annotations 模块导入 BroadcastingList2 和 BroadcastingList3 类型注解，忽略 F401 错误
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
# 导入 torch.nn.functional 模块，并使用 F 别名
import torch.nn.functional as F
# 导入 torch 核心库
import torch
# 导入 torch.cuda 模块
import torch.cuda
# 导入 torch.jit 核心模块
import torch.jit
# 导入 torch.jit._logging 模块
import torch.jit._logging
# 导入 torch.jit.frontend 模块
import torch.jit.frontend
# 从 torch.testing._internal.common_nn 模块导入 module_tests 和 new_module_tests
from torch.testing._internal.common_nn import module_tests, new_module_tests
# 从 torch.testing._internal.common_utils 模块导入 is_iterable_of_tensors 和 noncontiguous_like 函数
from torch.testing._internal.common_utils import is_iterable_of_tensors, noncontiguous_like

# 导入 collections 模块
import collections
# 从 copy 模块导入 deepcopy 函数
from copy import deepcopy
# 导入 typing 模块中的 Any、Dict、List、Union 类型
from typing import Any, Dict, List, Union
# 导入 math 模块，忽略 F401 错误
import math  # noqa: F401

# Testing utils
# 从 torch 模块中导入 inf 常量
from torch import inf

# 断言默认的 torch 浮点数类型为 float32
assert torch.get_default_dtype() == torch.float32

# 定义常量 L、M、S 分别为 20、10、5
L = 20
M = 10
S = 5

# 定义一个函数，用于递归解包变量 args
def unpack_variables(args):
    if isinstance(args, tuple):
        return tuple(unpack_variables(elem) for elem in args)
    else:
        return args

# 定义一个类 dont_convert，继承自 tuple 类
class dont_convert(tuple):
    pass

# 定义一个命名元组 non_differentiable，包含一个名为 tensor 的字段
non_differentiable = collections.namedtuple('non_differentiable', ['tensor'])

# 定义一个创建输入的函数 create_input，接受多个参数
def create_input(call_args, requires_grad=True, non_contiguous=False, call_kwargs=None, dtype=torch.float, device=None):
    # 如果 call_args 不是元组，则转换为单元素元组
    if not isinstance(call_args, tuple):
        call_args = (call_args,)

    # 定义 map_arg 函数，用于处理每个参数 arg
    def map_arg(arg):
        # 定义 maybe_non_contig 函数，根据 non_contiguous 参数复制非连续张量
        def maybe_non_contig(tensor):
            if not non_contiguous or tensor.numel() < 2:
                return tensor.clone()

            return noncontiguous_like(tensor)

        # 定义 conjugate 函数，返回张量的共轭
        def conjugate(tensor):
            return tensor.conj()

        # 根据 arg 的类型进行不同处理
        if isinstance(arg, (torch.Size, dont_convert)):
            return arg
        elif isinstance(arg, tuple) and len(arg) == 0:
            # 如果 arg 是空元组，则生成一个随机张量，并设置 requires_grad 属性
            var = conjugate(torch.randn((), dtype=dtype, device=device))
            var.requires_grad = requires_grad
            return var
        elif isinstance(arg, tuple) and not isinstance(arg[0], torch.Tensor):
            # 如果 arg 是非张量元组，则生成一个随机张量，并设置 requires_grad 属性
            return conjugate(maybe_non_contig(torch.randn(*arg, dtype=dtype, device=device))).requires_grad_(requires_grad)
        # 如果 arg 是 non_differentiable 类型
        elif isinstance(arg, non_differentiable):
            if isinstance(arg.tensor, torch.Tensor):
                return conjugate(maybe_non_contig(arg.tensor.to(device=device)))
            return conjugate(maybe_non_contig(arg.tensor.to(device=device)))
        elif isinstance(arg, torch.Tensor):
            # 如果 arg 是张量，则根据 dtype 和 device 生成张量，并设置 requires_grad 属性
            if arg.is_complex() != dtype.is_complex:
                raise RuntimeError("User provided tensor is real for a test that runs with complex dtype, ",
                                   "which is not supported for now")
            # 注意：在 detach() 后使用 clone()，以便能够在后续更改 v 的 size/storage
            v = conjugate(maybe_non_contig(arg)).detach().to(device=device).clone()
            v.requires_grad = requires_grad and (v.is_floating_point() or v.is_complex())
            return v
        elif callable(arg):
            # 如果 arg 是可调用对象，则调用 map_arg 函数
            return map_arg(arg(dtype=dtype, device=device))
        else:
            return arg

    # 对 call_args 中的每个参数应用 map_arg 函数，返回结果元组 args_out
    args_out = tuple(map_arg(arg) for arg in call_args)
    # 如果传入了关键字参数 call_kwargs，则将每个关键字参数的值经过 map_arg 函数处理后存入 kwargs_out 字典
    kwargs_out = {k: map_arg(v) for k, v in call_kwargs.items()} if call_kwargs else {}
    # 返回处理后的位置参数 args_out 和关键字参数 kwargs_out
    return args_out, kwargs_out
# 定义了一个名为 nn_functional_tests 的列表，用于存储神经网络功能接口的测试数据
nn_functional_tests = [
    # ('方法名', (输入张量的大小/构造函数), (参数元组表示张量参数的形状), '测试变体名称', (是否跳过梯度测试, 非可融合节点, 可融合节点) for autodiff, 跳过测试的函数映射, 输出映射到应进行梯度检查的部分, 函数的关键字参数)
    ('conv1d', (S, S, S), ((S, S, S),)),
    ('conv2d', (S, S, S, S), ((S, S, S, S),)),
    ('conv3d', (S, S, S, S, S), ((S, S, S, S, S),)),
    ('conv_transpose1d', (S, S, S), ((S, S, S),)),
    ('conv_transpose2d', (S, S, S, S), ((S, S, S, S),)),
    ('conv_transpose3d', (S, S, S, S, S), ((S, S, S, S, S),)),
    ('conv_tbc', (S, S, S), ((S, S, S), (S,), 2)),
    ('avg_pool1d', (S, S, S), (3,)),
    ('avg_pool2d', (S, S, S, S), (3,), '', (True,)),
    ('avg_pool3d', (S, S, S, S, S), (3,)),
    ('fractional_max_pool2d', (S, S, S, S), (3, [2, 3],)),
    ('max_pool1d', (S, S, S), (2, 1)),
    ('max_pool1d', (S, S, S), (2, 1, 1, 1, False, True), 'with_indices'),
    ('max_pool2d', (S, S, S, S), (2, 1), '', (True, 'aten::max_pool2d_with_indices')),
    ('max_pool2d', (S, S, S, S), (2, 1, 1, 1, False, True), 'with_indices', (True, 'aten::max_pool2d_with_indices')),
    ('max_pool3d', (S, S, S, S, S), (2, 1)),
    ('max_unpool1d', torch.tensor([[[2., 4]]]), (torch.tensor([[[1, 3]]]), 2, 2, 0)),
    ('max_unpool2d', torch.tensor([[[[2., 4]]]]), (torch.tensor([[[[1, 3]]]]), 2, 2, 0)),
    ('max_unpool3d', torch.tensor([[[[[2., 4]]]]]), (torch.tensor([[[[[1, 3]]]]]), 2, 2, 0)),
    ('lp_pool1d', (S, S, S), (2., 3, 2,)),
    ('lp_pool2d', (S, S, S, S), (2., 3, 2,)),
    ('lp_pool3d', (S, S, S, S, S), (2., 3, 2,)),
    ('adaptive_max_pool1d', (S, S, S), (5,)),
    ('adaptive_max_pool2d', (S, S, S, S), ([5, 7],)),
    ('adaptive_max_pool3d', (S, S, S, S, S), ([3, 2, 2],)),
    ('adaptive_avg_pool1d', (S, S, S), (5,), '', (True,)),
    ('adaptive_avg_pool2d', (S, S, S, S), ([5, 7],), '', (True,)),
    ('adaptive_avg_pool3d', (S, S, S, S, S), ([3, 2, 2],), '', (True,)),
    ('dropout', (S, S, S), (0.5,), '', (True, 'aten::native_dropout')),
    ('alpha_dropout', (S, S, S), (0.5,)),
    ('dropout2d', (S, S, S), (0.5,)),
    ('dropout2d', (S, S, S, S), (0.5,), 'batched'),
    ('dropout3d', (S, S, S, S), (0.5,)),
    ('dropout3d', (S, S, S, S, S), (0.5,), 'batched'),
    ('feature_alpha_dropout', (S, S, S), (0.5,)),
    ('threshold', (S, S, S), (0.1, 2.), '', (True,)),
    ('threshold', (S, S, S), (0.1, 2., True), 'inplace'),
    ('relu', (S, S, S), (), '', (True,)),
    ('relu', (S, S, S), (), 'inplace'),
]
    ('glu', (S - 1, S - 1, S - 1), (),),
    # 使用 glu 激活函数，输入形状为 (S-1, S-1, S-1)，无额外参数

    ('hardtanh', (S, S, S), (-0.5, 0.5), '', (True,)),
    # 使用 hardtanh 激活函数，输入形状为 (S, S, S)，参数为 (-0.5, 0.5)，无额外名称和 inplace 参数

    ('hardtanh', (S, S, S), (-0.5, 0.5, True), 'inplace'),
    # 使用 hardtanh 激活函数，输入形状为 (S, S, S)，参数为 (-0.5, 0.5)，并设置 inplace 参数为 True

    ('relu6', (S, S, S), (), '', (True,)),
    # 使用 relu6 激活函数，输入形状为 (S, S, S)，无额外参数和名称，设置 inplace 参数为 True

    ('relu6', (S, S, S), (True), 'inplace'),
    # 使用 relu6 激活函数，输入形状为 (S, S, S)，参数为 True，并设置 inplace 参数为 True

    ('elu', (S, S, S), (0.9,),),
    # 使用 elu 激活函数，输入形状为 (S, S, S)，参数为 (0.9)，无额外名称

    ('elu', (S, S, S), (0.9, True), 'inplace'),
    # 使用 elu 激活函数，输入形状为 (S, S, S)，参数为 (0.9)，并设置 inplace 参数为 True

    ('selu', (S, S, S), (),),
    # 使用 selu 激活函数，输入形状为 (S, S, S)，无额外参数

    ('selu', (S, S, S), (True), 'inplace'),
    # 使用 selu 激活函数，输入形状为 (S, S, S)，参数为 True，并设置 inplace 参数为 True

    ('celu', (S, S, S), (0.9,),),
    # 使用 celu 激活函数，输入形状为 (S, S, S)，参数为 (0.9)，无额外名称

    ('celu', (S, S, S), (0.9, True), 'inplace'),
    # 使用 celu 激活函数，输入形状为 (S, S, S)，参数为 (0.9)，并设置 inplace 参数为 True

    ('leaky_relu', (S, S, S), (0.02,), '', (True,)),
    # 使用 leaky_relu 激活函数，输入形状为 (S, S, S)，参数为 (0.02)，无额外名称和 inplace 参数，设置 inplace 参数为 True

    ('leaky_relu', (S, S, S), (0.02,), 'inplace'),
    # 使用 leaky_relu 激活函数，输入形状为 (S, S, S)，参数为 (0.02)，并设置 inplace 参数为 True

    ('rrelu', (S, S), (0.1, 0.3, False),),
    # 使用 rrelu 激活函数，输入形状为 (S, S)，参数为 (0.1, 0.3, False)

    ('rrelu', (S, S), (0.1, 0.3, False, True), 'inplace'),
    # 使用 rrelu 激活函数，输入形状为 (S, S)，参数为 (0.1, 0.3, False, True)，并设置 inplace 参数为 True

    ('hardshrink', (S, S, S), (0.4,), '', (True,)),
    # 使用 hardshrink 激活函数，输入形状为 (S, S, S)，参数为 (0.4)，无额外名称和 inplace 参数，设置 inplace 参数为 True

    ('tanhshrink', (S, S, S), (),),
    # 使用 tanhshrink 激活函数，输入形状为 (S, S, S)，无额外参数

    ('softsign', (S, S, S), (),),
    # 使用 softsign 激活函数，输入形状为 (S, S, S)，无额外参数

    ('softplus', (S, S, S), (), '', (True,)),
    # 使用 softplus 激活函数，输入形状为 (S, S, S)，无额外名称，设置 inplace 参数为 True

    ('softmin', (S, S, S), (0,),),
    # 使用 softmin 激活函数，输入形状为 (S, S, S)，参数为 (0)，无额外名称

    ('softmax', (S, S, S), (0,), '', (True,)),
    # 使用 softmax 激活函数，输入形状为 (S, S, S)，参数为 (0)，设置 inplace 参数为 True

    ('softmax', (S, S, S), (0, 3, torch.double), 'with_all_args', (True,)),
    # 使用 softmax 激活函数，输入形状为 (S, S, S)，参数为 (0, 3, torch.double)，设置额外名称为 'with_all_args'，设置 inplace 参数为 True

    ('tanh', (S, S, S), (), '', (True,)),
    # 使用 tanh 激活函数，输入形状为 (S, S, S)，无额外名称，设置 inplace 参数为 True

    ('sigmoid', (S, S, S), (), '', (True,)),
    # 使用 sigmoid 激活函数，输入形状为 (S, S, S)，无额外名称，设置 inplace 参数为 True

    ('silu', (S, S, S), (), '', (True,)),
    # 使用 silu 激活函数（也称为 swish），输入形状为 (S, S, S)，无额外名称，设置 inplace 参数为 True

    ('log_softmax', (S, S, S), (0,), '', (True,)),
    # 使用 log_softmax 激活函数，输入形状为 (S, S, S)，参数为 (0)，设置 inplace 参数为 True

    ('linear', (S, S), ((M, S),), '', (True, ['aten::linear'])),
    # 使用线性层，输入形状为 (S, S)，参数为 ((M, S)，)，设置额外名称为 ['aten::linear']，设置 inplace 参数为 True

    ('linear', (S, S), ((M, S), (M,)), 'addmm', (True, ['aten::linear'])),
    # 使用线性层，输入形状为 (S, S)，参数为 ((M, S)，(M,))，设置额外名称为 'addmm'，设置 inplace 参数为 True

    ('bilinear', (S, S, S), ((S, S, M), torch.zeros(M, S, M),),),
    # 使用双线性层，输入形状为 (S, S, S)，参数为 ((S, S, M)，torch.zeros(M, S, M))

    ('embedding', torch.tensor([[1, 2, 4, 5], [4, 3, 2, 5]]), (torch.rand(6, 3), ), '', (True,)),
    # 使用嵌入层，索引张量为 torch.tensor([[1, 2, 4, 5], [4, 3, 2, 5]])，权重张量为 torch.rand(6, 3)，设置 inplace 参数为 True

    ('embedding_bag', torch.tensor([1, 2, 4, 2]), (torch.rand(5, 3), torch.tensor([0, 4]),),),
    # 使用嵌入袋层，索引张量为 torch.tensor([1, 2, 4, 2])，权重张量为 torch.rand(5, 3)，偏置索引张量为 torch.tensor([0, 4])

    ('batch_norm', (S, S),
        (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)), None, None, True, ),
        'training', (True, 'aten::_batch_norm_impl_index')),
    # 使用批归一化层，输入形状为 (S, S)，均值张量为 non_differentiable(torch.randn(S))，方差张量为 non_differentiable(torch.ones(S))，无偏置和缩放张量，设置 training 参数为 True，设置额外名称为 'aten::_batch_norm_impl_index'

    ('batch_norm', (0, S, S, S),
        (
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
                                non_differentiable(torch.randn(S)), None, True, ),
            'with_only_weight_training', (True, 'aten::_batch_norm_impl_index')),
    
    # 执行批归一化操作，用于仅权重训练，输入为随机生成的张量和全为1的张量
    # 'aten::_batch_norm_impl_index' 是批归一化操作的实现索引
    
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
                                None, None, False, ),
            'inference', (True, 'aten::_batch_norm_impl_index')),
    
    # 执行批归一化操作，用于推断阶段，输入为随机生成的张量和全为1的张量
    # 'aten::_batch_norm_impl_index' 是批归一化操作的实现索引
    
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
                                non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)), False, ),
            'with_weight_and_bias_inference', (True, 'aten::_batch_norm_impl_index')),
    
    # 执行批归一化操作，用于带有权重和偏置的推断阶段，输入为随机生成的张量和全为1的张量
    # 'aten::_batch_norm_impl_index' 是批归一化操作的实现索引
    
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
                                None, non_differentiable(torch.ones(S)), False, ),
            'with_only_bias_inference', (True, 'aten::_batch_norm_impl_index')),
    
    # 执行批归一化操作，用于仅偏置推断阶段，输入为随机生成的张量和全为1的张量
    # 'aten::_batch_norm_impl_index' 是批归一化操作的实现索引
    
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
                                non_differentiable(torch.randn(S)), None, False, ),
            'with_only_weight_inference', (True, 'aten::_batch_norm_impl_index')),
    
    # 执行批归一化操作，用于仅权重推断阶段，输入为随机生成的张量和全为1的张量
    # 'aten::_batch_norm_impl_index' 是批归一化操作的实现索引
    
    ('instance_norm', (S, S, S), (non_differentiable(torch.zeros(S)), non_differentiable(torch.ones(S))),),
    
    # 执行实例归一化操作，输入为随机生成的张量和全为1的张量
    
    ('layer_norm', (S, S, S, S), ([5],), '',
         (False, ['aten::contiguous', 'aten::_batch_norm_impl_index'])),
    
    # 执行层归一化操作，输入为维度为5的张量列表，无权重和偏置，包含的操作有'aten::contiguous'和'aten::_batch_norm_impl_index'
    
    ('layer_norm', (S, S, S, S), ([5], non_differentiable(torch.rand(S)),), 'with_only_weight',
         (False, ['aten::contiguous', 'aten::_batch_norm_impl_index'])),
    
    # 执行层归一化操作，输入为维度为5的张量列表和随机生成的张量，用于仅权重的操作，包含的操作有'aten::contiguous'和'aten::_batch_norm_impl_index'
    
    ('layer_norm', (S, S, S, S), ([5], None, non_differentiable(torch.rand(S)),), 'with_only_bias',
         (False, ['aten::contiguous', 'aten::_batch_norm_impl_index'])),
    
    # 执行层归一化操作，输入为维度为5的张量列表和随机生成的张量，用于仅偏置的操作，包含的操作有'aten::contiguous'和'aten::_batch_norm_impl_index'
    
    ('layer_norm', (S, S, S, S), ([5], non_differentiable(torch.rand(S)),
                                      non_differentiable(torch.rand(S))), 'with_weight_and_bias',
         (False, ['aten::contiguous', 'aten::_batch_norm_impl_index', 'aten::addcmul'])),
    
    # 执行层归一化操作，输入为维度为5的张量列表和随机生成的张量，用于权重和偏置的操作，包含的操作有'aten::contiguous'、'aten::_batch_norm_impl_index'和'aten::addcmul'
    
    ('group_norm', (S, S, S), (1, torch.rand(5)),),
    
    # 执行组归一化操作，输入为1和维度为5的随机生成的张量
    
    ('local_response_norm', (S, S, S), (2,),),
    
    # 执行局部响应归一化操作，输入为2
    
    ('nll_loss', F.log_softmax(torch.randn(3, 5), dim=0), (torch.tensor([1, 0, 4]),), '',),
    
    # 执行负对数似然损失操作，输入为对第0维进行对数softmax处理的随机生成张量和标签张量[1, 0, 4]
    
    ('poisson_nll_loss', torch.rand(S, 2), (torch.rand(S, 2),),),
    
    # 执行泊松负对数似然损失操作，输入为随机生成的维度为Sx2的张量和同样维度的随机生成张量
    
    ('poisson_nll_loss', torch.rand(S, 2), (torch.rand(S, 2), True, True), 'full'),
    
    # 执行泊松负对数似然损失操作，输入为随机生成的维度为Sx2的张量、同样维度的随机生成张量和两个布尔值参数，用于完整模式
    
    ('kl_div', F.log_softmax(torch.randn(S, 10), 1), (F.softmax(torch.randn(S, 10), 1),),),
    
    # 执行KL散度操作，输入为对第1维进行对数softmax处理的随机生成张量和对第1维进行softmax处理的随机生成张量
    
    ('cross_entropy', (3, S), (torch.randint(S, (3,), dtype=torch.int64),),),
    
    # 执行交叉熵损失操作，输入为维度为(3, S)的张量和维度为3的随机整数张量
    
    ('binary_cross_entropy_with_logits', (3,), (torch.empty(3).random_(2), ),),
    
    # 执行带logits的二元交叉熵损失操作，输入为维度为3的张量和随机生成的0或1的张量
    
    ('smooth_l1_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    
    # 执行平滑L1损失操作，输入为随机生成的维度为(3, S)的张量
    
    ('huber_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    
    # 执行Huber损失操作，输入为随机生成的维度为(3, S)的张量
    
    ('l1_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    
    # 执行L1损失操作，输入为随机生成的维度为(3, S)的张量
    
    ('mse_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    
    # 执行均方误差损失操作，输入为随机生成的维度为(3, S)的张量
    # 调用 PyTorch 的 smooth_l1_loss 函数，计算平滑 L1 损失
    ('smooth_l1_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    
    # 调用 PyTorch 的 huber_loss 函数，计算 Huber 损失
    ('huber_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    
    # 调用 PyTorch 的 l1_loss 函数，计算 L1 损失
    ('l1_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    
    # 调用 PyTorch 的 mse_loss 函数，计算均方误差损失
    ('mse_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    
    # 调用 PyTorch 的 margin_ranking_loss 函数，计算排名损失
    ('margin_ranking_loss', (S,), ((S,), (S,)),),
    
    # 调用 PyTorch 的 hinge_embedding_loss 函数，计算铰链嵌入损失
    ('hinge_embedding_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    
    # 调用 PyTorch 的 soft_margin_loss 函数，计算软边损失
    ('soft_margin_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    
    # 调用 PyTorch 的 multilabel_soft_margin_loss 函数，计算多标签软边损失
    ('multilabel_soft_margin_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    
    # 调用 PyTorch 的 cosine_embedding_loss 函数，计算余弦嵌入损失
    ('cosine_embedding_loss', (S, S), ((S, S), non_differentiable(torch.rand(S,))),),
    
    # 调用 PyTorch 的 pixel_shuffle 函数，对像素进行混洗
    ('pixel_shuffle', (1, 9, 4, 4), (3,),),
    
    # 调用 PyTorch 的 pixel_unshuffle 函数，对像素进行反混洗
    ('pixel_unshuffle', (1, 1, 12, 12), (3,),),
    
    # 调用 PyTorch 的 affine_grid 函数，生成仿射变换的网格
    ('affine_grid', (S, 2, 3), (torch.Size([S, 1, 7, 7]),),),
    
    # 调用 PyTorch 的 pad 函数，对张量进行填充
    ('pad', (3, 3, 4, 2), ([1, 1],),),
    
    # 调用 PyTorch 的 pairwise_distance 函数，计算成对距离
    ('pairwise_distance', (S, S), ((S, S),),),
    
    # 调用 PyTorch 的 pdist 函数，计算张量的成对距离
    ('pdist', (S, S), (),),
    
    # 调用 PyTorch 的 cosine_similarity 函数，计算余弦相似度
    ('cosine_similarity', (S, S), ((S, S),),),
    
    # 调用 PyTorch 的 triplet_margin_loss 函数，计算三元组间隔损失
    ('triplet_margin_loss', (S, S), ((S, S), (S, S)),),
    
    # 调用 PyTorch 的 normalize 函数，对张量进行归一化
    ('normalize', (S, S, S), (),),
    
    # 调用 PyTorch 的 unfold 函数，对张量进行展开
    ('unfold', (S, S, S, S), ([2, 3]),),
    
    # 调用 PyTorch 的 fold 函数，对张量进行折叠
    ('fold', (1, 3 * 2 * 2, 12), ([4, 5], [2, 2]),),
    
    # 调用 PyTorch 的 grid_sample 函数，对网格进行采样
    ('grid_sample', (S, S, S, S), (non_differentiable(torch.rand(S, S, S, 2)),),),
    
    # 调用 PyTorch 的 gumbel_softmax 函数，实现 Gumbel Softmax 操作
    ('gumbel_softmax', (S, S), (2.,), '', (True, ['aten::softmax', 'aten::add', 'aten::div'], ['aten::neg'])),
    
    # 调用 PyTorch 的 gumbel_softmax 函数，实现硬化的 Gumbel Softmax 操作
    ('gumbel_softmax', (S, S), (2., True,), 'hard', (True, ['aten::softmax', 'aten::add', 'aten::div'], ['aten::neg'])),
    
    # 调用 PyTorch 的 multilabel_margin_loss 函数，计算多标签间隔损失
    ('multilabel_margin_loss', torch.tensor([[0.2, -0.2, 0.07]]), (torch.tensor([[0, 0, 1]]),),),
    
    # 调用 PyTorch 的 multi_margin_loss 函数，计算多间隔损失
    ('multi_margin_loss', (S, S), (non_differentiable(torch.randint(S, (S, ), dtype=torch.int64)),
                                   1, 1., non_differentiable(torch.randn(S))),),
    
    # 调用 PyTorch 的 binary_cross_entropy 函数，计算二元交叉熵损失
    ('binary_cross_entropy', torch.randn(3, 2).sigmoid(), (non_differentiable(torch.rand(3, 2)),
                                                           non_differentiable(torch.randn(3, 2))),),
    
    # 调用 PyTorch 的 binary_cross_entropy 函数，计算二元交叉熵损失（带参数）
    ('binary_cross_entropy', torch.randn(3, 2).sigmoid(),
        (non_differentiable(torch.rand(3, 2)),
         non_differentiable(torch.randn(3, 2)), None, None, 'mean'), 'size_average'),
    
    # 调用 PyTorch 的 ctc_loss 函数，计算连接时间分类损失
    ('ctc_loss', torch.rand(S, S, S).log_softmax(2).detach().requires_grad_(),
     (torch.randint(1, S, (S, S), dtype=torch.long), torch.full((S,), S, dtype=torch.long),
      torch.randint(1, S, (S,), dtype=torch.long))),
    
    # 调用 PyTorch 的 upsample 函数，进行上采样操作（带比例）
    ('upsample', torch.randn(S, S, M, M), (None, 2.), 'with_scale'),
    
    # 调用 PyTorch 的 upsample 函数，进行上采样操作（带尺寸）
    ('upsample', torch.randn(S, S, M, M), (4,), 'with_size'),
    
    # 调用 PyTorch 的 interpolate 函数，进行插值操作（最近邻，四维）
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,), 'nearest_4d'),
    
    # 调用 PyTorch 的 interpolate 函数，进行插值操作（最近邻，四维，带比例）
    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'nearest_4d_with_scale'),
    
    # 调用 PyTorch 的 interpolate 函数，进行插值操作（最近邻，四维，带尺寸）
    ('interpolate', torch.randn(S, S, M, M), (4,), 'nearest_4d_with_size'),
    
    # 调用 PyTorch 的 interpolate 函数，进行插值操作（区域插值，四维）
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,), 'area_4d'),
    
    # 调用 PyTorch 的 interpolate 函数，进行插值操作（区域插值，四维，带比例）
    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'area_4d_with_scale'),
    
    # 调用 PyTorch 的 interpolate 函数，进行插值操作（区域插值，四维，带尺
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,), 'bilinear_4d'),
    # 使用双线性插值对一个形状为 (1, 1, 3, 3) 的全零张量进行插值操作，输出结果的形状为 (2, 2, 3, 3)

    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'bilinear_4d_with_scale'),
    # 使用双线性插值对一个形状为 (S, S, M, M) 的正态分布张量进行插值操作，按比例因子 2 进行缩放

    ('interpolate', torch.randn(S, S, M, M), (4,), 'bilinear_4d_with_size'),
    # 使用双线性插值对一个形状为 (S, S, M, M) 的正态分布张量进行插值操作，目标输出大小为 4

    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,), 'bicubic_4d'),
    # 使用双三次插值对一个形状为 (1, 1, 3, 3) 的全零张量进行插值操作，输出结果的形状为 (2, 2, 3, 3)

    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'bicubic_4d_with_scale'),
    # 使用双三次插值对一个形状为 (S, S, M, M) 的正态分布张量进行插值操作，按比例因子 2 进行缩放

    ('interpolate', torch.randn(S, S, M, M), (4,), 'bicubic_4d_with_size'),
    # 使用双三次插值对一个形状为 (S, S, M, M) 的正态分布张量进行插值操作，目标输出大小为 4

    ('interpolate', torch.zeros(3, 3).view(1, 3, 3), (2,), 'nearest_3d'),
    # 使用最近邻插值对一个形状为 (1, 3, 3) 的全零张量进行插值操作，输出结果的形状为 (2, 3, 3)

    ('interpolate', torch.randn(S, M, M), (None, 2.), 'nearest_3d_with_scale'),
    # 使用最近邻插值对一个形状为 (S, M, M) 的正态分布张量进行插值操作，按比例因子 2 进行缩放

    ('interpolate', torch.randn(S, M, M), (4,), 'nearest_3d_with_size'),
    # 使用最近邻插值对一个形状为 (S, M, M) 的正态分布张量进行插值操作，目标输出大小为 4

    ('interpolate', torch.zeros(3, 3).view(1, 3, 3), (2,), 'area_3d'),
    # 使用区域插值对一个形状为 (1, 3, 3) 的全零张量进行插值操作，输出结果的形状为 (2, 3, 3)

    ('interpolate', torch.randn(S, M, M), (None, 2.), 'area_3d_with_scale'),
    # 使用区域插值对一个形状为 (S, M, M) 的正态分布张量进行插值操作，按比例因子 2 进行缩放

    ('interpolate', torch.randn(S, M, M), (4,), 'area_3d_with_size'),
    # 使用区域插值对一个形状为 (S, M, M) 的正态分布张量进行插值操作，目标输出大小为 4

    ('interpolate', torch.zeros(3, 3).view(1, 3, 3), (2,), 'linear_3d'),
    # 使用线性插值对一个形状为 (1, 3, 3) 的全零张量进行插值操作，输出结果的形状为 (2, 3, 3)

    ('interpolate', torch.randn(S, M, M), (None, 2.), 'linear_3d_with_scale'),
    # 使用线性插值对一个形状为 (S, M, M) 的正态分布张量进行插值操作，按比例因子 2 进行缩放

    ('interpolate', torch.randn(S, M, M), (4,), 'linear_3d_with_size'),
    # 使用线性插值对一个形状为 (S, M, M) 的正态分布张量进行插值操作，目标输出大小为 4

    ('interpolate', torch.randn(S, M, M, M, M), (None, 2.), 'nearest_5d_with_scale'),
    # 使用最近邻插值对一个形状为 (S, M, M, M, M) 的正态分布张量进行插值操作，按比例因子 2 进行缩放

    ('interpolate', torch.randn(S, M, M, M, M), (4,), 'nearest_5d_with_size'),
    # 使用最近邻插值对一个形状为 (S, M, M, M, M) 的正态分布张量进行插值操作，目标输出大小为 4

    ('interpolate', torch.zeros(3, 3, 3).view(1, 1, 3, 3, 3), (2,), 'area_5d'),
    # 使用区域插值对一个形状为 (1, 1, 3, 3, 3) 的全零张量进行插值操作，输出结果的形状为 (2, 1, 3, 3, 3)

    ('interpolate', torch.randn(S, M, M, M, M), (None, 2.), 'area_5d_with_scale'),
    # 使用区域插值对一个形状为 (S, M, M, M, M) 的正态分布张量进行插值操作，按比例因子 2 进行缩放

    ('interpolate', torch.randn(S, M, M, M, M), (4,), 'area_5d_with_size'),
    # 使用区域插值对一个形状为 (S, M, M, M, M) 的正态分布张量进行插值操作，目标输出大小为 4

    ('interpolate', torch.zeros(3, 3, 3).view(1, 1, 3, 3, 3), (2,), 'trilinear_5d'),
    # 使用三线性插值对一个形状为 (1, 1, 3, 3, 3) 的全零张量进行插值操作，输出结果的形状为 (2, 1, 3, 3, 3)

    ('interpolate', torch.randn(S, M, M, M, M), (None, 2.), 'trilinear_5d_with_scale'),
    # 使用三线性插值对一个形状为 (S, M, M, M, M) 的正态分布张量进行插值操作，按比例因子 2 进行缩放

    ('interpolate', torch.randn(S, M, M, M, M), (4,), 'trilinear_5d_with_size'),
    # 使用三线性插值对一个形状为 (S, M, M, M, M) 的正态分布张量进行插值操作，目标输出大小为 4

    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2, None, 'nearest', None, False),
     'nearest_4d_not_recompute_scale_factor'),
    # 使用最近邻插值对一个形状为 (1, 1, 3, 3) 的全零张量进行插值操作，禁止重新计算比例因子

    ('interpolate', torch.randn(S, S, M, M), (4, None, 'nearest', None, False),
     'nearest_4d_with_size_not_recompute_scale_factor'),
    # 使用最近邻插值对一个形状为 (S, S, M, M) 的正态分布
    # 调用 interpolate 函数进行张量的插值操作，以下是几种不同情况的示例：

    # 三维张量插值，使用线性插值方法，指定缩放因子但不重新计算缩放因子
    ('interpolate', torch.randn(S, M, M), (None, 2., 'linear', None, False),
     'linear_3d_with_scale_not_recompute_scale_factor'),

    # 三维张量插值，使用线性插值方法，指定目标大小但不重新计算缩放因子
    ('interpolate', torch.randn(S, M, M), (4, None, 'linear', None, False),
     'linear_3d_with_size_not_recompute_scale_factor'),

    # 五维张量插值，使用最近邻插值方法，指定缩放因子但不重新计算缩放因子
    ('interpolate', torch.randn(S, M, M, M, M), (None, 2., 'nearest', None, False),
     'nearest_5d_with_scale_not_recompute_scale_factor'),

    # 五维张量插值，使用最近邻插值方法，指定目标大小但不重新计算缩放因子
    ('interpolate', torch.randn(S, M, M, M, M), (4, None, 'nearest', None, False),
     'nearest_5d_with_size_not_recompute_scale_factor'),

    # 五维张量插值，使用三线性插值方法，指定缩放因子但不重新计算缩放因子
    ('interpolate', torch.randn(S, M, M, M, M), (None, 2., 'trilinear', None, False),
     'trilinear_5d_with_scale_not_recompute_scale_factor'),

    # 五维张量插值，使用三线性插值方法，指定目标大小但不重新计算缩放因子
    ('interpolate', torch.randn(S, M, M, M, M), (4, None, 'trilinear', None, False),
     'trilinear_5d_with_size_not_recompute_scale_factor'),
# 定义了一个脚本模板字符串，用于生成脚本函数
script_template = '''
def the_method({}):
    return {}
'''

# 将值转换为其字面量表示
def value_to_literal(value):
    if isinstance(value, str):
        # 将字符串加引号并转义特殊字符
        return ascii(value)
    if isinstance(value, torch.Tensor):
        # 返回表示 Torch 张量的字符串
        return 'torch.' + str(value)
    else:
        # 将值转换为字符串表示
        return str(value)

# 根据方法名、函数类型、参数和关键字参数生成调用字符串
def get_call(method_name, func_type, args, kwargs):
    kwargs_str = ', '.join([k + '=' + value_to_literal(v) for k, v in kwargs.items()])
    self_arg = args[0]
    if func_type == 'method':
        args = args[1:]

    argument_str = ', '.join(args)
    argument_str += ', ' if len(args) and len(kwargs) else ''
    argument_str += kwargs_str

    if func_type == 'functional' or func_type == 'function':
        # Torch 函数调用字符串
        call = f'torch.{method_name}({argument_str})'
    elif func_type == 'method':
        # 对象方法调用字符串
        call = f'{self_arg}.{method_name}({argument_str})'
    elif func_type == 'nn_functional':
        # Torch 的 nn.functional 函数调用字符串
        call = f'torch.nn.functional.{method_name}({argument_str})'
    else:
        # 不支持的函数类型错误
        raise TypeError('Unsupported function type')

    return call

# 获取常量的字符串表示
def get_constant(x):
    if x == inf:
        return 'math.inf'
    if x == -inf:
        return '-math.inf'
    return x

# 获取脚本函数的参数
def get_script_args(args):
    formals: List[str] = []
    tensors: List[Union[torch.Tensor, List[torch.Tensor]]] = []
    actuals: List[str] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            # 如果参数是 Torch 张量，生成一个形式参数和一个实际参数名，并将张量添加到列表中
            name = f'i{len(formals)}'
            formals.append(name)
            actuals.append(name)
            tensors.append(arg)
        elif is_iterable_of_tensors(arg):
            # 如果参数是张量的可迭代对象，生成一个形式参数和一个实际参数名，并将张量列表添加到列表中
            name = f'i{len(formals)}'
            formals.append(name + ': List[torch.Tensor]')
            actuals.append(name)
            tensors.append(list(arg))
        elif isinstance(arg, str):
            # 如果参数是字符串，将其加上引号
            actuals.append(f"'{arg}'")
        else:
            # 否则，将其转换为常量的字符串表示
            actuals.append(str(get_constant(arg)))
    return (formals, tensors, actuals)

# 生成一个脚本函数及其输入参数的示例
def gen_script_fn_and_args(method_name, func_type, *args, **kwargs):
    formals, tensors, actuals = get_script_args(args)
    call = get_call(method_name, func_type, actuals, kwargs)
    script = script_template.format(', '.join(formals), call)
    CU = torch.jit.CompilationUnit(script)
    return CU.the_method, tensors

# 创建一个脚本函数，返回一个函数，接受(args, kwargs)并运行编译后的函数
def create_script_fn(self, method_name, func_type):
    # 函数返回原始输出和用于梯度检查的过滤输出的元组
    def script_fn(*args, **kwargs):
        # 调用 gen_script_fn_and_args 函数生成脚本函数及其参数
        fn, tensors = gen_script_fn_and_args(method_name, func_type, *args, **kwargs)
        # 调用 assertExportImport 方法断言导出和导入，验证脚本函数的正确性
        self.assertExportImport(fn.graph, tensors)
        # 执行生成的脚本函数 fn，并传入参数 tensors，获取输出结果
        output = fn(*tensors)
        # 设置 script_fn 对象的 last_graph 属性为 fn 对应的计算图，忽略类型注解
        script_fn.last_graph = fn.graph_for(*tensors)  # type: ignore[attr-defined]
        # 返回脚本函数的执行结果
        return output
    # 返回生成的 script_fn 函数作为结果
    return script_fn
# 定义一个名为 SplitInputs 的类，用于处理函数参数的分类和分离
class SplitInputs:
    # 所有张量类型的参数列表
    all_tensors: List[Any]
    # 张量类型的普通参数列表
    tensor_args: List[Any]
    # 非张量类型的普通参数列表
    nontensor_args: List[Any]
    # 参数类型标记列表（'t' 表示张量，'s' 表示非张量）
    arg_types: List[str]
    # 张量类型的关键字参数字典
    tensor_kwargs: Dict[str, Any]
    # 关键字参数的顺序列表
    kwarg_order: List[str]
    # 非张量类型的关键字参数字典
    nontensor_kwargs: Dict[str, Any]
    # 关键字参数类型标记字典（'t' 表示张量，'s' 表示非张量）
    kwarg_types: Dict[str, Any]

    # 判断参数是否为张量输入的静态方法
    @staticmethod
    def _is_tensor_input(arg):
        return isinstance(arg, torch.Tensor) or is_iterable_of_tensors(arg)

    # 初始化方法，根据传入的普通参数和关键字参数对各类参数进行分类
    def __init__(self, args, kwargs):
        # 根据普通参数的类型（张量或非张量）生成标记列表
        self.arg_types = ['t' if self._is_tensor_input(arg) else 's' for arg in args]
        # 根据关键字参数的类型（张量或非张量）生成标记字典
        self.kwarg_types = {k: 't' if self._is_tensor_input(v) else 's' for k, v in kwargs.items()}
        # 将所有张量类型的普通参数收集到列表中
        self.tensor_args = [arg for arg in args if self._is_tensor_input(arg)]
        # 将所有非张量类型的普通参数收集到列表中
        self.nontensor_args = [arg for arg in args if not self._is_tensor_input(arg)]
        # 将所有张量类型的关键字参数收集到字典中
        self.tensor_kwargs = {k: v for k, v in kwargs.items() if self._is_tensor_input(v)}
        # 将所有非张量类型的关键字参数收集到字典中
        self.nontensor_kwargs = {k: v for k, v in kwargs.items() if not self._is_tensor_input(v)}
        # 收集所有张量类型的参数（普通参数和关键字参数）
        self.all_tensors = [*self.tensor_args, *[v for k, v in self.tensor_kwargs.items()]]
        # 收集关键字参数的顺序列表
        self.kwarg_order = [k for k, v in kwargs.items()]

    # 检查两个 SplitInputs 对象的非张量参数是否匹配
    def nontensors_match(self, other: 'SplitInputs'):
        # 检查普通参数类型标记列表是否相同
        if self.arg_types != other.arg_types:
            return False
        # 检查关键字参数类型标记字典是否相同
        if self.kwarg_types != other.kwarg_types:
            return False
        # 检查关键字参数顺序列表是否相同
        if self.kwarg_order != other.kwarg_order:
            return False
        # 检查普通参数列表是否相同
        if self.nontensor_args != other.nontensor_args:
            return False
        # 检查非张量类型的关键字参数字典是否相同
        if self.nontensor_kwargs != other.nontensor_kwargs:
            return False
        # 如果所有条件都匹配，则返回 True
        return True

# 创建一个新的函数，其中 'args' 中的所有非张量参数都已经被部分应用，而所有张量参数保持不变
# 用于在某些参数不是张量时跟踪函数的使用
def partial_apply_nontensors(fn, args, kwargs):
    # 使用 SplitInputs 类处理传入的参数，以便分类和分离
    inputs = SplitInputs(args, kwargs)

    # 定义一个新函数 new_fn，接受任意数量的张量参数 tensors_
    def new_fn(*tensors_):
        # 创建一个张量迭代器
        tensors = iter(tensors_)
        # 根据 inputs 中的参数类型标记，将张量和非张量参数重新组合成完整的参数列表 full_args
        full_args = [args[i] if s == 's' else next(tensors) for i, s in enumerate(inputs.arg_types)]
        # 根据 inputs 中的关键字参数类型标记，将张量和非张量关键字参数重新组合成完整的关键字参数字典 full_kwargs
        full_kwargs = {k: kwargs[k] if s == 's' else next(tensors) for k, s in inputs.kwarg_types.items()}
        # 调用原始函数 fn，传入重新组合后的参数和关键字参数，返回结果
        return fn(*full_args, **full_kwargs)

    # 返回新定义的函数 new_fn 和处理过的参数 inputs
    return new_fn, inputs

# 从输入的 fn 创建一个跟踪函数
def create_traced_fn(self, fn, cache_traced_fn=False):
    def traced_fn(*inputs, **kwargs):
        # `check_trace` 设置为 False 是因为 check_trace 是在 @no_grad 下运行的
        # 同时，`check_against_reference` 已经完成了对 Python 函数的所有检查
        # 调用 partial_apply_nontensors 函数，获取函数的张量输入和分割后的输入
        fn_tensors, split_inputs = partial_apply_nontensors(fn, inputs, kwargs)
        
        # 如果不缓存追踪的函数或者 traced_fn 没有 'traced' 属性
        if not cache_traced_fn or not hasattr(traced_fn, 'traced'):
            # 使用 torch.jit.trace 追踪函数的张量版本，不进行追踪检查
            traced = torch.jit.trace(fn_tensors, split_inputs.all_tensors, check_trace=False)
            # 断言导出和导入的正确性
            self.assertExportImport(traced.graph, split_inputs.all_tensors)
            # 使用追踪的函数进行计算
            output = traced(*split_inputs.all_tensors)
            
            # 如果需要缓存追踪的函数，则设置 traced_fn 的属性
            if cache_traced_fn:
                traced_fn.traced = traced
                traced_fn.split_inputs = split_inputs
        else:
            # 断言非张量输入在追踪时与当前输入相同
            self.assertTrue(traced_fn.split_inputs.nontensors_match(split_inputs))
            # 使用缓存的追踪函数进行计算
            output = traced_fn.traced(*split_inputs.all_tensors)
            traced = traced_fn.traced
        
        # 暂时跳过类型注解函数属性，参见：https://github.com/python/mypy/issues/2087
        # 设置 traced_fn 的最后一次计算的图
        traced_fn.last_graph = traced.graph_for(*split_inputs.all_tensors)  # type: ignore[attr-defined]
        # 设置 traced_fn 的图
        traced_fn.graph = traced.graph  # type: ignore[attr-defined]
        
        # 返回计算结果
        return output
    return traced_fn
# known to be failing in script
# 脚本中已知失败的测试用例集合
EXCLUDE_SCRIPT = {
    'test_norm_fro_default',
    'test_norm_fro_cpu',
    'test_norm_nuc',
    'test_norm_fro',
    'test_norm_nuc_batched',

    # aten op has additional cudnn argument
    # aten 操作含有额外的 cudnn 参数
    'test_nn_unfold',

    # flaky test - TODO fix
    # 不稳定的测试 - 待修复
    'test_nn_ctc_loss',

    # unknown builtin op
    # 未知的内置操作
    'test_nn_fold',

    # jit doesn't support sparse tensors.
    # jit 不支持稀疏张量
    'test_to_sparse',
    'test_to_sparse_dim',
}

# generates a script function and set of example inputs
# from a specified test in the format of nn_functional_tests
# 从 nn_functional_tests 格式中的特定测试生成脚本函数和示例输入集合
def get_nn_functional_compiled_fn_and_inputs(name, self_size, args, variant_name='', *extra_args):
    test_name = 'test_nn_' + name

    if variant_name != '':
        test_name = test_name + '_' + variant_name

    no_grad = variant_name == 'inplace'

    # create_input is assumed to create variables from inputs
    # create_input 假设从输入创建变量
    self_variable = create_input((self_size,))[0][0]
    kwargs = None

    # need to record this because methods can change the size (e.g. unsqueeze)
    # 需要记录这个因为方法可以改变大小（例如 unsqueeze）
    args_variable, kwargs_variable = create_input(args)

    # deepcopy is assumed to copy data from variables
    # deepcopy 假设从变量复制数据
    self_tensor = deepcopy(self_variable.data)
    args_tensor = deepcopy(unpack_variables(args_variable))

    f_args_variable = (self_variable,) + args_variable
    f_args_tensor = (self_tensor,) + args_tensor

    # Disable emission hooks temporarily for script generation
    # 暂时禁用发射钩子以进行脚本生成
    with torch._jit_internal._disable_emit_hooks():
        script_fn, inputs = gen_script_fn_and_args(name, "nn_functional", *f_args_variable)
    return script_fn, inputs


# additional modules test
# TODO: delete this list once we make all nn_tests work
# 额外的模块测试
# TODO: 一旦所有 nn_tests 都能正常工作，删除这个列表
additional_module_tests = [
    {
        'module_name': 'Bilinear',
        'constructor_args': (S, S, M),
        'input_size': (S, S),
        'extra_args': ((S, S),)
    },
    {
        'module_name': 'RNNCell',
        'constructor_args': (S, S),
        'input_size': (S, S),
    },
    {
        'module_name': 'LSTMCell',
        'constructor_args': (S, S),
        'input_size': (S, S),
    },
    {
        'module_name': 'GRUCell',
        'constructor_args': (S, S),
        'input_size': (S, S),
    },
    {
        'module_name': 'MultiheadAttention',
        'constructor_args': (128, 8),
        'input_size': (10, 8, 128),
        'extra_args': (torch.randn(10, 8, 128), torch.randn(10, 8, 128)),
        'slowTest': True
    },
    {
        'module_name': 'Transformer',
        'constructor_args': (1, 1, 1, 1, 2),
        'input_size': (3, 1, 1),
        'extra_args': (torch.randn(1, 1, 1),),
        'slowTest': True
    }
]

# EXCLUDE_SCRIPT_MODULES contains names of tests to exclude from script generation
# EXCLUDE_SCRIPT_MODULES 包含需要在脚本生成中排除的测试名称
EXCLUDE_SCRIPT_MODULES = {
    'test_nn_AdaptiveAvgPool2d_tuple_none',
    'test_nn_AdaptiveAvgPool3d_tuple_none',
    'test_nn_AdaptiveMaxPool2d_tuple_none',
    'test_nn_AdaptiveMaxPool3d_tuple_none',

    # Doesn't use future division, so this is not supported
    # 不使用 future division，因此不支持
    'test_nn_CrossMapLRN2d',

    # Derivative for aten::_scaled_dot_product_flash_attention_backward is not implemented
    # aten::_scaled_dot_product_flash_attention_backward 的导数未实现
    'test_nn_TransformerDecoderLayer_gelu_activation',
    'test_nn_TransformerDecoderLayer_relu_activation',
    'test_nn_TransformerEncoderLayer_gelu_activation',
}
    'test_nn_TransformerEncoderLayer_relu_activation',
    'test_nn_Transformer_multilayer_coder',


# 定义一个包含两个字符串的元组，用于测试
'test_nn_TransformerEncoderLayer_relu_activation',
'test_nn_Transformer_multilayer_coder',
}

script_method_template = '''
def forward({}):
    return {}
'''

# 创建脚本模块的方法，接受神经网络模块、构造器参数和其他参数
def create_script_module(self, nn_module, constructor_args, *args, **kwargs):
    # 定义内部函数 script_module，用于创建脚本模块
    def script_module(*args, **kwargs):
        # 获取脚本方法的形参、张量和实参
        formals, tensors, actuals = get_script_args(args)

        # 构建方法参数字符串，包括 self 和实际参数
        method_args = ', '.join(['self'] + actuals)
        call_args_str = ', '.join(actuals)
        # 构建调用字符串，调用 self.submodule() 方法
        call = f"self.submodule({call_args_str})"
        # 使用脚本方法模板，填充方法参数和调用
        script = script_method_template.format(method_args, call)

        submodule_constants = []
        # 如果传入的 kwargs 中有 'is_constant'，则设置 submodule 为常量
        if kwargs.get('is_constant'):
            submodule_constants = ['submodule']

        # 创建使用脚本方法的模块类
        class TheModule(torch.jit.ScriptModule):
            __constants__ = submodule_constants

            def __init__(self):
                super().__init__()
                # 初始化 submodule 属性为传入的神经网络模块
                self.submodule = nn_module(*constructor_args)

        # 内部函数，用于创建模块实例并定义脚本
        def make_module(script):
            module = TheModule()
            # 检查模块的字符串表示
            str(module)
            # 定义模块的脚本方法
            module.define(script)
            return module

        # 调用 make_module 函数创建模块
        module = make_module(script)
        # 如果有 self，则调用 assertExportImportModule 方法验证模块的导出和导入
        if self:
            self.assertExportImportModule(module, tensors)
            # 调用模块实例，执行脚本方法
            module(*args)
        # 设置最后一个图形的属性为模块的图形
        create_script_module.last_graph = module.graph  # type: ignore[attr-defined]
        return module
    return script_module

# 检查别名注释的方法，接受方法名称、参数和关键字参数，以及 ATen 函数名称和函数类型
def check_alias_annotation(method_name, args, kwargs, *, aten_name, func_type='method'):
    # 获取脚本方法的形参、张量和实参
    formals, tensors, actuals = get_script_args(args)
    # 获取方法调用字符串
    call = get_call(method_name, func_type, actuals, kwargs)
    # 使用脚本模板填充形参和调用字符串
    script = script_template.format(', '.join(formals), call)
    # 创建编译单元对象 CU
    CU = torch.jit.CompilationUnit(script)
    # 清理 IR
    torch._C._jit_pass_inline(CU.the_method.graph)
    torch._C._jit_pass_constant_propagation(CU.the_method.graph)
    # 检查别名注释
    torch._C._jit_check_alias_annotation(CU.the_method.graph, tuple(tensors), aten_name)

# 从关键字参数中获取神经网络模块名称
def get_nn_module_name_from_kwargs(**kwargs):
    if 'module_name' in kwargs:
        return kwargs['module_name']
    elif 'fullname' in kwargs:
        return kwargs['fullname']
    elif 'constructor' in kwargs:
        return kwargs['constructor'].__name__

# 从关键字参数中获取神经网络模块测试名称
def get_nn_mod_test_name(**kwargs):
    if 'fullname' in kwargs:
        test_name = kwargs['fullname']
    else:
        test_name = get_nn_module_name_from_kwargs(**kwargs)
        if 'desc' in kwargs:
            test_name = f"{test_name}_{kwargs['desc']}"
    return f'test_nn_{test_name}'

# 从关键字参数中获取神经网络模块类名
def get_nn_module_class_from_kwargs(**kwargs):
    name = get_nn_module_name_from_kwargs(**kwargs)
    index = name.find("_")
    if index == -1:
        return name
    else:
        return name[0:name.find("_")]

# 尝试获取编译模块和输入的神经网络模块名称
def try_get_nn_module_compiled_mod_and_inputs(*args, **kwargs):
    name = get_nn_module_name_from_kwargs(**kwargs)

    if 'desc' in kwargs and 'eval' in kwargs['desc']:
        # 如果描述中包含 'eval'，则跳过这些测试
        return

    test_name = name
    # 检查是否存在 'desc' 参数，若存在则将测试名称修改为带描述的形式
    if 'desc' in kwargs:
        test_name = f"{test_name}_{kwargs['desc']}"

    # 根据给定的参数获取测试名称
    test_name = get_nn_mod_test_name(**kwargs)

    # 如果测试名称在排除的脚本模块列表中，则直接返回，不进行后续操作
    if test_name in EXCLUDE_SCRIPT_MODULES:
        return

    # 如果参数中存在 'constructor' 键，则将 nn_module 设置为对应的构造器函数
    if 'constructor' in kwargs:
        nn_module = kwargs['constructor']
    else:
        # 否则从 torch.nn 模块中获取对应名称的模块
        nn_module = getattr(torch.nn, name)

    # 如果 nn_module 的字符串表示中包含 "FunctionalModule"，则直接返回
    if "FunctionalModule" in str(nn_module):
        return

    # 如果参数中存在 'constructor_args_fn' 键，则调用该函数获取构造函数的参数
    if 'constructor_args_fn' in kwargs:
        constructor_args = kwargs['constructor_args_fn']()
    else:
        # 否则使用参数中提供的 'constructor_args'，默认为空元组
        constructor_args = kwargs.get('constructor_args', ())

    # 根据 'input_fn' 参数生成输入数据，如果返回的是单个 Tensor，则转换为元组
    input_dtype = torch.double
    if 'input_fn' in kwargs:
        input = kwargs['input_fn']()
        if isinstance(input, torch.Tensor):
            input = (input,)

        # 如果所有输入都是复数类型，则将输入数据类型设置为 torch.cdouble
        if all(tensor.is_complex() for tensor in input):
            input_dtype = torch.cdouble
    else:
        # 否则根据 'input_size' 参数生成输入数据的大小信息
        input = (kwargs['input_size'],)

    # 如果参数中存在 'extra_args' 键，则将其添加到输入参数中
    if 'extra_args' in kwargs:
        input = input + kwargs['extra_args']

    # 如果存在 'target_size' 参数，则将其作为目标大小添加到输入参数中
    if 'target_size' in kwargs:
        input = input + (kwargs['target_size'],)
    elif 'target_fn' in kwargs:
        # 如果 'target_fn' 存在，则生成目标数据并添加到输入参数中
        if torch.is_tensor(input):
            input = (input,)
        input = input + (kwargs['target_fn'](),)

    # 调用 create_input 函数创建输入的可变参数和关键字参数
    args_variable, kwargs_variable = create_input(input, dtype=input_dtype)
    f_args_variable = deepcopy(unpack_variables(args_variable))
    out_var = deepcopy(f_args_variable)

    # 使用 create_script_module 函数创建脚本模块实例，并调用其 forward 方法进行前向传播
    args, mod = f_args_variable, create_script_module(None, nn_module, constructor_args, *f_args_variable)(*f_args_variable)

    # 返回创建的模块实例和前向传播的输出变量
    return mod, out_var
# 定义一个函数，用于获取所有的神经网络模块测试
def get_all_nn_module_tests():
    # 返回合并了 module_tests、new_module_tests 和 additional_module_tests 的列表
    return module_tests + new_module_tests + additional_module_tests
```