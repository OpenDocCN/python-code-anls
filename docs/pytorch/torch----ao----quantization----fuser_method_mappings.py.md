# `.\pytorch\torch\ao\quantization\fuser_method_mappings.py`

```
# mypy: allow-untyped-defs
# 导入PyTorch的神经网络模块和AO量化的内置模块
import torch.nn as nn
import torch.ao.nn.intrinsic as nni

# 导入类型相关的模块和函数
from typing import Any, Union, Callable, List, Tuple, Dict, Optional, Type
from torch.ao.quantization.utils import Pattern, get_combined_dict, MatchAllNode
import itertools

# 定义__all__变量，列出模块中公开的函数名
__all__ = [
    "fuse_conv_bn",
    "fuse_conv_bn_relu",
    "fuse_linear_bn",
    "fuse_convtranspose_bn",
    "get_fuser_method",
    "get_fuser_method_new",
]

# 定义函数fuse_conv_bn，用于融合卷积和批量归一化模块
def fuse_conv_bn(is_qat, conv, bn):
    r"""Return the fused the conv and bn modules.
    Given the conv and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    # 断言，确保卷积和批量归一化处于相同的训练状态（训练模式或评估模式）
    assert conv.training == bn.training, \
        "Conv and BN both must be in the same mode (train or eval)."

    # 定义融合后的模块类别映射
    fused_module_class_map = {
        nn.Conv1d: nni.ConvBn1d,
        nn.Conv2d: nni.ConvBn2d,
        nn.Conv3d: nni.ConvBn3d,
    }

    # 如果是量化感知训练（QAT）模式
    if is_qat:
        # 断言，确保批量归一化的输出通道数与卷积的输出通道数相匹配
        assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        # 断言，只支持将affine参数设置为True的批量归一化模块进行融合
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        # 断言，只支持跟踪运行统计数据的批量归一化模块进行融合
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        # 根据卷积的类型选择对应的融合后模块类别
        fused_module_class = fused_module_class_map.get((type(conv)), None)
        # 如果存在对应的融合后模块类别，则返回该类别的实例
        if fused_module_class is not None:
            return fused_module_class(conv, bn)
        else:
            # 抛出未实现错误，说明无法融合给定的训练模块
            raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn)}")
    else:
        # 返回在评估模式下融合卷积和批量归一化的结果
        return nn.utils.fuse_conv_bn_eval(conv, bn)

# 定义函数fuse_conv_bn_relu，用于融合卷积、批量归一化和ReLU激活函数模块
def fuse_conv_bn_relu(is_qat, conv, bn, relu):
    r"""Return the fused conv and bv modules.

    Given the conv and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> r1 = nn.ReLU(inplace=False)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_conv_bn_relu(m1, b1, r1)
    """
    # 断言，确保卷积、批量归一化和ReLU激活函数处于相同的训练状态（训练模式或评估模式）
    assert conv.training == bn.training == relu.training, \
        "Conv and BN both must be in the same mode (train or eval)."
    # 定义融合后模块的可选类型
    fused_module : Optional[Type[nn.Sequential]] = None
    # 如果是在量化感知训练（QAT）模式下：
    if is_qat:
        # 定义训练模式下的卷积与融合模块的映射关系
        map_to_fused_module_train = {
            nn.Conv1d: nni.ConvBnReLU1d,
            nn.Conv2d: nni.ConvBnReLU2d,
            nn.Conv3d: nni.ConvBnReLU3d,
        }
        # 断言，确保批归一化层的输出通道数与卷积层的输出通道数一致
        assert bn.num_features == conv.out_channels, 'Output channel of Conv must match num_features of BatchNorm'
        # 断言，仅支持将批归一化层与 affine 设置为 True 的卷积层融合
        assert bn.affine, 'Only support fusing BatchNorm with affine set to True'
        # 断言，仅支持将批归一化层与 tracking_running_stats 设置为 True 的卷积层融合
        assert bn.track_running_stats, 'Only support fusing BatchNorm with tracking_running_stats set to True'
        # 根据卷积层的类型选择对应的融合模块
        fused_module = map_to_fused_module_train.get(type(conv), None)
        # 如果找到了对应的融合模块，则返回融合后的模块
        if fused_module is not None:
            return fused_module(conv, bn, relu)
        else:
            # 如果没有找到对应的融合模块，则抛出未实现错误，显示无法融合的模块信息
            raise NotImplementedError(f"Cannot fuse train modules: {(conv, bn, relu)}")
    else:
        # 定义评估模式下的卷积与融合模块的映射关系
        map_to_fused_module_eval = {
            nn.Conv1d: nni.ConvReLU1d,
            nn.Conv2d: nni.ConvReLU2d,
            nn.Conv3d: nni.ConvReLU3d,
        }
        # 根据卷积层的类型选择对应的融合模块
        fused_module = map_to_fused_module_eval.get(type(conv), None)
        # 如果找到了对应的融合模块，则将卷积与批归一化层融合并返回融合后的模块
        if fused_module is not None:
            fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
            return fused_module(fused_conv, relu)
        else:
            # 如果没有找到对应的融合模块，则抛出未实现错误，显示无法融合的模块信息
            raise NotImplementedError(f"Cannot fuse eval modules: {(conv, bn, relu)}")
# 返回融合后的线性层和批归一化（BN）层模块。
# 给定线性层和BN层模块，将它们融合并返回融合后的模块。

def fuse_linear_bn(is_qat, linear, bn):
    # 断言线性层和BN层必须处于相同的模式（训练或评估）。
    assert linear.training == bn.training, \
        "Linear and BN both must be in the same mode (train or eval)."

    # 如果是量化感知训练（QAT）模式：
    if is_qat:
        # 断言BN层的输出特征数必须与线性层的输出特征数相匹配。
        assert bn.num_features == linear.out_features, \
            "Output features of Linear must match num_features of BatchNorm1d"
        # 断言BN层必须是可仿射的。
        assert bn.affine, "Only support fusing BatchNorm1d with affine set to True"
        # 断言BN层必须跟踪运行统计信息。
        assert bn.track_running_stats, \
            "Only support fusing BatchNorm1d with tracking_running_stats set to True"
        # 返回融合后的线性+BN层模块。
        return nni.LinearBn1d(linear, bn)
    else:
        # 返回使用评估模式融合线性+BN层模块。
        return nn.utils.fusion.fuse_linear_bn_eval(linear, bn)

# 返回融合后的转置卷积层和BN层模块。
# 给定转置卷积层和BN层模块，将它们融合并返回融合后的模块。

def fuse_convtranspose_bn(is_qat, convt, bn):
    # 断言转置卷积层和BN层必须处于相同的模式（训练或评估）。
    assert convt.training == bn.training, \
        "ConvTranspose and BN both must be in the same mode (train or eval)."

    # 如果是量化感知训练（QAT）模式：
    if is_qat:
        # 抛出异常，因为QAT模式下尚不支持融合转置卷积+BN层。
        raise Exception("Fusing ConvTranspose+BatchNorm not yet supported in QAT.")  # noqa: TRY002
    else:
        # 返回使用评估模式融合转置卷积+BN层模块。
        return nn.utils.fusion.fuse_conv_bn_eval(convt, bn, transpose=True)

# 返回一个对顺序模块进行封装的函数，用于is_qat和两个模块。
# 给定一个顺序类用于两个模块，返回一个函数，该函数接受is_qat以及两个模块作为参数，
# 忽略is_qat标志并始终返回结合了两个输入模块的顺序模块。

def _sequential_wrapper2(sequential):
    def fuser_method(is_qat, m1, m2):
        return sequential(m1, m2)
    return fuser_method

# 用于默认操作列表到融合方法的映射。
_DEFAULT_OP_LIST_TO_FUSER_METHOD: Dict[Tuple, Union[nn.Sequential, Callable]] = {
    # 当卷积层为Conv1d，BN层为BatchNorm1d时，使用fuse_conv_bn进行融合。
    (nn.Conv1d, nn.BatchNorm1d): fuse_conv_bn,
    # 当卷积层为Conv1d，BN层为BatchNorm1d，且后面接ReLU激活时，使用fuse_conv_bn_relu进行融合。
    (nn.Conv1d, nn.BatchNorm1d, nn.ReLU): fuse_conv_bn_relu,
    # 当卷积层为Conv2d，BN层为BatchNorm2d时，使用fuse_conv_bn进行融合。
    (nn.Conv2d, nn.BatchNorm2d): fuse_conv_bn,
    # 当卷积层为Conv2d，BN层为BatchNorm2d，且后面接ReLU激活时，使用fuse_conv_bn_relu进行融合。
    (nn.Conv2d, nn.BatchNorm2d, nn.ReLU): fuse_conv_bn_relu,
    # 当卷积层为Conv3d，BN层为BatchNorm3d时，使用fuse_conv_bn进行融合。
    (nn.Conv3d, nn.BatchNorm3d): fuse_conv_bn,
    # 当卷积层为Conv3d，BN层为BatchNorm3d，且后面接ReLU激活时，使用fuse_conv_bn_relu进行融合。
    (nn.Conv3d, nn.BatchNorm3d, nn.ReLU): fuse_conv_bn_relu,
    # 当卷积层为Conv1d，且后面接ReLU激活时，使用nni.ConvReLU1d进行融合。
    (nn.Conv1d, nn.ReLU): _sequential_wrapper2(nni.ConvReLU1d),
}
    # 定义一个字典，将特定的神经网络模块组合映射到相应的函数或类
    {
        # 当遇到 (nn.Conv2d, nn.ReLU) 组合时，使用 _sequential_wrapper2(nni.ConvReLU2d)
        (nn.Conv2d, nn.ReLU): _sequential_wrapper2(nni.ConvReLU2d),
        # 当遇到 (nn.Conv3d, nn.ReLU) 组合时，使用 _sequential_wrapper2(nni.ConvReLU3d)
        (nn.Conv3d, nn.ReLU): _sequential_wrapper2(nni.ConvReLU3d),
        # 当遇到 (nn.Linear, nn.BatchNorm1d) 组合时，使用 fuse_linear_bn 函数
        (nn.Linear, nn.BatchNorm1d): fuse_linear_bn,
        # 当遇到 (nn.Linear, nn.ReLU) 组合时，使用 _sequential_wrapper2(nni.LinearReLU)
        (nn.Linear, nn.ReLU): _sequential_wrapper2(nni.LinearReLU),
        # 当遇到 (nn.BatchNorm2d, nn.ReLU) 组合时，使用 _sequential_wrapper2(nni.BNReLU2d)
        (nn.BatchNorm2d, nn.ReLU): _sequential_wrapper2(nni.BNReLU2d),
        # 当遇到 (nn.BatchNorm3d, nn.ReLU) 组合时，使用 _sequential_wrapper2(nni.BNReLU3d)
        (nn.BatchNorm3d, nn.ReLU): _sequential_wrapper2(nni.BNReLU3d),
        # 当遇到 (nn.ConvTranspose1d, nn.BatchNorm1d) 组合时，使用 fuse_convtranspose_bn 函数
        (nn.ConvTranspose1d, nn.BatchNorm1d): fuse_convtranspose_bn,
        # 当遇到 (nn.ConvTranspose2d, nn.BatchNorm2d) 组合时，使用 fuse_convtranspose_bn 函数
        (nn.ConvTranspose2d, nn.BatchNorm2d): fuse_convtranspose_bn,
        # 当遇到 (nn.ConvTranspose3d, nn.BatchNorm3d) 组合时，使用 fuse_convtranspose_bn 函数
        (nn.ConvTranspose3d, nn.BatchNorm3d): fuse_convtranspose_bn,
    }
}

# 定义函数get_fuser_method，用于获取给定模块类型列表的融合方法
def get_fuser_method(op_list, additional_fuser_method_mapping=None):
    """Get fuser method for the given list of module types.

    Get fuser method for the given list of module types,
    return None if fuser method does not exist
    """
    # 如果未提供额外的融合方法映射，则初始化为空字典
    if additional_fuser_method_mapping is None:
        additional_fuser_method_mapping = {}
    # 获取合并后的映射字典，包括默认的和额外的映射
    all_mappings = get_combined_dict(_DEFAULT_OP_LIST_TO_FUSER_METHOD,
                                     additional_fuser_method_mapping)
    # 根据 op_list 获取对应的融合方法
    fuser_method = all_mappings.get(op_list, None)
    # 断言确保找到了对应的融合方法，否则抛出异常
    assert fuser_method is not None, f"did not find fuser method for: {op_list} "
    return fuser_method

# 定义函数_reverse2，用于生成一个新的反转函数
def _reverse2(f):
    def reversed(is_qat, x, y):
        return f(is_qat, y, x)
    return reversed

# 定义函数_reverse3，用于生成一个新的反转函数，处理包含两个元素的元组
def _reverse3(f):
    def reversed(is_qat, x, w):
        y, z = w
        return f(is_qat, z, y, x)
    return reversed

# 定义函数_get_valid_patterns，用于生成从给定操作模式生成的有效模式列表
def _get_valid_patterns(op_pattern):
    """Return a list of valid patterns generated from the op_pattern.

    Returns a list of valid patterns generated from the op_pattern,
    since MatchAllNode can match all types of nodes,
    e.g. pattern (torch.nn.Conv2d, torch.add) should also be able to match keys like
    (MatchAllNode, torch.add) and (torch.nn.Conv2d, MatchAllNode)

    Example Input:
    (torch.add, (torch.nn.ReLU, torch.nn.Conv2d))

    Example Output:
    [(torch.add, (torch.nn.ReLU, torch.nn.Conv2d)),
     (torch.add, (torch.nn.ReLU, MatchAllNode)),
     (torch.add, (MatchAllNode, torch.nn.Conv2d)),
     (torch.add, (MatchAllNode, MatchAllNode)),
     (MatchAllNode, (torch.nn.ReLU, torch.nn.Conv2d)),
     (MatchAllNode, (torch.nn.ReLU, MatchAllNode)),
     (MatchAllNode, (MatchAllNode, torch.nn.Conv2d)),
     (MatchAllNode, (MatchAllNode, MatchAllNode)),
    ]
    """
    result: List[Any]
    # 如果op_pattern是元组或列表，则递归处理每个子模式
    if isinstance(op_pattern, (tuple, list)):
        sub_combs = []
        for sub_pattern in op_pattern:
            sub_combs.append(_get_valid_patterns(sub_pattern))
        result = list(itertools.product(*sub_combs))
    else:
        # 如果op_pattern不是元组或列表，直接生成一个包含op_pattern和MatchAllNode的列表
        result = [op_pattern, MatchAllNode]
    return result

# 定义函数get_fuser_method_new，用于获取融合方法，并支持更复杂的操作模式
def get_fuser_method_new(
        op_pattern: Pattern,
        fuser_method_mapping: Dict[Pattern, Union[nn.Sequential, Callable]]):
    """Get fuser method.

    This will be made default after we deprecate the get_fuser_method
    Would like to implement this first and have a separate PR for deprecation
    """
    # 生成有效的操作模式列表
    op_patterns = _get_valid_patterns(op_pattern)
    fuser_method = None
    # 遍历所有操作模式，查找匹配的融合方法
    for op_pattern in op_patterns:
        fuser_method = fuser_method_mapping.get(op_pattern, None)
        if fuser_method is not None:
            break
    # 断言确保找到了对应的融合方法，否则抛出异常
    assert fuser_method is not None, f"did not find fuser method for: {op_pattern} "
    return fuser_method
```