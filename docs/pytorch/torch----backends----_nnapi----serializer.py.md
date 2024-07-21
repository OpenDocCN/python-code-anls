# `.\pytorch\torch\backends\_nnapi\serializer.py`

```
# mypy: allow-untyped-defs
# 导入所需模块
import array                     # 提供高效的数值序列操作
import enum                      # 提供创建枚举的支持
import functools                 # 提供高阶函数：部分应用、函数组合等
import logging                   # 提供日志记录功能
import operator                  # 提供Python内置运算符的函数形式
import struct                    # 提供基于字节的数据结构解析和打包
import sys                       # 提供与Python解释器交互的功能
from typing import List, NamedTuple, Optional, Tuple  # 提供静态类型检查支持

import torch                     # 导入PyTorch库，进行深度学习相关操作


LOG = logging.getLogger("nnapi_serialize")  # 创建一个名为 "nnapi_serialize" 的日志记录器


class NNAPI_OperandCode:
    FLOAT32 = 0                   # 浮点数（32位）
    INT32 = 1                     # 整数（32位有符号）
    UINT32 = 2                    # 整数（32位无符号）
    TENSOR_FLOAT32 = 3            # 浮点数张量（32位）
    TENSOR_INT32 = 4              # 整数张量（32位有符号）
    TENSOR_QUANT8_ASYMM = 5       # 8位异形量化张量
    BOOL = 6                      # 布尔值
    TENSOR_QUANT16_SYMM = 7       # 16位对称量化张量
    TENSOR_FLOAT16 = 8            # 浮点数张量（16位）
    TENSOR_BOOL8 = 9              # 布尔值张量（8位）
    FLOAT16 = 10                  # 浮点数（16位）
    TENSOR_QUANT8_SYMM_PER_CHANNEL = 11  # 通道间8位对称量化张量
    TENSOR_QUANT16_ASYMM = 12     # 16位异形量化张量


class NNAPI_OperationCode:
    ADD = 0                       # 加法操作
    AVERAGE_POOL_2D = 1           # 2D平均池化操作
    CONCATENATION = 2             # 连接操作
    CONV_2D = 3                   # 2D卷积操作
    DEPTHWISE_CONV_2D = 4         # 深度可分离2D卷积操作
    DEPTH_TO_SPACE = 5            # 深度转空间操作
    DEQUANTIZE = 6                # 反量化操作
    EMBEDDING_LOOKUP = 7          # 嵌入查找操作
    FLOOR = 8                     # 向下取整操作
    FULLY_CONNECTED = 9           # 全连接操作
    HASHTABLE_LOOKUP = 10         # 哈希表查找操作
    L2_NORMALIZATION = 11         # L2范数归一化操作
    L2_POOL_2D = 12               # 2D L2池化操作
    LOCAL_RESPONSE_NORMALIZATION = 13  # 本地响应归一化操作
    LOGISTIC = 14                 # 逻辑操作
    LSH_PROJECTION = 15           # 局部敏感哈希投影操作
    LSTM = 16                     # 长短期记忆网络操作
    MAX_POOL_2D = 17              # 2D最大池化操作
    MUL = 18                      # 乘法操作
    RELU = 19                     # ReLU激活函数操作
    RELU1 = 20                    # ReLU1激活函数操作
    RELU6 = 21                    # ReLU6激活函数操作
    RESHAPE = 22                  # 重塑操作
    RESIZE_BILINEAR = 23          # 双线性重缩放操作
    RNN = 24                      # 递归神经网络操作
    SOFTMAX = 25                  # Softmax操作
    SPACE_TO_DEPTH = 26           # 空间转深度操作
    SVDF = 27                     # 奇异值分解操作
    TANH = 28                     # 双曲正切操作
    BATCH_TO_SPACE_ND = 29        # N维批量转空间操作
    DIV = 30                      # 除法操作
    MEAN = 31                     # 平均值操作
    PAD = 32                      # 填充操作
    SPACE_TO_BATCH_ND = 33        # N维空间转批量操作
    SQUEEZE = 34                  # 压缩操作
    STRIDED_SLICE = 35            # 跨步切片操作
    SUB = 36                      # 减法操作
    TRANSPOSE = 37                # 转置操作
    ABS = 38                      # 绝对值操作
    ARGMAX = 39                   # 最大值索引操作
    ARGMIN = 40                   # 最小值索引操作
    AXIS_ALIGNED_BBOX_TRANSFORM = 41  # 轴对齐边界框变换操作
    BIDIRECTIONAL_SEQUENCE_LSTM = 42  # 双向序列LSTM操作
    BIDIRECTIONAL_SEQUENCE_RNN = 43   # 双向序列RNN操作
    BOX_WITH_NMS_LIMIT = 44       # 带有NMS限制的框操作
    CAST = 45                     # 类型转换操作
    CHANNEL_SHUFFLE = 46          # 通道洗牌操作
    DETECTION_POSTPROCESSING = 47  # 检测后处理操作
    EQUAL = 48                    # 等于操作
    EXP = 49                      # 指数操作
    EXPAND_DIMS = 50              # 扩展维度操作
    GATHER = 51                   # 收集操作
    GENERATE_PROPOSALS = 52       # 生成建议操作
    GREATER = 53                  # 大于操作
    GREATER_EQUAL = 54            # 大于等于操作
    GROUPED_CONV_2D = 55          # 分组2D卷积操作
    HEATMAP_MAX_KEYPOINT = 56     # 热图最大关键点操作
    INSTANCE_NORMALIZATION = 57   # 实例归一化操作
    LESS = 58                     # 小于操作
    LESS_EQUAL = 59               # 小于等于操作
    LOG = 60                      # 对数操作
    LOGICAL_AND = 61              # 逻辑与操作
    LOGICAL_NOT = 62              # 逻辑非操作
    LOGICAL_OR = 63               # 逻辑或操作
    LOG_SOFTMAX = 64              # 对数Softmax操作
    MAXIMUM = 65                  # 最大值操作
    MINIMUM = 66                  # 最小值操作
    NEG = 67                      # 取反操作
    NOT_EQUAL = 68                # 不等于操作
    PAD_V2 = 69                   # V2填充操作
    POW = 70                      # 幂操作
    PRELU = 71                    # PReLU激活函数操作
    QUANTIZE = 72                 # 量化操作
    QUANTIZED_16BIT_LSTM = 73     # 16位量化LSTM操作
    RANDOM_MULTINOMIAL = 74       # 随机多项式操作
    REDUCE_ALL = 75               # 全部减少操作
    REDUCE_ANY = 76               # 任意减少操作
    REDUCE_MAX = 77               # 最大减少操作
    REDUCE_MIN = 78               # 最小减少操作
    REDUCE_PROD = 79              # 乘积减少操作
    REDUCE_SUM = 80               # 总和减少操作
    ROI_ALIGN = 81                # ROI对齐操作
    ROI_POOLING = 82              # ROI池化操作
    RSQRT = 83                    # 平方根倒数操作
    SELECT = 84                   # 选择操作
    SIN = 85                      # 正弦操作
    SLICE = 86                    # 切片操作
    SPLIT = 87                    # 分割操作
    SQRT = 88                     # 平方根操作
    TILE = 89                     # 平铺操作
    TOPK_V2 = 90                  # V2前K大操作
    TRANSPOSE_CONV_2D = 91        # 转置2D卷
class TorchScalarTypes(enum.Enum):
    QUINT8 = 13

# 定义一个枚举类 TorchScalarTypes，包含一个成员 QUINT8，其值为 13



def approx_equal(lhs, rhs, tolerance=1e-6):
    return abs(lhs - rhs) <= tolerance * min(lhs, rhs)

# 判断 lhs 和 rhs 是否在指定容差 tolerance 范围内近似相等，返回布尔值



def tensor_size(op_type, dims):
    ITEM_SIZES = {
        NNAPI_OperandCode.TENSOR_FLOAT32: 4,
        NNAPI_OperandCode.TENSOR_INT32: 4,
        NNAPI_OperandCode.TENSOR_QUANT8_ASYMM: 1,
        NNAPI_OperandCode.TENSOR_QUANT16_SYMM: 2,
        NNAPI_OperandCode.TENSOR_QUANT16_ASYMM: 2,
    }
    size = ITEM_SIZES[op_type]
    for d in dims:
        size *= d
    return size

# 根据操作类型 op_type 和维度 dims 计算张量的大小，通过 ITEM_SIZES 字典查找 op_type 对应的大小，并乘以每个维度的大小



def change_element(tup, index, value):
    ls = list(tup)
    ls[index] = value
    return tuple(ls)

# 将元组 tup 转换为列表 ls，修改索引 index 处的值为 value，然后将列表 ls 转换回元组并返回



class ConvPoolArgs2d(NamedTuple):
    """Configuration arguments for a convolution."""

    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_t: int
    pad_b: int
    pad_l: int
    pad_r: int
    dilation_h: int
    dilation_w: int
    group: int

# 定义一个命名元组 ConvPoolArgs2d，表示二维卷积的配置参数，包括内核大小、步长、填充以及扩张率等



class DimOrder(enum.Enum):
    PRESUMED_CONTIGUOUS = 0
    CHANNELS_LAST = 1
    SCALAR_OR_VECTOR = 2
    UNKNOWN_CONSTANT = 999

# 定义一个枚举类 DimOrder，包含几种不同的维度顺序类型，分别是 PRESUMED_CONTIGUOUS、CHANNELS_LAST、SCALAR_OR_VECTOR 和 UNKNOWN_CONSTANT



class Operand(NamedTuple):
    """Represenation of an NNAPI operand."""

    # NNAPI operand type.  One of NNAPI_OperandCode.
    # TODO: Make this an enum.
    op_type: int

    # This is always the PyTorch shape, which is NCHW for feature maps.
    # The actual NNAPI operand might have a transposed shape.
    # we use 0 for load time dynamic shapes & -1 for runtime dynamic shapes
    shape: Tuple[int, ...]

    # Specifies how the shape of the operand that we define in NNAPI
    # relates to the shape we track above.
    # - PRESUMED_CONTIGUOUS: physical NNAPI operand will exactly match
    #   the shape of the PyTorch tensor.
    # - CHANNELS_LAST: The PyTorch tensor is expected to be NCHW, and
    #   the NNAPI operand will be represented explicitly as NHWC.
    dim_order: DimOrder

    # Quantization params
    scale: float
    zero_point: int

    def use_nchw(self):
        if self.dim_order is DimOrder.PRESUMED_CONTIGUOUS:
            return True
        if self.dim_order is DimOrder.CHANNELS_LAST:
            return False
        raise Exception("Unknown dim order")  # noqa: TRY002

# 定义一个命名元组 Operand，表示 NNAPI 操作数的表示形式，包括操作类型 op_type、形状 shape、维度顺序 dim_order、量化参数 scale 和 zero_point，以及一个方法 use_nchw 判断是否使用 NCHW 维度顺序



def broadcast_shapes(shape1, shape2):
    assert len(shape1) > 0
    assert len(shape2) > 0
    s1 = list(shape1)
    s2 = list(shape2)
    # TODO: Support non-equal-rank broadcast where semantics match.
    # This can be tricky for NHWC tensors because dimension orders
    # don't match between PT and NNAPI, even though semantics match.
    if len(s1) > len(s2):
        # s2 = [1] * (len(s1) - len(s2)) + s2
        raise Exception(  # noqa: TRY002
            "Non-equal-rank broadcast is not supported yet."
        )  # noqa: TRY002
    if len(s2) > len(s1):
        # s3 = [1] * (len(s2) - len(s1)) + s1
        raise Exception(  # noqa: TRY002
            "Non-equal-rank broadcast is not supported yet."
        )  # noqa: TRY002
    ret = []

# 定义函数 broadcast_shapes，用于广播两个形状 shape1 和 shape2，确保它们在语义上匹配的情况下支持非等级广播。
    # 遍历两个可迭代对象 s1 和 s2 中的元素，同时比较它们
    for d1, d2 in zip(s1, s2):
        # 如果 s1 中的元素等于 1，则将 s2 中的对应元素加入结果列表 ret
        if d1 == 1:
            ret.append(d2)
        # 如果 s2 中的元素等于 1，则将 s1 中的对应元素加入结果列表 ret
        elif d2 == 1:
            ret.append(d1)
        # 如果 s1 和 s2 中的对应元素相等，则将其中任意一个加入结果列表 ret
        elif d1 == d2:
            ret.append(d1)
        else:
            # 如果以上条件都不满足，则抛出异常，指示形状无法广播的情况
            raise Exception(
                f"Cannot broadcast shapes: {shape1} and {shape2}"
            )
    # 将结果列表 ret 转换为元组并返回
    return tuple(ret)
# 初始化_NnapiSerializer类的构造函数，接受一个配置config和一个可选的布尔参数use_int16_for_qint16。
class _NnapiSerializer:
    def __init__(self, config, use_int16_for_qint16=False):
        # 初始化各种成员变量，用于存储操作数、数值、操作、值数据、操作参数、输入输出等信息。
        self.operands = []
        self.values = []
        self.operations = []
        self.value_data = []
        self.operation_args = []
        self.inputs = []
        self.outputs = []
        self.flexible_shape_computation_lines = []

        # 初始化模块、常量、张量序列、JIT值操作映射、缓存的立即数等信息。
        self.modules = {}
        self.constants = {}
        self.tensor_sequences = {}
        self.jitval_operand_map = {}
        self.cached_immediates = {}
        self.used_weights = []
        self.weight_offset = 0
        self.use_int16_for_qint16 = use_int16_for_qint16

        # 如果配置config为None，将其设为空字典。
        if config is None:
            config = {}

    # 获取下一个操作数的ID，即当前操作数列表的长度。
    def get_next_operand_id(self):
        return len(self.operands)

    # 添加一个对应于JIT值的张量操作数。
    # 返回NNAPI操作数ID，稍后可以使用它进行查找。
    # 添加一个张量操作数到操作数列表中，使用 JIT 值作为键
    def add_tensor_operand(self, jitval, oper):
        # 断言操作数是 Operand 类的实例
        assert isinstance(oper, Operand)
        # 如果 JIT 值已经存在于映射中，则抛出异常
        if jitval in self.jitval_operand_map:
            raise Exception(f"Duplicate tensor: {jitval!r}")  # noqa: TRY002

        # 获取下一个操作数的唯一标识符
        operand_id = self.get_next_operand_id()
        # 将操作数添加到操作数列表中
        self.operands.append(oper)
        # 将 JIT 值映射到操作数的唯一标识符
        self.jitval_operand_map[jitval] = operand_id
        # 返回操作数的唯一标识符
        return operand_id

    # 添加一个不对应 JIT 值的张量操作数
    # 用于需要多个 NNAPI 操作数来实现一个 JIT IR 节点的情况
    # 返回 NNAPI 操作数的唯一标识符
    def add_anonymous_tensor_operand(self, oper):
        # 断言操作数是 Operand 类的实例
        assert isinstance(oper, Operand)
        # 获取下一个操作数的唯一标识符
        operand_id = self.get_next_operand_id()
        # 将操作数添加到操作数列表中
        self.operands.append(oper)
        # 返回操作数的唯一标识符
        return operand_id

    # 将 Torch 张量转换为 Operand 对象
    def torch_tensor_to_operand(self, tensor, dim_order):
        # 获取张量的数据类型，去掉前缀 'torch.'
        dtype = str(tensor.dtype).replace("torch.", "")
        # 默认的缩放因子和零点
        scale = 0.0
        zero_point = 0

        # 根据数据类型选择 NNAPI 的操作码
        if dtype == "float32":
            op_type = NNAPI_OperandCode.TENSOR_FLOAT32
        elif dtype == "int32":
            op_type = NNAPI_OperandCode.TENSOR_INT32
        elif dtype == "quint8":
            op_type = NNAPI_OperandCode.TENSOR_QUANT8_ASYMM
            scale = tensor.q_scale()
            zero_point = tensor.q_zero_point()
        elif dtype == "qint32":
            op_type = NNAPI_OperandCode.TENSOR_INT32
            scale = tensor.q_scale()
            zero_point = tensor.q_zero_point()
            # 对于 qint32 类型，零点必须为 0
            assert zero_point == 0
        elif dtype == "int16":
            # 如果允许使用 int16 代替 qint16
            if self.use_int16_for_qint16:
                nnapi_dtype = getattr(tensor, "nnapi_dtype", None)
                op_codes = (
                    NNAPI_OperandCode.TENSOR_QUANT16_SYMM,
                    NNAPI_OperandCode.TENSOR_QUANT16_ASYMM,
                )
                # 检查 nnapi_dtype 是否为有效的操作码之一
                if nnapi_dtype in op_codes:
                    op_type = nnapi_dtype
                    scale = tensor.nnapi_scale
                    zero_point = tensor.nnapi_zero_point
                else:
                    raise Exception(  # noqa: TRY002
                        f"`nnapi_type` needs to be one of {op_codes} for `int16`"
                    )
            else:
                raise Exception(  # noqa: TRY002
                    "`int16` isn't supported. If you're trying to represent NNAPI"
                    " qint16 with Pytorch int16, set `use_int16_for_qint16 = True`"
                )
        else:
            # 如果数据类型不支持，抛出异常
            raise Exception(  # noqa: TRY002
                f"Can't handle input with dtype '{tensor.dtype}'"
            )  # noqa: TRY002
        
        # 返回一个 Operand 对象，表示转换后的张量
        return Operand(
            shape=tuple(tensor.shape),
            op_type=op_type,
            dim_order=dim_order,
            scale=scale,
            zero_point=zero_point,
        )
    # 为输入参数添加张量操作数
    def add_tensor_operand_for_input(self, arg_idx, jitval, tensor):
        # 确定张量的维度顺序，根据 nnapi_nhwc 属性判断是否为 CHANNELS_LAST，否则使用 PRESUMED_CONTIGUOUS
        dim_order = (
            DimOrder.CHANNELS_LAST
            if getattr(tensor, "nnapi_nhwc", False)
            else DimOrder.PRESUMED_CONTIGUOUS
        )
        # 将 torch 张量转换为操作数，并根据指定的维度顺序
        toper = self.torch_tensor_to_operand(tensor, dim_order)
        # 添加张量操作数，并获取其操作数 ID
        operand_id = self.add_tensor_operand(jitval, toper)
        # 将操作数 ID 添加到输入列表中
        self.inputs.append(operand_id)
        # 检查张量的每个维度，如果维度大小为 0，则计算操作数的形状
        for dim, size in enumerate(tensor.shape):
            if size == 0:
                self.compute_operand_shape(
                    operand_id, dim, f"args[{arg_idx}].shape[{dim}]"
                )
        # 返回操作数 ID
        return operand_id

    # 为权重张量添加操作数
    def add_tensor_operand_for_weight(
        self, tensor, dim_order=DimOrder.UNKNOWN_CONSTANT
    ):
        # 将 torch 张量转换为操作数，并根据指定的维度顺序
        toper = self.torch_tensor_to_operand(tensor, dim_order)
        # 分配新的操作数 ID 给该操作数
        operand_id = len(self.operands)
        # 将操作数添加到操作数列表中
        self.operands.append(toper)
        # 计算张量的尺寸
        tsize = tensor_size(toper.op_type, toper.shape)
        # 对 tsize 进行对齐处理，并添加到 values 列表中
        psize = ((tsize - 1) | 0x3) + 1
        self.values.append((operand_id, OperandValueSourceType.NUMBERED_BUFFER))
        # 记录该权重的缓冲区编号，并偏移量为 0
        buf_num = len(self.used_weights)
        offset = 0
        self.value_data.append(struct.pack("iii", buf_num, offset, tsize))
        # 如果 dim_order 为 CHANNELS_LAST，则按照 NHWC NNAPI 操作的要求重新排列张量数据
        if dim_order == DimOrder.CHANNELS_LAST:
            tensor = tensor.permute(0, 2, 3, 1)
        # 将处理后的张量添加到 used_weights 列表中
        self.used_weights.append(tensor)
        # 返回操作数 ID
        return operand_id

    # 添加立即数操作数
    def add_immediate_operand(self, code, value, dims):
        # 确保 dims 是元组类型
        assert isinstance(dims, tuple)
        # 创建缓存键值对，用于缓存已添加的立即数操作数
        cache_key = (code, value)
        # 如果缓存中不存在该立即数操作数，则进行添加
        if cache_key not in self.cached_immediates:
            # 分配新的操作数 ID 给该操作数
            operand_id = len(self.operands)
            # 创建并添加立即数操作数到操作数列表中
            self.operands.append(Operand(code, dims, DimOrder.SCALAR_OR_VECTOR, 0.0, 0))
            # 将操作数 ID 和操作数值类型添加到 values 列表中
            self.values.append((operand_id, OperandValueSourceType.IMMEDIATE))
            # 添加操作数值到 value_data 列表中
            self.value_data.append(value)
            # 将新添加的操作数 ID 缓存起来
            self.cached_immediates[cache_key] = operand_id
        # 返回该立即数操作数的操作数 ID
        return self.cached_immediates[cache_key]

    # 添加整数类型的立即数操作数
    def add_immediate_int_scalar(self, value):
        # 调用通用的立即数操作数添加方法，指定整数类型码和打包的整数值
        return self.add_immediate_operand(
            NNAPI_OperandCode.INT32, struct.pack("i", value), ()
        )

    # 添加浮点数类型的立即数操作数
    def add_immediate_float_scalar(self, value):
        # 调用通用的立即数操作数添加方法，指定浮点数类型码和打包的浮点数值
        return self.add_immediate_operand(
            NNAPI_OperandCode.FLOAT32, struct.pack("f", value), ()
        )

    # 添加布尔类型的立即数操作数
    def add_immediate_bool_scalar(self, value):
        # 调用通用的立即数操作数添加方法，指定布尔类型码和对应的字节值
        return self.add_immediate_operand(
            NNAPI_OperandCode.BOOL, b"\x01" if value else b"\x00", ()
        )

    # 添加整数向量类型的立即数操作数
    def add_immediate_int_vector(self, value):
        # 调用通用的立即数操作数添加方法，指定整数向量类型码和打包的整数数组值
        return self.add_immediate_operand(
            NNAPI_OperandCode.TENSOR_INT32,
            array.array("i", value).tobytes(),
            (len(value),),
        )

    # 检查是否已为给定的 jitval 添加过操作数
    def has_operand_for_jitval(self, jitval):
        # 返回该 jitval 是否存在于 jitval_operand_map 中的布尔值
        return jitval in self.jitval_operand_map

    # 根据 jitval 获取张量操作数及其操作数 ID
    def get_tensor_operand_by_jitval(self, jitval):
        # 根据 jitval 查找其对应的操作数 ID
        operand_id = self.jitval_operand_map[jitval]
        # 返回操作数 ID 及其在 operands 列表中的操作数对象
        return (operand_id, self.operands[operand_id])
    # 根据给定的 JIT 值获取固定大小的张量操作数和操作数 ID
    def get_tensor_operand_by_jitval_fixed_size(self, jitval):
        # 调用 get_tensor_operand_by_jitval 方法获取操作数 ID 和操作数对象
        op_id, oper = self.get_tensor_operand_by_jitval(jitval)
        # 遍历操作数对象的形状
        for s in oper.shape:
            if s == 0:
                # 如果形状为 0，则抛出异常，指出不支持灵活大小的操作数
                raise Exception(
                    "Flexible size is not supported for this operand."
                )
            if s < 0:
                # 如果形状为负数，记录警告信息，表明操作数具有运行时灵活的形状
                LOG.warning("Operand %s has runtime flex shape", oper)
        # 返回操作数 ID 和操作数对象
        return op_id, oper

    # 根据 JIT 值获取张量操作数或常量
    def get_tensor_operand_or_constant(
        self, jitval, dim_order=DimOrder.PRESUMED_CONTIGUOUS
    ):
        # 查询 JIT 值在映射中对应的操作数 ID
        operand_id = self.jitval_operand_map.get(jitval)
        if operand_id is None:
            # 如果未找到对应的操作数 ID，则获取 JIT 值的常量值并添加为张量操作数
            _, value = self.get_constant_value(jitval, "TensorType")
            operand_id = self.add_tensor_operand_for_weight(value, dim_order)
        # 返回操作数 ID 和操作数对象
        return (operand_id, self.operands[operand_id])

    # 根据 JIT 值获取张量操作数，用于权重
    def get_tensor_operand_for_weight(self, jitval):
        # 获取 JIT 值对应的常量值，类型为张量类型
        _, value = self.get_constant_value(jitval, "TensorType")
        # 添加该常量值为权重的张量操作数，并获取操作数 ID
        operand_id = self.add_tensor_operand_for_weight(value)
        # 返回操作数 ID 和操作数对象
        return (operand_id, self.operands[operand_id])

    # 向操作列表中添加新的操作
    def add_operation(self, opcode, inputs, outputs):
        # 添加操作码、输入数量和输出数量的元组到操作列表中
        self.operations.append((opcode, len(inputs), len(outputs)))
        # 扩展操作参数列表，包括输入和输出
        self.operation_args.extend(inputs + outputs)

    # 向 tensor_sequences 字典中添加新的 JIT 值和对应的值序列
    def add_tensor_sequence(self, jitval, values):
        # 断言确保 JIT 值不在 tensor_sequences 中，然后添加 JIT 值及其对应的值序列
        assert jitval not in self.tensor_sequences
        self.tensor_sequences[jitval] = values

    # 向 constants 字典中添加新的 JIT 值、常量类型和值的记录
    def add_constant_value(self, jitval, ctype, value):
        # 断言确保 JIT 值不在 constants 中，然后添加 JIT 值及其常量类型和值的记录
        assert jitval not in self.constants
        self.constants[jitval] = (ctype, value)

    # 根据 JIT 值获取常量值
    def get_constant_value(self, jitval, typekind=None):
        # 获取 JIT 值在 constants 中对应的记录
        record = self.constants.get(jitval)
        if record is None:
            # 如果未找到记录，抛出异常，表明找不到该 JIT 值的常量值
            raise Exception(
                f"Could not find constant value for '{jitval!r}'."
            )
        ctype, _ = record
        if typekind is not None and ctype.kind() != typekind:
            # 如果指定了类型类别，并且记录的常量类型不符合期望，抛出异常
            raise Exception(
                f"Expected constant value of type {typekind}, but got {ctype.kind()} for value '{jitval!r}'"
            )
        # 返回常量类型和值的记录
        return record
    # 将操作数转换为 TorchScript 表达式以构建给定操作数的模板
    def operand_to_template_torchscript(self, op_id, oper, shape=None):
        """Return a TorchScript expression to build a template for a given operand."""
        # 如果未指定形状，则使用操作数本身的形状
        if shape is None:
            shape = oper.shape
        else:
            # 确保指定的形状与操作数的形状长度相同
            assert len(shape) == len(oper.shape)

        # 构建形状的字符串表示
        shape_parts = ["("]
        for d, s in enumerate(shape):
            if s > 0:
                # 固定形状维度：直接添加其值
                shape_parts.append(str(s))
            elif s == 0:
                # 在加载时可以变化的形状维度：应该在变量中计算得出
                shape_parts.append(flex_name(op_id, d))
            elif s == -1:
                # 运行时可变形状
                shape_parts.append("0")
            else:
                # 抛出异常，因为维度应该大于等于 -1
                raise Exception(
                    "Unknown dim value, dimensions should be >= -1"
                )

            shape_parts.append(",")
        shape_parts.append(")")
        shape_code = "".join(shape_parts)

        # 根据操作数的类型返回相应的 TorchScript 表达式
        if oper.op_type == NNAPI_OperandCode.TENSOR_FLOAT32:
            return f"torch.zeros({shape_code}, dtype=torch.float32)"
        elif oper.op_type == NNAPI_OperandCode.TENSOR_INT32:
            return f"torch.zeros({shape_code}, dtype=torch.int32)"
        elif oper.op_type == NNAPI_OperandCode.TENSOR_QUANT8_ASYMM:
            return (
                f"torch.quantize_per_tensor("
                f"torch.zeros(1), scale={oper.scale}, zero_point={oper.zero_point}, dtype=torch.quint8)"
                f".expand({shape_code}).contiguous()"
            )
        elif oper.op_type in (
            NNAPI_OperandCode.TENSOR_QUANT16_ASYMM,
            NNAPI_OperandCode.TENSOR_QUANT16_SYMM,
        ):
            if self.use_int16_for_qint16:
                return f"torch.zeros({shape_code}, dtype=torch.int16)"
            else:
                # 如果尝试使用 int16 表示 NNAPI qint16，则抛出异常
                raise Exception(
                    "`int16` isn't supported. If you're trying to represent NNAPI"
                    " qint16 with Pytorch int16, set `use_int16_for_qint16 = True`"
                )

        # 如果操作数类型不支持，则抛出异常
        raise Exception(
            f"Unsupported output operand type: {oper.op_type}"
        )

    # 前向操作数形状推断
    def forward_operand_shape(self, out_op_id, out_dim, in_op_id, in_dim):
        self.compute_operand_shape(out_op_id, out_dim, flex_name(in_op_id, in_dim))

    # 计算操作数形状
    def compute_operand_shape(self, op_id, dim, expr):
        # 将灵活形状计算表达式添加到列表中
        self.flexible_shape_computation_lines.append(
            f"{flex_name(op_id, dim)} = {expr}"
        )
    # 将输入张量转置为NHWC格式，以适应广播操作。
    def transpose_to_nhwc(self, in_id, oper):
        # 检查操作的高度和宽度是否为1，否则抛出异常
        if oper.shape[2:] != (1, 1):
            raise Exception(
                "Automatic transpose only supported for H,W == 1,1"
            )

        # 将操作数的维度顺序替换为CHANNELS_LAST
        out_oper = oper._replace(dim_order=DimOrder.CHANNELS_LAST)

        # 创建输入列表，其中包括in_id和一个转置后的整数向量
        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector([0, 2, 3, 1])

        # 创建输出列表，其中包括一个匿名的张量操作数
        outputs = [None] * 1
        outputs[0] = self.add_anonymous_tensor_operand(out_oper)

        # 添加一个转置操作，使用NNAPI_OperationCode.TRANSPOSE操作码
        self.add_operation(NNAPI_OperationCode.TRANSPOSE, inputs, outputs)

        # 返回输出张量和转置后的操作数
        return outputs[0], out_oper

    # 根据需要转置输入，以允许广播。
    def transpose_for_broadcast(self, in0_id, in0_oper, in1_id, in1_oper):
        # 如果输入的操作数维度顺序相同，则直接返回输入
        if in0_oper.dim_order == in1_oper.dim_order:
            return in0_id, in0_oper, in1_id, in1_oper

        # 如果维度顺序不同，假设NHWC是首选的
        orders = (in0_oper.dim_order, in1_oper.dim_order)
        if orders == (DimOrder.PRESUMED_CONTIGUOUS, DimOrder.CHANNELS_LAST):
            # 调用transpose_to_nhwc转置in0_id
            return self.transpose_to_nhwc(in0_id, in0_oper) + (in1_id, in1_oper)
        if orders == (DimOrder.CHANNELS_LAST, DimOrder.PRESUMED_CONTIGUOUS):
            # 调用transpose_to_nhwc转置in1_id
            return (in0_id, in0_oper) + self.transpose_to_nhwc(in1_id, in1_oper)

        # 如果维度顺序不支持自动转置，抛出异常
        raise Exception(
            f"Automatic transpose not supported for dim_orders: {in0_oper.dim_order!r}, {in1_oper.dim_order!r}"
        )

    # 从JIT获取大小参数
    def get_size_arg(self, jitval):
        # 获取常量类型和值
        ctype, value = self.get_constant_value(jitval)
        # 如果类型为ListType，且元素类型为IntType，则返回其值
        if ctype.kind() == "ListType":
            assert ctype.getElementType().kind() == "IntType"
            return value
        # 否则，抛出异常，指示无法处理该类型的大小参数
        raise Exception(
            f"Can't handle size arg of type '{ctype!r}' for '{jitval!r}'"
        )

    # 从打包配置中获取二维卷积池化参数
    def get_conv_pool_args_2d_from_pack(self, kernel_size, packed_config):
        # 将打包配置转换为Python列表
        pc = [i.item() for i in packed_config]
        # 断言打包配置的第一个元素为2
        assert pc[0] == 2
        # 从打包配置中提取步幅、填充、膨胀和输出填充
        strides = [pc[1], pc[2]]
        paddings = [pc[3], pc[4]]
        dilations = [pc[5], pc[6]]
        output_padding = [pc[7], pc[8]]
        group_num = pc[9]

        # 断言打包配置的长度为11，并且输出填充为[0, 0]
        assert len(pc) == 11
        assert output_padding == [0, 0]

        # 调用get_conv_pool_args_2d_common，获取二维卷积池化的通用参数
        return self.get_conv_pool_args_2d_common(
            kernel_size, strides, paddings, dilations, group_num
        )

    # 从JIT获取二维卷积池化参数
    def get_conv_pool_args_2d_from_jit(
        self, kernel_size, stride, padding, dilation=None, group=None
    ):
        # 获取步幅、填充、膨胀参数的大小
        strides = self.get_size_arg(stride)
        paddings = self.get_size_arg(padding)
        # 如果膨胀参数为None，则默认为[1, 1]
        if dilation is None:
            dilations = [1, 1]
        else:
            dilations = self.get_size_arg(dilation)
        # 如果组数不为None，则获取其值；否则设为None
        if group is not None:
            _, group_num = self.get_constant_value(group, "IntType")
        else:
            group_num = None
        # 调用get_conv_pool_args_2d_common，获取二维卷积池化的通用参数
        return self.get_conv_pool_args_2d_common(
            kernel_size, strides, paddings, dilations, group_num
        )
    # 定义一个方法，用于获取二维卷积池化操作的共同参数
    def get_conv_pool_args_2d_common(
        self, kernel_size, strides, paddings, dilations, group_num
    ):
        # 将 kernel_size 转换为列表
        kernels = list(kernel_size)

        # 断言各参数的长度为2，即二维操作
        assert len(kernels) == 2
        assert len(strides) == 2
        assert len(paddings) == 2
        assert len(dilations) == 2

        # NNAPI 使用四个值表示填充
        ph, pw = paddings
        real_paddings = [ph, ph, pw, pw]

        # 返回一个 ConvPoolArgs2d 对象，包含所有参数
        return ConvPoolArgs2d(
            *(kernels + strides + real_paddings + dilations + [group_num])
        )

    # 序列化实例的值和值数据
    def serialize_values(self):
        serialized_values = []
        serialized_value_data = []
        # 断言值和值数据的长度相同
        assert len(self.values) == len(self.value_data)
        for (op_index, source_type), data in zip(self.values, self.value_data):
            source_length = len(data)

            # 计算物理长度，用于内存对齐，填充到4的倍数
            physical_length = ((source_length - 1) | 0x3) + 1
            padded_data = data + (b"\0" * (physical_length - source_length))

            # 将 op_index、source_type 和 source_length 打包为字节流，添加到序列化值中
            serialized_values.append(
                struct.pack("iii", op_index, source_type, source_length)
            )
            serialized_value_data.append(padded_data)

        # 返回序列化后的值和值数据
        return serialized_values, serialized_value_data

    # 静态方法，将整数数组序列化为字节流
    @staticmethod
    def serialize_ints(ints):
        return array.array("i", ints).tobytes()

    # 添加节点到图中的方法
    def add_node(self, node):
        # 获取对应节点类型的添加器函数
        adder = self.ADDER_MAP.get(node.kind())
        if not adder:
            # 如果未找到对应的添加器，抛出异常
            raise Exception(
                f"Unsupported node kind ({node.kind()!r}) in node {node!r}"
            )
        adder(self, node)

    # 内部方法，用于处理标识节点
    def _identity(self, node):
        # 获取节点的输入标识符和操作数
        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        jitval = node.outputsAt(0)
        # 将节点的 JIT 值映射到输入标识符
        self.jitval_operand_map[jitval] = in_id

    # 添加获取属性的节点方法
    def add_getattr(self, node):
        # 断言输入输出节点的大小为1
        assert node.inputsSize() == 1
        assert node.outputsSize() == 1
        # 获取节点输入的常量类型和值
        obj_ctype, obj = self.get_constant_value(node.inputsAt(0))
        # 断言对象类型以 "__torch__." 开头
        assert str(obj_ctype).startswith("__torch__.")
        name = node.s("name")
        # 获取对象的指定属性值
        value = getattr(obj, name)
        output = node.outputsAt(0)
        ctype = output.type()
        # 将属性值添加为常量值
        self.add_constant_value(output, ctype, value)

    # 添加常量节点到图中的方法
    def add_constant_node(self, node):
        # 断言节点没有输入，只有一个输出
        assert node.inputsSize() == 0
        assert node.outputsSize() == 1
        output = node.outputsAt(0)
        ctype = output.type()
        # 获取输出节点的整数值
        value = output.toIValue()
        # 将整数值作为常量值添加到图中
        self.add_constant_value(output, ctype, value)
    def add_list_construct(self, node):
        # 确保节点只有一个输出
        assert node.outputsSize() == 1
        # 获取节点的输出
        output = node.outputsAt(0)
        # 获取输出的类型
        ctype = output.type()
        # 可选的常量值列表和张量列表初始化为空
        const_vals: Optional[List] = []
        tensors: Optional[List] = []
        # 遍历节点的输入
        for inp in node.inputs():
            # 如果const_vals不为空且输入在self.constants中，则获取常量值并添加到const_vals中
            if const_vals is not None and inp in self.constants:
                _, val = self.get_constant_value(inp)
                const_vals.append(val)
            else:
                const_vals = None
            # 如果tensors不为空且输入的类型是张量类型，则将输入添加到tensors列表中
            if tensors is not None and inp.type().kind() == "TensorType":
                tensors.append(inp)
            else:
                tensors = None

        # 如果const_vals不为空，则调用self.add_constant_value添加常量值
        if const_vals is not None:
            # 注意: 现在 TorchScript 支持列表常量，可能不再使用这段代码路径。
            self.add_constant_value(output, ctype, const_vals)
        
        # 如果tensors不为空，则调用self.add_tensor_sequence添加张量序列
        if tensors is not None:
            self.add_tensor_sequence(output, tensors)
        
        # 如果既没有const_vals也没有tensors，则抛出异常
        if const_vals is None and tensors is None:
            raise Exception(
                f"Unable to handle ListConstruct node.  Neither all constants nor all tensors. {node!r}"
            )

    def add_tuple_construct(self, node):
        # 确保节点只有一个输出
        assert node.outputsSize() == 1
        # 获取节点的输出
        output = node.outputsAt(0)
        # 获取所有输入节点作为值列表
        values = list(node.inputs())
        # 调用self.add_tensor_sequence添加张量序列
        self.add_tensor_sequence(output, values)

    def add_unsqueeze(self, node):
        # 确保节点有两个输入和一个输出
        assert node.inputsSize() == 2
        assert node.outputsSize() == 1

        # 获取第一个输入节点的标识符和操作
        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))

        # 获取第二个输入节点的常量值作为整数类型
        _, dim = self.get_constant_value(node.inputsAt(1), "IntType")
        # 确保in_oper.dim_order为DimOrder.PRESUMED_CONTIGUOUS
        assert in_oper.dim_order == DimOrder.PRESUMED_CONTIGUOUS

        # 计算实际的维度
        real_dim = dim if dim >= 0 else dim + len(in_oper.shape) + 1
        # 创建修改后的输出形状列表
        out_shape_list = list(in_oper.shape)
        out_shape_list.insert(real_dim, 1)
        out_shape = tuple(out_shape_list)
        # 使用新形状创建新的操作对象
        out_oper = in_oper._replace(shape=out_shape)

        # 初始化输入和输出列表
        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_scalar(dim)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        # 添加操作到操作列表
        self.add_operation(NNAPI_OperationCode.EXPAND_DIMS, inputs, outputs)

    def add_to(self, node):
        # 处理 to("cpu") / to("gpu") 的情况，直接调用self._identity
        self._identity(node)
    # 添加重塑操作到神经网络图中的方法，接受一个节点作为参数
    def add_reshape(self, node):
        # 确保节点有两个输入
        assert node.inputsSize() == 2
        # 确保节点有一个输出
        assert node.outputsSize() == 1

        # 获取第一个输入张量操作数的标识符和操作数
        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))

        # 获取第二个输入节点的常量值和类型
        shape_ctype, shape = self.get_constant_value(node.inputsAt(1))
        # 确保常量类型是列表类型
        assert shape_ctype.kind() == "ListType"
        # 确保列表中元素的类型是整数类型
        assert shape_ctype.getElementType().kind() == "IntType"
        # 检查是否是简单的重塑，即形状列表长度为2且第二个元素为-1
        is_trivial_reshape = len(shape) == 2 and shape[1] == -1

        # 如果输入张量操作数的维度顺序不是预设的连续顺序并且不是简单的重塑，则抛出异常
        if in_oper.dim_order != DimOrder.PRESUMED_CONTIGUOUS and not is_trivial_reshape:
            raise Exception(
                "Currently, reshape is only supported on NHWC tensors if the target size is [X, -1]."
            )

        # 通过使用一个真实的张量来推断输出形状，这里有点技巧性
        out_shape = torch.zeros(1).expand(in_oper.shape).reshape(shape).shape
        # 使用新的输出形状和预设的连续顺序来替换输入张量操作数的属性
        out_oper = in_oper._replace(
            shape=out_shape, dim_order=DimOrder.PRESUMED_CONTIGUOUS
        )

        # 准备输入和输出列表
        inputs = [None] * 2
        inputs[0] = in_id
        # 将形状列表作为立即整数向量添加到输入列表
        inputs[1] = self.add_immediate_int_vector(shape)

        outputs = [None] * 1
        # 添加新的张量操作数并将其作为输出列表的第一个元素
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        # 将重塑操作添加到神经网络图中，使用输入和输出列表
        self.add_operation(NNAPI_OperationCode.RESHAPE, inputs, outputs)
    # 定义一个方法用于向计算图中添加 flatten 操作节点
    def add_flatten(self, node):
        # 断言节点的输入张量数量为3
        assert node.inputsSize() == 3
        # 断言节点的输出张量数量为1
        assert node.outputsSize() == 1

        # 获取输入张量的操作数标识符和操作数对象
        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))

        # 获取起始维度和结束维度的常数值及类型
        start_ctype, start_dim = self.get_constant_value(node.inputsAt(1), "IntType")
        end_ctype, end_dim = self.get_constant_value(node.inputsAt(2), "IntType")

        # 检查是否为简单的 flatten 操作，即输入张量为四维且通道数为1，或者高度和宽度都为1
        is_trivial_flatten = len(in_oper.shape) == 4 and (
            in_oper.shape[1] == 1 or (in_oper.shape[2] == 1 and in_oper.shape[3] == 1)
        )
        # 如果输入张量的维度顺序不是预期的连续顺序并且不是简单的 flatten 操作，则抛出异常
        if in_oper.dim_order != DimOrder.PRESUMED_CONTIGUOUS and not is_trivial_flatten:
            raise Exception(
                "Currently, flatten is not supported on NHWC tensors unless C=1 or H=W=1"
            )

        # 将负数的起始维度和结束维度转换为有效索引
        if start_dim < 0:
            start_dim += len(in_oper.shape)
        if end_dim < 0:
            end_dim += len(in_oper.shape)

        # 计算输出张量的形状，进行 flatten 操作
        out_shape = (
            in_oper.shape[:start_dim]
            + (functools.reduce(operator.mul, in_oper.shape[start_dim : end_dim + 1]),)
            + in_oper.shape[end_dim + 1 :]
        )

        # 如果输出张量的某些维度为0，则抛出异常，因为不支持对灵活维度进行 flatten 操作
        if any(dim == 0 for dim in in_oper.shape[start_dim : end_dim + 1]):
            raise Exception(
                "Flattening flexible dims is not supported yet"
            )

        # 如果非 flatten 的维度中包含多个零，则抛出异常，因为只允许一个维度是灵活的
        non_flattened_dims = in_oper.shape[:start_dim] + in_oper.shape[end_dim + 1 :]
        if non_flattened_dims.count(0) > 1:
            raise Exception("Only 1 dim can be flexible")

        # 创建输出张量对象，替换原输入张量的形状和维度顺序
        out_oper = in_oper._replace(
            shape=out_shape, dim_order=DimOrder.PRESUMED_CONTIGUOUS
        )

        # 向计算图中添加输出张量操作，并获取其操作数标识符
        out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)

        # 对输出张量的形状进行进一步处理，如果某维度为0，则使用原输入张量对应维度的索引
        for idx, dim in enumerate(out_shape):
            if dim == 0:
                self.forward_operand_shape(out_id, idx, in_id, in_oper.shape.index(0))

        # 创建用于 NNAPI 操作的输入参数，将输出张量的形状添加到参数列表中
        inputs_1 = tuple(dim if dim != 0 else -1 for dim in out_shape)
        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector(inputs_1)

        # 创建用于 NNAPI 操作的输出参数列表
        outputs = [None] * 1
        outputs[0] = out_id

        # 向计算图中添加 NNAPI 操作节点（reshape 操作）
        self.add_operation(NNAPI_OperationCode.RESHAPE, inputs, outputs)
    # 定义一个方法，用于向节点添加切片操作
    def add_slice(self, node):
        # 断言节点的输入大小为5
        assert node.inputsSize() == 5
        # 断言节点的输出大小为1
        assert node.outputsSize() == 1

        # 获取第一个输入操作数的标识符和操作数
        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        # 获取第二个输入的常量值
        _, dim_value = self.get_constant_value(node.inputsAt(1))
        # 获取第三个输入的常量值
        _, start_value = self.get_constant_value(node.inputsAt(2))
        # 获取第四个输入的常量值
        _, stop_value = self.get_constant_value(node.inputsAt(3))
        # 获取第五个输入的常量值
        _, step_value = self.get_constant_value(node.inputsAt(4))

        # 如果开始值为None，则将其设为0
        if start_value is None:
            start_value = 0
        # 如果结束值为None，则将其设为系统最大值
        if stop_value is None:
            stop_value = sys.maxsize

        # 如果开始值为负数，则加上输入操作数的维度值
        if start_value < 0:
            start_value += in_oper.shape[dim_value]
        # 如果开始值等于系统最大值，则将其设为0
        elif start_value == sys.maxsize:
            start_value = 0

        # 如果开始值为0且结束值为系统最大值，则将节点设置为恒等操作
        if start_value == 0 and stop_value == sys.maxsize:
            self._identity(node)
            return

        # 如果输入操作数的维度值为0，则抛出异常
        if in_oper.shape[dim_value] == 0:
            raise Exception("Unable to slice with flexible shape")  # noqa: TRY002

        # 如果结束值为负数，则加上输入操作数的维度值
        if stop_value < 0:
            stop_value += in_oper.shape[dim_value]
        # 如果结束值等于系统最大值，则将其设为输入操作数的维度值
        elif stop_value == sys.maxsize:
            stop_value = in_oper.shape[dim_value]

        # 如果开始值大于等于结束值，则抛出异常
        if start_value >= stop_value:
            raise Exception(  # noqa: TRY002
                "Slice start value should be less than stop value"
            )  # noqa: TRY002

        # 计算输出长度
        out_len = (stop_value - start_value) // step_value
        # 计算输出形状
        out_shape = tuple(
            out_len if i == dim_value else dim for i, dim in enumerate(in_oper.shape)
        )
        # 添加张量操作数并获取其标识符
        out_id = self.add_tensor_operand(
            node.outputsAt(0), in_oper._replace(shape=out_shape)
        )

        # 处理灵活的输入
        end_mask = 0
        for idx, dim in enumerate(out_shape):
            if dim == 0:
                self.forward_operand_shape(out_id, idx, in_id, idx)
                end_mask |= 1 << idx

        # 准备输入列表
        inputs = [None] * 7
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector(
            [start_value if i == dim_value else 0 for i in range(len(in_oper.shape))]
        )
        inputs[2] = self.add_immediate_int_vector(
            [
                stop_value if i == dim_value else dim
                for i, dim in enumerate(in_oper.shape)
            ]
        )
        inputs[3] = self.add_immediate_int_vector(
            [step_value if i == dim_value else 1 for i in range(len(in_oper.shape))]
        )
        inputs[4] = self.add_immediate_int_scalar(0)  # begin mask
        inputs[5] = self.add_immediate_int_scalar(end_mask)
        inputs[6] = self.add_immediate_int_scalar(0)  # shrink axis mas

        # 准备输出列表
        outputs = [None] * 1
        outputs[0] = out_id

        # 添加 NNAPI 操作码为 STRIDED_SLICE 的操作
        self.add_operation(NNAPI_OperationCode.STRIDED_SLICE, inputs, outputs)
    # 添加节点大小信息，确保输入节点数为2
    def add_size(self, node):
        assert node.inputsSize() == 2
        # 确保输出节点数为1
        assert node.outputsSize() == 1

        # 获取第一个输入节点的张量操作数
        _, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        # 获取第二个输入节点的常量值
        _, value = self.constants[node.inputsAt(1)]
        # 从张量操作数中获取指定维度的大小
        res = in_oper.shape[value]
        # 获取输出节点
        output = node.outputsAt(0)
        # 将计算得到的大小作为常量值添加到输出节点
        self.add_constant_value(output, output.type(), res)

    # 添加节点的拼接操作
    def add_cat(self, node):
        assert node.inputsSize() == 2
        assert node.outputsSize() == 1

        # 获取第一个输入节点对应的张量序列及拼接的维度
        tensors = self.tensor_sequences[node.inputsAt(0)]
        _, dim = self.get_constant_value(node.inputsAt(1), "IntType")

        # 确保张量序列不为空
        assert len(tensors) > 0
        in_ids = []
        out_oper = None
        out_dim_size = 0
        # 遍历每个输入张量
        for inp in tensors:
            # 获取张量的操作数及标识符
            in_id, in_oper = self.get_tensor_operand_by_jitval(inp)
            # 如果输出操作数尚未设置，根据输入操作数的形状创建输出操作数
            if out_oper is None:
                out_shape = change_element(in_oper.shape, dim, -1)
                out_oper = in_oper._replace(shape=out_shape)
            # 确保输入操作数和输出操作数的类型和维度顺序一致
            assert in_oper.op_type == out_oper.op_type
            assert in_oper.dim_order == out_oper.dim_order
            # 确保经过维度拼接后的形状与预期一致
            assert change_element(in_oper.shape, dim, -1) == change_element(
                out_oper.shape, dim, -1
            )
            # 可能的TODO：检查比例和零点是否需要支持

            # 将输入操作数的标识符添加到列表中
            in_ids.append(in_id)
            # 可能的TODO：支持可变大小的输入

            # 累计输出操作数的拼接维度大小
            out_dim_size += in_oper.shape[dim]

        # 确保输出操作数已设置
        assert out_oper is not None
        # 根据累计的拼接维度大小更新输出操作数的形状
        out_oper = out_oper._replace(
            shape=change_element(out_oper.shape, dim, out_dim_size)
        )

        # 如果操作数维度顺序为CHANNELS_LAST，则进行特定处理
        if in_oper.dim_order == DimOrder.CHANNELS_LAST:  # type: ignore[possibly-undefined]
            assert len(out_oper.shape) == 4
            # 将维度映射到NNAPI的顺序
            nnapi_dim = [0, 3, 1, 2][dim]
        else:
            nnapi_dim = dim

        # 添加输出操作数并获取其标识符
        out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)
        # 遍历输出操作数的每个维度
        for idx, d in enumerate(out_oper.shape):
            # 如果维度大小为0，则根据特定规则计算形状
            if d == 0:
                if idx == dim:
                    shape = " + ".join(flex_name(ip_id, dim) for ip_id in in_ids)
                    self.compute_operand_shape(out_id, idx, shape)
                else:
                    self.forward_operand_shape(out_id, idx, in_ids[0], idx)

        # 将输入操作数和NNAPI维度作为输入列表
        inputs = in_ids + [self.add_immediate_int_scalar(nnapi_dim)]

        # 初始化输出列表
        outputs = [None] * 1
        # 将输出标识符添加到输出列表中
        outputs[0] = out_id

        # 添加拼接操作到操作列表中
        self.add_operation(NNAPI_OperationCode.CONCATENATION, inputs, outputs)
    # 定义一个方法用于向神经网络操作图中添加均值操作
    def add_mean(self, node):
        # 确保节点的输入大小为4
        assert node.inputsSize() == 4
        # 确保节点的输出大小为1
        assert node.outputsSize() == 1

        # 获取输入操作数的标识符和操作对象，通过节点的第一个输入
        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        # 获取维度常量的类型和值，通过节点的第二个输入
        dim_ctype, dim = self.get_constant_value(node.inputsAt(1))
        # 确保维度类型为列表类型
        assert dim_ctype.kind() == "ListType"
        # 确保列表元素类型为整数类型
        assert dim_ctype.getElementType().kind() == "IntType"
        # 获取保持维度参数的常量值，通过节点的第三个输入，预期为布尔类型
        _, keep_dim = self.get_constant_value(node.inputsAt(2), "BoolType")
        # 获取第四个输入的常量值，预期为NoneType，用于表示期望的数据类型
        # 此处的返回值在这个上下文中不用，因为它会直接传递给 get_constant_value()，以确保它是 NoneType

        # 如果输入操作对象的维度顺序是 CHANNELS_LAST
        if in_oper.dim_order == DimOrder.CHANNELS_LAST:
            # 确保输入操作对象的形状为四维
            assert len(in_oper.shape) == 4
            # 转换维度以符合 NNAPI 的预期顺序
            nnapi_dim = [[0, 3, 1, 2][d] for d in dim]
        else:
            nnapi_dim = dim

        # 创建一个集合，用于存储需要被合并的维度
        collapsed_dims = set()
        for d in dim:
            # 如果维度为负数，则将其转换为对应的正数索引
            if d < 0:
                d += len(in_oper.shape)
            collapsed_dims.add(d)

        # 如果输入操作对象的维度顺序是 CHANNELS_LAST 并且不保持维度
        if in_oper.dim_order == DimOrder.CHANNELS_LAST and not keep_dim:
            # 确保合并的维度包含 {2, 3}
            assert collapsed_dims.issuperset({2, 3})
            # 输出维度顺序设为 PRESUMED_CONTIGUOUS
            out_dim_order = DimOrder.PRESUMED_CONTIGUOUS
        else:
            # 否则，输出维度顺序与输入操作对象保持一致
            out_dim_order = in_oper.dim_order

        # 构建输出形状列表
        out_shape = []
        for i, s in enumerate(in_oper.shape):
            if i not in collapsed_dims:
                out_shape.append(s)
            elif keep_dim:
                out_shape.append(1)

        # 使用新的形状和维度顺序创建更新后的操作对象
        out_oper = in_oper._replace(shape=out_shape, dim_order=out_dim_order)

        # 初始化输入列表
        inputs = [None] * 3
        # 第一个输入为输入操作数的标识符
        inputs[0] = in_id
        # 第二个输入为转换后的 NNAPI 维度
        inputs[1] = self.add_immediate_int_vector(nnapi_dim)
        # 第三个输入为保持维度参数的整数标量
        inputs[2] = self.add_immediate_int_scalar(keep_dim)

        # 初始化输出列表
        outputs = [None] * 1
        # 第一个输出为经过更新后的输出操作对象
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        # 向神经网络操作图中添加 MEAN 操作，使用输入和输出列表
        self.add_operation(NNAPI_OperationCode.MEAN, inputs, outputs)
    # 添加量化操作到神经网络操作图中，该操作将输入节点进行量化处理
    def add_quantize(self, node):
        # 断言输入节点的输入大小为4
        assert node.inputsSize() == 4
        # 断言输入节点的输出大小为1
        assert node.outputsSize() == 1

        # 获取第一个输入节点的张量操作数及其标准化后的固定大小
        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        # 如果输入张量的维度顺序不是 CHANNELS_LAST，则抛出异常
        if in_oper.dim_order != DimOrder.CHANNELS_LAST:
            raise Exception(
                "Most hardware backends prefer NHWC quantized tensors.  "
                "Try setting `t.nnapi_nhwc = True` on your tensor inputs.  "
            )
        
        # 获取第二个输入节点的常量值及其浮点类型的尺度
        _, scale = self.get_constant_value(node.inputsAt(1), "FloatType")
        # 获取第三个输入节点的常量值及其整数类型的零点
        _, zero_point = self.get_constant_value(node.inputsAt(2), "IntType")
        # 获取第四个输入节点的常量值及其整数类型的标量类型
        _, scalar_type = self.get_constant_value(node.inputsAt(3), "IntType")
        # 如果标量类型不是 quint8 类型，则抛出异常
        if scalar_type != TorchScalarTypes.QUINT8.value:
            raise Exception(
                "PyTorch NNAPI export only supports quantized tensors "
                "with the quint8 dtype."
            )
        
        # 定义操作类型为 NNAPI_OperandCode.TENSOR_QUANT8_ASYMM
        op_type = NNAPI_OperandCode.TENSOR_QUANT8_ASYMM

        # 用输出操作数替换输入操作数，进行量化处理
        out_oper = in_oper._replace(
            op_type=op_type,
            scale=scale,
            zero_point=zero_point,
        )

        # 初始化输入列表和输出列表
        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        # 将量化后的输出操作添加到操作图中
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        # 添加量化操作到操作图中
        self.add_operation(NNAPI_OperationCode.QUANTIZE, inputs, outputs)

    # 添加反量化操作到神经网络操作图中，该操作将输入节点进行反量化处理
    def add_dequantize(self, node):
        # 断言输入节点的输入大小为1
        assert node.inputsSize() == 1
        # 断言输入节点的输出大小为1
        assert node.outputsSize() == 1

        # 获取输入节点的张量操作数及其标准化后的固定大小
        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        
        # 将输出操作数替换为浮点类型的反量化操作
        out_oper = in_oper._replace(
            op_type=NNAPI_OperandCode.TENSOR_FLOAT32,
            scale=0.0,
            zero_point=0,
        )

        # 初始化输入列表和输出列表
        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        # 将反量化后的输出操作添加到操作图中
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        # 添加反量化操作到操作图中
        self.add_operation(NNAPI_OperationCode.DEQUANTIZE, inputs, outputs)

    # 添加简单的逐点一元操作到神经网络操作图中，根据给定的操作码进行处理
    def add_pointwise_simple_unary_op(self, node, opcode):
        # 断言输入节点的输入大小为1
        assert node.inputsSize() == 1
        # 断言输入节点的输出大小为1
        assert node.outputsSize() == 1

        # 获取输入节点的张量操作数及其标准化后的尺寸
        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))

        # 将输出操作数初始化为输入操作数
        out_oper = in_oper
        # 如果操作码是 LOGISTIC，则根据 NNAPI 文档进行特定处理
        if opcode == NNAPI_OperationCode.LOGISTIC:
            # NNAPI 文档要求对于 ANEURALNETWORKS_TENSOR_QUANT8_ASYMM，scale 必须为 1.f / 256，zeroPoint 必须为 0
            if in_oper.op_type == NNAPI_OperandCode.TENSOR_QUANT8_ASYMM:
                out_oper = in_oper._replace(zero_point=0, scale=1.0 / 256)

        # 将输出操作数添加到操作图中，并获取其输出 ID
        out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)

        # 遍历输入操作数的每个维度，如果维度为 0，则进行前向操作数形状的设置
        for idx, dim in enumerate(in_oper.shape):
            if dim == 0:
                self.forward_operand_shape(out_id, idx, in_id, idx)

        # 初始化输入列表和输出列表
        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        outputs[0] = out_id

        # 添加逐点一元操作到操作图中
        self.add_operation(opcode, inputs, outputs)
    def _do_add_binary(self, node, opcode, fuse_code, *, qparams=None):  # noqa: D401
        """Helper for pointwise binary broadcast ops with superfluous extra args."""
        assert node.outputsSize() == 1  # 确保节点输出只有一个

        assert node.inputsAt(0).type().kind() == "TensorType"  # 确保第一个输入是张量类型
        assert node.inputsAt(1).type().kind() == "TensorType"  # 确保第二个输入是张量类型

        # 根据是否有用于 JIT 值的操作数，获取输入张量的操作数和标识
        if self.has_operand_for_jitval(node.inputsAt(0)):
            in0_id, in0_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
            in1_id, in1_oper = self.get_tensor_operand_or_constant(
                node.inputsAt(1), in0_oper.dim_order
            )
        elif self.has_operand_for_jitval(node.inputsAt(1)):
            in1_id, in1_oper = self.get_tensor_operand_by_jitval(node.inputsAt(1))
            in0_id, in0_oper = self.get_tensor_operand_or_constant(
                node.inputsAt(0), in1_oper.dim_order
            )
        else:
            raise Exception(  # 若两个输入都是常量，则抛出异常
                f"Can't do a NNAPI binary op: {opcode} on two constants"
            )

        assert in0_oper.op_type == in1_oper.op_type  # 确保两个操作数的操作类型相同
        in0_id, in0_oper, in1_id, in1_oper = self.transpose_for_broadcast(
            in0_id, in0_oper, in1_id, in1_oper
        )  # 为广播操作进行转置

        # 注意：PyTorch 和 NNAPI 具有相同的广播语义。
        out_shape = broadcast_shapes(in0_oper.shape, in1_oper.shape)  # 计算输出张量的形状
        out_oper = in0_oper._replace(shape=out_shape)  # 替换输出张量的形状
        if qparams is not None:
            scale, zp = qparams
            out_oper = out_oper._replace(scale=scale, zero_point=zp)  # 如果存在量化参数，则更新输出张量的参数

        out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)  # 添加输出张量操作数

        for idx, (d0, d1) in enumerate(zip(in0_oper.shape, in1_oper.shape)):
            if d0 == 1 and d1 == 0:
                self.forward_operand_shape(out_id, idx, in1_id, idx)  # 如果维度 d0 为 1，d1 为 0，更新输出形状
            elif d0 == 0 and d1 == 1:
                self.forward_operand_shape(out_id, idx, in0_id, idx)  # 如果维度 d0 为 0，d1 为 1，更新输出形状
            elif d0 == 0 and d1 == 0:
                self.flexible_shape_computation_lines.append(
                    f"assert {flex_name(in0_id, idx)} == {flex_name(in1_id, idx)}"
                )  # 如果维度 d0 和 d1 都为 0，添加灵活形状计算的行

                self.forward_operand_shape(out_id, idx, in0_id, idx)  # 更新输出形状

        inputs = [None] * 3
        inputs[0] = in0_id
        inputs[1] = in1_id
        inputs[2] = self.add_immediate_int_scalar(fuse_code)  # 添加立即整数标量作为输入之一

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(opcode, inputs, outputs)  # 添加操作到图中

    def add_pointwise_simple_binary_broadcast_op(self, node, opcode, fuse_code):
        assert node.inputsSize() == 2  # 确保节点有两个输入
        self._do_add_binary(node, opcode, fuse_code)  # 调用 _do_add_binary 方法进行简单点对点二进制广播操作的添加
    # 添加 add_add_sub_op 方法，处理节点的加法或减法操作
    def add_add_sub_op(self, node, opcode, fuse_code):
        # 确保节点的输入大小为3
        assert node.inputsSize() == 3

        # 获取第三个输入节点的常量值
        _, alpha = self.get_constant_value(node.inputsAt(2), "IntType")
        # 如果 alpha 不等于1，则抛出异常
        if alpha != 1:
            raise Exception(
                "NNAPI does not support add/sub with alpha."
            )

        # 执行二进制加法操作
        self._do_add_binary(node, opcode, fuse_code)

    # 添加 add_qadd 方法，处理节点的量化加法操作
    def add_qadd(self, node, opcode, fuse_code):
        # 确保节点的输入大小为4
        assert node.inputsSize() == 4

        # 获取第三个输入节点的常量值和第四个输入节点的常量值
        _, scale = self.get_constant_value(node.inputsAt(2), "FloatType")
        _, zero_point = self.get_constant_value(node.inputsAt(3), "IntType")

        # 执行二进制加法操作，传入量化参数 (scale, zero_point)
        self._do_add_binary(node, opcode, fuse_code, qparams=(scale, zero_point))

    # 添加 add_softmax 方法，处理节点的 softmax 操作
    def add_softmax(self, node):
        # 确保节点的输入大小为3
        assert node.inputsSize() == 3
        # 获取第一个输入节点的操作数和操作
        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))

        # 获取第二个输入节点的常量值，即 softmax 的维度
        _, softmax_dim = self.get_constant_value(node.inputsAt(1), "IntType")

        # 添加输出操作的操作数
        out_id = self.add_tensor_operand(node.outputsAt(0), in_oper)
        
        # 遍历输入操作的形状
        for dim, size in enumerate(in_oper.shape):
            # 如果某一维度为0，前向传播操作形状
            if size == 0:
                self.forward_operand_shape(out_id, dim, in_id, dim)

        # 构建输入列表，包括输入操作数、1.0 的浮点标量、softmax_dim 的整数标量
        inputs = [None] * 3
        inputs[0] = in_id
        inputs[1] = self.add_immediate_float_scalar(
            1.0
        )  # positive scaling factor of exponent, beta
        inputs[2] = self.add_immediate_int_scalar(softmax_dim)

        # 构建输出列表，仅包括输出操作数
        outputs = [None] * 1
        outputs[0] = out_id

        # 添加 NNAPI 操作码为 SOFTMAX 的操作
        self.add_operation(NNAPI_OperationCode.SOFTMAX, inputs, outputs)

    # 添加 add_hardtanh 方法，处理节点的硬切线操作
    def add_hardtanh(self, node):
        # 确保节点的输入大小为3和输出大小为1
        assert node.inputsSize() == 3
        assert node.outputsSize() == 1

        # 获取输入节点的操作数和固定大小的操作
        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        
        # 获取第二个输入节点的常量值和第三个输入节点的常量值，即最小值和最大值
        _, min_val = self.get_constant_value(node.inputsAt(1), "FloatType")
        _, max_val = self.get_constant_value(node.inputsAt(2), "FloatType")

        # 定义操作码映射
        op_map = {
            (-1, 1): NNAPI_OperationCode.RELU1,
            (0, 6): NNAPI_OperationCode.RELU6,  # noqa: E201
        }

        # 根据最小值和最大值从映射中获取操作码
        opcode = op_map.get((min_val, max_val))
        # 如果操作码为 None，则抛出异常
        if opcode is None:
            raise Exception(
                "NNAPI only supports hardtanh with args (-1, 1) or (0, 6)."
            )

        # 构建输入列表，仅包括输入操作数
        inputs = [None] * 1
        inputs[0] = in_id

        # 构建输出列表，包括输出操作数
        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), in_oper)

        # 添加相应操作码的操作
        self.add_operation(opcode, inputs, outputs)
    # 定义一个方法用于在计算图中添加 PReLU 操作节点
    def add_prelu_op(self, node):
        # 确保输入和输出的数量符合预期
        assert node.inputsSize() == 2
        assert node.outputsSize() == 1

        # 确保输入节点的类型是张量类型
        assert node.inputsAt(0).type().kind() == "TensorType"
        assert node.inputsAt(1).type().kind() == "TensorType"

        # 获取输入张量的操作数标识和操作对象
        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        # 获取权重张量的操作数标识和操作对象
        w_id, w_oper = self.get_tensor_operand_for_weight(node.inputsAt(1))
        
        # 确保权重张量是一维的
        assert len(w_oper.shape) == 1
        assert w_oper.shape[0] > 0
        
        # 如果权重张量的维度大于1且输入张量采用 channels first 的布局
        if w_oper.shape[0] > 1:
            if in_oper.use_nchw():
                # TODO: 通过在末尾添加1维度来支持此功能
                raise Exception(
                    "Per-channel PReLU only supports channels_last right now."
                )

        # 添加输出张量操作并获取其标识
        out_id = self.add_tensor_operand(node.outputsAt(0), in_oper)
        
        # 遍历输入张量的形状维度
        for dim, size in enumerate(in_oper.shape):
            # 如果维度大小大于0，则继续
            if size > 0:
                pass
            # 对于维度小于等于1的情况，抛出异常
            elif dim <= 1:
                raise Exception(
                    "PReLU requires fixed size for dim 0 and dim 1."
                )
            # 否则，根据输入张量的形状和维度，推断输出张量的形状
            else:
                self.forward_operand_shape(out_id, dim, in_id, dim)

        # 准备操作的输入和输出列表
        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = w_id

        outputs = [None] * 1
        outputs[0] = out_id

        # 添加 PReLU 操作到计算图中
        self.add_operation(NNAPI_OperationCode.PRELU, inputs, outputs)
    def add_pool2d_node(self, node, opcode):
        # 确保节点的输入参数数量为6
        assert node.inputsSize() == 6
        # 确保节点的输出参数数量为1
        assert node.outputsSize() == 1
        # 从节点的输入中获取图像、卷积核、步幅、填充、扩展、ceil_mode 参数
        image, kernel, stride, padding, dilation, ceil_mode = node.inputs()

        # 如果步幅为 None，则使用卷积核作为步幅
        stride = stride or kernel

        # TODO: 验证 ceil_mode 的语义

        # 从 JIT 中获取 2D 卷积池化的参数
        args = self.get_conv_pool_args_2d_from_jit(
            self.get_size_arg(kernel), stride, padding, dilation
        )
        # 如果扩展不是 1，抛出异常，因为 NNAPI 不支持扩展池化
        if args.dilation_h != 1 or args.dilation_w != 1:
            raise Exception("NNAPI does not support dilated pooling.")  # noqa: TRY002

        # 通过 JIT 值和固定大小获取图像操作数 ID 和操作
        image_id, image_oper = self.get_tensor_operand_by_jitval_fixed_size(image)
        # 确保图像操作的形状为四维
        assert len(image_oper.shape) == 4

        # 计算池化后的输出形状
        out_shape = get_conv_pool_shape(
            image_oper.shape, args, image_oper.shape[1], False
        )
        # 判断是否使用 NCHW 格式
        use_nchw = image_oper.use_nchw()

        # 创建长度为 11 的输入列表，初始化为 None
        inputs = [None] * 11
        # 设置各种参数值到输入列表中
        inputs[0] = image_id
        inputs[1] = self.add_immediate_int_scalar(args.pad_l)
        inputs[2] = self.add_immediate_int_scalar(args.pad_r)
        inputs[3] = self.add_immediate_int_scalar(args.pad_t)
        inputs[4] = self.add_immediate_int_scalar(args.pad_b)
        inputs[5] = self.add_immediate_int_scalar(args.stride_w)
        inputs[6] = self.add_immediate_int_scalar(args.stride_h)
        inputs[7] = self.add_immediate_int_scalar(args.kernel_w)
        inputs[8] = self.add_immediate_int_scalar(args.kernel_h)
        inputs[9] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)
        inputs[10] = self.add_immediate_bool_scalar(use_nchw)

        # 创建长度为 1 的输出列表，初始化为 None
        outputs = [None] * 1
        # 添加张量操作数到输出列表中，同时更新其形状
        outputs[0] = self.add_tensor_operand(
            node.outputsAt(0), image_oper._replace(shape=out_shape)
        )

        # 添加操作到操作列表中，包括操作码、输入和输出列表
        self.add_operation(opcode, inputs, outputs)
    # 定义一个方法，用于在神经网络模型中添加平均池化操作节点
    def add_avg_pool2d(self, node):
        # 断言输入节点的输入数量为7
        assert node.inputsSize() == 7
        # 断言输入节点的输出数量为1
        assert node.outputsSize() == 1
        
        # 将节点的输入解包为多个变量，分别表示图像、卷积核、步长、填充、取整模式、计算中包含填充、覆盖除数
        (
            image,
            kernel,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        ) = node.inputs()

        # 获取计算中包含填充和覆盖除数的常量值
        _, count_include_pad_value = self.get_constant_value(count_include_pad)
        _, divisor_override_value = self.get_constant_value(divisor_override)
        
        # 如果计算中不包含填充或覆盖除数，则抛出异常
        if not count_include_pad_value or divisor_override_value:
            raise Exception(
                "NNAPI doesn't support count_include_pad=False or divisor_override"
            )

        # 根据卷积核的大小、步长和填充，获取卷积池化的参数
        args = self.get_conv_pool_args_2d_from_jit(
            self.get_size_arg(kernel), stride, padding
        )

        # 获取图像操作数的ID和操作对象
        image_id, image_oper = self.get_tensor_operand_by_jitval(image)
        
        # 断言图像操作对象的形状为4维
        assert len(image_oper.shape) == 4

        # 计算输出形状，通过给定的卷积池化参数、通道数和是否使用NCHW格式
        out_shape = get_conv_pool_shape(
            image_oper.shape, args, image_oper.shape[1], False
        )
        
        # 检查是否使用NCHW格式
        use_nchw = image_oper.use_nchw()

        # 初始化输入列表，包含11个元素
        inputs = [None] * 11
        inputs[0] = image_id  # 图像操作数的ID
        inputs[1] = self.add_immediate_int_scalar(args.pad_l)  # 添加整数类型的填充左边界
        inputs[2] = self.add_immediate_int_scalar(args.pad_r)  # 添加整数类型的填充右边界
        inputs[3] = self.add_immediate_int_scalar(args.pad_t)  # 添加整数类型的填充顶部边界
        inputs[4] = self.add_immediate_int_scalar(args.pad_b)  # 添加整数类型的填充底部边界
        inputs[5] = self.add_immediate_int_scalar(args.stride_w)  # 添加整数类型的步长宽度
        inputs[6] = self.add_immediate_int_scalar(args.stride_h)  # 添加整数类型的步长高度
        inputs[7] = self.add_immediate_int_scalar(args.kernel_w)  # 添加整数类型的卷积核宽度
        inputs[8] = self.add_immediate_int_scalar(args.kernel_h)  # 添加整数类型的卷积核高度
        inputs[9] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)  # 添加整数类型的融合码
        inputs[10] = self.add_immediate_bool_scalar(use_nchw)  # 添加布尔类型的是否使用NCHW格式

        # 初始化输出列表，包含1个元素
        outputs = [None] * 1
        # 添加张量操作数，将输出张量ID和具有更新形状的图像操作数替换
        out_id = self.add_tensor_operand(
            node.outputsAt(0), image_oper._replace(shape=out_shape)
        )
        # 处理灵活输入的卷积池化
        self._handle_conv_pool_flexible_input(out_id, image, args, False)
        outputs[0] = out_id

        # 添加平均池化2D操作，使用输入和输出列表
        self.add_operation(NNAPI_OperationCode.AVERAGE_POOL_2D, inputs, outputs)
    # 添加自适应平均池化操作到神经网络计算图中
    def add_adaptive_avg_pool2d(self, node):
        # 确保节点输入张量数量为2
        assert node.inputsSize() == 2
        # 确保节点输出张量数量为1
        assert node.outputsSize() == 1

        # 从节点输入中获取图像操作数和图像操作
        image_id, image_oper = self.get_tensor_operand_by_jitval_fixed_size(
            node.inputsAt(0)
        )
        # 确保图像操作的形状是四维的
        assert len(image_oper.shape) == 4

        # 获取尺寸参数的类型和值
        size_ctype, size_arg = self.get_constant_value(node.inputsAt(1))
        # 确保尺寸参数的类型是列表类型
        assert size_ctype.kind() == "ListType"
        # 确保尺寸参数列表中的元素类型是整数类型
        assert size_ctype.getElementType().kind() == "IntType"
        # 如果尺寸参数不是[1, 1]，则抛出异常
        if size_arg != [1, 1]:
            raise Exception(
                "NNAPI only supports adaptive_avg_pool2d with output size (1, 1)."
            )

        # 计算输出张量的形状，将图像操作的前两个维度与尺寸参数拼接
        out_shape = image_oper.shape[0:2] + tuple(size_arg)
        # 检查是否使用NCHW格式
        use_nchw = image_oper.use_nchw()

        # 初始化输入参数列表，共11个元素
        inputs = [None] * 11
        inputs[0] = image_id
        inputs[1] = self.add_immediate_int_scalar(0)
        inputs[2] = self.add_immediate_int_scalar(0)
        inputs[3] = self.add_immediate_int_scalar(0)
        inputs[4] = self.add_immediate_int_scalar(0)
        inputs[5] = self.add_immediate_int_scalar(1)
        inputs[6] = self.add_immediate_int_scalar(1)
        inputs[7] = self.add_immediate_int_scalar(image_oper.shape[3])
        inputs[8] = self.add_immediate_int_scalar(image_oper.shape[2])
        inputs[9] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)
        inputs[10] = self.add_immediate_bool_scalar(use_nchw)

        # 初始化输出参数列表，共1个元素
        outputs = [None] * 1
        # 将输出张量添加到操作数列表中，并更新其形状为计算得到的形状
        outputs[0] = self.add_tensor_operand(
            node.outputsAt(0), image_oper._replace(shape=out_shape)
        )

        # 向神经网络计算图中添加平均池化操作
        self.add_operation(NNAPI_OperationCode.AVERAGE_POOL_2D, inputs, outputs)
    
    # 向神经网络计算图中添加加法矩阵乘法操作
    def add_addmm(self, node):
        # 确保节点输入张量数量为5
        assert node.inputsSize() == 5
        # 确保节点输出张量数量为1
        assert node.outputsSize() == 1
        # 从节点输入中获取偏置、输入、权重、beta和alpha的jit值
        jit_bias, jit_input, jit_weight, jit_beta, jit_alpha = node.inputs()

        # 遍历beta和alpha的jit值，检查其尺寸和类型
        for jitval in (jit_beta, jit_alpha):
            scale_ctype, scale_value = self.get_constant_value(jitval)
            # 确保尺寸值的类型是整数或浮点数类型
            assert scale_ctype.kind() in ("IntType", "FloatType")
            # 如果尺寸值不等于1，抛出异常
            if scale_value != 1:
                raise Exception(
                    "NNAPI Fully-Connected does not support alpha and beta."
                )

        # 向神经网络计算图中添加加法矩阵乘法或线性操作
        self.add_addmm_or_linear(node, True, jit_input, jit_weight, jit_bias)
    
    # 向神经网络计算图中添加线性操作
    def add_linear(self, node):
        # 确保节点输入张量数量为3
        assert node.inputsSize() == 3
        # 确保节点输出张量数量为1
        assert node.outputsSize() == 1
        # 从节点输入中获取输入、权重和偏置的jit值
        jit_input, jit_weight, jit_bias = node.inputs()

        # 向神经网络计算图中添加加法矩阵乘法或线性操作
        self.add_addmm_or_linear(node, False, jit_input, jit_weight, jit_bias)
    
    # 向神经网络计算图中添加加法矩阵乘法或线性操作
    def add_addmm_or_linear(
        self, node, transpose_weight, jit_input, jit_weight, jit_bias
    ):
        # 省略具体实现内容，不在这里添加注释
        ):
            # 从 JIT 值中获取输入张量的标识和操作对象
            input_id, input_oper = self.get_tensor_operand_by_jitval(jit_input)
            # 从 JIT 值中获取偏置张量的标识和操作对象
            bias_id, bias_oper = self.get_tensor_operand_for_weight(jit_bias)

            # 断言输入张量的形状为二维
            assert len(input_oper.shape) == 2
            # 断言偏置张量的形状为一维
            assert len(bias_oper.shape) == 1

            # TODO: 在加载时转换，与 CPU 模型共享权重
            # 获取常量值并确保返回的是 TensorType
            _, weight_tensor = self.get_constant_value(jit_weight, "TensorType")
            # 断言权重张量的形状为二维
            assert len(weight_tensor.shape) == 2
            # 如果需要转置权重张量，则进行转置并保持连续性
            if transpose_weight:
                nnapi_weight_tensor = weight_tensor.t().contiguous()
            else:
                nnapi_weight_tensor = weight_tensor.contiguous()
            # 添加转换后的权重张量的张量操作数，并获取操作对象
            weight_id = self.add_tensor_operand_for_weight(nnapi_weight_tensor)
            weight_oper = self.operands[weight_id]

            # 输出张量的形状为输入张量的行数和权重张量的行数
            out_shape = (input_oper.shape[0], weight_oper.shape[0])
            # 添加输出张量的张量操作数，使用新的形状替换输入张量的形状
            out_id = self.add_tensor_operand(
                node.outputsAt(0), input_oper._replace(shape=out_shape)
            )

            # 如果输入张量的行数为0，执行前向操作的张量形状设置
            if input_oper.shape[0] == 0:
                self.forward_operand_shape(out_id, 0, input_id, 0)

            # 构建输入张量的列表，包括输入标识、权重标识、偏置标识以及融合代码的整数标量
            inputs = [None] * 4
            inputs[0] = input_id
            inputs[1] = weight_id
            inputs[2] = bias_id
            inputs[3] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)

            # 构建输出张量的列表，仅包括输出标识
            outputs = [None] * 1
            outputs[0] = out_id

            # 添加一个全连接操作，使用 NNAPI 的全连接操作码、输入列表和输出列表
            self.add_operation(NNAPI_OperationCode.FULLY_CONNECTED, inputs, outputs)
    # 定义一个方法用于向量量化操作（QLinear）节点的添加
    def add_qlinear(self, node):
        # 断言节点的输入大小为4，确保输入正确
        assert node.inputsSize() == 4
        # 断言节点的输出大小为1，确保输出正确
        assert node.outputsSize() == 1
        (
            jit_input,
            jit_packed_weight,
            jit_scale,
            jit_zero_point,
        ) = node.inputs()

        # 通过节点的 JIT 值获取输入操作数的 ID 和操作数对象
        input_id, input_oper = self.get_tensor_operand_by_jitval_fixed_size(jit_input)
        # TODO: 支持自动重塑（reshape）
        # 断言输入操作数的形状为二维
        assert len(input_oper.shape) == 2

        # 通过 JIT 值获取输出比例和零点值的常数值
        _, out_scale = self.get_constant_value(jit_scale, "FloatType")
        _, out_zero_point = self.get_constant_value(jit_zero_point, "IntType")
        # 获取打包权重的常数值的类型和数据
        weight_ctype, packed_weight = self.get_constant_value(jit_packed_weight)
        # 断言权重的类型为 LinearPackedParamsBase
        assert weight_ctype.name() == "LinearPackedParamsBase"
        # 解包权重和偏置
        raw_weight, raw_bias = packed_weight.__getstate__()[0]
        # 断言偏置不为空
        assert raw_bias is not None

        # 断言解包的权重是二维的
        assert len(raw_weight.shape) == 2
        # 断言解包的偏置是一维的
        assert len(raw_bias.shape) == 1
        # 断言偏置的长度等于权重的行数
        assert raw_bias.shape[0] == raw_weight.shape[0]
        # 断言权重的列数等于输入操作数的列数
        assert raw_weight.shape[1] == input_oper.shape[1]

        # 断言权重的量化模式是每张量的仿射量化
        assert raw_weight.qscheme() == torch.per_tensor_affine
        # 如果权重的数据类型是 uint8，则直接使用它
        if raw_weight.dtype == torch.quint8:
            unsigned_weight = raw_weight
        else:
            # 否则，权重的数据类型应为 qint8
            assert raw_weight.dtype == torch.qint8
            # 将 qint8 类型的权重转换为 uint8 类型
            unsigned_weight = torch._make_per_tensor_quantized_tensor(
                (raw_weight.int_repr().int() + 128).to(torch.uint8),
                scale=raw_weight.q_scale(),
                zero_point=raw_weight.q_zero_point() + 128,
            )
        # 获取权重的量化比例
        weight_scale = unsigned_weight.q_scale()
        # 计算偏置的量化比例
        bias_scale = input_oper.scale * weight_scale
        # 对偏置进行整数量化
        int_bias = torch.quantize_per_tensor(raw_bias, bias_scale, 0, torch.qint32)
        # 将整数量化后的偏置添加到操作数中
        bias_id = self.add_tensor_operand_for_weight(int_bias)

        # 计算乘数，用于量化全连接操作
        multiplier = input_oper.scale * weight_scale / out_scale
        # 断言乘数大于0
        assert multiplier > 0
        # 如果乘数大于等于1，抛出异常
        if multiplier >= 1:
            raise Exception(  # noqa: TRY002
                "Quantized convolution multiplier is greater than 1.  "
                "This is supported by NNAPI, but not by most hardware backends.  "
                "Try training a model without quantization-aware training.  "
            )

        # TODO: 在加载时转换以与 CPU 模型共享权重
        # 获取 NNAPI 权重张量，并确保其是连续的
        nnapi_weight_tensor = unsigned_weight.contiguous()
        # 将 NNAPI 权重张量添加到操作数中
        weight_id = self.add_tensor_operand_for_weight(nnapi_weight_tensor)
        # 获取权重操作数对象
        weight_oper = self.operands[weight_id]

        # 计算输出形状
        out_shape = (input_oper.shape[0], weight_oper.shape[0])
        # 创建新的输出操作数对象，替换形状、比例和零点值
        out_oper = input_oper._replace(
            shape=out_shape,
            scale=out_scale,
            zero_point=out_zero_point,
        )

        # 创建一个长度为4的输入列表
        inputs = [None] * 4
        # 将输入操作数的 ID、权重操作数的 ID、偏置操作数的 ID 和融合代码添加到输入列表中
        inputs[0] = input_id
        inputs[1] = weight_id
        inputs[2] = bias_id
        inputs[3] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)

        # 创建一个长度为1的输出列表，并将输出操作数的 ID 添加到列表中
        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        # 添加一个全连接操作到操作列表中，指定操作码、输入和输出
        self.add_operation(NNAPI_OperationCode.FULLY_CONNECTED, inputs, outputs)
    # 获取可选的偏置项，根据情况返回偏置的操作数ID和操作对象
    def get_optional_bias(self, jit_bias, weight_tensor, transpose=False):
        # 获取常量值的类型和数值
        ctype, value = self.get_constant_value(jit_bias)
        # 如果常量类型为NoneType，则根据是否转置选择偏置索引
        if ctype.kind() == "NoneType":
            bias_idx = 1 if transpose else 0
            # 创建一个与权重张量同形状的零张量作为NNAPI偏置张量
            nnapi_bias_tensor = torch.zeros(
                weight_tensor.size()[bias_idx], dtype=weight_tensor.dtype
            )
            # 添加NNAPI偏置张量作为操作数，并获取操作对象
            bias_id = self.add_tensor_operand_for_weight(nnapi_bias_tensor)
            bias_oper = self.operands[bias_id]
            return bias_id, bias_oper
        else:
            # 否则，返回根据常量值获取的权重张量的操作数
            return self.get_tensor_operand_for_weight(jit_bias)

    # 添加2D卷积操作
    def add_conv2d(self, node):
        assert node.inputsSize() == 7
        assert node.outputsSize() == 1

        # 解包节点的输入
        (
            jit_image,
            jit_weight,
            jit_bias,
            jit_stride,
            jit_pad,
            jit_dilation,
            jit_groups,
        ) = node.inputs()

        # 获取常量值的类型和权重张量
        _, weight_tensor = self.get_constant_value(jit_weight, "TensorType")
        # 获取可选的偏置项的ID和操作对象
        bias_id, bias_oper = self.get_optional_bias(jit_bias, weight_tensor)
        # 从JIT获取2D卷积的参数
        args = self.get_conv_pool_args_2d_from_jit(
            weight_tensor.shape[2:4], jit_stride, jit_pad, jit_dilation, jit_groups
        )

        # 调用通用的2D卷积添加方法，返回操作数ID
        return self.add_conv2d_common(
            node.outputsAt(0),
            0.0,
            0,
            jit_image,
            weight_tensor,
            bias_id,
            args,
            False,  # transpose
            NNAPI_FuseCode.FUSED_NONE,
        )

    # 添加下划线命名的卷积操作
    def add_conv_underscore(self, node):
        assert node.inputsSize() == 13
        assert node.outputsSize() == 1

        # 解包节点的输入
        (
            jit_image,
            jit_weight,
            jit_bias,
            jit_stride,
            jit_pad,
            jit_dilation,
            jit_transpose,
            _,
            jit_groups,
            _,
            _,
            _,
            _,
        ) = node.inputs()

        # 获取常量值的类型和权重张量
        _, weight_tensor = self.get_constant_value(jit_weight, "TensorType")
        # 获取常量值的类型和是否转置
        _, transpose = self.get_constant_value(jit_transpose)
        # 获取可选的偏置项的ID和操作对象，传入是否转置的参数
        bias_id, bias_oper = self.get_optional_bias(jit_bias, weight_tensor, transpose)
        # 从JIT获取2D卷积的参数
        args = self.get_conv_pool_args_2d_from_jit(
            weight_tensor.shape[2:4], jit_stride, jit_pad, jit_dilation, jit_groups
        )

        # 调用通用的2D卷积添加方法，返回操作数ID
        return self.add_conv2d_common(
            node.outputsAt(0),
            0.0,
            0,
            jit_image,
            weight_tensor,
            bias_id,
            args,
            transpose,
            NNAPI_FuseCode.FUSED_NONE,
        )
    # 定义一个方法用来在神经网络模型中添加 log softmax 操作
    def add_log_softmax(self, node):
        # 断言输入节点有三个输入
        assert node.inputsSize() == 3
        # 断言输出节点有一个输出
        assert node.outputsSize() == 1

        # 从节点中获取三个输入值
        (jit_input, jit_dim, jit_half_to_float) = node.inputs()
        # 使用 self 对象的方法获取与 jit_input 关联的张量操作的标识符和操作对象
        input_id, input_oper = self.get_tensor_operand_by_jitval_fixed_size(jit_input)
        # 获取常量值 jit_dim，并将其解析为整数 dim
        _, dim = self.get_constant_value(jit_dim, "IntType")

        # 获取输入操作对象的形状，并将其保存在 out_shape 变量中
        out_shape = input_oper.shape

        # 创建一个长度为 3 的空列表 inputs
        inputs = [None] * 3
        # 将 input_id 放入 inputs 列表的第一个位置
        inputs[0] = input_id
        # 将 1 作为指数的缩放因子（即 beta）放入 inputs 列表的第二个位置
        inputs[1] = self.add_immediate_float_scalar(1)
        # 将 dim 放入 inputs 列表的第三个位置
        inputs[2] = self.add_immediate_int_scalar(dim)

        # 创建一个长度为 1 的空列表 outputs
        outputs = [None] * 1
        # 将使用 node.outputsAt(0) 和重新设定形状后的 input_oper 添加到 outputs 列表的第一个位置
        outputs[0] = self.add_tensor_operand(
            node.outputsAt(0), input_oper._replace(shape=out_shape)
        )
        # 向神经网络模型中添加一个 NNAPI_OperationCode.LOG_SOFTMAX 操作，使用 inputs 和 outputs
        self.add_operation(NNAPI_OperationCode.LOG_SOFTMAX, inputs, outputs)
    # 定义一个方法用于向计算图中添加量化卷积层节点
    def add_qconv2d(self, node, fuse_code, transpose=False):
        # 断言输入节点的输入张量数量为4
        assert node.inputsSize() == 4
        # 断言输入节点的输出张量数量为1
        assert node.outputsSize() == 1

        # 从输入节点中获取四个输入张量：图像、打包的权重、量化比例、量化零点
        (
            jit_image,
            jit_packed_weight,
            jit_scale,
            jit_zero_point,
        ) = node.inputs()

        # 获取量化比例和量化零点的常量值
        _, out_scale = self.get_constant_value(jit_scale, "FloatType")
        _, out_zero_point = self.get_constant_value(jit_zero_point, "IntType")
        # 获取打包的权重的常量值及其类型
        weight_ctype, packed_weight = self.get_constant_value(jit_packed_weight)
        # 断言打包的权重的类型为"Conv2dPackedParamsBase"
        assert weight_ctype.name() == "Conv2dPackedParamsBase"
        # 解析打包的权重，获取打包版本、张量数据、可选张量数据
        (
            pack_version,
            tensors,
            opt_tensors,
        ) = packed_weight.__getstate__()[0]
        # 断言打包版本为"2"
        assert pack_version == "2"
        # 解析张量数据和可选张量数据
        packed_config, raw_weight = tensors
        (raw_bias,) = opt_tensors
        # 断言存在原始偏置数据
        assert raw_bias is not None
        # 根据打包的权重和原始权重的形状获取卷积参数
        args = self.get_conv_pool_args_2d_from_pack(
            raw_weight.shape[2:4], packed_config
        )

        # 断言原始权重的量化方案为每张量仿射量化
        assert raw_weight.qscheme() == torch.per_tensor_affine
        # 如果原始权重的数据类型为uint8，则使用该权重
        if raw_weight.dtype == torch.quint8:
            unsigned_weight = raw_weight
        else:
            # 断言原始权重的数据类型为qint8
            assert raw_weight.dtype == torch.qint8
            # 将原始权重转换为每张量量化的张量
            unsigned_weight = torch._make_per_tensor_quantized_tensor(
                (raw_weight.int_repr().int() + 128).to(torch.uint8),
                scale=raw_weight.q_scale(),
                zero_point=raw_weight.q_zero_point() + 128,
            )
        # 获取权重的量化比例
        weight_scale = unsigned_weight.q_scale()
        # 获取图像操作数和比例的张量操作
        _, image_oper = self.get_tensor_operand_by_jitval(jit_image)
        # 计算偏置的比例
        bias_scale = image_oper.scale * weight_scale
        # 对原始偏置进行每张量量化
        int_bias = torch.quantize_per_tensor(raw_bias, bias_scale, 0, torch.qint32)
        # 为权重添加张量操作数
        bias_id = self.add_tensor_operand_for_weight(int_bias)

        # 计算乘数，用于量化卷积
        multiplier = image_oper.scale * weight_scale / out_scale
        # 断言乘数大于0
        assert multiplier > 0
        # 如果乘数大于等于1，则引发异常
        if multiplier >= 1:
            raise Exception(  # noqa: TRY002
                "Quantized convolution multiplier is greater than 1.  "
                "This is supported by NNAPI, but not by most hardware backends.  "
                "Try training a model without quantization-aware training.  "
            )

        # 调用通用卷积层添加方法，并返回结果
        return self.add_conv2d_common(
            node.outputsAt(0),
            out_scale,
            out_zero_point,
            jit_image,
            unsigned_weight,
            bias_id,
            args,
            transpose,
            fuse_code,
        )
    # 处理卷积和池化操作的灵活输入，根据输入参数确定输出形状
    def _handle_conv_pool_flexible_input(self, out_id, jit_image, args, transpose):
        # 获取图像操作数的 ID 和操作
        image_id, image_oper = self.get_tensor_operand_by_jitval(jit_image)
        # 获取图像的批次数、输入通道数、高度和宽度
        batch, in_ch, in_h, in_w = image_oper.shape

        # 如果批次数为0，则将输出形状设置为0
        if batch == 0:
            self.forward_operand_shape(out_id, 0, image_id, 0)
        # 如果输入通道数为0，则抛出异常
        if in_ch == 0:
            raise Exception("Input channels can't be flexible")  # noqa: TRY002
        # 处理高度和宽度
        if transpose:
            # 如果高度为0，则根据公式计算输出形状
            if in_h == 0:
                self.compute_operand_shape(
                    out_id,
                    2,
                    f"({flex_name(image_id, 2)} - 1) * {args.stride_h} + {args.kernel_h} - {args.pad_t} - {args.pad_b}",
                )
            # 如果宽度为0，则根据公式计算输出形状
            if in_w == 0:
                self.compute_operand_shape(
                    out_id,
                    3,
                    f"({flex_name(image_id, 3)} - 1) * {args.stride_w} + {args.kernel_w} - {args.pad_l} - {args.pad_r}",
                )
        else:
            # 如果高度为0，则根据公式计算输出形状
            if in_h == 0:
                self.compute_operand_shape(
                    out_id,
                    2,
                    f"({flex_name(image_id, 2)} - {args.kernel_h} + {args.pad_t} + {args.pad_b}) // {args.stride_h} + 1",
                )
            # 如果宽度为0，则根据公式计算输出形状
            if in_w == 0:
                self.compute_operand_shape(
                    out_id,
                    3,
                    f"({flex_name(image_id, 3)} - {args.kernel_w} + {args.pad_l} + {args.pad_r}) // {args.stride_w} + 1",
                )
def serialize_model(
    module, inputs, *, config=None, return_shapes=None, use_int16_for_qint16=False
):
    """Convert to NNAPI and serialize torchscript module.

    Parameters:
        module: Torchscript module to convert
        inputs: Tensors used to specify input details for NNAPI
        config (optional): Optional config to attach to module
        return_shapes (optional): Specify shape of outputs if
            your module uses runtime flexible shapes to set output
            buffer size for NNAPI
        use_int16_for_qint16 (optional): Use Pytorch int16 to represent NNAPI qint16 values
    """
    # 创建一个 _NnapiSerializer 的实例，用于序列化模型为 NNAPI 格式
    return _NnapiSerializer(config, use_int16_for_qint16).serialize_model(
        module, inputs, return_shapes
    )
```