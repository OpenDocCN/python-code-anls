# `.\pytorch\test\export\testing.py`

```py
# 导入 functools 模块，用于高阶函数（higher-order functions）操作
import functools
# 导入 unittest 模块，用于编写和运行单元测试
import unittest
# 从 unittest.mock 模块中导入 patch 函数，用于模拟对象的行为
from unittest.mock import patch

# 导入 PyTorch 库
import torch

# 获取 PyTorch aten 命名空间，表示 PyTorch 的底层运算函数
aten = torch.ops.aten

# 以下列表列出了一些复合操作，这些操作仅供测试时保留，不一定是完整的列表
# 这些操作涉及到 aten 命名空间中的各种数学和张量操作
_COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY = [
    aten.arctan2.default,                   # arctan2 函数的默认版本
    aten.divide.Tensor,                     # 张量除法操作
    aten.divide.Scalar,                     # 标量除法操作
    aten.divide.Tensor_mode,                # 带模式的张量除法
    aten.divide.Scalar_mode,                # 带模式的标量除法
    aten.multiply.Tensor,                   # 张量乘法
    aten.multiply.Scalar,                   # 标量乘法
    aten.subtract.Tensor,                   # 张量减法
    aten.subtract.Scalar,                   # 标量减法
    aten.true_divide.Tensor,                # 真除法操作
    aten.true_divide.Scalar,                # 真除法操作（标量）
    aten.greater.Tensor,                    # 张量比较操作（大于）
    aten.greater.Scalar,                    # 标量比较操作（大于）
    aten.greater_equal.Tensor,              # 张量比较操作（大于等于）
    aten.greater_equal.Scalar,              # 标量比较操作（大于等于）
    aten.less_equal.Tensor,                 # 张量比较操作（小于等于）
    aten.less_equal.Scalar,                 # 标量比较操作（小于等于）
    aten.less.Tensor,                       # 张量比较操作（小于）
    aten.less.Scalar,                       # 标量比较操作（小于）
    aten.not_equal.Tensor,                  # 张量比较操作（不等于）
    aten.not_equal.Scalar,                  # 标量比较操作（不等于）
    aten.cat.names,                         # 按名称连接张量序列
    aten.sum.dim_DimnameList,               # 沿指定维度求和
    aten.mean.names_dim,                    # 沿指定维度计算均值
    aten.prod.dim_Dimname,                  # 沿指定维度计算乘积
    aten.all.dimname,                       # 沿指定维度检查是否所有元素为真
    aten.norm.names_ScalarOpt_dim,          # 计算张量的范数
    aten.norm.names_ScalarOpt_dim_dtype,    # 指定数据类型计算张量的范数
    aten.var.default,                       # 计算张量的方差
    aten.var.dim,                           # 沿指定维度计算张量的方差
    aten.var.names_dim,                     # 按名称计算张量的方差
    aten.var.correction_names,              # 按名称计算张量的方差（考虑校正）
    aten.std.default,                       # 计算张量的标准差
    aten.std.dim,                           # 沿指定维度计算张量的标准差
    aten.std.names_dim,                     # 按名称计算张量的标准差
    aten.std.correction_names,              # 按名称计算张量的标准差（考虑校正）
    aten.absolute.default,                  # 计算张量的绝对值
    aten.arccos.default,                    # 计算反余弦值
    aten.arccosh.default,                   # 计算反双曲余弦值
    aten.arcsin.default,                    # 计算反正弦值
    aten.arcsinh.default,                   # 计算反双曲正弦值
    aten.arctan.default,                    # 计算反正切值
    aten.arctanh.default,                   # 计算反双曲正切值
    aten.clip.default,                      # 将张量值裁剪到指定范围内
    aten.clip.Tensor,                       # 将张量值裁剪到指定范围内
    aten.fix.default,                       # 计算张量的截断值
    aten.negative.default,                  # 计算张量的负值
    aten.square.default,                    # 计算张量的平方
    aten.size.int,                          # 获取张量的大小
    aten.size.Dimname,                      # 按名称获取张量的大小
    aten.stride.int,                        # 获取张量的步幅
    aten.stride.Dimname,                    # 按名称获取张量的步幅
    aten.repeat_interleave.self_Tensor,     # 按指定的张量重复插入元素
    aten.repeat_interleave.self_int,        # 按指定的整数重复插入元素
    aten.sym_size.int,                      # 获取张量的符号化大小
    aten.sym_stride.int,                    # 获取张量的符号化步幅
    aten.atleast_1d.Sequence,               # 将输入转换为至少具有一维的张量
    aten.atleast_2d.Sequence,               # 将输入转换为至少具有二维的张量
    aten.atleast_3d.Sequence,               # 将输入转换为至少具有三维的张量
    aten.linear.default,                    # 执行线性变换
    aten.conv2d.default,                    # 执行二维卷积操作
    aten.conv2d.padding,                    # 执行带填充的二维卷积操作
    aten.mish_backward.default,             # Mish 激活函数的反向传播
    aten.silu_backward.default,             # SiLU（Sigmoid Linear Unit）激活函数的反向传播
    aten.index_add.dimname,                 # 按名称对张量进行索引添加操作
    aten.pad_sequence.default,              # 对序列进行填充操作
    aten.index_copy.dimname,                # 按名称对张量进行索引复制操作
    aten.upsample_nearest1d.vec,            # 使用最近邻插值进行一维上采样
    aten.upsample_nearest2d.vec,            # 使用最近邻插值进行二维上采样
    aten.upsample_nearest3d.vec,            # 使用最近邻插值进行三维上采样
    aten._upsample_nearest_exact1d.vec,     # 使用最近邻插值进行精确一维上采样
    aten._upsample_nearest_exact2d.vec,     # 使用最近邻插值进行精确二维上采样
    aten._upsample_nearest_exact3d.vec,     # 使用最近邻插值进行精确三维上采样
    aten.rnn_tanh.input,                    # RNN tanh 激活函数的输入
    aten.rnn_tanh.data,                     # RNN tanh 激活函数的数据
    aten.rnn_relu.input,                    # RNN relu 激活函数的输入
    aten.rnn_relu.data,                     # RNN relu 激活函数的数据
    aten.lstm.input,                        # LSTM 网络的输入
    aten.lstm.data,                         # LSTM 网络的数据
    aten.gru.input,                         # GRU 网络的输入
    aten.gru.data,                          # GRU 网络的数据
    aten._upsample_bilinear2d_aa.vec,       # 使用双线性插值进行二维上
    # 返回默认值的 where 操作
    aten.where.default,
    # 返回默认值的 item 操作
    aten.item.default,
    # 在指定维度上进行 any 操作，使用维度名称
    aten.any.dimname,
    # 返回默认值的标准差和均值操作
    aten.std_mean.default,
    # 在指定维度上返回标准差和均值操作
    aten.std_mean.dim,
    # 在指定维度和名称上返回标准差和均值操作
    aten.std_mean.names_dim,
    # 在指定名称上进行标准差和均值操作
    aten.std_mean.correction_names,
    # 返回默认值的方差和均值操作
    aten.var_mean.default,
    # 在指定维度上返回方差和均值操作
    aten.var_mean.dim,
    # 在指定维度和名称上返回方差和均值操作
    aten.var_mean.names_dim,
    # 在指定名称上进行方差和均值操作
    aten.var_mean.correction_names,
    # 对默认张量进行广播操作
    aten.broadcast_tensors.default,
    # 返回默认值的短时傅里叶变换操作
    aten.stft.default,
    # 在进行短时傅里叶变换时进行中心化处理
    aten.stft.center,
    # 返回默认值的逆短时傅里叶变换操作
    aten.istft.default,
    # 使用维度名称和标量进行索引填充操作
    aten.index_fill.Dimname_Scalar,
    # 使用维度名称和张量进行索引填充操作
    aten.index_fill.Dimname_Tensor,
    # 在指定维度上进行索引选择操作
    aten.index_select.dimname,
    # 返回默认值的对角线操作
    aten.diag.default,
    # 在指定维度上进行累积和操作
    aten.cumsum.dimname,
    # 在指定维度上进行累积乘积操作
    aten.cumprod.dimname,
    # 返回默认值的网格生成操作
    aten.meshgrid.default,
    # 在网格生成时指定索引方式
    aten.meshgrid.indexing,
    # 返回默认值的快速傅里叶变换操作
    aten.fft_fft.default,
    # 返回默认值的逆快速傅里叶变换操作
    aten.fft_ifft.default,
    # 返回默认值的实数快速傅里叶变换操作
    aten.fft_rfft.default,
    # 返回默认值的逆实数快速傅里叶变换操作
    aten.fft_irfft.default,
    # 返回默认值的埃尔米特快速傅里叶变换操作
    aten.fft_hfft.default,
    # 返回默认值的逆埃尔米特快速傅里叶变换操作
    aten.fft_ihfft.default,
    # 返回默认值的多维快速傅里叶变换操作
    aten.fft_fftn.default,
    # 返回默认值的多维逆快速傅里叶变换操作
    aten.fft_ifftn.default,
    # 返回默认值的多维实数快速傅里叶变换操作
    aten.fft_rfftn.default,
    # 返回默认值的多维逆实数快速傅里叶变换操作
    aten.fft_ihfftn.default,
    # 返回默认值的多维埃尔米特快速傅里叶变换操作
    aten.fft_irfftn.default,
    # 返回默认值的多维逆埃尔米特快速傅里叶变换操作
    aten.fft_hfftn.default,
    # 返回默认值的二维快速傅里叶变换操作
    aten.fft_fft2.default,
    # 返回默认值的二维逆快速傅里叶变换操作
    aten.fft_ifft2.default,
    # 返回默认值的二维实数快速傅里叶变换操作
    aten.fft_rfft2.default,
    # 返回默认值的二维逆实数快速傅里叶变换操作
    aten.fft_irfft2.default,
    # 返回默认值的二维埃尔米特快速傅里叶变换操作
    aten.fft_hfft2.default,
    # 返回默认值的二维逆埃尔米特快速傅里叶变换操作
    aten.fft_ihfft2.default,
    # 返回默认值的快速傅里叶变换移位操作
    aten.fft_fftshift.default,
    # 返回默认值的逆快速傅里叶变换移位操作
    aten.fft_ifftshift.default,
    # 返回默认值的 SELU 激活函数操作
    aten.selu.default,
    # 返回默认值的边际排名损失函数操作
    aten.margin_ranking_loss.default,
    # 返回默认值的铰链嵌入损失函数操作
    aten.hinge_embedding_loss.default,
    # 返回默认值的负对数似然损失函数操作
    aten.nll_loss.default,
    # 返回默认值的参数化 ReLU 激活函数操作
    aten.prelu.default,
    # 返回默认值的 ReLU6 激活函数操作
    aten.relu6.default,
    # 返回默认值的两两距离计算操作
    aten.pairwise_distance.default,
    # 返回默认值的批次间距离计算操作
    aten.pdist.default,
    # 返回默认值的高斯累积分布函数操作
    aten.special_ndtr.default,
    # 在指定维度上进行累积最大值操作
    aten.cummax.dimname,
    # 在指定维度上进行累积最小值操作
    aten.cummin.dimname,
    # 在指定维度上进行对数累积和指数操作
    aten.logcumsumexp.dimname,
    # 返回其它张量的最大值操作
    aten.max.other,
    # 在指定维度和名称上返回最大值操作
    aten.max.names_dim,
    # 返回其它张量的最小值操作
    aten.min.other,
    # 在指定维度和名称上返回最小值操作
    aten.min.names_dim,
    # 返回默认值的特征值操作
    aten.linalg_eigvals.default,
    # 在指定维度和名称上返回中位数操作
    aten.median.names_dim,
    # 在指定维度和名称上返回非 NaN 值的中位数操作
    aten.nanmedian.names_dim,
    # 在指定维度上返回众数操作
    aten.mode.dimname,
    # 在指定维度上进行 gather 操作
    aten.gather.dimname,
    # 在指定维度上进行排序操作
    aten.sort.dimname,
    # 在指定维度上进行稳定排序操作
    aten.sort.dimname_stable,
    # 返回默认值的排序索引操作
    aten.argsort.default,
    # 在指定维度上返回排序索引操作
    aten.argsort.dimname,
    # 返回默认值的随机 ReLU 激活函数操作
    aten.rrelu.default,
    # 返回默认值的一维卷积转置操作
    aten.conv_transpose1d.default,
    # 在输入上进行一维卷积转置操作
    aten.conv_transpose2d.input,
    # 在输入上进行三维卷积转置操作
    aten.conv_transpose3d.input,
    # 返回默认值的一维卷积操作
    aten.conv1d.default,
    # 在一维卷积时进行填充操作
    aten.conv1d.padding,
    # 返回默认值的三维卷积操作
    aten.conv3d.default,
    # 在三维卷积时进行填充操作
    aten.conv3d.padding,
    # 返回张量与张量或标量的幂运算结果
# 定义函数，根据给定参数生成带有模拟导出功能的测试类
def make_test_cls_with_mocked_export(
    cls, cls_prefix, fn_suffix, mocked_export_fn, xfail_prop=None
):
    # 使用 type 动态创建新类，名称由 cls_prefix 和原类名组成
    MockedTestClass = type(f"{cls_prefix}{cls.__name__}", cls.__bases__, {})
    # 设置新类的限定名称与类名相同
    MockedTestClass.__qualname__ = MockedTestClass.__name__

    # 遍历原类的所有属性名
    for name in dir(cls):
        # 对于以 "test_" 开头的属性名
        if name.startswith("test_"):
            # 获取属性对应的函数
            fn = getattr(cls, name)
            # 如果该属性不是可调用的，直接将其赋给新类
            if not callable(fn):
                setattr(MockedTestClass, name, getattr(cls, name))
                continue
            # 构造新的函数名，加上指定的后缀
            new_name = f"{name}{fn_suffix}"
            # 创建一个新函数，该函数在调用原函数时模拟导出功能
            new_fn = _make_fn_with_mocked_export(fn, mocked_export_fn)
            new_fn.__name__ = new_name
            # 如果设置了 xfail_prop 并且原函数具有该属性，则标记新函数为预期失败
            if xfail_prop is not None and hasattr(fn, xfail_prop):
                new_fn = unittest.expectedFailure(new_fn)
            # 将新函数设置为新类的属性
            setattr(MockedTestClass, new_name, new_fn)
        # 处理其他属性（非以 "test_" 开头），将其直接赋给新类
        elif not hasattr(MockedTestClass, name):
            setattr(MockedTestClass, name, getattr(cls, name))

    # 返回创建的新类
    return MockedTestClass


# 定义一个内部函数，生成一个包装了模拟导出功能的新函数
def _make_fn_with_mocked_export(fn, mocked_export_fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        # 尝试导入 test_export 模块，如果失败则使用当前路径下的 test_export
        try:
            from . import test_export
        except ImportError:
            import test_export

        # 使用 patch 方法替换 test_export.export 函数为 mocked_export_fn
        with patch(f"{test_export.__name__}.export", mocked_export_fn):
            # 调用原函数并返回其结果
            return fn(*args, **kwargs)

    # 返回生成的新函数
    return _fn


# 控制在 test/export/test_export_training_ir_to_run_decomp.py 中生成的预期失败测试
def expectedFailureTrainingIRToRunDecomp(fn):
    # 将测试函数标记为在训练 IR 转运行解组中预期失败
    fn._expected_failure_training_ir_to_run_decomp = True
    return fn


# 控制在 test/export/test_export_nonstrict.py 中生成的预期失败测试
def expectedFailureNonStrict(fn):
    # 将测试函数标记为在非严格模式中预期失败
    fn._expected_failure_non_strict = True
    return fn


# 控制在 test/export/test_retraceability.py 中生成的预期失败测试
def expectedFailureRetraceability(fn):
    # 将测试函数标记为在追溯性测试中预期失败
    fn._expected_failure_retrace = True
    return fn


# 控制在 test/export/test_serdes.py 中生成的预期失败测试
def expectedFailureSerDer(fn):
    # 将测试函数标记为在序列化/反序列化测试中预期失败
    fn._expected_failure_serdes = True
    return fn


# 控制在 test/export/test_serdes.py 中生成的预期失败测试（预调度前解组）
def expectedFailureSerDerPreDispatch(fn):
    # 将测试函数标记为在序列化/反序列化测试中预期失败（预调度前解组）
    fn._expected_failure_serdes_pre_dispatch = True
    return fn


# 控制在预调度运行解组测试中生成的预期失败测试
def expectedFailurePreDispatchRunDecomp(fn):
    # 将测试函数标记为在预调度运行解组中预期失败
    fn._expected_failure_pre_dispatch = True
    return fn
```