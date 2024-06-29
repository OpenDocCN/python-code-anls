# `.\numpy\benchmarks\benchmarks\bench_ufunc_strides.py`

```
# 从 common 模块导入 Benchmark 类和 get_data 函数
from .common import Benchmark, get_data

# 导入 numpy 库并将其命名为 np
import numpy as np

# 从 numpy 的内部 umath 模块中筛选出所有的 ufunc 对象并存储在 UFUNCS 列表中
UFUNCS = [obj for obj in np._core.umath.__dict__.values() if
          isinstance(obj, np.ufunc)]

# 筛选出所有具有 "O->O" 类型签名的一元 ufunc 对象并存储在 UFUNCS_UNARY 列表中
UFUNCS_UNARY = [uf for uf in UFUNCS if "O->O" in uf.types]

# 创建一个 Benchmark 的子类 _AbstractBinary
class _AbstractBinary(Benchmark):
    # 参数列表为空
    params = []
    # 参数名称列表，包括 'ufunc', 'stride_in0', 'stride_in1', 'stride_out', 'dtype'
    param_names = ['ufunc', 'stride_in0', 'stride_in1', 'stride_out', 'dtype']
    # 设置超时时间为 10 秒
    timeout = 10
    # 数组长度为 10000
    arrlen = 10000
    # 数据是有限的
    data_finite = True
    # 数据中没有非规范数
    data_denormal = False
    # 数据中没有零值
    data_zeros = False

    # 设置初始化方法，接受 ufunc、stride_in0、stride_in1、stride_out 和 dtype 参数
    def setup(self, ufunc, stride_in0, stride_in1, stride_out, dtype):
        # 构建 ufunc 的输入签名
        ufunc_insig = f'{dtype}{dtype}->'

        # 如果 ufunc 的输入签名不在 ufunc 的类型列表中
        if ufunc_insig + dtype not in ufunc.types:
            # 在可能的签名中查找匹配的签名
            for st_sig in (ufunc_insig, dtype):
                test = [sig for sig in ufunc.types if sig.startswith(st_sig)]
                if test:
                    break
            # 如果没有找到匹配的签名，则抛出 NotImplementedError 异常
            if not test:
                raise NotImplementedError(
                    f"Ufunc {ufunc} doesn't support "
                    f"binary input of dtype {dtype}"
                ) from None
            # 将找到的输入和输出类型分割出来
            tin, tout = test[0].split('->')
        else:
            # 否则，直接使用指定的输入和输出类型
            tin = dtype + dtype
            tout = dtype

        # 初始化 ufunc_args 列表
        self.ufunc_args = []
        
        # 对于每个输入类型和对应的步长，生成测试数据并添加到 ufunc_args 中
        for i, (dt, stride) in enumerate(zip(tin, (stride_in0, stride_in1))):
            self.ufunc_args += [get_data(
                self.arrlen * stride, dt, i,
                zeros=self.data_zeros,
                finite=self.data_finite,
                denormal=self.data_denormal,
            )[::stride]]
        
        # 对于每个输出类型，生成空数组并添加到 ufunc_args 中
        for dt in tout:
            self.ufunc_args += [
                np.empty(stride_out * self.arrlen, dt)[::stride_out]
            ]

        # 设置忽略所有的 numpy 错误
        np.seterr(all='ignore')

    # 定义执行二元操作的方法 time_binary
    def time_binary(self, ufunc, stride_in0, stride_in1, stride_out,
                    dtype):
        ufunc(*self.ufunc_args)

    # 定义执行带有第一个标量输入的二元操作的方法 time_binary_scalar_in0
    def time_binary_scalar_in0(self, ufunc, stride_in0, stride_in1,
                               stride_out, dtype):
        ufunc(self.ufunc_args[0][0], *self.ufunc_args[1:])

    # 定义执行带有第二个标量输入的二元操作的方法 time_binary_scalar_in1
    def time_binary_scalar_in1(self, ufunc, stride_in0, stride_in1,
                               stride_out, dtype):
        ufunc(self.ufunc_args[0], self.ufunc_args[1][0], *self.ufunc_args[2:])

# 创建一个 Benchmark 的子类 _AbstractUnary
class _AbstractUnary(Benchmark):
    # 参数列表为空
    params = []
    # 参数名称列表，包括 'ufunc', 'stride_in', 'stride_out', 'dtype'
    param_names = ['ufunc', 'stride_in', 'stride_out', 'dtype']
    # 设置超时时间为 10 秒
    timeout = 10
    # 数组长度为 10000
    arrlen = 10000
    # 数据是有限的
    data_finite = True
    # 数据中没有非规范数
    data_denormal = False
    # 数据中没有零值
    data_zeros = False
    # 设置对象的初始状态，准备用于运算的输入数据和参数
    def setup(self, ufunc, stride_in, stride_out, dtype):
        # 调用函数获取特定条件下的输入数据数组
        arr_in = get_data(
            stride_in*self.arrlen, dtype,
            zeros=self.data_zeros,
            finite=self.data_finite,
            denormal=self.data_denormal,
        )
        # 将输入数据的切片作为ufunc的参数列表中的第一个参数
        self.ufunc_args = [arr_in[::stride_in]]

        # 构建ufunc的输入签名
        ufunc_insig = f'{dtype}->'
        # 检查ufunc是否支持指定的输入数据类型
        if ufunc_insig+dtype not in ufunc.types:
            # 如果不支持，尝试查找兼容的输入类型
            test = [sig for sig in ufunc.types if sig.startswith(ufunc_insig)]
            if not test:
                # 如果找不到兼容的输入类型，抛出未实现错误
                raise NotImplementedError(
                    f"Ufunc {ufunc} doesn't support "
                    f"unary input of dtype {dtype}"
                ) from None
            # 获取找到的第一个兼容类型的输出类型
            tout = test[0].split('->')[1]
        else:
            # 如果支持指定的输入类型，则直接使用该类型作为输出类型
            tout = dtype

        # 为每个输出类型创建空数组，并作为ufunc的参数列表中的一部分
        for dt in tout:
            self.ufunc_args += [
                np.empty(stride_out*self.arrlen, dt)[::stride_out]
            ]

        # 设置忽略所有数值错误的运算环境
        np.seterr(all='ignore')

    # 执行ufunc的一元运算，使用预设的参数
    def time_unary(self, ufunc, stride_in, stride_out, dtype):
        ufunc(*self.ufunc_args)
class UnaryFP(_AbstractUnary):
    # 定义类属性 params，包含一组列表，用于参数化基本的一元操作
    params = [[uf for uf in UFUNCS_UNARY
                   if uf not in (np.invert, np.bitwise_count)],
              [1, 4],
              [1, 2],
              ['e', 'f', 'd']]

    # 定义设置方法，用于配置一元操作的参数
    def setup(self, ufunc, stride_in, stride_out, dtype):
        # 调用父类的设置方法，进行基本的一元操作的设置
        _AbstractUnary.setup(self, ufunc, stride_in, stride_out, dtype)
        # 如果操作函数名称为 'arccosh'，则将参数列表的第一个元素加上 1.0
        if (ufunc.__name__ == 'arccosh'):
            self.ufunc_args[0] += 1.0

class UnaryFPSpecial(UnaryFP):
    # 设置数据属性，标识数据不是有限的、可能有非规格化数、可能有零值
    data_finite = False
    data_denormal = True
    data_zeros = True

class BinaryFP(_AbstractBinary):
    # 定义类属性 params，包含一组列表，用于参数化基本的二元浮点操作
    params = [
        [np.maximum, np.minimum, np.fmax, np.fmin, np.ldexp],
        [1, 2], [1, 4], [1, 2, 4], ['f', 'd']
    ]

class BinaryFPSpecial(BinaryFP):
    # 设置数据属性，标识数据不是有限的、可能有非规格化数、可能有零值
    data_finite = False
    data_denormal = True
    data_zeros = True

class BinaryComplex(_AbstractBinary):
    # 定义类属性 params，包含一组列表，用于参数化基本的二元复数操作
    params = [
        [np.add, np.subtract, np.multiply, np.divide],
        [1, 2, 4], [1, 2, 4], [1, 2, 4],
        ['F', 'D']
    ]

class UnaryComplex(_AbstractUnary):
    # 定义类属性 params，包含一组列表，用于参数化基本的一元复数操作
    params = [
        [np.reciprocal, np.absolute, np.square, np.conjugate],
        [1, 2, 4], [1, 2, 4], ['F', 'D']
    ]

class BinaryInt(_AbstractBinary):
    # 设置数组长度为 100000
    arrlen = 100000
    # 定义类属性 params，包含一组列表，用于参数化基本的二元整数操作
    params = [
        [np.maximum, np.minimum],
        [1, 2], [1, 2], [1, 2],
        ['b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q']
    ]

class BinaryIntContig(_AbstractBinary):
    # 定义类属性 params，包含一组列表，用于参数化基本的连续的二元整数操作
    params = [
        [getattr(np, uf) for uf in (
            'add', 'subtract', 'multiply', 'bitwise_and', 'bitwise_or',
            'bitwise_xor', 'logical_and', 'logical_or', 'logical_xor',
            'right_shift', 'left_shift',
        )],
        [1], [1], [1],
        ['b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q']
    ]

class UnaryIntContig(_AbstractUnary):
    # 设置数组长度为 100000
    arrlen = 100000
    # 定义类属性 params，包含一组列表，用于参数化基本的连续的一元整数操作
    params = [
        [getattr(np, uf) for uf in (
            'positive', 'square', 'reciprocal', 'conjugate', 'logical_not',
            'invert', 'isnan', 'isinf', 'isfinite',
            'absolute', 'sign', 'bitwise_count'
        )],
        [1], [1],
        ['b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q']
    ]

class Mandelbrot(Benchmark):
    # 定义方法 f，用于计算绝对值是否小于 4.0
    def f(self,z):
        return np.abs(z) < 4.0

    # 定义方法 g，用于计算 z*z + c 的和
    def g(self,z,c):
        return np.sum(np.multiply(z, z) + c)

    # 定义方法 mandelbrot_numpy，用于计算 Mandelbrot 集合的数据
    def mandelbrot_numpy(self, c, maxiter):
        output = np.zeros(c.shape, np.int32)
        z = np.empty(c.shape, np.complex64)
        for it in range(maxiter):
            notdone = self.f(z)
            output[notdone] = it
            z[notdone] = self.g(z[notdone],c[notdone])
        output[output == maxiter-1] = 0
        return output

    # 定义方法 mandelbrot_set，用于生成 Mandelbrot 集合的数据
    def mandelbrot_set(self,xmin,xmax,ymin,ymax,width,height,maxiter):
        r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
        r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
        c = r1 + r2[:,None]*1j
        n3 = self.mandelbrot_numpy(c,maxiter)
        return (r1,r2,n3.T)

    # 定义方法 time_mandel，用于测试 Mandelbrot 集合的计算时间
    def time_mandel(self):
        self.mandelbrot_set(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)
# LogisticRegression 类继承自 Benchmark 类，用于逻辑回归模型的训练和性能评估
class LogisticRegression(Benchmark):
    # 参数名称列表
    param_names = ['dtype']
    # 参数取值列表，包括 np.float32 和 np.float64
    params = [np.float32, np.float64]

    # 设置超时时间为 1000 毫秒
    timeout = 1000

    # 训练逻辑回归模型的方法，接受最大训练周期 max_epoch 作为参数
    def train(self, max_epoch):
        # 迭代训练 max_epoch 次
        for epoch in range(max_epoch):
            # 计算 z = X_train * W
            z = np.matmul(self.X_train, self.W)
            # 计算 A = sigmoid(z)
            A = 1 / (1 + np.exp(-z))
            # 计算交叉熵损失函数
            loss = -np.mean(self.Y_train * np.log(A) + (1 - self.Y_train) * np.log(1 - A))
            # 计算 dz = A - Y_train
            dz = A - self.Y_train
            # 计算 dw = (1/size) * X_train^T * dz
            dw = (1 / self.size) * np.matmul(self.X_train.T, dz)
            # 更新权重 W
            self.W = self.W - self.alpha * dw

    # 设置方法，初始化训练数据和模型参数
    def setup(self, dtype):
        # 设置随机种子为 42，以便结果可重复
        np.random.seed(42)
        # 训练集大小为 250
        self.size = 250
        # 特征数为 16
        features = 16
        # 生成随机训练数据 X_train，并将其类型转换为指定的 dtype
        self.X_train = np.random.rand(self.size, features).astype(dtype)
        # 生成随机标签数据 Y_train，并将其类型转换为指定的 dtype
        self.Y_train = np.random.choice(2, self.size).astype(dtype)
        # 初始化权重 W，全零数组，形状为 (features, 1)，类型为 dtype
        self.W = np.zeros((features, 1), dtype=dtype)
        # 初始化偏置 b，全零数组，形状为 (1, 1)，类型为 dtype
        self.b = np.zeros((1, 1), dtype=dtype)
        # 学习率设为 0.1
        self.alpha = 0.1

    # 性能评估方法，调用 train 方法训练模型 1000 次
    def time_train(self, dtype):
        self.train(1000)
```