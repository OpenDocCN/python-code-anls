# `D:\src\scipysrc\scikit-learn\sklearn\_loss\_loss.pxd`

```
# 定义一个融合类型，用于表示像 y_true、raw_prediction、sample_weights 这样的输入类型。
ctypedef fused floating_in:
    double  # 双精度浮点数
    float   # 单精度浮点数


# 定义一个融合类型，用于表示像梯度和黑塞矩阵这样的输出类型。
# 我们使用不同的融合类型来区分输入（floating_in）和输出（floating_out），使得在同一函数调用中输入和输出可以有不同的数据类型。
# 单个融合类型只能在一个函数调用中的所有参数中采用单一的数值类型。
ctypedef fused floating_out:
    double  # 双精度浮点数
    float   # 单精度浮点数


# 定义一个结构体，用于返回两个双精度浮点数。
ctypedef struct double_pair:
    double val1  # 第一个双精度浮点数
    double val2  # 第二个双精度浮点数


# CyLossFunction 的 C 基类，用于损失函数
cdef class CyLossFunction:
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil  # 计算损失函数值
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil  # 计算损失函数的梯度
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil  # 计算损失函数的梯度和黑塞矩阵


# CyHalfSquaredError 类，继承自 CyLossFunction，半平方误差损失函数
cdef class CyHalfSquaredError(CyLossFunction):
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil  # 计算半平方误差损失函数值
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil  # 计算半平方误差损失函数的梯度
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil  # 计算半平方误差损失函数的梯度和黑塞矩阵


# CyAbsoluteError 类，继承自 CyLossFunction，绝对误差损失函数
cdef class CyAbsoluteError(CyLossFunction):
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil  # 计算绝对误差损失函数值
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil  # 计算绝对误差损失函数的梯度
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil  # 计算绝对误差损失函数的梯度和黑塞矩阵


# CyPinballLoss 类，继承自 CyLossFunction，钉球损失函数
cdef class CyPinballLoss(CyLossFunction):
    cdef readonly double quantile  # readonly 使得 quantile 可以从 Python 访问，表示损失函数的分位数
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil  # 计算钉球损失函数值
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil  # 计算钉球损失函数的梯度
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil  # 计算钉球损失函数的梯度和黑塞矩阵


# CyHuberLoss 类，继承自 CyLossFunction，Huber 损失函数
cdef class CyHuberLoss(CyLossFunction):
    cdef public double delta  # public 使得 delta 可以从 Python 访问，表示损失函数的 delta 参数
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil  # 计算 Huber 损失函数值
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil  # 计算 Huber 损失函数的梯度
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil  # 计算 Huber 损失函数的梯度和黑塞矩阵


# CyHalfPoissonLoss 类，继承自 CyLossFunction，半 Poisson 损失函数
cdef class CyHalfPoissonLoss(CyLossFunction):
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil  # 计算半 Poisson 损失函数值
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil  # 计算半 Poisson 损失函数的梯度
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil  # 计算半 Poisson 损失函数的梯度和黑塞矩阵


# CyHalfGammaLoss 类，继承自 CyLossFunction，半 Gamma 损失函数
cdef class CyHalfGammaLoss(CyLossFunction):
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil  # 计算半 Gamma 损失函数值
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil  # 计算半 Gamma 损失函数的梯度
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil  # 计算半 Gamma 损失函数的梯度和黑塞矩阵


# CyHalfTweedieLoss 类，继承自 CyLossFunction，半 Tweedie 损失函数
cdef class CyHalfTweedieLoss(CyLossFunction):
    # 声明一个Cython变量，类型为double，带有readonly修饰符，使其可从Python中访问
    cdef readonly double power

    # 声明一个Cython函数cy_loss，接受两个double类型参数y_true和raw_prediction，
    # 使用noexcept声明不抛出异常，使用nogil声明释放GIL(Global Interpreter Lock)以支持并行执行
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil

    # 声明一个Cython函数cy_gradient，接受两个double类型参数y_true和raw_prediction，
    # 使用noexcept声明不抛出异常，使用nogil声明释放GIL以支持并行执行
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil

    # 声明一个Cython函数cy_grad_hess，接受两个double类型参数y_true和raw_prediction，
    # 返回一个double_pair类型，使用noexcept声明不抛出异常，使用nogil声明释放GIL以支持并行执行
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
# 定义 CyHalfTweedieLossIdentity 类，继承自 CyLossFunction 类
cdef class CyHalfTweedieLossIdentity(CyLossFunction):
    # 声明一个只读的 double 类型成员变量 power
    cdef readonly double power  # readonly makes it accessible from Python
    
    # 定义 cy_loss 方法，计算半 Tweedie 损失函数的损失值
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    
    # 定义 cy_gradient 方法，计算半 Tweedie 损失函数的梯度值
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    
    # 定义 cy_grad_hess 方法，计算半 Tweedie 损失函数的梯度和 Hessian 矩阵
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil


# 定义 CyHalfBinomialLoss 类，继承自 CyLossFunction 类
cdef class CyHalfBinomialLoss(CyLossFunction):
    # 定义 cy_loss 方法，计算半 Binomial 损失函数的损失值
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    
    # 定义 cy_gradient 方法，计算半 Binomial 损失函数的梯度值
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    
    # 定义 cy_grad_hess 方法，计算半 Binomial 损失函数的梯度和 Hessian 矩阵
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil


# 定义 CyExponentialLoss 类，继承自 CyLossFunction 类
cdef class CyExponentialLoss(CyLossFunction):
    # 定义 cy_loss 方法，计算指数损失函数的损失值
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    
    # 定义 cy_gradient 方法，计算指数损失函数的梯度值
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    
    # 定义 cy_grad_hess 方法，计算指数损失函数的梯度和 Hessian 矩阵
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
```