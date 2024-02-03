# `.\PaddleOCR\StyleText\arch\spectral_norm.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何担保或条件
# 请查看许可证以获取特定语言的权限和限制
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# 定义一个函数，用于生成服从正态分布的随机数填充到张量 x 中
def normal_(x, mean=0., std=1.):
    # 生成服从正态分布的随机数，填充到与 x 相同形状的张量中
    temp_value = paddle.normal(mean, std, shape=x.shape)
    # 将生成的随机数填充到 x 中
    x.set_value(temp_value)
    return x

# 定义一个类，用于实现谱范数归一化
class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(
                                 n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    # 将权重重塑为矩阵形式
    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # 将指定维度转置到最前面
            weight_mat = weight_mat.transpose([
                self.dim,
                * [d for d in range(weight_mat.dim()) if d != self.dim]
            ])

        height = weight_mat.shape[0]

        return weight_mat.reshape([height, -1])
    # 计算权重矩阵的特征向量，用于权重归一化
    def compute_weight(self, module, do_power_iteration):
        # 获取模块中原始权重
        weight = getattr(module, self.name + '_orig')
        # 获取模块中的特征向量 u
        u = getattr(module, self.name + '_u')
        # 获取模块中的特征向量 v
        v = getattr(module, self.name + '_v')
        # 将权重矩阵重塑为矩阵形式
        weight_mat = self.reshape_weight_to_matrix(weight)

        # 如果需要进行幂迭代
        if do_power_iteration:
            # 在不记录梯度的情况下
            with paddle.no_grad():
                # 进行指定次数的幂迭代
                for _ in range(self.n_power_iterations):
                    # 更新特征向量 v
                    v.set_value(
                        F.normalize(
                            paddle.matmul(
                                weight_mat,
                                u,
                                transpose_x=True,
                                transpose_y=False),
                            axis=0,
                            epsilon=self.eps, ))

                    # 更新特征向量 u
                    u.set_value(
                        F.normalize(
                            paddle.matmul(weight_mat, v),
                            axis=0,
                            epsilon=self.eps, ))
                # 如果进行了幂迭代，则克隆特征向量 u 和 v
                if self.n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()

        # 计算 sigma 值
        sigma = paddle.dot(u, paddle.mv(weight_mat, v))
        # 对权重进行归一化
        weight = weight / sigma
        return weight

    # 从模块中移除权重
    def remove(self, module):
        # 在不记录梯度的情况下，计算权重
        with paddle.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        # 删除模块中的权重及相关属性
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')

        # 将计算得到的权重作为模块的参数添加回去
        module.add_parameter(self.name, weight.detach())

    # 在模块中调用权重计算函数
    def __call__(self, module, inputs):
        # 设置模块中的权重为计算得到的归一化权重
        setattr(
            module,
            self.name,
            self.compute_weight(
                module, do_power_iteration=module.training))

    # 静态方法
    @staticmethod
    # 应用谱归一化到给定模块的指定参数上
    def apply(module, name, n_power_iterations, dim, eps):
        # 遍历模块的前向预处理钩子，检查是否已经存在相同名称的谱归一化钩子
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                # 如果存在相同名称的谱归一化钩子，则抛出运行时错误
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name))

        # 创建谱归一化对象
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        # 获取指定参数的权重
        weight = module._parameters[name]

        # 在不计算梯度的情况下进行以下操作
        with paddle.no_grad():
            # 将权重重塑为矩阵
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.shape

            # 随机初始化向量 u 和 v
            u = module.create_parameter([h])
            u = normal_(u, 0., 1.)
            v = module.create_parameter([w])
            v = normal_(v, 0., 1.)
            # 对 u 和 v 进行归一化处理
            u = F.normalize(u, axis=0, epsilon=fn.eps)
            v = F.normalize(v, axis=0, epsilon=fn.eps)

        # 从模块的参数中删除 fn.name，否则无法设置属性
        del module._parameters[fn.name]
        # 将原始权重添加为模块的参数
        module.add_parameter(fn.name + "_orig", weight)
        # 将权重设置回 fn.name，因为某些操作可能会假定它存在，例如初始化权重时
        # 但是，我们不能直接赋值，因为它可能是一个 Parameter 并被添加为参数。相反，我们将权重 * 1.0 注册为普通属性
        setattr(module, fn.name, weight * 1.0)
        # 注册 u 和 v 为模块的缓冲区
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        # 注册谱归一化对象为模块的前向预处理钩子
        module.register_forward_pre_hook(fn)
        # 返回谱归一化对象
        return fn
# 计算模块的谱范数，用于权重矩阵的规范化
def spectral_norm(module,
                  name='weight',
                  n_power_iterations=1,
                  eps=1e-12,
                  dim=None):
    # 如果未指定维度，则根据模块类型设置默认维度
    if dim is None:
        if isinstance(module, (nn.Conv1DTranspose, nn.Conv2DTranspose,
                               nn.Conv3DTranspose, nn.Linear)):
            dim = 1
        else:
            dim = 0
    # 应用谱范数到模块的指定参数上
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    # 返回应用谱范数后的模块
    return module
```