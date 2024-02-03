# `numpy-ml\numpy_ml\nonparametric\kernel_regression.py`

```
from ..utils.kernels import KernelInitializer
# 导入 KernelInitializer 模块

class KernelRegression:
    def __init__(self, kernel=None):
        """
        A Nadaraya-Watson kernel regression model.

        Notes
        -----
        The Nadaraya-Watson regression model is

        .. math::

            f(x) = \sum_i w_i(x) y_i

        where the sample weighting functions, :math:`w_i`, are simply

        .. math::

            w_i(x) = \\frac{k(x, x_i)}{\sum_j k(x, x_j)}

        with `k` being the kernel function.

        Observe that `k`-nearest neighbors
        (:class:`~numpy_ml.nonparametric.KNN`) regression is a special case of
        kernel regression where the `k` closest observations have a weight
        `1/k`, and all others have weight 0.

        Parameters
        ----------
        kernel : str, :doc:`Kernel <numpy_ml.utils.kernels>` object, or dict
            The kernel to use. If None, default to
            :class:`~numpy_ml.utils.kernels.LinearKernel`. Default is None.
        """
        self.parameters = {"X": None, "y": None}
        # 初始化参数字典，包含键值对 "X": None, "y": None
        self.hyperparameters = {"kernel": str(kernel)}
        # 初始化超参数字典，包含键值对 "kernel": str(kernel)
        self.kernel = KernelInitializer(kernel)()
        # 使用 KernelInitializer 初始化 kernel 对象

    def fit(self, X, y):
        """
        Fit the regression model to the data and targets in `X` and `y`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            An array of N examples to generate predictions on
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, ...)`
            Predicted targets for the `N` rows in `X`
        """
        self.parameters = {"X": X, "y": y}
        # 更新参数字典，将 "X" 和 "y" 更新为传入的 X 和 y
    # 为模型对象定义一个预测方法，用于生成与输入数据 `X` 相关的目标预测值
    def predict(self, X):
        """
        Generate predictions for the targets associated with the rows in `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N', M')`
            An array of `N'` examples to generate predictions on

        Returns
        -------
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N', ...)`
            Predicted targets for the `N'` rows in `X`
        """
        # 获取模型对象中的核函数和参数
        K = self.kernel
        P = self.parameters
        # 计算输入数据 `X` 与参数中的训练数据之间的相似度
        sim = K(P["X"], X)
        # 返回预测值，计算方法为相似度乘以参数中的目标值，然后按列求和并除以相似度的列和
        return (sim * P["y"][:, None]).sum(axis=0) / sim.sum(axis=0)
```