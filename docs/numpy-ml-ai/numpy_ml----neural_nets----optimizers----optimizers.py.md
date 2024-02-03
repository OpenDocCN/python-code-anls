# `numpy-ml\numpy_ml\neural_nets\optimizers\optimizers.py`

```
# 从 copy 模块中导入 deepcopy 函数
# 从 abc 模块中导入 ABC 和 abstractmethod 装饰器
import numpy as np
# 从 numpy.linalg 模块中导入 norm 函数

# 定义一个抽象基类 OptimizerBase，继承自 ABC 类
class OptimizerBase(ABC):
    # 初始化方法，接受学习率 lr 和调度器 scheduler 作为参数
    def __init__(self, lr, scheduler=None):
        """
        An abstract base class for all Optimizer objects.

        This should never be used directly.
        """
        # 从 ..initializers 模块中导入 SchedulerInitializer 类
        from ..initializers import SchedulerInitializer

        # 初始化缓存字典
        self.cache = {}
        # 初始化当前步数为 0
        self.cur_step = 0
        # 初始化超参数字典
        self.hyperparameters = {}
        # 使用 SchedulerInitializer 类创建学习率调度器对象
        self.lr_scheduler = SchedulerInitializer(scheduler, lr=lr)()

    # 定义 __call__ 方法，接受参数 param, param_grad, param_name, cur_loss，默认为 None
    def __call__(self, param, param_grad, param_name, cur_loss=None):
        return self.update(param, param_grad, param_name, cur_loss)

    # 定义 step 方法，用于将优化器步数加一
    def step(self):
        """Increment the optimizer step counter by 1"""
        self.cur_step += 1

    # 定义 reset_step 方法，用于将步数重置为 0
    def reset_step(self):
        """Reset the step counter to zero"""
        self.cur_step = 0

    # 定义 copy 方法，返回优化器对象的深拷贝
    def copy(self):
        """Return a copy of the optimizer object"""
        return deepcopy(self)

    # 定义 set_params 方法，从字典中设置优化器对象的参数
    def set_params(self, hparam_dict=None, cache_dict=None):
        """Set the parameters of the optimizer object from a dictionary"""
        # 从 ..initializers 模块中导入 SchedulerInitializer 类
        from ..initializers import SchedulerInitializer

        # 如果传入了超参数字典
        if hparam_dict is not None:
            # 遍历超参数字典
            for k, v in hparam_dict.items():
                # 如果键在超参数字典中
                if k in self.hyperparameters:
                    # 更新超参数字典的值
                    self.hyperparameters[k] = v
                    # 如果键是 "lr_scheduler"
                    if k == "lr_scheduler":
                        # 使用 SchedulerInitializer 类创建学习率调度器对象
                        self.lr_scheduler = SchedulerInitializer(v, lr=None)()

        # 如果传入了缓存字典
        if cache_dict is not None:
            # 遍历缓存字典
            for k, v in cache_dict.items():
                # 如果键在缓存字典中
                if k in self.cache:
                    # 更新缓存字典的值
                    self.cache[k] = v

    # 定义抽象方法 update，用于更新参数
    @abstractmethod
    def update(self, param, param_grad, param_name, cur_loss=None):
        raise NotImplementedError


# 定义 SGD 类，继承自 OptimizerBase 类
class SGD(OptimizerBase):
    # 初始化方法，接受学习率 lr、动量 momentum、梯度裁剪 clip_norm、学习率调度器 lr_scheduler 和其他关键字参数 kwargs
    def __init__(
        self, lr=0.01, momentum=0.0, clip_norm=None, lr_scheduler=None, **kwargs
    ):
        """
        A stochastic gradient descent optimizer.

        Notes
        -----
        For model parameters :math:`\\theta`, averaged parameter gradients
        :math:`\\nabla_{\\theta} \mathcal{L}`, and learning rate :math:`\eta`,
        the SGD update at timestep `t` is

        .. math::

            \\text{update}^{(t)}
                &=  \\text{momentum} \cdot \\text{update}^{(t-1)} + \eta^{(t)} \\nabla_{\\theta} \mathcal{L}\\\\
            \\theta^{(t+1)}
                &\leftarrow  \\theta^{(t)} - \\text{update}^{(t)}

        Parameters
        ----------
        lr : float
            Learning rate for SGD. If scheduler is not None, this is used as
            the starting learning rate. Default is 0.01.
        momentum : float in range [0, 1]
            The fraction of the previous update to add to the current update.
            If 0, no momentum is applied. Default is 0.
        clip_norm : float
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update. Default is None.
        lr_scheduler : str, :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object, or None
            The learning rate scheduler. If None, use a constant learning
            rate equal to `lr`. Default is None.
        """
        # 调用父类的构造函数，初始化学习率和学习率调度器
        super().__init__(lr, lr_scheduler)

        # 设置超参数字典
        self.hyperparameters = {
            "id": "SGD",
            "lr": lr,
            "momentum": momentum,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }

    # 返回优化器的字符串表示
    def __str__(self):
        # 获取超参数字典
        H = self.hyperparameters
        lr, mm, cn, sc = H["lr"], H["momentum"], H["clip_norm"], H["lr_scheduler"]
        # 返回优化器的字符串表示
        return "SGD(lr={}, momentum={}, clip_norm={}, lr_scheduler={})".format(
            lr, mm, cn, sc
        )
    # 定义一个方法，用于计算给定参数的 SGD 更新
    def update(self, param, param_grad, param_name, cur_loss=None):
        """
        Compute the SGD update for a given parameter

        Parameters
        ----------
        param : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of the parameter to be updated.
        param_grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the loss function with respect to `param_name`.
        param_name : str
            The name of the parameter.
        cur_loss : float
            The training or validation loss for the current minibatch. Used for
            learning rate scheduling e.g., by
            :class:`~numpy_ml.neural_nets.schedulers.KingScheduler`.
            Default is None.

        Returns
        -------
        updated_params : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of `param` after applying the momentum update.
        """
        # 获取缓存和超参数
        C = self.cache
        H = self.hyperparameters
        momentum, clip_norm = H["momentum"], H["clip_norm"]
        # 根据当前步数和当前损失计算学习率
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        # 如果参数名不在缓存中，则初始化为全零数组
        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # 缩放梯度以避免梯度爆炸
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        # 计算更新值，包括动量和学习率
        update = momentum * C[param_name] + lr * param_grad
        # 更新缓存中的参数值
        self.cache[param_name] = update
        # 返回更新后的参数值
        return param - update
# 自适应梯度方法
# 定义 AdaGrad 类，继承自 OptimizerBase 类
class AdaGrad(OptimizerBase):
    # 初始化 AdaGrad 优化器
    def __init__(self, lr=0.01, eps=1e-7, clip_norm=None, lr_scheduler=None, **kwargs):
        """
        An AdaGrad optimizer.

        Notes
        -----
        Weights that receive large gradients will have their effective learning
        rate reduced, while weights that receive small or infrequent updates
        will have their effective learning rate increased.

        Equations::

            cache[t] = cache[t-1] + grad[t] ** 2
            update[t] = lr * grad[t] / (np.sqrt(cache[t]) + eps)
            param[t+1] = param[t] - update[t]

        Note that the ``**`` and `/` operations are elementwise

        "A downside of Adagrad ... is that the monotonic learning rate usually
        proves too aggressive and stops learning too early." [1]

        References
        ----------
        .. [1] Karpathy, A. "CS231n: Convolutional neural networks for visual
           recognition" https://cs231n.github.io/neural-networks-3/

        Parameters
        ----------
        lr : float
            Global learning rate
        eps : float
            Smoothing term to avoid divide-by-zero errors in the update calc.
            Default is 1e-7.
        clip_norm : float or None
            If not None, all param gradients are scaled to have maximum `L2` norm of
            `clip_norm` before computing update. Default is None.
        lr_scheduler : str or :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object or None
            The learning rate scheduler. If None, use a constant learning
            rate equal to `lr`. Default is None.
        """
        # 调用父类的初始化方法，传入全局学习率 lr 和学习率调度器 lr_scheduler
        super().__init__(lr, lr_scheduler)

        # 初始化缓存字典
        self.cache = {}
        # 初始化超参数字典
        self.hyperparameters = {
            "id": "AdaGrad",
            "lr": lr,
            "eps": eps,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }
    # 定义对象的字符串表示形式，包括超参数 lr, eps, clip_norm, lr_scheduler
    def __str__(self):
        H = self.hyperparameters
        lr, eps, cn, sc = H["lr"], H["eps"], H["clip_norm"], H["lr_scheduler"]
        return "AdaGrad(lr={}, eps={}, clip_norm={}, lr_scheduler={})".format(
            lr, eps, cn, sc
        )

    # 更新给定参数的 AdaGrad 更新
    def update(self, param, param_grad, param_name, cur_loss=None):
        """
        Compute the AdaGrad update for a given parameter.

        Notes
        -----
        Adjusts the learning rate of each weight based on the magnitudes of its
        gradients (big gradient -> small lr, small gradient -> big lr).

        Parameters
        ----------
        param : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of the parameter to be updated
        param_grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the loss function with respect to `param_name`
        param_name : str
            The name of the parameter
        cur_loss : float or None
            The training or validation loss for the current minibatch. Used for
            learning rate scheduling e.g., by
            :class:`~numpy_ml.neural_nets.schedulers.KingScheduler`.
            Default is None.

        Returns
        -------
        updated_params : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of `param` after applying the AdaGrad update
        """
        C = self.cache
        H = self.hyperparameters
        eps, clip_norm = H["eps"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        # 如果参数名不在缓存中，则初始化为零数组
        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # 缩放梯度以避免梯度爆炸
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        # 更新缓存中的值
        C[param_name] += param_grad ** 2
        # 计算更新值
        update = lr * param_grad / (np.sqrt(C[param_name]) + eps)
        self.cache = C
        return param - update
class RMSProp(OptimizerBase):
    # RMSProp 优化器类，继承自 OptimizerBase 基类
    def __init__(
        self, lr=0.001, decay=0.9, eps=1e-7, clip_norm=None, lr_scheduler=None, **kwargs
    ):
        """
        RMSProp optimizer.

        Notes
        -----
        RMSProp was proposed as a refinement of :class:`AdaGrad` to reduce its
        aggressive, monotonically decreasing learning rate.

        RMSProp uses a *decaying average* of the previous squared gradients
        (second moment) rather than just the immediately preceding squared
        gradient for its `previous_update` value.

        Equations::

            cache[t] = decay * cache[t-1] + (1 - decay) * grad[t] ** 2
            update[t] = lr * grad[t] / (np.sqrt(cache[t]) + eps)
            param[t+1] = param[t] - update[t]

        Note that the ``**`` and ``/`` operations are elementwise.

        Parameters
        ----------
        lr : float
            Learning rate for update. Default is 0.001.
        decay : float in [0, 1]
            Rate of decay for the moving average. Typical values are [0.9,
            0.99, 0.999]. Default is 0.9.
        eps : float
            Constant term to avoid divide-by-zero errors during the update calc. Default is 1e-7.
        clip_norm : float or None
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update. Default is None.
        lr_scheduler : str or :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object or None
            The learning rate scheduler. If None, use a constant learning
            rate equal to `lr`. Default is None.
        """
        # 调用父类的初始化方法，传入学习率和学习率调度器
        super().__init__(lr, lr_scheduler)

        # 初始化缓存字典
        self.cache = {}
        # 初始化超参数字典
        self.hyperparameters = {
            "id": "RMSProp",
            "lr": lr,
            "eps": eps,
            "decay": decay,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }
    # 定义对象的字符串表示形式，包括超参数和学习率调度器信息
    def __str__(self):
        # 获取超参数字典和学习率调度器
        H = self.hyperparameters
        sc = H["lr_scheduler"]
        lr, eps, dc, cn = H["lr"], H["eps"], H["decay"], H["clip_norm"]
        # 返回对象的字符串表示形式
        return "RMSProp(lr={}, eps={}, decay={}, clip_norm={}, lr_scheduler={})".format(
            lr, eps, dc, cn, sc
        )

    # 更新给定参数的 RMSProp 更新
    def update(self, param, param_grad, param_name, cur_loss=None):
        """
        Compute the RMSProp update for a given parameter.

        Parameters
        ----------
        param : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of the parameter to be updated
        param_grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the loss function with respect to `param_name`
        param_name : str
            The name of the parameter
        cur_loss : float or None
            The training or validation loss for the current minibatch. Used for
            learning rate scheduling e.g., by
            :class:`~numpy_ml.neural_nets.schedulers.KingScheduler`.
            Default is None.

        Returns
        -------
        updated_params : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The value of `param` after applying the RMSProp update.
        """
        # 获取缓存和超参数字典
        C = self.cache
        H = self.hyperparameters
        eps, decay, clip_norm = H["eps"], H["decay"], H["clip_norm"]
        # 根据当前步数和损失计算学习率
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        # 如果参数名不在缓存中，则初始化为零数组
        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # 缩放梯度以避免梯度爆炸
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        # 计算 RMSProp 更新
        C[param_name] = decay * C[param_name] + (1 - decay) * param_grad ** 2
        update = lr * param_grad / (np.sqrt(C[param_name]) + eps)
        self.cache = C
        # 返回更新后的参数值
        return param - update
# 定义 Adam 优化器类，继承自 OptimizerBase 类
class Adam(OptimizerBase):
    # 初始化 Adam 优化器对象
    def __init__(
        self,
        lr=0.001,  # 学习率，默认为 0.001
        decay1=0.9,  # 第一矩估计的衰减率，默认为 0.9
        decay2=0.999,  # 第二矩估计的衰减率，默认为 0.999
        eps=1e-7,  # 避免除零错误的常数项，默认为 1e-7
        clip_norm=None,  # 梯度裁剪的最大 l2 范数，默认为 None
        lr_scheduler=None,  # 学习率调度器，默认为 None
        **kwargs
    ):
        """
        Adam (adaptive moment estimation) optimization algorithm.

        Notes
        -----
        Designed to combine the advantages of :class:`AdaGrad`, which works
        well with sparse gradients, and :class:`RMSProp`, which works well in
        online and non-stationary settings.

        Parameters
        ----------
        lr : float
            Learning rate for update. This parameter is ignored if using
            :class:`~numpy_ml.neural_nets.schedulers.NoamScheduler`.
            Default is 0.001.
        decay1 : float
            The rate of decay to use for in running estimate of the first
            moment (mean) of the gradient. Default is 0.9.
        decay2 : float
            The rate of decay to use for in running estimate of the second
            moment (variance) of the gradient. Default is 0.999.
        eps : float
            Constant term to avoid divide-by-zero errors during the update
            calc. Default is 1e-7.
        clip_norm : float
            If not None, all param gradients are scaled to have maximum l2 norm of
            `clip_norm` before computing update. Default is None.
        lr_scheduler : str, or :doc:`Scheduler <numpy_ml.neural_nets.schedulers>` object, or None
            The learning rate scheduler. If None, use a constant learning rate
            equal to `lr`. Default is None.
        """
        # 调用父类的初始化方法
        super().__init__(lr, lr_scheduler)

        # 初始化缓存字典
        self.cache = {}
        # 初始化超参数字典
        self.hyperparameters = {
            "id": "Adam",
            "lr": lr,
            "eps": eps,
            "decay1": decay1,
            "decay2": decay2,
            "clip_norm": clip_norm,
            "lr_scheduler": str(self.lr_scheduler),
        }
    # 定义类的字符串表示方法，返回 Adam 优化器的超参数信息
    def __str__(self):
        # 获取超参数字典
        H = self.hyperparameters
        # 从超参数字典中获取 lr, decay1, decay2 的值
        lr, d1, d2 = H["lr"], H["decay1"], H["decay2"]
        # 从超参数字典中获取 eps, clip_norm, lr_scheduler 的值
        eps, cn, sc = H["eps"], H["clip_norm"], H["lr_scheduler"]
        # 返回格式化后的字符串，包含 lr, decay1, decay2, eps, clip_norm, lr_scheduler 的值
        return "Adam(lr={}, decay1={}, decay2={}, eps={}, clip_norm={}, lr_scheduler={})".format(
            lr, d1, d2, eps, cn, sc
        )
```