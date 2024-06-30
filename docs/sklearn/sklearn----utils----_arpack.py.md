# `D:\src\scipysrc\scikit-learn\sklearn\utils\_arpack.py`

```
# 从.validation模块中导入check_random_state函数
from .validation import check_random_state

# 定义一个名为_init_arpack_v0的函数，用于初始化ARPACK函数中的起始向量
def _init_arpack_v0(size, random_state):
    """Initialize the starting vector for iteration in ARPACK functions.

    Initialize a ndarray with values sampled from the uniform distribution on
    [-1, 1]. This initialization model has been chosen to be consistent with
    the ARPACK one as another initialization can lead to convergence issues.

    Parameters
    ----------
    size : int
        The size of the eigenvalue vector to be initialized.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator used to generate a
        uniform distribution. If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is the
        random number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    Returns
    -------
    v0 : ndarray of shape (size,)
        The initialized vector.
    """
    # 使用check_random_state函数确保random_state是一个RandomState实例
    random_state = check_random_state(random_state)
    # 生成一个大小为size的均匀分布的随机数组成的ndarray，取值范围在[-1, 1]之间
    v0 = random_state.uniform(-1, 1, size)
    # 返回初始化后的向量
    return v0
```