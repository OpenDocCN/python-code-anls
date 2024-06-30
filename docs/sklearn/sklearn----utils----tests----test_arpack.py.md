# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_arpack.py`

```
# 导入 pytest 库，用于测试框架
import pytest
# 从 numpy.testing 模块中导入 assert_allclose 函数，用于比较数组是否接近
from numpy.testing import assert_allclose
# 从 sklearn.utils 中导入 check_random_state 函数，用于生成确定性的随机状态
from sklearn.utils import check_random_state
# 从 sklearn.utils._arpack 中导入 _init_arpack_v0 函数，即将测试的函数

# 使用 pytest.mark.parametrize 装饰器，参数化测试函数 test_init_arpack_v0，参数 seed 取值范围为 0 到 99
@pytest.mark.parametrize("seed", range(100))
# 定义测试函数 test_init_arpack_v0，参数 seed 作为随机数种子
def test_init_arpack_v0(seed):
    # 检查初始化是否从均匀分布中采样，其中可以固定随机状态
    # 定义数组的大小
    size = 1000
    # 调用 _init_arpack_v0 函数，生成长度为 size 的初始向量 v0
    v0 = _init_arpack_v0(size, seed)

    # 使用给定的 seed 创建随机状态 rng
    rng = check_random_state(seed)
    # 断言 v0 数组中的值接近于从均匀分布 [-1, 1) 中生成的随机数数组
    assert_allclose(v0, rng.uniform(-1, 1, size=size))
```