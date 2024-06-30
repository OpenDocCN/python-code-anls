# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_mask.py`

```
# 导入 pytest 测试框架
import pytest

# 从 sklearn.utils._mask 模块中导入 safe_mask 函数
from sklearn.utils._mask import safe_mask
# 从 sklearn.utils.fixes 模块中导入 CSR_CONTAINERS 常量
from sklearn.utils.fixes import CSR_CONTAINERS
# 从 sklearn.utils.validation 模块中导入 check_random_state 函数
from sklearn.utils.validation import check_random_state


# 使用 pytest 的 parametrize 装饰器，对 csr_container 参数化
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
# 定义测试函数 test_safe_mask，接受 csr_container 参数
def test_safe_mask(csr_container):
    # 使用种子 0 初始化随机状态生成器
    random_state = check_random_state(0)
    # 生成一个 5x4 的随机矩阵 X
    X = random_state.rand(5, 4)
    # 使用 csr_container 将 X 转换为 CSR 格式稀疏矩阵 X_csr
    X_csr = csr_container(X)
    # 定义一个布尔类型的掩码 mask
    mask = [False, False, True, True, True]

    # 使用 safe_mask 函数对 X 应用掩码 mask
    mask = safe_mask(X, mask)
    # 断言通过掩码后的 X 的行数为 3
    assert X[mask].shape[0] == 3

    # 使用 safe_mask 函数对 X_csr 应用掩码 mask
    mask = safe_mask(X_csr, mask)
    # 断言通过掩码后的 X_csr 的行数为 3
    assert X_csr[mask].shape[0] == 3
```