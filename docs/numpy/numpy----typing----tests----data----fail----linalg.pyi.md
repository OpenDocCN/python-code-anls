# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\linalg.pyi`

```
import numpy as np  # 导入 NumPy 库

AR_f8: npt.NDArray[np.float64]  # 定义 AR_f8 变量，类型为 NumPy 的 float64 类型数组
AR_O: npt.NDArray[np.object_]  # 定义 AR_O 变量，类型为 NumPy 的 object 类型数组
AR_M: npt.NDArray[np.datetime64]  # 定义 AR_M 变量，类型为 NumPy 的 datetime64 类型数组

np.linalg.tensorsolve(AR_O, AR_O)  # 调用 NumPy 的 tensorsolve 函数，求解张量方程，但类型不兼容

np.linalg.solve(AR_O, AR_O)  # 调用 NumPy 的 solve 函数，求解线性方程组，但类型不兼容

np.linalg.tensorinv(AR_O)  # 调用 NumPy 的 tensorinv 函数，计算张量的逆，但类型不兼容

np.linalg.inv(AR_O)  # 调用 NumPy 的 inv 函数，计算方阵的逆，但类型不兼容

np.linalg.matrix_power(AR_M, 5)  # 调用 NumPy 的 matrix_power 函数，计算矩阵的整数次幂，但类型不兼容

np.linalg.cholesky(AR_O)  # 调用 NumPy 的 cholesky 函数，计算正定矩阵的 Cholesky 分解，但类型不兼容

np.linalg.qr(AR_O)  # 调用 NumPy 的 qr 函数，计算矩阵的 QR 分解，但类型不兼容
np.linalg.qr(AR_f8, mode="bob")  # 调用 NumPy 的 qr 函数，但没有符合的重载变体

np.linalg.eigvals(AR_O)  # 调用 NumPy 的 eigvals 函数，计算方阵的特征值，但类型不兼容

np.linalg.eigvalsh(AR_O)  # 调用 NumPy 的 eigvalsh 函数，计算实对称或复共轭对称矩阵的特征值，但类型不兼容
np.linalg.eigvalsh(AR_O, UPLO="bob")  # 调用 NumPy 的 eigvalsh 函数，但没有符合的重载变体

np.linalg.eig(AR_O)  # 调用 NumPy 的 eig 函数，计算方阵的特征值和特征向量，但类型不兼容

np.linalg.eigh(AR_O)  # 调用 NumPy 的 eigh 函数，计算实对称或复共轭对称矩阵的特征值和特征向量，但类型不兼容
np.linalg.eigh(AR_O, UPLO="bob")  # 调用 NumPy 的 eigh 函数，但没有符合的重载变体

np.linalg.svd(AR_O)  # 调用 NumPy 的 svd 函数，计算矩阵的奇异值分解，但类型不兼容

np.linalg.cond(AR_O)  # 调用 NumPy 的 cond 函数，计算矩阵的条件数，但类型不兼容
np.linalg.cond(AR_f8, p="bob")  # 调用 NumPy 的 cond 函数，但没有符合的重载变体

np.linalg.matrix_rank(AR_O)  # 调用 NumPy 的 matrix_rank 函数，计算矩阵的秩，但类型不兼容

np.linalg.pinv(AR_O)  # 调用 NumPy 的 pinv 函数，计算矩阵的伪逆，但类型不兼容

np.linalg.slogdet(AR_O)  # 调用 NumPy 的 slogdet 函数，计算矩阵的符号和对数行列式的值，但类型不兼容

np.linalg.det(AR_O)  # 调用 NumPy 的 det 函数，计算矩阵的行列式，但类型不兼容

np.linalg.norm(AR_f8, ord="bob")  # 调用 NumPy 的 norm 函数，但没有符合的重载变体

np.linalg.multi_dot([AR_M])  # 调用 NumPy 的 multi_dot 函数，计算多个数组的点积，但类型不兼容
```