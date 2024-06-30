# `D:\src\scipysrc\scipy\scipy\stats\_qmc_cy.pyi`

```
# 导入NumPy库，简称为np
import numpy as np
# 从scipy库的内部工具中导入DecimalNumber和IntNumber类
from scipy._lib._util import DecimalNumber, IntNumber


# 定义一个名为_cy_wrapper_centered_discrepancy的函数，接受三个参数：
# sample: np.ndarray，代表样本数据的NumPy数组
# iterative: bool，布尔值，指示是否使用迭代模式
# workers: IntNumber，整数类型，指定工作线程数
# 返回值为float类型，表示计算得到的中心差异度
def _cy_wrapper_centered_discrepancy(
        sample: np.ndarray, 
        iterative: bool, 
        workers: IntNumber,
) -> float: ...


# 定义一个名为_cy_wrapper_wrap_around_discrepancy的函数，接受三个参数：
# sample: np.ndarray，代表样本数据的NumPy数组
# iterative: bool，布尔值，指示是否使用迭代模式
# workers: IntNumber，整数类型，指定工作线程数
# 返回值为float类型，表示计算得到的环绕差异度
def _cy_wrapper_wrap_around_discrepancy(
        sample: np.ndarray,
        iterative: bool, 
        workers: IntNumber,
) -> float: ...


# 定义一个名为_cy_wrapper_mixture_discrepancy的函数，接受三个参数：
# sample: np.ndarray，代表样本数据的NumPy数组
# iterative: bool，布尔值，指示是否使用迭代模式
# workers: IntNumber，整数类型，指定工作线程数
# 返回值为float类型，表示计算得到的混合差异度
def _cy_wrapper_mixture_discrepancy(
        sample: np.ndarray,
        iterative: bool, 
        workers: IntNumber,
) -> float: ...


# 定义一个名为_cy_wrapper_l2_star_discrepancy的函数，接受三个参数：
# sample: np.ndarray，代表样本数据的NumPy数组
# iterative: bool，布尔值，指示是否使用迭代模式
# workers: IntNumber，整数类型，指定工作线程数
# 返回值为float类型，表示计算得到的L2星差异度
def _cy_wrapper_l2_star_discrepancy(
        sample: np.ndarray,
        iterative: bool,
        workers: IntNumber,
) -> float: ...


# 定义一个名为_cy_wrapper_update_discrepancy的函数，接受三个参数：
# x_new_view: np.ndarray，代表更新后的新视图数据的NumPy数组
# sample_view: np.ndarray，代表样本视图数据的NumPy数组
# initial_disc: DecimalNumber，DecimalNumber类型，表示初始差异度
# 返回值为float类型，表示更新后的差异度
def _cy_wrapper_update_discrepancy(
        x_new_view: np.ndarray,
        sample_view: np.ndarray,
        initial_disc: DecimalNumber,
) -> float: ...


# 定义一个名为_cy_van_der_corput的函数，接受四个参数：
# n: IntNumber，整数类型，表示生成数列的长度
# base: IntNumber，整数类型，表示Van der Corput序列的基数
# start_index: IntNumber，整数类型，表示起始索引
# workers: IntNumber，整数类型，指定工作线程数
# 返回值为np.ndarray类型，表示生成的Van der Corput序列
def _cy_van_der_corput(
        n: IntNumber,
        base: IntNumber,
        start_index: IntNumber,
        workers: IntNumber,
) -> np.ndarray: ...


# 定义一个名为_cy_van_der_corput_scrambled的函数，接受五个参数：
# n: IntNumber，整数类型，表示生成数列的长度
# base: IntNumber，整数类型，表示Van der Corput序列的基数
# start_index: IntNumber，整数类型，表示起始索引
# permutations: np.ndarray，代表置换的NumPy数组
# workers: IntNumber，整数类型，指定工作线程数
# 返回值为np.ndarray类型，表示生成的混淆Van der Corput序列
def _cy_van_der_corput_scrambled(
        n: IntNumber,
        base: IntNumber,
        start_index: IntNumber,
        permutations: np.ndarray,
        workers: IntNumber,
) -> np.ndarray: ...
```