# `D:\src\scipysrc\scipy\benchmarks\benchmarks\peak_finding.py`

```
# 导入用于峰值查找相关功能的基准测试模块
from .common import Benchmark, safe_import

# 安全导入需要的模块，确保不存在导入错误
with safe_import():
    # 导入 scipy 库中的峰值查找、峰值显著性和峰值宽度函数，以及心电图数据集
    from scipy.signal import find_peaks, peak_prominences, peak_widths
    from scipy.datasets import electrocardiogram

# 定义一个 Benchmark 子类，用于对 `scipy.signal.find_peaks` 函数进行基准测试
class FindPeaks(Benchmark):
    """Benchmark `scipy.signal.find_peaks`.

    Notes
    -----
    The first value of `distance` is None in which case the benchmark shows
    the actual speed of the underlying maxima finding function.
    """

    # 参数名称为 `distance`，参数值包括 None 以及一系列整数
    param_names = ['distance']
    params = [[None, 8, 64, 512, 4096]]

    # 在测试运行前设置数据，读取心电图数据作为 self.x
    def setup(self, distance):
        self.x = electrocardiogram()

    # 测试 `find_peaks` 函数的执行时间，传入参数 `distance`
    def time_find_peaks(self, distance):
        find_peaks(self.x, distance=distance)


# 定义一个 Benchmark 子类，用于对 `scipy.signal.peak_prominences` 函数进行基准测试
class PeakProminences(Benchmark):
    """Benchmark `scipy.signal.peak_prominences`."""

    # 参数名称为 `wlen`，参数值包括 None 以及一系列整数
    param_names = ['wlen']
    params = [[None, 8, 64, 512, 4096]]

    # 在测试运行前设置数据，读取心电图数据作为 self.x，并使用 `find_peaks` 找到峰值
    def setup(self, wlen):
        self.x = electrocardiogram()
        self.peaks = find_peaks(self.x)[0]

    # 测试 `peak_prominences` 函数的执行时间，传入参数 `wlen`
    def time_peak_prominences(self, wlen):
        peak_prominences(self.x, self.peaks, wlen)


# 定义一个 Benchmark 子类，用于对 `scipy.signal.peak_widths` 函数进行基准测试
class PeakWidths(Benchmark):
    """Benchmark `scipy.signal.peak_widths`."""

    # 参数名称为 `rel_height`，参数值为一系列浮点数
    param_names = ['rel_height']
    params = [[0, 0.25, 0.5, 0.75, 1]]

    # 在测试运行前设置数据，读取心电图数据作为 self.x，并使用 `find_peaks` 找到峰值，
    # 使用 `peak_prominences` 计算峰值显著性数据
    def setup(self, rel_height):
        self.x = electrocardiogram()
        self.peaks = find_peaks(self.x)[0]
        self.prominence_data = peak_prominences(self.x, self.peaks)

    # 测试 `peak_widths` 函数的执行时间，传入参数 `rel_height` 和峰值显著性数据
    def time_peak_widths(self, rel_height):
        peak_widths(self.x, self.peaks, rel_height, self.prominence_data)
```