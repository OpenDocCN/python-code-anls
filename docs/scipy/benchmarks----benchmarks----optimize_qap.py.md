# `D:\src\scipysrc\scipy\benchmarks\benchmarks\optimize_qap.py`

```
import numpy as np   # 导入 NumPy 库，用于数值计算
from .common import Benchmark, safe_import   # 从当前目录下的 common 模块导入 Benchmark 类和 safe_import 函数
import os   # 导入操作系统相关的功能

with safe_import():   # 使用 safe_import 函数来安全地导入依赖，确保不会出现导入错误
    from scipy.optimize import quadratic_assignment   # 从 SciPy 库中导入 quadratic_assignment 函数，用于求解二次分配问题

# XXX this should probably have an is_xslow with selected tests.
# Even with this, it takes ~30 seconds to collect the ones to run
# (even if they will all be skipped in the `setup` function).

class QuadraticAssignment(Benchmark):   # 定义 QuadraticAssignment 类，继承自 Benchmark 类
    methods = ['faq', '2opt']   # 定义可用的方法列表
    probs = ["bur26a", "bur26b", "bur26c", "bur26d", "bur26e", "bur26f",
             "bur26g", "bur26h", "chr12a", "chr12b", "chr12c", "chr15a",
             "chr15b", "chr15c", "chr18a", "chr18b", "chr20a", "chr20b",
             "chr20c", "chr22a", "chr22b", "chr25a",
             "els19",
             "esc16a", "esc16b", "esc16c", "esc16d", "esc16e", "esc16g",
             "esc16h", "esc16i", "esc16j", "esc32e", "esc32g", "esc128",
             "had12", "had14", "had16", "had18", "had20", "kra30a",
             "kra30b", "kra32",
             "lipa20a", "lipa20b", "lipa30a", "lipa30b", "lipa40a", "lipa40b",
             "lipa50a", "lipa50b", "lipa60a", "lipa60b", "lipa70a", "lipa70b",
             "lipa80a", "lipa90a", "lipa90b",
             "nug12", "nug14", "nug16a", "nug16b", "nug17", "nug18", "nug20",
             "nug21", "nug22", "nug24", "nug25", "nug27", "nug28", "nug30",
             "rou12", "rou15", "rou20",
             "scr12", "scr15", "scr20",
             "sko42", "sko49", "sko56", "sko64", "sko72", "sko81", "sko90",
             "sko100a", "sko100b", "sko100c", "sko100d", "sko100e", "sko100f",
             "ste36b", "ste36c",
             "tai12a", "tai12b", "tai15a", "tai15b", "tai17a", "tai20a",
             "tai20b", "tai25a", "tai25b", "tai30a", "tai30b", "tai35a",
             "tai40a", "tai40b", "tai50a", "tai50b", "tai60a", "tai60b",
             "tai64c", "tai80a", "tai100a", "tai100b", "tai150b", "tai256c",
             "tho30", "tho40", "tho150", "wil50", "wil100"]   # 定义问题实例的名称列表
    params = [methods, probs]   # 参数列表包括方法和问题实例
    param_names = ['Method', 'QAP Problem']   # 参数的名称，分别为方法和问题实例

    def setup(self, method, qap_prob):   # 设置方法，用于初始化测试环境
        dir_path = os.path.dirname(os.path.realpath(__file__))   # 获取当前脚本文件的目录路径
        datafile = np.load(os.path.join(dir_path, "qapdata/qap_probs.npz"),   # 加载 QAP 数据文件
                           allow_pickle=True)
        slnfile = np.load(os.path.join(dir_path, "qapdata/qap_sols.npz"),   # 加载 QAP 解决方案文件
                          allow_pickle=True)
        self.A = datafile[qap_prob][0]   # 设置问题实例 A 的数据
        self.B = datafile[qap_prob][1]   # 设置问题实例 B 的数据
        self.opt_solution = slnfile[qap_prob]   # 设置最优解决方案
        self.method = method   # 设置使用的方法

    def time_evaluation(self, method, qap_prob):   # 定义时间评估函数，评估方法在问题实例上的运行时间
        quadratic_assignment(self.A, self.B, self.method)

    def track_score(self, method, qap_prob):   # 定义跟踪分数函数，评估方法的解决方案得分
        res = quadratic_assignment(self.A, self.B, self.method)   # 调用 quadratic_assignment 函数求解
        score = int(res['fun'])   # 获取目标函数的值（解决方案得分）
        percent_diff = (score - self.opt_solution) / self.opt_solution   # 计算解决方案得分相对最优解的百分比差异
        return percent_diff   # 返回百分比差异值
```