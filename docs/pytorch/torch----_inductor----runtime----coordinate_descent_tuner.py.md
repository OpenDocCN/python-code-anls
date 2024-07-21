# `.\pytorch\torch\_inductor\runtime\coordinate_descent_tuner.py`

```
    # 设置 mypy: allow-untyped-defs 以允许未类型化的函数定义
import copy  # 导入复制模块
import itertools  # 导入迭代工具模块
import logging  # 导入日志模块
from typing import Callable, Optional  # 导入类型提示相关模块

from .hints import TRITON_MAX_BLOCK  # 导入 TRITON_MAX_BLOCK 提示模块

from .runtime_utils import red_text, triton_config_to_hashable  # 导入运行时工具模块中的 red_text 和 triton_config_to_hashable 函数

try:
    import triton  # 尝试导入 triton 库
except ImportError:
    triton = None  # 如果导入失败，将 triton 设为 None

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def get_field(config, name):
    # 根据字段名获取配置对象中的值
    if name == "num_warps":
        return config.num_warps
    elif name == "num_stages":
        return config.num_stages
    else:
        return config.kwargs.get(name, None)


def set_field(config, name, value):
    # 根据字段名设置配置对象中的值
    if name == "num_warps":
        config.num_warps = value
    elif name == "num_stages":
        config.num_stages = value
    else:
        config.kwargs[name] = value


class CoordescTuner:
    """
    The coordinate descent tuner. Tune one field/coordinate at a time.

    TODO will it be necessary to tune multiple fields simultaneously.


    TODO: what if both increasing and decreasing a field can improve perf.
          i.e., there are multiple local optima..
    """

    def __init__(
        self, is_mm=False, name="unknown", size_hints=None, inductor_meta=None
    ):
        # 初始化坐标下降调节器对象
        self.is_mm = is_mm  # 是否为矩阵乘法调优
        self.cached_benchmark_results = {}  # 缓存性能基准结果的空字典
        self.name = name  # 调节器名称
        self.size_hints = size_hints  # 大小提示
        self.inductor_meta = inductor_meta or {}  # 电感元数据字典

    def get_xmax(self):
        # 获取 X 维度的最大值
        xmax = TRITON_MAX_BLOCK["X"]
        if self.size_hints and len(self.size_hints) > 0:
            xmax = min(xmax, self.size_hints[0])
        return xmax

    def get_ymax(self):
        # 获取 Y 维度的最大值
        ymax = TRITON_MAX_BLOCK["Y"]
        if self.size_hints and len(self.size_hints) > 1:
            ymax = min(ymax, self.size_hints[1])
        return ymax

    def get_zmax(self):
        # 获取 Z 维度的最大值
        zmax = TRITON_MAX_BLOCK["Z"]
        if self.size_hints and len(self.size_hints) > 2:
            zmax = min(zmax, self.size_hints[2])
        return zmax

    def get_rmax(self):
        # 获取 R 维度的最大值
        rmax = TRITON_MAX_BLOCK["R"]
        if self.size_hints and len(self.size_hints) > 0:
            rmax = min(rmax, self.size_hints[-1])  # 最后一个用于约简
        return rmax

    def get_warpsmax(self):
        # 目前 CUDA 最多有 1024 个线程，所以最大的线程束数为 32
        return 1024 // 32

    def cache_benchmark_result(self, config, timing):
        # 缓存基准测试结果
        self.cached_benchmark_results[triton_config_to_hashable(config)] = timing

    def lookup_in_cache(self, config):
        # 查找缓存中是否存在指定配置的结果
        return self.cached_benchmark_results.get(triton_config_to_hashable(config))

    def call_func(self, func, config):
        # 调用函数并缓存结果
        found = self.lookup_in_cache(config)
        if found is not None:
            log.debug("  CACHED")
            return found
        timing = func(config)
        self.cache_benchmark_result(config, timing)
        return timing

    @property
    # 返回一个包含所有可调整字段的列表
    def tunable_fields(self):
        out = [
            "XBLOCK",
            "YBLOCK",
            "ZBLOCK",
            # 注意：我们不应调整 RBLOCK 以进行持久性降低。
            # 我们依赖于持久性降低的 triton.Config 不包含 RBLOCK 字段来保证这一点。
            "RBLOCK",
            # 以下三个字段用于矩阵乘法
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
        ]
        # 如果是矩阵乘法，则添加 num_stages 字段
        if self.is_mm:
            out.append("num_stages")

        return out

    # 检查字段值是否过大
    def value_too_large(self, name, val):
        if name == "XBLOCK":
            return val > self.get_xmax()
        if name == "YBLOCK":
            return val > self.get_ymax()
        if name == "ZBLOCK":
            return val > self.get_zmax()
        if name == "RBLOCK":
            return val > self.get_rmax()
        if name == "num_warps":
            return val > self.get_warpsmax()

        return False

    # 获取邻居值列表，即指定半径内的值
    def get_neighbour_values(self, name, orig_val, radius=1, include_self=False):
        """
        Get neighbour values in 'radius' steps. The original value is not
        returned as it's own neighbour.
        """
        assert radius >= 1

        # 更新当前值的函数
        def update(cur_val, inc=True):
            if name == "num_stages":
                if inc:
                    return cur_val + 1
                else:
                    return cur_val - 1
            else:
                if inc:
                    return cur_val * 2
                else:
                    return cur_val // 2

        out = []
        # 增加循环
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, True)
            # 检查值是否过大
            if self.value_too_large(name, cur_val):
                break
            out.append(cur_val)

        # 减少循环
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, False)
            # 如果当前值小于等于 0，则停止循环
            if cur_val <= 0:
                break
            out.append(cur_val)

        # 是否包含原始值自身
        if include_self:
            out.append(orig_val)
        return out

    @staticmethod
    # 检查测试值是否优于基准值
    def has_improvement(baseline, test):
        threshold = 0.001  # 0.1%
        return test is not None and test < baseline * (1 - threshold)

    # 检查所有调优方向
    def check_all_tuning_directions(
        self,
        func: Callable[["triton.Config"], float],
        best_config,
        best_timing,
    ):
        """
        检查所有方向。只有在常规坐标下降调优找不到更好选择时才执行此操作。
        我们只有少数可调整字段，因此这应该没问题。
        """
        candidate_values_list = []
        effective_fields = []
        for field in self.tunable_fields:
            old_value = get_field(best_config, field)
            if old_value is None:
                continue
            # 获取邻近值列表，包括自身，根据指定的搜索半径
            candidate_values = self.get_neighbour_values(
                field,
                old_value,
                radius=self.inductor_meta.get("coordinate_descent_search_radius", 1),
                include_self=True,
            )
            candidate_values_list.append(candidate_values)
            effective_fields.append(field)

        # 生成所有可行的选择组合
        choices = itertools.product(*candidate_values_list)
        improved = False
        for choice in choices:
            assert len(choice) == len(effective_fields)
            # 深拷贝当前最佳配置
            candidate_config = copy.deepcopy(best_config)
            # 将新值应用到候选配置中
            for new_val, field in zip(choice, effective_fields):
                set_field(candidate_config, field, new_val)
            # 比较候选配置与当前最佳配置的性能
            cmp_res, candidate_timing = self.compare_config(
                func, candidate_config, best_config, best_timing
            )
            if cmp_res:
                # 如果候选配置更优，则更新最佳配置和最佳时间
                improved = True
                best_config = candidate_config
                best_timing = candidate_timing

        return improved, best_config, best_timing

    def compare_config(self, func, candidate_config, best_config, best_timing):
        """
        检查 candidate_config 是否比 best_config 更好。

        返回一个元组 (compare_result, candidate_timing)。
        如果 candidate_config 更好，compare_result 为 True。
        """
        log.debug("尝试配置 %s", candidate_config)
        try:
            # 调用函数并计时候选配置
            candidate_timing = self.call_func(func, candidate_config)
        except Exception as e:
            log.debug("遇到异常 %s", e)
            return False, float("inf")

        # 检查是否有性能改善
        if self.has_improvement(best_timing, candidate_timing):
            log.debug(
                "从 %s %f 调整到 %s %f",
                best_config,
                best_timing,
                candidate_config,
                candidate_timing,
            )
            return True, candidate_timing
        return False, candidate_timing

    def autotune(
        self,
        func: Callable[["triton.Config"], float],
        baseline_config: "triton.Config",
        baseline_timing: Optional[float] = None,
        ) -> "triton.Config":
        # 定义函数签名，指定返回类型为 "triton.Config"
        if baseline_timing is None:
            # 如果基准时间未定义，则通过调用函数获取基准时间
            baseline_timing = self.call_func(func, baseline_config)

        # 输出调试信息，显示当前进行坐标下降调优的对象名称
        log.debug("= Do coordinate descent tuning for %s =", self.name)
        # 输出调试信息，显示基准配置和基准时间
        log.debug(
            "Baseline Config %s, baseline timing %f", baseline_config, baseline_timing
        )
        # 初始化改进标志为 True
        improved = True
        # 将最佳配置和最佳时间设置为基准配置和基准时间
        best_config = baseline_config
        best_timing = baseline_timing
        # 获取可调整字段列表
        tunable_fields = self.tunable_fields

        # 进入改进循环，直到不再改进为止
        while improved:
            # 每次循环开始时，将改进标志重置为 False
            improved = False

            # 遍历所有可调整字段
            for name in tunable_fields:
                # 获取当前字段在最佳配置中的值
                cur_val = get_field(best_config, name)
                # 如果当前值为 None，则跳过本次循环
                if cur_val is None:
                    continue

                # 获取候选的下一个数值列表，用于尝试不同配置
                candidate_values = self.get_neighbour_values(name, cur_val)

                # 遍历候选数值列表
                for next_val in candidate_values:
                    # 深拷贝最佳配置，以便修改并测试新的候选配置
                    candidate_config = copy.deepcopy(best_config)
                    set_field(candidate_config, name, next_val)

                    # 比较新的候选配置和当前最佳配置，返回比较结果和候选时间
                    cmp_res, candidate_timing = self.compare_config(
                        func, candidate_config, best_config, best_timing
                    )
                    # 如果比较结果表明候选配置更优，则更新最佳配置和最佳时间
                    if cmp_res:
                        improved = True
                        best_config, best_timing = candidate_config, candidate_timing

            # 如果在当前轮次没有改进，并且指示需要检查所有调优方向
            if not improved and self.inductor_meta.get(
                "coordinate_descent_check_all_directions"
            ):
                # 保存旧的最佳时间
                old_best_timing = best_timing
                # 通过检查所有调优方向来尝试改进
                improved, best_config, best_timing = self.check_all_tuning_directions(
                    func, best_config, best_timing
                )

                # 如果改进成功，输出消息提示
                if improved:
                    msg = red_text(
                        "Coordinate descend tuning found improvement of %.3fx by looking in all directions."
                    )
                    log.debug(
                        msg,
                        old_best_timing / best_timing,
                    )

        # 输出调试信息，显示从基准配置和时间到最佳配置和时间的改进比例
        log.debug(
            "Improve from %s %f -> %s %f, %.3fx",
            baseline_config,
            baseline_timing,
            best_config,
            best_timing,
            baseline_timing / best_timing,
        )

        # 返回最佳配置
        return best_config
```