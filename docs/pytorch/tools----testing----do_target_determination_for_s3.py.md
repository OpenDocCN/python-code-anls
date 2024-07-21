# `.\pytorch\tools\testing\do_target_determination_for_s3.py`

```py
# 导入必要的模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统功能的模块
import sys  # 导入系统相关的功能模块
from pathlib import Path  # 导入处理文件路径的模块

# 确定项目根目录
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# 将项目根目录添加到系统路径中，以便导入项目内部模块
sys.path.insert(0, str(REPO_ROOT))

# 导入自定义的模块和函数
from tools.stats.import_test_stats import (
    copy_additional_previous_failures,
    copy_pytest_cache,
    get_td_heuristic_historial_edited_files_json,
    get_td_heuristic_profiling_json,
    get_test_class_ratings,
    get_test_class_times,
    get_test_file_ratings,
    get_test_times,
)
from tools.stats.upload_metrics import emit_metric

from tools.testing.discover_tests import TESTS  # 导入测试发现模块
from tools.testing.target_determination.determinator import (
    AggregatedHeuristics,
    get_test_prioritizations,
    TestPrioritizations,
)

# 从系统路径中移除项目根目录路径
sys.path.remove(str(REPO_ROOT))


# 导入测试结果函数
def import_results() -> TestPrioritizations:
    # 检查是否存在测试结果 JSON 文件
    if not (REPO_ROOT / ".additional_ci_files/td_results.json").exists():
        print("No TD results found")  # 输出未找到测试决策结果信息
        return TestPrioritizations([], {})  # 返回空的测试优先级对象

    # 读取测试决策结果 JSON 文件
    with open(REPO_ROOT / ".additional_ci_files/td_results.json") as f:
        td_results = json.load(f)  # 加载 JSON 文件内容
        tp = TestPrioritizations.from_json(td_results)  # 使用 JSON 数据创建测试优先级对象

    return tp  # 返回加载的测试优先级对象


# 主函数
def main() -> None:
    selected_tests = TESTS  # 选择要运行的测试

    # 创建聚合启发式对象
    aggregated_heuristics: AggregatedHeuristics = AggregatedHeuristics(selected_tests)

    # 获取测试运行时间信息
    get_test_times()
    # 获取测试类运行时间信息
    get_test_class_times()
    # 获取测试文件评级信息
    get_test_file_ratings()
    # 获取测试类评级信息
    get_test_class_ratings()
    # 获取历史编辑文件 JSON 的启发式决策信息
    get_td_heuristic_historial_edited_files_json()
    # 获取性能分析 JSON 的启发式决策信息
    get_td_heuristic_profiling_json()
    # 复制 pytest 缓存
    copy_pytest_cache()
    # 复制额外的先前失败
    copy_additional_previous_failures()

    # 获取测试优先级信息
    aggregated_heuristics = get_test_prioritizations(selected_tests)

    # 获取聚合优先级信息
    test_prioritizations = aggregated_heuristics.get_aggregated_priorities()

    # 打印聚合启发式信息
    print("Aggregated Heuristics")
    print(test_prioritizations.get_info_str(verbose=False))  # 输出详细的聚合优先级信息

    # 如果在持续集成环境中
    if os.getenv("CI") == "true":
        print("Emitting metrics")  # 输出正在发送指标信息
        # 发送测试决策结果的最终优先级信息
        emit_metric(
            "td_results_final_test_prioritizations",
            {"test_prioritizations": test_prioritizations.to_json()},
        )
        # 发送聚合启发式的信息
        emit_metric(
            "td_results_aggregated_heuristics",
            {"aggregated_heuristics": aggregated_heuristics.to_json()},
        )

    # 将测试优先级信息写入文件
    with open(REPO_ROOT / "td_results.json", "w") as f:
        f.write(json.dumps(test_prioritizations.to_json()))  # 将测试优先级对象转换为 JSON 格式并写入文件


# 程序入口点，如果直接运行该脚本则执行 main 函数
if __name__ == "__main__":
    main()
```