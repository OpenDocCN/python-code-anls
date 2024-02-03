# `.\AutoGPT\benchmark\agbenchmark\reports\processing\gen_combined_chart.py`

```py
import json
import os
from pathlib import Path

from agbenchmark.reports.processing.graphs import (
    save_combined_bar_chart,
    save_combined_radar_chart,
)
from agbenchmark.reports.processing.process_report import (
    all_agent_categories,
    get_reports_data,
)

# 定义生成组合图表的函数
def generate_combined_chart() -> None:
    # 获取所有代理报告的路径
    all_agents_path = Path(__file__).parent.parent.parent.parent / "reports"

    # 创建组合图表的文件夹
    combined_charts_folder = all_agents_path / "combined_charts"

    # 获取所有代理报告的数据
    reports_data = get_reports_data(str(all_agents_path))

    # 获取所有代理的类别
    categories = all_agent_categories(reports_data)

    # 统计此目录中的目录数量
    num_dirs = len([f for f in combined_charts_folder.iterdir() if f.is_dir()])

    # 创建运行图表的文件夹
    run_charts_folder = combined_charts_folder / f"run{num_dirs + 1}"

    # 如果运行图表的文件夹不存在，则创建
    if not os.path.exists(run_charts_folder):
        os.makedirs(run_charts_folder)

    # 从报告数据中提取信息数据
    info_data = {
        report_name: data.benchmark_start_time
        for report_name, data in reports_data.items()
        if report_name in categories
    }
    # 将信息数据写入 JSON 文件
    with open(Path(run_charts_folder) / "run_info.json", "w") as f:
        json.dump(info_data, f)

    # 保存组合雷达图表
    save_combined_radar_chart(categories, Path(run_charts_folder) / "radar_chart.png")
    # 保存组合柱状图表
    save_combined_bar_chart(categories, Path(run_charts_folder) / "bar_chart.png")


if __name__ == "__main__":
    # 调用生成组合图表的函数
    generate_combined_chart()
```