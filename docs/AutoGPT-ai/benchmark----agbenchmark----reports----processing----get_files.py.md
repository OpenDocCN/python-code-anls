# `.\AutoGPT\benchmark\agbenchmark\reports\processing\get_files.py`

```py
import os

# 获取指定目录下最新的子目录
def get_last_subdirectory(directory_path: str) -> str | None:
    # 获取目录中的所有子目录
    subdirs = [
        os.path.join(directory_path, name)
        for name in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, name))
    ]

    # 按创建时间对子目录进行排序
    subdirs.sort(key=os.path.getctime)

    # 返回列表中的最后一个子目录
    return subdirs[-1] if subdirs else None

# 从代理目录中获取最新的报告
def get_latest_report_from_agent_directories(
    directory_path: str,
) -> list[tuple[os.DirEntry[str], str]]:
    latest_reports = []

    # 遍历目录中的所有条目
    for subdir in os.scandir(directory_path):
        if subdir.is_dir():
            # 获取该代理目录中最近创建的子目录
            latest_subdir = get_last_subdirectory(subdir.path)
            if latest_subdir is not None:
                # 在子目录中查找'report.json'文件
                report_file = os.path.join(latest_subdir, "report.json")
                if os.path.isfile(report_file):
                    latest_reports.append((subdir, report_file))

    return latest_reports
```