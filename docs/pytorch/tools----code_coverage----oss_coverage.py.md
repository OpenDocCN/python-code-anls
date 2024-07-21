# `.\pytorch\tools\code_coverage\oss_coverage.py`

```py
#!/usr/bin/env python3
# 导入时间模块
import time

# 导入必要的自定义模块（可能是外部包）
from package.oss.cov_json import get_json_report  # type: ignore[import]
from package.oss.init import initialization  # type: ignore[import]
from package.tool.summarize_jsons import summarize_jsons  # type: ignore[import]
from package.util.setting import TestPlatform  # type: ignore[import]
from package.util.utils import print_time  # type: ignore[import]


# 定义一个函数，无返回值
def report_coverage() -> None:
    # 记录程序开始执行的时间点
    start_time = time.time()
    # 调用初始化函数，获取返回的元组数据
    (options, test_list, interested_folders) = initialization()
    # 运行 CPP 测试，生成 JSON 报告
    get_json_report(test_list, options)
    # 如果需要生成汇总报告
    if options.need_summary:
        # 调用函数汇总 JSON 报告数据
        summarize_jsons(test_list, interested_folders, [""], TestPlatform.OSS)
    # 打印程序总运行时间
    print_time("Program Total Time: ", start_time)


# 如果当前脚本作为主程序运行，则执行 report_coverage 函数
if __name__ == "__main__":
    report_coverage()
```