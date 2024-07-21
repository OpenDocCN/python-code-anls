# `.\pytorch\tools\render_junit.py`

```py
#!/usr/bin/env python3
从 __future__ 导入注解

导入命令行参数解析模块 argparse
导入操作系统相关模块 os
从 typing 模块导入 Any 类型

尝试导入 junitparser 模块，引入 Error、Failure、JUnitXml、TestCase、TestSuite 类型
如果导入失败，抛出 ImportError 异常，并提示安装 junitparser
如果导入成功，则忽略类型检查错误

尝试导入 rich 模块，如果导入失败则打印提示信息

定义函数 parse_junit_reports，接收一个路径字符串 path_to_reports，返回一个 TestCase 对象列表
    定义局部函数 parse_file，接收一个路径字符串 path，返回一个 TestCase 对象列表
        尝试将指定路径的 JUnit XML 文件转换为 TestCase 对象列表并返回
        如果出现异常，则使用 rich 模块打印警告信息，并返回空列表

    如果指定的路径不存在，则抛出 FileNotFoundError 异常
    如果指定的路径是一个文件，则直接调用 parse_file 处理并返回结果
    初始化一个空列表 ret_xml
    如果指定的路径是一个目录，则遍历目录下的所有文件和子目录
        对于目录下以 "xml" 结尾的文件，调用 parse_file 处理并将结果追加到 ret_xml 中
    返回 ret_xml 列表

定义函数 convert_junit_to_testcases，接收一个 JUnitXml 或 TestSuite 对象，返回一个 TestCase 对象列表
    初始化空列表 testcases
    遍历 xml 对象
        如果是 TestSuite 类型，则递归调用 convert_junit_to_testcases 处理并将结果扩展到 testcases 中
        否则，将当前对象直接追加到 testcases 中
    返回 testcases 列表

定义函数 render_tests，接收一个 TestCase 对象列表，不返回值
    初始化 num_passed、num_skipped、num_failed 计数器为 0
    遍历 testcases 列表
        如果 testcase 没有 result 属性，则将 num_passed 计数器加一并继续下一个循环
        否则，遍历 result 列表
            根据 result 的类型（Error、Failure 或其他），设置不同的图标和计数器
            使用 rich 模块打印带有颜色标记的测试结果信息
            打印 result 的文本信息
    使用 rich 模块打印通过、跳过和失败的测试数量信息

定义函数 parse_args，不接收参数，返回任意类型
    创建 argparse.ArgumentParser 对象 parser，设置描述信息
    # 添加一个位置参数到参数解析器，该参数用于指定 xunit 测试报告的路径
    parser.add_argument(
        "report_path",
        help="Base xunit reports (single file or directory) to compare to",
    )
    # 解析命令行参数并返回解析结果
    return parser.parse_args()
# 定义主函数入口，程序的起点
def main() -> None:
    # 解析命令行参数，获取用户输入的选项
    options = parse_args()
    # 解析 JUnit 测试报告，从指定路径读取测试用例信息
    testcases = parse_junit_reports(options.report_path)
    # 渲染测试结果，将测试用例信息展示或处理
    render_tests(testcases)

# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 调用主函数 main()
    main()
```