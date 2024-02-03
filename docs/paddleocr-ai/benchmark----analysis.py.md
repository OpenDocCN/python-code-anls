# `.\PaddleOCR\benchmark\analysis.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入未来的打印函数，确保代码同时兼容 Python 2 和 Python 3
from __future__ import print_function

# 导入解析命令行参数的模块
import argparse
# 导入处理 JSON 数据的模块
import json
# 导入操作系统相关的模块
import os
# 导入正则表达式模块
import re
# 导入异常追踪模块
import traceback

# 定义解析命令行参数的函数
def parse_args():
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser(description=__doc__)
    # 添加命令行参数，指定需要分析的日志文件名
    parser.add_argument(
        "--filename", type=str, help="The name of log which need to analysis.")
    # 添加命令行参数，指定带有性能分析器的训练日志路径
    parser.add_argument(
        "--log_with_profiler",
        type=str,
        help="The path of train log with profiler")
    # 添加命令行参数，指定性能分析器时间线日志的路径
    parser.add_argument(
        "--profiler_path", type=str, help="The path of profiler timeline log.")
    # 添加命令行参数，指定用于指定分析数据的关键字
    parser.add_argument(
        "--keyword", type=str, help="Keyword to specify analysis data")
    # 添加命令行参数，指定日志中不同字段的分隔符
    parser.add_argument(
        "--separator",
        type=str,
        default=None,
        help="Separator of different field in log")
    # 添加命令行参数，指定数据字段的位置
    parser.add_argument(
        '--position', type=int, default=None, help='The position of data field')
    # 添加命令行参数，指定要截取的数据字段范围
    parser.add_argument(
        '--range',
        type=str,
        default="",
        help='The range of data field to intercept')
    # 添加命令行参数，指定 GPU 上的基本批量大小
    parser.add_argument(
        '--base_batch_size', type=int, help='base_batch size on gpu')
    # 添加命令行参数，指定要跳过的步骤数
    parser.add_argument(
        '--skip_steps',
        type=int,
        default=0,
        help='The number of steps to be skipped')
    # 添加命令行参数，指定分析模式，默认值为 -1
    parser.add_argument(
        '--model_mode',
        type=int,
        default=-1,
        help='Analysis mode, default value is -1')
    # 添加命令行参数 --ips_unit，类型为字符串，默认为 None，帮助信息为 IPS unit
    parser.add_argument('--ips_unit', type=str, default=None, help='IPS unit')
    # 添加命令行参数 --model_name，类型为字符串，默认为 0，帮助信息为 training model_name, transformer_base
    parser.add_argument(
        '--model_name',
        type=str,
        default=0,
        help='training model_name, transformer_base')
    # 添加命令行参数 --mission_name，类型为字符串，默认为 0，帮助信息为 training mission name
    parser.add_argument(
        '--mission_name', type=str, default=0, help='training mission name')
    # 添加命令行参数 --direction_id，类型为整数，默认为 0，帮助信息为 training direction_id
    parser.add_argument(
        '--direction_id', type=int, default=0, help='training direction_id')
    # 添加命令行参数 --run_mode，类型为字符串，默认为 "sp"，帮助信息为 multi process or single process
    parser.add_argument(
        '--run_mode',
        type=str,
        default="sp",
        help='multi process or single process')
    # 添加命令行参数 --index，类型为整数，默认为 1，帮助信息为 {1: speed, 2:mem, 3:profiler, 6:max_batch_size}
    parser.add_argument(
        '--index',
        type=int,
        default=1,
        help='{1: speed, 2:mem, 3:profiler, 6:max_batch_size}')
    # 添加命令行参数 --gpu_num，类型为整数，默认为 1，帮助信息为 nums of training gpus
    parser.add_argument(
        '--gpu_num', type=int, default=1, help='nums of training gpus')
    # 解析命令行参数并返回结果
    args = parser.parse_args()
    # 如果参数 separator 的值为 "None"，则将其设置为 None，否则保持原值
    args.separator = None if args.separator == "None" else args.separator
    # 返回解析后的参数
    return args
# 判断输入的字符串是否为数字
def _is_number(num):
    # 编译正则表达式，匹配数字的模式
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    # 使用正则表达式匹配输入的字符串
    result = pattern.match(num)
    # 如果匹配成功，返回True；否则返回False
    if result:
        return True
    else:
        return False

# 定义一个时间分析器类
class TimeAnalyzer(object):
    # 初始化方法，接受文件名、关键字、分隔符、位置和范围等参数
    def __init__(self,
                 filename,
                 keyword=None,
                 separator=None,
                 position=None,
                 range="-1"):
        # 如果文件名为空，抛出异常
        if filename is None:
            raise Exception("Please specify the filename!")

        # 如果关键字为空，抛出异常
        if keyword is None:
            raise Exception("Please specify the keyword!")

        # 初始化实例变量
        self.filename = filename
        self.keyword = keyword
        self.separator = separator
        self.position = position
        self.range = range
        self.records = None
        # 调用内部方法_distil()进行处理
        self._distil()
    # 定义一个私有方法_distil，用于从文件中提取数据
    def _distil(self):
        # 初始化记录列表
        self.records = []
        # 打开文件并读取内容
        with open(self.filename, "r") as f_object:
            # 逐行读取文件内容
            lines = f_object.readlines()
            for line in lines:
                # 如果关键字不在当前行中，则跳过当前行
                if self.keyword not in line:
                    continue
                try:
                    result = None

                    # 从一行中提取字符串
                    line = line.strip()
                    # 根据分隔符拆分字符串，如果没有分隔符则按空格拆分
                    line_words = line.split(
                        self.separator) if self.separator else line.split()
                    # 如果指定了位置参数，则获取指定位置的字符串
                    if args.position:
                        result = line_words[self.position]
                    else:
                        # 在关键字后面提取字符串
                        for i in range(len(line_words) - 1):
                            if line_words[i] == self.keyword:
                                result = line_words[i + 1]
                                break

                    # 从选定的字符串中提取结果
                    if not self.range:
                        result = result[0:]
                    elif _is_number(self.range):
                        result = result[0:int(self.range)]
                    else:
                        result = result[int(self.range.split(":")[0]):int(
                            self.range.split(":")[1])]
                    # 将提取的结果转换为浮点数并添加到记录列表中
                    self.records.append(float(result))
                except Exception as exc:
                    # 捕获异常并打印相关信息
                    print("line is: {}; separator={}; position={}".format(
                        line, self.separator, self.position))

        # 打印提取的记录数量以及分隔符和位置参数
        print("Extract {} records: separator={}; position={}".format(
            len(self.records), self.separator, self.position))
    # 获取每秒处理的帧数（fps）和单位
    def _get_fps(self,
                 mode,
                 batch_size,
                 gpu_num,
                 avg_of_records,
                 run_mode,
                 unit=None):
        # 当 mode 为 -1 且运行模式为 'sp' 时
        if mode == -1 and run_mode == 'sp':
            # 断言 unit 不为空，否则抛出异常提示
            assert unit, "Please set the unit when mode is -1."
            # 计算 fps
            fps = gpu_num * avg_of_records
        # 当 mode 为 -1 且运行模式为 'mp' 时
        elif mode == -1 and run_mode == 'mp':
            # 断言 unit 不为空，否则抛出异常提示
            assert unit, "Please set the unit when mode is -1."
            # 计算 fps，暂时未使用
            fps = gpu_num * avg_of_records  #temporarily, not used now
            # 打印提示信息
            print("------------this is mp")
        # 当 mode 为 0 时
        elif mode == 0:
            # 计算 fps，将单位设置为 "samples/s"
            fps = (batch_size * gpu_num) / avg_of_records
            unit = "samples/s"
        # 当 mode 为 1 时
        elif mode == 1:
            # 将 fps 设置为 avg_of_records，单位设置为 "steps/s"
            fps = avg_of_records
            unit = "steps/s"
        # 当 mode 为 2 时
        elif mode == 2:
            # 计算 fps，将单位设置为 "steps/s"
            fps = 1 / avg_of_records
            unit = "steps/s"
        # 当 mode 为 3 时
        elif mode == 3:
            # 计算 fps，将单位设置为 "samples/s"
            fps = batch_size * gpu_num * avg_of_records
            unit = "samples/s"
        # 当 mode 为 4 时
        elif mode == 4:
            # 将 fps 设置为 avg_of_records，单位设置为 "s/epoch"
            fps = avg_of_records
            unit = "s/epoch"
        else:
            # 抛出异常，表示不支持的分析模式
            ValueError("Unsupported analysis mode.")

        # 返回计算得到的 fps 和单位
        return fps, unit
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 创建一个空字典用于存储运行信息
    run_info = dict()
    # 将命令行参数中的文件名存入字典
    run_info["log_file"] = args.filename
    # 将命令行参数中的模型名称存入字典
    run_info["model_name"] = args.model_name
    # 将命令行参数中的任务名称存入字典
    run_info["mission_name"] = args.mission_name
    # 将命令行参数中的方向 ID 存入字典
    run_info["direction_id"] = args.direction_id
    # 将命令行参数中的运行模式存入字典
    run_info["run_mode"] = args.run_mode
    # 将命令行参数中的索引存入字典
    run_info["index"] = args.index
    # 将命令行参数中的 GPU 数量存入字典
    run_info["gpu_num"] = args.gpu_num
    # 初始化最终结果为 0
    run_info["FINAL_RESULT"] = 0
    # 初始化任务失败标志为 0
    run_info["JOB_FAIL_FLAG"] = 0

    # 捕获异常并打印堆栈信息
    except Exception:
        traceback.print_exc()
    # 打印运行信息的 JSON 格式字符串，用于将日志文件路径插入数据库
    print("{}".format(json.dumps(run_info))
          )  # it's required, for the log file path  insert to the database
```