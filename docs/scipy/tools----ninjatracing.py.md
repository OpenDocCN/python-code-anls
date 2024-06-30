# `D:\src\scipysrc\scipy\tools\ninjatracing.py`

```
# 作者版权声明及许可信息
#
# 根据 Apache 许可证 2.0 版本授权，除非遵守许可证，否则不得使用此文件。
# 您可以在以下链接获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则分发的软件以"原样"方式分发，
# 没有任何明示或暗示的担保或条件。请参阅许可证了解详细信息。

"""将一个（或多个）.ninja_log文件转换为Chrome的about:tracing格式。

如果在生成的文件旁边发现clang -ftime-trace .json文件，则会嵌入它们。

用法：
    ninja -C $BUILDDIR
    python ninjatracing.py $BUILDDIR/.ninja_log > trace.json

然后将trace.json加载到Chrome或https://ui.perfetto.dev/中查看性能分析结果。
"""

import json
import os
import argparse
import re
import sys


class Target:
    """表示从.ninja_log文件中读取的单行。起始和结束时间以毫秒为单位。"""
    def __init__(self, start, end):
        self.start = int(start)
        self.end = int(end)
        self.targets = []


def read_targets(log, show_all):
    """从.ninja_log文件|log_file|中读取所有目标，按开始时间排序。"""
    header = log.readline()
    m = re.search(r'^# ninja log v(\d+)\n$', header)
    assert m, "unrecognized ninja log version %r" % header
    version = int(m.group(1))
    assert 5 <= version <= 6, "unsupported ninja log version %d" % version
    if version == 6:
        # 跳过头行
        next(log)

    targets = {}
    last_end_seen = 0
    for line in log:
        start, end, _, name, cmdhash = line.strip().split('\t')  # 忽略重新启动。
        if not show_all and int(end) < last_end_seen:
            # 较早的时间戳意味着这一步是新构建的第一步，可能是增量构建。丢弃先前的数据，
            # 以便此新构建将独立显示。
            targets = {}
        last_end_seen = int(end)
        targets.setdefault(cmdhash, Target(start, end)).targets.append(name)
    return sorted(targets.values(), key=lambda job: job.end, reverse=True)


class Threads:
    """试图从.ninja_log中重建并行性。"""
    def __init__(self):
        self.workers = []  # 将线程ID映射到线程占用时间。

    def alloc(self, target):
        """将目标放置在可用线程中，或添加一个新线程。"""
        for worker in range(len(self.workers)):
            if self.workers[worker] >= target.end:
                self.workers[worker] = target.start
                return worker
        self.workers.append(target.start)
        return len(self.workers) - 1


def read_events(trace, options):
    """从时间跟踪json文件|trace|中读取所有事件。"""
    trace_data = json.load(trace)
    # 定义一个函数 include_event，用于筛选事件数据
    def include_event(event, options):
        """Only include events if they are complete events, are longer than
        granularity, and are not totals."""
        # 返回条件：事件的 phase 字段为 'X'，持续时间大于等于指定的 granularity，
        # 并且事件名称不以 'Total' 开头
        return ((event['ph'] == 'X') and
                (event['dur'] >= options['granularity']) and
                (not event['name'].startswith('Total')))

    # 从 trace_data 中选择符合 include_event 函数条件的事件，并返回列表
    return [x for x in trace_data['traceEvents'] if include_event(x, options)]
def trace_to_dicts(target, trace, options, pid, tid):
    """Read a file-like object |trace| containing -ftime-trace data and yields
    about:tracing dict per eligible event in that log."""
    # 遍历从trace中读取的事件
    for event in read_events(trace, options):
        # 计算从ninja日志中获得的时间间隔
        ninja_time = (target.end - target.start) * 1000
        # 检查事件的持续时间是否大于ninja时间
        if event['dur'] > ninja_time:
            # 若持续时间不一致，则打印警告信息并退出程序
            print("Inconsistent timing found (clang time > ninja time). Please"
                  " ensure that timings are from consistent builds.")
            sys.exit(1)

        # 设置事件的pid和tid为从ninja日志中获得的值
        event['pid'] = pid
        event['tid'] = tid

        # 将事件的时间戳偏移至ninja启动时间
        event['ts'] += (target.start * 1000)

        # 生成当前事件的字典表示
        yield event


def embed_time_trace(ninja_log_dir, target, pid, tid, options):
    """Produce time trace output for the specified ninja target. Expects
    time-trace file to be in .json file named based on .o file."""
    # 遍历目标中的每个目标
    for t in target.targets:
        # 构建.o文件的路径
        o_path = os.path.join(ninja_log_dir, t)
        # 构建.json文件的路径
        json_trace_path = os.path.splitext(o_path)[0] + '.json'
        try:
            # 尝试打开.json文件作为追踪文件
            with open(json_trace_path) as trace:
                # 生成从追踪文件中读取的事件字典
                yield from trace_to_dicts(target, trace, options, pid, tid)
        except OSError:
            # 如果打开文件失败，则继续下一个目标
            pass


def log_to_dicts(log, pid, options):
    """Reads a file-like object |log| containing a .ninja_log, and yields one
    about:tracing dict per command found in the log."""
    # 初始化线程管理器
    threads = Threads()
    # 遍历从日志中读取的每个目标
    for target in read_targets(log, options['showall']):
        # 为当前目标分配一个线程ID
        tid = threads.alloc(target)

        # 生成关于当前目标的字典表示
        yield {
            'name': '%0s' % ', '.join(target.targets), 'cat': 'targets',
            'ph': 'X', 'ts': (target.start * 1000),
            'dur': ((target.end - target.start) * 1000),
            'pid': pid, 'tid': tid, 'args': {},
            }
        
        # 如果选项中包含embed_time_trace，尝试将时间追踪信息嵌入到ninja追踪中
        if options.get('embed_time_trace', False):
            try:
                # 获取ninja日志的目录
                ninja_log_dir = os.path.dirname(log.name)
            except AttributeError:
                # 如果获取失败，则继续下一个目标
                continue
            # 生成嵌入的时间追踪信息的事件字典
            yield from embed_time_trace(ninja_log_dir, target, pid, tid, options)


def main(argv):
    usage = __doc__
    # 解析命令行参数
    parser = argparse.ArgumentParser(usage)
    parser.add_argument("logfiles", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument('-a', '--showall', action='store_true',
                        dest='showall', default=False,
                        help='report on last build step for all outputs. '
                              'Default is to report just on the last '
                              '(possibly incremental) build')
    parser.add_argument('-g', '--granularity', type=int, default=50000,
                        dest='granularity',
                        help='minimum length time-trace event to embed in '
                             'microseconds. Default: 50000')
    # 添加一个布尔类型的命令行参数，用于指定是否嵌入 clang 的 -ftime-trace JSON 文件
    parser.add_argument('-e', '--embed-time-trace', action='store_true',
                        default=False, dest='embed_time_trace',
                        help='embed clang -ftime-trace json file found '
                             'adjacent to a target file')
    
    # 解析命令行参数，并将结果存储在 options 变量中
    options = parser.parse_args()

    # 初始化一个空列表用于存储日志条目
    entries = []
    
    # 遍历命令行参数中指定的每个日志文件及其对应的进程 ID
    for pid, log_file in enumerate(options.logfiles):
        # 打开日志文件
        with open(log_file) as log:
            # 调用 log_to_dicts 函数将日志文件转换为字典形式的条目，并将结果合并到 entries 列表中
            entries += list(log_to_dicts(log, pid, vars(options)))
    
    # 将 entries 列表中的字典数据序列化为 JSON 格式，并输出到标准输出流
    json.dump(entries, sys.stdout)
# 如果当前脚本作为主程序执行（而非被导入），则执行以下代码
if __name__ == '__main__':
    # 调用 main 函数，并将命令行参数列表传递给它，main 函数的返回值作为 sys.exit 的参数
    sys.exit(main(sys.argv[1:]))
```