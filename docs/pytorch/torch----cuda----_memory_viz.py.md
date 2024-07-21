# `.\pytorch\torch\cuda\_memory_viz.py`

```py
# 导入必要的模块
import pickle  # 用于序列化和反序列化 Python 对象
import sys  # 提供对 Python 解释器的访问和控制
import os  # 提供与操作系统交互的功能
import io  # 提供基本的 I/O 操作支持
import subprocess  # 用于启动新进程、连接它们的输入、输出和错误管道，并获取它们的返回码
import json  # 用于处理 JSON 数据的编码和解码
from functools import lru_cache  # 提供缓存函数的装饰器，支持最近最少使用 (LRU) 策略
from typing import Any  # 提供静态类型检查支持
from itertools import groupby  # 提供对序列进行分组操作的功能
import base64  # 提供对 base64 编码和解码的支持
import warnings  # 提供警告管理工具
import operator  # 提供 Python 中各种内置操作符的函数形式

# 使用 lru_cache 装饰器缓存函数的结果，不限制缓存大小
cache = lru_cache(None)

# 定义导出的模块接口
__all__ = ["format_flamegraph", "segments", "memory", "compare"]

# 格式化单个帧的输出格式
def _frame_fmt(f, full_filename=False):
    i = f['line']  # 获取帧的行号
    fname = f['filename']  # 获取帧的文件名
    if not full_filename:
        fname = fname.split('/')[-1]  # 如果不需要完整文件路径，只获取文件名部分
    func = f['name']  # 获取帧的函数名
    return f'{fname}:{i}:{func}'  # 返回格式化后的字符串

# 使用缓存过滤帧的函数名和文件名
@cache
def _frame_filter(name, filename):
    omit_functions = [  # 需要忽略的函数名列表
        "unwind::unwind",
        "CapturedTraceback::gather",
        "gather_with_cpp",
        "_start",
        "__libc_start_main",
        "PyEval_",
        "PyObject_",
        "PyFunction_",
    ]
    omit_filenames = [  # 需要忽略的文件名列表
        "core/boxing",
        "/Register",
        "/Redispatch",
        "pythonrun.c",
        "Modules/main.c",
        "Objects/call.c",
        "Objects/methodobject.c",
        "pycore_ceval.h",
        "ceval.c",
        "cpython/abstract.h",
    ]
    for of in omit_functions:
        if of in name:
            return False  # 如果函数名在忽略列表中，则返回 False
    for of in omit_filenames:
        if of in filename:
            return False  # 如果文件名在忽略列表中，则返回 False
    return True  # 否则返回 True，表示不需要过滤

# 格式化帧列表的输出格式，根据需要反向排序
def _frames_fmt(frames, full_filename=False, reverse=False):
    if reverse:
        frames = reversed(frames)  # 如果需要反向排序，则反转帧列表
    return [_frame_fmt(f, full_filename) for f in frames if _frame_filter(f['name'], f['filename'])]

# 处理旧版快照格式的块信息，获取帧和实际大小
def _block_extra_legacy(b):
    if 'history' in b:
        frames = b['history'][0].get('frames', [])  # 获取历史记录中的帧信息
        real_size = b['history'][0]['real_size']  # 获取历史记录中的实际大小
    else:
        real_size = b.get('requested_size', b['size'])  # 否则获取请求的大小或默认大小
        frames = []  # 如果没有帧信息，则为空列表
    return frames, real_size  # 返回帧列表和实际大小

# 获取块信息的帧和请求大小
def _block_extra(b):
    if 'frames' not in b:
        # 旧版快照格式处理更复杂，需要调用 _block_extra_legacy 处理
        return _block_extra_legacy(b)
    return b['frames'], b['requested_size']  # 返回帧列表和请求的大小

# 格式化火焰图输出
def format_flamegraph(flamegraph_lines, flamegraph_script=None):
    if flamegraph_script is None:
        flamegraph_script = f'/tmp/{os.getuid()}_flamegraph.pl'  # 默认使用临时文件路径
    if not os.path.exists(flamegraph_script):
        import urllib.request
        print(f"Downloading flamegraph.pl to: {flamegraph_script}")  # 下载火焰图脚本的消息提示
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl', flamegraph_script)  # 下载火焰图脚本
        subprocess.check_call(['chmod', '+x', flamegraph_script])  # 修改下载的脚本文件权限为可执行
    args = [flamegraph_script, '--countname', 'bytes']  # 设置火焰图脚本的参数
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf-8')  # 启动子进程执行火焰图脚本
    assert p.stdin is not None
    assert p.stdout is not None
    p.stdin.write(flamegraph_lines)  # 将火焰图数据写入子进程的标准输入
    p.stdin.close()  # 关闭标准输入
    result = p.stdout.read()  # 读取子进程的标准输出结果
    p.stdout.close()  # 关闭标准输出
    p.wait()  # 等待子进程执行完成
    assert p.wait() == 0  # 检查子进程返回码，确保正常执行
    return result  # 返回火焰图处理结果

# 写入块信息的辅助函数，用于处理旧版快照格式
def _write_blocks(f, prefix, blocks):
    # 对给定的 blocks 进行迭代处理
    for b in blocks:
        # 如果块 b 中不包含 'history' 键
        if 'history' not in b:
            # 调用 _block_extra 函数获取 frames 和 accounted_for_size
            frames, accounted_for_size = _block_extra(b)
            # 写入文件 f，格式为 prefix;state;frames_fragment(accounted_for_size)\n
            f.write(f'{prefix};{b["state"]};{frames_fragment(frames)} {accounted_for_size}\n')
        else:
            # 初始化 accounted_for_size 为 0
            accounted_for_size = 0
            # 遍历块 b 中的 'history' 列表
            for h in b['history']:
                # 获取当前历史记录 h 的实际大小
                sz = h['real_size']
                # 累加到 accounted_for_size 中
                accounted_for_size += sz
                # 如果当前历史记录 h 中包含 'frames' 键
                if 'frames' in h:
                    # 获取 frames
                    frames = h['frames']
                    # 写入文件 f，格式为 prefix;state;frames_fragment(frames) sz\n
                    f.write(f'{prefix};{b["state"]};{frames_fragment(frames)} {sz}\n')
                else:
                    # 如果当前历史记录 h 不包含 'frames' 键，写入文件 f，格式为 prefix;state;<no-context> sz\n
                    f.write(f'{prefix};{b["state"]};<no-context> {sz}\n')
        # 计算缺口大小
        gaps = b['size'] - accounted_for_size
        # 如果存在缺口 gaps
        if gaps:
            # 写入文件 f，格式为 prefix;state;<gaps> gaps\n
            f.write(f'{prefix};{b["state"]};<gaps> {gaps}\n')
# 定义函数 segments，接受一个快照参数和一个格式化函数，默认为 format_flamegraph
def segments(snapshot, format_flamegraph=format_flamegraph):
    # 创建一个 String I/O 对象 f，用于存储数据
    f = io.StringIO()
    # 遍历快照中的每个段落
    for seg in snapshot['segments']:
        # 创建段落的前缀，格式为'stream_{seg["stream"]};seg_{seg["address"]}'
        prefix = f'stream_{seg["stream"]};seg_{seg["address"]}'
        # 调用 _write_blocks 函数，将段落的块信息写入到 f 中
        _write_blocks(f, prefix, seg['blocks'])
    # 使用 format_flamegraph 函数对 f 中的数据进行格式化并返回结果
    return format_flamegraph(f.getvalue())

# 定义函数 memory，接受一个快照参数和一个格式化函数，默认为 format_flamegraph
def memory(snapshot, format_flamegraph=format_flamegraph):
    # 创建一个 String I/O 对象 f，用于存储数据
    f = io.StringIO()
    # 遍历快照中的每个段落
    for seg in snapshot['segments']:
        # 创建段落的前缀，格式为'stream_{seg["stream"]}'
        prefix = f'stream_{seg["stream"]}'
        # 调用 _write_blocks 函数，将段落的块信息写入到 f 中
        _write_blocks(f, prefix, seg['blocks'])
    # 使用 format_flamegraph 函数对 f 中的数据进行格式化并返回结果
    return format_flamegraph(f.getvalue())

# 定义函数 compare，接受两个快照参数和一个格式化函数，默认为 format_flamegraph
def compare(before, after, format_flamegraph=format_flamegraph):
    # 定义内部函数 _seg_key，返回一个元组 (seg['address'], seg['total_size'])
    def _seg_key(seg):
        return (seg['address'], seg['total_size'])

    # 定义内部函数 _seg_info，返回格式为 'stream_{seg["stream"]};seg_{seg["address"]}' 的字符串
    def _seg_info(seg):
        return f'stream_{seg["stream"]};seg_{seg["address"]}'

    # 创建一个 String I/O 对象 f，用于存储数据
    f = io.StringIO()

    # 计算 before 和 after 中段落的集合
    before_segs = {_seg_key(seg) for seg in before}
    after_segs = {_seg_key(seg) for seg in after}

    # 打印仅在 before 中出现的段落的地址列表
    print(f'only_before = {[a for a, _ in (before_segs - after_segs)]}')
    # 打印仅在 after 中出现的段落的地址列表
    print(f'only_after = {[a for a, _ in (after_segs - before_segs)]}')

    # 遍历 before 中的段落
    for seg in before:
        # 如果段落不在 after_segs 集合中，则将其信息写入到 f 中
        if _seg_key(seg) not in after_segs:
            _write_blocks(f, f'only_before;{_seg_info(seg)}', seg['blocks'])

    # 遍历 after 中的段落
    for seg in after:
        # 如果段落不在 before_segs 集合中，则将其信息写入到 f 中
        if _seg_key(seg) not in before_segs:
            _write_blocks(f, f'only_after;{_seg_info(seg)}', seg['blocks'])

    # 使用 format_flamegraph 函数对 f 中的数据进行格式化并返回结果
    return format_flamegraph(f.getvalue())

# 定义内部函数 _format_size，用于将字节数转换为人类可读的格式
def _format_size(num):
    # 根据字节数确定合适的单位进行显示，最高支持到 ZiB 级别
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}B"
        num /= 1024.0
    return f"{num:.1f}YiB"

# 定义类 Bytes
class Bytes:
    # 初始化方法，接受一个 value 参数
    def __init__(self, value):
        self.value = value

    # 重载 + 运算符，用于计算字节数的和
    def __add__(self, rhs):
        return Bytes(self.value + rhs)

    # 重载 repr 方法，用于打印 Bytes 对象的字符串表示，调用 _format_size 方法
    def __repr__(self):
        return _format_size(self.value)

# 定义常量 PAGE_SIZE，表示页面大小为 20MB
PAGE_SIZE = 1024 * 1024 * 20

# 定义 legend 变量，存储描述信息，包括页面大小的格式化输出
legend = f"""\

Legend:
    [a     ] - a segment in the allocator
     ^-- a page {Bytes(PAGE_SIZE)} of memory in the segment
    a-z: pages filled with a single block's content
    ' ': page is completely free
    *: page if completely full with multiple blocks
    0-9: page is partially full with tensors of multiple blocks (9 == 90% full)
    (X% internal) - of the free memory, X% is free because we rounded the size of the allocation.
"""

# 定义函数 segsum，接受一个数据参数
def segsum(data):
    r"""Visually reports how the allocator has filled its segments.

    This printout can help debug fragmentation issues since free fragments
    will appear as gaps in this printout.  The amount of free space is reported
    for each segment.
    """
    We distinguish between internal free memory which occurs because the
    allocator rounds the allocation size, and external free memory, which are
    the gaps between allocations in a segment.
    Args:
        data: snapshot dictionary created from _snapshot()
    """
    segments = []  # 初始化一个空列表用于存储处理后的段信息
    out = io.StringIO()  # 创建一个字符串IO对象，用于构建输出内容
    out.write(f"Summary of segments >= {Bytes(PAGE_SIZE)} in size\n")  # 输出段大小大于等于PAGE_SIZE的段的总结信息
    total_reserved = 0  # 初始化总保留内存大小计数器
    total_allocated = 0  # 初始化总分配内存大小计数器
    free_external = 0  # 初始化外部空闲内存计数器
    free_internal = 0  # 初始化内部空闲内存计数器

    # 遍历所有段，按照总大小和活跃状态排序
    for seg in sorted(data['segments'], key=lambda x: (x['total_size'], calc_active(x))):
        total_reserved += seg['total_size']  # 累加总保留内存大小

        seg_free_external = 0  # 初始化段的外部空闲内存大小计数器
        seg_free_internal = 0  # 初始化段的内部空闲内存大小计数器
        seg_allocated = 0  # 初始化段的分配内存大小计数器
        all_ranges = []  # 初始化用于存储所有内存块范围的列表
        boffset = 0  # 初始化内存块偏移量

        # 遍历段内的所有内存块
        for b in seg['blocks']:
            active = b['state'] == 'active_allocated'  # 检查内存块是否处于活跃分配状态
            if active:
                _, allocated_size = _block_extra(b)  # 获取额外分配的内存大小
                all_ranges.append((boffset, allocated_size, True))  # 将活跃分配的内存块范围添加到列表
                seg_allocated += allocated_size  # 累加段的分配内存大小
                seg_free_internal += b['size'] - allocated_size  # 计算段的内部空闲内存大小
            else:
                seg_free_external += b['size']  # 计算段的外部空闲内存大小

            boffset += b['size']  # 更新内存块偏移量

        total_allocated += seg_allocated  # 累加总分配内存大小
        free_external += seg_free_external  # 累加外部空闲内存大小
        free_internal += seg_free_internal  # 累加内部空闲内存大小

        nseg = (seg['total_size'] - 1) // PAGE_SIZE + 1  # 计算段的页数
        occupied = [' ' for _ in range(nseg)]  # 初始化用于标记页被占用情况的列表
        frac = [0.0 for _ in range(nseg)]  # 初始化用于存储每页被占用比例的列表
        active_size = 0  # 初始化活跃内存大小计数器
        for i, (start_, size, active) in enumerate(all_ranges):
            active_size += size  # 累加活跃内存大小
            finish_ = (start_ + size)
            start = start_ // PAGE_SIZE
            finish = (finish_ - 1) // PAGE_SIZE + 1
            m = chr(ord('a' if active else 'A') + (i % 26))  # 生成用于表示内存块的字符
            for j in range(start, finish):
                s = max(start_, j * PAGE_SIZE)
                e = min(finish_, (j + 1) * PAGE_SIZE)
                frac[j] += (e - s) / PAGE_SIZE  # 更新页被占用比例
                if occupied[j] != ' ':
                    occupied[j] = '0123456789*'[int(frac[j] * 10)]  # 根据被占用比例设置页的占用标记
                else:
                    occupied[j] = m

        stream = '' if seg['stream'] == 0 else f', stream_{seg["stream"]}'  # 根据段的流信息设置输出流标记
        body = ''.join(occupied)  # 将页的占用情况列表转换为字符串
        assert seg_free_external + seg_free_internal + seg_allocated == seg['total_size']  # 检查段的内存统计是否正确
        stream = f' stream_{seg["stream"]}' if seg['stream'] != 0 else ''  # 根据段的流信息设置输出流标记
        if seg['total_size'] >= PAGE_SIZE:
            out.write(f'[{body}] {Bytes(seg["total_size"])} allocated, '
                      f'{_report_free(seg_free_external, seg_free_internal)} free{stream}\n')  # 输出段的总结信息

    out.write(f'segments: {len(data["segments"])}\n')  # 输出处理过的段的总数
    out.write(f'total_reserved: {Bytes(total_reserved)}\n')  # 输出总保留内存大小
    out.write(f'total_allocated: {Bytes(total_allocated)}\n')  # 输出总分配内存大小
    internal_external = f' ({Bytes(free_internal)} internal + {Bytes(free_external)} external)' if free_internal else ''  # 计算并设置内部和外部空闲内存大小的输出格式
    out.write(f'total_free: {_report_free(free_external, free_internal)}\n')  # 输出总空闲内存信息
    out.write(legend)  # 输出说明信息
    # 断言语句，用于确保自由内存、外部自由内存和总分配量之和等于总保留量
    assert free_internal + free_external + total_allocated == total_reserved
    # 返回对象 `out` 的字节表示
    return out.getvalue()
def trace(data):
    # 创建一个StringIO对象，用于存储输出结果
    out = io.StringIO()

    # 遍历传入数据中的设备追踪信息
    for i, d in enumerate(data['device_traces']):
        if d:
            # 将设备追踪信息写入输出对象
            out.write(f'Device {i} ----------------\n')
            # 调用format函数对数据进行格式化（缺失实现细节）
            format(d)

    # 返回StringIO对象中的所有内容作为字符串
    return out.getvalue()


_memory_viz_template = r"""
<!DOCTYPE html>
<html>
<head>
</head>
<body>
<script type="module">
import {add_local_files} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"
const local_files = $SNAPSHOT
add_local_files(local_files, $VIZ_KIND)
</script>
</body>
"""

def _format_viz(data, viz_kind, device):
    # 如果device参数不为None，则发出警告，指出该参数已经不推荐使用
    if device is not None:
        warnings.warn(
            'device argument is deprecated, plots now contain all device',
            FutureWarning,
            stacklevel=3,
        )
    
    # 将数据序列化为pickle格式的字节流
    buffer = pickle.dumps(data)
    buffer += b'\x00' * (3 - len(buffer) % 3)
    # 使用base64编码对字节流进行编码
    encoded_buffer = base64.b64encode(buffer).decode('utf-8')

    # 将数据格式化为JSON格式
    json_format = json.dumps([{"name": 'snapshot.pickle', "base64": encoded_buffer}])
    
    # 使用_memory_viz_template模板替换其中的占位符$VIZ_KIND和$SNAPSHOT
    return _memory_viz_template.replace('$VIZ_KIND', repr(viz_kind)) \
                               .replace('$SNAPSHOT', json_format)

def trace_plot(data, device=None, plot_segments=False):
    """Generate a visualization over time of the memory usage recorded by the trace as an html file.

    Args:
        data: Memory snapshot as generated from torch.cuda.memory._snapshot()
        device (torch.device, optional): Generate the trace for this device, needed if multiple devices have allocations.
        plot_segments (bool, optional): Plots memory returned from cudaMalloc, rather than individual allocations.
                                        Defaults to False.

    Returns:
        str: HTML of visualization
    """
    # 调用_format_viz函数生成并返回内存使用可视化的HTML内容
    return _format_viz(data, 'Active Memory Timeline' if not plot_segments else 'Active Cached Memory Timeline', device)


def _profile_to_snapshot(profile):
    import torch
    from torch.profiler._memory_profiler import Action, TensorKey
    from torch._C._profiler import _EventType
    
    # 获取内存分析器的内存分析结果
    memory_profile = profile._memory_profile()

    # 创建一个空字典，用于存储分配栈信息
    allocation_stacks = {}

    # 遍历内存分析结果中的事件节点
    for event in memory_profile._op_tree.sorted_nodes:
        # 如果事件类型为Allocation
        if event.tag == _EventType.Allocation:
            parent = event.parent
            python_parents = []
            
            # 寻找Python调用事件的父节点
            while parent:
                if parent.tag in (_EventType.PyCall, _EventType.PyCCall):
                    python_parents.append(parent)
                parent = parent.parent
            
            # 根据事件附加字段创建TensorKey对象
            key = TensorKey.from_allocation(event.extra_fields)

            # 特殊情况处理：如果Allocation事件没有ID或分配大小为0，则跳过
            if key and event.extra_fields.alloc_size > 0:
                allocation_stacks[key] = python_parents

    # 获取当前系统中的CUDA设备数量
    device_count = torch.cuda.device_count()
    # 创建一个初始的快照状态，包括设备追踪和段信息
    snapshot = {
        'device_traces': [[] for _ in range(device_count + 1)],  # 为每个设备创建空列表的列表，用于记录设备的内存分配追踪
        'segments': [{'device': device,
                      'address': None,
                      'total_size': 0,
                      'stream': 0,
                      'blocks': []} for device in range(device_count + 1)]  # 为每个设备创建初始段信息，每个段有设备号、地址、总大小、流和块列表
    }

    # 将设备类型转换为设备索引或默认设备数
    def to_device(device):
        if device.type == 'cuda':
            return device.index  # 如果设备类型为 'cuda'，返回其索引号
        else:
            return device_count  # 否则返回默认设备数

    # 分配内存给张量并记录分配的细节
    def allocate(size, tensor_key, version, during_trace=True):
        device = to_device(tensor_key.device)  # 获取张量所在的设备索引
        addr = tensor_key.storage.ptr  # 获取张量的存储指针地址

        seg = snapshot['segments'][device]  # 获取设备对应的段信息
        if seg['address'] is None or seg['address'] > addr:
            seg['address'] = addr  # 如果段的地址为空或大于当前张量的地址，则更新段的起始地址
        seg['total_size'] = max(seg['total_size'], addr + size)  # 记录当前段的最大地址加上张量大小
        category = memory_profile._categories.get(tensor_key, version)  # 获取张量的内存类别
        category = category.name.lower() if category is not None else "unknown"  # 将内存类别转换为小写字符串，若无则为 "unknown"
        stack = allocation_stacks.get(tensor_key, ())  # 获取张量的分配栈信息
        stack = [{'filename': 'none', 'line': 0, 'name': p.name} for p in stack]  # 将栈中的函数信息转换为包含文件名、行号和函数名的字典列表
        r = {'action': 'alloc', 'addr': addr, 'size': size, 'stream': 0, 'frames': stack, 'category': category}  # 创建分配操作的事件记录
        if during_trace:
            snapshot['device_traces'][device].append(r)  # 将分配事件记录添加到对应设备的追踪列表中
        return r  # 返回分配的事件记录

    # 释放内存并记录释放的细节
    def free(alloc, device):
        for e in ('free_requested', 'free_completed'):
            snapshot['device_traces'][device].append({'action': e,  # 记录释放操作的事件类型
                                                      'addr': alloc['addr'],  # 记录被释放的内存地址
                                                      'size': alloc['size'],  # 记录被释放的内存大小
                                                      'stream': 0,
                                                      'frames': alloc['frames']})  # 记录释放操作对应的调用栈

    kv_to_elem = {}

    # 遍历内存分配的时间线，根据不同的操作类型进行相应的处理
    for time, action, (tensor_key, version), size in memory_profile.timeline:
        if not isinstance(tensor_key, TensorKey):
            continue  # 如果键不是张量键类型，则跳过
        if action == Action.CREATE:
            kv_to_elem[(tensor_key, version)] = allocate(size, tensor_key, version)  # 执行分配操作并将分配的事件记录加入到字典中
        elif action == Action.DESTROY:
            free(kv_to_elem.pop((tensor_key, version)), to_device(tensor_key.device))  # 执行释放操作并将释放的事件记录加入到设备追踪中
        elif action == Action.INCREMENT_VERSION:
            free(kv_to_elem.pop((tensor_key, version)), to_device(tensor_key.device))  # 执行释放操作并将释放的事件记录加入到设备追踪中
            kv_to_elem[(tensor_key, version + 1)] = allocate(size, tensor_key, version + 1)  # 执行增加版本操作并将分配的事件记录加入到字典中
        elif action == Action.PREEXISTING:
            kv_to_elem[(tensor_key, version)] = allocate(size, tensor_key, version, during_trace=False)  # 执行预存在操作并将分配的事件记录加入到字典中

    # 创建最终的快照状态，包含所有设备上的块信息
    blocks_at_end = [(to_device(tensor_key.device), event['addr'], event['size'], event['frames'])
                     for (tensor_key, version), event in kv_to_elem.items()]  # 将最终分配的块信息存储为设备、地址、大小和帧的列表
    # 根据 'blocks_at_end' 中设备分组并排序，遍历每个设备及其关联的块列表
    for device, blocks in groupby(sorted(blocks_at_end), key=operator.itemgetter(0)):
        # 获取快照中该设备对应的段信息
        seg = snapshot['segments'][device]  # type: ignore[index]
        # 初始化上一个块的地址为该段的起始地址
        last_addr = seg['address']
        # 遍历该设备的每个块信息：地址、大小、帧列表
        for _, addr, size, frames in blocks:
            # 如果上一个块的地址小于当前块的地址，则添加一个不活跃的块到段中
            if last_addr < addr:
                seg['blocks'].append({'size': addr - last_addr, 'state': 'inactive'})
            # 添加当前块到段中，标记为活跃已分配，包括请求大小和帧列表
            seg['blocks'].append({'size': size, 'state': 'active_allocated', 'requested_size': size, 'frames': frames})
            # 更新上一个块的地址为当前块结束后的地址
            last_addr = addr + size
        # 如果最后一个块后面还有空闲空间，则添加一个不活跃的块到段中
        if last_addr < seg['total_size']:
            seg['blocks'].append({'size': seg['total_size'] - last_addr, 'state': 'inactive'})

    # 过滤掉不包含块的段，并更新快照中的段列表
    snapshot['segments'] = [seg for seg in snapshot['segments'] if seg['blocks']]  # type: ignore[attr-defined]

    # 遍历快照中的每个段，更新其总大小并检查是否有未使用的段
    for seg in snapshot['segments']:  # type: ignore[attr-defined, name-defined, no-redef]
        seg['total_size'] -= seg['address']
        # 如果段中没有块，则添加一个完整大小的不活跃块
        if not seg['blocks']:
            seg['blocks'].append({'size': seg['total_size'], 'state': 'inactive'})

    # 返回更新后的快照
    return snapshot
if __name__ == "__main__":
    # 获取当前脚本所在目录的绝对路径
    import os.path
    thedir = os.path.realpath(os.path.dirname(__file__))
    # 如果当前目录在系统路径中，移除它，以避免冲突
    if thedir in sys.path:
        sys.path.remove(thedir)

    # 导入参数解析模块
    import argparse

    # 定义函数名称字符串
    fn_name = 'torch.cuda.memory._snapshot()'
    # 创建包含函数名称的序列化内存统计的字符串
    pickled = f'pickled memory statistics from {fn_name}'

    # 创建参数解析对象，描述为根据函数名称生成的内存转储可视化
    parser = argparse.ArgumentParser(description=f'Visualize memory dumps produced by {fn_name}')

    # 添加子命令解析器
    subparsers = parser.add_subparsers(dest='action')

    # 定义内存统计子命令及其描述
    def _output(p):
        p.add_argument('-o', '--output', default='output.svg', help='flamegraph svg (default: output.svg)')

    # 定义各子命令的描述
    description = 'Prints overall allocation statistics and a visualization of how the allocators segments are currently filled.'
    stats_a = subparsers.add_parser('stats', description=description)
    stats_a.add_argument('input', help=pickled)

    description = 'Prints buffer of the most recent allocation events embedded in the snapshot in a Pythonic style.'
    trace_a = subparsers.add_parser('trace', description=description)
    trace_a.add_argument('input', help=pickled)

    description = 'Generate a flamegraph that visualizes what memory is stored in each allocator segment (aka block)'
    segments_a = subparsers.add_parser('segments', description=description)
    segments_a.add_argument('input', help=pickled)
    _output(segments_a)

    description = "Generate a flamegraph the program locations contributing to CUDA memory usage."
    memory_a = subparsers.add_parser('memory', description=description)
    memory_a.add_argument('input', help=pickled)
    _output(memory_a)

    description = 'Generate a flamegraph that shows segments (aka blocks) that have been added ' \
        'or removed between two different memorys snapshots.'
    compare_a = subparsers.add_parser('compare', description=description)
    compare_a.add_argument('before', help=pickled)
    compare_a.add_argument('after', help=pickled)
    _output(compare_a)

    # 定义多个绘图任务的元组
    plots = (
        ("trace_plot", "Generate a visualization over time of the memory usage recorded by the trace as an html file."),
        ("segment_plot", "Visualize how allocations are packed into allocator segments at each point in a trace as an html file.")
    )
    # 遍历plots列表中的每对(cmd, description)，cmd是命令，description是描述
    for cmd, description in plots:
        # 添加一个子命令解析器，使用cmd作为命令名称，description作为帮助描述
        trace_plot_a = subparsers.add_parser(cmd, description=description)
        # 添加一个参数，用于指定输入文件的路径，帮助信息为pickled
        trace_plot_a.add_argument('input', help=pickled)
        # 设备编号参数，可选，帮助信息为默认选择具有跟踪信息或错误的唯一设备
        help = 'visualize trace from this device (default: chooses the only device with trace info or errors)'
        trace_plot_a.add_argument('-d', '--device', type=int, default=None, help=help)
        # 输出文件路径参数，默认为'output.html'，帮助信息为保存可视化结果的路径
        help = 'path to save the visualization(default: output.html)'
        trace_plot_a.add_argument('-o', '--output', default='output.html', help=help)
        # 如果命令是"trace_plot"，添加一个选项参数，用于可视化变更到段而不是单个分配
        if cmd == "trace_plot":
            help = 'visualize change to segments rather than individual allocations'
            trace_plot_a.add_argument('-s', '--segments', action='store_true', help=help)

    # 解析命令行参数
    args = parser.parse_args()

    # 定义一个函数_read，根据给定的文件名读取数据
    def _read(name):
        # 如果name为'-'，从标准输入读取数据
        if name == '-':
            f = sys.stdin.buffer
        else:
            # 否则打开以二进制模式读取的文件
            f = open(name, 'rb')
        # 使用pickle加载文件中的数据
        data = pickle.load(f)
        # 如果数据是列表，则假设仅包含段...
        if isinstance(data, list):  # segments only...
            data = {'segments': data, 'traces': []}
        return data

    # 定义一个函数_write，将数据写入指定的文件名
    def _write(name, data):
        # 使用写模式打开文件
        with open(name, 'w') as f:
            # 将数据写入文件
            f.write(data)

    # 根据args中的action字段执行相应的操作
    if args.action == 'segments':
        # 读取输入数据
        data = _read(args.input)
        # 执行segments函数并将结果写入输出文件
        _write(args.output, segments(data))
    elif args.action == 'memory':
        # 读取输入数据
        data = _read(args.input)
        # 执行memory函数并将结果写入输出文件
        _write(args.output, memory(data))
    elif args.action == 'stats':
        # 读取输入数据
        data = _read(args.input)
        # 执行segsum函数并打印结果
        print(segsum(data))
    elif args.action == 'trace':
        # 读取输入数据
        data = _read(args.input)
        # 执行trace函数并打印结果
        print(trace(data))
    elif args.action == 'compare':
        # 分别读取两个输入数据
        before = _read(args.before)
        after = _read(args.after)
        # 执行compare函数并将结果写入输出文件
        _write(args.output, compare(before, after))
    elif args.action == 'trace_plot':
        # 读取输入数据
        data = _read(args.input)
        # 执行trace_plot函数并将结果写入输出文件，传递设备参数和是否绘制段的选项
        _write(args.output, trace_plot(data, device=args.device, plot_segments=args.segments))
    elif args.action == 'segment_plot':
        # 读取输入数据
        data = _read(args.input)
        # 执行segment_plot函数并将结果写入输出文件，传递设备参数
        _write(args.output, segment_plot(data, device=args.device))
```