# `.\pytorch\torch\utils\viz\_cycles.py`

```
# mypy: allow-untyped-defs
# 导入必要的库和模块
import gc  # 导入垃圾回收模块
import sys  # 导入系统模块
from typing import Any, Dict, List, NamedTuple, Optional, Tuple  # 导入类型相关的模块
import types  # 导入类型模块
import weakref  # 导入弱引用模块
import json  # 导入 JSON 模块
from tempfile import NamedTemporaryFile  # 导入临时文件模块
import torch  # 导入 PyTorch 模块
from torch.cuda._memory_viz import _frames_fmt, _block_extra  # 导入 CUDA 内存可视化相关模块
import atexit  # 导入退出时执行模块
import logging  # 导入日志模块
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

# 定义观察垃圾回收的函数
def observe_garbage(observer):
    enabled = True  # 开启观察标志

    # 注册退出时执行的函数，用于禁用回调
    def disable():
        # 在退出时运行垃圾回收时，像 `sys` 这样的内容可能已经卸载，
        # 所以必须禁用回调以避免触发错误。
        nonlocal enabled
        enabled = False
    atexit.register(disable)  # 注册禁用函数到退出时执行

    # 定义垃圾回收回调函数
    def gc_callback(phase, info):
        nonlocal enabled
        if not enabled:
            return
        
        # 开始垃圾回收阶段，设置 DEBUG_SAVEALL 标志
        if phase == "start":
            gc.set_debug(gc.DEBUG_SAVEALL)
        # 结束垃圾回收阶段
        elif phase == "stop":
            orig_trace = sys.getprofile()  # 获取原始的函数调用追踪

            self_return = [False]

            # 定义执行垃圾回收的函数
            def do_collect(*args, **kwargs):
                nonlocal enabled
                if not self_return[0]:
                    self_return[0] = True
                else:
                    sys.setprofile(orig_trace)  # 恢复原始的函数调用追踪
                    enabled = False
                    try:
                        # gc.garbage 中的对象经历了一次收集
                        # 因此我们必须收集比它们大的一代，以释放它们
                        # 但这可能会释放其他我们不想丢失的对象。因此，我们在这里
                        # 强制在最高级别运行 gc，报告我们找到的所有内容，然后我们可以释放它。
                        if info['generation'] != 2:
                            gc.collect()
                        observer(gc.garbage)  # 观察垃圾对象
                        gc.garbage.clear()  # 清空 gc.garbage
                        # 我们必须重新运行 GC 以清理之前保存的循环引用
                        gc.set_debug(0)
                        before = torch.cuda.memory_allocated()
                        gc.collect()
                        after = torch.cuda.memory_allocated()
                        if before != after:
                            logger.warning("CUDA Memory changed during GC, %d bytes freed.", before - after)
                    finally:
                        enabled = True
                if orig_trace is not None:
                    return orig_trace(*args, **kwargs)
            sys.setprofile(do_collect)  # 设置函数调用追踪为 do_collect

    gc.callbacks.append(gc_callback)  # 添加垃圾回收回调函数到 gc.callbacks

    # 提供一个取消回调函数的方法
    def remove():
        gc.callbacks.remove(gc_callback)
    return remove  # 返回取消回调函数的方法

# Function to visualize cycles adapted from refcycle:
# Copyright 2013 Mark Dickinson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# 定义一个内部函数_f，该函数返回一个闭包，闭包的默认值为None
def _get_cell_type():
    def f(x=None):
        return lambda: x
    return type(f().__closure__[0])

# 使用_get_cell_type函数获取闭包类型，赋值给CellType变量
CellType = _get_cell_type()

# 定义函数annotated_references，返回给定对象的已知引用信息
def annotated_references(obj):
    """
    Return known information about references held by the given object.

    Returns a mapping from referents to lists of descriptions.  Note that there
    may be more than one edge leading to any particular referent; hence the
    need for a list.  Descriptions are currently strings.
    """

    # 定义一个字典references，用于存储对象引用信息，键为对象ID，值为描述列表
    references: Dict[int, List[str]] = {}

    # 定义内部函数add_reference，将指定名称和对象添加到references字典中
    def add_reference(name, obj):
        references.setdefault(id(obj), []).append(name)

    # 定义内部函数add_attrs，依次检查obj对象的各个属性是否存在，并添加到references字典中
    def add_attrs(*attrs):
        for attr in attrs:
            if hasattr(obj, attr):
                add_reference(attr, getattr(obj, attr))

    # 定义内部函数add_cell_references，尝试添加cell_contents属性到references字典中
    def add_cell_references():
        try:
            add_attrs("cell_contents")
        except ValueError:
            # 如果cell_contents为空，访问它会引发ValueError，此时不做任何操作
            pass

    # 定义内部函数add_function_references，添加函数对象的各个关联属性到references字典中
    def add_function_references():
        add_attrs("__defaults__",
                  "__closure__",
                  "__globals__",
                  "__code__",
                  "__name__",
                  "__module__",
                  "__doc__"
                  "__qualname__",
                  "__annotations__",
                  "__kwdefaults__")

    # 定义内部函数add_sequence_references，遍历序列对象obj，将每个元素添加到references字典中
    def add_sequence_references():
        for position, item in enumerate(obj):
            add_reference(f"[{position}]", item)

    # 定义内部函数add_dict_references，遍历字典对象obj，将每个键和值添加到references字典中
    def add_dict_references():
        for key, value in obj.items():
            add_reference("key", key)
            add_reference(f"[{repr(key)}]", value)

    # 定义内部函数add_set_references，遍历集合对象obj，将每个元素添加到references字典中
    def add_set_references():
        for elt in obj:
            add_reference("element", elt)

    # 定义内部函数add_bound_method_references，添加绑定方法对象的相关属性到references字典中
    def add_bound_method_references():
        add_attrs("__self__", "__func__", "im_class")

    # 定义内部函数add_weakref_references，对弱引用对象进行处理，添加相关属性到references字典中
    def add_weakref_references():
        # 对于weakref的子类，无法可靠地区分回调函数与其他属性
        if type(obj) is weakref.ref:
            referents = gc.get_referents(obj)
            if len(referents) == 1:
                target = referents[0]
                add_reference("__callback__", target)
    # 定义一个函数，用于为对象添加帧引用
    def add_frame_references():
        # 获取对象的局部变量字典
        f_locals = obj.f_locals
        # 调用 add_attrs 函数，为对象添加多个属性引用
        add_attrs("f_back", "f_code", "f_builtins", "f_globals", "f_trace", "f_locals")
        
        # 有些不良行为的代码会将 f_locals 字典替换为不支持完整字典接口的对象。
        # 因此，只有当 f_locals 是 Python 字典时，我们才继续进行注解。
        if type(f_locals) is dict:
            # 遍历对象的局部变量字典，为每个局部变量添加引用
            for name, local in obj.f_locals.items():
                add_reference(f"local {name}", local)

    # 定义一个函数，用于为对象添加获取设置描述符的引用
    def add_getset_descriptor_references():
        # 调用 add_attrs 函数，为对象添加 __objclass__、__name__、__doc__ 属性引用
        add_attrs("__objclass__", "__name__", "__doc__")

    # 根据对象类型选择相应的引用处理函数
    type_based_references = {
        tuple: add_sequence_references,
        list: add_sequence_references,
        dict: add_dict_references,
        set: add_set_references,
        frozenset: add_set_references,
        types.FunctionType: add_function_references,
        types.FrameType: add_frame_references,  # 对象类型为 FrameType 时，调用 add_frame_references 函数
        CellType: add_cell_references,
        types.MethodType: add_bound_method_references,
        weakref.ref: add_weakref_references,
        types.GetSetDescriptorType: add_getset_descriptor_references,  # 对象类型为 GetSetDescriptorType 时，调用 add_getset_descriptor_references 函数
    }

    # 遍历对象类型的方法解析顺序（Method Resolution Order，MRO）
    for type_ in type(obj).__mro__:
        # 如果当前对象类型在 type_based_references 字典中存在对应的处理函数，则调用该函数
        if type_ in type_based_references:
            type_based_references[type_]()

    # 调用 add_attrs 函数，为对象添加 __dict__、__class__ 属性引用
    add_attrs("__dict__", "__class__")
    
    # 如果对象是类对象（instance），则调用 add_attrs 函数，为对象添加 __mro__ 属性引用
    if isinstance(obj, type):
        add_attrs("__mro__")

    # 返回最终收集到的所有引用信息
    return references
###############################################################################
# Object annotations.

# 定义基本类型集合，包括整数、浮点数、复数、None 类型、字符串和字节串
BASE_TYPES = (int, float, complex, type(None), str, bytes)
# 定义帧文件名的长度限制
FRAME_FILENAME_LIMIT = 32

# 定义对象注释函数，返回用于 Graphviz 节点的字符串描述
def object_annotation(obj):
    """
    Return a string to be used for Graphviz nodes.

    The string should be short but as informative as possible.
    """

    # 定义格式化序列的辅助函数，将序列对象转换为字符串表示
    def format_sequence(obj):
        # 取前8个元素，如果元素是基本类型则使用其 repr()，否则使用其类型名
        body = ','.join(repr(x) if isinstance(x, BASE_TYPES) else type(x).__name__ for i, x in zip(range(8), obj))
        # 如果序列长度超过8个元素，则显示省略号和剩余元素数量
        if len(obj) > 8:
            body = f'{body}, ...{len(obj) - 8}'
        return body

    # 对于基本类型，直接使用其 repr() 表示
    if isinstance(obj, BASE_TYPES):
        return repr(obj)
    # 对于函数类型，显示为 "function\n函数名"
    if type(obj).__name__ == 'function':
        return f"function\n{obj.__name__}"
    # 对于方法类型，显示为 "instancemethod\n方法全名"
    elif isinstance(obj, types.MethodType):
        try:
            func_name = obj.__func__.__qualname__
        except AttributeError:
            func_name = "<anonymous>"
        return f"instancemethod\n{func_name}"
    # 对于列表类型，显示为 "[元素1,元素2,...]" 的格式
    elif isinstance(obj, list):
        return f"[{format_sequence(obj)}]"
    # 对于元组类型，显示为 "(元素1,元素2,...)" 的格式
    elif isinstance(obj, tuple):
        return f"({format_sequence(obj)})"
    # 对于字典类型，显示为 "dict[元素数量]" 的格式
    elif isinstance(obj, dict):
        return f"dict[{len(obj)}]"
    # 对于模块类型，显示为 "module\n模块名"
    elif isinstance(obj, types.ModuleType):
        return f"module\n{obj.__name__}"
    # 对于类类型，显示为 "type\n类名"
    elif isinstance(obj, type):
        return f"type\n{obj.__name__}"
    # 对于弱引用类型，显示为 "weakref to id 0x十六进制对象ID" 或 "weakref (dead referent)"
    elif isinstance(obj, weakref.ref):
        referent = obj()
        if referent is None:
            return "weakref (dead referent)"
        else:
            return f"weakref to id 0x{id(referent):x}"
    # 对于帧类型，显示为 "frame\n文件名:行号"
    elif isinstance(obj, types.FrameType):
        filename = obj.f_code.co_filename
        if len(filename) > FRAME_FILENAME_LIMIT:
            filename = "..." + filename[-(FRAME_FILENAME_LIMIT - 3):]
        return f"frame\n{filename}:{obj.f_lineno}"
    # 对于其他对象，显示为 "object\n模块名.类名"
    else:
        return f"object\n{type(obj).__module__}.{type(obj).__name__}"


# 定义节点类，表示 Graphviz 中的节点
class Node(NamedTuple):
    label: str                # 节点标签，使用 object_annotation 函数生成
    context: Optional[str]    # 节点上下文信息，由 context 函数生成
    root: bool                # 节点是否是根节点，由 filter 函数生成
    referrents: List[Tuple[str, int]]  # 节点引用的其他节点列表


# 定义创建图函数，生成对象的节点及其之间的引用关系
def create_graph(objects, *, context=None, filter=None):
    if context is None:
        context = cuda_allocation_context()  # 获取 CUDA 分配的上下文信息
    if filter is None:
        filter = is_cuda_tensor  # 检查对象是否是 CUDA 张量

    # 创建节点列表，每个节点由 object_annotation、context 和 filter 生成
    nodes = [Node(object_annotation(obj), context(obj), filter(obj), []) for obj in objects]
    node_referrers: List[List[int]] = [[] for obj in objects]

    # 构建对象 ID 到节点索引的映射
    id_to_node = {id(obj): i for i, obj in enumerate(objects)}
    for obj in objects:
        fidx = id_to_node[id(obj)]
        f = nodes[fidx]
        # 获取对象的注释引用
        references = annotated_references(obj)
        for referrent in gc.get_referents(obj):
            rid = id(referrent)
            tidx = id_to_node.get(rid, None)
            if tidx is None:
                continue
            t = nodes[tidx]
            labels = references.get(rid, ["?"])
            node_referrers[tidx].append(fidx)
            for label in labels:
                f.referrents.append((label, tidx))
    # 找出所有根节点的索引并存储在列表中
    to_search = [i for i, n in enumerate(nodes) if n.root]
    
    # 用于存储要保留的节点索引的集合
    to_keep = set()
    
    # 使用深度优先搜索遍历节点图
    while to_search:
        # 弹出待搜索列表的最后一个索引
        idx = to_search.pop()
        
        # 如果索引已经在保留集合中，则跳过
        if idx in to_keep:
            continue
        
        # 将当前索引添加到保留集合中
        to_keep.add(idx)
        
        # 获取当前节点的引用者列表
        referrers = node_referrers[idx]
        
        # 将当前节点的引用者列表加入待搜索列表中
        to_search.extend(referrers)
    
    # 创建一个从原始节点索引到筛选后索引的映射字典
    id_to_filtered_id: Dict[int, int] = {}
    
    # 存储筛选后的节点列表
    filtered: List[Any] = []
    
    # 遍历所有节点，将筛选后的节点添加到filtered列表，并建立映射关系
    for i, n in enumerate(nodes):
        if i in to_keep:
            id_to_filtered_id[i] = len(id_to_filtered_id)
            filtered.append(n)
    
    # 更新筛选后节点的引用者列表，根据映射关系进行更新
    for n in filtered:
        n.referrents[:] = [(label, id_to_filtered_id[idx])
                           for (label, idx) in n.referrents
                           if idx in id_to_filtered_id]
    
    # 返回筛选后的节点列表
    return filtered
# 将输入的对象 n 转换为 JSON 格式的字符串并返回
def escape(n):
    return json.dumps(n)

# 检查对象 obj 是否为 CUDA Tensor，并且在 CUDA 上分配，但不是 FakeTensor
def is_cuda_tensor(obj):
    return isinstance(obj, torch.Tensor) and obj.is_cuda and not isinstance(obj, torch._subclasses.FakeTensor)

# 获取当前 CUDA 内存分配的快照，并建立地址到帧的映射关系字典
def cuda_allocation_context():
    snapshot = torch.cuda.memory._snapshot()
    addr_to_frame = {}
    for seg in snapshot['segments']:
        addr = seg['address']
        for blk in seg['blocks']:
            if blk['state'] == 'active_allocated':
                frames, real_size = _block_extra(blk)
                addr_to_frame[addr] = frames
            addr += blk['size']

    # 返回一个函数，用于获取 CUDA Tensor 对象的分配上下文
    def object_context(obj):
        if is_cuda_tensor(obj):
            addr = obj.untyped_storage().data_ptr()
            frames = addr_to_frame.get(addr)
            if frames is not None:
                return '\n'.join(_frames_fmt(frames, full_filename=True))
        return None
    return object_context

# 将节点列表转换为 DOT 格式的图形描述字符串
def to_dot(nodes):
    lines = ["digraph GraphName {", "node [shape=rect];", 'rankdir=LR;']
    for i, n in enumerate(nodes):
        lines.append(f'{i} [label={escape(n.label)}, color={"red" if n.root else "black"}];')

    for i, f in enumerate(nodes):
        for label, j in f.referrents:
            lines.append(f'{i} -> {j} [label={escape(label)}]')
    lines.append("}\n")
    return '\n'.join(lines)

# HTML 模板，用于将 DOT 格式的图转换为可视化的 SVG 图形
_template = """
<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      margin: 0;
      padding: 0;
      overflow: hidden;
    }

    #container {
      display: flex;
      flex-direction: column;
      height: 100vh;
    }

    #main {
      flex: 2;
      overflow: auto;
    }

    #preContainer {
      flex: 1;
      overflow: auto;
    }

    svg {
        overflow: scroll;
    }

    pre {
      margin: 0;
      padding: 10px;
    }
  </style>
</head>
<body>
  <div id="container">
    <div id="main">
    </div>
    <div id="preContainer">
      <pre id="stacktrace">Mouse over tensor objects to see where they were allocated.</pre>
    </div>
  </div>
<script src='https://cdnjs.cloudflare.com/ajax/libs/viz.js/1.8.0/viz-lite.js'></script>
<script>
let dot = $DOT
let image = Viz(dot, {format: 'svg'});
document.getElementById('main').innerHTML = image
$LISTENERS
</script>
</body>
</html>
"""

# 监听器模板，生成用于节点上鼠标悬停事件的 JavaScript 监听器
_listener_template = """
document.getElementById('node{id}').addEventListener('mouseover', function(event) {{
  document.getElementById("stacktrace").textContent = {stack}
}})
"""

# 将节点列表转换为 HTML 格式的字符串
def to_html(nodes):
    listeners = []
    for i, n in enumerate(nodes):
        if n.context is None:
            continue
        s = _listener_template.format(id=str(i + 1), stack=escape(f'{n.label}:\n{n.context}'))
        listeners.append(s)
    dot = to_dot(nodes)
    # 使用模板替换 DOT 数据和监听器数据
    return _template.replace('$DOT', repr(dot)).replace('$LISTENERS', '\n'.join(listeners))

# 记录 CUDA Tensor 的内存历史，最多记录 100000 条记录
def observe_tensor_cycles(callback):
    torch.cuda.memory._record_memory_history(max_entries=100000)
    # 定义名为 observer 的函数，接受一个参数 garbage
    def observer(garbage):
        # 检查参数 garbage 是否为真值（非空、非零等）
        if garbage:
            # 如果 garbage 非空，进入条件判断
            # 检查 garbage 中是否存在不是 CUDA 张量的对象
            if not any(is_cuda_tensor(obj) for obj in garbage):
                # 如果没有找到 CUDA 张量，记录日志信息
                logger.info("No CUDA Tensors found in garbage")
                # 函数结束
                return
            # 如果存在 CUDA 张量，生成对象间关系图的 HTML，并调用回调函数
            callback(to_html(create_graph(garbage)))
    
    # 调用 observe_garbage 函数，并将 observer 函数作为参数传递
    return observe_garbage(observer)
# 定义一个函数，用于安装一个警告，以报告每当观察到占用 CUDA 内存的循环时。

# 在警告产生时，生成一个 .html 文件来可视化该循环，并将其链接到分配了 CUDA 张量的堆栈帧上。

# 引用循环在对象首次变得不可达时不会被立即清理，而是由循环收集器释放。如果一个循环指向一个张量，
# 那么直到垃圾收集运行时，该张量的 CUDA 内存才会被释放。CUDA 分配的积累可能导致内存不足错误（OOMs），
# 以及难以调试的非确定性分配行为。

def warn_tensor_cycles():
    # 使用 logger 记录信息，指示正在监视 Python 引用循环中的 CUDA 张量。
    logger.info("Watching Python reference cycles for CUDA Tensors.")

    # 定义一个内部函数 write_and_log，用于将生成的 HTML 写入临时文件，并记录警告消息，指出循环包括一个 CUDA 张量，
    # 并提供循环可视化的文件链接。
    def write_and_log(html):
        with NamedTemporaryFile('w', suffix='.html', delete=False) as f:
            f.write(html)
            logger.warning('Reference cycle includes a CUDA Tensor see visualization of cycle %s', f.name)
    
    # 返回 observe_tensor_cycles 函数的调用结果，传递 write_and_log 函数作为参数。
    return observe_tensor_cycles(write_and_log)
```