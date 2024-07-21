# `.\pytorch\torch\utils\viz\MemoryViz.js`

```py
'use strict';

// 导入 d3 库的相关模块
import * as d3 from "https://cdn.skypack.dev/d3@5";
import {axisLeft} from "https://cdn.skypack.dev/d3-axis@1";
import {scaleLinear} from "https://cdn.skypack.dev/d3-scale@1";
import {zoom, zoomIdentity} from "https://cdn.skypack.dev/d3-zoom@1";
import {brushX} from "https://cdn.skypack.dev/d3-brush@1";

// 定义颜色方案 Tableau 10
const schemeTableau10 = [
  '#4e79a7',
  '#f28e2c',
  '#e15759',
  '#76b7b2',
  '#59a14f',
  '#edc949',
  '#af7aa1',
  '#ff9da7',
  '#9c755f',
  '#bab0ab',
];

// 定义闭包函数 version_space，用于管理地址的版本号
function version_space() {
  const version = {};
  return (addr, increment) => {
    if (!(addr in version)) {
      version[addr] = 0;
    }
    const r = version[addr];
    if (increment) {
      version[addr]++;
    }
    return r;
  };
}

// 定义函数 Segment，用于创建一个段的对象
function Segment(addr, size, stream, frames, version) {
  return {addr, size, stream, version, frames};
}

// 定义函数 Block，用于创建一个块的对象
function Block(addr, size, requested_size, frames, free_requested, version) {
  return {addr, size, requested_size, frames, free_requested, version};
}

// 定义函数 EventSelector，用于创建一个事件选择器对象
function EventSelector(outer, events, stack_info, memory_view) {
  // 创建包含事件的 DIV 元素
  const events_div = outer
    .append('div')
    .attr(
      'style',
      'grid-column: 1; grid-row: 1; overflow: auto; font-family: monospace',
    );

  // 选择所有事件，绑定数据并创建 PRE 元素，并显示事件内容
  const events_selection = events_div
    .selectAll('pre')
    .data(events)
    .enter()
    .append('pre')
    .text(e => formatEvent(e)) // 使用 formatEvent 函数格式化事件
    .attr('style', '');

  let selected_event_idx = null;

  // 定义事件选择器 es
  const es = {
    select(idx) {
      // 取消选中之前选择的事件
      if (selected_event_idx !== null) {
        const selected_event = d3.select(
          events_div.node().children[selected_event_idx],
        );
        selected_event.attr('style', '');
      }
      // 选中新的事件
      if (idx !== null) {
        const div = d3.select(events_div.node().children[idx]);
        div.attr('style', `background-color: ${schemeTableau10[5]}`);
        // 调用 memory_view.draw 方法，并进入事件堆栈
        const [reserved, allocated] = memory_view.draw(idx);
        const enter = () => eventStack(div.datum(), allocated, reserved);
        stack_info.highlight(enter);
        div.node().scrollIntoViewIfNeeded(false);
      } else {
        memory_view.draw(0);
      }
      selected_event_idx = idx;
    },
  };

  // 监听键盘事件，实现事件选择
  d3.select('body').on('keydown', _e => {
    const key = d3.event.key;
    const actions = {ArrowDown: 1, ArrowUp: -1};
    if (selected_event_idx !== null && key in actions) {
      const new_idx = selected_event_idx + actions[key];
      es.select(Math.max(0, Math.min(new_idx, events.length - 1)));
      d3.event.preventDefault();
    }
  });

  // 注册事件处理函数
  stack_info.register(
    events_selection,
    t => eventStack(t.datum()), // 调用 eventStack 函数处理事件
    _t => {},
    d => es.select(d.datum().idx),
  );

  return es;
}

// 格式化文件大小，以人类可读的方式显示
function formatSize(num) {
  const orig = num;
  // 根据文件大小选择适当的单位
  const units = ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'];
  for (const unit of units) {
    if (Math.abs(num) < 1024.0) {
      return `${num.toFixed(1)}${unit}B (${orig} bytes)`;
    }
    num /= 1024.0;
  }
  return `${num.toFixed(1)}YiB`;
}
function formatAddr(event) {
  // 根据事件动作的前缀判断是段落还是块，生成对应的前缀
  const prefix = event.action.startsWith('segment') ? 's' : 'b';
  // 返回格式化后的地址字符串，格式为 前缀十六进制地址_版本号
  return `${prefix}${event.addr.toString(16)}_${event.version}`;
}

function formatEvent(event) {
  // 根据事件的类型不同，格式化并返回对应的事件描述字符串
  const stream =
    event.stream === null ? '' : `\n              (stream ${event.stream})`;
  switch (event.action) {
    case 'oom':
      // OOM（内存耗尽）事件的描述，包含请求大小和 CUDA 可用内存大小
      return `OOM (requested ${formatSize(event.size)}, CUDA has ${formatSize(
        event.device_free,
      )} memory free)${stream}`;
    case 'snapshot':
      // 快照事件的简单描述
      return 'snapshot';
    default:
      // 默认情况下，返回动作、地址和大小的格式化字符串
      return `${event.action.padEnd(14)} ${formatAddr(event).padEnd(
        18,
      )} ${formatSize(event.size)}${stream}`;
  }
}

function eventStack(e, allocated, reserved) {
  // 根据事件格式化事件堆栈的字符串描述
  let event = formatEvent(e);
  if (reserved !== undefined) {
    // 如果有保留内存信息，添加在事件描述之前
    event = `(${formatSize(allocated)} allocated / ${formatSize(
      reserved,
    )} reserved)\n${event}`;
  }
  // 返回格式化后的事件堆栈字符串，包括事件描述和帧信息
  return event + '\n' + format_frames(e.frames);
}

function hashCode(num) {
  // 计算输入数字的哈希码（32位整数）
  const numStr = num.toString();
  let hash = 0;
  for (let i = 0; i < numStr.length; i++) {
    const charCode = numStr.charCodeAt(i);
    hash = (hash << 5) - hash + charCode;
    hash = hash & hash; // 转换为32位整数
  }
  return hash;
}

function addStroke(d) {
  // 给输入的元素添加红色描边样式
  d.attr('stroke', 'red')
    .attr('stroke-width', '2')
    .attr('vector-effect', 'non-scaling-stroke');
}

function removeStroke(d) {
  // 移除输入元素的描边样式
  d.attr('stroke', '');
}

function calculate_fragmentation(blocks, sorted_segments) {
  // 计算内存碎片化指数
  const sorted_blocks = Object.values(blocks).sort((a, b) => a.addr - b.addr);
  let block_i = 0;
  let total_size = 0;
  let sum_squared_free = 0;
  for (const seg of sorted_segments) {
    let addr = seg.addr;
    total_size += seg.size;
    while (
      block_i < sorted_blocks.length &&
      sorted_blocks[block_i].addr < seg.addr + seg.size
    ) {
      const block = sorted_blocks[block_i];
      if (block.addr > addr) {
        sum_squared_free += (block.addr - addr) ** 2;
      }
      addr = block.addr + block.size;
      block_i += 1;
    }
    if (addr < seg.addr + seg.size) {
      sum_squared_free += (seg.addr + seg.size - addr) ** 2;
    }
  }
  // 输出内存碎片化指数
  console.log(sum_squared_free / total_size ** 2);
}

function MemoryView(outer, stack_info, snapshot, device) {
  // 创建内存视图对象
  const svg = outer
    .append('svg')
    .attr('style', 'grid-column: 2; grid-row: 1; width: 100%; height: 100%;')
    .attr('viewBox', '0 0 200 100')
    .attr('preserveAspectRatio', 'xMinYMin meet');
  const g = svg.append('g');
  const seg_zoom = zoom();
  seg_zoom.on('zoom', () => {
    g.attr('transform', d3.event.transform);
  });
  svg.call(seg_zoom);

  const sorted_segments = [];
  const block_map = {};
  for (const seg of snapshot.segments) {
    // 选择特定设备的快照段落，创建并添加段对象到排序段列表
    if (seg.device !== device) {
      continue;
    }
    sorted_segments.push(
      Segment(
        seg.address,
        seg.total_size,
        seg.stream,
        seg.frames || [],
        seg.version,
      ),
    );
  }
    for (const b of seg.blocks) {
      // 遍历每个段的块列表
      if (b.state !== 'active_pending_free' && b.state !== 'active_allocated') {
        // 如果块的状态不是 'active_pending_free' 或 'active_allocated'，跳过当前块
        continue;
      }
      // 将块信息映射到 block_map 中
      block_map[b.addr] = Block(
        b.addr,
        b.size,
        b.requested_size,
        b.frames,
        b.state === 'active_pending_free',
        b.version,
      );
    }
  }
  // 根据段的地址升序排序
  sorted_segments.sort((x, y) => x.addr - y.addr);

  function simulate_memory(idx) {
    // 创建 sorted_segments 的副本，因为接下来会修改其大小属性
    const l_segments = sorted_segments.map(x => {
      return {...x};
    });
    // 创建 block_map 的副本
    const l_block_map = {...block_map};

    function map_segment(merge, seg) {
      // 查找要插入的段的位置
      let idx = l_segments.findIndex(e => e.addr > seg.addr);
      if (!merge) {
        // 如果不需要合并，则直接插入段
        l_segments.splice(idx, 0, seg);
        return;
      }
      // 如果找不到位置，则插入到末尾
      if (idx === -1) {
        idx = l_segments.length;
      }
      // 插入段并尝试合并相邻的同一流的段
      l_segments.splice(idx, 0, seg);
      if (idx + 1 < l_segments.length) {
        const next = l_segments[idx + 1];
        if (seg.addr + seg.size === next.addr && seg.stream === next.stream) {
          seg.size += next.size;
          l_segments.splice(idx + 1, 1);
        }
      }
      if (idx > 0) {
        const prev = l_segments[idx - 1];
        if (prev.addr + prev.size === seg.addr && prev.stream === seg.stream) {
          prev.size += seg.size;
          l_segments.splice(idx, 1);
        }
      }
    }
    function unmap_segment(merge, seg) {
      if (!merge) {
        // 如果不需要合并，则直接移除指定地址的段
        l_segments.splice(
          l_segments.findIndex(x => x.addr === seg.addr),
          1,
        );
        return;
      }
      // 计算段的结束地址
      const seg_end = seg.addr + seg.size;
      // 查找要移除的段的位置
      const idx = l_segments.findIndex(
        e => e.addr <= seg.addr && seg_end <= e.addr + e.size,
      );
      const existing = l_segments[idx];
      const existing_end = existing.addr + existing.size;
      if (existing.addr === seg.addr) {
        // 如果段的起始地址匹配，调整现有段的起始地址和大小
        existing.addr += seg.size;
        existing.size -= seg.size;
        if (existing.size === 0) {
          // 如果现有段的大小为0，则移除该段
          l_segments.splice(idx, 1);
        }
      } else if (existing_end === seg_end) {
        // 如果段的结束地址匹配，调整现有段的大小
        existing.size -= seg.size;
      } else {
        // 否则，将现有段分割为两段
        existing.size = seg.addr - existing.addr;
        seg.addr = seg_end;
        seg.size = existing_end - seg_end;
        l_segments.splice(idx + 1, 0, seg);
      }
    }
    // 获取设备跟踪事件
    const events = snapshot.device_traces[device];
    // 从最后一个事件开始向前遍历到指定索引 idx
    for (let i = events.length - 1; i > idx; i--) {
      // 获取当前事件对象
      const event = events[i];
      // 根据事件的动作类型进行不同的操作
      switch (event.action) {
        // 如果动作是 'free'，则更新 l_block_map 中对应地址的 Block 对象
        case 'free':
          l_block_map[event.addr] = Block(
            event.addr,
            event.size,
            event.size,
            event.frames,
            false,
            event.version,
          );
          break;
        // 如果动作是 'free_requested'，则设置对应 Block 的 free_requested 属性为 false
        case 'free_requested':
          l_block_map[event.addr].free_requested = false;
          break;
        // 如果动作是 'free_completed'，则更新 l_block_map 中对应地址的 Block 对象
        case 'free_completed':
          l_block_map[event.addr] = Block(
            event.addr,
            event.size,
            event.size,
            event.frames,
            true,
            event.version,
          );
          break;
        // 如果动作是 'alloc'，则从 l_block_map 中删除对应地址的 Block 对象
        case 'alloc':
          delete l_block_map[event.addr];
          break;
        // 如果动作是 'segment_free' 或 'segment_unmap'，则调用 map_segment 函数处理段的释放或取消映射
        case 'segment_free':
        case 'segment_unmap':
          map_segment(
            event.action === 'segment_unmap',
            Segment(
              event.addr,
              event.size,
              event.stream,
              event.frames,
              event.version,
            ),
          );
          break;
        // 如果动作是 'segment_alloc' 或 'segment_map'，则调用 unmap_segment 函数处理段的分配或映射
        case 'segment_alloc':
        case 'segment_map':
          unmap_segment(
            event.action === 'segment_map',
            Segment(
              event.addr,
              event.size,
              event.stream,
              event.frames,
              event.version,
            ),
          );
          break;
        // 如果动作是 'oom'，则不执行任何操作
        case 'oom':
          break;
        // 默认情况下，不执行任何操作
        default:
          break;
      }
    }
    // 构造新的 blocks 数组，包含 l_block_map 中所有的 Block 对象
    const new_blocks = Object.values(l_block_map);
    // 返回更新后的段列表和新的 blocks 数组
    return [l_segments, new_blocks];
  }

  // 如果未进入循环，则返回空对象
  return {
    },
  };
}

// StackInfo 类用于管理和操作堆栈信息的显示和交互
function StackInfo(outer) {
  // 创建一个预格式化的区块用于显示堆栈信息，并设置样式和布局属性
  const stack_trace = outer
    .append('pre')
    .attr('style', 'grid-column: 1 / 3; grid-row: 2; overflow: auto');
  let selected = {
    // 当鼠标进入时清空堆栈信息显示区域
    enter: () => {
      stack_trace.text('');
    },
    leave: () => {},  // 默认的离开事件为空函数
  };
  // 返回一个对象，包含注册事件、高亮显示等方法
  return {
    // 注册事件处理函数，包括鼠标悬停、按下和离开时的操作
    register(dom, enter, leave = _e => {}, select = _e => {}) {
      dom
        .on('mouseover', _e => {
          // 当鼠标悬停时，执行进入事件并显示相关堆栈信息
          selected.leave();
          stack_trace.text(enter(d3.select(d3.event.target)));
        })
        .on('mousedown', _e => {
          // 当鼠标按下时，更新选择的对象和相应的操作
          const obj = d3.select(d3.event.target);
          selected = {
            enter: () => stack_trace.text(enter(obj)),
            leave: () => leave(obj),
          };
          select(obj);
        })
        .on('mouseleave', _e => {
          // 当鼠标离开时执行离开事件，并重新显示选中的堆栈信息
          leave(d3.select(d3.event.target));
          selected.enter();
        });
    },
    // 高亮显示指定的堆栈信息
    highlight(enter, leave = () => {}) {
      selected = {enter: () => stack_trace.text(enter()), leave};
      selected.enter();
    },
  };
}

// 创建段视图，显示指定设备的内存视图和事件选择器
function create_segment_view(dst, snapshot, device) {
  // 在指定的目标区域创建一个包含布局和样式的容器
  const outer = dst
    .append('div')
    .attr(
      'style',
      'display: grid; grid-template-columns: 1fr 2fr; grid-template-rows: 2fr 1fr; height: 100%; gap: 10px',
    );

  // 获取指定设备的事件列表
  const events = snapshot.device_traces[device];
  // 创建堆栈信息管理对象
  const stack_info = StackInfo(outer);
  // 创建内存视图对象，关联堆栈信息和快照数据
  const memory_view = MemoryView(outer, stack_info, snapshot, device);
  // 创建事件选择器，关联外部容器、事件列表、堆栈信息和内存视图
  const event_selector = EventSelector(outer, events, stack_info, memory_view);

  // 使用动画帧请求执行事件选择器的最后一个事件选择
  window.requestAnimationFrame(function () {
    event_selector.select(events.length > 0 ? events.length - 1 : null);
  });
}

// 为快照对象添加注释信息
function annotate_snapshot(snapshot) {
  // 初始化快照的版本空间和类别列表
  snapshot.segment_version = version_space();
  snapshot.block_version = version_space();
  snapshot.categories = [];
  // 初始化空列表和下一个流的索引
  const empty_list = [];
  let next_stream = 1;
  // 创建流名称映射对象，初始包含一个默认流
  const stream_names = {0: 0};
  // 获取流名称的函数，如果不存在则创建新的流编号
  function stream_name(s) {
    if (!(s in stream_names)) {
      stream_names[s] = next_stream++;
    }
    return stream_names[s];
  }
  // 创建新的设备跟踪列表，并进行流编号处理
  const new_traces = [];
  for (const device_trace of snapshot.device_traces) {
    const new_trace = [];
    new_traces.push(new_trace);
    // 遍历设备追踪事件数组
    for (const t of device_trace) {
      // 如果事件对象 t 中不存在 frames 属性，则将其设为一个空列表
      if (!('frames' in t)) {
        t.frames = empty_list;
      }
      // 对流名称进行处理，确保每个流名称都唯一
      t.stream = stream_name(t.stream);
      // 根据 t 的 action 属性执行不同的操作
      switch (t.action) {
        case 'free_completed':
          // 设置 t 的 version 属性为释放块的快照版本号
          t.version = snapshot.block_version(t.addr, true);
          // 如果 new_trace 数组长度大于 0，则合并相邻的 free_requested 和 free_completed 事件
          if (new_trace.length > 0) {
            const prev = new_trace.at(-1);
            if (prev.action === 'free_requested' && prev.addr === t.addr) {
              prev.action = 'free';
              continue;
            }
          }
          break;
        case 'free_requested':
        case 'alloc':
          // 设置 t 的 version 属性为分配块的快照版本号
          t.version = snapshot.block_version(t.addr, false);
          break;
        case 'segment_free':
        case 'segment_unmap':
          // 设置 t 的 version 属性为释放段的快照版本号
          t.version = snapshot.segment_version(t.addr, true);
          break;
        case 'segment_alloc':
        case 'segment_map':
          // 设置 t 的 version 属性为映射段的快照版本号
          t.version = snapshot.segment_version(t.addr, false);
          break;
        default:
          break;
      }
      // 如果 t 中有 category 属性，并且该 category 不在 snapshot.categories 数组中，则加入该 category
      if ('category' in t && !snapshot.categories.includes(t.category)) {
        snapshot.categories.push(t.category);
      }
      // 设置 t 的 idx 属性为当前 new_trace 数组的长度，并将 t 添加到 new_trace 数组中
      t.idx = new_trace.length;
      new_trace.push(t);
    }
  }
  // 将更新后的 new_trace 数组赋值给 snapshot.device_traces
  snapshot.device_traces = new_trace;
  // 如果所有事件都在默认流上，则清空流名称以优化显示
  if (next_stream == 1) {
    // 遍历 snapshot.device_traces 数组
    for (const device_trace of snapshot.device_traces) {
      // 将每个事件对象 t 的 stream 属性设为 null
      for (const t of device_trace) {
        t.stream = null;
      }
    }
  }

  // 遍历 snapshot.segments 数组
  for (const seg of snapshot.segments) {
    // 对段对象 seg 的 stream 属性进行处理，确保其唯一性
    seg.stream = stream_name(seg.stream);
    // 设置段对象 seg 的 version 属性为其分配的快照版本号
    seg.version = snapshot.segment_version(seg.address, false);
    let addr = seg.address;
    // 遍历段对象 seg 的 blocks 数组
    for (const b of seg.blocks) {
      // 设置块对象 b 的 addr 属性为当前地址 addr
      b.addr = addr;
      // 如果块对象 b 中不存在 frames 属性，则根据历史记录进行设置
      if (!('frames' in b)) {
        if ('history' in b) {
          b.frames = b.history[0].frames || empty_list;
          b.requested_size = b.requested_size || b.history[0].real_size;
        } else {
          b.frames = empty_list;
          b.requested_size = b.requested_size || b.size;
        }
      }
      // 设置块对象 b 的 version 属性为其分配的快照版本号
      b.version = snapshot.block_version(b.addr, false);
      // 更新地址 addr 为下一个块的地址
      addr += b.size;
    }
  }

  // 如果 snapshot.categories 数组中有元素，并且不包含 'unknown'，则加入 'unknown'
  if (
    snapshot.categories.length > 0 &&
    !snapshot.categories.includes('unknown')
  ) {
    snapshot.categories.push('unknown');
  }
}

// 函数 elideRepeats 接收 frames 数组作为参数，用于处理重复的堆栈帧
function elideRepeats(frames) {
  const result = []; // 初始化结果数组
  const length = frames.length; // 获取 frames 数组的长度
  for (let i = 0; i < length; ) { // 循环遍历 frames 数组
    let j = i + 1; // 设置 j 的初始值为 i + 1
    const f = frames[i]; // 获取当前索引 i 处的帧
    while (j < length && f === frames[j]) { // 查找连续重复的帧直到不重复为止
      j++;
    }
    switch (j - i) { // 根据重复次数决定如何处理帧
      case 1:
        result.push(f); // 单次重复，直接添加到结果数组中
        break;
      case 2:
        result.push(f, f); // 两次重复，将帧添加两次到结果数组中
        break;
      default:
        result.push(f, `<repeats ${j - i - 1} times>`); // 多次重复，使用 "<repeats n times>" 形式表示
        break;
    }
    i = j; // 更新 i 为 j，继续处理下一个不重复的帧
  }
  return result; // 返回处理后的结果数组
}

// 函数 frameFilter 用于过滤需要忽略的函数名和文件名
function frameFilter({name, filename}) {
  // 需要忽略的函数名列表
  const omitFunctions = [
    'unwind::unwind',
    'CapturedTraceback::gather',
    'gather_with_cpp',
    '_start',
    '__libc_start_main',
    'PyEval_',
    'PyObject_',
    'PyFunction_',
  ];

  // 需要忽略的文件名列表
  const omitFilenames = [
    'core/boxing',
    '/Register',
    '/Redispatch',
    'pythonrun.c',
    'Modules/main.c',
    'Objects/call.c',
    'Objects/methodobject.c',
    'pycore_ceval.h',
    'ceval.c',
    'cpython/abstract.h',
  ];

  // 检查给定的函数名是否在需要忽略的列表中
  for (const of of omitFunctions) {
    if (name.includes(of)) {
      return false; // 如果找到匹配的函数名，返回 false
    }
  }

  // 检查给定的文件名是否在需要忽略的列表中
  for (const of of omitFilenames) {
    if (filename.includes(of)) {
      return false; // 如果找到匹配的文件名，返回 false
    }
  }

  return true; // 如果未匹配到需要忽略的函数名或文件名，返回 true
}

// 函数 format_frames 用于格式化堆栈帧信息
function format_frames(frames) {
  if (frames.length === 0) { // 如果 frames 数组为空
    return (
      `This block has no frames. Potential causes:\n` +
      `1) This block was allocated before _record_memory_history was enabled.\n` +
      `2) The context or stacks passed to _record_memory_history does not include this block. Consider changing context to 'state', 'alloc', or 'all', or changing stacks to 'all'.\n` +
      `3) This event occurred during backward, which has no python frames, and memory history did not include C++ frames. Use stacks='all' to record both C++ and python frames.`
    ); // 返回未找到堆栈帧的可能原因的错误信息
  }
  const frame_strings = frames
    .filter(frameFilter) // 使用 frameFilter 函数过滤要显示的堆栈帧
    .map(f => `${f.filename}:${f.line}:${f.name}`); // 将过滤后的帧信息转换成字符串形式
  return elideRepeats(frame_strings).join('\n'); // 调用 elideRepeats 函数处理重复的帧，并将结果数组转换成字符串返回
}

// 函数 process_alloc_data 处理分配数据快照，根据设备和操作类型进行分类
function process_alloc_data(snapshot, device, plot_segments, max_entries) {
  const elements = []; // 初始化元素数组
  const initially_allocated = []; // 初始化初始分配数组
  const actions = []; // 初始化操作数组
  const addr_to_alloc = {}; // 初始化地址到分配映射表

  // 根据 plot_segments 的值选择不同的操作类型
  const alloc = plot_segments ? 'segment_alloc' : 'alloc';
  const [free, free_completed] = plot_segments
    ? ['segment_free', 'segment_free']
    : ['free', 'free_completed'];

  // 遍历设备的追踪数据中的每个事件 e
  for (const e of snapshot.device_traces[device]) {
    switch (e.action) {
      case alloc:
        elements.push(e); // 如果是分配操作，将事件 e 添加到元素数组中
        addr_to_alloc[e.addr] = elements.length - 1; // 记录地址到分配映射
        actions.push(elements.length - 1); // 记录操作的索引
        break;
      case free:
      case free_completed:
        if (e.addr in addr_to_alloc) { // 如果是释放操作且地址在映射表中
          actions.push(addr_to_alloc[e.addr]); // 记录释放的操作索引
          delete addr_to_alloc[e.addr]; // 删除地址的分配映射
        } else {
          elements.push(e); // 否则将事件 e 添加到元素数组中
          initially_allocated.push(elements.length - 1); // 记录初始分配的索引
          actions.push(elements.length - 1); // 记录操作的索引
        }
        break;
      default:
        break;
    }
  }

  // 遍历快照的段信息
  for (const seg of snapshot.segments) {
    if (seg.device !== device) { // 如果段信息的设备与当前设备不匹配，跳过该段信息
      continue;
    }
    }
    // 如果需要绘制分段图，则执行以下操作
    if (plot_segments) {
      // 如果当前段落地址不在addr_to_alloc中
      if (!(seg.address in addr_to_alloc)) {
        // 创建一个新的元素对象，表示分配操作
        const element = {
          action: 'alloc',
          addr: seg.address,
          size: seg.total_size,
          frames: [],
          stream: seg.stream,
          version: seg.version,
        };
        // 将新元素加入elements数组
        elements.push(element);
        // 将新元素在elements数组中的索引加入initially_allocated数组
        initially_allocated.push(elements.length - 1);
      }
    } else {
      // 如果不需要绘制分段图，则遍历每个分块
      for (const b of seg.blocks) {
        // 如果当前块状态为'active_allocated'且地址不在addr_to_alloc中
        if (b.state === 'active_allocated' && !(b.addr in addr_to_alloc)) {
          // 创建一个新的元素对象，表示分配操作
          const element = {
            action: 'alloc',
            addr: b.addr,
            size: b.requested_size,
            frames: b.frames,
            stream: seg.stream,
            version: b.version,
          };
          // 将新元素加入elements数组
          elements.push(element);
          // 将新元素在elements数组中的索引加入initially_allocated数组
          initially_allocated.push(elements.length - 1);
        }
      }
    }
  }
  // 反转initially_allocated数组，以便于后续正确显示分配顺序
  initially_allocated.reverse();
  // 如果没有操作动作(actions)，且initially_allocated数组不为空
  // 则将initially_allocated数组的最后一个元素加入actions数组
  if (actions.length === 0 && initially_allocated.length > 0) {
    actions.push(initially_allocated.pop());
  }

  // 初始化数组和变量
  const current = [];
  const current_data = [];
  const data = [];
  let max_size = 0;

  let total_mem = 0;
  let total_summarized_mem = 0;
  let timestep = 0;

  const max_at_time = [];

  // 初始化summarized_mem对象
  const summarized_mem = {
    elem: 'summarized',
    timesteps: [],
    offsets: [total_mem],
    size: [],
    color: 0,
  };
  // 初始化summarized_elems对象
  const summarized_elems = {};

  // 定义advance函数，用于推进时间步进
  function advance(n) {
    // 将当前时间步进和总内存及总汇总内存记录到summarized_mem对象
    summarized_mem.timesteps.push(timestep);
    summarized_mem.offsets.push(total_mem);
    summarized_mem.size.push(total_summarized_mem);
    timestep += n;
    // 将n次时间步进中的最大内存记录到max_at_time数组中
    for (let i = 0; i < n; i++) {
      max_at_time.push(total_mem + total_summarized_mem);
    }
  }

  // 对elements数组按size从大到小排序，并选取前max_entries个元素构成sizes数组
  const sizes = elements
    .map((x, i) => [x.size, i])
    .sort(([x, _xi], [y, _yi]) => y - x);

  // 初始化draw_elem对象，用于标记需要绘制的元素
  const draw_elem = {};
  // 将sizes数组中的前max_entries个元素加入draw_elem对象
  for (const [_s, e] of sizes.slice(0, max_entries)) {
    draw_elem[e] = true;
  }

  // 定义add_allocation函数，用于添加分配操作
  function add_allocation(elem) {
    // 获取elements数组中对应索引的元素对象
    const element_obj = elements[elem];
    const size = element_obj.size;
    // 将当前元素索引加入current数组
    current.push(elem);
    // 根据元素属性创建新的数据对象e
    let color = elem;
    if (snapshot.categories.length > 0) {
      color = snapshot.categories.indexOf(element_obj.category || 'unknown');
    }
    const e = {
      elem,
      timesteps: [timestep],
      offsets: [total_mem],
      size,
      color,
    };
    // 将e加入current_data和data数组
    current_data.push(e);
    data.push(e);
    // 更新总内存和元素对象的最大分配内存属性
    total_mem += size;
    element_obj.max_allocated_mem = total_mem + total_summarized_mem;
  }

  // 遍历initially_allocated数组，根据需要绘制的元素判断是否调用add_allocation函数
  for (const elem of initially_allocated) {
    if (elem in draw_elem) {
      add_allocation(elem);
    } else {
      // 将不需要绘制的元素大小加入total_summarized_mem
      total_summarized_mem += elements[elem].size;
      summarized_elems[elem] = true;
    }
  }

  // 遍历actions数组，根据元素索引调用add_allocation函数
  for (const elem of actions) {
    const size = elements[elem].size;
    // ...
  }
    // 如果元素不在draw_elem中
    if (!(elem in draw_elem)) {
      // 如果元素在summarized_elems中
      if (elem in summarized_elems) {
        // 前进1步
        advance(1);
        // 减去被总结的内存大小
        total_summarized_mem -= size;
        // 将summarized_elems中的元素设为null
        summarized_elems[elem] = null;
      } else {
        // 增加被总结的内存大小
        total_summarized_mem += size;
        // 在summarized_elems中标记该元素为true
        summarized_elems[elem] = true;
        // 前进1步
        advance(1);
      }
      // 继续循环处理下一个元素
      continue;
    }
    // 查找当前数组中元素elem最后出现的索引
    const idx = current.findLastIndex(x => x === elem);
    // 如果在当前数组中找不到该元素
    if (idx === -1) {
      // 添加分配操作
      add_allocation(elem);
      // 前进1步
      advance(1);
    } else {
      // 前进1步
      advance(1);
      // 获取被移除的数据
      const removed = current_data[idx];
      // 向被移除的数据添加时间步
      removed.timesteps.push(timestep);
      // 向被移除的数据添加偏移量，复制最后一个偏移量
      removed.offsets.push(removed.offsets.at(-1));
      // 从当前数组和当前数据中移除该元素
      current.splice(idx, 1);
      current_data.splice(idx, 1);

      // 如果移除操作后的索引小于当前数组长度
      if (idx < current.length) {
        // 遍历当前数组中剩余元素
        for (let j = idx; j < current.length; j++) {
          const e = current_data[j];
          // 向元素的时间步数组添加当前时间步
          e.timesteps.push(timestep);
          // 向元素的偏移量数组添加最后一个偏移量
          e.offsets.push(e.offsets.at(-1));
          // 向元素的时间步数组再添加当前时间步加3
          e.timesteps.push(timestep + 3);
          // 向元素的偏移量数组添加当前偏移量减去大小
          e.offsets.push(e.offsets.at(-1) - size);
        }
        // 前进3步
        advance(3);
      }
      // 减去总内存大小
      total_mem -= size;
    }
    // 更新最大内存大小
    max_size = Math.max(total_mem + total_summarized_mem, max_size);
  }

  // 遍历当前数据数组中的元素
  for (const elem of current_data) {
    // 向元素的时间步数组添加当前时间步
    elem.timesteps.push(timestep);
    // 向元素的偏移量数组添加最后一个偏移量
    elem.offsets.push(elem.offsets.at(-1));
  }
  // 将summarized_mem添加到data数组中
  data.push(summarized_mem);

  // 返回结果对象
  return {
    max_size,
    allocations_over_time: data,
    max_at_time,
    summarized_mem,
    elements_length: elements.length,
    // 为给定ID提供上下文
    context_for_id: id => {
      const elem = elements[id];
      let text = `Addr: ${formatAddr(elem)}`;
      text = `${text}, Size: ${formatSize(elem.size)} allocation`;
      text = `${text}, Total memory used after allocation: ${formatSize(
        elem.max_allocated_mem,
      )}`;
      if (elem.stream !== null) {
        text = `${text}, stream ${elem.stream}`;
      }
      if (!elem.action.includes('alloc')) {
        text = `${text}\nalloc not recorded, stack trace for free:`;
      }
      text = `${text}\n${format_frames(elem.frames)}`;
      return text;
    },
  };
  }
  
  // MemoryPlot函数：用于绘制内存分配图的函数
  function MemoryPlot(
    svg,            // SVG元素，用于绘制图形
    data,           // 数据对象，包含绘制所需的数据
    left_pad,       // 左侧内边距
    width,          // 图形总宽度
    height,         // 图形总高度
    colors = schemeTableau10,  // 颜色方案，默认为Tableau的颜色方案
  ) {
    // format_points函数：格式化数据点
    function format_points(d) {
      const size = d.size;   // 数据点的大小
      const xs = d.timesteps.map(t => xscale(t));  // X轴坐标数组
      const bottom = d.offsets.map(t => yscale(t));  // 底部坐标数组
      const m = Array.isArray(size)
        ? (t, i) => yscale(t + size[i])
        : t => yscale(t + size);
      const top = d.offsets.map(m);  // 顶部坐标数组
      const p0 = xs.map((x, i) => `${x},${bottom[i]}`);  // 底部点坐标数组
      const p1 = xs.map((x, i) => `${x},${top[i]}`).reverse();  // 顶部点坐标数组，反转顺序
      return `${p0.join(' ')} ${p1.join(' ')}`;  // 返回格式化后的点坐标字符串
    }

    // 数据的最大时间步数和最大大小
    const max_timestep = data.max_at_time.length;
    const max_size = data.max_size;

    // 绘图区域的宽度和高度
    const plot_width = width - left_pad;
    const plot_height = height;

    // Y轴比例尺和坐标轴
    const yscale = scaleLinear().domain([0, max_size]).range([plot_height, 0]);
    const yaxis = axisLeft(yscale).tickFormat(d3.format('.3s'));

    // X轴比例尺
    const xscale = scaleLinear().domain([0, max_timestep]).range([0, plot_width]);

    // 绘图的坐标空间
    const plot_coordinate_space = svg
      .append('g')
      .attr('transform', `translate(${left_pad}, ${0})`);
    const plot_outer = plot_coordinate_space.append('g');

    // 创建视图矩形
    function view_rect(a) {
      return a
        .append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', plot_width)
        .attr('height', plot_height)
        .attr('fill', 'white');
    }

    // 在外部绘图区域创建视图矩形
    view_rect(plot_outer);

    // 创建裁剪路径
    const cp = svg.append('clipPath').attr('id', 'clip');
    view_rect(cp);
    plot_outer.attr('clip-path', 'url(#clip)');

    // 创建缩放组和刷选组
    const zoom_group = plot_outer.append('g');
    const scrub_group = zoom_group.append('g');

    // 创建多边形图形并绑定数据
    const plot = scrub_group
      .selectAll('polygon')
      .data(data.allocations_over_time)
      .enter()
      .append('polygon')
      .attr('points', format_points)
      .attr('fill', d => colors[d.color % colors.length]);

    // 创建Y轴坐标轴
    const axis = plot_coordinate_space.append('g').call(yaxis);

    // 处理缩放事件
    function handleZoom() {
      const t = d3.event.transform;
      zoom_group.attr('transform', t);
      axis.call(yaxis.scale(d3.event.transform.rescaleY(yscale)));
    }

    // 创建缩放操作
    const thezoom = zoom().on('zoom', handleZoom);
    plot_outer.call(thezoom);

    // 返回对象，包含select_window方法
    return {
      // select_window方法：选择窗口
      select_window: (stepbegin, stepend, max) => {
        const begin = xscale(stepbegin);  // 起始时间步数对应的X轴坐标
        const size = xscale(stepend) - xscale(stepbegin);  // 时间步数区间对应的宽度
        const scale = plot_width / size;  // 缩放比例
        const translate = -begin;  // X轴平移量
        const yscale = max_size / max;  // Y轴缩放比例
        scrub_group.attr(
          'transform',
          `scale(${scale / yscale}, 1) translate(${translate}, 0)`,
        );
        plot_outer.call(
          thezoom.transform,
          zoomIdentity
            .scale(yscale)
            .translate(0, -(plot_height - plot_height / yscale)),
        );
      },
    };
  }
    # 设置委托函数，该函数接受一个代理对象作为参数
    set_delegate: delegate => {
      # 给绘图区域绑定鼠标悬停事件，当鼠标悬停在元素上时触发
      plot
        .on('mouseover', function (_e, _d) {
          # 调用委托对象的方法，将当前被选中的元素传递给委托对象处理
          delegate.set_selected(d3.select(this));
        })
        # 给绘图区域绑定鼠标按下事件，当鼠标按下元素时触发
        .on('mousedown', function (_e, _d) {
          # 设置委托对象的默认选中元素为当前被选中的元素
          delegate.default_selected = d3.select(this);
        })
        # 给绘图区域绑定鼠标离开事件，当鼠标移出元素时触发
        .on('mouseleave', function (_e, _d) {
          # 调用委托对象的方法，将默认选中的元素传递给委托对象处理
          delegate.set_selected(delegate.default_selected);
        });
    },
}

// 定义一个名为 ContextViewer 的函数，用于管理上下文显示及选择状态
function ContextViewer(text, data) {
  let current_selected = null;

  return {
    default_selected: null, // 默认选择为空
    set_selected: d => { // 设置选中的元素
      if (current_selected !== null) { // 如果当前选中不为空
        current_selected.attr('stroke', null).attr('stroke-width', null); // 取消之前选中元素的样式
      }
      if (d === null) { // 如果传入的数据为空
        text.text(''); // 清空文本显示
      } else {
        const dd = d.datum(); // 获取数据绑定
        if (dd.elem === 'summarized') { // 如果元素类型为 'summarized'
          text.html( // 设置 HTML 内容，显示提示信息
            'Small tensors that were not plotted to cutdown on render time.\n' +
              'Use detail slider to see smaller allocations.'
          );
        } else {
          text.text(`${dd.elem} ${data.context_for_id(dd.elem)}`); // 显示元素名称及其上下文
        }
        d.attr('stroke', 'black') // 设置选中元素的边框颜色
          .attr('stroke-width', 1) // 设置选中元素的边框宽度
          .attr('vector-effect', 'non-scaling-stroke'); // 避免边框随缩放而缩放
      }
      current_selected = d; // 更新当前选中元素
    },
  };
}

// 定义一个名为 MiniMap 的函数，用于绘制缩略图
function MiniMap(mini_svg, plot, data, left_pad, width, height = 70) {
  const max_at_time = data.max_at_time; // 获取最大值数组
  const plot_width = width - left_pad; // 计算绘图区域宽度
  const yscale = scaleLinear().domain([0, data.max_size]).range([height, 0]); // 创建 Y 轴比例尺
  const minixscale = scaleLinear() // 创建 X 轴比例尺
    .domain([0, max_at_time.length])
    .range([left_pad, width]);

  const mini_points = [ // 创建缩略图的多边形顶点数组
    [max_at_time.length, 0],
    [0, 0],
  ];

  for (const [i, m] of max_at_time.entries()) { // 遍历最大值数组
    const [_lastx, lasty] = mini_points[mini_points.length - 1]; // 获取最后一个顶点的坐标
    if (m !== lasty) { // 如果当前最大值不等于最后一个顶点的 Y 值
      mini_points.push([i, lasty]); // 添加水平线段顶点
      mini_points.push([i, m]); // 添加竖直线段顶点
    } else if (i === max_at_time.length - 1) { // 如果是数组的最后一个元素
      mini_points.push([i, m]); // 添加最后一个顶点
    }
  }

  let points = mini_points.map(([t, o]) => `${minixscale(t)}, ${yscale(o)}`); // 根据比例尺转换顶点坐标
  points = points.join(' '); // 将顶点坐标转换为字符串
  mini_svg
    .append('polygon') // 在缩略图 SVG 上绘制多边形
    .attr('points', points) // 设置多边形的顶点坐标
    .attr('fill', schemeTableau10[0]); // 设置填充颜色

  const xscale = scaleLinear() // 创建主绘图区域的 X 轴比例尺
    .domain([0, max_at_time.length])
    .range([0, plot_width]);

  const brush = brushX(); // 创建 X 轴刷子
  brush.extent([
    [left_pad, 0], // 刷子范围从 left_pad 开始到 width 结束
    [width, height],
  ]);
  brush.on('brush', function () { // 定义刷子的 brush 事件处理函数
    const [begin, end] = d3.event.selection.map(x => x - left_pad); // 获取刷选范围

    const stepbegin = Math.floor(xscale.invert(begin)); // 计算开始步数
    const stepend = Math.floor(xscale.invert(end)); // 计算结束步数
    let max = 0; // 初始化最大值

    for (let i = stepbegin; i < stepend; i++) { // 遍历刷选范围内的步数
      max = Math.max(max, max_at_time[i]); // 更新最大值
    }
    plot.select_window(stepbegin, stepend, max); // 选择主绘图区域的窗口
  });
  mini_svg.call(brush); // 调用刷子函数应用于缩略图 SVG
  return {}; // 返回空对象
}

// 定义一个名为 Legend 的函数，用于绘制图例
function Legend(plot_svg, categories) {
  const xstart = 100; // 图例 X 起始位置
  const ystart = 5; // 图例 Y 起始位置

  plot_svg
    .append('g') // 在主绘图 SVG 上创建图例组
    .selectAll('rect') // 选择所有矩形元素
    .data(categories) // 绑定数据
    .enter() // 进入数据集
    .append('rect') // 添加矩形元素
    .attr('x', (c, i) => xstart) // 设置矩形 X 坐标
    .attr('y', (c, i) => ystart + i * 15) // 设置矩形 Y 坐标
    .attr('width', 10) // 设置矩形宽度
    .attr('height', 10) // 设置矩形高度
    .attr('fill', (c, i) => schemeTableau10[i % schemeTableau10.length]); // 根据索引设置填充颜色

  plot_svg
    .append('g') // 在主绘图 SVG 上创建文本组
    .selectAll('text') // 选择所有文本元素
    .data(categories) // 绑定数据
    .enter() // 进入数据集
    .append('text') // 添加文本元素
    .attr('x', (c, i) => xstart + 20) // 设置文本 X 坐标
    .attr('y', (c, i) => ystart + i * 15 + 8) // 设置文本 Y 坐标
    .attr('font-family', 'helvetica') // 设置字体
    .attr('font-size', 10) // 设置字号
    .text(c => c); // 设置文本内容为分类名称
}
    .text(c => c);
  return {};


    # 以函数 .text() 的返回值作为参数，传递给函数 c，并返回结果
    .text(c => c);
    # 返回一个空的字典对象
    return {};
// 定义函数 create_trace_view，用于在给定的目标元素中创建内存跟踪视图
function create_trace_view(
  dst, // 目标元素，视图将被创建并附加到此元素上
  snapshot, // 快照数据，用于生成视图的数据源
  device, // 设备信息，可能用于特定设备相关的视图显示
  plot_segments = false, // 是否绘制分段图（可选，默认为 false）
  max_entries = 15000, // 最大条目数（可选，默认为 15000）
) {
  const left_pad = 70; // 左侧填充像素，用于图表布局

  // 处理快照数据，生成适合绘制的分配数据
  const data = process_alloc_data(snapshot, device, plot_segments, max_entries);

  // 清空目标元素中已存在的 SVG 和 DIV 元素
  dst.selectAll('svg').remove();
  dst.selectAll('div').remove();

  // 在目标元素中创建一个新的 DIV 元素
  const d = dst.append('div');

  // 向新创建的 DIV 元素中添加一个 input 元素，用于选择条目数量
  d.append('input')
    .attr('type', 'range')
    .attr('min', 0)
    .attr('max', data.elements_length)
    .attr('value', max_entries)
    .on('change', function () {
      // 当 input 值改变时，重新创建内存跟踪视图并更新条目数量
      create_trace_view(dst, snapshot, device, plot_segments, this.value);
    });
  d.append('label').text('Detail'); // 向 DIV 元素添加一个标签文本

  // 在目标元素中创建一个新的 DIV 元素作为网格容器，并设置样式
  const grid_container = dst
    .append('div')
    .attr(
      'style',
      'display: grid; grid-template-columns: 1fr; grid-template-rows: 10fr 1fr 8fr; height: 100%; gap: 10px',
    );

  // 在网格容器中添加一个 SVG 元素作为主绘图区域，设置视图框和样式
  const plot_svg = grid_container
    .append('svg')
    .attr('display', 'block')
    .attr('viewBox', '0 0 1024 576')
    .attr('preserveAspectRatio', 'none')
    .attr('style', 'grid-column: 1; grid-row: 1; width: 100%; height: 100%;');

  // 在主绘图区域中绘制内存图表，使用 MemoryPlot 函数
  const plot = MemoryPlot(plot_svg, data, left_pad, 1024, 576);

  // 如果快照数据包含类别信息，则在主绘图区域添加图例
  if (snapshot.categories.length !== 0) {
    Legend(plot_svg.append('g'), snapshot.categories);
  }

  // 在网格容器中添加一个 SVG 元素作为缩略图区域，设置视图框和样式
  const mini_svg = grid_container
    .append('svg')
    .attr('display', 'block')
    .attr('viewBox', '0 0 1024 60')
    .attr('preserveAspectRatio', 'none')
    .attr('style', 'grid-column: 1; grid-row: 2; width: 100%; height: 100%;');

  // 在缩略图区域中绘制缩略图，使用 MiniMap 函数
  MiniMap(mini_svg, plot, data, left_pad, 1024);

  // 在网格容器中添加一个 DIV 元素作为上下文查看器容器，设置样式
  const context_div = grid_container
    .append('div')
    .attr(
      'style',
      'grid-column: 1; grid-row: 3; width: 100%; height: 100%; overflow: auto;',
    );

  // 在上下文查看器容器中添加一个预格式化文本区域，并使用 ContextViewer 函数处理数据
  const delegate = ContextViewer(context_div.append('pre').text('none'), data);
  
  // 将上下文查看器委托给内存图表对象
  plot.set_delegate(delegate);
}
// 定义函数用于反序列化缓冲区数据
function unpickle(buffer) {
  // 将输入的缓冲区转换为 Uint8Array 格式
  const bytebuffer = new Uint8Array(buffer);
  // 创建文本解码器对象
  const decoder = new TextDecoder();

  // 声明用于解析过程中的堆栈、标记和备忘录
  const stack = [];
  const marks = [];
  const memo = [];
  let offset = 0; // 初始化偏移量为 0
  let memo_id = 0; // 初始化备忘录 ID 为 0

  // 定义和初始化用于处理特定操作码的常量
  const APPENDS = 'e'.charCodeAt(0);
  const BINGET = 'h'.charCodeAt(0);
  const BININT = 'J'.charCodeAt(0);
  const BININT1 = 'K'.charCodeAt(0);
  const BININT2 = 'M'.charCodeAt(0);
  const EMPTY_DICT = '}'.charCodeAt(0);
  const EMPTY_LIST = ']'.charCodeAt(0);
  const FRAME = 0x95;
  const LONG1 = 0x8a;
  const LONG_BINGET = 'j'.charCodeAt(0);
  const MARK = '('.charCodeAt(0);
  const MEMOIZE = 0x94;
  const PROTO = 0x80;
  const SETITEMS = 'u'.charCodeAt(0);
  const SHORT_BINUNICODE = 0x8c;
  const STOP = '.'.charCodeAt(0);
  const TUPLE2 = 0x86;
  const APPEND = 'a'.charCodeAt(0);
  const NEWFALSE = 0x89;
  const BINPUT = 'q'.charCodeAt(0);
  const BINUNICODE = 'X'.charCodeAt(0);
  const EMPTY_TUPLE = ')'.charCodeAt(0);
  const NEWTRUE = 0x88;
  const NONE = 'N'.charCodeAt(0);
  const BINFLOAT = 'G'.charCodeAt(0);
  const TUPLE = 't'.charCodeAt(0);
  const TUPLE1 = 0x85;
  const TUPLE3 = 0x87;
  // untested
  const LONG_BINPUT = 'r'.charCodeAt(0);
  const LIST = 'l'.charCodeAt(0);
  const DICT = 'd'.charCodeAt(0);
  const SETITEM = 's'.charCodeAt(0);

  // 创建用于临时存储数据的 ArrayBuffer 和相关的视图
  const scratch_buffer = new ArrayBuffer(8);
  const scratch_bytes = new Uint8Array(scratch_buffer);
  const big = new BigInt64Array(scratch_buffer);
  const float64 = new Float64Array(scratch_buffer);

  // 定义函数，用于从缓冲区中读取 4 字节无符号整数
  function read_uint4() {
    const n =
      bytebuffer[offset] +
      bytebuffer[offset + 1] * 256 +
      bytebuffer[offset + 2] * 65536 +
      bytebuffer[offset + 3] * 16777216;
    offset += 4;
    return n;
  }
  
  // 定义函数，用于将堆栈中的键值对设置到目标字典中
  function setitems(d, mark) {
    for (let i = mark; i < stack.length; i += 2) {
      d[stack[i]] = stack[i + 1];
    }
    stack.splice(mark, Infinity);
  }

  // 进入无限循环，处理序列化数据的每个操作码
  while (true) {
    const opcode = bytebuffer[offset++];
    // 在这里处理具体的操作码逻辑
    // （这里的代码片段不完整，需要根据具体操作码的定义来编写相应的逻辑）
    // 例如：
    // switch (opcode) {
    //   case APPENDS:
    //     // 处理 APPENDS 操作码
    //     break;
    //   case BINGET:
    //     // 处理 BINGET 操作码
    //     break;
    //   // 其他操作码的处理逻辑
    // }
  }
}

// 定义函数，用于对 base64 编码的输入进行解码
function decode_base64(input) {
  // 定义函数，用于解码单个字符
  function decode_char(i, shift) {
    const nChr = input.charCodeAt(i);
    const r =
      nChr > 64 && nChr < 91
        ? nChr - 65
        : nChr > 96 && nChr < 123
        ? nChr - 71
        : nChr > 47 && nChr < 58
        ? nChr + 4
        : nChr === 43
        ? 62
        : nChr === 47
        ? 63
        : 0;
    return r << shift;
  }

  // 创建用于存储解码后数据的 Uint8Array
  const output = new Uint8Array((input.length / 4) * 3);
  
  // 遍历输入字符串，每次处理 4 个字符进行解码
  for (let i = 0, j = 0; i < input.length; i += 4, j += 3) {
    const u24 =
      decode_char(i, 18) +
      decode_char(i + 1, 12) +
      decode_char(i + 2, 6) +
      decode_char(i + 3);
    output[j] = u24 >> 16;
    output[j + 1] = (u24 >> 8) & 0xff;
    output[j + 2] = u24 & 0xff;
  }
  
  // 返回解码后的数据 ArrayBuffer
  return output.buffer;
}

// 定义对象 kinds，包含不同类型的处理函数映射
const kinds = {
  'Active Memory Timeline': create_trace_view,
  'Allocator State History': create_segment_view,
  'Active Cached Segment Timeline': (dst, snapshot, device) =>
    create_trace_view(dst, snapshot, device, true),
};

// 创建空对象用作快照缓存
const snapshot_cache = {};

// 创建空对象用作快照到加载器的映射
const snapshot_to_loader = {};
const snapshot_to_url = {};  // 创建一个空对象，用于存储快照名称到URL的映射关系
const selection_to_div = {};  // 创建一个空对象，用于存储选择器到div元素的映射关系

const style = `
pre {
  margin: 0px;
}
html, body {
  height: 100%;
  overflow: clip;
}`;  // 定义样式字符串，包含了对<pre>和html, body元素的样式设置

const head = d3.select('head');  // 选择文档的<head>元素
head.append('style').text(style);  // 向<head>元素中添加一个<style>标签，并设置其文本内容为前面定义的样式字符串
const body = d3.select('body');  // 选择文档的<body>元素
const snapshot_select = body.append('select');  // 向<body>元素中添加一个<select>标签，用于显示快照选项
const view = body.append('select');  // 向<body>元素中添加一个<select>标签，用于显示视图选项
for (const x in kinds) {  // 遍历kinds对象的属性
  view.append('option').text(x);  // 向视图选项<select>中添加一个<option>标签，其文本内容为当前属性名x
}
const gpu = body.append('select');  // 向<body>元素中添加一个<select>标签，用于显示GPU选项

function unpickle_and_annotate(data) {  // 定义一个函数，接受data参数，用于反序列化和注释数据
  data = unpickle(data);  // 调用unpickle函数对data进行反序列化处理
  console.log(data);  // 在控制台打印反序列化后的data内容
  annotate_snapshot(data);  // 调用annotate_snapshot函数对data进行注释处理
  return data;  // 返回处理后的data
}

function snapshot_change(f) {  // 定义一个函数，处理快照变化时的逻辑，接受参数f作为当前快照名
  const view_value = view.node().value;  // 获取视图<select>当前选中的值
  let device = Number(gpu.node().value);  // 获取GPU<select>当前选中的值，并转换为数字类型
  const snapshot = snapshot_cache[f];  // 获取快照缓存中对应名称f的快照数据
  gpu.selectAll('option').remove();  // 清空GPU<select>中的所有<option>选项
  const has_segments = {};  // 创建一个空对象，用于存储是否存在段的信息
  for (const s of snapshot.segments) {  // 遍历快照中的段数据
    has_segments[s.device] = true;  // 将段的设备信息标记为true，表示该设备上有段数据
  }
  let device_valid = false;  // 创建一个布尔变量，表示设备是否有效，默认为false
  for (const [i, trace] of snapshot.device_traces.entries()) {  // 遍历快照中的设备跟踪数据
    if (trace.length > 0 || i in has_segments) {  // 如果设备跟踪数据不为空或者设备存在段数据
      gpu.append('option').text(i);  // 向GPU<select>中添加一个<option>标签，其文本内容为设备名称i
      if (i === device) {  // 如果当前设备名称i与选中设备相同
        device_valid = true;  // 标记设备有效为true
        gpu.node().selectedIndex = gpu.node().children.length - 1;  // 设置GPU<select>的选中项为最后添加的项
      }
    }
  }
  if (!device_valid) {  // 如果设备无效
    device = Number(gpu.node().value);  // 获取GPU<select>当前选中的值，并转换为数字类型
  }
  const key = [f, view_value, device];  // 创建一个数组key，存储当前快照名、视图值和设备值
  if (!(key in selection_to_div)) {  // 如果key不在选择器到div映射中
    selection_to_div[key] = d3.select('body').append('div');  // 向<body>元素中添加一个<div>元素，并存储到选择器到div映射中
    kinds[view_value](selection_to_div[key], snapshot, device);  // 调用kinds对象中对应视图值的函数，传入新创建的<div>、快照和设备信息
  }
  const selected_div = selection_to_div[key];  // 获取选择器对应的<div>元素

  selected_div.attr('style', 'display: float; height: 100%');  // 设置选中的<div>元素样式，使其显示为浮动且高度为100%
}

function selected_change() {  // 定义一个函数，处理选择变化时的逻辑
  for (const d of Object.values(selection_to_div)) {  // 遍历选择器到div映射中的所有值
    d.attr('style', 'display: none; height: 100%');  // 将每个<div>元素的样式设置为不显示且高度为100%
  }
  const f = snapshot_select.node().value;  // 获取快照<select>当前选中的值
  if (f === '') {  // 如果快照名为空
    return;  // 结束函数执行
  }
  if (!(f in snapshot_cache)) {  // 如果快照名不在快照缓存中
    snapshot_to_loader[f](f);  // 调用快照加载器中对应快照名的函数
  } else {  // 否则
    snapshot_change(f);  // 调用快照变化处理函数，传入当前快照名f
  }
}

snapshot_select.on('change', selected_change);  // 监听快照<select>的change事件，触发selected_change函数
view.on('change', selected_change);  // 监听视图<select>的change事件，触发selected_change函数
gpu.on('change', selected_change);  // 监听GPU<select>的change事件，触发selected_change函数

body.on('dragover', e => {  // 监听<body>元素的dragover事件，当拖拽操作在元素上时触发
  event.preventDefault();  // 阻止事件的默认行为
});

body.on('drop', () => {  // 监听<body>元素的drop事件，当拖拽操作结束时触发
  console.log(event.dataTransfer.files);  // 在控制台打印拖拽操作中传输的文件信息
  Array.from(event.dataTransfer.files).forEach(file => {  // 遍历拖拽操作中传输的每一个文件
    add_snapshot(file.name, unique_name => {  // 调用add_snapshot函数，传入文件名和一个处理加载完成的回调函数
      const reader = new FileReader();  // 创建一个FileReader对象
      reader.onload = e => {  // 定义文件加载完成时的处理函数
        finished_loading(unique_name, e.target.result);  // 调用加载完成处理函数，传入唯一名称和加载的结果
      };
      reader.readAsArrayBuffer(file);  // 以ArrayBuffer格式读取文件内容
    });
  });
  event.preventDefault();  // 阻止事件的默认行为
  snapshot_select.node().selectedIndex =
    snapshot_select.node().options.length - 1;  // 将快照<select>的选中项设置为最后一项
  selected_change();  // 调用选择变化处理函数
});

selection_to_div[''] = body
  .append('div')
  .text(
    'Drag and drop a file to load a local snapshot. No data from the snapshot is uploaded.',
  );  // 将空字符串作为键，添加一个包含指定文本内容的<div>元素到<body>中

let next_unique_n = 1;  // 创建一个变量，用于生成唯一编号，初始值为1
function add_snapshot(name, loader) {  // 定义一个函数，用于添加快照
  if (name in snapshot_to_loader) {  // 如果指定名称的快照已经存在
    name = `${name} (${next_unique_n++})`;  //
function finished_loading(name, data) {
  // 将数据解析和注释，并将结果存储到快照缓存中
  snapshot_cache[name] = unpickle_and_annotate(data);
  // 通知快照已经改变
  snapshot_change(name);
}

export function add_remote_files(files) {
  // 对每一个远程文件执行操作
  files.forEach(f =>
    // 添加快照，使用文件名和唯一名称作为参数
    add_snapshot(f.name, unique_name => {
      // 打印日志，显示正在获取文件
      console.log('fetching', f.url);
      // 从远程 URL 获取数据，将其转换为 ArrayBuffer
      fetch(f.url)
        .then(x => x.arrayBuffer())
        .then(data => finished_loading(unique_name, data));
    }),
  );
  // 如果有文件被添加，则触发选中改变事件
  if (files.length > 0) {
    selected_change();
  }
}

export function add_local_files(files, view_value) {
  // 设置视图节点的值为给定视图值
  view.node().value = view_value;
  // 对每一个本地文件执行操作
  files.forEach(f =>
    // 添加快照，使用文件名和唯一名称作为参数
    add_snapshot(f.name, unique_name => {
      // 使用 Base64 解码本地文件的数据，并完成加载
      finished_loading(unique_name, decode_base64(f.base64));
    }),
  );
  // 如果有文件被添加，则触发选中改变事件
  if (files.length > 0) {
    selected_change();
  }
}
```