# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\fold\foldgutter.js`

```
// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"), require("./foldcode"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror", "./foldcode"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  CodeMirror.defineOption("foldGutter", false, function(cm, val, old) {
    if (old && old != CodeMirror.Init) {
      cm.clearGutter(cm.state.foldGutter.options.gutter);
      cm.state.foldGutter = null;
      cm.off("gutterClick", onGutterClick);
      cm.off("changes", onChange);
      cm.off("viewportChange", onViewportChange);
      cm.off("fold", onFold);
      cm.off("unfold", onFold);
      cm.off("swapDoc", onChange);
    }
    if (val) {
      cm.state.foldGutter = new State(parseOptions(val));
      updateInViewport(cm);
      cm.on("gutterClick", onGutterClick);
      cm.on("changes", onChange);
      cm.on("viewportChange", onViewportChange);
      cm.on("fold", onFold);
      cm.on("unfold", onFold);
      cm.on("swapDoc", onChange);
    }
  });

  var Pos = CodeMirror.Pos;

  function State(options) {
    this.options = options;
    this.from = this.to = 0;
  }

  function parseOptions(opts) {
    if (opts === true) opts = {};
    if (opts.gutter == null) opts.gutter = "CodeMirror-foldgutter";
    if (opts.indicatorOpen == null) opts.indicatorOpen = "CodeMirror-foldgutter-open";
    if (opts.indicatorFolded == null) opts.indicatorFolded = "CodeMirror-foldgutter-folded";
    return opts;
  }

  function isFolded(cm, line) {
    var marks = cm.findMarks(Pos(line, 0), Pos(line + 1, 0));
    # 遍历 marks 数组
    for (var i = 0; i < marks.length; ++i) {
      # 如果 marks[i] 的 __isFold 属性为真
      if (marks[i].__isFold) {
        # 查找 marks[i] 中值为 -1 的位置
        var fromPos = marks[i].find(-1);
        # 如果 fromPos 存在且其行号等于 line
        if (fromPos && fromPos.line === line)
          # 返回 marks[i]
          return marks[i];
      }
    }
  }

  # 根据 spec 参数创建 marker 元素
  function marker(spec) {
    # 如果 spec 的类型为字符串
    if (typeof spec == "string") {
      # 创建一个 div 元素
      var elt = document.createElement("div");
      # 设置 div 元素的类名
      elt.className = spec + " CodeMirror-guttermarker-subtle";
      # 返回创建的 div 元素
      return elt;
    } else {
      # 返回 spec 的克隆节点
      return spec.cloneNode(true);
    }
  }

  # 更新折叠信息
  function updateFoldInfo(cm, from, to) {
    # 获取折叠选项和当前行号
    var opts = cm.state.foldGutter.options, cur = from - 1;
    # 获取最小折叠大小和范围查找器
    var minSize = cm.foldOption(opts, "minFoldSize");
    var func = cm.foldOption(opts, "rangeFinder");
    # 如果内置指示器元素的类名与新状态匹配，则可以重用它
    var clsFolded = typeof opts.indicatorFolded == "string" && classTest(opts.indicatorFolded);
    var clsOpen = typeof opts.indicatorOpen == "string" && classTest(opts.indicatorOpen);
    # 遍历 from 到 to 之间的每一行
    cm.eachLine(from, to, function(line) {
      ++cur;
      var mark = null;
      var old = line.gutterMarkers;
      # 如果 old 存在且其类名与 gutter 匹配
      if (old) old = old[opts.gutter];
      # 如果当前行被折叠
      if (isFolded(cm, cur)) {
        # 如果 clsFolded 存在且 old 存在且其类名与 clsFolded 匹配，则返回
        if (clsFolded && old && clsFolded.test(old.className)) return;
        # 创建折叠指示器元素
        mark = marker(opts.indicatorFolded);
      } else {
        # 获取当前行的位置和范围
        var pos = Pos(cur, 0);
        var range = func && func(cm, pos);
        # 如果范围存在且其行数大于等于最小折叠大小
        if (range && range.to.line - range.from.line >= minSize) {
          # 如果 clsOpen 存在且 old 存在且其类名与 clsOpen 匹配，则返回
          if (clsOpen && old && clsOpen.test(old.className)) return;
          # 创建展开指示器元素
          mark = marker(opts.indicatorOpen);
        }
      }
      # 如果 mark 和 old 都不存在，则返回
      if (!mark && !old) return;
      # 设置当前行的 gutterMarker
      cm.setGutterMarker(line, opts.gutter, mark);
    });
  }

  # 从 CodeMirror/src/util/dom.js 复制过来的函数
  function classTest(cls) { return new RegExp("(^|\\s)" + cls + "(?:$|\\s)\\s*") }

  # 更新视口中的内容
  function updateInViewport(cm) {
    # 获取编辑器的视口和折叠状态
    var vp = cm.getViewport(), state = cm.state.foldGutter;
    # 如果折叠状态不存在，则返回
    if (!state) return;
    # 执行更新折叠信息的操作
    cm.operation(function() {
      updateFoldInfo(cm, vp.from, vp.to);
    });
  // 将视口的起始和结束行号赋值给状态对象
  state.from = vp.from; state.to = vp.to;
}

function onGutterClick(cm, line, gutter) {
  // 获取折叠栏状态对象
  var state = cm.state.foldGutter;
  // 如果状态对象不存在，则返回
  if (!state) return;
  // 获取状态对象的选项
  var opts = state.options;
  // 如果点击的折叠栏不是指定的折叠栏，则返回
  if (gutter != opts.gutter) return;
  // 判断当前行是否已经折叠，如果是则清除折叠，否则折叠当前行
  var folded = isFolded(cm, line);
  if (folded) folded.clear();
  else cm.foldCode(Pos(line, 0), opts);
}

function onChange(cm) {
  // 获取折叠栏状态对象
  var state = cm.state.foldGutter;
  // 如果状态对象不存在，则返回
  if (!state) return;
  // 获取状态对象的选项
  var opts = state.options;
  // 将状态对象的起始和结束行号都设置为0
  state.from = state.to = 0;
  // 清除之前的定时器，设置新的定时器来更新视口中的折叠信息
  clearTimeout(state.changeUpdate);
  state.changeUpdate = setTimeout(function() { updateInViewport(cm); }, opts.foldOnChangeTimeSpan || 600);
}

function onViewportChange(cm) {
  // 获取折叠栏状态对象
  var state = cm.state.foldGutter;
  // 如果状态对象不存在，则返回
  if (!state) return;
  // 获取状态对象的选项
  var opts = state.options;
  // 清除之前的定时器，设置新的定时器来更新视口中的折叠信息
  clearTimeout(state.changeUpdate);
  state.changeUpdate = setTimeout(function() {
    // 获取当前视口的起始和结束行号
    var vp = cm.getViewport();
    // 如果状态对象的起始和结束行号相等，或者视口的起始行号与状态对象的结束行号之差大于20，或者状态对象的起始行号与视口的结束行号之差大于20，则更新视口中的折叠信息
    if (state.from == state.to || vp.from - state.to > 20 || state.from - vp.to > 20) {
      updateInViewport(cm);
    } else {
      // 否则，进行操作来更新折叠信息
      cm.operation(function() {
        if (vp.from < state.from) {
          updateFoldInfo(cm, vp.from, state.from);
          state.from = vp.from;
        }
        if (vp.to > state.to) {
          updateFoldInfo(cm, state.to, vp.to);
          state.to = vp.to;
        }
      });
    }
  }, opts.updateViewportTimeSpan || 400);
}

function onFold(cm, from) {
  // 获取折叠栏状态对象
  var state = cm.state.foldGutter;
  // 如果状态对象不存在，则返回
  if (!state) return;
  // 获取折叠的行号
  var line = from.line;
  // 如果折叠的行号在状态对象的起始和结束行号之间，则更新折叠信息
  if (line >= state.from && line < state.to)
    updateFoldInfo(cm, line, line + 1);
}
# 闭合了一个代码块或者函数的结束
```