# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\selection\mark-selection.js`

```
// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

// 因为有时需要标记所选的*文本*。
//
// 添加一个名为'styleSelectedText'的选项，当启用时，给所选文本添加指定的 CSS 类，或者当值不是字符串时，添加"CodeMirror-selectedtext"类。

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义'styleSelectedText'选项
  CodeMirror.defineOption("styleSelectedText", false, function(cm, val, old) {
    var prev = old && old != CodeMirror.Init;
    if (val && !prev) {
      // 初始化标记选择的文本数组和样式
      cm.state.markedSelection = [];
      cm.state.markedSelectionStyle = typeof val == "string" ? val : "CodeMirror-selectedtext";
      reset(cm);
      // 监听光标活动事件和文本改变事件
      cm.on("cursorActivity", onCursorActivity);
      cm.on("change", onChange);
    } else if (!val && prev) {
      // 取消监听光标活动事件和文本改变事件
      cm.off("cursorActivity", onCursorActivity);
      cm.off("change", onChange);
      clear(cm);
      cm.state.markedSelection = cm.state.markedSelectionStyle = null;
    }
  });

  // 光标活动事件处理函数
  function onCursorActivity(cm) {
    if (cm.state.markedSelection)
      cm.operation(function() { update(cm); });
  }

  // 文本改变事件处理函数
  function onChange(cm) {
    if (cm.state.markedSelection && cm.state.markedSelection.length)
      cm.operation(function() { clear(cm); });
  }

  // 定义常量和函数
  var CHUNK_SIZE = 8;
  var Pos = CodeMirror.Pos;
  var cmp = CodeMirror.cmpPos;

  // 覆盖范围函数
  function coverRange(cm, from, to, addAt) {
    if (cmp(from, to) == 0) return;
    var array = cm.state.markedSelection;
    var cls = cm.state.markedSelectionStyle;
    # 从起始行开始循环处理
    for (var line = from.line;;) {
      # 如果当前行等于起始行，则将起始位置设为起始点，否则将起始位置设为当前行的第一个字符
      var start = line == from.line ? from : Pos(line, 0);
      # 计算结束行的位置，判断是否已经到达目标行
      var endLine = line + CHUNK_SIZE, atEnd = endLine >= to.line;
      # 如果已经到达目标行，则将结束位置设为目标点，否则将结束位置设为结束行的第一个字符
      var end = atEnd ? to : Pos(endLine, 0);
      # 在编辑器中标记起始位置到结束位置的文本，并设置样式
      var mark = cm.markText(start, end, {className: cls});
      # 如果没有指定插入位置，则将标记添加到数组末尾，否则在指定位置插入标记
      if (addAt == null) array.push(mark);
      else array.splice(addAt++, 0, mark);
      # 如果已经到达目标行，则跳出循环
      if (atEnd) break;
      # 更新当前行为结束行
      line = endLine;
    }
  }

  # 清除编辑器中的标记
  function clear(cm) {
    # 获取编辑器中的标记数组
    var array = cm.state.markedSelection;
    # 循环清除所有标记
    for (var i = 0; i < array.length; ++i) array[i].clear();
    # 清空标记数组
    array.length = 0;
  }

  # 重置编辑器中的标记
  function reset(cm) {
    # 清除编辑器中的标记
    clear(cm);
    # 获取编辑器中的选择范围数组
    var ranges = cm.listSelections();
    # 遍历选择范围数组，重新标记文本
    for (var i = 0; i < ranges.length; i++)
      coverRange(cm, ranges[i].from(), ranges[i].to());
  }

  # 更新编辑器中的标记
  function update(cm) {
    # 如果没有选中文本，则清除编辑器中的标记
    if (!cm.somethingSelected()) return clear(cm);
    # 如果选择范围数量大于1，则重置编辑器中的标记
    if (cm.listSelections().length > 1) return reset(cm);

    # 获取选中文本的起始位置和结束位置
    var from = cm.getCursor("start"), to = cm.getCursor("end");

    # 获取编辑器中的标记数组
    var array = cm.state.markedSelection;
    # 如果标记数组为空，则标记选中文本
    if (!array.length) return coverRange(cm, from, to);

    # 获取第一个标记的起始位置和最后一个标记的结束位置
    var coverStart = array[0].find(), coverEnd = array[array.length - 1].find();
    # 如果起始位置或结束位置为空，或者选中文本行数小于等于CHUNK_SIZE，或者选中文本超出已有标记范围，则重置编辑器中的标记
    if (!coverStart || !coverEnd || to.line - from.line <= CHUNK_SIZE ||
        cmp(from, coverEnd.to) >= 0 || cmp(to, coverStart.from) <= 0)
      return reset(cm);

    # 循环移除超出选中文本范围的标记
    while (cmp(from, coverStart.from) > 0) {
      array.shift().clear();
      coverStart = array[0].find();
    }
    # 如果起始位置小于第一个标记的起始位置，则根据条件重新标记文本
    if (cmp(from, coverStart.from) < 0) {
      if (coverStart.to.line - from.line < CHUNK_SIZE) {
        array.shift().clear();
        coverRange(cm, from, coverStart.to, 0);
      } else {
        coverRange(cm, from, coverStart.from, 0);
      }
    }

    # 循环移除超出选中文本范围的标记
    while (cmp(to, coverEnd.to) < 0) {
      array.pop().clear();
      coverEnd = array[array.length - 1].find();
    }
    # 如果 to 大于 coverEnd.to
    if (cmp(to, coverEnd.to) > 0) {
      # 如果 to 的行数减去 coverEnd.from 的行数小于 CHUNK_SIZE
      if (to.line - coverEnd.from.line < CHUNK_SIZE) {
        # 弹出数组最后一个元素并清空
        array.pop().clear();
        # 覆盖范围从 coverEnd.from 到 to
        coverRange(cm, coverEnd.from, to);
      } else {
        # 覆盖范围从 coverEnd.to 到 to
        coverRange(cm, coverEnd.to, to);
      }
    }
  }
# 闭合了一个代码块或者函数的结束
```