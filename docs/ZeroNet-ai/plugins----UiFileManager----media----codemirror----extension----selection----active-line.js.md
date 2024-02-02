# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\selection\active-line.js`

```py
// 使用立即执行函数表达式（IIFE）来封装代码，防止变量污染全局作用域
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 导入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 导入 CodeMirror 模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接导入 CodeMirror 模块
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";
  // 定义活动行的样式类名
  var WRAP_CLASS = "CodeMirror-activeline";
  var BACK_CLASS = "CodeMirror-activeline-background";
  var GUTT_CLASS = "CodeMirror-activeline-gutter";

  // 定义 CodeMirror 的 styleActiveLine 选项
  CodeMirror.defineOption("styleActiveLine", false, function(cm, val, old) {
    // 获取之前的选项值
    var prev = old == CodeMirror.Init ? false : old;
    // 如果新值和之前的值相同，则不做任何操作
    if (val == prev) return
    // 如果之前有活动行样式，移除事件监听和样式类
    if (prev) {
      cm.off("beforeSelectionChange", selectionChange);
      clearActiveLines(cm);
      delete cm.state.activeLines;
    }
    // 如果新值为 true，添加事件监听和样式类
    if (val) {
      cm.state.activeLines = [];
      updateActiveLines(cm, cm.listSelections());
      cm.on("beforeSelectionChange", selectionChange);
    }
  });

  // 清除活动行的样式
  function clearActiveLines(cm) {
    for (var i = 0; i < cm.state.activeLines.length; i++) {
      cm.removeLineClass(cm.state.activeLines[i], "wrap", WRAP_CLASS);
      cm.removeLineClass(cm.state.activeLines[i], "background", BACK_CLASS);
      cm.removeLineClass(cm.state.activeLines[i], "gutter", GUTT_CLASS);
    }
  }

  // 比较两个数组是否相同
  function sameArray(a, b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++)
      if (a[i] != b[i]) return false;
    return true;
  }

  // 更新活动行的样式
  function updateActiveLines(cm, ranges) {
    var active = [];
    # 遍历ranges数组
    for (var i = 0; i < ranges.length; i++) {
      # 获取当前range
      var range = ranges[i];
      # 获取styleActiveLine选项
      var option = cm.getOption("styleActiveLine");
      # 判断styleActiveLine选项是否为对象且非空，如果是则判断range的起始行和结束行是否相同，如果不是则判断range是否为空
      if (typeof option == "object" && option.nonEmpty ? range.anchor.line != range.head.line : !range.empty())
        # 如果条件成立则跳过本次循环
        continue
      # 获取range头部行的可视起始行
      var line = cm.getLineHandleVisualStart(range.head.line);
      # 如果active数组中最后一个元素不等于line，则将line添加到active数组中
      if (active[active.length - 1] != line) active.push(line);
    }
    # 判断cm.state.activeLines和active数组是否相同，如果相同则返回
    if (sameArray(cm.state.activeLines, active)) return;
    # 执行操作
    cm.operation(function() {
      # 清除activeLines
      clearActiveLines(cm);
      # 遍历active数组
      for (var i = 0; i < active.length; i++) {
        # 为active数组中的每个元素添加wrap类
        cm.addLineClass(active[i], "wrap", WRAP_CLASS);
        # 为active数组中的每个元素添加background类
        cm.addLineClass(active[i], "background", BACK_CLASS);
        # 为active数组中的每个元素添加gutter类
        cm.addLineClass(active[i], "gutter", GUTT_CLASS);
      }
      # 将active数组赋值给cm.state.activeLines
      cm.state.activeLines = active;
    });
  }

  # 当选择发生变化时调用
  function selectionChange(cm, sel) {
    # 更新activeLines
    updateActiveLines(cm, sel.ranges);
  }
# 闭合了一个代码块或者函数的结束
```