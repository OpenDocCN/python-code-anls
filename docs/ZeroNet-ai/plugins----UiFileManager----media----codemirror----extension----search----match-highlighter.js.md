# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\search\match-highlighter.js`

```py
// 定义一个匿名函数，传入一个 mod 参数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"), require("./matchesonscrollbar"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror", "./matchesonscrollbar"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 默认配置项
  var defaults = {
    style: "matchhighlight",
    minChars: 2,
    delay: 100,
    wordsOnly: false,
    annotateScrollbar: false,
    showToken: false,
    trim: true
  }

  // State 对象构造函数
  function State(options) {
    this.options = {}
    // 遍历默认配置项，将传入的配置项覆盖默认配置项
    for (var name in defaults)
      this.options[name] = (options && options.hasOwnProperty(name) ? options : defaults)[name]
    this.overlay = this.timeout = null;
    this.matchesonscroll = null;
    # 设置 active 属性为 false
    this.active = false;
  }

  # 定义 highlightSelectionMatches 选项
  CodeMirror.defineOption("highlightSelectionMatches", false, function(cm, val, old) {
    # 如果旧值存在且不是初始值，则移除覆盖层，清除定时器，取消事件监听
    if (old && old != CodeMirror.Init) {
      removeOverlay(cm);
      clearTimeout(cm.state.matchHighlighter.timeout);
      cm.state.matchHighlighter = null;
      cm.off("cursorActivity", cursorActivity);
      cm.off("focus", onFocus)
    }
    # 如果新值为 true
    if (val) {
      # 创建状态对象
      var state = cm.state.matchHighlighter = new State(val);
      # 如果编辑器有焦点，则设置状态为 active，并高亮匹配项
      if (cm.hasFocus()) {
        state.active = true
        highlightMatches(cm)
      } else {
        # 否则，在焦点事件时高亮匹配项
        cm.on("focus", onFocus)
      }
      # 在光标活动时触发事件
      cm.on("cursorActivity", cursorActivity);
    }
  });

  # 光标活动时的处理函数
  function cursorActivity(cm) {
    var state = cm.state.matchHighlighter;
    # 如果状态为 active 或编辑器有焦点，则调度高亮匹配项
    if (state.active || cm.hasFocus()) scheduleHighlight(cm, state)
  }

  # 编辑器获得焦点时的处理函数
  function onFocus(cm) {
    var state = cm.state.matchHighlighter
    # 如果状态不为 active，则设置状态为 active，并调度高亮匹配项
    if (!state.active) {
      state.active = true
      scheduleHighlight(cm, state)
    }
  }

  # 调度高亮匹配项的函数
  function scheduleHighlight(cm, state) {
    clearTimeout(state.timeout);
    state.timeout = setTimeout(function() {highlightMatches(cm);}, state.options.delay);
  }

  # 添加覆盖层的函数
  function addOverlay(cm, query, hasBoundary, style) {
    var state = cm.state.matchHighlighter;
    # 添加覆盖层
    cm.addOverlay(state.overlay = makeOverlay(query, hasBoundary, style));
    # 如果选项为 annotateScrollbar 且编辑器支持在滚动条上显示匹配项
    if (state.options.annotateScrollbar && cm.showMatchesOnScrollbar) {
      # 创建搜索正则表达式
      var searchFor = hasBoundary ? new RegExp((/\w/.test(query.charAt(0)) ? "\\b" : "") +
                                               query.replace(/[\\\[.+*?(){|^$]/g, "\\$&") +
                                               (/\w/.test(query.charAt(query.length - 1)) ? "\\b" : "")) : query;
      # 在滚动条上显示匹配项
      state.matchesonscroll = cm.showMatchesOnScrollbar(searchFor, false,
        {className: "CodeMirror-selection-highlight-scrollbar"});
    }
  }

  # 移除覆盖层的函数
  function removeOverlay(cm) {
    var state = cm.state.matchHighlighter;
    # 如果存在覆盖层
    if (state.overlay):
      # 移除覆盖层
      cm.removeOverlay(state.overlay);
      # 重置覆盖层为 null
      state.overlay = null;
      # 如果滚动时匹配存在
      if (state.matchesonscroll):
        # 清空滚动时匹配
        state.matchesonscroll.clear();
        # 重置滚动时匹配为 null
        state.matchesonscroll = null;
  
  # 高亮匹配
  def highlightMatches(cm):
    # 执行操作
    cm.operation(function():
      # 获取匹配高亮状态
      var state = cm.state.matchHighlighter;
      # 移除覆盖层
      removeOverlay(cm);
      # 如果没有选中内容且选项为显示标记
      if (!cm.somethingSelected() && state.options.showToken):
        # 获取当前光标位置和所在行内容
        var cur = cm.getCursor(), line = cm.getLine(cur.line), start = cur.ch, end = start;
        # 查找标记的起始和结束位置
        while (start && re.test(line.charAt(start - 1))) --start;
        while (end < line.length && re.test(line.charAt(end))) ++end;
        # 如果存在标记
        if (start < end)
          # 添加覆盖层
          addOverlay(cm, line.slice(start, end), re, state.options.style);
        return;
      # 获取选中内容的起始和结束位置
      var from = cm.getCursor("from"), to = cm.getCursor("to");
      # 如果起始和结束在不同行
      if (from.line != to.line) return;
      # 如果选项为仅限单词且不是单词
      if (state.options.wordsOnly && !isWord(cm, from, to)) return;
      # 获取选中内容
      var selection = cm.getRange(from, to)
      # 如果选项为修剪空白字符
      if (state.options.trim) selection = selection.replace(/^\s+|\s+$/g, "")
      # 如果选中内容长度大于等于最小字符数
      if (selection.length >= state.options.minChars)
        # 添加覆盖层
        addOverlay(cm, selection, false, state.options.style);
    );
  
  # 判断是否为单词
  def isWord(cm, from, to):
    # 获取选中内容
    var str = cm.getRange(from, to);
    # 如果匹配单词
    if (str.match(/^\w+$/) !== null):
        # 如果起始位置大于 0
        if (from.ch > 0):
            # 获取前一个字符
            var pos = {line: from.line, ch: from.ch - 1};
            var chr = cm.getRange(pos, from);
            # 如果前一个字符不是非单词字符
            if (chr.match(/\W/) === null) return false;
        # 如果结束位置小于所在行长度
        if (to.ch < cm.getLine(from.line).length):
            # 获取后一个字符
            var pos = {line: to.line, ch: to.ch + 1};
            var chr = cm.getRange(to, pos);
            # 如果后一个字符不是非单词字符
            if (chr.match(/\W/) === null) return false;
        return true;
    else return false;
  
  # 获取边界周围的内容
  def boundariesAround(stream, re):
    # 返回一个布尔值，判断当前位置是否在字符串的开头或者前一个字符不符合正则表达式要求，并且当前位置在字符串末尾或者后一个字符不符合正则表达式要求
    return (!stream.start || !re.test(stream.string.charAt(stream.start - 1))) &&
      (stream.pos == stream.string.length || !re.test(stream.string.charAt(stream.pos)));
  }

  # 创建一个覆盖层，用于匹配指定的查询字符串，并根据条件和样式返回结果
  function makeOverlay(query, hasBoundary, style) {
    return {token: function(stream) {
      # 如果匹配到查询字符串，并且满足边界条件（如果有），则返回指定的样式
      if (stream.match(query) &&
          (!hasBoundary || boundariesAround(stream, hasBoundary)))
        return style;
      # 否则，继续向前匹配，直到匹配到查询字符串的第一个字符，或者匹配到字符串末尾
      stream.next();
      stream.skipTo(query.charAt(0)) || stream.skipToEnd();
    }};
  }
# 闭合了一个代码块或者函数的结束
```