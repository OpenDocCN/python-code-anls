# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\search\matchesonscrollbar.js`

```
// 将代码封装在立即执行函数中，传入 CodeMirror 对象
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"), require("./searchcursor"), require("../scroll/annotatescrollbar"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror", "./searchcursor", "../scroll/annotatescrollbar"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义 CodeMirror 的扩展方法 showMatchesOnScrollbar
  CodeMirror.defineExtension("showMatchesOnScrollbar", function(query, caseFold, options) {
    // 如果 options 是字符串，则转换为对象
    if (typeof options == "string") options = {className: options};
    // 如果 options 不存在，则初始化为空对象
    if (!options) options = {};
    // 创建 SearchAnnotation 实例
    return new SearchAnnotation(this, query, caseFold, options);
  });

  // 定义 SearchAnnotation 类
  function SearchAnnotation(cm, query, caseFold, options) {
    this.cm = cm;
    this.options = options;
    // 初始化 annotateOptions 对象
    var annotateOptions = {listenForChanges: false};
    // 将 options 的属性复制到 annotateOptions
    for (var prop in options) annotateOptions[prop] = options[prop];
    // 如果 annotateOptions 中没有 className 属性，则设置为默认值 "CodeMirror-search-match"
    if (!annotateOptions.className) annotateOptions.className = "CodeMirror-search-match";
    // 在滚动条上创建注解
    this.annotation = cm.annotateScrollbar(annotateOptions);
    this.query = query;
    this.caseFold = caseFold;
    // 初始化 gap 对象
    this.gap = {from: cm.firstLine(), to: cm.lastLine() + 1};
    this.matches = [];
    this.update = null;

    // 查找匹配项并更新注解
    this.findMatches();
    this.annotation.update(this.matches);

    // 监听编辑器内容变化事件
    var self = this;
    cm.on("change", this.changeHandler = function(_cm, change) { self.onChange(change); });
  }

  // 定义最大匹配项数量
  var MAX_MATCHES = 1000;

  // 查找匹配项的方法
  SearchAnnotation.prototype.findMatches = function() {
    // 如果 gap 不存在，则直接返回
    if (!this.gap) return;
    // 遍历匹配项数组
    for (var i = 0; i < this.matches.length; i++) {
      var match = this.matches[i];
      // 如果匹配项的起始行大于等于 gap 的结束行，则跳出循环
      if (match.from.line >= this.gap.to) break;
      // 如果匹配项的结束行大于等于 gap 的起始行，则从匹配项数组中删除该匹配项
      if (match.to.line >= this.gap.from) this.matches.splice(i--, 1);
    }
    # 根据查询内容和位置创建搜索游标对象
    var cursor = this.cm.getSearchCursor(this.query, CodeMirror.Pos(this.gap.from, 0), {caseFold: this.caseFold, multiline: this.options.multiline});
    # 设置最大匹配数
    var maxMatches = this.options && this.options.maxMatches || MAX_MATCHES;
    # 循环查找匹配项
    while (cursor.findNext()) {
      # 获取匹配项的起始和结束位置
      var match = {from: cursor.from(), to: cursor.to()};
      # 如果匹配项的起始行大于等于 gap 的结束行，则跳出循环
      if (match.from.line >= this.gap.to) break;
      # 将匹配项插入到 matches 数组中
      this.matches.splice(i++, 0, match);
      # 如果匹配项数量超过最大匹配数，则跳出循环
      if (this.matches.length > maxMatches) break;
    }
    # 重置 gap 为 null
    this.gap = null;
  };

  # 根据改变的起始行和大小改变，计算偏移后的行数
  function offsetLine(line, changeStart, sizeChange) {
    if (line <= changeStart) return line;
    return Math.max(changeStart, line + sizeChange);
  }

  # 当搜索注释对象发生改变时的处理函数
  SearchAnnotation.prototype.onChange = function(change) {
    # 获取改变的起始行和结束行
    var startLine = change.from.line;
    var endLine = CodeMirror.changeEnd(change).line;
    # 计算大小改变
    var sizeChange = endLine - change.to.line;
    # 如果存在 gap
    if (this.gap) {
      # 更新 gap 的起始行和结束行
      this.gap.from = Math.min(offsetLine(this.gap.from, startLine, sizeChange), change.from.line);
      this.gap.to = Math.max(offsetLine(this.gap.to, startLine, sizeChange), change.from.line);
    } else {
      # 创建新的 gap 对象
      this.gap = {from: change.from.line, to: endLine + 1};
    }

    # 如果存在大小改变，更新匹配项的位置
    if (sizeChange) for (var i = 0; i < this.matches.length; i++) {
      var match = this.matches[i];
      var newFrom = offsetLine(match.from.line, startLine, sizeChange);
      if (newFrom != match.from.line) match.from = CodeMirror.Pos(newFrom, match.from.ch);
      var newTo = offsetLine(match.to.line, startLine, sizeChange);
      if (newTo != match.to.line) match.to = CodeMirror.Pos(newTo, match.to.ch);
    }
    # 清除更新定时器
    clearTimeout(this.update);
    # 设置 self 为 this
    var self = this;
    # 设置更新定时器
    this.update = setTimeout(function() { self.updateAfterChange(); }, 250);
  };

  # 在改变后更新匹配项
  SearchAnnotation.prototype.updateAfterChange = function() {
    this.findMatches();
    this.annotation.update(this.matches);
  };

  # 清除搜索注释对象
  SearchAnnotation.prototype.clear = function() {
    this.cm.off("change", this.changeHandler);
    this.annotation.clear();
  };
# 闭合了一个代码块或者函数的结束
```