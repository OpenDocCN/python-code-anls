# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\edit\matchtags.js`

```py
// 将整个代码块包裹在一个立即执行的匿名函数中，传入CodeMirror对象
(function(mod) {
  // 如果是CommonJS环境，使用require引入依赖
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"), require("../fold/xml-fold"));
  // 如果是AMD环境，使用define引入依赖
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror", "../fold/xml-fold"], mod);
  // 如果是普通的浏览器环境，直接传入CodeMirror对象
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义CodeMirror的matchTags选项
  CodeMirror.defineOption("matchTags", false, function(cm, val, old) {
    // 如果旧值存在且不是初始值，则移除事件监听器并清除标记
    if (old && old != CodeMirror.Init) {
      cm.off("cursorActivity", doMatchTags);
      cm.off("viewportChange", maybeUpdateMatch);
      clear(cm);
    }
    // 如果新值存在
    if (val) {
      // 设置匹配标签的状态
      cm.state.matchBothTags = typeof val == "object" && val.bothTags;
      // 添加光标活动事件监听器和视口变化事件监听器
      cm.on("cursorActivity", doMatchTags);
      cm.on("viewportChange", maybeUpdateMatch);
      // 执行匹配标签函数
      doMatchTags(cm);
    }
  });

  // 清除标记的函数
  function clear(cm) {
    if (cm.state.tagHit) cm.state.tagHit.clear();
    if (cm.state.tagOther) cm.state.tagOther.clear();
    cm.state.tagHit = cm.state.tagOther = null;
  }

  // 执行匹配标签的函数
  function doMatchTags(cm) {
    cm.state.failedTagMatch = false;
    // 执行操作
    cm.operation(function() {
      clear(cm);
      // 如果有选中内容，则返回
      if (cm.somethingSelected()) return;
      var cur = cm.getCursor(), range = cm.getViewport();
      range.from = Math.min(range.from, cur.line); range.to = Math.max(cur.line + 1, range.to);
      var match = CodeMirror.findMatchingTag(cm, cur, range);
      // 如果没有匹配的标签，则返回
      if (!match) return;
      // 如果需要匹配两种标签
      if (cm.state.matchBothTags) {
        var hit = match.at == "open" ? match.open : match.close;
        if (hit) cm.state.tagHit = cm.markText(hit.from, hit.to, {className: "CodeMirror-matchingtag"});
      }
      var other = match.at == "close" ? match.open : match.close;
      if (other)
        cm.state.tagOther = cm.markText(other.from, other.to, {className: "CodeMirror-matchingtag"});
      else
        cm.state.failedTagMatch = true;
    });
  }
});
  // 定义一个匿名函数，该函数在匹配标签失败时执行匹配标签操作
  function maybeUpdateMatch(cm) {
    if (cm.state.failedTagMatch) doMatchTags(cm);
  }

  // 定义一个命令，用于将光标移动到匹配的标签处
  CodeMirror.commands.toMatchingTag = function(cm) {
    // 查找匹配的标签，并返回匹配结果
    var found = CodeMirror.findMatchingTag(cm, cm.getCursor());
    if (found) {
      // 如果找到匹配的标签，则将光标扩展到匹配标签的位置
      var other = found.at == "close" ? found.open : found.close;
      if (other) cm.extendSelection(other.to, other.from);
    }
  };
# 闭合了一个代码块，可能是函数、循环或条件语句的结束
```