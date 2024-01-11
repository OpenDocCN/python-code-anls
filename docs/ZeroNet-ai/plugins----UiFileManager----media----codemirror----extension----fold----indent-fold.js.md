# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\fold\indent-fold.js`

```
// 使用立即执行函数表达式（IIFE）来定义模块
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 导入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 导入 CodeMirror 模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接使用 CodeMirror 对象
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义函数 lineIndent，用于获取指定行的缩进
  function lineIndent(cm, lineNo) {
    var text = cm.getLine(lineNo)
    var spaceTo = text.search(/\S/)
    // 如果行为空或者是注释行，则返回 -1
    if (spaceTo == -1 || /\bcomment\b/.test(cm.getTokenTypeAt(CodeMirror.Pos(lineNo, spaceTo + 1))))
      return -1
    // 否则返回行的缩进
    return CodeMirror.countColumn(text, null, cm.getOption("tabSize"))
  }

  // 注册折叠辅助函数，根据缩进进行折叠
  CodeMirror.registerHelper("fold", "indent", function(cm, start) {
    var myIndent = lineIndent(cm, start.line)
    // 如果行没有缩进，则返回
    if (myIndent < 0) return
    var lastLineInFold = null

    // 遍历行，找到不属于当前折叠块的行或者到达末尾
    for (var i = start.line + 1, end = cm.lastLine(); i <= end; ++i) {
      var indent = lineIndent(cm, i)
      if (indent == -1) {
        // 如果是空行或者注释行，则继续
      } else if (indent > myIndent) {
        // 如果缩进大于当前行，则认为是当前折叠块的一部分
        lastLineInFold = i;
      } else {
        // 如果这一行有非空格、非注释内容，并且缩进小于等于起始行，则是另一个折叠块的开始
        break;
      }
    }
    // 返回折叠的起始和结束位置
    if (lastLineInFold) return {
      from: CodeMirror.Pos(start.line, cm.getLine(start.line).length),
      to: CodeMirror.Pos(lastLineInFold, cm.getLine(lastLineInFold).length)
    };
  });

});
```