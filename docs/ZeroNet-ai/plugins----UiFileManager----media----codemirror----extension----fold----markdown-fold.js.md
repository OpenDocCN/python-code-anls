# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\fold\markdown-fold.js`

```
// CodeMirror, copyright (c) by Marijn Haverbeke and others
// Distributed under an MIT license: https://codemirror.net/LICENSE

// 使用立即执行函数表达式，传入 mod 函数
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 注册 markdown 折叠辅助函数
  CodeMirror.registerHelper("fold", "markdown", function(cm, start) {
    // 设置最大深度
    var maxDepth = 100;

    // 判断是否为标题行
    function isHeader(lineNo) {
      var tokentype = cm.getTokenTypeAt(CodeMirror.Pos(lineNo, 0));
      return tokentype && /\bheader\b/.test(tokentype);
    }

    // 获取标题级别
    function headerLevel(lineNo, line, nextLine) {
      var match = line && line.match(/^#+/);
      if (match && isHeader(lineNo)) return match[0].length;
      match = nextLine && nextLine.match(/^[=\-]+\s*$/);
      if (match && isHeader(lineNo + 1)) return nextLine[0] == "=" ? 1 : 2;
      return maxDepth;
    }

    // 获取起始行和下一行的内容
    var firstLine = cm.getLine(start.line), nextLine = cm.getLine(start.line + 1);
    var level = headerLevel(start.line, firstLine, nextLine);
    if (level === maxDepth) return undefined;

    var lastLineNo = cm.lastLine();
    var end = start.line, nextNextLine = cm.getLine(end + 2);
    // 循环查找折叠结束位置
    while (end < lastLineNo) {
      if (headerLevel(end + 1, nextLine, nextNextLine) <= level) break;
      ++end;
      nextLine = nextNextLine;
      nextNextLine = cm.getLine(end + 2);
    }

    // 返回折叠的起始和结束位置
    return {
      from: CodeMirror.Pos(start.line, firstLine.length),
      to: CodeMirror.Pos(end, cm.getLine(end).length)
    };
  });

});
```