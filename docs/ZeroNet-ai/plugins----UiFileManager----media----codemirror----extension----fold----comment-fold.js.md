# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\fold\comment-fold.js`

```py
// 将代码包装在自执行函数中，传入 CodeMirror 对象
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 引入 CodeMirror
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 引入 CodeMirror
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接传入 CodeMirror 对象
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 注册全局的折叠帮助器，用于折叠注释
  CodeMirror.registerGlobalHelper("fold", "comment", function(mode) {
    // 判断当前模式是否有块注释的开始和结束标记
    return mode.blockCommentStart && mode.blockCommentEnd;
  }, function(cm, start) {
    // 获取当前光标位置的模式，以及块注释的开始和结束标记
    var mode = cm.getModeAt(start), startToken = mode.blockCommentStart, endToken = mode.blockCommentEnd;
    if (!startToken || !endToken) return;
    var line = start.line, lineText = cm.getLine(line);

    var startCh;
    // 循环查找块注释的开始位置
    for (var at = start.ch, pass = 0;;) {
      var found = at <= 0 ? -1 : lineText.lastIndexOf(startToken, at - 1);
      if (found == -1) {
        if (pass == 1) return;
        pass = 1;
        at = lineText.length;
        continue;
      }
      if (pass == 1 && found < start.ch) return;
      // 判断找到的开始位置是否为注释类型
      if (/comment/.test(cm.getTokenTypeAt(CodeMirror.Pos(line, found + 1))) &&
          (found == 0 || lineText.slice(found - endToken.length, found) == endToken ||
           !/comment/.test(cm.getTokenTypeAt(CodeMirror.Pos(line, found))))) {
        startCh = found + startToken.length;
        break;
      }
      at = found - 1;
    }

    var depth = 1, lastLine = cm.lastLine(), end, endCh;
    // 循环查找块注释的结束位置
    outer: for (var i = line; i <= lastLine; ++i) {
      var text = cm.getLine(i), pos = i == line ? startCh : 0;
      for (;;) {
        var nextOpen = text.indexOf(startToken, pos), nextClose = text.indexOf(endToken, pos);
        if (nextOpen < 0) nextOpen = text.length;
        if (nextClose < 0) nextClose = text.length;
        pos = Math.min(nextOpen, nextClose);
        if (pos == text.length) break;
        if (pos == nextOpen) ++depth;
        else if (!--depth) { end = i; endCh = pos; break outer; }
        ++pos;
      }
    }
    // 折叠块注释
    cm.foldCode(CodeMirror.Pos(line, startCh), CodeMirror.Pos(end, endCh));
  });
});
    }
  }  # 结束 if 语句块
  # 如果结束位置为空，或者行数等于结束行数并且结束字符等于开始字符，则返回
  if (end == null || line == end && endCh == startCh) return;
  # 返回一个对象，包含起始位置和结束位置
  return {from: CodeMirror.Pos(line, startCh),
          to: CodeMirror.Pos(end, endCh)};
# 闭合两个嵌套的匿名函数
});
```