# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\fold\brace-fold.js`

```py
// 将代码封装在立即执行函数中，传入 CodeMirror 对象
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 引入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 引入 CodeMirror 模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接传入 CodeMirror 对象
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 注册折叠辅助函数，用于大括号和方括号的折叠
  CodeMirror.registerHelper("fold", "brace", function(cm, start) {
    // 获取起始行的行号和内容
    var line = start.line, lineText = cm.getLine(line);
    var tokenType;

    // 查找起始大括号或方括号的位置
    function findOpening(openCh) {
      for (var at = start.ch, pass = 0;;) {
        var found = at <= 0 ? -1 : lineText.lastIndexOf(openCh, at - 1);
        if (found == -1) {
          if (pass == 1) break;
          pass = 1;
          at = lineText.length;
          continue;
        }
        if (pass == 1 && found < start.ch) break;
        // 获取括号内的 token 类型
        tokenType = cm.getTokenTypeAt(CodeMirror.Pos(line, found + 1));
        // 如果 token 类型不是注释或字符串，返回找到的位置
        if (!/^(comment|string)/.test(tokenType)) return found + 1;
        at = found - 1;
      }
    }

    // 初始化起始和结束的括号类型和位置
    var startToken = "{", endToken = "}", startCh = findOpening("{");
    if (startCh == null) {
      startToken = "[", endToken = "]";
      startCh = findOpening("[");
    }

    // 如果找不到起始位置，返回
    if (startCh == null) return;
    var count = 1, lastLine = cm.lastLine(), end, endCh;
    // 遍历每一行
    outer: for (var i = line; i <= lastLine; ++i) {
      var text = cm.getLine(i), pos = i == line ? startCh : 0;
      for (;;) {
        var nextOpen = text.indexOf(startToken, pos), nextClose = text.indexOf(endToken, pos);
        if (nextOpen < 0) nextOpen = text.length;
        if (nextClose < 0) nextClose = text.length;
        pos = Math.min(nextOpen, nextClose);
        if (pos == text.length) break;
        // 获取当前位置的 token 类型
        if (cm.getTokenTypeAt(CodeMirror.Pos(i, pos + 1)) == tokenType) {
          if (pos == nextOpen) ++count;
          else if (!--count) { end = i; endCh = pos; break outer; }
        }
        ++pos;
      }
    }
  });
});
    }
  }
  # 如果结束位置为空或者当前行等于结束位置，则返回
  if (end == null || line == end) return;
  # 返回一个对象，包含起始位置和结束位置的信息
  return {from: CodeMirror.Pos(line, startCh),
          to: CodeMirror.Pos(end, endCh)};
// 注册一个名为 "import" 的折叠辅助函数
CodeMirror.registerHelper("fold", "import", function(cm, start) {
  // 判断指定行是否包含 import 关键字
  function hasImport(line) {
    if (line < cm.firstLine() || line > cm.lastLine()) return null;
    var start = cm.getTokenAt(CodeMirror.Pos(line, 1));
    if (!/\S/.test(start.string)) start = cm.getTokenAt(CodeMirror.Pos(line, start.end + 1));
    if (start.type != "keyword" || start.string != "import") return null;
    // 寻找下一个分号的位置，返回其位置
    for (var i = line, e = Math.min(cm.lastLine(), line + 10); i <= e; ++i) {
      var text = cm.getLine(i), semi = text.indexOf(";");
      if (semi != -1) return {startCh: start.end, end: CodeMirror.Pos(i, semi)};
    }
  }

  var startLine = start.line, has = hasImport(startLine), prev;
  // 如果没有 import 或者上一行也有 import 或者上上一行有 import，则返回 null
  if (!has || hasImport(startLine - 1) || ((prev = hasImport(startLine - 2)) && prev.end.line == startLine - 1))
    return null;
  // 寻找 import 块的结束位置
  for (var end = has.end;;) {
    var next = hasImport(end.line + 1);
    if (next == null) break;
    end = next.end;
  }
  // 返回 import 块的起始和结束位置
  return {from: cm.clipPos(CodeMirror.Pos(startLine, has.startCh + 1)), to: end};
});

// 注册一个名为 "include" 的折叠辅助函数
CodeMirror.registerHelper("fold", "include", function(cm, start) {
  // 判断指定行是否包含 #include
  function hasInclude(line) {
    if (line < cm.firstLine() || line > cm.lastLine()) return null;
    var start = cm.getTokenAt(CodeMirror.Pos(line, 1));
    if (!/\S/.test(start.string)) start = cm.getTokenAt(CodeMirror.Pos(line, start.end + 1));
    if (start.type == "meta" && start.string.slice(0, 8) == "#include") return start.start + 8;
  }

  var startLine = start.line, has = hasInclude(startLine);
  // 如果没有 #include 或者上一行有 #include，则返回 null
  if (has == null || hasInclude(startLine - 1) != null) return null;
  // 寻找 #include 块的结束位置
  for (var end = startLine;;) {
    var next = hasInclude(end + 1);
    if (next == null) break;
    ++end;
  }
  // 返回 #include 块的起始和结束位置
  return {from: CodeMirror.Pos(startLine, has + 1),
          to: cm.clipPos(CodeMirror.Pos(end))};
});
```