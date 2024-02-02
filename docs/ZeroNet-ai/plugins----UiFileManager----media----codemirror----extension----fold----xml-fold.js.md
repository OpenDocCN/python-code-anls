# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\fold\xml-fold.js`

```py
// 使用立即执行函数表达式（IIFE）包装代码，传入 CodeMirror 对象
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

  // 定义 Pos 函数，用于表示位置
  var Pos = CodeMirror.Pos;
  // 定义 cmp 函数，用于比较两个位置的大小
  function cmp(a, b) { return a.line - b.line || a.ch - b.ch; }

  // 定义变量 nameStartChar，表示 XML 标签名的起始字符
  var nameStartChar = "A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD";
  // 定义变量 nameChar，表示 XML 标签名的字符集合
  var nameChar = nameStartChar + "\-\:\.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040";
  // 定义正则表达式 xmlTagStart，用于匹配 XML 标签的起始部分
  var xmlTagStart = new RegExp("<(/?)([" + nameStartChar + "][" + nameChar + "]*)", "g");

  // 定义 Iter 类，用于迭代器
  function Iter(cm, line, ch, range) {
    this.line = line; this.ch = ch;
    this.cm = cm; this.text = cm.getLine(line);
    this.min = range ? Math.max(range.from, cm.firstLine()) : cm.firstLine();
    this.max = range ? Math.min(range.to - 1, cm.lastLine()) : cm.lastLine();
  }

  // 定义 tagAt 函数，用于判断指定位置是否为标签
  function tagAt(iter, ch) {
    var type = iter.cm.getTokenTypeAt(Pos(iter.line, ch));
    return type && /\btag\b/.test(type);
  }

  // 定义 nextLine 函数，用于获取下一行的内容
  function nextLine(iter) {
    if (iter.line >= iter.max) return;
    iter.ch = 0;
    iter.text = iter.cm.getLine(++iter.line);
    return true;
  }
  // 定义 prevLine 函数，用于获取上一行的内容
  function prevLine(iter) {
    if (iter.line <= iter.min) return;
    iter.text = iter.cm.getLine(--iter.line);
    iter.ch = iter.text.length;
    return true;
  }

  // 定义 toTagEnd 函数，用于获取标签的结束位置
  function toTagStart(iter) {
    // 无限循环，查找标签的起始位置
    for (;;) {
      // 在当前位置之后查找">"的位置
      var gt = iter.text.indexOf(">", iter.ch);
      // 如果找不到">"，则继续下一行查找，如果没有下一行则返回
      if (gt == -1) { if (nextLine(iter)) continue; else return; }
      // 如果">"后面不是标签，则继续查找
      if (!tagAt(iter, gt + 1)) { iter.ch = gt + 1; continue; }
      // 在">"之前查找最后一个"/"的位置
      var lastSlash = iter.text.lastIndexOf("/", gt);
      // 判断是否是自闭合标签
      var selfClose = lastSlash > -1 && !/\S/.test(iter.text.slice(lastSlash + 1, gt));
      // 设置当前位置为">"的位置
      iter.ch = gt + 1;
      // 返回标签类型，自闭合或者普通
      return selfClose ? "selfClose" : "regular";
    }
  }

  function toTagStart(iter) {
    // 无限循环，查找标签的起始位置
    for (;;) {
      // 在当前位置之前查找"<"的位置
      var lt = iter.ch ? iter.text.lastIndexOf("<", iter.ch - 1) : -1;
      // 如果找不到"<"，则继续上一行查找，如果没有上一行则返回
      if (lt == -1) { if (prevLine(iter)) continue; else return; }
      // 如果"<"后面不是标签，则继续查找
      if (!tagAt(iter, lt + 1)) { iter.ch = lt; continue; }
      // 设置正则表达式的起始位置
      xmlTagStart.lastIndex = lt;
      // 设置当前位置为"<"的位置
      iter.ch = lt;
      // 在文本中查找标签的匹配
      var match = xmlTagStart.exec(iter.text);
      // 如果找到匹配则返回
      if (match && match.index == lt) return match;
    }
  }

  function toNextTag(iter) {
    // 无限循环，查找下一个标签的位置
    for (;;) {
      // 设置正则表达式的起始位置
      xmlTagStart.lastIndex = iter.ch;
      // 在文本中查找下一个标签的位置
      var found = xmlTagStart.exec(iter.text);
      // 如果找不到下一个标签，则继续下一行查找，如果没有下一行则返回
      if (!found) { if (nextLine(iter)) continue; else return; }
      // 如果找到的位置不是标签，则继续查找
      if (!tagAt(iter, found.index + 1)) { iter.ch = found.index + 1; continue; }
      // 设置当前位置为找到的标签的结束位置
      iter.ch = found.index + found[0].length;
      // 返回找到的标签
      return found;
    }
  }

  function toPrevTag(iter) {
    // 无限循环，查找上一个标签的位置
    for (;;) {
      // 在当前位置之前查找">"的位置
      var gt = iter.ch ? iter.text.lastIndexOf(">", iter.ch - 1) : -1;
      // 如果找不到">"，则继续上一行查找，如果没有上一行则返回
      if (gt == -1) { if (prevLine(iter)) continue; else return; }
      // 如果">"后面不是标签，则继续查找
      if (!tagAt(iter, gt + 1)) { iter.ch = gt; continue; }
      // 在">"之前查找最后一个"/"的位置
      var lastSlash = iter.text.lastIndexOf("/", gt);
      // 判断是否是自闭合标签
      var selfClose = lastSlash > -1 && !/\S/.test(iter.text.slice(lastSlash + 1, gt));
      // 设置当前位置为">"的位置
      iter.ch = gt + 1;
      // 返回标签类型，自闭合或者普通
      return selfClose ? "selfClose" : "regular";
    }
  }

  function findMatchingClose(iter, tag) {
    // 初始化一个空栈
    var stack = [];
    for (;;) {
      // 无限循环，用于遍历 XML 标签
      var next = toNextTag(iter), end, startLine = iter.line, startCh = iter.ch - (next ? next[0].length : 0);
      // 获取下一个标签的信息，记录起始行号和起始字符位置
      if (!next || !(end = toTagEnd(iter))) return;
      // 如果没有下一个标签或者找不到标签结束位置，则退出循环
      if (end == "selfClose") continue;
      // 如果标签是自闭合的，则继续下一次循环
      if (next[1]) { // closing tag
        // 如果是闭合标签
        for (var i = stack.length - 1; i >= 0; --i) if (stack[i] == next[2]) {
          stack.length = i;
          break;
        }
        // 从堆栈中移除对应的开放标签
        if (i < 0 && (!tag || tag == next[2])) return {
          tag: next[2],
          from: Pos(startLine, startCh),
          to: Pos(iter.line, iter.ch)
        };
        // 如果找到对应的开放标签，则返回标签信息
      } else { // opening tag
        // 如果是开放标签
        stack.push(next[2]);
        // 将标签名加入堆栈
      }
    }
  }
  function findMatchingOpen(iter, tag) {
    // 查找匹配的开放标签
    var stack = [];
    for (;;) {
      var prev = toPrevTag(iter);
      // 获取前一个标签的信息
      if (!prev) return;
      // 如果没有前一个标签，则退出循环
      if (prev == "selfClose") { toTagStart(iter); continue; }
      // 如果是自闭合标签，则继续下一次循环
      var endLine = iter.line, endCh = iter.ch;
      var start = toTagStart(iter);
      // 获取标签的起始位置
      if (!start) return;
      // 如果找不到标签的起始位置，则退出循环
      if (start[1]) { // closing tag
        // 如果是闭合标签
        stack.push(start[2]);
        // 将标签名加入堆栈
      } else { // opening tag
        // 如果是开放标签
        for (var i = stack.length - 1; i >= 0; --i) if (stack[i] == start[2]) {
          stack.length = i;
          break;
        }
        // 从堆栈中移除对应的开放标签
        if (i < 0 && (!tag || tag == start[2])) return {
          tag: start[2],
          from: Pos(iter.line, iter.ch),
          to: Pos(endLine, endCh)
        };
        // 如果找到对应的闭合标签，则返回标签信息
      }
    }
  }

  CodeMirror.registerHelper("fold", "xml", function(cm, start) {
    // 注册 XML 折叠功能
    var iter = new Iter(cm, start.line, 0);
    // 创建迭代器
    for (;;) {
      var openTag = toNextTag(iter)
      // 获取下一个标签
      if (!openTag || iter.line != start.line) return
      // 如果没有下一个标签或者行号不匹配，则退出循环
      var end = toTagEnd(iter)
      // 获取标签的结束位置
      if (!end) return
      // 如果找不到标签的结束位置，则退出循环
      if (!openTag[1] && end != "selfClose") {
        // 如果是开放标签且不是自闭合标签
        var startPos = Pos(iter.line, iter.ch);
        // 记录起始位置
        var endPos = findMatchingClose(iter, openTag[2]);
        // 查找匹配的闭合标签
        return endPos && cmp(endPos.from, startPos) > 0 ? {from: startPos, to: endPos.from} : null
        // 如果找到匹配的闭合标签，则返回折叠的范围
      }
    }
  });
  CodeMirror.findMatchingTag = function(cm, pos, range) {
  // 查找匹配的标签
    // 创建一个迭代器对象，用于在指定范围内查找标签
    var iter = new Iter(cm, pos.line, pos.ch, range);
    // 如果当前行文本中不包含"<"和">"，则直接返回
    if (iter.text.indexOf(">") == -1 && iter.text.indexOf("<") == -1) return;
    // 查找标签的结束位置
    var end = toTagEnd(iter), to = end && Pos(iter.line, iter.ch);
    // 查找标签的开始位置
    var start = end && toTagStart(iter);
    // 如果没有找到开始或结束位置，或者当前位置在开始位置之后，则直接返回
    if (!end || !start || cmp(iter, pos) > 0) return;
    // 创建一个对象表示当前标签的位置和类型
    var here = {from: Pos(iter.line, iter.ch), to: to, tag: start[2]};
    // 如果标签是自闭合的，则返回一个包含当前标签信息的对象
    if (end == "selfClose") return {open: here, close: null, at: "open"};

    // 如果是闭合标签，则返回一个包含闭合标签和对应开放标签信息的对象
    if (start[1]) { // closing tag
      return {open: findMatchingOpen(iter, start[2]), close: here, at: "close"};
    } else { // opening tag
      // 创建一个新的迭代器对象，用于查找对应的闭合标签
      iter = new Iter(cm, to.line, to.ch, range);
      return {open: here, close: findMatchingClose(iter, start[2]), at: "open"};
    }
  };

  // 在给定范围内查找包含指定标签的开放和闭合标签
  CodeMirror.findEnclosingTag = function(cm, pos, range, tag) {
    var iter = new Iter(cm, pos.line, pos.ch, range);
    for (;;) {
      // 查找包含指定标签的开放标签
      var open = findMatchingOpen(iter, tag);
      if (!open) break;
      // 创建一个新的迭代器对象，用于查找对应的闭合标签
      var forward = new Iter(cm, pos.line, pos.ch, range);
      var close = findMatchingClose(forward, open.tag);
      if (close) return {open: open, close: close};
    }
  };

  // 用于编辑器插件，根据指定位置和标签名称查找对应的闭合标签
  CodeMirror.scanForClosingTag = function(cm, pos, name, end) {
    var iter = new Iter(cm, pos.line, pos.ch, end ? {from: 0, to: end} : null);
    return findMatchingClose(iter, name);
  };
# 闭合了一个代码块或者函数的结束
```