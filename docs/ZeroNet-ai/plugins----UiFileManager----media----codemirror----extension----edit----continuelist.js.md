# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\edit\continuelist.js`

```
// 使用立即执行函数表达式（IIFE）将模块封装起来，避免变量污染全局作用域
(function(mod) {
  // 如果是 CommonJS 环境，使用 require 引入 CodeMirror 模块
  if (typeof exports == "object" && typeof module == "object") 
    mod(require("../../lib/codemirror"));
  // 如果是 AMD 环境，使用 define 引入 CodeMirror 模块
  else if (typeof define == "function" && define.amd) 
    define(["../../lib/codemirror"], mod);
  // 如果是普通的浏览器环境，直接引入 CodeMirror 模块
  else 
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义正则表达式，用于匹配 Markdown 列表的不同格式
  var listRE = /^(\s*)(>[> ]*|[*+-] \[[x ]\]\s|[*+-]\s|(\d+)([.)]))(\s*)/,
      emptyListRE = /^(\s*)(>[> ]*|[*+-] \[[x ]\]|[*+-]|(\d+)[.)])(\s*)$/,
      unorderedListRE = /[*+-]\s/;

  // 定义 CodeMirror 命令，用于在 Markdown 列表中插入新行并缩进
  CodeMirror.commands.newlineAndIndentContinueMarkdownList = function(cm) {
    // 如果禁用输入，则返回 CodeMirror.Pass
    if (cm.getOption("disableInput")) return CodeMirror.Pass;
    // 获取当前选择的文本范围
    var ranges = cm.listSelections(), replacements = [];
    // 遍历ranges数组，处理每个范围
    for (var i = 0; i < ranges.length; i++) {
      // 获取当前范围的头部位置
      var pos = ranges[i].head;

      // 获取当前行的状态，如果不是Markdown模式，则执行默认的newlineAndIndent命令
      var eolState = cm.getStateAfter(pos.line);
      var inner = CodeMirror.innerMode(cm.getMode(), eolState);
      if (inner.mode.name !== "markdown") {
        cm.execCommand("newlineAndIndent");
        return;
      } else {
        eolState = inner.state;
      }

      // 判断当前行是否在列表中
      var inList = eolState.list !== false;
      // 判断当前行是否在引用中
      var inQuote = eolState.quote !== 0;

      // 获取当前行的内容和匹配列表项的正则表达式
      var line = cm.getLine(pos.line), match = listRE.exec(line);
      // 判断当前范围是否为空，或者不在列表或引用中，或者不匹配列表项的正则，或者光标前为空白
      var cursorBeforeBullet = /^\s*$/.test(line.slice(0, pos.ch));
      if (!ranges[i].empty() || (!inList && !inQuote) || !match || cursorBeforeBullet) {
        cm.execCommand("newlineAndIndent");
        return;
      }
      // 如果当前行是空列表项
      if (emptyListRE.test(line)) {
        // 判断是否在引用的末尾或者列表的末尾，如果是则删除当前行
        var endOfQuote = inQuote && />\s*$/.test(line)
        var endOfList = !/>\s*$/.test(line)
        if (endOfQuote || endOfList) cm.replaceRange("", {
          line: pos.line, ch: 0
        }, {
          line: pos.line, ch: pos.ch + 1
        });
        // 将当前范围的替换内容设置为换行符
        replacements[i] = "\n";
      } else {
        // 如果不是空列表项，则处理列表项的缩进和标记
        var indent = match[1], after = match[5];
        var numbered = !(unorderedListRE.test(match[2]) || match[2].indexOf(">") >= 0);
        var bullet = numbered ? (parseInt(match[3], 10) + 1) + match[4] : match[2].replace("x", " ");
        // 将当前范围的替换内容设置为新的列表项
        replacements[i] = "\n" + indent + bullet + after;

        // 如果是有序列表，则增加后续列表项的编号
        if (numbered) incrementRemainingMarkdownListNumbers(cm, pos);
      }
    }

    // 执行替换操作，将replacements数组中的内容替换到对应的范围中
    cm.replaceSelections(replacements);
  };

  // 当在列表中间添加新项时，自动更新Markdown列表的编号
  function incrementRemainingMarkdownListNumbers(cm, pos) {
    var startLine = pos.line, lookAhead = 0, skipCount = 0;
    var startItem = listRE.exec(cm.getLine(startLine)), startIndent = startItem[1];
    # 循环开始
    do {
      # 增加查找下一行的偏移量
      lookAhead += 1;
      # 计算下一行的行号
      var nextLineNumber = startLine + lookAhead;
      # 获取下一行的内容和列表项的正则匹配结果
      var nextLine = cm.getLine(nextLineNumber), nextItem = listRE.exec(nextLine);

      # 如果匹配到了下一行的列表项
      if (nextItem) {
        # 获取下一行的缩进
        var nextIndent = nextItem[1];
        # 计算新的列表项编号
        var newNumber = (parseInt(startItem[3], 10) + lookAhead - skipCount);
        # 获取下一行的列表项编号
        var nextNumber = (parseInt(nextItem[3], 10)), itemNumber = nextNumber;

        # 如果起始行和下一行的缩进相同且下一行的编号是数字
        if (startIndent === nextIndent && !isNaN(nextNumber)) {
          # 如果新的编号等于下一行的编号，则使用下一行的编号加1
          if (newNumber === nextNumber) itemNumber = nextNumber + 1;
          # 如果新的编号大于下一行的编号，则使用新的编号加1
          if (newNumber > nextNumber) itemNumber = newNumber + 1;
          # 替换下一行的内容，更新列表项编号
          cm.replaceRange(
            nextLine.replace(listRE, nextIndent + itemNumber + nextItem[4] + nextItem[5]),
          {
            line: nextLineNumber, ch: 0
          }, {
            line: nextLineNumber, ch: nextLine.length
          });
        } else {
          # 如果起始行的缩进大于下一行的缩进，则直接返回
          if (startIndent.length > nextIndent.length) return;
          # 如果下一行的缩进大于起始行的缩进，并且偏移量为1，则直接返回
          # 这是因为无法确定用户意图（是新的缩进项还是同级别的项）
          if ((startIndent.length < nextIndent.length) && (lookAhead === 1)) return;
          # 增加跳过计数
          skipCount += 1;
        }
      }
    } while (nextItem);
  }
# 闭合一个代码块或函数的结束括号
```