# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\hint\anyword-hint.js`

```py
// 使用立即执行函数表达式（IIFE）封装代码，传入 CodeMirror 对象
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

  // 定义单词匹配的正则表达式和范围
  var WORD = /[\w$]+/, RANGE = 500;

  // 注册一个名为 "anyword" 的提示辅助函数
  CodeMirror.registerHelper("hint", "anyword", function(editor, options) {
    // 获取单词和范围的配置，如果没有则使用默认值
    var word = options && options.word || WORD;
    var range = options && options.range || RANGE;
    // 获取当前光标位置和所在行的文本
    var cur = editor.getCursor(), curLine = editor.getLine(cur.line);
    var end = cur.ch, start = end;
    // 向前查找当前单词的起始位置
    while (start && word.test(curLine.charAt(start - 1))) --start;
    // 获取当前单词
    var curWord = start != end && curLine.slice(start, end);

    // 获取提示列表和已经出现过的单词
    var list = options && options.list || [], seen = {};
    // 创建匹配单词的正则表达式
    var re = new RegExp(word.source, "g");
    // 向前和向后搜索范围内的单词
    for (var dir = -1; dir <= 1; dir += 2) {
      var line = cur.line, endLine = Math.min(Math.max(line + dir * range, editor.firstLine()), editor.lastLine()) + dir;
      for (; line != endLine; line += dir) {
        var text = editor.getLine(line), m;
        // 在每行文本中匹配单词
        while (m = re.exec(text)) {
          // 排除当前单词，并且确保单词未出现过，然后添加到提示列表中
          if (line == cur.line && m[0] === curWord) continue;
          if ((!curWord || m[0].lastIndexOf(curWord, 0) == 0) && !Object.prototype.hasOwnProperty.call(seen, m[0])) {
            seen[m[0]] = true;
            list.push(m[0]);
          }
        }
      }
    }
    // 返回提示列表和当前单词的位置范围
    return {list: list, from: CodeMirror.Pos(cur.line, start), to: CodeMirror.Pos(cur.line, end)};
  });
});
```