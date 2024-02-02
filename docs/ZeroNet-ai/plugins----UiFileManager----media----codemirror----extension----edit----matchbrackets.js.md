# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\edit\matchbrackets.js`

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
  // 检测是否是 IE8 及以下版本
  var ie_lt8 = /MSIE \d/.test(navigator.userAgent) &&
    (document.documentMode == null || document.documentMode < 8);

  // 定义 Pos 为 CodeMirror.Pos
  var Pos = CodeMirror.Pos;

  // 定义匹配的括号对
  var matching = {"(": ")>", ")": "(<", "[": "]>", "]": "[<", "{": "}>", "}": "{<", "<": ">>", ">": "<<"};

  // 定义括号的正则表达式
  function bracketRegex(config) {
    return config && config.bracketRegex || /[(){}[\]]/
  }

  // 查找匹配的括号
  function findMatchingBracket(cm, where, config) {
    // 获取当前行的内容和光标位置
    var line = cm.getLineHandle(where.line), pos = where.ch - 1;
    var afterCursor = config && config.afterCursor
    if (afterCursor == null)
      afterCursor = /(^| )cm-fat-cursor($| )/.test(cm.getWrapperElement().className)
    var re = bracketRegex(config)

    // 判断光标位置的括号是否匹配
    var match = (!afterCursor && pos >= 0 && re.test(line.text.charAt(pos)) && matching[line.text.charAt(pos)]) ||
        re.test(line.text.charAt(pos + 1)) && matching[line.text.charAt(++pos)];
    if (!match) return null;
    var dir = match.charAt(1) == ">" ? 1 : -1;
    if (config && config.strict && (dir > 0) != (pos == where.ch)) return null;
    var style = cm.getTokenTypeAt(Pos(where.line, pos + 1));

    // 在指定方向上扫描匹配的括号
    var found = scanForBracket(cm, Pos(where.line, pos + (dir > 0 ? 1 : 0)), dir, style || null, config);
    if (found == null) return null;
  }
  # 返回一个对象，包含匹配括号的位置信息和是否匹配的布尔值
  return {from: Pos(where.line, pos), to: found && found.pos,
          match: found && found.ch == match.charAt(0), forward: dir > 0};
}

// bracketRegex is used to specify which type of bracket to scan
// should be a regexp, e.g. /[[\]]/
//
// Note: If "where" is on an open bracket, then this bracket is ignored.
//
// Returns false when no bracket was found, null when it reached
// maxScanLines and gave up
function scanForBracket(cm, where, dir, style, config) {
  var maxScanLen = (config && config.maxScanLineLength) || 10000;
  var maxScanLines = (config && config.maxScanLines) || 1000;

  var stack = [];
  var re = bracketRegex(config)
  var lineEnd = dir > 0 ? Math.min(where.line + maxScanLines, cm.lastLine() + 1)
                        : Math.max(cm.firstLine() - 1, where.line - maxScanLines);
  for (var lineNo = where.line; lineNo != lineEnd; lineNo += dir) {
    var line = cm.getLine(lineNo);
    if (!line) continue;
    var pos = dir > 0 ? 0 : line.length - 1, end = dir > 0 ? line.length : -1;
    if (line.length > maxScanLen) continue;
    if (lineNo == where.line) pos = where.ch - (dir < 0 ? 1 : 0);
    for (; pos != end; pos += dir) {
      var ch = line.charAt(pos);
      if (re.test(ch) && (style === undefined || cm.getTokenTypeAt(Pos(lineNo, pos + 1)) == style)) {
        var match = matching[ch];
        if (match && (match.charAt(1) == ">") == (dir > 0)) stack.push(ch);
        else if (!stack.length) return {pos: Pos(lineNo, pos), ch: ch};
        else stack.pop();
      }
    }
  }
  return lineNo - dir == (dir > 0 ? cm.lastLine() : cm.firstLine()) ? false : null;
}

function matchBrackets(cm, autoclear, config) {
  // Disable brace matching in long lines, since it'll cause hugely slow updates
  var maxHighlightLen = cm.state.matchBrackets.maxHighlightLineLength || 1000;
  var marks = [], ranges = cm.listSelections();
    // 遍历 ranges 数组，对每个元素执行以下操作
    for (var i = 0; i < ranges.length; i++) {
      // 检查当前 range 是否为空，如果是，则查找匹配的括号
      var match = ranges[i].empty() && findMatchingBracket(cm, ranges[i].head, config);
      // 如果找到匹配的括号，并且匹配的行长度小于等于 maxHighlightLen，则执行以下操作
      if (match && cm.getLine(match.from.line).length <= maxHighlightLen) {
        // 根据匹配结果选择样式
        var style = match.match ? "CodeMirror-matchingbracket" : "CodeMirror-nonmatchingbracket";
        // 在匹配的位置创建标记
        marks.push(cm.markText(match.from, Pos(match.from.line, match.from.ch + 1), {className: style}));
        // 如果有匹配的结束位置，并且匹配的行长度小于等于 maxHighlightLen，则在结束位置创建标记
        if (match.to && cm.getLine(match.to.line).length <= maxHighlightLen)
          marks.push(cm.markText(match.to, Pos(match.to.line, match.to.ch + 1), {className: style}));
      }
    }

    // 如果存在标记
    if (marks.length) {
      // 修复 IE bug，解决 issue #1193，当触发此事件时，文本输入停止输入到文本区域
      if (ie_lt8 && cm.state.focused) cm.focus();

      // 定义清除标记的函数
      var clear = function() {
        cm.operation(function() {
          for (var i = 0; i < marks.length; i++) marks[i].clear();
        });
      };
      // 如果 autoclear 为真，则延迟 800 毫秒后执行清除函数，否则返回清除函数
      if (autoclear) setTimeout(clear, 800);
      else return clear;
    }
  }

  // 执行匹配括号的操作
  function doMatchBrackets(cm) {
    cm.operation(function() {
      // 如果已经有高亮的匹配括号，则清除
      if (cm.state.matchBrackets.currentlyHighlighted) {
        cm.state.matchBrackets.currentlyHighlighted();
        cm.state.matchBrackets.currentlyHighlighted = null;
      }
      // 执行匹配括号的操作
      cm.state.matchBrackets.currentlyHighlighted = matchBrackets(cm, false, cm.state.matchBrackets);
    });
  }

  // 定义 matchBrackets 选项
  CodeMirror.defineOption("matchBrackets", false, function(cm, val, old) {
    // 定义清除函数
    function clear(cm) {
      if (cm.state.matchBrackets && cm.state.matchBrackets.currentlyHighlighted) {
        cm.state.matchBrackets.currentlyHighlighted();
        cm.state.matchBrackets.currentlyHighlighted = null;
      }
    }

    // 如果旧值存在且不等于 CodeMirror.Init
    if (old && old != CodeMirror.Init) {
      // 移除事件监听
      cm.off("cursorActivity", doMatchBrackets);
      cm.off("focus", doMatchBrackets)
      cm.off("blur", clear)
      // 清除匹配括号的高亮
      clear(cm);
    }
    # 如果val存在，则执行以下操作
    if (val) {
      # 如果val是对象，则将cm.state.matchBrackets设置为val，否则设置为空对象
      cm.state.matchBrackets = typeof val == "object" ? val : {};
      # 当光标活动时执行doMatchBrackets函数
      cm.on("cursorActivity", doMatchBrackets);
      # 当编辑器获得焦点时执行doMatchBrackets函数
      cm.on("focus", doMatchBrackets)
      # 当编辑器失去焦点时执行clear函数
      cm.on("blur", clear)
    }
  });

  # 定义matchBrackets方法，调用matchBrackets函数并传入当前编辑器对象和true作为参数
  CodeMirror.defineExtension("matchBrackets", function() {matchBrackets(this, true);});
  # 定义findMatchingBracket方法，根据传入的位置、配置和旧配置查找匹配的括号
  CodeMirror.defineExtension("findMatchingBracket", function(pos, config, oldConfig){
    # 向后兼容的修补措施
    if (oldConfig || typeof config == "boolean") {
      # 如果没有旧配置，则根据config的值设置严格模式，否则将旧配置的严格模式设置为config的值
      if (!oldConfig) {
        config = config ? {strict: true} : null
      } else {
        oldConfig.strict = config
        config = oldConfig
      }
    }
    # 调用findMatchingBracket函数并传入当前编辑器对象、位置和配置作为参数
    return findMatchingBracket(this, pos, config)
  });
  # 定义scanForBracket方法，根据传入的位置、方向、样式和配置扫描括号
  CodeMirror.defineExtension("scanForBracket", function(pos, dir, style, config){
    # 调用scanForBracket函数并传入当前编辑器对象、位置、方向、样式和配置作为参数
    return scanForBracket(this, pos, dir, style, config);
  });
# 闭合代码块的右大括号
```