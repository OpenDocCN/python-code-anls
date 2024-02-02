# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\sublime.js`

```py
// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

// Sublime Text 键绑定的粗略近似
// 依赖于 addon/search/searchcursor.js 和可选的 addon/dialog/dialogs.js

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../lib/codemirror"), require("../addon/search/searchcursor"), require("../addon/edit/matchbrackets"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../lib/codemirror", "../addon/search/searchcursor", "../addon/edit/matchbrackets"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  var cmds = CodeMirror.commands;
  var Pos = CodeMirror.Pos;

  // 这并不完全是 Sublime 的算法。我无法理解那个。
  function findPosSubword(doc, start, dir) {
    if (dir < 0 && start.ch == 0) return doc.clipPos(Pos(start.line - 1));
    var line = doc.getLine(start.line);
    if (dir > 0 && start.ch >= line.length) return doc.clipPos(Pos(start.line + 1, 0));
    var state = "start", type, startPos = start.ch;
    for (var pos = startPos, e = dir < 0 ? 0 : line.length, i = 0; pos != e; pos += dir, i++) {
      var next = line.charAt(dir < 0 ? pos - 1 : pos);
      var cat = next != "_" && CodeMirror.isWordChar(next) ? "w" : "o";
      if (cat == "w" && next.toUpperCase() == next) cat = "W";
      if (state == "start") {
        if (cat != "o") { state = "in"; type = cat; }
        else startPos = pos + dir
      } else if (state == "in") {
        if (type != cat) {
          if (type == "w" && cat == "W" && dir < 0) pos--;
          if (type == "W" && cat == "w" && dir > 0) { // 从大写转换为小写
            if (pos == startPos + 1) { type = "w"; continue; }
            else pos--;
          }
          break;
        }
      }
    }
  # 返回一个包含指定行和位置的位置对象
  return Pos(start.line, pos);
}

# 向前或向后移动子词
function moveSubword(cm, dir) {
  # 通过扩展选择范围来移动子词
  cm.extendSelectionsBy(function(range) {
    if (cm.display.shift || cm.doc.extend || range.empty())
      # 如果按下了 shift 键或者文档处于扩展模式，或者选择范围为空，则查找子词位置
      return findPosSubword(cm.doc, range.head, dir);
    else
      # 否则返回当前选择范围的起始位置或结束位置
      return dir < 0 ? range.from() : range.to();
  });
}

# 向左移动子词
cmds.goSubwordLeft = function(cm) { moveSubword(cm, -1); };
# 向右移动子词
cmds.goSubwordRight = function(cm) { moveSubword(cm, 1); };

# 向上滚动一行
cmds.scrollLineUp = function(cm) {
  var info = cm.getScrollInfo();
  if (!cm.somethingSelected()) {
    var visibleBottomLine = cm.lineAtHeight(info.top + info.clientHeight, "local");
    if (cm.getCursor().line >= visibleBottomLine)
      cm.execCommand("goLineUp");
  }
  # 滚动到指定位置
  cm.scrollTo(null, info.top - cm.defaultTextHeight());
};
# 向下滚动一行
cmds.scrollLineDown = function(cm) {
  var info = cm.getScrollInfo();
  if (!cm.somethingSelected()) {
    var visibleTopLine = cm.lineAtHeight(info.top, "local")+1;
    if (cm.getCursor().line <= visibleTopLine)
      cm.execCommand("goLineDown");
  }
  # 滚动到指定位置
  cm.scrollTo(null, info.top + cm.defaultTextHeight());
};

# 通过行拆分选择
cmds.splitSelectionByLine = function(cm) {
  var ranges = cm.listSelections(), lineRanges = [];
  for (var i = 0; i < ranges.length; i++) {
    var from = ranges[i].from(), to = ranges[i].to();
    for (var line = from.line; line <= to.line; ++line)
      if (!(to.line > from.line && line == to.line && to.ch == 0))
        lineRanges.push({anchor: line == from.line ? from : Pos(line, 0),
                         head: line == to.line ? to : Pos(line)});
  }
  # 设置选择范围
  cm.setSelections(lineRanges, 0);
};

# 将单个选择移动到顶部
cmds.singleSelectionTop = function(cm) {
  var range = cm.listSelections()[0];
  # 设置选择范围
  cm.setSelection(range.anchor, range.head, {scroll: false});
};

# 选择整行
cmds.selectLine = function(cm) {
  var ranges = cm.listSelections(), extended = [];
  # ...
}
    # 遍历 ranges 数组
    for (var i = 0; i < ranges.length; i++) {
      # 获取当前 range
      var range = ranges[i];
      # 将当前 range 转换为 anchor 和 head 对象，并添加到 extended 数组中
      extended.push({anchor: Pos(range.from().line, 0),
                     head: Pos(range.to().line + 1, 0)});
    }
    # 设置编辑器的选择范围为 extended 数组中的内容
    cm.setSelections(extended);
  };

  # 在当前光标位置上方或下方插入一行
  function insertLine(cm, above) {
    # 如果编辑器为只读状态，则返回 CodeMirror.Pass
    if (cm.isReadOnly()) return CodeMirror.Pass
    # 执行编辑操作
    cm.operation(function() {
      # 获取当前选择的数量和新的选择数组
      var len = cm.listSelections().length, newSelection = [], last = -1;
      # 遍历当前选择
      for (var i = 0; i < len; i++) {
        # 获取当前选择的头部位置
        var head = cm.listSelections()[i].head;
        # 如果头部位置小于等于上一个位置，则继续下一次循环
        if (head.line <= last) continue;
        # 计算新行的位置
        var at = Pos(head.line + (above ? 0 : 1), 0);
        # 在新行的位置插入换行符
        cm.replaceRange("\n", at, null, "+insertLine");
        # 自动缩进新行
        cm.indentLine(at.line, null, true);
        # 将新行的位置添加到新选择数组中
        newSelection.push({head: at, anchor: at});
        # 更新上一个位置
        last = head.line + 1;
      }
      # 设置编辑器的选择范围为新选择数组中的内容
      cm.setSelections(newSelection);
    });
    # 执行自动缩进命令
    cm.execCommand("indentAuto");
  }

  # 定义在当前光标位置下方插入一行的命令
  cmds.insertLineAfter = function(cm) { return insertLine(cm, false); };

  # 定义在当前光标位置上方插入一行的命令
  cmds.insertLineBefore = function(cm) { return insertLine(cm, true); };

  # 获取指定位置的单词信息
  function wordAt(cm, pos) {
    # 初始化单词的起始和结束位置，以及所在行的内容
    var start = pos.ch, end = start, line = cm.getLine(pos.line);
    # 向前查找单词的起始位置
    while (start && CodeMirror.isWordChar(line.charAt(start - 1))) --start;
    # 向后查找单词的结束位置
    while (end < line.length && CodeMirror.isWordChar(line.charAt(end))) ++end;
    # 返回单词的起始位置、结束位置和内容
    return {from: Pos(pos.line, start), to: Pos(pos.line, end), word: line.slice(start, end)};
  }

  # 定义选择下一个相同单词的命令
  cmds.selectNextOccurrence = function(cm) {
    # 获取当前光标的起始位置和结束位置
    var from = cm.getCursor("from"), to = cm.getCursor("to");
    # 判断是否需要选择整个单词
    var fullWord = cm.state.sublimeFindFullWord == cm.doc.sel;
    # 如果起始位置和结束位置相同
    if (CodeMirror.cmpPos(from, to) == 0) {
      # 获取当前位置的单词信息
      var word = wordAt(cm, from);
      # 如果没有找到单词，则返回
      if (!word.word) return;
      # 选择当前单词的范围
      cm.setSelection(word.from, word.to);
      # 设置需要选择整个单词
      fullWord = true;
  } else {
    // 获取选中文本
    var text = cm.getRange(from, to);
    // 根据是否全词匹配创建查询正则表达式
    var query = fullWord ? new RegExp("\\b" + text + "\\b") : text;
    // 获取当前光标位置的搜索游标
    var cur = cm.getSearchCursor(query, to);
    // 查找下一个匹配项
    var found = cur.findNext();
    // 如果没有找到，则从文档开头重新查找
    if (!found) {
      cur = cm.getSearchCursor(query, Pos(cm.firstLine(), 0));
      found = cur.findNext();
    }
    // 如果没有找到或者选中范围已经包含该匹配项，则返回
    if (!found || isSelectedRange(cm.listSelections(), cur.from(), cur.to())) return
    // 将匹配项添加到选中范围
    cm.addSelection(cur.from(), cur.to());
  }
  // 如果全词匹配，则保存当前选中范围
  if (fullWord)
    cm.state.sublimeFindFullWord = cm.doc.sel;
};

// 跳过当前匹配项并选择下一个匹配项
cmds.skipAndSelectNextOccurrence = function(cm) {
  // 保存上一个锚点和头部位置
  var prevAnchor = cm.getCursor("anchor"), prevHead = cm.getCursor("head");
  // 选择下一个匹配项
  cmds.selectNextOccurrence(cm);
  // 如果上一个锚点和头部位置不相同，则更新选中范围
  if (CodeMirror.cmpPos(prevAnchor, prevHead) != 0) {
    cm.doc.setSelections(cm.doc.listSelections()
        .filter(function (sel) {
          return sel.anchor != prevAnchor || sel.head != prevHead;
        }));
  }
}

// 将光标添加到选中范围的上一行
function addCursorToSelection(cm, dir) {
  var ranges = cm.listSelections(), newRanges = [];
  for (var i = 0; i < ranges.length; i++) {
    var range = ranges[i];
    var newAnchor = cm.findPosV(
        range.anchor, dir, "line", range.anchor.goalColumn);
    var newHead = cm.findPosV(
        range.head, dir, "line", range.head.goalColumn);
    newAnchor.goalColumn = range.anchor.goalColumn != null ?
        range.anchor.goalColumn : cm.cursorCoords(range.anchor, "div").left;
    newHead.goalColumn = range.head.goalColumn != null ?
        range.head.goalColumn : cm.cursorCoords(range.head, "div").left;
    var newRange = {anchor: newAnchor, head: newHead};
    newRanges.push(range);
    newRanges.push(newRange);
  }
  // 更新选中范围
  cm.setSelections(newRanges);
}
// 将光标添加到选中范围的上一行
cmds.addCursorToPrevLine = function(cm) { addCursorToSelection(cm, -1); };
// 将光标添加到选中范围的下一行
cmds.addCursorToNextLine = function(cm) { addCursorToSelection(cm, 1); };

// 判断选中范围是否包含指定范围
function isSelectedRange(ranges, from, to) {
  // 遍历 ranges 数组
  for (var i = 0; i < ranges.length; i++)
    // 检查当前 range 是否与给定 from 和 to 相等，如果相等则返回 true
    if (CodeMirror.cmpPos(ranges[i].from(), from) == 0 &&
        CodeMirror.cmpPos(ranges[i].to(), to) == 0) return true
  // 如果没有相等的 range，则返回 false
  return false
}

// 定义 mirror 变量，包含括号和方括号
var mirror = "(){}[]";
// 定义 selectBetweenBrackets 函数，参数为 cm
function selectBetweenBrackets(cm) {
  // 获取当前选中的 ranges
  var ranges = cm.listSelections(), newRanges = []
  // 遍历 ranges 数组
  for (var i = 0; i < ranges.length; i++) {
    // 获取当前 range 的头部位置
    var range = ranges[i], pos = range.head, opening = cm.scanForBracket(pos, -1);
    // 如果没有找到匹配的开括号，则返回 false
    if (!opening) return false;
    // 循环查找匹配的闭括号
    for (;;) {
      var closing = cm.scanForBracket(pos, 1);
      // 如果没有找到匹配的闭括号，则返回 false
      if (!closing) return false;
      // 如果找到匹配的闭括号，则判断是否与对应的开括号匹配
      if (closing.ch == mirror.charAt(mirror.indexOf(opening.ch) + 1)) {
        // 获取起始位置
        var startPos = Pos(opening.pos.line, opening.pos.ch + 1);
        // 如果起始位置和结束位置与 range 的 from 和 to 相等，则继续查找下一个开括号
        if (CodeMirror.cmpPos(startPos, range.from()) == 0 &&
            CodeMirror.cmpPos(closing.pos, range.to()) == 0) {
          opening = cm.scanForBracket(opening.pos, -1);
          // 如果没有找到匹配的开括号，则返回 false
          if (!opening) return false;
        } else {
          // 将匹配的括号范围添加到 newRanges 数组中
          newRanges.push({anchor: startPos, head: closing.pos});
          break;
        }
      }
      // 更新位置为下一个字符
      pos = Pos(closing.pos.line, closing.pos.ch + 1);
    }
  }
  // 设置新的选中范围
  cm.setSelections(newRanges);
  return true;
}

// 定义 selectScope 命令，参数为 cm
cmds.selectScope = function(cm) {
  // 调用 selectBetweenBrackets 函数，如果返回 false 则执行 selectAll 命令
  selectBetweenBrackets(cm) || cm.execCommand("selectAll");
};
// 定义 selectBetweenBrackets 命令，参数为 cm
cmds.selectBetweenBrackets = function(cm) {
  // 如果 selectBetweenBrackets 函数返回 false，则返回 CodeMirror.Pass
  if (!selectBetweenBrackets(cm)) return CodeMirror.Pass;
};

// 定义 puncType 函数，参数为 type
function puncType(type) {
  // 如果 type 为假值，则返回 null，否则返回 type 是否包含 'punctuation' 的正则匹配结果
  return !type ? null : /\bpunctuation\b/.test(type) ? type : undefined
}

// 定义 goToBracket 命令，参数为 cm
cmds.goToBracket = function(cm) {
  // 根据当前位置和类型查找下一个匹配的括号位置
  cm.extendSelectionsBy(function(range) {
    var next = cm.scanForBracket(range.head, 1, puncType(cm.getTokenTypeAt(range.head)));
    // 如果找到下一个匹配的括号位置，则返回该位置
    if (next && CodeMirror.cmpPos(next.pos, range.head) != 0) return next.pos;
    // 否则，查找前一个匹配的括号位置
    var prev = cm.scanForBracket(range.head, -1, puncType(cm.getTokenTypeAt(Pos(range.head.line, range.head.ch + 1))));
    // 如果找到前一个匹配的括号位置，则返回该位置，否则返回当前位置
    return prev && Pos(prev.pos.line, prev.pos.ch + 1) || range.head;
  // 定义 swapLineUp 命令，用于将选中的行向上移动
  cmds.swapLineUp = function(cm) {
    // 如果编辑器是只读的，则返回 CodeMirror.Pass
    if (cm.isReadOnly()) return CodeMirror.Pass
    // 获取选中的文本范围和行数
    var ranges = cm.listSelections(), linesToMove = [], at = cm.firstLine() - 1, newSels = [];
    // 遍历选中的文本范围
    for (var i = 0; i < ranges.length; i++) {
      var range = ranges[i], from = range.from().line - 1, to = range.to().line;
      // 更新新的选择范围
      newSels.push({anchor: Pos(range.anchor.line - 1, range.anchor.ch),
                    head: Pos(range.head.line - 1, range.head.ch)});
      // 如果选中的行不为空且结束位置为0，则将结束位置减1
      if (range.to().ch == 0 && !range.empty()) --to;
      // 判断行是否需要移动，并更新行数
      if (from > at) linesToMove.push(from, to);
      else if (linesToMove.length) linesToMove[linesToMove.length - 1] = to;
      at = to;
    }
    // 执行操作
    cm.operation(function() {
      // 遍历需要移动的行
      for (var i = 0; i < linesToMove.length; i += 2) {
        var from = linesToMove[i], to = linesToMove[i + 1];
        var line = cm.getLine(from);
        // 移动行
        cm.replaceRange("", Pos(from, 0), Pos(from + 1, 0), "+swapLine");
        // 判断是否需要在末尾添加新行
        if (to > cm.lastLine())
          cm.replaceRange("\n" + line, Pos(cm.lastLine()), null, "+swapLine");
        else
          cm.replaceRange(line + "\n", Pos(to, 0), null, "+swapLine");
      }
      // 更新选择范围并滚动视图
      cm.setSelections(newSels);
      cm.scrollIntoView();
    });
  };

  // 定义 swapLineDown 命令，用于将选中的行向下移动
  cmds.swapLineDown = function(cm) {
    // 如果编辑器是只读的，则返回 CodeMirror.Pass
    if (cm.isReadOnly()) return CodeMirror.Pass
    // 获取选中的文本范围和行数
    var ranges = cm.listSelections(), linesToMove = [], at = cm.lastLine() + 1;
    // 倒序遍历选中的文本范围
    for (var i = ranges.length - 1; i >= 0; i--) {
      var range = ranges[i], from = range.to().line + 1, to = range.from().line;
      // 如果选中的行不为空且结束位置为0，则将结束位置减1
      if (range.to().ch == 0 && !range.empty()) from--;
      // 判断行是否需要移动，并更新行数
      if (from < at) linesToMove.push(from, to);
      else if (linesToMove.length) linesToMove[linesToMove.length - 1] = to;
      at = to;
    }
    // 执行操作函数
    cm.operation(function() {
      // 遍历需要移动的行
      for (var i = linesToMove.length - 2; i >= 0; i -= 2) {
        // 获取起始行和目标行
        var from = linesToMove[i], to = linesToMove[i + 1];
        // 获取起始行的内容
        var line = cm.getLine(from);
        // 如果起始行是最后一行，则删除起始行
        if (from == cm.lastLine())
          cm.replaceRange("", Pos(from - 1), Pos(from), "+swapLine");
        // 否则删除起始行到下一行的内容
        else
          cm.replaceRange("", Pos(from, 0), Pos(from + 1, 0), "+swapLine");
        // 在目标行插入起始行的内容
        cm.replaceRange(line + "\n", Pos(to, 0), null, "+swapLine");
      }
      // 滚动视图以确保操作后的内容可见
      cm.scrollIntoView();
    });
  };

  // 切换缩进注释
  cmds.toggleCommentIndented = function(cm) {
    cm.toggleComment({ indent: true });
  }

  // 合并多行为一行
  cmds.joinLines = function(cm) {
    // 获取选中的文本范围
    var ranges = cm.listSelections(), joined = [];
    // 遍历选中的文本范围
    for (var i = 0; i < ranges.length; i++) {
      var range = ranges[i], from = range.from();
      var start = from.line, end = range.to().line;
      // 合并相邻的选中文本范围
      while (i < ranges.length - 1 && ranges[i + 1].from().line == end)
        end = ranges[++i].to().line;
      joined.push({start: start, end: end, anchor: !range.empty() && from});
    }
    // 执行操作函数
    cm.operation(function() {
      var offset = 0, ranges = [];
      // 遍历合并后的文本范围
      for (var i = 0; i < joined.length; i++) {
        var obj = joined[i];
        var anchor = obj.anchor && Pos(obj.anchor.line - offset, obj.anchor.ch), head;
        // 遍历合并后的每一行
        for (var line = obj.start; line <= obj.end; line++) {
          var actual = line - offset;
          // 如果不是最后一行，则删除行尾空白字符并增加偏移量
          if (line == obj.end) head = Pos(actual, cm.getLine(actual).length + 1);
          if (actual < cm.lastLine()) {
            cm.replaceRange(" ", Pos(actual), Pos(actual + 1, /^\s*/.exec(cm.getLine(actual + 1))[0].length));
            ++offset;
          }
        }
        ranges.push({anchor: anchor || head, head: head});
      }
      // 设置新的文本范围
      cm.setSelections(ranges, 0);
    });
  };

  // 复制当前行
  cmds.duplicateLine = function(cm) {
  // 定义操作函数，接受一个匿名函数作为参数
  cm.operation(function() {
    // 获取当前选择的文本范围数量
    var rangeCount = cm.listSelections().length;
    // 遍历每个文本范围
    for (var i = 0; i < rangeCount; i++) {
      // 获取当前文本范围
      var range = cm.listSelections()[i];
      // 如果范围为空
      if (range.empty())
        // 替换范围内的内容为当前行内容加上换行符
        cm.replaceRange(cm.getLine(range.head.line) + "\n", Pos(range.head.line, 0));
      else
        // 否则替换范围内的内容为选中的文本
        cm.replaceRange(cm.getRange(range.from(), range.to()), range.from());
    }
    // 滚动视图以确保操作后的内容可见
    cm.scrollIntoView();
  });

  // 定义排序函数，接受 CodeMirror 对象和是否区分大小写作为参数
  function sortLines(cm, caseSensitive) {
    // 如果编辑器为只读状态，则返回
    if (cm.isReadOnly()) return CodeMirror.Pass
    // 获取选择的文本范围
    var ranges = cm.listSelections(), toSort = [], selected;
    // 遍历每个文本范围
    for (var i = 0; i < ranges.length; i++) {
      // 获取当前文本范围
      var range = ranges[i];
      // 如果范围为空，则继续下一次循环
      if (range.empty()) continue;
      // 获取范围的起始行和结束行
      var from = range.from().line, to = range.to().line;
      // 如果下一个范围的起始行与当前范围的结束行相同，则合并范围
      while (i < ranges.length - 1 && ranges[i + 1].from().line == to)
        to = ranges[++i].to().line;
      // 如果范围的结束位置为行首，则结束行减一
      if (!ranges[i].to().ch) to--;
      // 将需要排序的范围起始行和结束行添加到数组中
      toSort.push(from, to);
    }
    // 如果存在需要排序的范围，则设置 selected 为 true
    if (toSort.length) selected = true;
    else toSort.push(cm.firstLine(), cm.lastLine());

    // 执行操作函数
    cm.operation(function() {
      // 定义范围数组
      var ranges = [];
      // 遍历需要排序的范围
      for (var i = 0; i < toSort.length; i += 2) {
        // 获取范围的起始行和结束行
        var from = toSort[i], to = toSort[i + 1];
        // 创建起始和结束位置
        var start = Pos(from, 0), end = Pos(to);
        // 获取范围内的文本内容
        var lines = cm.getRange(start, end, false);
        // 如果区分大小写，则直接排序
        if (caseSensitive)
          lines.sort();
        else
          // 否则不区分大小写排序
          lines.sort(function(a, b) {
            var au = a.toUpperCase(), bu = b.toUpperCase();
            if (au != bu) { a = au; b = bu; }
            return a < b ? -1 : a == b ? 0 : 1;
          });
        // 替换范围内的内容为排序后的文本
        cm.replaceRange(lines, start, end);
        // 如果存在需要选择的范围，则添加到范围数组中
        if (selected) ranges.push({anchor: start, head: Pos(to + 1, 0)});
      }
      // 如果存在需要选择的范围，则设置编辑器的选择范围
      if (selected) cm.setSelections(ranges, 0);
    });
  }

  // 定义排序命令，调用排序函数并设置区分大小写为 true
  cmds.sortLines = function(cm) { sortLines(cm, true); };
  // 定义不区分大小写排序命令，调用排序函数并设置区分大小写为 false
  cmds.sortLinesInsensitive = function(cm) { sortLines(cm, false); };

  // 定义下一个书签命令
  cmds.nextBookmark = function(cm) {
    // 获取编辑器的书签状态
    var marks = cm.state.sublimeBookmarks;
  // 如果存在书签，则循环处理每个书签
  if (marks) while (marks.length) {
    // 从书签数组中取出第一个书签
    var current = marks.shift();
    // 查找当前书签所在位置
    var found = current.find();
    // 如果找到了书签所在位置
    if (found) {
      // 将当前书签重新放回书签数组的末尾
      marks.push(current);
      // 设置编辑器的选中范围为书签所在位置
      return cm.setSelection(found.from, found.to);
    }
  }
};

// 向前查找书签
cmds.prevBookmark = function(cm) {
  // 获取当前编辑器中的书签数组
  var marks = cm.state.sublimeBookmarks;
  // 如果存在书签，则循环处理每个书签
  if (marks) while (marks.length) {
    // 将书签数组的最后一个书签移到数组的开头
    marks.unshift(marks.pop());
    // 查找最后一个书签所在位置
    var found = marks[marks.length - 1].find();
    // 如果未找到书签所在位置
    if (!found)
      // 移除最后一个书签
      marks.pop();
    else
      // 设置编辑器的选中范围为书签所在位置
      return cm.setSelection(found.from, found.to);
  }
};

// 切换书签状态
cmds.toggleBookmark = function(cm) {
  // 获取当前编辑器中的选中范围
  var ranges = cm.listSelections();
  // 获取当前编辑器中的书签数组，如果不存在则初始化为空数组
  var marks = cm.state.sublimeBookmarks || (cm.state.sublimeBookmarks = []);
  // 循环处理每个选中范围
  for (var i = 0; i < ranges.length; i++) {
    // 获取选中范围的起始和结束位置
    var from = ranges[i].from(), to = ranges[i].to();
    // 在选中范围内查找书签
    var found = ranges[i].empty() ? cm.findMarksAt(from) : cm.findMarks(from, to);
    // 循环处理每个找到的书签
    for (var j = 0; j < found.length; j++) {
      // 如果找到了已存在的书签
      if (found[j].sublimeBookmark) {
        // 清除该书签
        found[j].clear();
        // 在书签数组中找到并移除该书签
        for (var k = 0; k < marks.length; k++)
          if (marks[k] == found[j])
            marks.splice(k--, 1);
        break;
      }
    }
    // 如果未找到已存在的书签
    if (j == found.length)
      // 在选中范围内创建新的书签
      marks.push(cm.markText(from, to, {sublimeBookmark: true, clearWhenEmpty: false}));
  }
};

// 清除所有书签
cmds.clearBookmarks = function(cm) {
  // 获取当前编辑器中的书签数组
  var marks = cm.state.sublimeBookmarks;
  // 如果存在书签，则循环清除每个书签
  if (marks) for (var i = 0; i < marks.length; i++) marks[i].clear();
  // 清空书签数组
  marks.length = 0;
};

// 选择所有书签
cmds.selectBookmarks = function(cm) {
  // 获取当前编辑器中的书签数组和选中范围数组
  var marks = cm.state.sublimeBookmarks, ranges = [];
  // 如果存在书签，则循环处理每个书签
  if (marks) for (var i = 0; i < marks.length; i++) {
    // 查找每个书签所在位置
    var found = marks[i].find();
    // 如果未找到书签所在位置，则移除该书签
    if (!found)
      marks.splice(i--, 0);
    else
      // 将书签所在位置添加到选中范围数组中
      ranges.push({anchor: found.from, head: found.to});
  }
  // 如果存在选中范围，则设置编辑器的选中范围为选中范围数组中的位置
  if (ranges.length)
    cm.setSelections(ranges, 0);
};

// 修改单词或选中范围
function modifyWordOrSelection(cm, mod) {
    # 执行编辑操作
    cm.operation(function() {
      # 获取当前选中文本的范围
      var ranges = cm.listSelections(), indices = [], replacements = [];
      # 遍历选中文本的范围
      for (var i = 0; i < ranges.length; i++) {
        var range = ranges[i];
        # 如果范围为空，则将索引添加到数组中，并将替换内容置为空字符串
        if (range.empty()) { indices.push(i); replacements.push(""); }
        # 否则，将替换内容设置为经过修改的选中文本
        else replacements.push(mod(cm.getRange(range.from(), range.to())));
      }
      # 用替换内容替换选中文本
      cm.replaceSelections(replacements, "around", "case");
      # 遍历索引数组，逆序处理选中文本
      for (var i = indices.length - 1, at; i >= 0; i--) {
        var range = ranges[indices[i]];
        # 如果存在at且当前选中文本的位置大于at，则继续下一次循环
        if (at && CodeMirror.cmpPos(range.head, at) > 0) continue;
        # 获取当前选中文本所在单词的范围
        var word = wordAt(cm, range.head);
        at = word.from;
        # 用修改后的单词替换当前选中文本
        cm.replaceRange(mod(word.word), word.from, word.to);
      }
    });
  }

  # 定义智能退格操作
  cmds.smartBackspace = function(cm) {
    # 如果有选中文本，则返回CodeMirror.Pass
    if (cm.somethingSelected()) return CodeMirror.Pass;

    # 执行编辑操作
    cm.operation(function() {
      # 获取当前光标位置
      var cursors = cm.listSelections();
      var indentUnit = cm.getOption("indentUnit");

      # 逆序遍历光标数组
      for (var i = cursors.length - 1; i >= 0; i--) {
        var cursor = cursors[i].head;
        # 获取光标所在行的起始位置到光标位置的文本
        var toStartOfLine = cm.getRange({line: cursor.line, ch: 0}, cursor);
        # 计算光标所在列的位置
        var column = CodeMirror.countColumn(toStartOfLine, null, cm.getOption("tabSize"));

        # 默认向左删除一个字符
        var deletePos = cm.findPosH(cursor, -1, "char", false);

        # 如果光标所在行为空且光标所在列为缩进单位的整数倍
        if (toStartOfLine && !/\S/.test(toStartOfLine) && column % indentUnit == 0) {
          # 获取前一个缩进位置
          var prevIndent = new Pos(cursor.line,
            CodeMirror.findColumn(toStartOfLine, column - indentUnit, indentUnit));

          # 只有在找到有效的前一个缩进位置时才进行智能删除
          if (prevIndent.ch != cursor.ch) deletePos = prevIndent;
        }

        # 删除光标位置到指定位置的文本
        cm.replaceRange("", deletePos, cursor, "+delete");
      }
    });
  };

  # 定义向右删除整行操作
  cmds.delLineRight = function(cm) {
  // 定义一个操作函数，接受一个函数作为参数
  cm.operation(function() {
    // 获取当前编辑器中所有选中文本的范围
    var ranges = cm.listSelections();
    // 遍历选中文本的范围，逐个删除
    for (var i = ranges.length - 1; i >= 0; i--)
      cm.replaceRange("", ranges[i].anchor, Pos(ranges[i].to().line), "+delete");
    // 滚动编辑器视图，确保光标可见
    cm.scrollIntoView();
  });

  // 定义一个在光标处将文本转换为大写的命令
  cmds.upcaseAtCursor = function(cm) {
    modifyWordOrSelection(cm, function(str) { return str.toUpperCase(); });
  };
  // 定义一个在光标处将文本转换为小写的命令
  cmds.downcaseAtCursor = function(cm) {
    modifyWordOrSelection(cm, function(str) { return str.toLowerCase(); });
  };

  // 定义一个设置Sublime标记的命令
  cmds.setSublimeMark = function(cm) {
    // 如果已经存在Sublime标记，则清除
    if (cm.state.sublimeMark) cm.state.sublimeMark.clear();
    // 设置Sublime标记为当前光标位置
    cm.state.sublimeMark = cm.setBookmark(cm.getCursor());
  };
  // 定义一个选择到Sublime标记的命令
  cmds.selectToSublimeMark = function(cm) {
    // 查找Sublime标记，并将光标移动到该位置
    var found = cm.state.sublimeMark && cm.state.sublimeMark.find();
    if (found) cm.setSelection(cm.getCursor(), found);
  };
  // 定义一个删除到Sublime标记的命令
  cmds.deleteToSublimeMark = function(cm) {
    // 查找Sublime标记，并删除光标位置到标记位置的文本
    var found = cm.state.sublimeMark && cm.state.sublimeMark.find();
    if (found) {
      var from = cm.getCursor(), to = found;
      if (CodeMirror.cmpPos(from, to) > 0) { var tmp = to; to = from; from = tmp; }
      cm.state.sublimeKilled = cm.getRange(from, to);
      cm.replaceRange("", from, to);
    }
  };
  // 定义一个与Sublime标记交换位置的命令
  cmds.swapWithSublimeMark = function(cm) {
    // 查找Sublime标记，并将光标位置与标记位置交换
    var found = cm.state.sublimeMark && cm.state.sublimeMark.find();
    if (found) {
      cm.state.sublimeMark.clear();
      cm.state.sublimeMark = cm.setBookmark(cm.getCursor());
      cm.setCursor(found);
    }
  };
  // 定义一个Sublime粘贴的命令
  cmds.sublimeYank = function(cm) {
    // 如果存在Sublime删除的文本，则粘贴到光标位置
    if (cm.state.sublimeKilled != null)
      cm.replaceSelection(cm.state.sublimeKilled, null, "paste");
  };

  // 定义一个在编辑器中心显示光标位置的命令
  cmds.showInCenter = function(cm) {
    // 获取光标位置的坐标，并滚动编辑器视图使其居中显示
    var pos = cm.cursorCoords(null, "local");
    cm.scrollTo(null, (pos.top + pos.bottom) / 2 - cm.getScrollInfo().clientHeight / 2);
  };

  // 定义一个获取目标范围的函数
  function getTarget(cm) {
    // 获取光标起始位置和结束位置的范围
    var from = cm.getCursor("from"), to = cm.getCursor("to");
    # 如果起始位置和结束位置相同
    if (CodeMirror.cmpPos(from, to) == 0):
      # 获取起始位置的单词
      var word = wordAt(cm, from);
      # 如果没有单词则返回
      if (!word.word) return;
      # 更新起始位置和结束位置为单词的起始和结束位置
      from = word.from;
      to = word.to;
    # 返回起始位置、结束位置、查询内容和单词
    return {from: from, to: to, query: cm.getRange(from, to), word: word};
  }

  # 查找并跳转到目标位置
  function findAndGoTo(cm, forward):
    # 获取目标位置
    var target = getTarget(cm);
    # 如果没有目标位置则返回
    if (!target) return;
    # 获取查询内容
    var query = target.query;
    # 获取搜索光标
    var cur = cm.getSearchCursor(query, forward ? target.to : target.from);

    # 如果向前查找则找到下一个匹配，否则找到上一个匹配
    if (forward ? cur.findNext() : cur.findPrevious()):
      # 设置选中内容为匹配的位置
      cm.setSelection(cur.from(), cur.to());
    else:
      # 重新获取搜索光标
      cur = cm.getSearchCursor(query, forward ? Pos(cm.firstLine(), 0)
                                              : cm.clipPos(Pos(cm.lastLine())));
      # 如果向前查找则找到下一个匹配，否则找到上一个匹配
      if (forward ? cur.findNext() : cur.findPrevious()):
        # 设置选中内容为匹配的位置
        cm.setSelection(cur.from(), cur.to());
      # 如果存在单词
      else if (target.word)
        # 设置选中内容为单词的起始和结束位置
        cm.setSelection(target.from, target.to);
    # 定义命令 findUnder
  cmds.findUnder = function(cm) { findAndGoTo(cm, true); };
  # 定义命令 findUnderPrevious
  cmds.findUnderPrevious = function(cm) { findAndGoTo(cm,false); };
  # 定义命令 findAllUnder
  cmds.findAllUnder = function(cm):
    # 获取目标位置
    var target = getTarget(cm);
    # 如果没有目标位置则返回
    if (!target) return;
    # 获取搜索光标
    var cur = cm.getSearchCursor(target.query);
    # 定义匹配数组
    var matches = [];
    # 定义主要索引
    var primaryIndex = -1;
    # 循环查找所有匹配
    while (cur.findNext()):
      # 将匹配的起始和结束位置添加到匹配数组
      matches.push({anchor: cur.from(), head: cur.to()});
      # 如果匹配的起始位置在目标位置之前，则主要索引加一
      if (cur.from().line <= target.from.line && cur.from().ch <= target.from.ch)
        primaryIndex++;
    # 设置选中内容为匹配数组中的位置，主要索引为主要索引
    cm.setSelections(matches, primaryIndex);


  # 定义键盘映射
  var keyMap = CodeMirror.keyMap;
  keyMap.macSublime = {
    "Cmd-Left": "goLineStartSmart",
    "Shift-Tab": "indentLess",
    "Shift-Ctrl-K": "deleteLine",
    "Alt-Q": "wrapLines",
    "Ctrl-Left": "goSubwordLeft",
    "Ctrl-Right": "goSubwordRight",
    "Ctrl-Alt-Up": "scrollLineUp",
    "Ctrl-Alt-Down": "scrollLineDown",
    "Cmd-L": "selectLine",
    "Shift-Cmd-L": "splitSelectionByLine",
    "Esc": "singleSelectionTop",
    "Cmd-Enter": "insertLineAfter",
    "Shift-Cmd-Enter": "insertLineBefore",
    # 定义键盘快捷键映射，选择下一个出现的内容
    "Cmd-D": "selectNextOccurrence",
    # 定义键盘快捷键映射，选择范围
    "Shift-Cmd-Space": "selectScope",
    # 定义键盘快捷键映射，选择括号之间的内容
    "Shift-Cmd-M": "selectBetweenBrackets",
    # 定义键盘快捷键映射，跳转到括号
    "Cmd-M": "goToBracket",
    # 定义键盘快捷键映射，向上交换当前行和上一行
    "Cmd-Ctrl-Up": "swapLineUp",
    # 定义键盘快捷键映射，向下交换当前行和下一行
    "Cmd-Ctrl-Down": "swapLineDown",
    # 定义键盘快捷键映射，切换缩进的注释
    "Cmd-/": "toggleCommentIndented",
    # 定义键盘快捷键映射，合并当前行和下一行
    "Cmd-J": "joinLines",
    # 定义键盘快捷键映射，复制当前行
    "Shift-Cmd-D": "duplicateLine",
    # 定义键盘快捷键映射，对选中的行进行排序
    "F5": "sortLines",
    # 定义键盘快捷键映射，对选中的行进行不区分大小写的排序
    "Cmd-F5": "sortLinesInsensitive",
    # 定义键盘快捷键映射，跳转到下一个书签
    "F2": "nextBookmark",
    # 定义键盘快捷键映射，跳转到上一个书签
    "Shift-F2": "prevBookmark",
    # 定义键盘快捷键映射，切换书签
    "Cmd-F2": "toggleBookmark",
    # 定义键盘快捷键映射，清除所有书签
    "Shift-Cmd-F2": "clearBookmarks",
    # 定义键盘快捷键映射，选择所有书签
    "Alt-F2": "selectBookmarks",
    # 定义键盘快捷键映射，智能退格
    "Backspace": "smartBackspace",
    # 定义键盘快捷键映射，跳过并选择下一个出现的内容
    "Cmd-K Cmd-D": "skipAndSelectNextOccurrence",
    # 定义键盘快捷键映射，删除右侧的行
    "Cmd-K Cmd-K": "delLineRight",
    # 定义键盘快捷键映射，将光标处的内容转换为大写
    "Cmd-K Cmd-U": "upcaseAtCursor",
    # 定义键盘快捷键映射，将光标处的内容转换为小写
    "Cmd-K Cmd-L": "downcaseAtCursor",
    # 定义键盘快捷键映射，设置 Sublime 标记
    "Cmd-K Cmd-Space": "setSublimeMark",
    # 定义键盘快捷键映射，选择到 Sublime 标记
    "Cmd-K Cmd-A": "selectToSublimeMark",
    # 定义键盘快捷键映射，删除到 Sublime 标记
    "Cmd-K Cmd-W": "deleteToSublimeMark",
    # 定义键盘快捷键映射，与 Sublime 标记交换内容
    "Cmd-K Cmd-X": "swapWithSublimeMark",
    # 定义键盘快捷键映射，Sublime 粘贴
    "Cmd-K Cmd-Y": "sublimeYank",
    # 定义键盘快捷键映射，在中心显示
    "Cmd-K Cmd-C": "showInCenter",
    # 定义键盘快捷键映射，清除所有书签
    "Cmd-K Cmd-G": "clearBookmarks",
    # 定义键盘快捷键映射，删除左侧的行
    "Cmd-K Cmd-Backspace": "delLineLeft",
    # 定义键盘快捷键映射，折叠所有内容
    "Cmd-K Cmd-1": "foldAll",
    # 定义键盘快捷键映射，展开所有内容
    "Cmd-K Cmd-0": "unfoldAll",
    # 定义键盘快捷键映射，展开所有内容
    "Cmd-K Cmd-J": "unfoldAll",
    # 定义键盘快捷键映射，将光标添加到上一行
    "Ctrl-Shift-Up": "addCursorToPrevLine",
    # 定义键盘快捷键映射，将光标添加到下一行
    "Ctrl-Shift-Down": "addCursorToNextLine",
    # 定义键盘快捷键映射，查找并选择下一个匹配内容
    "Cmd-F3": "findUnder",
    # 定义键盘快捷键映射，查找并选择上一个匹配内容
    "Shift-Cmd-F3": "findUnderPrevious",
    # 定义键盘快捷键映射，查找并选择所有匹配内容
    "Alt-F3": "findAllUnder",
    # 定义键盘快捷键映射，折叠内容
    "Shift-Cmd-[": "fold",
    # 定义键盘快捷键映射，展开内容
    "Shift-Cmd-]": "unfold",
    # 定义键盘快捷键映射，增量查找
    "Cmd-I": "findIncremental",
    # 定义键盘快捷键映射，反向增量查找
    "Shift-Cmd-I": "findIncrementalReverse",
    # 定义键盘快捷键映射，替换
    "Cmd-H": "replace",
    # 定义键盘快捷键映射，查找下一个匹配内容
    "F3": "findNext",
    # 定义键盘快捷键映射，查找上一个匹配内容
    "Shift-F3": "findPrev",
    # 定义键盘快捷键映射，mac 默认
    "fallthrough": "macDefault"
  };
  # 标准化键盘映射
  CodeMirror.normalizeKeyMap(keyMap.macSublime);

  # 定义 PC Sublime 键盘映射
  keyMap.pcSublime = {
    # 定义键盘快捷键映射，减少缩进
    "Shift-Tab": "indentLess",
    # 定义键盘快捷键映射，删除当前行
    "Shift-Ctrl-K": "deleteLine",
    # 定义键盘快捷键映射，换行
    "Alt-Q": "wrapLines",
    # 定义键盘快捷键映射，交换字符
    "Ctrl-T": "transposeChars",
    # 定义键盘快捷键映射，向左跳转一个子词
    "Alt-Left": "goSubwordLeft",
    # 定义键盘快捷键映射，向右跳转一个子词
    "Alt-Right": "goSubwordRight",
    # 定义键盘快捷键映射，向上滚动一行
    "Ctrl-Up": "scrollLineUp",
    # 定义键盘快捷键映射，向下滚动一行
    "Ctrl-Down": "scrollLineDown",
    # 定义键盘快捷键映射，选择当前行
    "Ctrl-L": "selectLine",
    # 定义键盘快捷键映射，根据行进行分割选择
    "Shift-Ctrl-L": "splitSelectionByLine",
    # 定义键盘快捷键映射，将"Esc"映射到"singleSelectionTop"
    "Esc": "singleSelectionTop",
    # 将"Ctrl-Enter"映射到"insertLineAfter"
    "Ctrl-Enter": "insertLineAfter",
    # 将"Shift-Ctrl-Enter"映射到"insertLineBefore"
    "Shift-Ctrl-Enter": "insertLineBefore",
    # 将"Ctrl-D"映射到"selectNextOccurrence"
    "Ctrl-D": "selectNextOccurrence",
    # 将"Shift-Ctrl-Space"映射到"selectScope"
    "Shift-Ctrl-Space": "selectScope",
    # 将"Shift-Ctrl-M"映射到"selectBetweenBrackets"
    "Shift-Ctrl-M": "selectBetweenBrackets",
    # 将"Ctrl-M"映射到"goToBracket"
    "Ctrl-M": "goToBracket",
    # 将"Shift-Ctrl-Up"映射到"swapLineUp"
    "Shift-Ctrl-Up": "swapLineUp",
    # 将"Shift-Ctrl-Down"映射到"swapLineDown"
    "Shift-Ctrl-Down": "swapLineDown",
    # 将"Ctrl-/"映射到"toggleCommentIndented"
    "Ctrl-/": "toggleCommentIndented",
    # 将"Ctrl-J"映射到"joinLines"
    "Ctrl-J": "joinLines",
    # 将"Shift-Ctrl-D"映射到"duplicateLine"
    "Shift-Ctrl-D": "duplicateLine",
    # 将"F9"映射到"sortLines"
    "F9": "sortLines",
    # 将"Ctrl-F9"映射到"sortLinesInsensitive"
    "Ctrl-F9": "sortLinesInsensitive",
    # 将"F2"映射到"nextBookmark"
    "F2": "nextBookmark",
    # 将"Shift-F2"映射到"prevBookmark"
    "Shift-F2": "prevBookmark",
    # 将"Ctrl-F2"映射到"toggleBookmark"
    "Ctrl-F2": "toggleBookmark",
    # 将"Shift-Ctrl-F2"映射到"clearBookmarks"
    "Shift-Ctrl-F2": "clearBookmarks",
    # 将"Alt-F2"映射到"selectBookmarks"
    "Alt-F2": "selectBookmarks",
    # 将"Backspace"映射到"smartBackspace"
    "Backspace": "smartBackspace",
    # 将"Ctrl-K Ctrl-D"映射到"skipAndSelectNextOccurrence"
    "Ctrl-K Ctrl-D": "skipAndSelectNextOccurrence",
    # 将"Ctrl-K Ctrl-K"映射到"delLineRight"
    "Ctrl-K Ctrl-K": "delLineRight",
    # 将"Ctrl-K Ctrl-U"映射到"upcaseAtCursor"
    "Ctrl-K Ctrl-U": "upcaseAtCursor",
    # 将"Ctrl-K Ctrl-L"映射到"downcaseAtCursor"
    "Ctrl-K Ctrl-L": "downcaseAtCursor",
    # 将"Ctrl-K Ctrl-Space"映射到"setSublimeMark"
    "Ctrl-K Ctrl-Space": "setSublimeMark",
    # 将"Ctrl-K Ctrl-A"映射到"selectToSublimeMark"
    "Ctrl-K Ctrl-A": "selectToSublimeMark",
    # 将"Ctrl-K Ctrl-W"映射到"deleteToSublimeMark"
    "Ctrl-K Ctrl-W": "deleteToSublimeMark",
    # 将"Ctrl-K Ctrl-X"映射到"swapWithSublimeMark"
    "Ctrl-K Ctrl-X": "swapWithSublimeMark",
    # 将"Ctrl-K Ctrl-Y"映射到"sublimeYank"
    "Ctrl-K Ctrl-Y": "sublimeYank",
    # 将"Ctrl-K Ctrl-C"映射到"showInCenter"
    "Ctrl-K Ctrl-C": "showInCenter",
    # 将"Ctrl-K Ctrl-G"映射到"clearBookmarks"
    "Ctrl-K Ctrl-G": "clearBookmarks",
    # 将"Ctrl-K Ctrl-Backspace"映射到"delLineLeft"
    "Ctrl-K Ctrl-Backspace": "delLineLeft",
    # 将"Ctrl-K Ctrl-1"映射到"foldAll"
    "Ctrl-K Ctrl-1": "foldAll",
    # 将"Ctrl-K Ctrl-0"映射到"unfoldAll"
    "Ctrl-K Ctrl-0": "unfoldAll",
    # 将"Ctrl-K Ctrl-J"映射到"unfoldAll"
    "Ctrl-K Ctrl-J": "unfoldAll",
    # 将"Ctrl-Alt-Up"映射到"addCursorToPrevLine"
    "Ctrl-Alt-Up": "addCursorToPrevLine",
    # 将"Ctrl-Alt-Down"映射到"addCursorToNextLine"
    "Ctrl-Alt-Down": "addCursorToNextLine",
    # 将"Ctrl-F3"映射到"findUnder"
    "Ctrl-F3": "findUnder",
    # 将"Shift-Ctrl-F3"映射到"findUnderPrevious"
    "Shift-Ctrl-F3": "findUnderPrevious",
    # 将"Alt-F3"映射到"findAllUnder"
    "Alt-F3": "findAllUnder",
    # 将"Shift-Ctrl-["映射到"fold"
    "Shift-Ctrl-[": "fold",
    # 将"Shift-Ctrl-]"映射到"unfold"
    "Shift-Ctrl-]": "unfold",
    # 将"Ctrl-I"映射到"findIncremental"
    "Ctrl-I": "findIncremental",
    # 将"Shift-Ctrl-I"映射到"findIncrementalReverse"
    "Shift-Ctrl-I": "findIncrementalReverse",
    # 将"Ctrl-H"映射到"replace"
    "Ctrl-H": "replace",
    # 将"F3"映射到"findNext"
    "F3": "findNext",
    # 将"Shift-F3"映射到"findPrev"
    "Shift-F3": "findPrev",
    # 将"fallthrough"映射到"pcDefault"
    "fallthrough": "pcDefault"
    # 标准化键盘映射
    CodeMirror.normalizeKeyMap(keyMap.pcSublime);
    # 判断是否为 Mac 系统，选择相应的键盘映射
    var mac = keyMap.default == keyMap.macDefault;
    keyMap.sublime = mac ? keyMap.macSublime : keyMap.pcSublime;
# 闭合了一个代码块或者函数的结束
```