# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\search\search.js`

```py
// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

// 定义搜索命令。依赖于 dialog.js 或另一个
// 实现 openDialog 方法的库

// Replace 的工作方式有点奇怪 - 它将在下一次按下 Ctrl-G（或绑定到 findNext 的任何键）时进行替换。确保匹配在按下 Ctrl-G 时不再被选中，即可阻止替换。

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"), require("./searchcursor"), require("../dialog/dialog"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror", "./searchcursor", "../dialog/dialog"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  function searchOverlay(query, caseInsensitive) {
    if (typeof query == "string")
      query = new RegExp(query.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, "\\$&"), caseInsensitive ? "gi" : "g");
    else if (!query.global)
      query = new RegExp(query.source, query.ignoreCase ? "gi" : "g");

    return {token: function(stream) {
      query.lastIndex = stream.pos;
      var match = query.exec(stream.string);
      if (match && match.index == stream.pos) {
        stream.pos += match[0].length || 1;
        return "searching";
      } else if (match) {
        stream.pos = match.index;
      } else {
        stream.skipToEnd();
      }
    }};
  }

  function SearchState() {
    this.posFrom = this.posTo = this.lastQuery = this.query = null;
    this.overlay = null;
  }

  function getSearchState(cm) {
    return cm.state.search || (cm.state.search = new SearchState());
  }

  function queryCaseInsensitive(query) {
    return typeof query == "string" && query == query.toLowerCase();
  }

  function getSearchCursor(cm, query, pos) {
  // 如果查询字符串全为小写，则进行不区分大小写的搜索
  return cm.getSearchCursor(query, pos, {caseFold: queryCaseInsensitive(query), multiline: true});
}

function persistentDialog(cm, text, deflt, onEnter, onKeyDown) {
  // 打开持久对话框，设置默认值、回车事件、按键事件等
  cm.openDialog(text, onEnter, {
    value: deflt,
    selectValueOnOpen: true,
    closeOnEnter: false,
    onClose: function() { clearSearch(cm); },
    onKeyDown: onKeyDown
  });
}

function dialog(cm, text, shortText, deflt, f) {
  // 如果支持对话框，则打开对话框，否则使用 prompt 函数
  if (cm.openDialog) cm.openDialog(text, f, {value: deflt, selectValueOnOpen: true});
  else f(prompt(shortText, deflt));
}

function confirmDialog(cm, text, shortText, fs) {
  // 如果支持确认对话框，则打开确认对话框，否则使用 confirm 函数
  if (cm.openConfirm) cm.openConfirm(text, fs);
  else if (confirm(shortText)) fs[0]();
}

function parseString(string) {
  // 解析字符串中的转义字符
  return string.replace(/\\([nrt\\])/g, function(match, ch) {
    if (ch == "n") return "\n"
    if (ch == "r") return "\r"
    if (ch == "t") return "\t"
    if (ch == "\\") return "\\"
    return match
  })
}

function parseQuery(query) {
  // 解析查询字符串，如果是正则表达式则转换为正则对象，否则解析转义字符
  var isRE = query.match(/^\/(.*)\/([a-z]*)$/);
  if (isRE) {
    try { query = new RegExp(isRE[1], isRE[2].indexOf("i") == -1 ? "" : "i"); }
    catch(e) {} // 如果不是正则表达式，则进行字符串搜索
  } else {
    query = parseString(query)
  }
  // 如果查询为空，则设置默认查询
  if (typeof query == "string" ? query == "" : query.test(""))
    query = /x^/;
  return query;
}

function startSearch(cm, state, query) {
  // 设置查询文本和查询对象
  state.queryText = query;
  state.query = parseQuery(query);
  // 移除旧的覆盖层，添加新的覆盖层
  cm.removeOverlay(state.overlay, queryCaseInsensitive(state.query));
  state.overlay = searchOverlay(state.query, queryCaseInsensitive(state.query));
  cm.addOverlay(state.overlay);
  // 如果支持在滚动条上显示匹配项，则进行相应操作
  if (cm.showMatchesOnScrollbar) {
    if (state.annotate) { state.annotate.clear(); state.annotate = null; }
    state.annotate = cm.showMatchesOnScrollbar(state.query, queryCaseInsensitive(state.query));
    }
  }

  // 执行搜索操作
  function doSearch(cm, rev, persistent, immediate) {
    // 获取搜索状态
    var state = getSearchState(cm);
    // 如果存在查询条件，则执行查找下一个匹配项
    if (state.query) return findNext(cm, rev);
    // 获取当前选中的文本，如果没有则使用上次的查询条件
    var q = cm.getSelection() || state.lastQuery;
    // 如果查询条件是正则表达式并且内容为"x^"，则置空
    if (q instanceof RegExp && q.source == "x^") q = null
    // 如果需要持久化对话框，并且编辑器支持对话框
    if (persistent && cm.openDialog) {
      var hiding = null
      // 定义搜索下一个匹配项的函数
      var searchNext = function(query, event) {
        CodeMirror.e_stop(event);
        if (!query) return;
        if (query != state.queryText) {
          startSearch(cm, state, query);
          state.posFrom = state.posTo = cm.getCursor();
        }
        if (hiding) hiding.style.opacity = 1
        // 查找下一个匹配项
        findNext(cm, event.shiftKey, function(_, to) {
          var dialog
          // 如果匹配项在前三行，并且编辑器支持查询对话框
          if (to.line < 3 && document.querySelector &&
              (dialog = cm.display.wrapper.querySelector(".CodeMirror-dialog")) &&
              dialog.getBoundingClientRect().bottom - 4 > cm.cursorCoords(to, "window").top)
            (hiding = dialog).style.opacity = .4
        })
      };
      // 持久化对话框
      persistentDialog(cm, getQueryDialog(cm), q, searchNext, function(event, query) {
        var keyName = CodeMirror.keyName(event)
        var extra = cm.getOption('extraKeys'), cmd = (extra && extra[keyName]) || CodeMirror.keyMap[cm.getOption("keyMap")][keyName]
        if (cmd == "findNext" || cmd == "findPrev" ||
          cmd == "findPersistentNext" || cmd == "findPersistentPrev") {
          CodeMirror.e_stop(event);
          startSearch(cm, getSearchState(cm), query);
          cm.execCommand(cmd);
        } else if (cmd == "find" || cmd == "findPersistent") {
          CodeMirror.e_stop(event);
          searchNext(query, event);
        }
      });
      // 如果需要立即搜索并且存在查询条件，则开始搜索并查找下一个匹配项
      if (immediate && q) {
        startSearch(cm, state, q);
        findNext(cm, rev);
      }
  } else {
    # 如果条件不成立，弹出对话框，获取查询内容
    dialog(cm, getQueryDialog(cm), "Search for:", q, function(query) {
      # 如果查询内容存在且状态中的查询内容为空，则执行操作
      if (query && !state.query) cm.operation(function() {
        # 开始搜索
        startSearch(cm, state, query);
        # 设置搜索起始和结束位置为当前光标位置
        state.posFrom = state.posTo = cm.getCursor();
        # 查找下一个匹配项
        findNext(cm, rev);
      });
    });
  }
}

# 查找下一个匹配项
function findNext(cm, rev, callback) {cm.operation(function() {
  # 获取搜索状态
  var state = getSearchState(cm);
  # 获取搜索光标
  var cursor = getSearchCursor(cm, state.query, rev ? state.posFrom : state.posTo);
  # 如果没有找到匹配项
  if (!cursor.find(rev)) {
    # 设置搜索光标为文档末尾或开头
    cursor = getSearchCursor(cm, state.query, rev ? CodeMirror.Pos(cm.lastLine()) : CodeMirror.Pos(cm.firstLine(), 0));
    # 如果还是没有找到匹配项，则返回
    if (!cursor.find(rev)) return;
  }
  # 设置编辑器选中匹配项
  cm.setSelection(cursor.from(), cursor.to());
  # 滚动编辑器视图到匹配项位置
  cm.scrollIntoView({from: cursor.from(), to: cursor.to()}, 20);
  # 设置状态中的搜索起始和结束位置
  state.posFrom = cursor.from(); state.posTo = cursor.to();
  # 如果有回调函数，则执行回调
  if (callback) callback(cursor.from(), cursor.to())
});}

# 清除搜索状态
function clearSearch(cm) {cm.operation(function() {
  # 获取搜索状态
  var state = getSearchState(cm);
  # 将上次的查询内容保存到状态中
  state.lastQuery = state.query;
  # 如果查询内容为空，则返回
  if (!state.query) return;
  # 清空查询内容和查询文本
  state.query = state.queryText = null;
  # 移除搜索效果
  cm.removeOverlay(state.overlay);
  # 如果有标注，则清除标注
  if (state.annotate) { state.annotate.clear(); state.annotate = null; }
});}

# 获取查询对话框内容
function getQueryDialog(cm)  {
  return '<span class="CodeMirror-search-label">' + cm.phrase("Search:") + '</span> <input type="text" style="width: 10em" class="CodeMirror-search-field"/> <span style="color: #888" class="CodeMirror-search-hint">' + cm.phrase("(Use /re/ syntax for regexp search)") + '</span>';
}
# 获取替换查询对话框内容
function getReplaceQueryDialog(cm) {
  return ' <input type="text" style="width: 10em" class="CodeMirror-search-field"/> <span style="color: #888" class="CodeMirror-search-hint">' + cm.phrase("(Use /re/ syntax for regexp search)") + '</span>';
}
# 获取替换内容对话框内容
function getReplacementQueryDialog(cm) {
  # 返回一个包含搜索标签和输入框的字符串，用于搜索功能
  function getSearchLabel(cm) {
    return '<span class="CodeMirror-search-label">' + cm.phrase("With:") + '</span> <input type="text" style="width: 10em" class="CodeMirror-search-field"/>';
  }
  # 返回一个包含确认替换按钮的字符串，用于确认替换操作
  function getDoReplaceConfirm(cm) {
    return '<span class="CodeMirror-search-label">' + cm.phrase("Replace?") + '</span> <button>' + cm.phrase("Yes") + '</button> <button>' + cm.phrase("No") + '</button> <button>' + cm.phrase("All") + '</button> <button>' + cm.phrase("Stop") + '</button> ';
  }

  # 替换所有匹配的文本
  function replaceAll(cm, query, text) {
    # 执行替换操作
    cm.operation(function() {
      # 遍历所有匹配的位置
      for (var cursor = getSearchCursor(cm, query); cursor.findNext();) {
        # 如果查询条件不是字符串，则进行特殊处理
        if (typeof query != "string") {
          # 获取匹配的文本
          var match = cm.getRange(cursor.from(), cursor.to()).match(query);
          # 替换文本
          cursor.replace(text.replace(/\$(\d)/g, function(_, i) {return match[i];}));
        } else 
          # 直接替换文本
          cursor.replace(text);
      }
    });
  }

  # 执行替换操作
  function replace(cm, all) {
    # 如果编辑器是只读模式，则直接返回
    if (cm.getOption("readOnly")) return;
    # 获取选中的文本或上次查询的文本
    var query = cm.getSelection() || getSearchState(cm).lastQuery;
    # 构建对话框文本
    var dialogText = '<span class="CodeMirror-search-label">' + (all ? cm.phrase("Replace all:") : cm.phrase("Replace:")) + '</span>';
    // 弹出对话框，显示替换查询的文本，并等待用户输入
    dialog(cm, dialogText + getReplaceQueryDialog(cm), dialogText, query, function(query) {
      // 如果用户没有输入查询内容，则直接返回
      if (!query) return;
      // 解析用户输入的查询内容
      query = parseQuery(query);
      // 弹出对话框，显示替换文本的输入框，并等待用户输入
      dialog(cm, getReplacementQueryDialog(cm), cm.phrase("Replace with:"), "", function(text) {
        // 解析用户输入的替换文本
        text = parseString(text)
        // 如果需要替换所有匹配项
        if (all) {
          // 调用替换所有匹配项的函数
          replaceAll(cm, query, text)
        } else {
          // 清除搜索高亮
          clearSearch(cm);
          // 获取搜索光标对象
          var cursor = getSearchCursor(cm, query, cm.getCursor("from"));
          // 定义光标移动的函数
          var advance = function() {
            // 获取当前匹配项的起始位置
            var start = cursor.from(), match;
            // 如果没有找到下一个匹配项
            if (!(match = cursor.findNext())) {
              // 重置搜索光标对象
              cursor = getSearchCursor(cm, query);
              // 如果还是没有找到下一个匹配项，或者匹配项的位置没有改变，则返回
              if (!(match = cursor.findNext()) ||
                  (start && cursor.from().line == start.line && cursor.from().ch == start.ch)) return;
            }
            // 选中匹配项
            cm.setSelection(cursor.from(), cursor.to());
            // 滚动到匹配项的位置
            cm.scrollIntoView({from: cursor.from(), to: cursor.to()});
            // 弹出确认对话框，询问用户是否替换当前匹配项
            confirmDialog(cm, getDoReplaceConfirm(cm), cm.phrase("Replace?"),
                          [function() {doReplace(match);}, advance,
                           function() {replaceAll(cm, query, text)}]);
          };
          // 定义替换匹配项的函数
          var doReplace = function(match) {
            // 替换匹配项
            cursor.replace(typeof query == "string" ? text :
                           text.replace(/\$(\d)/g, function(_, i) {return match[i];}));
            // 移动光标到下一个匹配项
            advance();
          };
          // 调用光标移动函数，开始替换操作
          advance();
        }
      });
  // 定义一个匿名函数，该函数用于清除搜索结果
  });
  
  // 定义一个命令，用于触发搜索功能
  CodeMirror.commands.find = function(cm) {clearSearch(cm); doSearch(cm);};
  
  // 定义一个命令，用于触发持续搜索功能
  CodeMirror.commands.findPersistent = function(cm) {clearSearch(cm); doSearch(cm, false, true);};
  
  // 定义一个命令，用于触发持续搜索下一个功能
  CodeMirror.commands.findPersistentNext = function(cm) {doSearch(cm, false, true, true);};
  
  // 定义一个命令，用于触发持续搜索上一个功能
  CodeMirror.commands.findPersistentPrev = function(cm) {doSearch(cm, true, true, true);};
  
  // 定义一个命令，用于触发搜索下一个功能
  CodeMirror.commands.findNext = doSearch;
  
  // 定义一个命令，用于触发搜索上一个功能
  CodeMirror.commands.findPrev = function(cm) {doSearch(cm, true);};
  
  // 定义一个命令，用于触发清除搜索结果功能
  CodeMirror.commands.clearSearch = clearSearch;
  
  // 定义一个命令，用于触发替换功能
  CodeMirror.commands.replace = replace;
  
  // 定义一个命令，用于触发替换所有功能
  CodeMirror.commands.replaceAll = function(cm) {replace(cm, true);};
# 闭合了一个代码块或者函数的结束
```