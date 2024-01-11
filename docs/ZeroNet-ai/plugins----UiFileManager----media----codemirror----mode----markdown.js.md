# `ZeroNet\plugins\UiFileManager\media\codemirror\mode\markdown.js`

```
// 使用立即执行函数表达式（IIFE）定义模块
(function(mod) {
  // 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"), require("../xml/xml"), require("../meta"));
  // 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror", "../xml/xml", "../meta"], mod);
  // 如果是普通的浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  // 定义 Markdown 模式
  CodeMirror.defineMode("markdown", function(cmCfg, modeCfg) {

    // 获取 HTML 模式
    var htmlMode = CodeMirror.getMode(cmCfg, "text/html");
    // 检查 HTML 模式是否存在
    var htmlModeMissing = htmlMode.name == "null"

    // 获取指定名称的模式
    function getMode(name) {
      if (CodeMirror.findModeByName) {
        var found = CodeMirror.findModeByName(name);
        if (found) name = found.mime || found.mimes[0];
      }
      var mode = CodeMirror.getMode(cmCfg, name);
      return mode.name == "null" ? null : mode;
    }

    // 是否应该单独高亮影响语法高亮的字符？
    if (modeCfg.highlightFormatting === undefined)
      modeCfg.highlightFormatting = false;

    // 最大嵌套引用块数。设置为 0 表示无限嵌套。
    // 多余的 `>` 将发出 `error` 标记。
    if (modeCfg.maxBlockquoteDepth === undefined)
      modeCfg.maxBlockquoteDepth = 0;

    // 是否启用任务列表？（"- [ ] " 和 "- [x] "）
    if (modeCfg.taskLists === undefined) modeCfg.taskLists = false;

    // 是否启用删除线语法
    if (modeCfg.strikethrough === undefined)
      modeCfg.strikethrough = false;

    // 是否启用表情符号
    if (modeCfg.emoji === undefined)
      modeCfg.emoji = false;

    // 是否启用围栏代码块高亮
    if (modeCfg.fencedCodeBlockHighlighting === undefined)
      modeCfg.fencedCodeBlockHighlighting = true;

    // 围栏代码块默认模式
    if (modeCfg.fencedCodeBlockDefaultMode === undefined)
      modeCfg.fencedCodeBlockDefaultMode = 'text/plain';

    // 是否启用 XML 模式
    if (modeCfg.xml === undefined)
    // 设置 modeCfg.xml 为 true
    modeCfg.xml = true;

    // 允许用户提供的 token 类型覆盖默认的 token 类型
    if (modeCfg.tokenTypeOverrides === undefined)
        modeCfg.tokenTypeOverrides = {};

    // 定义默认的 token 类型
    var tokenTypes = {
        header: "header",
        code: "comment",
        quote: "quote",
        list1: "variable-2",
        list2: "variable-3",
        list3: "keyword",
        hr: "hr",
        image: "image",
        imageAltText: "image-alt-text",
        imageMarker: "image-marker",
        formatting: "formatting",
        linkInline: "link",
        linkEmail: "link",
        linkText: "link",
        linkHref: "string",
        em: "em",
        strong: "strong",
        strikethrough: "strikethrough",
        emoji: "builtin"
    };

    // 遍历 tokenTypes 对象，如果存在用户提供的 token 类型，则覆盖默认的 token 类型
    for (var tokenType in tokenTypes) {
        if (tokenTypes.hasOwnProperty(tokenType) && modeCfg.tokenTypeOverrides[tokenType]) {
            tokenTypes[tokenType] = modeCfg.tokenTypeOverrides[tokenType];
        state.f = state.inline = f;
        return f(stream, state);
    }

    // 定义 switchBlock 函数，用于切换块级状态
    function switchBlock(stream, state, f) {
        state.f = state.block = f;
        return f(stream, state);
    }

    // 判断一行是否为空
    function lineIsEmpty(line) {
        return !line || !/\S/.test(line.string)
    }

    // Blocks

    // 定义空行的处理函数
    function blankLine(state) {
        // 重置 linkTitle 状态
        state.linkTitle = false;
        state.linkHref = false;
        state.linkText = false;
        // 重置 EM 状态
        state.em = false;
        // 重置 STRONG 状态
        state.strong = false;
        // 重置 strikethrough 状态
        state.strikethrough = false;
        // 重置 quote 状态
        state.quote = 0;
        // 重置 indentedCode 状态
        state.indentedCode = false;
        if (state.f == htmlBlock) {
            var exit = htmlModeMissing
            if (!exit) {
                var inner = CodeMirror.innerMode(htmlMode, state.htmlState)
                exit = inner.mode.name == "xml" && inner.state.tagStart === null &&
                    (!inner.state.context && inner.state.tokenize.isInText)
            }
            if (exit) {
                state.f = inlineNormal;
                state.block = blockNormal;
                state.htmlState = null;
            }
        }
    }
    // 重置状态中的 trailingSpace
    state.trailingSpace = 0;
    state.trailingSpaceNewLine = false;
    // 标记这一行为空行
    state.prevLine = state.thisLine
    state.thisLine = {stream: null}
    // 返回空值
    return null;
  }

  function blockNormal(stream, state) {
    var firstTokenOnLine = stream.column() === state.indentation;
    var prevLineLineIsEmpty = lineIsEmpty(state.prevLine.stream);
    var prevLineIsIndentedCode = state.indentedCode;
    var prevLineIsHr = state.prevLine.hr;
    var prevLineIsList = state.list !== false;
    var maxNonCodeIndentation = (state.listStack[state.listStack.length - 1] || 0) + 3;

    state.indentedCode = false;

    var lineIndentation = state.indentation;
    // 每行计算一次（在第一个标记上）
    if (state.indentationDiff === null) {
      state.indentationDiff = state.indentation;
      if (prevLineIsList) {
        state.list = null;
        // 当此列表项的标记缩进小于最深列表项内容的缩进时，弹出最深列表项的缩进，并更新块缩进状态
        while (lineIndentation < state.listStack[state.listStack.length - 1]) {
          state.listStack.pop();
          if (state.listStack.length) {
            state.indentation = state.listStack[state.listStack.length - 1];
          // 小于第一个列表的缩进 -> 该行不再是列表
          } else {
            state.list = false;
          }
        }
        if (state.list !== false) {
          state.indentationDiff = lineIndentation - state.listStack[state.listStack.length - 1]
        }
      }
    }

    // 不全面（目前仅用于 setext 检测目的）
    var allowsInlineContinuation = (
        !prevLineLineIsEmpty && !prevLineIsHr && !state.prevLine.header &&
        (!prevLineIsList || !prevLineIsIndentedCode) &&
        !state.prevLine.fencedCodeEnd
    );
    # 检查当前行是否为水平线
    var isHr = (state.list === false || prevLineIsHr || prevLineLineIsEmpty) &&
      state.indentation <= maxNonCodeIndentation && stream.match(hrRE);

    var match = null;
    # 如果缩进增加了4个以上，并且前一行是缩进代码或者前一行是分隔代码结束或者前一行是标题或者前一行是空行
    if (state.indentationDiff >= 4 && (prevLineIsIndentedCode || state.prevLine.fencedCodeEnd ||
         state.prevLine.header || prevLineLineIsEmpty)) {
      # 跳过当前行的所有内容
      stream.skipToEnd();
      # 设置当前行为缩进代码
      state.indentedCode = true;
      return tokenTypes.code;
    } else if (stream.eatSpace()) {
      return null;
    } else if (firstTokenOnLine && state.indentation <= maxNonCodeIndentation && (match = stream.match(atxHeaderRE)) && match[1].length <= 6) {
      # 设置引用的级别和标题的级别
      state.quote = 0;
      state.header = match[1].length;
      state.thisLine.header = true;
      if (modeCfg.highlightFormatting) state.formatting = "header";
      state.f = state.inline;
      return getType(state);
    } else if (state.indentation <= maxNonCodeIndentation && stream.eat('>')) {
      # 如果当前行以>开头，设置引用的级别
      state.quote = firstTokenOnLine ? 1 : state.quote + 1;
      if (modeCfg.highlightFormatting) state.formatting = "quote";
      stream.eatSpace();
      return getType(state);
    } else if (!isHr && !state.setext && firstTokenOnLine && state.indentation <= maxNonCodeIndentation && (match = stream.match(listRE))) {
      # 匹配列表的类型
      var listType = match[1] ? "ol" : "ul";

      # 设置缩进和列表的状态
      state.indentation = lineIndentation + stream.current().length;
      state.list = true;
      state.quote = 0;

      # 将此列表项的内容缩进添加到堆栈中
      state.listStack.push(state.indentation);
      # 重置不应传播到列表项的内联样式
      state.em = false;
      state.strong = false;
      state.code = false;
      state.strikethrough = false;

      # 如果启用了任务列表，并且匹配到任务列表
      if (modeCfg.taskLists && stream.match(taskListRE, false)) {
        state.taskList = true;
      }
      state.f = state.inline;
      if (modeCfg.highlightFormatting) state.formatting = ["list", "list-" + listType];
      return getType(state);
    // 如果当前行的第一个标记并且缩进小于等于最大非代码缩进，并且匹配到了 fencedCodeRE
    } else if (firstTokenOnLine && state.indentation <= maxNonCodeIndentation && (match = stream.match(fencedCodeRE, true))) {
      // 重置引用状态
      state.quote = 0;
      // 设置 fencedEndRE 为匹配到的字符串结尾的正则表达式
      state.fencedEndRE = new RegExp(match[1] + "+ *$");
      // 尝试切换模式
      state.localMode = modeCfg.fencedCodeBlockHighlighting && getMode(match[2] || modeCfg.fencedCodeBlockDefaultMode );
      // 如果存在本地模式，则设置本地状态为本地模式的起始状态
      if (state.localMode) state.localState = CodeMirror.startState(state.localMode);
      // 设置 f 和 block 为 local
      state.f = state.block = local;
      // 如果 highlightFormatting 为真，则设置 formatting 为 "code-block"
      if (modeCfg.highlightFormatting) state.formatting = "code-block";
      // 设置 code 为 -1
      state.code = -1
      // 返回状态类型
      return getType(state);
    // SETEXT 在 HR 之后具有最低的块范围优先级，因此在其他情况（code, blockquote, list...）之后检查它
    } else if (
      // 如果 setext 已设置，表示---/===后的行
      state.setext || (
        // ---/===前的行
        (!allowsInlineContinuation || !prevLineIsList) && !state.quote && state.list === false &&
        !state.code && !isHr && !linkDefRE.test(stream.string) &&
        (match = stream.lookAhead(1)) && (match = match.match(setextHeaderRE))
      )
    ) {
      // 如果 setext 未设置
      if ( !state.setext ) {
        // 设置 header 为匹配到的字符串的第一个字符是'='则为1，否则为2
        state.header = match[0].charAt(0) == '=' ? 1 : 2;
        // 设置 setext 为 header
        state.setext = state.header;
      } else {
        // 设置 header 为 setext
        state.header = state.setext;
        // 对类型没有影响，所以现在可以重置它
        state.setext = 0;
        // 跳过当前行
        stream.skipToEnd();
        // 如果 highlightFormatting 为真，则设置 formatting 为 "header"
        if (modeCfg.highlightFormatting) state.formatting = "header";
      }
      // 设置当前行为 header
      state.thisLine.header = true;
      // 设置 f 为 inline
      state.f = state.inline;
      // 返回状态类型
      return getType(state);
    } else if (isHr) {
      // 跳过当前行
      stream.skipToEnd();
      // 设置 hr 为 true
      state.hr = true;
      // 设置当前行为 hr
      state.thisLine.hr = true;
      // 返回 tokenTypes.hr
      return tokenTypes.hr;
    } else if (stream.peek() === '[') {
      // 切换到 footnoteLink 模式
      return switchInline(stream, state, footnoteLink);
    }
    // 切换到 inline 模式
    return switchInline(stream, state, state.inline);
  }

  // 处理 HTML 块
  function htmlBlock(stream, state) {
    // 调用 htmlMode 的 token 方法处理 HTML 块
    var style = htmlMode.token(stream, state.htmlState);
  // 如果不缺少 HTML 模式
  if (!htmlModeMissing) {
    // 获取 HTML 模式的内部模式和状态
    var inner = CodeMirror.innerMode(htmlMode, state.htmlState)
    // 如果内部模式是 XML 并且不在标签内，并且在文本中，或者在 Markdown 内部并且当前行包含 ">" 符号
    if ((inner.mode.name == "xml" && inner.state.tagStart === null &&
         (!inner.state.context && inner.state.tokenize.isInText)) ||
        (state.md_inside && stream.current().indexOf(">") > -1)) {
      // 设置状态为内联普通模式
      state.f = inlineNormal;
      // 设置状态为块普通模式
      state.block = blockNormal;
      // 重置 HTML 状态
      state.htmlState = null;
    }
  }
  // 返回样式
  return style;
}

// 本地模式
function local(stream, state) {
  // 获取当前列表的缩进
  var currListInd = state.listStack[state.listStack.length - 1] || 0;
  // 判断是否已经退出列表
  var hasExitedList = state.indentation < currListInd;
  // 计算最大围栏结束缩进
  var maxFencedEndInd = currListInd + 3;
  // 如果存在围栏结束正则表达式，并且缩进小于等于最大围栏结束缩进，并且已经退出列表或者匹配到围栏结束
  if (state.fencedEndRE && state.indentation <= maxFencedEndInd && (hasExitedList || stream.match(state.fencedEndRE))) {
    // 如果配置了高亮格式，则设置格式为 "code-block"
    if (modeCfg.highlightFormatting) state.formatting = "code-block";
    var returnType;
    // 如果没有退出列表，则获取类型
    if (!hasExitedList) returnType = getType(state)
    // 重置本地模式和状态
    state.localMode = state.localState = null;
    // 设置状态为块普通模式
    state.block = blockNormal;
    // 设置状态为内联普通模式
    state.f = inlineNormal;
    // 重置围栏结束正则表达式
    state.fencedEndRE = null;
    // 重置代码计数
    state.code = 0
    // 设置当前行为围栏代码结束
    state.thisLine.fencedCodeEnd = true;
    // 如果已经退出列表，则切换到块模式
    if (hasExitedList) return switchBlock(stream, state, state.block);
    // 返回类型
    return returnType;
  } else if (state.localMode) {
    // 如果存在本地模式，则调用本地模式的 token 方法
    return state.localMode.token(stream, state.localState);
  } else {
    // 跳过到行尾
    stream.skipToEnd();
    // 返回代码类型
    return tokenTypes.code;
  }
}

// 内联
function getType(state) {
  // 初始化样式数组
  var styles = [];
    # 如果正在进行格式化处理
    if (state.formatting):
      # 将格式化类型添加到样式列表中
      styles.push(tokenTypes.formatting);

      # 如果格式化类型是字符串，则转换为数组
      if (typeof state.formatting === "string") state.formatting = [state.formatting];

      # 遍历格式化类型数组
      for (var i = 0; i < state.formatting.length; i++):
        # 将格式化类型和具体类型添加到样式列表中
        styles.push(tokenTypes.formatting + "-" + state.formatting[i]);

        # 如果是标题类型，添加标题级别到样式列表中
        if (state.formatting[i] === "header"):
          styles.push(tokenTypes.formatting + "-" + state.formatting[i] + "-" + state.header);

        # 对于引用类型，根据最大引用嵌套深度添加样式
        if (state.formatting[i] === "quote"):
          if (!modeCfg.maxBlockquoteDepth || modeCfg.maxBlockquoteDepth >= state.quote):
            styles.push(tokenTypes.formatting + "-" + state.formatting[i] + "-" + state.quote);
          else:
            styles.push("error");

    # 如果存在未完成的任务
    if (state.taskOpen):
      styles.push("meta");
      return styles.length ? styles.join(' ') : null;

    # 如果任务已完成
    if (state.taskClosed):
      styles.push("property");
      return styles.length ? styles.join(' ') : null;

    # 如果存在链接地址
    if (state.linkHref):
      styles.push(tokenTypes.linkHref, "url");
    else: # 只对非链接文本应用内联样式
      if (state.strong): styles.push(tokenTypes.strong);
      if (state.em): styles.push(tokenTypes.em);
      if (state.strikethrough): styles.push(tokenTypes.strikethrough);
      if (state.emoji): styles.push(tokenTypes.emoji);
      if (state.linkText): styles.push(tokenTypes.linkText);
      if (state.code): styles.push(tokenTypes.code);
      if (state.image): styles.push(tokenTypes.image);
      if (state.imageAltText): styles.push(tokenTypes.imageAltText, "link");
      if (state.imageMarker): styles.push(tokenTypes.imageMarker);

    # 如果存在标题类型，添加标题级别到样式列表中
    if (state.header): styles.push(tokenTypes.header, tokenTypes.header + "-" + state.header);
    // 如果当前状态为引用状态
    if (state.quote) {
      // 将引用样式添加到样式数组中
      styles.push(tokenTypes.quote);

      // 添加 `quote-#` 样式，其中 `#` 的最大值为 modeCfg.maxBlockquoteDepth
      if (!modeCfg.maxBlockquoteDepth || modeCfg.maxBlockquoteDepth >= state.quote) {
        styles.push(tokenTypes.quote + "-" + state.quote);
      } else {
        styles.push(tokenTypes.quote + "-" + modeCfg.maxBlockquoteDepth);
      }
    }

    // 如果当前状态为列表状态
    if (state.list !== false) {
      // 计算列表模数
      var listMod = (state.listStack.length - 1) % 3;
      // 根据模数添加不同的列表样式
      if (!listMod) {
        styles.push(tokenTypes.list1);
      } else if (listMod === 1) {
        styles.push(tokenTypes.list2);
      } else {
        styles.push(tokenTypes.list3);
      }
    }

    // 如果当前状态为有尾随空格和换行符
    if (state.trailingSpaceNewLine) {
      styles.push("trailing-space-new-line");
    } else if (state.trailingSpace) {
      // 如果有尾随空格，根据空格数量添加不同的样式
      styles.push("trailing-space-" + (state.trailingSpace % 2 ? "a" : "b"));
    }

    // 返回样式数组的字符串表示，如果为空则返回 null
    return styles.length ? styles.join(' ') : null;
  }

  // 处理文本
  function handleText(stream, state) {
    // 如果匹配到文本正则表达式
    if (stream.match(textRE, true)) {
      // 返回当前状态的类型
      return getType(state);
    }
    // 否则返回 undefined
    return undefined;
  }

  // 处理普通内联文本
  function inlineNormal(stream, state) {
    // 调用 state 的 text 方法处理文本，获取样式
    var style = state.text(stream, state);
    // 如果样式不为 undefined，则返回样式
    if (typeof style !== 'undefined')
      return style;

    // 如果当前状态为列表状态
    if (state.list) { // List marker (*, +, -, 1., etc)
      // 重置列表状态，并返回当前状态的类型
      state.list = null;
      return getType(state);
    }

    // 如果当前状态为任务列表状态
    if (state.taskList) {
      // 匹配任务列表正则表达式，判断任务是否打开
      var taskOpen = stream.match(taskListRE, true)[1] === " ";
      if (taskOpen) state.taskOpen = true;
      else state.taskClosed = true;
      // 如果需要突出显示格式，则设置格式为 "task"
      if (modeCfg.highlightFormatting) state.formatting = "task";
      state.taskList = false;
      return getType(state);
    }

    // 重置任务打开和关闭状态
    state.taskOpen = false;
    state.taskClosed = false;

    // 如果当前状态为标题状态，并且匹配到标题的正则表达式
    if (state.header && stream.match(/^#+$/, true)) {
      // 如果需要突出显示格式，则设置格式为 "header"
      if (modeCfg.highlightFormatting) state.formatting = "header";
      return getType(state);
    }

    // 获取下一个字符
    var ch = stream.next();

    // 匹配下一行上的链接标题
    // 如果状态中存在链接标题
    if (state.linkTitle) {
      // 将链接标题状态设置为false
      state.linkTitle = false;
      // 记录当前字符到matchCh
      var matchCh = ch;
      // 如果当前字符是'('，则将matchCh设置为')'
      if (ch === '(') {
        matchCh = ')';
      }
      // 将matchCh中的特殊字符转义
      matchCh = (matchCh+'').replace(/([.?*+^\[\]\\(){}|-])/g, "\\$1");
      // 构建正则表达式，用于匹配链接地址
      var regex = '^\\s*(?:[^' + matchCh + '\\\\]+|\\\\\\\\|\\\\.)' + matchCh;
      // 如果匹配成功，则返回链接地址类型
      if (stream.match(new RegExp(regex), true)) {
        return tokenTypes.linkHref;
      }
    }

    // 如果代码块发生变化，可能需要在GFM模式下更新
    if (ch === '`') {
      // 保存之前的格式化状态
      var previousFormatting = state.formatting;
      // 如果启用了高亮格式化，则将格式化状态设置为"code"
      if (modeCfg.highlightFormatting) state.formatting = "code";
      // 吃掉连续的反引号
      stream.eatWhile('`');
      // 获取当前连续反引号的数量
      var count = stream.current().length
      // 如果当前不在代码块中且不在引用块中，且连续反引号数量为1，则将状态设置为代码块数量，返回相应类型
      if (state.code == 0 && (!state.quote || count == 1)) {
        state.code = count
        return getType(state)
      } else if (count == state.code) { // 必须完全匹配
        var t = getType(state)
        state.code = 0
        return t
      } else {
        // 恢复之前的格式化状态，返回相应类型
        state.formatting = previousFormatting
        return getType(state)
      }
    } else if (state.code) {
      // 如果在代码块中，则返回相应类型
      return getType(state);
    }

    // 如果当前字符是反斜杠
    if (ch === '\\') {
      // 吃掉下一个字符
      stream.next();
      // 如果启用了高亮格式化
      if (modeCfg.highlightFormatting) {
        // 获取当前类型
        var type = getType(state);
        // 设置格式化转义类型
        var formattingEscape = tokenTypes.formatting + "-escape";
        // 如果存在类型，则返回类型和格式化转义类型，否则返回格式化转义类型
        return type ? type + " " + formattingEscape : formattingEscape;
      }
    }

    // 如果当前字符是'!'且下一个字符匹配图片链接的正则表达式
    if (ch === '!' && stream.match(/\[[^\]]*\] ?(?:\(|\[)/, false)) {
      // 设置图片标记为true，设置图片状态为true
      state.imageMarker = true;
      state.image = true;
      // 如果启用了高亮格式化，则将格式化状态设置为"image"
      if (modeCfg.highlightFormatting) state.formatting = "image";
      // 返回相应类型
      return getType(state);
    }

    // 如果当前字符是'['且图片标记为true且下一个字符匹配图片链接的正则表达式
    if (ch === '[' && state.imageMarker && stream.match(/[^\]]*\](\(.*?\)| ?\[.*?\])/, false)) {
      // 将图片标记设置为false，将图片alt文本状态设置为true
      state.imageMarker = false;
      state.imageAltText = true
      // 如果启用了高亮格式化，则将格式化状态设置为"image"
      if (modeCfg.highlightFormatting) state.formatting = "image";
      // 返回相应类型
      return getType(state);
    }
    # 如果当前字符是 ']' 并且状态中存在图片的替代文本
    if (ch === ']' && state.imageAltText) {
      # 如果启用了高亮格式设置，将状态的格式设置为 "image"
      if (modeCfg.highlightFormatting) state.formatting = "image";
      # 获取类型
      var type = getType(state);
      # 重置状态
      state.imageAltText = false;
      state.image = false;
      state.inline = state.f = linkHref;
      # 返回类型
      return type;
    }

    # 如果当前字符是 '[' 并且状态中不存在图片
    if (ch === '[' && !state.image) {
      # 如果存在链接文本并且匹配到链接文本
      if (state.linkText && stream.match(/^.*?\]/)) return getType(state)
      # 设置状态中存在链接文本
      state.linkText = true;
      # 如果启用了高亮格式设置，将状态的格式设置为 "link"
      if (modeCfg.highlightFormatting) state.formatting = "link";
      # 返回类型
      return getType(state);
    }

    # 如果当前字符是 ']' 并且状态中存在链接文本
    if (ch === ']' && state.linkText) {
      # 如果启用了高亮格式设置，将状态的格式设置为 "link"
      if (modeCfg.highlightFormatting) state.formatting = "link";
      # 获取类型
      var type = getType(state);
      # 重置状态
      state.linkText = false;
      state.inline = state.f = stream.match(/\(.*?\)| ?\[.*?\]/, false) ? linkHref : inlineNormal
      # 返回类型
      return type;
    }

    # 如果当前字符是 '<' 并且匹配到链接地址
    if (ch === '<' && stream.match(/^(https?|ftps?):\/\/(?:[^\\>]|\\.)+>/, false)) {
      state.f = state.inline = linkInline;
      # 如果启用了高亮格式设置，将状态的格式设置为 "link"
      if (modeCfg.highlightFormatting) state.formatting = "link";
      # 获取类型
      var type = getType(state);
      # 如果类型存在，则在后面添加空格
      if (type){
        type += " ";
      } else {
        type = "";
      }
      # 返回类型和链接内联的 token 类型
      return type + tokenTypes.linkInline;
    }

    # 如果当前字符是 '<' 并且匹配到电子邮件地址
    if (ch === '<' && stream.match(/^[^> \\]+@(?:[^\\>]|\\.)+>/, false)) {
      state.f = state.inline = linkInline;
      # 如果启用了高亮格式设置，将状态的格式设置为 "link"
      if (modeCfg.highlightFormatting) state.formatting = "link";
      # 获取类型
      var type = getType(state);
      # 如果类型存在，则在后面添加空格
      if (type){
        type += " ";
      } else {
        type = "";
      }
      # 返回类型和链接电子邮件的 token 类型
      return type + tokenTypes.linkEmail;
    }
    # 如果配置为 XML 模式，并且当前字符为 '<'，并且下一个内容匹配 XML 标签的正则表达式
    if (modeCfg.xml && ch === '<' && stream.match(/^(!--|\?|!\[CDATA\[|[a-z][a-z0-9-]*(?:\s+[a-z_:.\-]+(?:\s*=\s*[^>]+)?)*\s*(?:>|$))/i, false)) {
      # 获取当前标签的结束位置
      var end = stream.string.indexOf(">", stream.pos);
      # 如果找到结束位置
      if (end != -1) {
        # 获取标签的属性部分
        var atts = stream.string.substring(stream.start, end);
        # 如果属性中包含 markdown=1，则设置状态为 md_inside
        if (/markdown\s*=\s*('|"){0,1}1('|"){0,1}/.test(atts)) state.md_inside = true;
      }
      # 回退一个字符
      stream.backUp(1);
      # 设置 HTML 状态为起始状态
      state.htmlState = CodeMirror.startState(htmlMode);
      # 返回切换到 HTML 模式的结果
      return switchBlock(stream, state, htmlBlock);
    }

    # 如果配置为 XML 模式，并且当前字符为 '<'，并且下一个内容匹配闭合标签的正则表达式
    if (modeCfg.xml && ch === '<' && stream.match(/^\/\w*?>/)) {
      # 设置状态为非 md_inside
      state.md_inside = false;
      # 返回标签类型为 "tag"
      return "tag";
    } else if (ch === "*" || ch === "_") {
      // 如果当前字符是 * 或者 _，则执行以下操作
      var len = 1, before = stream.pos == 1 ? " " : stream.string.charAt(stream.pos - 2)
      // 初始化 len 为 1，before 为当前字符前一个字符，如果当前位置在第一个字符，则 before 为一个空格
      while (len < 3 && stream.eat(ch)) len++
      // 当 len 小于 3 且当前字符为 * 或者 _ 时，循环读取字符并增加 len 的值
      var after = stream.peek() || " "
      // 获取当前字符后一个字符，如果不存在则为一个空格
      // See http://spec.commonmark.org/0.27/#emphasis-and-strong-emphasis
      // 参考链接，关于强调和加粗的规范
      var leftFlanking = !/\s/.test(after) && (!punctuation.test(after) || /\s/.test(before) || punctuation.test(before))
      var rightFlanking = !/\s/.test(before) && (!punctuation.test(before) || /\s/.test(after) || punctuation.test(after))
      // 判断左右是否为边界字符
      var setEm = null, setStrong = null
      // 初始化 setEm 和 setStrong 为 null
      if (len % 2) { // Em
        // 如果 len 为奇数，表示强调
        if (!state.em && leftFlanking && (ch === "*" || !rightFlanking || punctuation.test(before)))
          setEm = true
        else if (state.em == ch && rightFlanking && (ch === "*" || !leftFlanking || punctuation.test(after)))
          setEm = false
      }
      if (len > 1) { // Strong
        // 如果 len 大于 1，表示加粗
        if (!state.strong && leftFlanking && (ch === "*" || !rightFlanking || punctuation.test(before)))
          setStrong = true
        else if (state.strong == ch && rightFlanking && (ch === "*" || !leftFlanking || punctuation.test(after)))
          setStrong = false
      }
      // 设置强调和加粗的状态
      if (setStrong != null || setEm != null) {
        // 如果设置了强调或者加粗
        if (modeCfg.highlightFormatting) state.formatting = setEm == null ? "strong" : setStrong == null ? "em" : "strong em"
        if (setEm === true) state.em = ch
        if (setStrong === true) state.strong = ch
        var t = getType(state)
        if (setEm === false) state.em = false
        if (setStrong === false) state.strong = false
        return t
      }
    } else if (ch === ' ') {
      // 如果当前字符是空格
      if (stream.eat('*') || stream.eat('_')) { // Probably surrounded by spaces
        // 如果下一个字符是 * 或者 _
        if (stream.peek() === ' ') { // Surrounded by spaces, ignore
          // 如果下一个字符是空格，则忽略
          return getType(state);
        } else { // Not surrounded by spaces, back up pointer
          // 如果下一个字符不是空格，则回退指针
          stream.backUp(1);
        }
      }
    }
    // 如果 modeCfg.strikethrough 为真，则执行以下代码块
    if (modeCfg.strikethrough) {
      // 如果当前字符为'~'，并且后面的字符都是'~'，则执行以下代码块
      if (ch === '~' && stream.eatWhile(ch)) {
        // 如果当前状态为strikethrough，则移除strikethrough
        if (state.strikethrough) {
          // 如果 modeCfg.highlightFormatting 为真，则将当前格式设置为"strikethrough"
          if (modeCfg.highlightFormatting) state.formatting = "strikethrough";
          // 获取当前类型并返回
          var t = getType(state);
          state.strikethrough = false;
          return t;
        } else if (stream.match(/^[^\s]/, false)) {
          // 如果当前字符不为空白，则添加strikethrough
          state.strikethrough = true;
          // 如果 modeCfg.highlightFormatting 为真，则将当前格式设置为"strikethrough"
          if (modeCfg.highlightFormatting) state.formatting = "strikethrough";
          // 获取当前类型并返回
          return getType(state);
        }
      } else if (ch === ' ') {
        if (stream.match(/^~~/, true)) {
          // 如果前后都是'~'，则返回当前类型
          if (stream.peek() === ' ') {
            return getType(state);
          } else {
            // 如果前后不都是空格，则回退指针
            stream.backUp(2);
          }
        }
      }
    }

    // 如果 modeCfg.emoji 为真，并且当前字符为':'，并且满足指定的正则表达式，则执行以下代码块
    if (modeCfg.emoji && ch === ":" && stream.match(/^(?:[a-z_\d+][a-z_\d+-]*|\-[a-z_\d+][a-z_\d+-]*):/)) {
      state.emoji = true;
      // 如果 modeCfg.highlightFormatting 为真，则将当前格式设置为"emoji"
      if (modeCfg.highlightFormatting) state.formatting = "emoji";
      // 获取当前类型并返回
      var retType = getType(state);
      state.emoji = false;
      return retType;
    }

    // 如果当前字符为空格，则执行以下代码块
    if (ch === ' ') {
      if (stream.match(/^ +$/, false)) {
        // 如果后面都是空格，则增加尾随空格计数
        state.trailingSpace++;
      } else if (state.trailingSpace) {
        // 如果之前有尾随空格，则设置尾随空格新行为真
        state.trailingSpaceNewLine = true;
      }
    }

    // 获取当前类型并返回
    return getType(state);
  }

  // 内联链接处理函数
  function linkInline(stream, state) {
    // 获取下一个字符
    var ch = stream.next();

    // 如果当前字符为">"，则执行以下代码块
    if (ch === ">") {
      state.f = state.inline = inlineNormal;
      // 如果 modeCfg.highlightFormatting 为真，则将当前格式设置为"link"
      if (modeCfg.highlightFormatting) state.formatting = "link";
      // 获取当前类型并返回
      var type = getType(state);
      if (type){
        type += " ";
      } else {
        type = "";
      }
      return type + tokenTypes.linkInline;
    }

    // 匹配非">"的字符
    stream.match(/^[^>]+/, true);

    // 返回内联链接类型
    return tokenTypes.linkInline;
  }

  // 链接地址处理函数
  function linkHref(stream, state) {
    // 检查是否为空格，如果是则返回NULL（避免标记空格）
    if(stream.eatSpace()){
      return null;
  }
  // 获取下一个字符
  var ch = stream.next();
  // 如果下一个字符是 '(' 或者 '['
  if (ch === '(' || ch === '[') {
    // 设置状态为获取链接内部的内容
    state.f = state.inline = getLinkHrefInside(ch === "(" ? ")" : "]");
    // 如果需要突出显示格式，则设置格式为 "link-string"
    if (modeCfg.highlightFormatting) state.formatting = "link-string";
    // 设置链接地址标记为 true
    state.linkHref = true;
    // 返回状态类型
    return getType(state);
  }
  // 返回错误类型
  return 'error';
}

// 链接正则表达式
var linkRE = {
  ")": /^(?:[^\\\(\)]|\\.|\((?:[^\\\(\)]|\\.)*\))*?(?=\))/,
  "]": /^(?:[^\\\[\]]|\\.|\[(?:[^\\\[\]]|\\.)*\])*?(?=\])/
}

// 获取链接内部的内容
function getLinkHrefInside(endChar) {
  return function(stream, state) {
    // 获取下一个字符
    var ch = stream.next();

    // 如果下一个字符是结束字符
    if (ch === endChar) {
      // 设置状态为内联正常状态
      state.f = state.inline = inlineNormal;
      // 如果需要突出显示格式，则设置格式为 "link-string"
      if (modeCfg.highlightFormatting) state.formatting = "link-string";
      // 获取当前状态类型
      var returnState = getType(state);
      // 设置链接地址标记为 false
      state.linkHref = false;
      // 返回状态类型
      return returnState;
    }

    // 匹配链接正则表达式
    stream.match(linkRE[endChar])
    // 设置链接地址标记为 true
    state.linkHref = true;
    // 返回状态类型
    return getType(state);
  };
}

// 脚注链接
function footnoteLink(stream, state) {
  // 如果匹配到脚注链接
  if (stream.match(/^([^\]\\]|\\.)*\]:/, false)) {
    // 设置状态为脚注链接内部
    state.f = footnoteLinkInside;
    // 消耗掉 [
    stream.next();
    // 如果需要突出显示格式，则设置格式为 "link"
    if (modeCfg.highlightFormatting) state.formatting = "link";
    // 设置链接文本标记为 true
    state.linkText = true;
    // 返回状态类型
    return getType(state);
  }
  // 切换到内联正常状态
  return switchInline(stream, state, inlineNormal);
}

// 脚注链接内部
function footnoteLinkInside(stream, state) {
  // 如果匹配到脚注链接结束
  if (stream.match(/^\]:/, true)) {
    // 设置状态为内联脚注 URL
    state.f = state.inline = footnoteUrl;
    // 如果需要突出显示格式，则设置格式为 "link"
    if (modeCfg.highlightFormatting) state.formatting = "link";
    // 获取当前状态类型
    var returnType = getType(state);
    // 设置链接文本标记为 false
    state.linkText = false;
    // 返回状态类型
    return returnType;
  }

  // 匹配非转义字符或转义字符
  stream.match(/^([^\]\\]|\\.)+/, true);

  // 返回链接文本类型
  return tokenTypes.linkText;
}

// 脚注链接 URL
function footnoteUrl(stream, state) {
  // 检查是否是空格，如果是则返回空
  if(stream.eatSpace()){
    return null;
  }
  // 匹配 URL
  stream.match(/^[^\s]+/, true);
  // 检查链接标题
    if (stream.peek() === undefined) { // 如果流的下一个字符是未定义的，表示行末尾，设置标志以检查下一行
      state.linkTitle = true;
    } else { // 如果行上还有内容，检查是否是链接标题
      stream.match(/^(?:\s+(?:"(?:[^"\\]|\\\\|\\.)+"|'(?:[^'\\]|\\\\|\\.)+'|\((?:[^)\\]|\\\\|\\.)+\)))?/, true);
    }
    state.f = state.inline = inlineNormal; // 设置状态为内联普通状态
    return tokenTypes.linkHref + " url"; // 返回链接的类型和 URL
  }

  var mode = {
    startState: function() { // 定义编辑器的起始状态
      return {
        f: blockNormal, // 设置默认块状态为普通块

        prevLine: {stream: null}, // 上一行的流
        thisLine: {stream: null}, // 当前行的流

        block: blockNormal, // 块状态为普通块
        htmlState: null, // HTML 状态为空
        indentation: 0, // 缩进为 0

        inline: inlineNormal, // 内联状态为普通内联
        text: handleText, // 文本处理函数

        formatting: false, // 格式化标志为假
        linkText: false, // 链接文本标志为假
        linkHref: false, // 链接地址标志为假
        linkTitle: false, // 链接标题标志为假
        code: 0, // 代码标志为 0
        em: false, // 强调标志为假
        strong: false, // 加粗标志为假
        header: 0, // 标题级别为 0
        setext: 0, // Setext 标题为 0
        hr: false, // 水平线标志为假
        taskList: false, // 任务列表标志为假
        list: false, // 列表标志为假
        listStack: [], // 列表堆栈为空
        quote: 0, // 引用级别为 0
        trailingSpace: 0, // 尾随空格为 0
        trailingSpaceNewLine: false, // 尾随空格换行标志为假
        strikethrough: false, // 删除线标志为假
        emoji: false, // 表情标志为假
        fencedEndRE: null // 围栏结束正则为空
      };
    },
    // 复制给定状态对象的属性，创建并返回一个新的状态对象
    copyState: function(s) {
      return {
        // 复制当前行的标记
        f: s.f,
        // 复制上一行的内容
        prevLine: s.prevLine,
        // 复制当前行的内容
        thisLine: s.thisLine,
        // 复制当前块的标记
        block: s.block,
        // 复制 HTML 模式的状态
        htmlState: s.htmlState && CodeMirror.copyState(htmlMode, s.htmlState),
        // 复制缩进
        indentation: s.indentation,
        // 复制本地模式
        localMode: s.localMode,
        // 复制本地模式的状态
        localState: s.localMode ? CodeMirror.copyState(s.localMode, s.localState) : null,
        // 复制内联标记
        inline: s.inline,
        // 复制文本
        text: s.text,
        // 格式化标记
        formatting: false,
        // 复制链接文本
        linkText: s.linkText,
        // 复制链接标题
        linkTitle: s.linkTitle,
        // 复制链接地址
        linkHref: s.linkHref,
        // 复制代码标记
        code: s.code,
        // 复制强调标记
        em: s.em,
        // 复制加粗标记
        strong: s.strong,
        // 复制删除线标记
        strikethrough: s.strikethrough,
        // 复制表情标记
        emoji: s.emoji,
        // 复制标题标记
        header: s.header,
        // 复制 setext 标记
        setext: s.setext,
        // 复制水平线标记
        hr: s.hr,
        // 复制任务列表标记
        taskList: s.taskList,
        // 复制列表标记
        list: s.list,
        // 复制列表堆栈
        listStack: s.listStack.slice(0),
        // 复制引用标记
        quote: s.quote,
        // 复制缩进代码标记
        indentedCode: s.indentedCode,
        // 复制尾随空格标记
        trailingSpace: s.trailingSpace,
        // 复制尾随空格换行标记
        trailingSpaceNewLine: s.trailingSpaceNewLine,
        // 复制 md_inside 标记
        md_inside: s.md_inside,
        // 复制 fencedEndRE 标记
        fencedEndRE: s.fencedEndRE
      };
    },
    // 定义 token 方法，用于处理代码中的标记
    token: function(stream, state) {

      // 重置 state.formatting 为 false
      state.formatting = false;

      // 如果当前流不等于 state.thisLine.stream
      if (stream != state.thisLine.stream) {
        // 重置 state.header 为 0
        state.header = 0;
        // 重置 state.hr 为 false
        state.hr = false;

        // 如果流匹配空白行
        if (stream.match(/^\s*$/, true)) {
          // 调用 blankLine 方法处理空白行
          blankLine(state);
          // 返回 null
          return null;
        }

        // 保存当前行到 state.prevLine
        state.prevLine = state.thisLine
        // 设置当前行为 state.thisLine
        state.thisLine = {stream: stream}

        // 重置 state.taskList 为 false
        state.taskList = false;

        // 重置 state.trailingSpace 为 0
        state.trailingSpace = 0;
        // 重置 state.trailingSpaceNewLine 为 false
        state.trailingSpaceNewLine = false;

        // 如果没有 localState
        if (!state.localState) {
          // 设置 state.f 为 state.block
          state.f = state.block;
          // 如果 state.f 不等于 htmlBlock
          if (state.f != htmlBlock) {
            // 匹配行首空白，计算缩进
            var indentation = stream.match(/^\s*/, true)[0].replace(/\t/g, expandedTab).length;
            // 设置 state.indentation 为缩进值
            state.indentation = indentation;
            // 设置 state.indentationDiff 为 null
            state.indentationDiff = null;
            // 如果缩进值大于 0，返回 null
            if (indentation > 0) return null;
          }
        }
      }
      // 调用 state.f 方法处理流和状态
      return state.f(stream, state);
    },

    // 定义 innerMode 方法，用于处理内部模式
    innerMode: function(state) {
      // 如果 state.block 等于 htmlBlock，返回 state.htmlState 和 htmlMode
      if (state.block == htmlBlock) return {state: state.htmlState, mode: htmlMode};
      // 如果有 localState，返回 state.localState 和 state.localMode
      if (state.localState) return {state: state.localState, mode: state.localMode};
      // 否则返回 state 和 mode
      return {state: state, mode: mode};
    },

    // 定义 indent 方法，用于处理缩进
    indent: function(state, textAfter, line) {
      // 如果 state.block 等于 htmlBlock 并且 htmlMode.indent 存在，调用 htmlMode.indent 处理缩进
      if (state.block == htmlBlock && htmlMode.indent) return htmlMode.indent(state.htmlState, textAfter, line)
      // 如果有 localState 并且 state.localMode.indent 存在，调用 state.localMode.indent 处理缩进
      if (state.localState && state.localMode.indent) return state.localMode.indent(state.localState, textAfter, line)
      // 否则返回 CodeMirror.Pass
      return CodeMirror.Pass
    },

    // 设置 blankLine 方法为之前定义的 blankLine 方法
    blankLine: blankLine,

    // 设置 getType 方法为之前定义的 getType 方法
    getType: getType,

    // 设置 blockCommentStart 为 "<!--"
    blockCommentStart: "<!--",
    // 设置 blockCommentEnd 为 "-->"
    blockCommentEnd: "-->",
    // 设置 closeBrackets 为 "()[]{}''\"\"``"
    closeBrackets: "()[]{}''\"\"``",
    // 设置 fold 为 "markdown"
    fold: "markdown"
  };
  // 返回 mode
  return mode;
# 定义 MIME 类型为 "xml" 的语法高亮规则
CodeMirror.defineMIME("text/xml", "xml");

# 定义 MIME 类型为 "markdown" 的语法高亮规则
CodeMirror.defineMIME("text/markdown", "markdown");

# 定义 MIME 类型为 "text/x-markdown" 的语法高亮规则
CodeMirror.defineMIME("text/x-markdown", "markdown");
```