# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\edit\closetag.js`

```py
// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

/**
 * CodeMirror 的标签闭合扩展。
 *
 * 此扩展添加了一个 "autoCloseTags" 选项，可以设置为 true 以获取默认行为，或者设置为对象以进一步配置其行为。
 *
 * 支持的选项包括：
 *
 * `whenClosing`（默认为 true）
 *   当输入关闭标签的 '/' 时是否自动闭合。
 * `whenOpening`（默认为 true）
 *   当输入开放标签的最后一个 '>' 时是否自动闭合标签。
 * `dontCloseTags`（默认为空标签对于 HTML，对于 XML 为无）
 *   一个不应该自动闭合的标签名称数组。
 * `indentTags`（默认为 HTML 的块标签，对于 XML 为无）
 *   一个应该在打开时导致在标签内添加空行，并且缩进空行和闭合行的标签名称数组。
 * `emptyTags`（默认为无）
 *   一个应该使用 '/>' 自动闭合的 XML 标签名称数组。
 *
 * 请参阅 demos/closetag.html 以获取使用示例。
 */

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"), require("../fold/xml-fold"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror", "../fold/xml-fold"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  CodeMirror.defineOption("autoCloseTags", false, function(cm, val, old) {
    if (old != CodeMirror.Init && old)
      cm.removeKeyMap("autoCloseTags");
    if (!val) return;
    var map = {name: "autoCloseTags"};
    if (typeof val != "object" || val.whenClosing !== false)
      map["'/'"] = function(cm) { return autoCloseSlash(cm); };
    if (typeof val != "object" || val.whenOpening !== false)
      map["'>'"] = function(cm) { return autoCloseGT(cm); };
  # 将键盘映射添加到代码编辑器中
  cm.addKeyMap(map);
  });

  # 不需要自动闭合的 HTML 标签
  var htmlDontClose = ["area", "base", "br", "col", "command", "embed", "hr", "img", "input", "keygen", "link", "meta", "param",
                       "source", "track", "wbr"];
  # 需要缩进的 HTML 标签
  var htmlIndent = ["applet", "blockquote", "body", "button", "div", "dl", "fieldset", "form", "frameset", "h1", "h2", "h3", "h4",
                    "h5", "h6", "head", "html", "iframe", "layer", "legend", "object", "ol", "p", "select", "table", "ul"];

  # 自动闭合 HTML 标签的函数
  function autoCloseGT(cm) {
    # 如果禁用输入，则返回
    if (cm.getOption("disableInput")) return CodeMirror.Pass;
    # 获取选中的文本范围和替换内容
    var ranges = cm.listSelections(), replacements = [];
    # 获取自动闭合标签的选项
    var opt = cm.getOption("autoCloseTags");
    # 遍历ranges数组
    for (var i = 0; i < ranges.length; i++) {
      # 如果当前range不为空，则返回CodeMirror.Pass
      if (!ranges[i].empty()) return CodeMirror.Pass;
      # 获取当前range的头部位置和token
      var pos = ranges[i].head, tok = cm.getTokenAt(pos);
      # 获取当前token所在的内部模式和状态
      var inner = CodeMirror.innerMode(cm.getMode(), tok.state), state = inner.state;
      # 获取当前XML标签的信息
      var tagInfo = inner.mode.xmlCurrentTag && inner.mode.xmlCurrentTag(state)
      # 获取当前XML标签的名称
      var tagName = tagInfo && tagInfo.name
      # 如果没有标签名称，则返回CodeMirror.Pass
      if (!tagName) return CodeMirror.Pass

      # 判断当前模式是否为HTML
      var html = inner.mode.configuration == "html";
      # 获取不需要关闭的标签列表
      var dontCloseTags = (typeof opt == "object" && opt.dontCloseTags) || (html && htmlDontClose);
      # 获取需要缩进的标签列表
      var indentTags = (typeof opt == "object" && opt.indentTags) || (html && htmlIndent);

      # 如果token的结束位置大于当前位置的字符数，则修正tagName的长度
      if (tok.end > pos.ch) tagName = tagName.slice(0, tagName.length - tok.end + pos.ch);
      # 将tagName转换为小写
      var lowerTagName = tagName.toLowerCase();
      # 不处理结束标签或自闭合标签末尾的'>'
      if (!tagName ||
          tok.type == "string" && (tok.end != pos.ch || !/[\"\']/.test(tok.string.charAt(tok.string.length - 1)) || tok.string.length == 1) ||
          tok.type == "tag" && tagInfo.close ||
          tok.string.indexOf("/") == (pos.ch - tok.start - 1) || // match something like <someTagName />
          dontCloseTags && indexOf(dontCloseTags, lowerTagName) > -1 ||
          closingTagExists(cm, inner.mode.xmlCurrentContext && inner.mode.xmlCurrentContext(state) || [], tagName, pos, true))
        return CodeMirror.Pass;

      # 获取空标签列表
      var emptyTags = typeof opt == "object" && opt.emptyTags;
      # 如果当前标签在空标签列表中，则设置替换文本和新位置
      if (emptyTags && indexOf(emptyTags, tagName) > -1) {
        replacements[i] = { text: "/>", newPos: CodeMirror.Pos(pos.line, pos.ch + 2) };
        continue;
      }

      # 判断是否需要缩进
      var indent = indentTags && indexOf(indentTags, lowerTagName) > -1;
      # 设置替换文本和新位置
      replacements[i] = {indent: indent,
                         text: ">" + (indent ? "\n\n" : "") + "</" + tagName + ">",
                         newPos: indent ? CodeMirror.Pos(pos.line + 1, 0) : CodeMirror.Pos(pos.line, pos.ch + 1)};
    }
    # 检查是否存在 opt 对象并且 opt.dontIndentOnAutoClose 为真
    var dontIndentOnAutoClose = (typeof opt == "object" && opt.dontIndentOnAutoClose);
    # 从后向前遍历 ranges 数组
    for (var i = ranges.length - 1; i >= 0; i--) {
      # 获取替换信息
      var info = replacements[i];
      # 替换指定范围的文本
      cm.replaceRange(info.text, ranges[i].head, ranges[i].anchor, "+insert");
      # 复制当前选择
      var sel = cm.listSelections().slice(0);
      # 更新选择范围
      sel[i] = {head: info.newPos, anchor: info.newPos};
      cm.setSelections(sel);
      # 如果不是在自动关闭标签时不缩进，并且存在缩进信息
      if (!dontIndentOnAutoClose && info.indent) {
        # 缩进当前行
        cm.indentLine(info.newPos.line, null, true);
        # 缩进下一行
        cm.indentLine(info.newPos.line + 1, null, true);
      }
    }
  }

  # 自动关闭当前标签
  function autoCloseCurrent(cm, typingSlash) {
    # 获取当前选择范围
    var ranges = cm.listSelections(), replacements = [];
    # 设置头部文本
    var head = typingSlash ? "/" : "</";
    # 获取自动关闭标签的选项
    var opt = cm.getOption("autoCloseTags");
    # 检查是否存在 opt 对象并且 opt.dontIndentOnSlash 为真
    var dontIndentOnAutoClose = (typeof opt == "object" && opt.dontIndentOnSlash);
    // 遍历ranges数组，对每个元素执行以下操作
    for (var i = 0; i < ranges.length; i++) {
        // 如果当前range不为空，则返回CodeMirror.Pass
        if (!ranges[i].empty()) return CodeMirror.Pass;
        // 获取当前range的头部位置和对应的token
        var pos = ranges[i].head, tok = cm.getTokenAt(pos);
        // 获取当前token所在的inner mode和对应的state
        var inner = CodeMirror.innerMode(cm.getMode(), tok.state), state = inner.state;
        // 如果正在输入斜杠并且当前token是字符串，或者token的第一个字符不是"<"，或者token的起始位置不是pos.ch - 1，则返回CodeMirror.Pass
        if (typingSlash && (tok.type == "string" || tok.string.charAt(0) != "<" ||
                            tok.start != pos.ch - 1))
            return CodeMirror.Pass;
        // 用于解决在htmlmixed模式下在JS/CSS片段中自动补全时不在XML模式下的问题
        var replacement, mixed = inner.mode.name != "xml" && cm.getMode().name == "htmlmixed"
        // 如果mixed为true并且inner mode的名称为javascript
        if (mixed && inner.mode.name == "javascript") {
            replacement = head + "script";
        } else if (mixed && inner.mode.name == "css") {
            replacement = head + "style";
        } else {
            // 获取当前inner mode的xmlCurrentContext并执行对应的操作
            var context = inner.mode.xmlCurrentContext && inner.mode.xmlCurrentContext(state)
            // 如果context不存在或者context长度大于0并且在当前位置存在closing tag，则返回CodeMirror.Pass
            if (!context || (context.length && closingTagExists(cm, context, context[context.length - 1], pos)))
                return CodeMirror.Pass;
            // 根据context的最后一个元素构建replacement
            replacement = head + context[context.length - 1]
        }
        // 如果当前位置的行不是">"，则在replacement后面添加">"
        if (cm.getLine(pos.line).charAt(tok.end) != ">") replacement += ">";
        // 将replacement添加到replacements数组中
        replacements[i] = replacement;
    }
    // 用replacements数组替换当前选择的文本
    cm.replaceSelections(replacements);
    // 更新ranges数组
    ranges = cm.listSelections();
    // 如果dontIndentOnAutoClose为false，则对每个range执行以下操作
    if (!dontIndentOnAutoClose) {
        for (var i = 0; i < ranges.length; i++)
            if (i == ranges.length - 1 || ranges[i].head.line < ranges[i + 1].head.line)
                cm.indentLine(ranges[i].head.line);
    }
  }

  // 自动关闭斜杠
  function autoCloseSlash(cm) {
    // 如果disableInput选项为true，则返回CodeMirror.Pass
    if (cm.getOption("disableInput")) return CodeMirror.Pass;
    // 调用autoCloseCurrent函数并传入true作为参数
    return autoCloseCurrent(cm, true);
  }

  // 将closeTag命令绑定到autoCloseCurrent函数
  CodeMirror.commands.closeTag = function(cm) { return autoCloseCurrent(cm); };

  // 查找collection中elt的索引
  function indexOf(collection, elt) {
    // 如果集合对象有 indexOf 方法，则使用该方法查找元素的索引
    if (collection.indexOf) return collection.indexOf(elt);
    // 遍历集合对象，查找元素的索引
    for (var i = 0, e = collection.length; i < e; ++i)
      if (collection[i] == elt) return i;
    // 如果未找到元素，则返回 -1
    return -1;
    }
    
    // 如果加载了 xml-fold 模块，使用其功能尝试验证给定标签是否未闭合
    function closingTagExists(cm, context, tagName, pos, newTag) {
      // 如果未加载 CodeMirror.scanForClosingTag 模块，则返回 false
      if (!CodeMirror.scanForClosingTag) return false;
      // 计算结束位置，最大为当前行号加上 500
      var end = Math.min(cm.lastLine() + 1, pos.line + 500);
      // 查找下一个闭合标签
      var nextClose = CodeMirror.scanForClosingTag(cm, pos, null, end);
      // 如果未找到下一个闭合标签，或者闭合标签不是指定的标签，则返回 false
      if (!nextClose || nextClose.tag != tagName) return false;
      // 如果新标签存在，则将 onCx 设置为 1，否则为 0
      var onCx = newTag ? 1 : 0
      // 遍历上下文数组，查找相同标签的实例数量
      for (var i = context.length - 1; i >= 0; i--) {
        if (context[i] == tagName) ++onCx
        else break
      }
      // 更新位置为下一个闭合标签的位置
      pos = nextClose.to;
      // 如果存在多个相同标签的实例，需要检查是否有足够数量的闭合标签
      for (var i = 1; i < onCx; i++) {
        var next = CodeMirror.scanForClosingTag(cm, pos, null, end);
        if (!next || next.tag != tagName) return false;
        pos = next.to;
      }
      // 返回是否存在闭合标签的结果
      return true;
    }
# 闭合了一个代码块，可能是函数、循环、条件语句等的结束
```