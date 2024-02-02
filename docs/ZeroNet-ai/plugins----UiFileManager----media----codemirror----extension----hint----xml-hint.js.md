# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\hint\xml-hint.js`

```py
// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  var Pos = CodeMirror.Pos;

  function matches(hint, typed, matchInMiddle) {
    // 如果 matchInMiddle 为真，则返回提示是否包含已输入的内容
    if (matchInMiddle) return hint.indexOf(typed) >= 0;
    // 否则返回提示是否以已输入的内容开头
    else return hint.lastIndexOf(typed, 0) == 0;
  }

  function getHints(cm, options) {
    var tags = options && options.schemaInfo;
    var quote = (options && options.quoteChar) || '"';
    var matchInMiddle = options && options.matchInMiddle;
    // 如果没有标签信息，则返回
    if (!tags) return;
    var cur = cm.getCursor(), token = cm.getTokenAt(cur);
    // 如果 token 的结束位置大于光标位置，则修正 token 的结束位置和字符串内容
    if (token.end > cur.ch) {
      token.end = cur.ch;
      token.string = token.string.slice(0, cur.ch - token.start);
    }
    // 获取内部模式和当前标签信息
    var inner = CodeMirror.innerMode(cm.getMode(), token.state);
    if (!inner.mode.xmlCurrentTag) return
    var result = [], replaceToken = false, prefix;
    var tag = /\btag\b/.test(token.type) && !/>$/.test(token.string);
    var tagName = tag && /^\w/.test(token.string), tagStart;

    if (tagName) {
      var before = cm.getLine(cur.line).slice(Math.max(0, token.start - 2), token.start);
      var tagType = /<\/$/.test(before) ? "close" : /<$/.test(before) ? "open" : null;
      if (tagType) tagStart = token.start - (tagType == "close" ? 2 : 1);
    } else if (tag && token.string == "<") {
      tagType = "open";
    } else if (tag && token.string == "</") {
      tagType = "close";
    }

    var tagInfo = inner.mode.xmlCurrentTag(inner.state)
    # 如果标签不存在并且标签信息也不存在，或者标签类型存在
    if (!tag && !tagInfo || tagType) {
      # 如果存在标签名，则将其赋值给前缀
      if (tagName)
        prefix = token.string;
      # 将替换标记设置为标签类型
      replaceToken = tagType;
      # 获取当前 XML 上下文，如果存在则获取最后一个内部标签
      var context = inner.mode.xmlCurrentContext ? inner.mode.xmlCurrentContext(inner.state) : []
      var inner = context.length && context[context.length - 1]
      # 获取当前内部标签和其子标签列表
      var curTag = inner && tags[inner]
      var childList = inner ? curTag && curTag.children : tags["!top"];
      # 如果子标签列表存在且标签类型不是关闭标签
      if (childList && tagType != "close") {
        # 遍历子标签列表，如果前缀为空或者匹配中间位置，则将标签名加入结果列表
        for (var i = 0; i < childList.length; ++i) if (!prefix || matches(childList[i], prefix, matchInMiddle))
          result.push("<" + childList[i]);
      } else if (tagType != "close") {
        # 如果子标签列表不存在且标签类型不是关闭标签
        # 遍历所有标签，如果标签名不是顶级标签、不是属性标签且前缀为空或者匹配中间位置，则将标签名加入结果列表
        for (var name in tags)
          if (tags.hasOwnProperty(name) && name != "!top" && name != "!attrs" && (!prefix || matches(name, prefix, matchInMiddle)))
            result.push("<" + name);
      }
      # 如果存在内部标签且前缀为空或者标签类型是关闭标签且匹配中间位置
      if (inner && (!prefix || tagType == "close" && matches(inner, prefix, matchInMiddle)))
        result.push("</" + inner + ">");
    }
    # 返回结果对象，包括结果列表、起始位置和结束位置
    return {
      list: result,
      from: replaceToken ? Pos(cur.line, tagStart == null ? token.start : tagStart) : cur,
      to: replaceToken ? Pos(cur.line, token.end) : cur
    };
  }

  # 注册 XML 提示的帮助函数
  CodeMirror.registerHelper("hint", "xml", getHints);
# 闭合了一个代码块或者函数的结束括号
```