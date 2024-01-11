# `ZeroNet\plugins\UiFileManager\media\codemirror\mode\htmlmixed.js`

```
// CodeMirror, 版权由 Marijn Haverbeke 和其他人拥有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"), require("../xml/xml"), require("../javascript/javascript"), require("../css/css"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror", "../xml/xml", "../javascript/javascript", "../css/css"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  var defaultTags = {
    script: [
      ["lang", /(javascript|babel)/i, "javascript"],
      ["type", /^(?:text|application)\/(?:x-)?(?:java|ecma)script$|^module$|^$/i, "javascript"],
      ["type", /./, "text/plain"],
      [null, null, "javascript"]
    ],
    style:  [
      ["lang", /^css$/i, "css"],
      ["type", /^(text\/)?(x-)?(stylesheet|css)$/i, "css"],
      ["type", /./, "text/plain"],
      [null, null, "css"]
    ]
  };

  function maybeBackup(stream, pat, style) {
    var cur = stream.current(), close = cur.search(pat);
    if (close > -1) {
      stream.backUp(cur.length - close);
    } else if (cur.match(/<\/?$/)) {
      stream.backUp(cur.length);
      if (!stream.match(pat, false)) stream.match(cur);
    }
    return style;
  }

  var attrRegexpCache = {};
  function getAttrRegexp(attr) {
    var regexp = attrRegexpCache[attr];
    if (regexp) return regexp;
    return attrRegexpCache[attr] = new RegExp("\\s+" + attr + "\\s*=\\s*('|\")?([^'\"]+)('|\")?\\s*");
  }

  function getAttrValue(text, attr) {
    var match = text.match(getAttrRegexp(attr))
    return match ? /^\s*(.*?)\s*$/.exec(match[2])[1] : ""
  }

  function getTagRegexp(tagName, anchored) {
    return new RegExp((anchored ? "^" : "") + "<\/\s*" + tagName + "\s*>", "i");
  }

  function addTags(from, to) {
    // 遍历 from 对象的属性
    for (var tag in from) {
      // 如果 to 对象中不存在当前属性，则创建一个空数组
      var dest = to[tag] || (to[tag] = []);
      // 获取 from 对象中当前属性的值
      var source = from[tag];
      // 遍历 from 对象中当前属性的值，并将其逆序添加到 to 对象中当前属性对应的数组中
      for (var i = source.length - 1; i >= 0; i--)
        dest.unshift(source[i])
    }
  }

  // 根据标签信息和标签文本查找匹配的模式
  function findMatchingMode(tagInfo, tagText) {
    // 遍历标签信息数组
    for (var i = 0; i < tagInfo.length; i++) {
      // 获取当前标签信息
      var spec = tagInfo[i];
      // 如果当前标签信息为空或者匹配标签文本的属性值，则返回对应的模式
      if (!spec[0] || spec[1].test(getAttrValue(tagText, spec[0]))) return spec[2];
    }
  }

  // 定义名为 "htmlmixed" 的 CodeMirror 模式
  CodeMirror.defineMode("htmlmixed", function (config, parserConfig) {
    // 获取 XML 模式，并设置相关配置
    var htmlMode = CodeMirror.getMode(config, {
      name: "xml",
      htmlMode: true,
      multilineTagIndentFactor: parserConfig.multilineTagIndentFactor,
      multilineTagIndentPastTag: parserConfig.multilineTagIndentPastTag
    });

    // 创建空对象 tags
    var tags = {};
    // 获取配置中的标签和脚本类型
    var configTags = parserConfig && parserConfig.tags, configScript = parserConfig && parserConfig.scriptTypes;
    // 添加默认标签
    addTags(defaultTags, tags);
    // 如果存在配置中的标签，则添加到 tags 对象中
    if (configTags) addTags(configTags, tags);
    // 如果存在配置中的脚本类型，则遍历添加到 tags 对象中
    if (configScript) for (var i = configScript.length - 1; i >= 0; i--)
      tags.script.unshift(["type", configScript[i].matches, configScript[i].mode])
    // 定义处理 HTML 内容的函数
    function html(stream, state) {
      // 调用 htmlMode.token 处理 HTML 内容，获取样式
      var style = htmlMode.token(stream, state.htmlState), tag = /\btag\b/.test(style), tagName
      // 如果当前是标签，并且当前字符不是 <、>、空格或斜杠
      // 并且存在标签名，并且标签名在 tags 对象中
      if (tag && !/[<>\s\/]/.test(stream.current()) &&
          (tagName = state.htmlState.tagName && state.htmlState.tagName.toLowerCase()) &&
          tags.hasOwnProperty(tagName)) {
        // 设置状态中的 inTag 为标签名加一个空格
        state.inTag = tagName + " "
      } else if (state.inTag && tag && />$/.test(stream.current())) {
        // 如果存在 inTag 并且当前是标签结束符 >
        // 通过正则表达式获取标签名和属性
        var inTag = /^([\S]+) (.*)/.exec(state.inTag)
        state.inTag = null
        // 获取标签对应的模式
        var modeSpec = stream.current() == ">" && findMatchingMode(tags[inTag[1]], inTag[2])
        var mode = CodeMirror.getMode(config, modeSpec)
        // 获取标签的结束标签正则表达式
        var endTagA = getTagRegexp(inTag[1], true), endTag = getTagRegexp(inTag[1], false);
        // 设置状态中的 token 函数
        state.token = function (stream, state) {
          // 如果匹配到结束标签
          if (stream.match(endTagA, false)) {
            state.token = html;
            state.localState = state.localMode = null;
            return null;
          }
          // 否则调用当前模式的 token 函数处理内容
          return maybeBackup(stream, endTag, state.localMode.token(stream, state.localState));
        };
        state.localMode = mode;
        // 初始化局部模式和状态
        state.localState = CodeMirror.startState(mode, htmlMode.indent(state.htmlState, "", ""));
      } else if (state.inTag) {
        // 如果存在 inTag，则将当前字符添加到 inTag 中
        state.inTag += stream.current()
        // 如果当前是行尾，则在 inTag 后面添加一个空格
        if (stream.eol()) state.inTag += " "
      }
      // 返回样式
      return style;
    };
    # 返回一个对象，包含了开始状态的函数和其他方法
    return {
      # 开始状态的函数
      startState: function () {
        # 使用 htmlMode 的开始状态作为 state 的初始状态
        var state = CodeMirror.startState(htmlMode);
        # 返回一个对象，包含 token、inTag、localMode、localState 和 htmlState
        return {token: html, inTag: null, localMode: null, localState: null, htmlState: state};
      },

      # 复制状态的函数
      copyState: function (state) {
        var local;
        # 如果 state 中有 localState，则使用 CodeMirror.copyState 复制 localState
        if (state.localState) {
          local = CodeMirror.copyState(state.localMode, state.localState);
        }
        # 返回一个对象，包含 token、inTag、localMode、localState 和 htmlState
        return {token: state.token, inTag: state.inTag,
                localMode: state.localMode, localState: local,
                htmlState: CodeMirror.copyState(htmlMode, state.htmlState)};
      },

      # token 函数
      token: function (stream, state) {
        # 调用 state 中的 token 函数
        return state.token(stream, state);
      },

      # 缩进函数
      indent: function (state, textAfter, line) {
        # 如果没有 localMode 或者 textAfter 以 </ 开头，则调用 htmlMode 的 indent 函数
        if (!state.localMode || /^\s*<\//.test(textAfter))
          return htmlMode.indent(state.htmlState, textAfter, line);
        # 如果有 localMode 并且 localMode 中有 indent 函数，则调用 localMode 的 indent 函数
        else if (state.localMode.indent)
          return state.localMode.indent(state.localState, textAfter, line);
        # 否则返回 CodeMirror.Pass
        else
          return CodeMirror.Pass;
      },

      # 内部模式函数
      innerMode: function (state) {
        # 返回一个对象，包含 localState 或 htmlState 以及 localMode 或 htmlMode
        return {state: state.localState || state.htmlState, mode: state.localMode || htmlMode};
      }
    };
  }, "xml", "javascript", "css");

  # 定义 MIME 类型为 text/html 的语法为 htmlmixed
  CodeMirror.defineMIME("text/html", "htmlmixed");
# 闭合了一个代码块或者函数的结束
```