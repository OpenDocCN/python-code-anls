# `ZeroNet\plugins\UiFileManager\media\codemirror\mode\htmlembedded.js`

```
# 导入模块
(function(mod) {
  # 如果是 CommonJS 环境
  if (typeof exports == "object" && typeof module == "object")
    mod(require("../../lib/codemirror"), require("../htmlmixed/htmlmixed"),
        require("../../addon/mode/multiplex"));
  # 如果是 AMD 环境
  else if (typeof define == "function" && define.amd)
    define(["../../lib/codemirror", "../htmlmixed/htmlmixed",
            "../../addon/mode/multiplex"], mod);
  # 如果是普通浏览器环境
  else
    mod(CodeMirror);
})(function(CodeMirror) {
  "use strict";

  # 定义模式为 htmlembedded
  CodeMirror.defineMode("htmlembedded", function(config, parserConfig) {
    # 获取关闭注释的标记
    var closeComment = parserConfig.closeComment || "--%>"
    # 返回多重模式，包含 htmlmixed 和 scriptingModeSpec
    return CodeMirror.multiplexingMode(CodeMirror.getMode(config, "htmlmixed"), {
      open: parserConfig.openComment || "<%--",
      close: closeComment,
      delimStyle: "comment",
      mode: {token: function(stream) {
        stream.skipTo(closeComment) || stream.skipToEnd()
        return "comment"
      }}
    }, {
      open: parserConfig.open || parserConfig.scriptStartRegex || "<%",
      close: parserConfig.close || parserConfig.scriptEndRegex || "%>",
      mode: CodeMirror.getMode(config, parserConfig.scriptingModeSpec)
    });
  }, "htmlmixed");

  # 定义 MIME 类型为 application/x-ejs
  CodeMirror.defineMIME("application/x-ejs", {name: "htmlembedded", scriptingModeSpec:"javascript"});
  # 定义 MIME 类型为 application/x-aspx
  CodeMirror.defineMIME("application/x-aspx", {name: "htmlembedded", scriptingModeSpec:"text/x-csharp"});
  # 定义 MIME 类型为 application/x-jsp
  CodeMirror.defineMIME("application/x-jsp", {name: "htmlembedded", scriptingModeSpec:"text/x-java"});
  # 定义 MIME 类型为 application/x-erb
  CodeMirror.defineMIME("application/x-erb", {name: "htmlembedded", scriptingModeSpec:"ruby"});
});
```