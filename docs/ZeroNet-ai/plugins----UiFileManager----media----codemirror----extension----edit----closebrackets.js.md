# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\edit\closebrackets.js`

```py
// 使用立即执行函数表达式（IIFE）来创建一个闭包，避免变量污染全局作用域
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
  // 默认配置项
  var defaults = {
    pairs: "()[]{}''\"\"",
    closeBefore: ")]}'\":;>",
    triples: "",
    explode: "[]{}"
  };

  // 定义 Pos 对象，用于表示 CodeMirror 中的位置
  var Pos = CodeMirror.Pos;

  // 定义 autoCloseBrackets 选项，初始化配置
  CodeMirror.defineOption("autoCloseBrackets", false, function(cm, val, old) {
    // 移除旧的按键映射
    if (old && old != CodeMirror.Init) {
      cm.removeKeyMap(keyMap);
      cm.state.closeBrackets = null;
    }
    // 如果新的配置存在，则添加新的按键映射
    if (val) {
      ensureBound(getOption(val, "pairs"))
      cm.state.closeBrackets = val;
      cm.addKeyMap(keyMap);
    }
  });

  // 获取配置项的值
  function getOption(conf, name) {
    if (name == "pairs" && typeof conf == "string") return conf;
    if (typeof conf == "object" && conf[name] != null) return conf[name];
    return defaults[name];
  }

  // 定义按键映射
  var keyMap = {Backspace: handleBackspace, Enter: handleEnter};
  // 确保按键映射中包含配置项中的字符
  function ensureBound(chars) {
    for (var i = 0; i < chars.length; i++) {
      var ch = chars.charAt(i), key = "'" + ch + "'"
      if (!keyMap[key]) keyMap[key] = handler(ch)
    }
  }
  ensureBound(defaults.pairs + "`")

  // 处理特定字符的按键事件
  function handler(ch) {
    return function(cm) { return handleChar(cm, ch); };
  }

  // 获取配置项的值
  function getConfig(cm) {
    var deflt = cm.state.closeBrackets;
    if (!deflt || deflt.override) return deflt;
    var mode = cm.getModeAt(cm.getCursor());
    return mode.closeBrackets || deflt;
  }

  // 处理 Backspace 按键事件
  function handleBackspace(cm) {
    var conf = getConfig(cm);
    if (!conf || cm.getOption("disableInput")) return CodeMirror.Pass;

    var pairs = getOption(conf, "pairs");
    var ranges = cm.listSelections();
    // 遍历ranges数组，对每个range进行处理
    for (var i = 0; i < ranges.length; i++) {
      // 如果range不为空，则返回CodeMirror.Pass
      if (!ranges[i].empty()) return CodeMirror.Pass;
      // 获取range头部字符周围的字符
      var around = charsAround(cm, ranges[i].head);
      // 如果周围字符不存在或者不在pairs数组中，返回CodeMirror.Pass
      if (!around || pairs.indexOf(around) % 2 != 0) return CodeMirror.Pass;
    }
    // 逆序遍历ranges数组
    for (var i = ranges.length - 1; i >= 0; i--) {
      // 获取当前range的头部位置
      var cur = ranges[i].head;
      // 在当前位置前后各删除一个字符
      cm.replaceRange("", Pos(cur.line, cur.ch - 1), Pos(cur.line, cur.ch + 1), "+delete");
    }
  }

  // 处理回车键事件
  function handleEnter(cm) {
    // 获取配置信息
    var conf = getConfig(cm);
    // 获取是否explode配置
    var explode = conf && getOption(conf, "explode");
    // 如果不存在explode配置或者输入被禁用，则返回CodeMirror.Pass
    if (!explode || cm.getOption("disableInput")) return CodeMirror.Pass;

    // 获取选择范围
    var ranges = cm.listSelections();
    // 遍历ranges数组
    for (var i = 0; i < ranges.length; i++) {
      // 如果range不为空，则返回CodeMirror.Pass
      if (!ranges[i].empty()) return CodeMirror.Pass;
      // 获取range头部字符周围的字符
      var around = charsAround(cm, ranges[i].head);
      // 如果周围字符不存在或者不在explode数组中，返回CodeMirror.Pass
      if (!around || explode.indexOf(around) % 2 != 0) return CodeMirror.Pass;
    }
    // 执行操作
    cm.operation(function() {
      // 获取行分隔符
      var linesep = cm.lineSeparator() || "\n";
      // 在选择范围内插入两个行分隔符
      cm.replaceSelection(linesep + linesep, null);
      // 执行goCharLeft命令
      cm.execCommand("goCharLeft");
      // 重新获取选择范围
      ranges = cm.listSelections();
      // 遍历ranges数组
      for (var i = 0; i < ranges.length; i++) {
        // 获取行号
        var line = ranges[i].head.line;
        // 对当前行和下一行进行缩进
        cm.indentLine(line, null, true);
        cm.indentLine(line + 1, null, true);
      }
    });
  }

  // 收缩选择范围
  function contractSelection(sel) {
    // 判断选择范围是否倒置
    var inverted = CodeMirror.cmpPos(sel.anchor, sel.head) > 0;
    // 返回收缩后的选择范围
    return {anchor: new Pos(sel.anchor.line, sel.anchor.ch + (inverted ? -1 : 1)),
            head: new Pos(sel.head.line, sel.head.ch + (inverted ? 1 : -1))};
  }

  // 处理字符输入事件
  function handleChar(cm, ch) {
    // 获取配置信息
    var conf = getConfig(cm);
    // 如果不存在配置信息或者输入被禁用，则返回CodeMirror.Pass
    if (!conf || cm.getOption("disableInput")) return CodeMirror.Pass;

    // 获取pairs配置
    var pairs = getOption(conf, "pairs");
    // 获取字符在pairs中的位置
    var pos = pairs.indexOf(ch);
    // 如果字符不在pairs中，则返回CodeMirror.Pass
    if (pos == -1) return CodeMirror.Pass;

    // 获取closeBefore配置
    var closeBefore = getOption(conf,"closeBefore");

    // 获取triples配置
    var triples = getOption(conf, "triples");

    // 判断是否为相同字符
    var identical = pairs.charAt(pos + 1) == ch;
    // 获取选择范围
    var ranges = cm.listSelections();
    # 判断当前光标位置是否在偶数列
    var opening = pos % 2 == 0;

    # 定义变量 type
    var type;
    # 遍历 ranges 数组
    for (var i = 0; i < ranges.length; i++) {
      # 获取当前 range 的头部位置和当前类型
      var range = ranges[i], cur = range.head, curType;
      # 获取当前位置的下一个字符
      var next = cm.getRange(cur, Pos(cur.line, cur.ch + 1));
      # 判断是否为开放符号并且 range 不为空
      if (opening && !range.empty()) {
        curType = "surround";
      } else if ((identical || !opening) && next == ch) {
        # 判断是否为相同符号并且下一个字符等于当前字符
        if (identical && stringStartsAfter(cm, cur))
          curType = "both";
        else if (triples.indexOf(ch) >= 0 && cm.getRange(cur, Pos(cur.line, cur.ch + 3)) == ch + ch + ch)
          curType = "skipThree";
        else
          curType = "skip";
      } else if (identical && cur.ch > 1 && triples.indexOf(ch) >= 0 &&
                 cm.getRange(Pos(cur.line, cur.ch - 2), cur) == ch + ch) {
        # 判断是否为相同符号并且当前列大于1且为三重符号
        if (cur.ch > 2 && /\bstring/.test(cm.getTokenTypeAt(Pos(cur.line, cur.ch - 2)))) return CodeMirror.Pass;
        curType = "addFour";
      } else if (identical) {
        # 判断是否为相同符号
        var prev = cur.ch == 0 ? " " : cm.getRange(Pos(cur.line, cur.ch - 1), cur)
        if (!CodeMirror.isWordChar(next) && prev != ch && !CodeMirror.isWordChar(prev)) curType = "both";
        else return CodeMirror.Pass;
      } else if (opening && (next.length === 0 || /\s/.test(next) || closeBefore.indexOf(next) > -1)) {
        # 判断是否为开放符号并且下一个字符为空或者为空格或者在 closeBefore 中
        curType = "both";
      } else {
        return CodeMirror.Pass;
      }
      # 如果 type 为空，则赋值为当前类型，否则如果 type 不等于当前类型，则返回 CodeMirror.Pass
      if (!type) type = curType;
      else if (type != curType) return CodeMirror.Pass;
    }

    # 定义 left 和 right 变量
    var left = pos % 2 ? pairs.charAt(pos - 1) : ch;
    var right = pos % 2 ? ch : pairs.charAt(pos + 1);
    # 执行编辑器操作
    cm.operation(function() {
      # 如果类型为"skip"，执行向右移动一个字符的命令
      if (type == "skip") {
        cm.execCommand("goCharRight");
      } 
      # 如果类型为"skipThree"，执行向右移动三个字符的命令
      else if (type == "skipThree") {
        for (var i = 0; i < 3; i++)
          cm.execCommand("goCharRight");
      } 
      # 如果类型为"surround"，执行包围文本的操作
      else if (type == "surround") {
        # 获取当前选中的文本
        var sels = cm.getSelections();
        # 遍历选中的文本，添加左右包围字符
        for (var i = 0; i < sels.length; i++)
          sels[i] = left + sels[i] + right;
        # 替换选中的文本
        cm.replaceSelections(sels, "around");
        # 复制选中的文本的位置
        sels = cm.listSelections().slice();
        # 缩小选中的文本的范围
        for (var i = 0; i < sels.length; i++)
          sels[i] = contractSelection(sels[i]);
        # 设置新的选中文本
        cm.setSelections(sels);
      } 
      # 如果类型为"both"，执行添加左右字符的操作
      else if (type == "both") {
        # 替换选中的文本为左右字符
        cm.replaceSelection(left + right, null);
        # 触发电动力
        cm.triggerElectric(left + right);
        # 执行向左移动一个字符的命令
        cm.execCommand("goCharLeft");
      } 
      # 如果类型为"addFour"，执行在选中文本前添加四个字符的操作
      else if (type == "addFour") {
        cm.replaceSelection(left + left + left + left, "before");
        # 执行向右移动一个字符的命令
        cm.execCommand("goCharRight");
      }
    });
  }

  # 获取指定位置周围的字符
  function charsAround(cm, pos) {
    var str = cm.getRange(Pos(pos.line, pos.ch - 1),
                          Pos(pos.line, pos.ch + 1));
    return str.length == 2 ? str : null;
  }

  # 检查指定位置是否在字符串之后
  function stringStartsAfter(cm, pos) {
    # 获取指定位置后面的标记
    var token = cm.getTokenAt(Pos(pos.line, pos.ch + 1))
    # 检查是否在字符串之后
    return /\bstring/.test(token.type) && token.start == pos.ch &&
      (pos.ch == 0 || !/\bstring/.test(cm.getTokenTypeAt(pos)))
  }
# 闭合了一个代码块或者函数的结束
```