# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\fold\foldcode.js`

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

  function doFold(cm, pos, options, force) {
    if (options && options.call) {
      var finder = options;
      options = null;
    } else {
      var finder = getOption(cm, options, "rangeFinder");
    }
    if (typeof pos == "number") pos = CodeMirror.Pos(pos, 0);
    var minSize = getOption(cm, options, "minFoldSize");

    function getRange(allowFolded) {
      var range = finder(cm, pos);
      if (!range || range.to.line - range.from.line < minSize) return null;
      var marks = cm.findMarksAt(range.from);
      for (var i = 0; i < marks.length; ++i) {
        if (marks[i].__isFold && force !== "fold") {
          if (!allowFolded) return null;
          range.cleared = true;
          marks[i].clear();
        }
      }
      return range;
    }

    var range = getRange(true);
    if (getOption(cm, options, "scanUp")) while (!range && pos.line > cm.firstLine()) {
      pos = CodeMirror.Pos(pos.line - 1, 0);
      range = getRange(false);
    }
    if (!range || range.cleared || force === "unfold") return;

    var myWidget = makeWidget(cm, options, range);
    CodeMirror.on(myWidget, "mousedown", function(e) {
      myRange.clear();
      CodeMirror.e_preventDefault(e);
    });
    var myRange = cm.markText(range.from, range.to, {
      replacedWith: myWidget,
      clearOnEnter: getOption(cm, options, "clearOnEnter"),
      __isFold: true
    });
    myRange.on("clear", function(from, to) {
      CodeMirror.signal(cm, "unfold", cm, from, to);
    });
  // 发出折叠信号，通知 CodeMirror 折叠了一段代码
  CodeMirror.signal(cm, "fold", cm, range.from, range.to);
}

// 创建折叠小部件
function makeWidget(cm, options, range) {
  // 从选项中获取小部件
  var widget = getOption(cm, options, "widget");

  // 如果小部件是函数，则调用函数获取小部件
  if (typeof widget == "function") {
    widget = widget(range.from, range.to);
  }

  // 如果小部件是字符串，则创建一个包含该字符串的 span 元素
  if (typeof widget == "string") {
    var text = document.createTextNode(widget);
    widget = document.createElement("span");
    widget.appendChild(text);
    widget.className = "CodeMirror-foldmarker";
  } else if (widget) {
    // 如果小部件存在，则克隆该小部件
    widget = widget.cloneNode(true)
  }
  return widget;
}

// 旧式接口
CodeMirror.newFoldFunction = function(rangeFinder, widget) {
  return function(cm, pos) { doFold(cm, pos, {rangeFinder: rangeFinder, widget: widget}); };
};

// 新式接口
CodeMirror.defineExtension("foldCode", function(pos, options, force) {
  doFold(this, pos, options, force);
});

// 判断指定位置是否被折叠
CodeMirror.defineExtension("isFolded", function(pos) {
  var marks = this.findMarksAt(pos);
  for (var i = 0; i < marks.length; ++i)
    if (marks[i].__isFold) return true;
});

// 折叠命令
CodeMirror.commands.toggleFold = function(cm) {
  cm.foldCode(cm.getCursor());
};
CodeMirror.commands.fold = function(cm) {
  cm.foldCode(cm.getCursor(), null, "fold");
};
CodeMirror.commands.unfold = function(cm) {
  cm.foldCode(cm.getCursor(), null, "unfold");
};
CodeMirror.commands.foldAll = function(cm) {
  cm.operation(function() {
    for (var i = cm.firstLine(), e = cm.lastLine(); i <= e; i++)
      cm.foldCode(CodeMirror.Pos(i, 0), null, "fold");
  });
};
CodeMirror.commands.unfoldAll = function(cm) {
  cm.operation(function() {
    for (var i = cm.firstLine(), e = cm.lastLine(); i <= e; i++)
      cm.foldCode(CodeMirror.Pos(i, 0), null, "unfold");
  });
};

// 注册折叠辅助函数
CodeMirror.registerHelper("fold", "combine", function() {
  var funcs = Array.prototype.slice.call(arguments, 0);
    # 定义一个函数，接受两个参数 cm 和 start
    return function(cm, start) {
      # 遍历 funcs 数组
      for (var i = 0; i < funcs.length; ++i) {
        # 调用 funcs[i] 函数，传入参数 cm 和 start
        var found = funcs[i](cm, start);
        # 如果找到了结果，则返回该结果
        if (found) return found;
      }
    };
  });

  # 注册一个自动折叠的辅助函数
  CodeMirror.registerHelper("fold", "auto", function(cm, start) {
    # 获取与折叠相关的辅助函数
    var helpers = cm.getHelpers(start, "fold");
    # 遍历辅助函数数组
    for (var i = 0; i < helpers.length; i++) {
      # 调用当前辅助函数，传入参数 cm 和 start
      var cur = helpers[i](cm, start);
      # 如果找到了结果，则返回该结果
      if (cur) return cur;
    }
  });

  # 定义默认选项对象
  var defaultOptions = {
    rangeFinder: CodeMirror.fold.auto,
    widget: "\u2194",
    minFoldSize: 0,
    scanUp: false,
    clearOnEnter: true
  };

  # 定义折叠选项
  CodeMirror.defineOption("foldOptions", null);

  # 定义一个函数，用于获取选项值
  function getOption(cm, options, name) {
    # 如果 options 存在且包含指定名称的值，则返回该值
    if (options && options[name] !== undefined)
      return options[name];
    # 否则，获取编辑器的折叠选项对象，并返回其中指定名称的值
    var editorOptions = cm.options.foldOptions;
    if (editorOptions && editorOptions[name] !== undefined)
      return editorOptions[name];
    # 否则，返回默认选项对象中指定名称的值
    return defaultOptions[name];
  }

  # 定义一个扩展函数，用于获取折叠选项值
  CodeMirror.defineExtension("foldOption", function(options, name) {
    return getOption(this, options, name);
  });
# 闭合了一个代码块或者函数的结束
```