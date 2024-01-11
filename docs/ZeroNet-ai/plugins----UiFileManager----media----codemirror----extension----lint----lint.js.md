# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\lint\lint.js`

```
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
  var GUTTER_ID = "CodeMirror-lint-markers";

  function showTooltip(cm, e, content) {
    // 创建提示框元素
    var tt = document.createElement("div");
    tt.className = "CodeMirror-lint-tooltip cm-s-" + cm.options.theme;
    tt.appendChild(content.cloneNode(true));
    // 将提示框添加到编辑器或者页面上
    if (cm.state.lint.options.selfContain)
      cm.getWrapperElement().appendChild(tt);
    else
      document.body.appendChild(tt);

    function position(e) {
      // 根据鼠标位置调整提示框位置
      if (!tt.parentNode) return CodeMirror.off(document, "mousemove", position);
      tt.style.top = Math.max(0, e.clientY - tt.offsetHeight - 5) + "px";
      tt.style.left = (e.clientX + 5) + "px";
    }
    CodeMirror.on(document, "mousemove", position);
    position(e);
    if (tt.style.opacity != null) tt.style.opacity = 1;
    return tt;
  }
  function rm(elt) {
    // 移除元素
    if (elt.parentNode) elt.parentNode.removeChild(elt);
  }
  function hideTooltip(tt) {
    // 隐藏提示框
    if (!tt.parentNode) return;
    if (tt.style.opacity == null) rm(tt);
    tt.style.opacity = 0;
    setTimeout(function() { rm(tt); }, 600);
  }

  function showTooltipFor(cm, e, content, node) {
    // 显示带有节点的提示框
    var tooltip = showTooltip(cm, e, content);
    function hide() {
      CodeMirror.off(node, "mouseout", hide);
      if (tooltip) { hideTooltip(tooltip); tooltip = null; }
    }
    var poll = setInterval(function() {
      if (tooltip) for (var n = node;; n = n.parentNode) {
        if (n && n.nodeType == 11) n = n.host;
        if (n == document.body) return;
        if (!n) { hide(); break; }
      }
      if (!tooltip) return clearInterval(poll);
    }
  // 在指定时间后执行函数
  }, 400);
  // 当鼠标移出指定区域时触发隐藏函数
  CodeMirror.on(node, "mouseout", hide);
}

// LintState 类的构造函数
function LintState(cm, options, hasGutter) {
  // 已标记的数组
  this.marked = [];
  // 选项
  this.options = options;
  // 超时
  this.timeout = null;
  // 是否有 gutter
  this.hasGutter = hasGutter;
  // 鼠标悬停事件处理函数
  this.onMouseOver = function(e) { onMouseOver(cm, e); };
  // 等待中的数量
  this.waitingFor = 0
}

// 解析选项
function parseOptions(_cm, options) {
  // 如果选项是函数，则返回包含 getAnnotations 函数的对象
  if (options instanceof Function) return {getAnnotations: options};
  // 如果选项为空或为 true，则返回空对象
  if (!options || options === true) options = {};
  return options;
}

// 清除标记
function clearMarks(cm) {
  var state = cm.state.lint;
  // 如果有 gutter，则清除 gutter
  if (state.hasGutter) cm.clearGutter(GUTTER_ID);
  // 清除所有标记
  for (var i = 0; i < state.marked.length; ++i)
    state.marked[i].clear();
  state.marked.length = 0;
}

// 创建标记
function makeMarker(cm, labels, severity, multiple, tooltips) {
  var marker = document.createElement("div"), inner = marker;
  // 设置标记的类名
  marker.className = "CodeMirror-lint-marker-" + severity;
  // 如果是多个标记，则创建内部标记
  if (multiple) {
    inner = marker.appendChild(document.createElement("div"));
    inner.className = "CodeMirror-lint-marker-multiple";
  }

  // 如果 tooltips 不为 false，则绑定鼠标悬停事件
  if (tooltips != false) CodeMirror.on(inner, "mouseover", function(e) {
    showTooltipFor(cm, e, labels, inner);
  });

  return marker;
}

// 获取最大的严重程度
function getMaxSeverity(a, b) {
  if (a == "error") return a;
  else return b;
}

// 按行分组注解
function groupByLine(annotations) {
  var lines = [];
  for (var i = 0; i < annotations.length; ++i) {
    var ann = annotations[i], line = ann.from.line;
    (lines[line] || (lines[line] = [])).push(ann);
  }
  return lines;
}

// 注解提示
function annotationTooltip(ann) {
  var severity = ann.severity;
  if (!severity) severity = "error";
  var tip = document.createElement("div");
  tip.className = "CodeMirror-lint-message-" + severity;
  if (typeof ann.messageHTML != 'undefined') {
    tip.innerHTML = ann.messageHTML;
  } else {
    tip.appendChild(document.createTextNode(ann.message));
  }
    // 返回提示
    return tip;
  }

  // 异步检查代码
  function lintAsync(cm, getAnnotations, passOptions) {
    // 获取编辑器的 lint 状态
    var state = cm.state.lint
    // 递增等待计数
    var id = ++state.waitingFor
    // 定义中止函数
    function abort() {
      id = -1
      cm.off("change", abort)
    }
    // 监听编辑器变化事件，触发中止函数
    cm.on("change", abort)
    // 获取代码注解
    getAnnotations(cm.getValue(), function(annotations, arg2) {
      cm.off("change", abort)
      // 如果等待计数不匹配，则返回
      if (state.waitingFor != id) return
      // 如果有第二个参数并且注解是 CodeMirror 类型，则赋值给 annotations
      if (arg2 && annotations instanceof CodeMirror) annotations = arg2
      // 执行更新代码检查
      cm.operation(function() {updateLinting(cm, annotations)})
    }, passOptions, cm);
  }

  // 开始代码检查
  function startLinting(cm) {
    // 获取编辑器的 lint 状态和选项
    var state = cm.state.lint, options = state.options;
    /*
     * 通过 `options` 属性传递规则，防止 JSHint（和其他检查器）抱怨无法识别的规则，如 `onUpdateLinting`、`delay`、`lintOnChange` 等。
     */
    var passOptions = options.options || options;
    // 获取注解或者获取代码辅助函数
    var getAnnotations = options.getAnnotations || cm.getHelper(CodeMirror.Pos(0, 0), "lint");
    // 如果没有获取到注解或者代码辅助函数，则返回
    if (!getAnnotations) return;
    // 如果选项中有异步属性或者获取注解的函数有异步属性，则执行异步代码检查
    if (options.async || getAnnotations.async) {
      lintAsync(cm, getAnnotations, passOptions)
    } else {
      // 否则执行同步代码检查
      var annotations = getAnnotations(cm.getValue(), passOptions, cm);
      // 如果没有注解，则返回
      if (!annotations) return;
      // 如果注解是一个 Promise 对象，则执行 Promise 的回调函数
      if (annotations.then) annotations.then(function(issues) {
        cm.operation(function() {updateLinting(cm, issues)})
      });
      // 否则执行更新代码检查
      else cm.operation(function() {updateLinting(cm, annotations)})
    }
  }

  // 更新代码检查
  function updateLinting(cm, annotationsNotSorted) {
    // 清除标记
    clearMarks(cm);
    // 获取编辑器的 lint 状态和选项
    var state = cm.state.lint, options = state.options;

    // 按行分组注解
    var annotations = groupByLine(annotationsNotSorted);
    // 遍历注释数组，处理每一行的注释
    for (var line = 0; line < annotations.length; ++line) {
      // 获取当前行的注释
      var anns = annotations[line];
      // 如果当前行没有注释，则跳过
      if (!anns) continue;

      // 初始化最大严重性和提示标签
      var maxSeverity = null;
      var tipLabel = state.hasGutter && document.createDocumentFragment();

      // 遍历当前行的注释
      for (var i = 0; i < anns.length; ++i) {
        // 获取当前注释
        var ann = anns[i];
        // 获取注释的严重性，如果没有则默认为"error"
        var severity = ann.severity;
        if (!severity) severity = "error";
        // 更新最大严重性
        maxSeverity = getMaxSeverity(maxSeverity, severity);

        // 如果有自定义的注释格式化函数，则对注释进行格式化
        if (options.formatAnnotation) ann = options.formatAnnotation(ann);
        // 如果存在提示标签，则添加注释的提示
        if (state.hasGutter) tipLabel.appendChild(annotationTooltip(ann));

        // 如果注释有结束位置，则在编辑器中标记出注释的位置范围
        if (ann.to) state.marked.push(cm.markText(ann.from, ann.to, {
          className: "CodeMirror-lint-mark-" + severity,
          __annotation: ann
        }));
      }

      // 如果存在提示标签，则设置编辑器的行号标记
      if (state.hasGutter)
        cm.setGutterMarker(line, GUTTER_ID, makeMarker(cm, tipLabel, maxSeverity, anns.length > 1,
                                                       state.options.tooltips));
    }
    // 如果存在更新注释的回调函数，则调用该函数
    if (options.onUpdateLinting) options.onUpdateLinting(annotationsNotSorted, annotations, cm);
  }

  // 编辑器内容改变时的事件处理函数
  function onChange(cm) {
    var state = cm.state.lint;
    if (!state) return;
    // 清除之前的定时器，设置新的定时器用于延迟触发代码检查
    clearTimeout(state.timeout);
    state.timeout = setTimeout(function(){startLinting(cm);}, state.options.delay || 500);
  }

  // 鼠标悬停时显示提示信息
  function popupTooltips(cm, annotations, e) {
    var target = e.target || e.srcElement;
    var tooltip = document.createDocumentFragment();
    // 遍历注释数组，为每个注释添加提示信息
    for (var i = 0; i < annotations.length; i++) {
      var ann = annotations[i];
      tooltip.appendChild(annotationTooltip(ann));
    }
    // 在鼠标位置显示提示信息
    showTooltipFor(cm, e, tooltip, target);
  }

  // 鼠标悬停在标记上时的事件处理函数
  function onMouseOver(cm, e) {
    var target = e.target || e.srcElement;
    // 如果鼠标悬停在标记上，则获取标记的位置信息
    if (!/\bCodeMirror-lint-mark-/.test(target.className)) return;
    var box = target.getBoundingClientRect(), x = (box.left + box.right) / 2, y = (box.top + box.bottom) / 2;
    var spans = cm.findMarksAt(cm.coordsChar({left: x, top: y}, "client"));

    var annotations = [];
    // 遍历 spans 数组，获取每个元素的 __annotation 属性，如果存在则将其添加到 annotations 数组中
    for (var i = 0; i < spans.length; ++i) {
      var ann = spans[i].__annotation;
      if (ann) annotations.push(ann);
    }
    // 如果 annotations 数组不为空，则调用 popupTooltips 方法显示提示框
    if (annotations.length) popupTooltips(cm, annotations, e);
  }

  // 定义 lint 选项，当值发生变化时执行相应操作
  CodeMirror.defineOption("lint", false, function(cm, val, old) {
    // 如果旧值存在且不等于 CodeMirror.Init，则清除标记、解绑事件和定时器，并删除 lint 状态
    if (old && old != CodeMirror.Init) {
      clearMarks(cm);
      if (cm.state.lint.options.lintOnChange !== false)
        cm.off("change", onChange);
      CodeMirror.off(cm.getWrapperElement(), "mouseover", cm.state.lint.onMouseOver);
      clearTimeout(cm.state.lint.timeout);
      delete cm.state.lint;
    }

    // 如果新值存在
    if (val) {
      // 获取 gutters 和是否存在 lint gutter 的状态
      var gutters = cm.getOption("gutters"), hasLintGutter = false;
      for (var i = 0; i < gutters.length; ++i) if (gutters[i] == GUTTER_ID) hasLintGutter = true;
      // 创建 lint 状态对象，并根据选项绑定事件
      var state = cm.state.lint = new LintState(cm, parseOptions(cm, val), hasLintGutter);
      if (state.options.lintOnChange !== false)
        cm.on("change", onChange);
      // 如果选项中 tooltips 不为 false 且不为 "gutter"，则绑定鼠标悬停事件
      if (state.options.tooltips != false && state.options.tooltips != "gutter")
        CodeMirror.on(cm.getWrapperElement(), "mouseover", state.onMouseOver);

      // 开始 linting
      startLinting(cm);
    }
  });

  // 定义 performLint 方法，执行 lint 操作
  CodeMirror.defineExtension("performLint", function() {
    // 如果存在 lint 状态，则开始 linting
    if (this.state.lint) startLinting(this);
  });
# 闭合了一个代码块或者函数的结束
```