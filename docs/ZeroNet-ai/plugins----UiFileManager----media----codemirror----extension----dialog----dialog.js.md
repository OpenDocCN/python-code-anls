# `ZeroNet\plugins\UiFileManager\media\codemirror\extension\dialog\dialog.js`

```
// CodeMirror, 版权归 Marijn Haverbeke 和其他人所有
// 根据 MIT 许可证分发：https://codemirror.net/LICENSE

// 在编辑器顶部打开简单对话框。依赖于 dialog.css。

(function(mod) {
  if (typeof exports == "object" && typeof module == "object") // CommonJS
    mod(require("../../lib/codemirror"));
  else if (typeof define == "function" && define.amd) // AMD
    define(["../../lib/codemirror"], mod);
  else // Plain browser env
    mod(CodeMirror);
})(function(CodeMirror) {
  // 创建对话框的 div 元素
  function dialogDiv(cm, template, bottom) {
    var wrap = cm.getWrapperElement();
    var dialog;
    dialog = wrap.appendChild(document.createElement("div"));
    // 根据参数决定对话框的位置
    if (bottom)
      dialog.className = "CodeMirror-dialog CodeMirror-dialog-bottom";
    else
      dialog.className = "CodeMirror-dialog CodeMirror-dialog-top";

    // 根据模板类型决定对话框内容
    if (typeof template == "string") {
      dialog.innerHTML = template;
    } else { // 假设它是一个独立的 DOM 元素。
      dialog.appendChild(template);
    }
    // 给编辑器添加类名，表示对话框已打开
    CodeMirror.addClass(wrap, 'dialog-opened');
    return dialog;
  }

  // 关闭通知
  function closeNotification(cm, newVal) {
    if (cm.state.currentNotificationClose)
      cm.state.currentNotificationClose();
    cm.state.currentNotificationClose = newVal;
  }

  // 定义 openDialog 方法
  CodeMirror.defineExtension("openDialog", function(template, callback, options) {
    if (!options) options = {};

    closeNotification(this, null);

    // 创建对话框
    var dialog = dialogDiv(this, template, options.bottom);
    var closed = false, me = this;
    function close(newVal) {
      if (typeof newVal == 'string') {
        inp.value = newVal;
      } else {
        if (closed) return;
        closed = true;
        // 移除对话框并恢复编辑器焦点
        CodeMirror.rmClass(dialog.parentNode, 'dialog-opened');
        dialog.parentNode.removeChild(dialog);
        me.focus();

        if (options.onClose) options.onClose(dialog);
      }
    }

    // 获取对话框中的输入框和按钮
    var inp = dialog.getElementsByTagName("input")[0], button;
    // 如果输入框存在
    if (inp) {
      // 让输入框获得焦点
      inp.focus();

      // 如果选项中有数值，则将输入框的值设置为该数值，并根据选项决定是否在打开时选择输入框中的值
      if (options.value) {
        inp.value = options.value;
        if (options.selectValueOnOpen !== false) {
          inp.select();
        }
      }

      // 如果选项中有 onInput 回调函数，则在输入框输入时触发该回调函数
      if (options.onInput)
        CodeMirror.on(inp, "input", function(e) { options.onInput(e, inp.value, close);});
      // 如果选项中有 onKeyUp 回调函数，则在输入框按键弹起时触发该回调函数
      if (options.onKeyUp)
        CodeMirror.on(inp, "keyup", function(e) {options.onKeyUp(e, inp.value, close);});

      // 在输入框按键按下时触发的事件处理函数
      CodeMirror.on(inp, "keydown", function(e) {
        // 如果选项中有 onKeyDown 回调函数，则在按键按下时触发该回调函数
        if (options && options.onKeyDown && options.onKeyDown(e, inp.value, close)) { return; }
        // 如果按下的是 ESC 键或者按下的是回车键且选项中没有设置不关闭，则让输入框失去焦点并关闭对话框
        if (e.keyCode == 27 || (options.closeOnEnter !== false && e.keyCode == 13)) {
          inp.blur();
          CodeMirror.e_stop(e);
          close();
        }
        // 如果按下的是回车键，则调用回调函数，并传入输入框的值和事件对象
        if (e.keyCode == 13) callback(inp.value, e);
      });

      // 如果选项中没有设置不关闭，则在对话框失去焦点时关闭对话框
      if (options.closeOnBlur !== false) CodeMirror.on(dialog, "focusout", function (evt) {
        if (evt.relatedTarget !== null) close();
      });
    } 
    // 如果输入框不存在，且对话框中有按钮
    else if (button = dialog.getElementsByTagName("button")[0]) {
      // 在按钮点击时关闭对话框并让编辑器获得焦点
      CodeMirror.on(button, "click", function() {
        close();
        me.focus();
      });

      // 如果选项中没有设置不关闭，则在按钮失去焦点时关闭对话框
      if (options.closeOnBlur !== false) CodeMirror.on(button, "blur", close);

      // 让按钮获得焦点
      button.focus();
    }
    // 返回关闭函数
    return close;
  });

  // 定义一个名为 openConfirm 的编辑器扩展方法
  CodeMirror.defineExtension("openConfirm", function(template, callbacks, options) {
    // 关闭当前通知
    closeNotification(this, null);
    // 创建对话框
    var dialog = dialogDiv(this, template, options && options.bottom);
    var buttons = dialog.getElementsByTagName("button");
    var closed = false, me = this, blurring = 1;
    // 定义关闭函数
    function close() {
      if (closed) return;
      closed = true;
      // 移除对话框的打开样式，并移除对话框
      CodeMirror.rmClass(dialog.parentNode, 'dialog-opened');
      dialog.parentNode.removeChild(dialog);
      // 让编辑器获得焦点
      me.focus();
    }
    // 让第一个按钮获得焦点
    buttons[0].focus();
    // 遍历按钮数组
    for (var i = 0; i < buttons.length; ++i) {
      // 获取当前按钮
      var b = buttons[i];
      // 创建闭包，将回调函数作为参数传入
      (function(callback) {
        // 给按钮添加点击事件监听器
        CodeMirror.on(b, "click", function(e) {
          // 阻止默认事件
          CodeMirror.e_preventDefault(e);
          // 关闭通知
          close();
          // 如果存在回调函数，则执行回调函数
          if (callback) callback(me);
        });
      })(callbacks[i]);
      // 给按钮添加失焦事件监听器
      CodeMirror.on(b, "blur", function() {
        // 减少失焦计数
        --blurring;
        // 设置定时器，在200ms后如果失焦计数小于等于0，则关闭通知
        setTimeout(function() { if (blurring <= 0) close(); }, 200);
      });
      // 给按钮添加获取焦点事件监听器
      CodeMirror.on(b, "focus", function() { ++blurring; });
    }
  });

  /*
   * openNotification
   * 打开一个通知，可以使用可选的定时器关闭（默认5000ms定时器），并且始终在点击时关闭。
   *
   * 如果在打开另一个通知时已经有一个通知打开，则会关闭当前打开的通知，并立即打开新的通知。
   */
  CodeMirror.defineExtension("openNotification", function(template, options) {
    // 关闭通知
    closeNotification(this, close);
    // 创建对话框
    var dialog = dialogDiv(this, template, options && options.bottom);
    var closed = false, doneTimer;
    // 设置持续时间，默认为5000ms
    var duration = options && typeof options.duration !== "undefined" ? options.duration : 5000;

    function close() {
      // 如果已关闭，则直接返回
      if (closed) return;
      // 设置已关闭标志
      closed = true;
      // 清除定时器
      clearTimeout(doneTimer);
      // 移除对话框的打开样式类，并移除对话框
      CodeMirror.rmClass(dialog.parentNode, 'dialog-opened');
      dialog.parentNode.removeChild(dialog);
    }

    // 给对话框添加点击事件监听器
    CodeMirror.on(dialog, 'click', function(e) {
      // 阻止默认事件
      CodeMirror.e_preventDefault(e);
      // 关闭通知
      close();
    });

    // 如果设置了持续时间，则设置定时器，在持续时间后关闭通知
    if (duration)
      doneTimer = setTimeout(close, duration);

    // 返回关闭函数
    return close;
  });
# 闭合了一个代码块或者函数的结束
```