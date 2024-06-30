# `D:\src\scipysrc\seaborn\doc\_static\copybutton.js`

```
// 当文档加载完毕后执行的函数
$(document).ready(function() {
    /* 在代码示例的右上角添加一个按钮 [>>>]，用于隐藏
     * >>> 和 ... 的提示符以及输出，使得代码可以被复制。
     * 注意：此 JS 片段来自于官方 python.org 文档网站。 */
    
    // 选择所有带有 highlight-python, highlight-python3, highlight-pycon 类的代码块
    var div = $('.highlight-python .highlight,' +
                '.highlight-python3 .highlight,' +
                '.highlight-pycon .highlight');
    var pre = div.find('pre');

    // 获取当前主题的样式并应用到父元素上
    pre.parent().parent().css('position', 'relative');
    var hide_text = '隐藏提示符和输出';
    var show_text = '显示提示符和输出';
    var border_width = pre.css('border-top-width');
    var border_style = pre.css('border-top-style');
    var border_color = pre.css('border-top-color');
    var button_styles = {
        'cursor':'pointer', 'position': 'absolute', 'top': '0', 'right': '0',
        'border-color': border_color, 'border-style': border_style,
        'border-width': border_width, 'color': border_color, 'text-size': '75%',
        'font-family': 'monospace', 'padding-left': '0.2em', 'padding-right': '0.2em'
    }

    // 对每个代码块添加按钮
    div.each(function(index) {
        var jthis = $(this);
        // 如果代码块包含 .gp 类，则添加按钮
        if (jthis.find('.gp').length > 0) {
            var button = $('<span class="copybutton">&gt;&gt;&gt;</span>');
            button.css(button_styles)
            button.attr('title', hide_text);
            jthis.prepend(button);
        }
        // 处理跟踪信息中的文本节点，将其包装在 span 元素中，以便后续处理
        jthis.find('pre:has(.gt)').contents().filter(function() {
            return ((this.nodeType == 3) && (this.data.trim().length > 0));
        }).wrap('<span>');
    });

    // 定义按钮点击时的行为
    $('.copybutton').toggle(
        function() {
            var button = $(this);
            // 隐藏提示符和输出
            button.parent().find('.go, .gp, .gt').hide();
            button.next('pre').find('.gt').nextUntil('.gp, .go').css('visibility', 'hidden');
            button.css('text-decoration', 'line-through');
            button.attr('title', show_text);
        },
        function() {
            var button = $(this);
            // 显示提示符和输出
            button.parent().find('.go, .gp, .gt').show();
            button.next('pre').find('.gt').nextUntil('.gp, .go').css('visibility', 'visible');
            button.css('text-decoration', 'none');
            button.attr('title', hide_text);
        });
});
```