# `stable-diffusion-webui\javascript\ui_settings_hints.js`

```py
// 设置提示和额外信息的变量
var settingsHintsSetup = false;

// 当选项改变时执行的函数
onOptionsChanged(function() {
    // 如果设置提示已经设置过，则直接返回
    if (settingsHintsSetup) return;
    // 设置提示已经设置过的标志为 true
    settingsHintsSetup = true;

    // 获取所有以'setting_'开头的元素
    gradioApp().querySelectorAll('#settings [id^=setting_]').forEach(function(div) {
        // 获取设置的名称
        var name = div.id.substr(8);
        // 获取设置前的注释和设置后的注释
        var commentBefore = opts._comments_before[name];
        var commentAfter = opts._comments_after[name];

        // 如果没有设置前的注释和设置后的注释，则直接返回
        if (!commentBefore && !commentAfter) return;

        // 初始化 span 变量
        var span = null;
        // 根据不同的类别找到对应的 span 元素
        if (div.classList.contains('gradio-checkbox')) span = div.querySelector('label span');
        else if (div.classList.contains('gradio-checkboxgroup')) span = div.querySelector('span').firstChild;
        else if (div.classList.contains('gradio-radio')) span = div.querySelector('span').firstChild;
        else span = div.querySelector('label span').firstChild;

        // 如果找不到 span 元素，则直接返回
        if (!span) return;

        // 如果有设置前的注释，则创建一个包含注释内容的 DIV 元素，并插入到 span 元素之前
        if (commentBefore) {
            var comment = document.createElement('DIV');
            comment.className = 'settings-comment';
            comment.innerHTML = commentBefore;
            span.parentElement.insertBefore(document.createTextNode('\xa0'), span);
            span.parentElement.insertBefore(comment, span);
            span.parentElement.insertBefore(document.createTextNode('\xa0'), span);
        }
        // 如果有设置后的注释，则创建一个包含注释内容的 DIV 元素，并插入到 span 元素之后
        if (commentAfter) {
            comment = document.createElement('DIV');
            comment.className = 'settings-comment';
            comment.innerHTML = commentAfter;
            span.parentElement.insertBefore(comment, span.nextSibling);
            span.parentElement.insertBefore(document.createTextNode('\xa0'), span.nextSibling);
        }
    });
});

// 显示快速设置提示的函数
function settingsHintsShowQuicksettings() {
    // 发送 GET 请求到指定路径，获取数据并执行回调函数
    requestGet("./internal/quicksettings-hint", {}, function(data) {
        // 创建一个表格元素
        var table = document.createElement('table');
        // 设置表格的类名为 'popup-table'
        table.className = 'popup-table';

        // 遍历数据数组中的每个对象
        data.forEach(function(obj) {
            // 创建一个表格行元素
            var tr = document.createElement('tr');
            // 创建一个表格数据元素
            var td = document.createElement('td');
            // 设置表格数据的文本内容为对象的名称
            td.textContent = obj.name;
            // 将表格数据添加到表格行中
            tr.appendChild(td);

            // 创建另一个表格数据元素
            td = document.createElement('td');
            // 设置表格数据的文本内容为对象的标签
            td.textContent = obj.label;
            // 将表格数据添加到表格行中
            tr.appendChild(td);

            // 将表格行添加到表格中
            table.appendChild(tr);
        });

        // 在页面上弹出包含表格的弹出框
        popup(table);
    });
# 闭合之前的代码块
```