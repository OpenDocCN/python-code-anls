# `stable-diffusion-webui\javascript\settings.js`

```
// 定义一个对象，用于存储需要排除在显示所有选项卡之外的选项卡
let settingsExcludeTabsFromShowAll = {
    settings_tab_defaults: 1,
    settings_tab_sysinfo: 1,
    settings_tab_actions: 1,
    settings_tab_licenses: 1,
};

// 显示所有选项卡的函数
function settingsShowAllTabs() {
    // 遍历所有选项卡元素
    gradioApp().querySelectorAll('#settings > div').forEach(function(elem) {
        // 如果该选项卡在排除列表中，则跳过
        if (settingsExcludeTabsFromShowAll[elem.id]) return;

        // 显示该选项卡
        elem.style.display = "block";
    });
}

// 显示一个选项卡的函数
function settingsShowOneTab() {
    // 点击显示一个选项卡的按钮
    gradioApp().querySelector('#settings_show_one_page').click();
}

// 当 UI 加载完成时执行的函数
onUiLoaded(function() {
    // 获取搜索框元素
    var edit = gradioApp().querySelector('#settings_search');
    // 获取搜索框中的输入框元素
    var editTextarea = gradioApp().querySelector('#settings_search > label > input');
    // 获取显示所有页面按钮元素
    var buttonShowAllPages = gradioApp().getElementById('settings_show_all_pages');
    // 获取设置选项卡元素
    var settings_tabs = gradioApp().querySelector('#settings div');

    // 当搜索框内容发生变化时执行的函数
    onEdit('settingsSearch', editTextarea, 250, function() {
        // 获取搜索框中的文本内容并转换为小写
        var searchText = (editTextarea.value || "").trim().toLowerCase();

        // 遍历所有设置选项卡中的元素
        gradioApp().querySelectorAll('#settings > div[id^=settings_] div[id^=column_settings_] > *').forEach(function(elem) {
            // 判断元素的文本内容是否包含搜索文本
            var visible = elem.textContent.trim().toLowerCase().indexOf(searchText) != -1;
            // 根据搜索结果显示或隐藏元素
            elem.style.display = visible ? "" : "none";
        });

        // 根据搜索框内容是否为空来显示所有选项卡或显示一个选项卡
        if (searchText != "") {
            settingsShowAllTabs();
        } else {
            settingsShowOneTab();
        }
    });

    // 将搜索框插入到设置选项卡元素中的第一个位置
    settings_tabs.insertBefore(edit, settings_tabs.firstChild);
    // 将显示所有页面按钮添加到设置选项卡元素的末尾
    settings_tabs.appendChild(buttonShowAllPages);

    // 点击显示所有页面按钮时执行显示所有选项卡的函数
    buttonShowAllPages.addEventListener("click", settingsShowAllTabs);
});

// 当选项发生变化时执行的函数
onOptionsChanged(function() {
    // 如果存在设置类别元素，则返回
    if (gradioApp().querySelector('#settings .settings-category')) return;

    // 创建一个映射对象，用于存储选项卡名称和对应的按钮元素
    var sectionMap = {};
    // 遍历所有按钮元素
    gradioApp().querySelectorAll('#settings > div > button').forEach(function(x) {
        // 将按钮文本内容作为键，按钮元素作为值存储到映射对象中
        sectionMap[x.textContent.trim()] = x;
    });
    # 遍历选项的类别数组，对每个元素执行指定的函数
    opts._categories.forEach(function(x) {
        # 获取当前元素的第一个值作为 section
        var section = x[0];
        # 获取当前元素的第二个值作为 category
        var category = x[1];

        # 创建一个新的 SPAN 元素
        var span = document.createElement('SPAN');
        # 设置 SPAN 元素的文本内容为 category
        span.textContent = category;
        # 设置 SPAN 元素的类名为 'settings-category'
        span.className = 'settings-category';

        # 获取 sectionMap 中对应 section 的元素
        var sectionElem = sectionMap[section];
        # 如果 sectionElem 不存在，则返回
        if (!sectionElem) return;

        # 在 sectionElem 的父元素之前插入新创建的 span 元素
        sectionElem.parentElement.insertBefore(span, sectionElem);
    });
# 结束一个 JavaScript 函数或代码块
});
```