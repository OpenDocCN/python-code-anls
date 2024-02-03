# `stable-diffusion-webui\javascript\inputAccordion.js`

```
function inputAccordionChecked(id, checked) {
    // 根据 id 获取对应的折叠面板元素
    var accordion = gradioApp().getElementById(id);
    // 设置折叠面板的可见复选框状态为 checked
    accordion.visibleCheckbox.checked = checked;
    // 调用折叠面板的可见复选框改变事件处理函数
    accordion.onVisibleCheckboxChange();
}

function setupAccordion(accordion) {
    // 获取折叠面板中的标签包裹元素
    var labelWrap = accordion.querySelector('.label-wrap');
    // 获取折叠面板中的 gradio 复选框元素
    var gradioCheckbox = gradioApp().querySelector('#' + accordion.id + "-checkbox input");
    // 获取折叠面板中的额外元素
    var extra = gradioApp().querySelector('#' + accordion.id + "-extra");
    // 获取标签包裹元素中的 span 元素
    var span = labelWrap.querySelector('span');
    // 设置 linked 变量为 true
    var linked = true;

    // 定义 isOpen 函数，用于判断折叠面板是否展开
    var isOpen = function() {
        return labelWrap.classList.contains('open');
    };

    // 创建 MutationObserver 实例，用于监听标签包裹元素的 class 属性变化
    var observerAccordionOpen = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutationRecord) {
            // 根据折叠面板是否展开来切换 input-accordion-open 类的存在
            accordion.classList.toggle('input-accordion-open', isOpen());

            if (linked) {
                // 如果 linked 为 true，则设置折叠面板的可见复选框状态为折叠面板是否展开
                accordion.visibleCheckbox.checked = isOpen();
                // 调用折叠面板的可见复选框改变事件处理函数
                accordion.onVisibleCheckboxChange();
            }
        });
    });
    // 开始观察标签包裹元素的 class 属性变化
    observerAccordionOpen.observe(labelWrap, {attributes: true, attributeFilter: ['class']});

    // 如果存在额外元素，则将其插入到标签包裹元素中
    if (extra) {
        labelWrap.insertBefore(extra, labelWrap.lastElementChild);
    }

    // 定义折叠面板的 onChecked 方法，用于处理折叠面板的展开状态
    accordion.onChecked = function(checked) {
        if (isOpen() != checked) {
            // 如果折叠面板的展开状态与传入的 checked 参数不一致，则模拟点击标签包裹元素
            labelWrap.click();
        }
    };

    // 创建一个可见复选框元素，并设置相关属性
    var visibleCheckbox = document.createElement('INPUT');
    visibleCheckbox.type = 'checkbox';
    visibleCheckbox.checked = isOpen();
    visibleCheckbox.id = accordion.id + "-visible-checkbox";
    visibleCheckbox.className = gradioCheckbox.className + " input-accordion-checkbox";
    // 将可见复选框插入到 span 元素中
    span.insertBefore(visibleCheckbox, span.firstChild);

    // 将可见复选框元素赋值给折叠面板的 visibleCheckbox 属性
    accordion.visibleCheckbox = visibleCheckbox;
    // 定义折叠面板的可见复选框改变事件处理函数
    accordion.onVisibleCheckboxChange = function() {
        if (linked && isOpen() != visibleCheckbox.checked) {
            // 如果 linked 为 true 且折叠面板的展开状态与可见复选框状态不一致，则模拟点击标签包裹元素
            labelWrap.click();
        }

        // 设置 gradio 复选框的状态为可见复选框的状态，并更新输入
        gradioCheckbox.checked = visibleCheckbox.checked;
        updateInput(gradioCheckbox);
    };
    // 添加点击事件监听器，当复选框被点击时执行回调函数
    visibleCheckbox.addEventListener('click', function(event) {
        // 设置 linked 变量为 false
        linked = false;
        // 阻止事件冒泡
        event.stopPropagation();
    });
    // 添加输入事件监听器，当复选框的值发生改变时执行 accordion.onVisibleCheckboxChange 函数
    visibleCheckbox.addEventListener('input', accordion.onVisibleCheckboxChange);
# 当页面加载完成后执行的回调函数
onUiLoaded(function() {
    # 遍历所有类名为 'input-accordion' 的元素
    for (var accordion of gradioApp().querySelectorAll('.input-accordion')) {
        # 对每个 'input-accordion' 元素进行设置折叠功能
        setupAccordion(accordion);
    }
});
```