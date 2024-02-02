# `ZeroNet\plugins\Sidebar\media\Scrollable.js`

```py
// 初始化滚动条功能
window.initScrollable = function () {

    // 获取滚动容器
    var scrollContainer = document.querySelector('.scrollable'),
        // 获取滚动内容包裹器
        scrollContentWrapper = document.querySelector('.scrollable .content-wrapper'),
        // 获取滚动内容
        scrollContent = document.querySelector('.scrollable .content'),
        // 内容位置
        contentPosition = 0,
        // 是否正在拖动滚动条
        scrollerBeingDragged = false,
        // 滚动条
        scroller,
        // 顶部位置
        topPosition,
        // 滚动条高度
        scrollerHeight;

    // 计算滚动条高度
    function calculateScrollerHeight() {
        // 计算滚动条应该有多高
        var visibleRatio = scrollContainer.offsetHeight / scrollContentWrapper.scrollHeight;
        if (visibleRatio == 1)
            scroller.style.display = "none";
        else
            scroller.style.display = "block";
        return visibleRatio * scrollContainer.offsetHeight;
    }

    // 移动滚动条
    function moveScroller(evt) {
        // 将滚动条移动到顶部偏移
        var scrollPercentage = evt.target.scrollTop / scrollContentWrapper.scrollHeight;
        topPosition = scrollPercentage * (scrollContainer.offsetHeight - 5); // 5px 的任意偏移，以防滚动条移动超出内容包裹器的边界框
        scroller.style.top = topPosition + 'px';
    }

    // 开始拖动
    function startDrag(evt) {
        normalizedPosition = evt.pageY;
        contentPosition = scrollContentWrapper.scrollTop;
        scrollerBeingDragged = true;
        window.addEventListener('mousemove', scrollBarScroll);
        return false;
    }

    // 停止拖动
    function stopDrag(evt) {
        scrollerBeingDragged = false;
        window.removeEventListener('mousemove', scrollBarScroll);
    }
    function scrollBarScroll(evt) {
        // 如果正在拖动滚动条，则阻止默认事件
        if (scrollerBeingDragged === true) {
            evt.preventDefault();
            // 计算鼠标位置和滚动条位置的差值
            var mouseDifferential = evt.pageY - normalizedPosition;
            // 计算滚动条等效的滚动距离
            var scrollEquivalent = mouseDifferential * (scrollContentWrapper.scrollHeight / scrollContainer.offsetHeight);
            // 设置滚动内容的滚动位置
            scrollContentWrapper.scrollTop = contentPosition + scrollEquivalent;
        }
    }

    function updateHeight() {
        // 计算滚动条高度并减去10像素
        scrollerHeight = calculateScrollerHeight() - 10;
        // 设置滚动条的高度
        scroller.style.height = scrollerHeight + 'px';
    }

    function createScroller() {
        // 创建滚动条元素并添加到'.scrollable' div中
        // 创建滚动条元素
        scroller = document.createElement("div");
        scroller.className = 'scroller';

        // 根据内容确定滚动条的大小
        scrollerHeight = calculateScrollerHeight() - 10;

        if (scrollerHeight / scrollContainer.offsetHeight < 1) {
            // 如果需要根据内容大小设置滚动条
            scroller.style.height = scrollerHeight + 'px';

            // 将滚动条添加到scrollContainer div中
            scrollContainer.appendChild(scroller);

            // 显示滚动路径
            scrollContainer.className += ' showScroll';

            // 添加相关的可拖动监听器
            scroller.addEventListener('mousedown', startDrag);
            window.addEventListener('mouseup', stopDrag);
        }

    }

    createScroller();


    // 监听器
    scrollContentWrapper.addEventListener('scroll', moveScroller);

    return updateHeight;
# 代码结尾的分号，可能是用于结束一个代码块或语句的标记
```