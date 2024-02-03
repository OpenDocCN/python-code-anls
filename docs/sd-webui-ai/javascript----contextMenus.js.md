# `stable-diffusion-webui\javascript\contextMenus.js`

```
# 初始化右键菜单
var contextMenuInit = function() {
    # 标记事件监听器是否已经应用
    let eventListenerApplied = false;
    # 创建一个存储菜单规格的 Map 对象
    let menuSpecs = new Map();

    # 生成一个唯一的 ID
    const uid = function() {
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    };
    // 显示右键菜单
    function showContextMenu(event, element, menuEntries) {
        // 计算鼠标点击位置的坐标
        let posx = event.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
        let posy = event.clientY + document.body.scrollTop + document.documentElement.scrollTop;

        // 获取已存在的右键菜单并移除
        let oldMenu = gradioApp().querySelector('#context-menu');
        if (oldMenu) {
            oldMenu.remove();
        }

        // 创建右键菜单元素
        let baseStyle = window.getComputedStyle(uiCurrentTab);
        const contextMenu = document.createElement('nav');
        contextMenu.id = "context-menu";
        contextMenu.style.background = baseStyle.background;
        contextMenu.style.color = baseStyle.color;
        contextMenu.style.fontFamily = baseStyle.fontFamily;
        contextMenu.style.top = posy + 'px';
        contextMenu.style.left = posx + 'px';

        // 创建右键菜单项列表
        const contextMenuList = document.createElement('ul');
        contextMenuList.className = 'context-menu-items';
        contextMenu.append(contextMenuList);

        // 遍历菜单项数组，创建菜单项并添加点击事件
        menuEntries.forEach(function(entry) {
            let contextMenuEntry = document.createElement('a');
            contextMenuEntry.innerHTML = entry['name'];
            contextMenuEntry.addEventListener("click", function() {
                entry['func']();
            });
            contextMenuList.append(contextMenuEntry);
        });

        // 将右键菜单添加到页面
        gradioApp().appendChild(contextMenu);

        // 计算右键菜单的宽度和高度
        let menuWidth = contextMenu.offsetWidth + 4;
        let menuHeight = contextMenu.offsetHeight + 4;

        // 获取窗口的宽度和高度
        let windowWidth = window.innerWidth;
        let windowHeight = window.innerHeight;

        // 如果右键菜单超出窗口边界，调整位置
        if ((windowWidth - posx) < menuWidth) {
            contextMenu.style.left = windowWidth - menuWidth + "px";
        }

        if ((windowHeight - posy) < menuHeight) {
            contextMenu.style.top = windowHeight - menuHeight + "px";
        }
    }
    // 向上下文菜单选项中添加新的选项
    function appendContextMenuOption(targetElementSelector, entryName, entryFunction) {

        // 获取目标元素的当前菜单选项
        var currentItems = menuSpecs.get(targetElementSelector);

        // 如果当前菜单选项不存在，则创建一个空数组
        if (!currentItems) {
            currentItems = [];
            menuSpecs.set(targetElementSelector, currentItems);
        }
        // 创建新的菜单选项对象
        let newItem = {
            id: targetElementSelector + '_' + uid(),
            name: entryName,
            func: entryFunction,
            isNew: true
        };

        // 将新的菜单选项添加到当前菜单选项数组中
        currentItems.push(newItem);
        // 返回新菜单选项的 ID
        return newItem['id'];
    }

    // 从上下文菜单选项中移除指定 ID 的选项
    function removeContextMenuOption(uid) {
        // 遍历所有菜单选项
        menuSpecs.forEach(function(v) {
            let index = -1;
            // 查找指定 ID 的菜单选项并记录索引
            v.forEach(function(e, ei) {
                if (e['id'] == uid) {
                    index = ei;
                }
            });
            // 如果找到指定 ID 的菜单选项，则移除
            if (index >= 0) {
                v.splice(index, 1);
            }
        });
    }

    // 添加上下文菜单事件监听器
    function addContextMenuEventListener() {
        // 如果事件监听器已经应用，则直接返回
        if (eventListenerApplied) {
            return;
        }
        // 添加点击事件监听器，用于移除上下文菜单
        gradioApp().addEventListener("click", function(e) {
            if (!e.isTrusted) {
                return;
            }

            let oldMenu = gradioApp().querySelector('#context-menu');
            if (oldMenu) {
                oldMenu.remove();
            }
        });
        // 添加右键菜单事件监听器，根据目标元素显示对应的上下文菜单
        gradioApp().addEventListener("contextmenu", function(e) {
            let oldMenu = gradioApp().querySelector('#context-menu');
            if (oldMenu) {
                oldMenu.remove();
            }
            // 遍历所有菜单选项，根据目标元素匹配显示对应的上下文菜单
            menuSpecs.forEach(function(v, k) {
                if (e.composedPath()[0].matches(k)) {
                    showContextMenu(e, e.composedPath()[0], v);
                    e.preventDefault();
                }
            });
        });
        // 标记事件监听器已应用
        eventListenerApplied = true;

    }

    // 返回上下文菜单操作函数数组
    return [appendContextMenuOption, removeContextMenuOption, addContextMenuEventListener];
};

var initResponse = contextMenuInit();
var appendContextMenuOption = initResponse[0];
var removeContextMenuOption = initResponse[1];
var addContextMenuEventListener = initResponse[2];

(function() {
    //Start example Context Menu Items

    // 定义一个函数，用于生成重复点击事件
    let generateOnRepeat = function(genbuttonid, interruptbuttonid) {
        // 获取生成按钮和中断按钮的元素
        let genbutton = gradioApp().querySelector(genbuttonid);
        let interruptbutton = gradioApp().querySelector(interruptbuttonid);
        // 如果中断按钮不可见，则点击生成按钮
        if (!interruptbutton.offsetParent) {
            genbutton.click();
        }
        // 清除之前的定时器，设置新的定时器，每500毫秒点击一次生成按钮
        clearInterval(window.generateOnRepeatInterval);
        window.generateOnRepeatInterval = setInterval(function() {
            if (!interruptbutton.offsetParent) {
                genbutton.click();
            }
        },
        500);
    };

    // 定义生成文本到图像的重复点击事件
    let generateOnRepeat_txt2img = function() {
        generateOnRepeat('#txt2img_generate', '#txt2img_interrupt');
    };

    // 定义生成图像到图像的重复点击事件
    let generateOnRepeat_img2img = function() {
        generateOnRepeat('#img2img_generate', '#img2img_interrupt');
    };

    // 添加生成文本到图像的重复点击事件到右键菜单
    appendContextMenuOption('#txt2img_generate', 'Generate forever', generateOnRepeat_txt2img);
    appendContextMenuOption('#txt2img_interrupt', 'Generate forever', generateOnRepeat_txt2img);
    // 添加生成图像到图像的重复点击事件到右键菜单
    appendContextMenuOption('#img2img_generate', 'Generate forever', generateOnRepeat_img2img);
    appendContextMenuOption('#img2img_interrupt', 'Generate forever', generateOnRepeat_img2img);

    // 取消生成 forever 的事件处理函数
    let cancelGenerateForever = function() {
        clearInterval(window.generateOnRepeatInterval);
    };

    // 添加取消生成 forever 的事件到右键菜单
    appendContextMenuOption('#txt2img_interrupt', 'Cancel generate forever', cancelGenerateForever);
    appendContextMenuOption('#txt2img_generate', 'Cancel generate forever', cancelGenerateForever);
    appendContextMenuOption('#img2img_interrupt', 'Cancel generate forever', cancelGenerateForever);
    appendContextMenuOption('#img2img_generate', 'Cancel generate forever', cancelGenerateForever);

})();
//End example Context Menu Items
# 在 UI 更新后添加右键菜单事件监听器
onAfterUiUpdate(addContextMenuEventListener);
```