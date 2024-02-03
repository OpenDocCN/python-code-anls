# `stable-diffusion-webui\javascript\extraNetworks.js`

```
function toggleCss(key, css, enable) {
    // 获取指定 ID 的样式元素
    var style = document.getElementById(key);
    // 如果需要启用样式且样式元素不存在，则创建新的样式元素并添加到 head 中
    if (enable && !style) {
        style = document.createElement('style');
        style.id = key;
        style.type = 'text/css';
        document.head.appendChild(style);
    }
    // 如果样式元素存在且需要禁用样式，则从 head 中移除该样式元素
    if (style && !enable) {
        document.head.removeChild(style);
    }
    // 如果样式元素存在，则清空其内容并添加新的样式内容
    if (style) {
        style.innerHTML == '';
        style.appendChild(document.createTextNode(css));
    }
}

function setupExtraNetworksForTab(tabname) {
    // 为指定标签页设置额外网络
    gradioApp().querySelector('#' + tabname + '_extra_tabs').classList.add('extra-networks');

    // 获取相关元素
    var tabs = gradioApp().querySelector('#' + tabname + '_extra_tabs > div');
    var searchDiv = gradioApp().getElementById(tabname + '_extra_search');
    var search = searchDiv.querySelector('textarea');
    var sort = gradioApp().getElementById(tabname + '_extra_sort');
    var sortOrder = gradioApp().getElementById(tabname + '_extra_sortorder');
    var refresh = gradioApp().getElementById(tabname + '_extra_refresh');
    var showDirsDiv = gradioApp().getElementById(tabname + '_extra_show_dirs');
    var showDirs = gradioApp().querySelector('#' + tabname + '_extra_show_dirs input');
    var promptContainer = gradioApp().querySelector('.prompt-container-compact#' + tabname + '_prompt_container');
    var negativePrompt = gradioApp().querySelector('#' + tabname + '_neg_prompt');

    // 将相关元素添加到标签页中
    tabs.appendChild(searchDiv);
    tabs.appendChild(sort);
    tabs.appendChild(sortOrder);
    tabs.appendChild(refresh);
    tabs.appendChild(showDirsDiv);
}
    # 定义一个函数 applyFilter，用于根据搜索条件过滤元素
    var applyFilter = function() {
        # 获取搜索框中的搜索条件，并转换为小写
        var searchTerm = search.value.toLowerCase();
    
        # 获取所有符合条件的元素，并遍历处理
        gradioApp().querySelectorAll('#' + tabname + '_extra_tabs div.card').forEach(function(elem) {
            # 获取当前元素中的搜索关键词和额外搜索条件，并转换为小写
            var searchOnly = elem.querySelector('.search_only');
            var text = elem.querySelector('.name').textContent.toLowerCase() + " " + elem.querySelector('.search_term').textContent.toLowerCase();
    
            # 判断当前元素是否包含搜索条件
            var visible = text.indexOf(searchTerm) != -1;
    
            # 如果是搜索专用元素且搜索条件长度小于4，则隐藏元素
            if (searchOnly && searchTerm.length < 4) {
                visible = false;
            }
    
            # 根据元素是否可见设置显示样式
            elem.style.display = visible ? "" : "none";
        });
    
        # 调用排序函数
        applySort();
    };
    // 定义一个函数，用于对卡片进行排序操作
    var applySort = function() {
        // 获取所有具有特定 ID 的卡片元素
        var cards = gradioApp().querySelectorAll('#' + tabname + '_extra_tabs div.card');

        // 检查是否需要反向排序
        var reverse = sortOrder.classList.contains("sortReverse");
        // 获取排序关键字并格式化
        var sortKey = sort.querySelector("input").value.toLowerCase().replace("sort", "").replaceAll(" ", "_").replace(/_+$/, "").trim() || "name";
        sortKey = "sort" + sortKey.charAt(0).toUpperCase() + sortKey.slice(1);
        // 生成排序关键字的存储值
        var sortKeyStore = sortKey + "-" + (reverse ? "Descending" : "Ascending") + "-" + cards.length;

        // 如果当前排序关键字与之前相同，则直接返回
        if (sortKeyStore == sort.dataset.sortkey) {
            return;
        }
        // 更新排序关键字的存储值
        sort.dataset.sortkey = sortKeyStore;

        // 为每个卡片元素设置原始父元素属性
        cards.forEach(function(card) {
            card.originalParentElement = card.parentElement;
        });
        // 复制卡片元素数组并根据排序关键字进行排序
        var sortedCards = Array.from(cards);
        sortedCards.sort(function(cardA, cardB) {
            var a = cardA.dataset[sortKey];
            var b = cardB.dataset[sortKey];
            if (!isNaN(a) && !isNaN(b)) {
                return parseInt(a) - parseInt(b);
            }
            return (a < b ? -1 : (a > b ? 1 : 0));
        });
        // 如果需要反向排序，则反转排序后的数组
        if (reverse) {
            sortedCards.reverse();
        }
        // 移除所有卡片元素
        cards.forEach(function(card) {
            card.remove();
        });
        // 将排序后的卡片元素添加回其原始父元素
        sortedCards.forEach(function(card) {
            card.originalParentElement.appendChild(card);
        });
    };

    // 监听搜索框输入事件，触发筛选操作
    search.addEventListener("input", applyFilter);
    // 监听排序按钮点击事件，切换排序方式并执行排序操作
    sortOrder.addEventListener("click", function() {
        sortOrder.classList.toggle("sortReverse");
        applySort();
    });
    // 执行筛选操作
    applyFilter();

    // 将排序函数和筛选函数存储到全局对象中
    extraNetworksApplySort[tabname] = applySort;
    extraNetworksApplyFilter[tabname] = applyFilter;

    // 定义一个函数，用于更新显示目录的样式
    var showDirsUpdate = function() {
        // 构建 CSS 样式字符串
        var css = '#' + tabname + '_extra_tabs .extra-network-subdirs { display: none; }';
        // 根据复选框状态切换 CSS 样式
        toggleCss(tabname + '_extra_show_dirs_style', css, !showDirs.checked);
        // 将显示目录的状态存储到本地存储中
        localSet('extra-networks-show-dirs', showDirs.checked ? 1 : 0);
    };
    # 将 showDirs.checked 设置为 localGet('extra-networks-show-dirs', 1) 等于 1 的布尔值
    showDirs.checked = localGet('extra-networks-show-dirs', 1) == 1;
    # 添加事件监听器，当 showDirs 元素发生改变时调用 showDirsUpdate 函数
    showDirs.addEventListener("change", showDirsUpdate);
    # 调用 showDirsUpdate 函数，初始化或更新 showDirs 元素的状态
    showDirsUpdate();
}

// 将提示移动到指定选项卡中
function extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt) {
    // 只适用于紧凑提示布局
    if (!gradioApp().querySelector('.toprow-compact-tools')) return;

    // 获取提示容器和提示元素
    var promptContainer = gradioApp().getElementById(tabname + '_prompt_container');
    var prompt = gradioApp().getElementById(tabname + '_prompt_row');
    var negPrompt = gradioApp().getElementById(tabname + '_neg_prompt_row');
    var elem = id ? gradioApp().getElementById(id) : null;

    // 根据条件将提示插入到指定位置
    if (showNegativePrompt && elem) {
        elem.insertBefore(negPrompt, elem.firstChild);
    } else {
        promptContainer.insertBefore(negPrompt, promptContainer.firstChild);
    }

    if (showPrompt && elem) {
        elem.insertBefore(prompt, elem.firstChild);
    } else {
        promptContainer.insertBefore(prompt, promptContainer.firstChild);
    }

    // 根据条件设置元素的类
    if (elem) {
        elem.classList.toggle('extra-page-prompts-active', showNegativePrompt || showPrompt);
    }
}

// 当用户选择一个不相关的选项卡时调用，将提示移动到指定选项卡
function extraNetworksUrelatedTabSelected(tabname) {
    extraNetworksMovePromptToTab(tabname, '', false, false);
}

// 当用户选择一个额外网络选项卡时调用，将提示移动到指定选项卡
function extraNetworksTabSelected(tabname, id, showPrompt, showNegativePrompt) {
    extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt);
}

// 应用额外网络过滤器
function applyExtraNetworkFilter(tabname) {
    setTimeout(extraNetworksApplyFilter[tabname], 1);
}

// 应用额外网络排序
function applyExtraNetworkSort(tabname) {
    setTimeout(extraNetworksApplySort[tabname], 1);
}

// 初始化额外网络过滤器和排序器
var extraNetworksApplyFilter = {};
var extraNetworksApplySort = {};
var activePromptTextarea = {};

// 设置额外网络
function setupExtraNetworks() {
    setupExtraNetworksForTab('txt2img');
    setupExtraNetworksForTab('img2img');
}
    # 定义一个函数用于注册提示框，参数为标签名和ID
    function registerPrompt(tabname, id) {
        # 在当前页面中查找指定ID的textarea元素
        var textarea = gradioApp().querySelector("#" + id + " > label > textarea");
    
        # 如果当前标签页没有激活的提示框textarea，则将当前找到的textarea设置为激活的提示框
        if (!activePromptTextarea[tabname]) {
            activePromptTextarea[tabname] = textarea;
        }
    
        # 给当前找到的textarea元素添加一个focus事件监听器，当textarea获得焦点时，将其设置为激活的提示框
        textarea.addEventListener("focus", function() {
            activePromptTextarea[tabname] = textarea;
        });
    }
    
    # 注册不同标签页下的不同提示框
    registerPrompt('txt2img', 'txt2img_prompt');
    registerPrompt('txt2img', 'txt2img_neg_prompt');
    registerPrompt('img2img', 'img2img_prompt');
    registerPrompt('img2img', 'img2img_neg_prompt');
// 当 UI 加载完成后调用 setupExtraNetworks 函数
onUiLoaded(setupExtraNetworks);

// 匹配形如 <protocol:port>text 的正则表达式
var re_extranet = /<([^:^>]+:[^:]+):[\d.]+>(.*)/;
// 全局匹配形如 <protocol:port> 的正则表达式
var re_extranet_g = /<([^:^>]+:[^:]+):[\d.]+>/g;

// 尝试从提示文本中移除额外网络信息
function tryToRemoveExtraNetworkFromPrompt(textarea, text) {
    // 匹配正则表达式 re_extranet
    var m = text.match(re_extranet);
    var replaced = false;
    var newTextareaText;
    if (m) {
        // 获取额外网络信息前的文本和额外网络信息后的文本
        var extraTextBeforeNet = opts.extra_networks_add_text_separator;
        var extraTextAfterNet = m[2];
        var partToSearch = m[1];
        var foundAtPosition = -1;
        // 替换文本中匹配到的额外网络信息
        newTextareaText = textarea.value.replaceAll(re_extranet_g, function(found, net, pos) {
            m = found.match(re_extranet);
            if (m[1] == partToSearch) {
                replaced = true;
                foundAtPosition = pos;
                return "";
            }
            return found;
        });

        // 如果找到额外网络信息位置
        if (foundAtPosition >= 0) {
            // 移除额外网络信息后的文本
            if (newTextareaText.substr(foundAtPosition, extraTextAfterNet.length) == extraTextAfterNet) {
                newTextareaText = newTextareaText.substr(0, foundAtPosition) + newTextareaText.substr(foundAtPosition + extraTextAfterNet.length);
            }
            // 移除额外网络信息前的文本
            if (newTextareaText.substr(foundAtPosition - extraTextBeforeNet.length, extraTextBeforeNet.length) == extraTextBeforeNet) {
                newTextareaText = newTextareaText.substr(0, foundAtPosition - extraTextBeforeNet.length) + newTextareaText.substr(foundAtPosition);
            }
        }
    } else {
        // 替换文本中匹配到的 text
        newTextareaText = textarea.value.replaceAll(new RegExp(text, "g"), function(found) {
            if (found == text) {
                replaced = true;
                return "";
            }
            return found;
        });
    }

    // 如果有替换操作，则更新文本框内容并返回 true，否则返回 false
    if (replaced) {
        textarea.value = newTextareaText;
        return true;
    }

    return false;
}

// 处理卡片点击事件
function cardClicked(tabname, textToAdd, allowNegativePrompt) {
    // 获取文本框元素
    var textarea = allowNegativePrompt ? activePromptTextarea[tabname] : gradioApp().querySelector("#" + tabname + "_prompt > label > textarea");
}
    # 如果无法从文本区域中删除额外的网络，则将额外的网络文本添加到文本区域中
    if (!tryToRemoveExtraNetworkFromPrompt(textarea, textToAdd)) {
        textarea.value = textarea.value + opts.extra_networks_add_text_separator + textToAdd;
    }

    # 更新输入框
    updateInput(textarea);
// 保存卡片预览的文件名，并触发更新输入框操作
function saveCardPreview(event, tabname, filename) {
    // 获取指定标签页下的文本框元素
    var textarea = gradioApp().querySelector("#" + tabname + '_preview_filename  > label > textarea');
    // 获取保存预览按钮元素
    var button = gradioApp().getElementById(tabname + '_save_preview');

    // 设置文本框的值为文件名
    textarea.value = filename;
    // 更新输入框
    updateInput(textarea);

    // 模拟点击保存按钮
    button.click();

    // 阻止事件冒泡
    event.stopPropagation();
    // 阻止默认事件
    event.preventDefault();
}

// 处理额外网络搜索按钮点击事件
function extraNetworksSearchButton(tabs_id, event) {
    // 获取搜索文本框元素
    var searchTextarea = gradioApp().querySelector("#" + tabs_id + ' > label > textarea');
    // 获取点击的按钮元素
    var button = event.target;
    // 获取按钮文本内容，如果是搜索全部按钮则为空
    var text = button.classList.contains("search-all") ? "" : button.textContent.trim();

    // 设置搜索文本框的值为按钮文本内容
    searchTextarea.value = text;
    // 更新输入框
    updateInput(searchTextarea);
}

// 全局弹出框相关函数
var globalPopup = null;
var globalPopupInner = null;

// 关闭弹出框
function closePopup() {
    // 如果弹出框不存在，则直接返回
    if (!globalPopup) return;
    // 隐藏弹出框
    globalPopup.style.display = "none";
}

// 弹出框显示内容
function popup(contents) {
    // 如果全局弹出框不存在，则创建
    if (!globalPopup) {
        globalPopup = document.createElement('div');
        globalPopup.classList.add('global-popup');

        var close = document.createElement('div');
        close.classList.add('global-popup-close');
        close.addEventListener("click", closePopup);
        close.title = "Close";
        globalPopup.appendChild(close);

        globalPopupInner = document.createElement('div');
        globalPopupInner.classList.add('global-popup-inner');
        globalPopup.appendChild(globalPopupInner);

        // 将弹出框添加到主要内容区域
        gradioApp().querySelector('.main').appendChild(globalPopup);
    }

    // 清空弹出框内部内容
    globalPopupInner.innerHTML = '';
    // 添加新内容到弹出框内部
    globalPopupInner.appendChild(contents);

    // 显示弹出框
    globalPopup.style.display = "flex";
}

// 存储弹出框内容的对象
var storedPopupIds = {};
// 根据ID弹出对应内容
function popupId(id) {
    // 如果对应ID的内容未存储，则存储
    if (!storedPopupIds[id]) {
        storedPopupIds[id] = gradioApp().getElementById(id);
    }

    // 弹出对应ID的内容
    popup(storedPopupIds[id]);
}

// 显示额外网络的元数据信息
function extraNetworksShowMetadata(text) {
    // 创建包含文本的预格式化元素
    var elem = document.createElement('pre');
    elem.classList.add('popup-metadata');
    elem.textContent = text;

    // 弹出元素内容
    popup(elem);
}
// 发起 GET 请求，传入 URL、数据、成功处理函数和错误处理函数
function requestGet(url, data, handler, errorHandler) {
    // 创建 XMLHttpRequest 对象
    var xhr = new XMLHttpRequest();
    // 将数据对象转换为 URL 查询字符串
    var args = Object.keys(data).map(function(k) {
        return encodeURIComponent(k) + '=' + encodeURIComponent(data[k]);
    }).join('&');
    // 打开 GET 请求
    xhr.open("GET", url + "?" + args, true);

    // 监听状态变化
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    // 解析响应数据为 JSON
                    var js = JSON.parse(xhr.responseText);
                    // 调用成功处理函数
                    handler(js);
                } catch (error) {
                    console.error(error);
                    // 调用错误处理函数
                    errorHandler();
                }
            } else {
                // 调用错误处理函数
                errorHandler();
            }
        }
    };
    // 将数据对象转换为 JSON 字符串并发送请求
    var js = JSON.stringify(data);
    xhr.send(js);
}

// 获取额外网络请求的元数据
function extraNetworksRequestMetadata(event, extraPage, cardName) {
    // 定义错误处理函数
    var showError = function() {
        extraNetworksShowMetadata("there was an error getting metadata");
    };

    // 发起 GET 请求获取元数据
    requestGet("./sd_extra_networks/metadata", {page: extraPage, item: cardName}, function(data) {
        if (data && data.metadata) {
            // 显示元数据
            extraNetworksShowMetadata(data.metadata);
        } else {
            // 显示错误信息
            showError();
        }
    }, showError);

    // 阻止事件冒泡
    event.stopPropagation();
}

// 编辑用户元数据
var extraPageUserMetadataEditors = {};

function extraNetworksEditUserMetadata(event, tabname, extraPage, cardName) {
    // 构建编辑器 ID
    var id = tabname + '_' + extraPage + '_edit_user_metadata';

    // 获取或创建编辑器对象
    var editor = extraPageUserMetadataEditors[id];
    if (!editor) {
        editor = {};
        editor.page = gradioApp().getElementById(id);
        editor.nameTextarea = gradioApp().querySelector("#" + id + "_name" + ' textarea');
        editor.button = gradioApp().querySelector("#" + id + "_button");
        extraPageUserMetadataEditors[id] = editor;
    }

    // 设置编辑器名称文本框的值并更新输入
    editor.nameTextarea.value = cardName;
    updateInput(editor.nameTextarea);

    // 触发按钮点击事件
    editor.button.click();

    // 弹出编辑页面
    popup(editor.page);

    // 阻止事件冒泡
    event.stopPropagation();
}
// 刷新单个卡片的额外网络信息
function extraNetworksRefreshSingleCard(page, tabname, name) {
    // 发送 GET 请求获取单个卡片的额外网络信息
    requestGet("./sd_extra_networks/get-single-card", {page: page, tabname: tabname, name: name}, function(data) {
        // 如果返回数据存在且包含 HTML 内容
        if (data && data.html) {
            // 查找指定名称的卡片元素
            var card = gradioApp().querySelector(`#${tabname}_${page.replace(" ", "_")}_cards > .card[data-name="${name}"]`);

            // 创建一个新的 DIV 元素，并将返回的 HTML 内容填充进去
            var newDiv = document.createElement('DIV');
            newDiv.innerHTML = data.html;
            var newCard = newDiv.firstElementChild;

            // 设置新卡片显示
            newCard.style.display = '';
            // 在原卡片之前插入新卡片，并移除原卡片
            card.parentElement.insertBefore(newCard, card);
            card.parentElement.removeChild(card);
        }
    });
}

// 监听键盘事件，如果按下 Escape 键，则关闭弹出窗口
window.addEventListener("keydown", function(event) {
    if (event.key == "Escape") {
        closePopup();
    }
});
```