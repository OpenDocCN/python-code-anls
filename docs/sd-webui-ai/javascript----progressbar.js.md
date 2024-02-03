# `stable-diffusion-webui\javascript\progressbar.js`

```
// 记录图库选择的内容
function rememberGallerySelection() {

}

// 获取图库中选中的索引
function getGallerySelectedIndex() {

}

// 发送 POST 请求到指定 URL，处理成功和失败的回调
function request(url, data, handler, errorHandler) {
    // 创建 XMLHttpRequest 对象
    var xhr = new XMLHttpRequest();
    // 打开连接
    xhr.open("POST", url, true);
    // 设置请求头
    xhr.setRequestHeader("Content-Type", "application/json");
    // 监听状态变化
    xhr.onreadystatechange = function() {
        // 请求完成
        if (xhr.readyState === 4) {
            // 请求成功
            if (xhr.status === 200) {
                try {
                    // 解析返回的 JSON 数据
                    var js = JSON.parse(xhr.responseText);
                    // 调用成功处理函数
                    handler(js);
                } catch (error) {
                    // 捕获异常并输出错误信息
                    console.error(error);
                    // 调用错误处理函数
                    errorHandler();
                }
            } else {
                // 请求失败，调用错误处理函数
                errorHandler();
            }
        }
    };
    // 将数据转换为 JSON 字符串
    var js = JSON.stringify(data);
    // 发送请求
    xhr.send(js);
}

// 将数字补齐为两位数
function pad2(x) {
    return x < 10 ? '0' + x : x;
}

// 格式化时间为 HH:MM:SS 或 MM:SS 或 秒
function formatTime(secs) {
    if (secs > 3600) {
        return pad2(Math.floor(secs / 60 / 60)) + ":" + pad2(Math.floor(secs / 60) % 60) + ":" + pad2(Math.floor(secs) % 60);
    } else if (secs > 60) {
        return pad2(Math.floor(secs / 60)) + ":" + pad2(Math.floor(secs) % 60);
    } else {
        return Math.floor(secs) + "s";
    }
}

// 设置页面标题，包括进度信息
function setTitle(progress) {
    var title = 'Stable Diffusion';

    if (opts.show_progress_in_title && progress) {
        title = '[' + progress.trim() + '] ' + title;
    }

    if (document.title != title) {
        document.title = title;
    }
}

// 生成随机 ID
function randomId() {
    return "task(" + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + ")";
}

// 开始向 "/internal/progress" URI 发送进度请求，创建进度条和预览，任务结束时清理所有创建的内容并调用 atEnd 函数，每次有进度更新时调用 onProgress
# 定义一个函数，用于处理任务的进度更新
function requestProgress(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout = 40) {
    # 记录任务开始时间
    var dateStart = new Date();
    # 标记是否曾经活跃过
    var wasEverActive = false;
    # 获取进度条容器的父节点
    var parentProgressbar = progressbarContainer.parentNode;

    # 创建一个用于显示进度的 div 元素
    var divProgress = document.createElement('div');
    divProgress.className = 'progressDiv';
    divProgress.style.display = opts.show_progressbar ? "block" : "none";
    # 创建一个内部的 div 元素用于显示进度条
    var divInner = document.createElement('div');
    divInner.className = 'progress';

    # 将内部的 div 元素添加到进度条 div 元素中
    divProgress.appendChild(divInner);
    # 将进度条 div 元素插入到进度条容器之前
    parentProgressbar.insertBefore(divProgress, progressbarContainer);

    # 初始化 livePreview 变量为 null
    var livePreview = null;

    # 定义一个函数，用于移除进度条
    var removeProgressBar = function() {
        # 如果进度条不存在，则直接返回
        if (!divProgress) return;

        # 设置标题为空
        setTitle("");
        # 从父节点中移除进度条
        parentProgressbar.removeChild(divProgress);
        # 如果存在 gallery 和 livePreview，则从 gallery 中移除 livePreview
        if (gallery && livePreview) gallery.removeChild(livePreview);
        # 调用结束回调函数
        atEnd();

        # 将进度条设为 null
        divProgress = null;
    };
    # 定义一个名为funProgress的函数，用于更新任务进度条
    var funProgress = function(id_task) {
        # 发送请求到指定路径，获取任务进度信息
        request("./internal/progress", {id_task: id_task, live_preview: false}, function(res) {
            # 如果任务已完成，则移除进度条并返回
            if (res.completed) {
                removeProgressBar();
                return;
            }

            # 初始化进度文本为空字符串
            let progressText = "";

            # 设置进度条的宽度和背景颜色
            divInner.style.width = ((res.progress || 0) * 100.0) + '%';
            divInner.style.background = res.progress ? "" : "transparent";

            # 如果进度大于0，则更新进度文本
            if (res.progress > 0) {
                progressText = ((res.progress || 0) * 100.0).toFixed(0) + '%';
            }

            # 如果存在预计剩余时间，则添加到进度文本中
            if (res.eta) {
                progressText += " ETA: " + formatTime(res.eta);
            }

            # 设置页面标题为进度文本
            setTitle(progressText);

            # 如果存在文本信息且不包含换行符，则将文本信息添加到进度文本中
            if (res.textinfo && res.textinfo.indexOf("\n") == -1) {
                progressText = res.textinfo + " " + progressText;
            }

            # 设置进度条的文本内容为进度文本
            divInner.textContent = progressText;

            # 计算从开始到现在的经过时间
            var elapsedFromStart = (new Date() - dateStart) / 1000;

            # 如果任务仍在进行中，则标记为活跃状态
            if (res.active) wasEverActive = true;

            # 如果任务不再活跃且之前有活跃状态，则移除进度条并返回
            if (!res.active && wasEverActive) {
                removeProgressBar();
                return;
            }

            # 如果经过时间超过不活动超时时间且任务不在队列中且不再活跃，则移除进度条并返回
            if (elapsedFromStart > inactivityTimeout && !res.queued && !res.active) {
                removeProgressBar();
                return;
            }

            # 如果存在进度回调函数，则调用该函数并传入进度信息
            if (onProgress) {
                onProgress(res);
            }

            # 设置定时器，定时调用funProgress函数以更新进度条
            setTimeout(() => {
                funProgress(id_task, res.id_live_preview);
            }, opts.live_preview_refresh_period || 500);
        }, function() {
            # 如果请求失败，则移除进度条
            removeProgressBar();
        });
    };
    // 定义一个名为funLivePreview的函数，用于实时预览任务进度
    var funLivePreview = function(id_task, id_live_preview) {
        // 发送请求获取任务进度信息
        request("./internal/progress", {id_task: id_task, id_live_preview: id_live_preview}, function(res) {
            // 如果divProgress不存在，则返回
            if (!divProgress) {
                return;
            }

            // 如果返回的结果包含实时预览信息并且gallery存在
            if (res.live_preview && gallery) {
                // 创建一个新的图片对象
                var img = new Image();
                // 当图片加载完成时执行以下操作
                img.onload = function() {
                    // 如果livePreview不存在，则创建一个div元素并添加到gallery中
                    if (!livePreview) {
                        livePreview = document.createElement('div');
                        livePreview.className = 'livePreview';
                        gallery.insertBefore(livePreview, gallery.firstElementChild);
                    }

                    // 将图片添加到livePreview中
                    livePreview.appendChild(img);
                    // 如果livePreview中子元素数量大于2，则移除第一个子元素
                    if (livePreview.childElementCount > 2) {
                        livePreview.removeChild(livePreview.firstElementChild);
                    }
                };
                // 设置图片的src属性为实时预览的URL
                img.src = res.live_preview;
            }

            // 每隔一段时间执行funLivePreview函数，实现实时更新
            setTimeout(() => {
                funLivePreview(id_task, res.id_live_preview);
            }, opts.live_preview_refresh_period || 500);
        }, function() {
            // 请求失败时移除进度条
            removeProgressBar();
        });
    };

    // 调用funProgress函数，显示任务进度
    funProgress(id_task, 0);

    // 如果gallery存在，则调用funLivePreview函数，实现实时预览
    if (gallery) {
        funLivePreview(id_task, 0);
    }
# 闭合之前的代码块
```