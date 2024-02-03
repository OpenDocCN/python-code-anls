# `stable-diffusion-webui\javascript\notification.js`

```py
// 监视画廊并在主要图像为新时发送浏览器通知

// 上一个主要图像
let lastHeadImg = null;

// 通知按钮
let notificationButton = null;

// 在 UI 更新后执行的函数
onAfterUiUpdate(function() {
    // 如果通知按钮为空
    if (notificationButton == null) {
        // 获取请求通知按钮
        notificationButton = gradioApp().getElementById('request_notifications');

        // 如果通知按钮存在
        if (notificationButton != null) {
            // 添加点击事件监听器，请求通知权限
            notificationButton.addEventListener('click', () => {
                void Notification.requestPermission();
            }, true);
        }
    }

    // 获取画廊预览图像
    const galleryPreviews = gradioApp().querySelectorAll('div[id^="tab_"] div[id$="_results"] .thumbnail-item > img');

    // 如果画廊预览图像为空，则返回
    if (galleryPreviews == null) return;

    // 获取主要图像的 URL
    const headImg = galleryPreviews[0]?.src;

    // 如果主要图像为空或与上一个主要图像相同，则返回
    if (headImg == null || headImg == lastHeadImg) return;

    // 更新上一个主要图像
    lastHeadImg = headImg;

    // 如果通知音频可用
    const notificationAudio = gradioApp().querySelector('#audio_notification audio');
    if (notificationAudio) {
        // 设置音频音量并播放
        notificationAudio.volume = opts.notification_volume / 100.0 || 1.0;
        notificationAudio.play();
    }

    // 如果文档处于焦点状态，则返回
    if (document.hasFocus()) return;

    // 通过 Set 去重获取真实生成的图像数量
    const imgs = new Set(Array.from(galleryPreviews).map(img => img.src));

    // 创建通知
    const notification = new Notification(
        'Stable Diffusion',
        {
            body: `Generated ${imgs.size > 1 ? imgs.size - opts.return_grid : 1} image${imgs.size > 1 ? 's' : ''}`,
            icon: headImg,
            image: headImg,
        }
    );

    // 点击通知时聚焦父窗口并关闭通知
    notification.onclick = function(_) {
        parent.focus();
        this.close();
    };
});
```