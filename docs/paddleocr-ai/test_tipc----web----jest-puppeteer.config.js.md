# `.\PaddleOCR\test_tipc\web\jest-puppeteer.config.js`

```py
// 配置文件，用于设置 Jest Puppeteer 的相关配置
module.exports = {
    // 启动配置，设置浏览器是否为无头模式，以及使用的浏览器产品
    launch: {
        headless: false,
        product: 'chrome'
    },
    // 浏览器上下文设置为默认值
    browserContext: 'default',
    // 服务器配置，设置启动命令、端口号、启动超时时间和是否启用调试模式
    server: {
        command: 'python3 -m http.server 9811',
        port: 9811,
        launchTimeout: 10000,
        debug: true
    }
};
```