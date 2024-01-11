# `ZeroNet\plugins\ContentFilter\media\js\ZeroFrame.js`

```
// Version 1.0.0 - Initial release
// Version 1.1.0 (2017-08-02) - Added cmdp function that returns promise instead of using callback
// Version 1.2.0 (2017-08-02) - Added Ajax monkey patch to emulate XMLHttpRequest over ZeroFrame API

// 定义 ZeroFrame 类
const CMD_INNER_READY = 'innerReady'  // 内部准备完毕命令
const CMD_RESPONSE = 'response'  // 响应命令
const CMD_WRAPPER_READY = 'wrapperReady'  // 包装器准备完毕命令
const CMD_PING = 'ping'  // ping 命令
const CMD_PONG = 'pong'  // pong 命令
const CMD_WRAPPER_OPENED_WEBSOCKET = 'wrapperOpenedWebsocket'  // 包装器打开 WebSocket 命令
const CMD_WRAPPER_CLOSE_WEBSOCKET = 'wrapperClosedWebsocket'  // 包装器关闭 WebSocket 命令

class ZeroFrame {
    constructor(url) {
        this.url = url  // 初始化 ZeroFrame 对象的 URL
        this.waiting_cb = {}  // 初始化等待回调的空对象
        this.wrapper_nonce = document.location.href.replace(/.*wrapper_nonce=([A-Za-z0-9]+).*/, "$1")  // 获取包装器的随机数
        this.connect()  // 调用 connect 方法
        this.next_message_id = 1  // 初始化下一个消息的 ID
        this.init()  // 调用 init 方法
    }

    init() {
        return this  // 返回当前对象
    }

    connect() {
        this.target = window.parent  // 将父窗口赋给目标
        window.addEventListener('message', e => this.onMessage(e), false)  // 监听消息事件，调用 onMessage 方法
        this.cmd(CMD_INNER_READY)  // 发送内部准备完毕命令
    }

    onMessage(e) {
        let message = e.data  // 获取消息数据
        let cmd = message.cmd  // 获取消息命令
        if (cmd === CMD_RESPONSE) {  // 如果命令是响应命令
            if (this.waiting_cb[message.to] !== undefined) {  // 如果等待回调中存在该消息的目标
                this.waiting_cb[message.to](message.result)  // 调用回调函数并传入结果
            }
            else {
                this.log("Websocket callback not found:", message)  // 否则记录错误信息
            }
        } else if (cmd === CMD_WRAPPER_READY) {  // 如果命令是包装器准备完毕命令
            this.cmd(CMD_INNER_READY)  // 发送内部准备完毕命令
        } else if (cmd === CMD_PING) {  // 如果命令是 ping 命令
            this.response(message.id, CMD_PONG)  // 发送 pong 响应
        } else if (cmd === CMD_WRAPPER_OPENED_WEBSOCKET) {  // 如果命令是包装器打开 WebSocket 命令
            this.onOpenWebsocket()  // 调用打开 WebSocket 方法
        } else if (cmd === CMD_WRAPPER_CLOSE_WEBSOCKET) {  // 如果命令是包装器关闭 WebSocket 命令
            this.onCloseWebsocket()  // 调用关闭 WebSocket 方法
        } else {
            this.onRequest(cmd, message)  // 否则调用请求方法
        }
    }

    onRequest(cmd, message) {
        this.log("Unknown request", message)  // 记录未知请求的信息
    }
}
    # 定义一个响应方法，用于发送响应消息
    response(to, result) {
        # 发送包含命令、接收者和结果的消息
        this.send({
            cmd: CMD_RESPONSE,
            to: to,
            result: result
        })
    }

    # 定义一个发送命令的方法，可以包含参数和回调函数
    cmd(cmd, params={}, cb=null) {
        # 发送包含命令和参数的消息，并可选地包含回调函数
        this.send({
            cmd: cmd,
            params: params
        }, cb)
    }

    # 定义一个发送命令并返回 Promise 的方法
    cmdp(cmd, params={}) {
        # 返回一个 Promise 对象，用于发送命令并处理结果
        return new Promise((resolve, reject) => {
            # 调用 cmd 方法发送命令，并根据结果执行 resolve 或 reject
            this.cmd(cmd, params, (res) => {
                if (res && res.error) {
                    reject(res.error)
                } else {
                    resolve(res)
                }
            })
        })
    }

    # 定义一个发送消息的方法
    send(message, cb=null) {
        # 设置消息的包装随机数和 ID
        message.wrapper_nonce = this.wrapper_nonce
        message.id = this.next_message_id
        this.next_message_id++
        # 发送消息到目标窗口
        this.target.postMessage(message, '*')
        # 如果存在回调函数，将其存储到等待回调的对象中
        if (cb) {
            this.waiting_cb[message.id] = cb
        }
    }

    # 定义一个日志输出方法
    log(...args) {
        # 输出带有前缀的日志信息
        console.log.apply(console, ['[ZeroFrame]'].concat(args))
    }

    # 定义一个 WebSocket 打开时的回调方法
    onOpenWebsocket() {
        # 输出 WebSocket 打开的日志信息
        this.log('Websocket open')
    }

    # 定义一个 WebSocket 关闭时的回调方法
    onCloseWebsocket() {
        # 输出 WebSocket 关闭的日志信息
        this.log('Websocket close')
    }

    # 定义一个用于修改 Ajax 的方法
    monkeyPatchAjax() {
        # 获取当前页面对象的引用
        var page = this
        # 保存原始的 XMLHttpRequest.open 方法
        XMLHttpRequest.prototype.realOpen = XMLHttpRequest.prototype.open
        # 调用 cmd 方法获取 Ajax 密钥
        this.cmd("wrapperGetAjaxKey", [], (res) => { this.ajax_key = res })
        # 定义一个新的 XMLHttpRequest.open 方法，用于在 URL 中添加 Ajax 密钥
        var newOpen = function (method, url, async) {
            url += "?ajax_key=" + page.ajax_key
            return this.realOpen(method, url, async)
        }
        # 覆盖原始的 XMLHttpRequest.open 方法
        XMLHttpRequest.prototype.open = newOpen
    }
# 闭合前面的函数定义
```