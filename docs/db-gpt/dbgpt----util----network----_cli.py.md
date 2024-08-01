# `.\DB-GPT-src\dbgpt\util\network\_cli.py`

```py
# 导入标准库模块
import os
import socket  # 导入 socket 模块，用于网络通信
import ssl as py_ssl  # 导入 ssl 模块，用于加密通信
import threading  # 导入 threading 模块，支持多线程编程

# 导入第三方库
import click  # 导入 click 库，用于创建命令行接口
from ..console import CliLogger  # 导入自定义模块 CliLogger，用于日志记录

# 创建日志记录器对象
logger = CliLogger()


def forward_data(source, destination):
    """Forward data from source to destination."""
    try:
        while True:
            # 从源端口接收数据
            data = source.recv(4096)
            # 如果接收到空数据，则发送并终止循环
            if b"" == data:
                destination.sendall(data)
                break
            # 如果没有接收到数据，直接终止循环
            if not data:
                break  # no more data or connection closed
            # 将接收到的数据转发到目标端口
            destination.sendall(data)
    except Exception as e:
        # 发生异常时记录错误日志
        logger.error(f"Error forwarding data: {e}")


def handle_client(
    client_socket,
    remote_host: str,
    remote_port: int,
    is_ssl: bool = False,
    http_proxy=None,
):
    """Handle client connection.

    Create a connection to the remote host and port, and forward data between the
    client and the remote host.

    Close the client socket and remote socket when all forwarding threads are done.
    """
    # 如果有 HTTP 代理，则通过代理连接远程主机
    if http_proxy:
        proxy_host, proxy_port = http_proxy
        remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 连接 HTTP 代理服务器
        remote_socket.connect((proxy_host, proxy_port))
        # 获取客户端 IP 地址
        client_ip = client_socket.getpeername()[0]
        # 确定连接协议（http 或 https）
        scheme = "https" if is_ssl else "http"
        # 构建 CONNECT 请求头
        connect_request = (
            f"CONNECT {remote_host}:{remote_port} HTTP/1.1\r\n"
            f"Host: {remote_host}\r\n"
            f"Connection: keep-alive\r\n"
            f"X-Real-IP: {client_ip}\r\n"
            f"X-Forwarded-For: {client_ip}\r\n"
            f"X-Forwarded-Proto: {scheme}\r\n\r\n"
        )
        # 发送 CONNECT 请求到代理服务器
        logger.info(f"Sending connect request: {connect_request}")
        remote_socket.sendall(connect_request.encode())

        # 接收代理服务器的响应
        response = b""
        while True:
            part = remote_socket.recv(4096)
            response += part
            # 当收到完整的响应头时停止接收
            if b"\r\n\r\n" in part:
                break

        # 如果连接未成功建立，记录错误日志并返回
        if b"200 Connection established" not in response:
            logger.error("Failed to establish connection through proxy")
            return

    else:
        # 如果没有代理，则直接连接远程主机
        remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        remote_socket.connect((remote_host, remote_port))

    # 如果需要 SSL 加密，则创建 SSL 套接字
    if is_ssl:
        # 创建 SSL 上下文
        context = py_ssl.create_default_context(py_ssl.Purpose.SERVER_AUTH)
        # 在已有连接上包装成 SSL 套接字
        ssl_target_socket = context.wrap_socket(
            remote_socket, server_hostname=remote_host
        )
    else:
        # 否则直接使用普通套接字
        ssl_target_socket = remote_socket
    try:
        # 尝试建立 SSL 连接
        ssl_target_socket.connect((remote_host, remote_port))
        
        # 创建线程，将客户端数据转发到服务器
        client_to_server = threading.Thread(
            target=forward_data, args=(client_socket, ssl_target_socket)
        )
        client_to_server.start()

        # 创建线程，将服务器数据转发到客户端
        server_to_client = threading.Thread(
            target=forward_data, args=(ssl_target_socket, client_socket)
        )
        server_to_client.start()

        # 等待线程结束
        client_to_server.join()
        server_to_client.join()
    except Exception as e:
        # 捕获并记录异常信息
        logger.error(f"Error handling client connection: {e}")
    finally:
        # 关闭客户端和服务器套接字
        client_socket.close()
        ssl_target_socket.close()
@click.command(name="forward")
@click.option("--local-port", required=True, type=int, help="Local port to listen on.")
@click.option(
    "--remote-host", required=True, type=str, help="Remote host to forward to."
)
@click.option(
    "--remote-port", required=True, type=int, help="Remote port to forward to."
)
@click.option(
    "--ssl",
    is_flag=True,
    help="Whether to use SSL for the connection to the remote host.",
)
@click.option(
    "--tcp",
    is_flag=True,
    help="Whether to forward TCP traffic. "
    "Default is HTTP. TCP has higher performance but not support proxies now.",
)
@click.option("--timeout", type=int, default=120, help="Timeout for the connection.")
@click.option(
    "--proxies",
    type=str,
    help="HTTP proxy to use for forwarding requests. e.g. http://127.0.0.1:7890, "
    "if not specified, try to read from environment variable http_proxy and "
    "https_proxy.",
)
def start_forward(
    local_port,
    remote_host,
    remote_port,
    ssl: bool,
    tcp: bool,
    timeout: int,
    proxies: str | None = None,
):
    """Start a TCP/HTTP proxy server that forwards traffic from a local port to a remote
    host and port, just for debugging purposes, please don't use it in production
    environment.
    """

    """
    Example:
        1. Forward HTTP traffic:

        ```
        dbgpt net forward --local-port 5010 \
            --remote-host api.openai.com \
            --remote-port 443 \
            --ssl \
            --proxies http://127.0.0.1:7890 \
            --timeout 30    
        ```py
        Then you can set your environment variable `OPENAI_API_BASE` to 
        `http://127.0.0.1:5010/v1`
    """
    
    # 根据是否选择 TCP 或 HTTP 进行不同的转发方式
    if not tcp:
        _start_http_forward(local_port, remote_host, remote_port, ssl, timeout, proxies)
    else:
        # 使用 TCP 协议建立服务器
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            # 绑定服务器到指定的本地端口
            server.bind(("0.0.0.0", local_port))
            # 启动监听，最大连接数设为 5
            server.listen(5)
            logger.info(
                f"[*] Listening on 0.0.0.0:{local_port}, forwarding to "
                f"{remote_host}:{remote_port}"
            )
            # 获取代理信息，如果没有指定则尝试从环境变量中读取
            proxies = (
                proxies or os.environ.get("http_proxy") or os.environ.get("https_proxy")
            )
            if proxies:
                # 如果指定了代理，则解析代理地址和端口
                if proxies.startswith("http://") or proxies.startswith("https://"):
                    proxies = proxies.split("//")[1]
                http_proxy = proxies.split(":")[0], int(proxies.split(":")[1])

            # 持续接受客户端连接并处理
            while True:
                client_socket, addr = server.accept()
                logger.info(f"[*] Accepted connection from: {addr[0]}:{addr[1]}")
                # 创建处理客户端连接的线程
                client_thread = threading.Thread(
                    target=handle_client,
                    args=(client_socket, remote_host, remote_port, ssl, http_proxy),
                )
                client_thread.start()


def _start_http_forward(
    local_port, remote_host, remote_port, ssl: bool, timeout, proxies: str | None = None


    # 定义函数参数：本地端口号，远程主机地址，远程端口号，SSL 连接标志（布尔型），超时时间，代理设置（字符串或None）
    # local_port: 本地端口号
    # remote_host: 远程主机地址
    # remote_port: 远程端口号
    # ssl: 是否使用 SSL 连接，布尔类型
    # timeout: 连接超时时间
    # proxies: 可选的代理设置，可以是字符串或者 None
# 导入httpx库，用于处理HTTP请求
# 导入uvicorn库，用于运行FastAPI应用
# 导入FastAPI框架中的BackgroundTasks、Request、Response类
# 导入StreamingResponse类，用于生成流式响应
from fastapi import BackgroundTasks, Request, Response
from fastapi.responses import StreamingResponse

# 从dbgpt.util.fastapi模块中导入create_app函数
from dbgpt.util.fastapi import create_app

# 使用create_app函数创建FastAPI应用实例
app = create_app()

# 将uvicorn作为FastAPI应用的中间件，监听HTTP请求
@app.middleware("http")
# 以非阻塞方式运行FastAPI应用，指定监听的主机为0.0.0.0，端口号由local_port决定
uvicorn.run(app, host="0.0.0.0", port=local_port)
```