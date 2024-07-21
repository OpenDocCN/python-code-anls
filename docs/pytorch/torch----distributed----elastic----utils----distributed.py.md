# `.\pytorch\torch\distributed\elastic\utils\distributed.py`

```
# 定义一个函数，用于创建基于 C10d 的分布式存储
def create_c10d_store(
    is_server: bool,                         # 标识是否为服务端
    server_addr: str,                        # 服务端地址
    server_port: int = -1,                   # 服务端端口，默认为-1
    world_size: int = 1,                     # 全局进程数，默认为1
    timeout: float = (60 * 10),              # 超时时间，单位秒，默认为10分钟
    wait_for_workers: bool = True,           # 是否等待工作进程连接，默认为True
    retries=3,                               # 连接重试次数，默认为3次
    use_libuv: Optional[bool] = None,        # 是否使用 Libuv 库进行网络通信，可选
):
    # 当 world_size 大于 1 时，必须指定 server_port
    if server_port == -1 and world_size > 1:
        raise ValueError(
            f"server_port must be specified when world_size > 1, got server_port={server_port}, world_size={world_size}"
        )

    # 如果指定了 server_port，则记录日志并忽略重试设置
    if server_port != -1:
        logger.info("sever_port: %s, specified, ignoring retries", server_port)

    # 只有在 server_port 不是静态指定时才进行多次尝试连接
    attempt = retries if server_port == -1 else 1
    # 无限循环，直到成功创建了 c10d 存储对象或者遇到无法处理的运行时错误
    while True:
        # 如果指定了服务器端口，则使用该端口；否则获取一个空闲端口
        if server_port != -1:
            port = server_port
        else:
            port = get_free_port()

        # 记录创建 c10d 存储对象的详细信息，包括服务器地址、端口号、集群规模、是否为服务器、超时时间、是否使用 libuv
        logger.info(
            "Creating c10d store on %s:%s\n"
            "  world_size  : %s\n"
            "  is_server   : %s\n"
            "  timeout(sec): %s\n"
            "  use_libuv   : %s\n",
            server_addr,
            port,
            world_size,
            is_server,
            timeout,
            use_libuv,
        )

        try:
            # 部分构造函数，用于创建 TCPStore 实例
            store_builder = functools.partial(
                dist.TCPStore,
                host_name=server_addr,
                port=port,
                world_size=world_size,
                is_master=is_server,
                timeout=datetime.timedelta(seconds=timeout),
                wait_for_workers=wait_for_workers,
            )
            # 根据 use_libuv 参数决定是否指定 TCPStore 使用 libuv
            if use_libuv is None:
                # 创建 TCPStore 对象，使用默认的后端实现
                store = store_builder()
            else:
                # 创建 TCPStore 对象，指定使用 libuv
                store = store_builder(use_libuv=use_libuv)

            # 如果需要等待所有工作进程加入，进行完整的排名检查
            if wait_for_workers:
                _check_full_rank(store, world_size, timeout=timeout)
            # 记录成功创建 c10d 存储对象的信息
            logger.info("Successfully created c10d store")
            # 返回创建的存储对象
            return store
        except RuntimeError as e:
            # 捕获运行时错误，通常是端口冲突或超时异常
            # 当端口已被占用时，根据错误消息判断并处理
            if str(e) == _ADDRESS_IN_USE:  # 这种情况仅发生在服务器端口冲突时
                if attempt < retries:
                    # 如果尝试次数未达到上限，记录警告信息并增加尝试次数
                    logger.warning(
                        "port: %s already in use, attempt: [%s/%s]",
                        port,
                        attempt,
                        retries,
                    )
                    attempt += 1
                else:
                    # 如果已达到尝试次数上限，抛出详细的运行时错误信息
                    raise RuntimeError(
                        f"on {server_addr}, port: {port} already in use"
                    ) from e
            else:
                # 如果捕获的运行时错误不是由端口冲突引起的，抛出原始异常
                raise
# 检查是否所有成员已经加入，通过调用 barrier 函数进行同步
def _check_full_rank(store, world_size, timeout):
    try:
        # 调用 barrier 函数，等待所有成员加入完成
        barrier(store, world_size, key_prefix=_TCP_STORE_INIT, barrier_timeout=timeout)
    except RuntimeError as e:
        # 捕获 RuntimeError 异常
        if str(e) == _SOCKET_TIMEOUT:
            # 如果异常信息为超时错误，则抛出 TimeoutError 异常
            raise TimeoutError(
                f"timed out waiting for all {world_size} members to join"
            ) from e
        else:
            # 如果异常信息不是超时错误，则继续抛出原始异常
            raise


def get_free_port():
    """
    Returns an unused port on localhost.

    This function finds an unused port on localhost by opening to socket to bind
    to a port and then closing it.

    Returns:
        int: an unused port on localhost

    Example:
        >>> # xdoctest: +SKIP("Nondeterministic")
        >>> get_free_port()
        63976

    ..note:
        The port returned by :func:`get_free_port` is not reserved and may be
        taken by another process after this function returns.
    """
    # 获取一个带有未使用端口的 socket 对象
    sock = get_socket_with_port()
    # 使用 closing 上下文管理器确保 socket 在退出时关闭
    with closing(sock):
        # 返回 socket 的端口号
        return sock.getsockname()[1]


def get_socket_with_port() -> socket.socket:
    """
    Returns a free port on localhost that is "reserved" by binding a temporary
    socket on it. Close the socket before passing the port to the entity
    that requires it. Usage example

    ::

    sock = _get_socket_with_port()
    with closing(sock):
        port = sock.getsockname()[1]
        sock.close()
        # there is still a race-condition that some other process
        # may grab this port before func() runs
        func(port)
    """

    # 获取 localhost 的地址信息，包括各种协议族和端口信息
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    for addr in addrs:
        # 解析地址信息
        family, type, proto, _, _ = addr
        # 创建一个 socket 对象
        s = socket.socket(family, type, proto)
        try:
            # 绑定 socket 到一个临时端口
            s.bind(("localhost", 0))
            # 开始监听该端口
            s.listen(0)
            # 返回成功绑定的 socket 对象
            return s
        except OSError as e:
            # 如果绑定失败，则关闭 socket 对象并记录警告日志
            s.close()
            logger.warning("Socket creation attempt failed.", exc_info=e)
    # 如果所有尝试都失败，则抛出 RuntimeError 异常
    raise RuntimeError("Failed to create a socket")
```