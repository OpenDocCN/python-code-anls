# `.\pytorch\torch\distributed\elastic\rendezvous\c10d_rendezvous_backend.py`

```py
    def __init__(self, store: Store, run_id: str) -> None:
        # 检查运行 ID 是否为空，如果是则抛出数值错误异常
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        # 将传入的 Store 实例赋给对象属性 _store
        self._store = store

        # 根据运行 ID 构建键名，用于存储在分布式存储中
        self._key = "torch.rendezvous." + run_id

        # 使用分布式存储的 compare_set 方法，在 _key 对应的位置设置空字符串，并使用 _NULL_SENTINEL 作为哨兵值
        # 这样设置是为了解决使用存储作为常规键值字典时的阻塞读取问题
        self._call_store("compare_set", self._key, "", self._NULL_SENTINEL)

    @property
    def name(self) -> str:
        """See base class."""
        # 返回当前后端的名称
        return "c10d"

    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """See base class."""
        # 从存储中获取当前状态的 Base64 编码表示
        base64_state: bytes = self._call_store("get", self._key)

        # 将 Base64 编码的状态解码并返回状态数据和令牌
        return self._decode_state(base64_state)

    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> None:
        """See base class."""
        # 将给定的状态数据和可选的令牌保存到存储中
        self._call_store("set", self._key, self._encode_state(state, token))
    ) -> Optional[Tuple[bytes, Token, bool]]:
        """See base class."""
        # 将状态对象转换为Base64编码的字符串
        base64_state_str: str = b64encode(state).decode()

        if token:
            # 如果token存在，则进行下列操作
            if not isinstance(token, bytes):
                # 如果token不是字节对象，则获取当前状态，并返回带有False标志的临时元组
                result = self.get_state()
                if result is not None:
                    tmp = *result, False
                    # Python 3.6不支持在return语句中的元组解包
                    return tmp
                return None

            # 将字节类型的token解码为字符串类型
            token = token.decode()
        else:
            # 如果token不存在，则将其设置为_NULL_SENTINEL
            token = self._NULL_SENTINEL

        # 调用存储对象的compare_set方法，比较本地token和状态的Base64编码字符串
        base64_state: bytes = self._call_store(
            "compare_set", self._key, token, base64_state_str
        )

        # 解码Base64编码的状态，获取新的状态和token
        state_token_pair = self._decode_state(base64_state)
        if state_token_pair is None:
            return None

        new_state, new_token = state_token_pair

        # C10d Store的compare_set方法无法直接确定写入是否成功，因此使用位比较进行验证
        return new_state, new_token, new_state == state

    def _call_store(self, store_op: str, *args, **kwargs) -> Any:
        # 尝试调用存储对象的指定操作，并返回其结果
        try:
            return getattr(self._store, store_op)(*args, **kwargs)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            # 捕获可能的异常，并封装成RendezvousConnectionError抛出
            raise RendezvousConnectionError(
                "The connection to the C10d store has failed. See inner exception for details."
            ) from exc

    def _decode_state(self, base64_state: bytes) -> Optional[Tuple[bytes, Token]]:
        # 如果base64编码的状态等于_NULL_SENTINEL编码，则返回None
        if base64_state == self._NULL_SENTINEL.encode():
            return None

        try:
            # 尝试解码base64编码的状态
            state = b64decode(base64_state)
        except binascii.Error as exc:
            # 捕获解码错误，并封装成RendezvousStateError抛出
            raise RendezvousStateError(
                "The state object is corrupt. See inner exception for details."
            ) from exc

        # 返回解码后的状态和原始base64编码的状态
        return state, base64_state
# 根据给定的参数创建一个 TCPStore 实例
def _create_tcp_store(params: RendezvousParameters) -> TCPStore:
    # 解析参数中的端点地址和默认端口号，并分配给 host 和 port
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=29400)

    # 从参数中获取是否作为主机的配置信息
    cfg_is_host = params.get_as_bool("is_host")

    # 如果用户显式指定了进程是否应该托管存储，则使用指定的值
    if cfg_is_host is not None:
        is_host = cfg_is_host
    # 否则根据主机名和 IP 地址判断当前进程是否应该托管存储
    else:
        is_host = _matches_machine_hostname(host)

    # 从参数中获取是否使用 libuv 的配置信息，默认为 False
    use_libuv = params.get_as_bool("use_libuv", False)

    # 设置读取超时时间，如果小于等于 0，则抛出异常
    read_timeout = cast(int, params.get_as_int("read_timeout", 60))
    if read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    # 在特定情况下，尝试实例化存储两次。详细信息请参见下面 except 子句中的说明。
    for is_server in [is_host, False]:
        try:
            # 创建 TCPStore 实例
            store = TCPStore(
                host,
                port,
                is_master=is_server,
                multi_tenant=True,
                timeout=timedelta(seconds=read_timeout),
                use_libuv=use_libuv,
            )

            # 如果当前进程被确定为主机，记录相应的事件和日志信息
            if is_server:
                msg = f"Process {os.getpid()} hosts the TCP store for the C10d rendezvous backend."
                construct_and_record_rdzv_event(
                    run_id=params.run_id, message=msg, node_state=NodeState.INIT
                )
                logger.info(msg)

            # 成功创建存储实例后退出循环
            break
        except (ValueError, RuntimeError, TimeoutError) as exc:
            # 如果启发式地将 is_host 的值推断为 True，并且第一次尝试实例化 TCPStore 失败，
            # 则将 is_host 设置为 False 再尝试一次。在边缘情况下，同一机器上可能有多个进程
            # 参与相同的 rendezvous，但最终只有一个进程会托管存储。

            # 如果不是作为服务端，或者用户已显式指定 is_host，则将异常传递出去
            if not is_server or cfg_is_host is not None:
                raise RendezvousConnectionError(
                    "The connection to the C10d store has failed. See inner exception for details."
                ) from exc

    # 返回创建的存储实例
    return store  # type: ignore[possibly-undefined]


# 根据给定的参数创建一个 FileStore 实例
def _create_file_store(params: RendezvousParameters) -> FileStore:
    # 如果用户指定了 endpoint，则将其视为文件路径
    if params.endpoint:
        path = params.endpoint
    else:
        try:
            # 创建一个临时文件，仅由当前进程可读写
            _, path = tempfile.mkstemp()
        except OSError as exc:
            # 如果文件创建失败，抛出异常
            raise RendezvousError(
                "The file creation for C10d store has failed. See inner exception for details."
            ) from exc

    try:
        # 创建 FileStore 实例
        store = FileStore(path)
    # 捕获可能的异常类型：ValueError 和 RuntimeError，并将其存储在变量 exc 中
    except (ValueError, RuntimeError) as exc:
        # 抛出自定义的 RendezvousConnectionError 异常，提供连接到 C10d 存储失败的错误信息
        raise RendezvousConnectionError(
            "The connection to the C10d store has failed. See inner exception for details."
        ) from exc

    # 返回变量 store，这个变量可能在异常处理之外被定义或修改
    return store
    # 从参数中获取存储类型，默认为 "tcp"
    store_type = params.get("store_type", "tcp").strip().lower()
    # 声明一个变量 store，类型为 Store，用于后续赋值
    store: Store
    try:
        # 尝试根据指定的存储类型创建存储对象
        if store_type == "file":
            # 如果存储类型是文件，调用函数创建文件存储对象
            store = _create_file_store(params)
        elif store_type == "tcp":
            # 如果存储类型是 TCP，调用函数创建 TCP 存储对象
            store = _create_tcp_store(params)
        else:
            # 如果存储类型既不是文件也不是 TCP，抛出值错误异常
            raise ValueError(
                "Invalid store type given. Currently only supports file and tcp."
            )

        # 使用创建的存储对象和运行 ID 创建 C10dRendezvousBackend 后端对象
        backend = C10dRendezvousBackend(store, params.run_id)

    except Exception as e:
        # 如果在创建过程中发生异常，构造并记录 Rendezvous 事件
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        # 将捕获的异常向上层抛出
        raise

    # 返回创建的后端对象和存储对象
    return backend, store
```