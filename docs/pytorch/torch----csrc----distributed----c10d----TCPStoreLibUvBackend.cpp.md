# `.\pytorch\torch\csrc\distributed\c10d\TCPStoreLibUvBackend.cpp`

```
// 包含必要的标准库和第三方库头文件
#include <algorithm>
#include <deque>
#include <exception>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// 包含 C10D 和 LibUV 的特定头文件
#include <c10/util/thread_name.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStoreBackend.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

#ifdef TORCH_USE_LIBUV
// 在使用 LibUV 的情况下，包含 LibUV 头文件
#include <uv.h>
#endif

// 定义 c10d::detail 命名空间，包含以下实现细节
namespace c10d::detail {

#ifdef TORCH_USE_LIBUV

/*
异常安全性：

在客户端处理期间使用异常是安全的。
其他回调函数不提供异常安全性，因此要避免使用异常。
*/

// 控制等待在后台的未接受 TCP 连接的最大数量，默认为主机的最大值
#define DEFAULT_BACKLOG -1
// 最大键值对数和最大字符串长度和最大负载长度的定义
#define MAX_KEY_COUNT (128 * 1024)
#define MAX_STRING_LEN (8 * 1024)
#define MAX_PAYLOAD_LEN (8 * 1024 * 1024)

// 分配缓冲区的推荐大小
#define ALLOC_BUFFER_SIZE ((size_t)4000)

// UvHandle 类，用于管理 LibUV 句柄的基础类
class UvHandle : public c10::intrusive_ptr_target {
 public:
  ~UvHandle() override = default;

  // 获取 UvHandle 的智能指针
  c10::intrusive_ptr<UvHandle> iptr() {
    return c10::intrusive_ptr<UvHandle>::reclaim_copy(this);
  }

  // 关闭 LibUV 句柄
  void close() {
    if (uv_is_closing(unsafeGetHandle())) {
      return;
    }
    uv_close(unsafeGetHandle(), on_close);
  }

  // 获取 LibUV 句柄的抽象方法，需要子类实现
  virtual uv_handle_t* unsafeGetHandle() = 0;

 protected:
  // 设置句柄数据和增加引用计数
  void handleReady() {
    uv_handle_set_data(unsafeGetHandle(), this);
    at::raw::intrusive_ptr::incref(this);
  }

  // LibUV 句柄关闭时的回调函数，需要子类实现
  virtual void onClose() = 0;

 private:
  // 从 LibUV 句柄中获取 UvHandle 智能指针
  static c10::intrusive_ptr<UvHandle> reclaim(uv_handle_t* handle) {
    auto h = (UvHandle*)uv_handle_get_data(handle);
    return c10::intrusive_ptr<UvHandle>::reclaim(h);
  }

  // LibUV 句柄关闭的静态回调函数
  static void on_close(uv_handle_t* uv_handle) {
    auto handle = reclaim(uv_handle);
    handle->onClose();
  }
};

// UvTcpSocket 类，继承自 UvHandle，用于管理 LibUV TCP 套接字
class UvTcpSocket : public UvHandle {
  uv_tcp_t client{}; // LibUV TCP 客户端对象

  // 获取 UvTcpSocket 的智能指针
  c10::intrusive_ptr<UvTcpSocket> iptr() {
    return c10::intrusive_ptr<UvTcpSocket>::reclaim_copy(this);
  }

  // 从 LibUV 流句柄中获取 UvTcpSocket 智能指针
  static c10::intrusive_ptr<UvTcpSocket> borrow(uv_stream_t* handle) {
    auto h = (UvTcpSocket*)uv_handle_get_data((uv_handle_t*)handle);
    return h->iptr();
  }

  // 分配缓冲区的回调函数
  static void alloc_buffer(
      uv_handle_t* handle,
      size_t suggested_size,
      uv_buf_t* buf) {
    suggested_size = std::min(suggested_size, (size_t)ALLOC_BUFFER_SIZE);
    buf->base = (char*)malloc(suggested_size);
    buf->len = suggested_size;
  }

  // 读取数据的回调函数
  static void read_callback(
      uv_stream_t* client,
      ssize_t nread,
      const uv_buf_t* buf) {
    // 读取数据并处理的具体逻辑在此处实现

      // 读取数据并处理的具体逻辑在此处实现
    // 从客户端借用一个 UV TCP Socket 对象
    auto uv_socket = UvTcpSocket::borrow(client);

    // 如果读取操作返回小于 0 的值，表示读取失败
    if (nread < 0) {
      // 输出调试信息，记录读取回调失败的具体信息
      C10D_DEBUG(
          "Read callback failed. code:{} name:{} desc:{}",
          nread,
          uv_err_name(nread),
          uv_strerror(nread));
      // 关闭 UV TCP Socket 对象
      uv_socket->close();
      return;
    }
    // 如果成功读取到数据
    if (nread > 0) {
      try {
        // 处理接收到的数据缓冲区
        uv_socket->processBuf(buf, nread);
      } catch (std::exception& ex) {
        // 捕获处理数据时可能抛出的异常，输出警告信息
        C10D_WARNING("Error processing client message: {}", ex.what());
        // 关闭 UV TCP Socket 对象
        uv_socket->close();
      }
    }
  }

 public:
  // 显式构造函数，初始化 UV TCP Socket 对象
  explicit UvTcpSocket(uv_loop_t* loop) {
    // 使用给定的事件循环和客户端 TCP 对象初始化 UV TCP 对象
    uv_tcp_init(loop, &client);
    // 设置客户端 TCP 对象为无延迟模式
    if (int err = uv_tcp_nodelay(&client, 1)) {
      // 如果设置无延迟模式失败，输出警告信息
      C10D_WARNING(
          "The no-delay option cannot be enabled for the client socket. err={}",
          err);
    }
  }

  // 启动读取操作
  void startRead() {
    // 启动从客户端读取数据的回调函数
    int res = uv_read_start((uv_stream_t*)&client, alloc_buffer, read_callback);
    // 如果启动读取操作失败
    if (res) {
      // 输出警告信息，记录读取回调设置失败的具体信息
      C10D_WARNING(
          "Failed to setup read callback. client:{} code:{} name:{} desc:{}.",
          (void*)this,
          res,
          uv_err_name(res),
          uv_strerror(res));
      // 关闭 UV TCP Socket 对象
      close();
    }
  }

  // 获取不安全的事件处理句柄
  uv_handle_t* unsafeGetHandle() override {
    return (uv_handle_t*)&client;
  }

 protected:
  // 获取不安全的流对象
  uv_stream_t* unsafeGetStream() {
    return (uv_stream_t*)&client;
  }

  // 获取不安全的 TCP Socket 对象
  uv_tcp_t* unsafeGetSocket() {
    return &client;
  }

  // 虚拟函数，处理接收到的数据缓冲区
  virtual void processBuf(const uv_buf_t* buf, size_t nread) {
    // 抛出错误，提示未实现接收数据处理函数
    TORCH_CHECK(
        false, "Trying to read from a socket subclass that lacks processBuf");
  }

  // 关闭连接时的回调函数
  void onClose() override {
    // TODO 使用 registerClient（并重命名为 registerHandle）- 这将极大简化事务。
  }
  };

// UvTcpServer 类继承自 UvTcpSocket 类
class UvTcpServer : public UvTcpSocket {
 public:
  // 定义 OnConnectCallback 类型为接受 int 参数的函数对象
  typedef std::function<void(int)> OnConnectCallback;
  
  // 构造函数，接受一个 uv_loop_t* 参数，调用 UvTcpSocket 构造函数，并初始化 onConnectCb 为 missingOnConnect 函数对象
  explicit UvTcpServer(uv_loop_t* loop)
      : UvTcpSocket(loop), onConnectCb(missingOnConnect) {}

  // 静态方法，使用现有的 socket 创建 UvTcpServer 对象
  static c10::intrusive_ptr<UvTcpServer> makeWithSocket(
      uv_loop_t* loop,
      int socket) {
    auto res = c10::make_intrusive<UvTcpServer>(loop);
    res->handleReady();
    try {
      // 打开给定的 socket
      int uv_res = uv_tcp_open((uv_tcp_t*)res->unsafeGetStream(), socket);
      TORCH_CHECK(
          uv_res == 0,
          "Failed to open existing socket. ",
          "socket: ",
          socket,
          ", code: ",
          uv_res,
          ", name: ",
          uv_err_name(uv_res),
          ", message: ",
          uv_strerror(uv_res));

      // 缓存 socket 的端口号
      res->cacheSocketPort();
    } catch (std::exception& ex) {
      // 出现异常时关闭对象
      res->close();
      throw;
    }

    return res;
  }

  // 设置连接回调函数
  void setOnConnectCallback(OnConnectCallback&& callback) {
    onConnectCb = std::move(callback);
  }

  // 静态方法，使用指定的端口创建 UvTcpServer 对象
  static c10::intrusive_ptr<UvTcpServer> makeWithPort(
      uv_loop_t* loop,
      uint16_t port,
      bool useIpv6) {
    auto res = c10::make_intrusive<UvTcpServer>(loop);
    res->handleReady();
    try {
      struct sockaddr_storage addr {};
      int uv_res = 0;
      // 根据 useIpv6 的值解析地址
      if (useIpv6) {
        uv_res = uv_ip6_addr("::", port, (struct sockaddr_in6*)&addr);
      } else {
        uv_res = uv_ip4_addr("0.0.0.0", port, (struct sockaddr_in*)&addr);
      }
      TORCH_CHECK(
          uv_res == 0,
          "UV Store addr parsing failure. ",
          "useIpv6: ",
          useIpv6,
          ", code: ",
          uv_res,
          ", name: ",
          uv_err_name(uv_res),
          ", message: ",
          uv_strerror(uv_res));

      // 将 socket 绑定到指定地址和端口
      uv_res =
          uv_tcp_bind(res->unsafeGetSocket(), (const struct sockaddr*)&addr, 0);
      TORCH_CHECK(
          uv_res == 0,
          "The server socket has failed to bind. ",
          "useIpv6: ",
          useIpv6,
          ", code: ",
          uv_res,
          ", name: ",
          uv_err_name(uv_res),
          ", message: ",
          uv_strerror(uv_res));

      // 监听新连接
      uv_res =
          uv_listen(res->unsafeGetStream(), DEFAULT_BACKLOG, on_new_connection);
      TORCH_CHECK(
          uv_res == 0,
          "The server socket has failed to listen on any local network address. ",
          "useIpv6: ",
          useIpv6,
          ", code: ",
          uv_res,
          ", name: ",
          uv_err_name(uv_res),
          ", message: ",
          uv_strerror(uv_res));

      // 缓存 socket 的端口号
      res->cacheSocketPort();
    } catch (std::exception& ex) {
      // 出现异常时关闭对象
      res->close();
      throw;
    }

    return res;
  }

  // 返回对象绑定的端口号
  uint16_t port() const {
    return portNum;
  }

  // 接受连接请求，并将连接对象传递给指定的 socket
  void accept(const c10::intrusive_ptr<UvTcpSocket>& socket) {
    int res =
        uv_accept(unsafeGetStream(), (uv_stream_t*)socket->unsafeGetHandle());
    # 使用 TORCH_CHECK 宏来检查条件，确保 res 的值为 0，否则抛出异常信息
    TORCH_CHECK(
        res == 0,
        "Failed to accept socket. ",
        "code: ",
        res,
        ", name: ",
        uv_err_name(res),
        ", message: ",
        uv_strerror(res));
  }

 private:
  # 定义成员变量，用于存储连接成功时的回调函数
  OnConnectCallback onConnectCb;
  # 端口号变量的初始化，默认为 0
  uint16_t portNum{};

  # 返回当前对象的强引用，使用 intrusive_ptr
  c10::intrusive_ptr<UvTcpServer> iptr() {
    return c10::intrusive_ptr<UvTcpServer>::reclaim_copy(this);
  }

  # 通过给定的 handle 返回对应的 UvTcpServer 对象的引用
  static c10::intrusive_ptr<UvTcpServer> borrow(uv_stream_t* handle) {
    auto h = (UvTcpServer*)uv_handle_get_data((uv_handle_t*)handle);
    return h->iptr();
  }

  # 缓存当前 socket 的端口号信息
  void cacheSocketPort() {
    sockaddr_storage addr_s{};  # 定义 socket 地址结构体变量

    int addr_len = sizeof(addr_s);  # 获取地址结构体的大小

    # 使用 uv_tcp_getsockname 函数获取 socket 的本地地址信息
    if (uv_tcp_getsockname(
            (uv_tcp_t*)unsafeGetStream(),
            reinterpret_cast<sockaddr*>(&addr_s),
            &addr_len) != 0) {
      throw std::runtime_error(
          "The port number of the socket cannot be retrieved.");
    }

    # 根据地址家族判断并获取端口号，存储在 portNum 变量中
    if (addr_s.ss_family == AF_INET) {
      portNum = ntohs(reinterpret_cast<sockaddr_in*>(&addr_s)->sin_port);
    } else {
      portNum = ntohs(reinterpret_cast<sockaddr_in6*>(&addr_s)->sin6_port);
    }
  }

  # 当没有设置连接回调函数时，抛出异常
  static void missingOnConnect(int status) {
    TORCH_CHECK(false, "Socket accepted byt onConnect callback missing");
  }

  # 当有新连接时的回调函数，调用 borrow 函数来获取当前对象并调用 onConnectCb 回调
  static void on_new_connection(uv_stream_t* server, int status) {
    borrow(server)->onConnectCb(status);
  }
};

class WriterPayload : public c10::intrusive_ptr_target {
  static c10::intrusive_ptr<WriterPayload> reclaim(uv_write_t* request) {
    /* This method returns a intrusive_ptr that does not increase the refcount.
     * It reclaims ownership of the WriterPayload object from the given uv_write_t request. */
    auto h = (WriterPayload*)uv_req_get_data((uv_req_t*)request);
    return c10::intrusive_ptr<WriterPayload>::reclaim(h);
  }

  void registeredInLoop() {
    /*
    This refcount increment must be matched by a reclaim call.
    Call this method after sucessfully scheduling this handle with a loop.
    */
    at::raw::intrusive_ptr::incref(this);
  }

  static void write_done(uv_write_t* req, int status) {
    /* Since we're no longer actively used by the event loop, transfer ownership
     * to this frame. */
    auto wp = WriterPayload::reclaim(req);
    auto handle = wp->handle;

    if (status) {
      C10D_WARNING(
          "Write to client failed. code:{} name:{} desc:{}.",
          status,
          uv_err_name(status),
          uv_strerror(status));
      handle->close();
    }
  }

  std::vector<uint8_t> data;
  uv_write_t req = {};
  uv_buf_t buf = {};
  c10::intrusive_ptr<UvHandle> handle;

 public:
  WriterPayload(
      std::vector<uint8_t>&& in_data,
      c10::intrusive_ptr<UvHandle> handle)
      : data(std::move(in_data)), handle(std::move(handle)) {
    uv_req_set_data((uv_req_t*)&req, this);
  }

  ~WriterPayload() override = default;

  void send() {
    buf = uv_buf_init((char*)data.data(), data.size());
    int res = uv_write(
        &req, (uv_stream_t*)handle->unsafeGetHandle(), &buf, 1, write_done);

    if (res) {
      C10D_WARNING(
          "Write setup to client failed. code:{} name:{} desc:{}.",
          res,
          uv_err_name(res),
          uv_strerror(res));
      handle->close();
    } else {
      /* This object was successfully registered with the event loop, so keep it
       * alive until it's unregistered. */
      registeredInLoop();
    }
  }
};

class StreamWriter {
  std::vector<uint8_t> data;
  c10::intrusive_ptr<UvHandle> handle;

  // must be stack allocated
  void* operator new(size_t);

 public:
  StreamWriter(c10::intrusive_ptr<UvHandle> handle)
      : handle(std::move(handle)) {}

  void write1(uint8_t val) {
    data.push_back(val);
  }

  template <typename T>
  void write_value(T val) {
    uint8_t* val_ptr = (uint8_t*)&val;
    data.insert(data.end(), val_ptr, val_ptr + sizeof(T));
  }

  void write_vector(const std::vector<uint8_t>& val) {
    write_value<uint64_t>(val.size());
    data.insert(data.end(), val.begin(), val.end());
  }

  void write_string(const std::string& val) {
    write_value<uint64_t>(val.size());
    data.insert(data.end(), val.data(), val.data() + val.size());
  }

  void send() {
    auto wd = c10::make_intrusive<WriterPayload>(std::move(data), handle);
    wd->send();
  }
};
class ChunkedStream {
  std::deque<uv_buf_t> buffers;  // 存储数据块的双端队列
  size_t buff_idx{0};  // 当前读取的数据块索引
  size_t buff_offset{0};  // 当前数据块中的偏移量
  size_t capacity{0};  // 已缓存数据的总容量
  size_t buff_offset_commit{0};  // 提交数据块时的偏移量
  size_t read_offset{0};  // 已读取的字节数

 public:
  ChunkedStream() = default;  // 默认构造函数

  size_t buf_count() {  // 返回当前缓存的数据块数量
    return buffers.size();
  }

  void append(uv_buf_t buf) {  // 向缓存中追加数据块
    if (buf.len == 0) {  // 如果数据块长度为0，释放基础内存
      free(buf.base);
    } else {
      capacity += buf.len;  // 增加总容量
      buffers.push_back(buf);  // 将数据块添加到队尾
    }
  }

  bool read_many(char* dest, size_t size) {  // 从缓存中读取多个字节到目标位置
    if (available() < size) {  // 如果可用数据不足以满足读取请求，返回失败
      return false;
    }

    size_t remaining = size;
    char* write_base = dest;
    while (remaining > 0) {  // 循环直到读取完所有请求的字节数
      auto to_read = std::min(buffers[buff_idx].len - buff_offset, remaining);  // 计算当前数据块中可读取的字节数
      ::memcpy(write_base, buffers[buff_idx].base + buff_offset, to_read);  // 将数据块中的数据复制到目标位置
      buff_offset += to_read;  // 更新当前数据块中的偏移量
      remaining -= to_read;  // 更新剩余需读取的字节数
      write_base += to_read;  // 更新目标位置指针
      if (buff_offset >= buffers[buff_idx].len) {  // 如果当前数据块已全部读取
        buff_offset = 0;  // 重置偏移量
        ++buff_idx;  // 切换到下一个数据块
        if (buff_idx >= buffers.size() && remaining > 0) {  // 如果已经没有更多数据块可读且仍有剩余需读取的字节
          TORCH_CHECK(
              false,
              "Trying to read past end of buffer. ",
              "buffer_idx: ",
              buff_idx,
              ", available: ",
              buffers.size(),
              ", remaining: ",
              remaining);
        }
      }
    }
    read_offset += size;  // 更新已读取的字节数
    return true;
  }

  bool read1(uint8_t& byte) {  // 从缓存中读取一个字节
    while (true) {
      if (buff_idx >= buffers.size())  // 如果已经超出缓存范围，返回失败
        return false;
      if (buff_offset >= buffers[buff_idx].len) {  // 如果当前数据块已全部读取
        buff_offset = 0;  // 重置偏移量
        ++buff_idx;  // 切换到下一个数据块
        continue;
      }
      break;
    }

    byte = buffers[buff_idx].base[buff_offset];  // 读取当前字节
    ++buff_offset;  // 更新偏移量
    ++read_offset;  // 更新已读取的字节数
    return true;
  }

  template <typename T>
  bool read_value(T& value) {  // 从缓存中读取一个值
    return read_many((char*)&value, sizeof(T));
  }

  bool read_key(std::string& str) {  // 从缓存中读取一个字符串键
    uint64_t size = 0;
    if (!read_value(size))  // 先读取字符串长度
      return false;
    TORCH_CHECK(
        size <= MAX_STRING_LEN,
        "Invalid string size. ",
        "size: ",
        size,
        ", max: ",
        MAX_STRING_LEN);

    if (available() < size)  // 检查可用数据是否足够读取整个字符串
      return false;
    str.resize(size);  // 调整字符串容器大小
    return read_many((char*)str.data(), size);  // 从缓存中读取字符串内容
  }

  bool read_payload(std::vector<uint8_t>& data) {  // 从缓存中读取有效负载数据
    uint64_t size = 0;
    if (!read_value(size))  // 先读取有效负载数据长度
      return false;
    auto size_in_bytes = size * sizeof(uint8_t);
    TORCH_CHECK(
        size_in_bytes <= MAX_PAYLOAD_LEN,
        "Invalid payload size. ",
        "size: ",
        size_in_bytes,
        ", max: ",
        MAX_PAYLOAD_LEN);

    if (available() < size_in_bytes)  // 检查可用数据是否足够读取整个有效负载
      return false;
    data.resize(size);  // 调整有效负载数据容器大小
    return read_many((char*)data.data(), size_in_bytes);  // 从缓存中读取有效负载数据内容
  }

  size_t available() {  // 返回当前可用的数据量
    return capacity - read_offset;
  }

  void commit() {  // 提交当前读取位置
    if (buff_idx >= buffers.size() || buff_offset >= buffers[buff_idx].len) {
      buff_offset = 0;  // 重置偏移量
      if (buff_idx < buffers.size())
        ++buff_idx;  // 切换到下一个数据块
    }
  }
};
    # 遍历循环，释放动态分配的内存并调整缓冲区的容量
    for (size_t i = 0; i < buff_idx; ++i) {
      # 释放当前缓冲区队列中第一个缓冲区的基地址指针
      free(buffers[0].base);
      # 减去第一个缓冲区的长度，更新总容量
      capacity -= buffers[0].len;
      # 移除缓冲区队列中的第一个缓冲区
      buffers.pop_front();
    }
    
    # 重置缓冲区索引为零，表示缓冲区已经清空
    buff_idx = 0;
    
    # 将读取偏移量、缓冲区提交偏移量和缓冲区偏移量重置为相同值
    read_offset = buff_offset_commit = buff_offset;
    }
    
    void reset() {
      # 将缓冲区索引重置为零，表示当前无任何缓冲区
      buff_idx = 0;
      # 将读取偏移量重置为缓冲区偏移量的值，表示读取位置归位
      read_offset = buff_offset = buff_offset_commit;
    }
};

// LibUVStoreDaemon 类定义，继承自 BackgroundThread
class LibUVStoreDaemon : public BackgroundThread {
 public:
  // 构造函数，初始化使用指定端口
  explicit LibUVStoreDaemon(int port);
  // 析构函数，释放资源
  ~LibUVStoreDaemon() override;

  // 返回当前守护进程监听的端口号
  uint16_t port() const override;

  // 设置给定 key 对应的值为指定的字节向量
  void set(const std::string& key, const std::vector<uint8_t>& value);
  
  // 比较并设置给定 key 的值，如果当前值与期望值匹配，则设置为新值
  const std::vector<uint8_t>& compareAndSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& newValue);
  
  // 获取给定 key 对应的值
  const std::vector<uint8_t>& get(const std::string& key);
  
  // 将给定 key 对应的值增加指定的整数值，并返回增加后的结果
  int64_t add(const std::string& key, int64_t addVal);
  
  // 检查给定多个 key 是否存在
  bool checkKeys(const std::vector<std::string>& keys);
  
  // 等待多个 key 被设置，使用给定的客户端句柄
  bool waitKeys(
      const std::vector<std::string>& keys,
      const c10::intrusive_ptr<UvHandle>& client);
  
  // 返回存储的键值对数量
  int64_t size();
  
  // 删除给定 key 及其对应的值
  int64_t deleteKey(const std::string& key);
  
  // 将指定值追加到给定 key 对应的值后面
  void append(const std::string& key, const std::vector<uint8_t>& value);

  // 注册客户端，使用指定的 UvHandle
  void registerClient(const c10::intrusive_ptr<UvHandle>& client);
  
  // 注销客户端，使用指定的 UvHandle
  void unregisterClient(const c10::intrusive_ptr<UvHandle>& client);
  
  // 清除客户端的等待状态，使用指定的 UvHandle
  void clearClientWaitState(const c10::intrusive_ptr<UvHandle>& client);
  
  // 检查是否是杂项客户端，使用指定的 UvHandle
  bool isMiscellaneousClient(const c10::intrusive_ptr<UvHandle>& client);

  // 获取给定 UV TCP 句柄的端口号
  uint16_t get_socket_port(uv_tcp_t* handle);
  
  // 初始化方法，使用给定的 TCPStoreOptions
  void init(const TCPStoreOptions& opts);

 protected:
  // 后台线程运行的方法，继承自 BackgroundThread
  void run() override;
  
  // 停止后台线程的方法，继承自 BackgroundThread
  void stop() override;

 private:
  // libuv 事件循环
  uv_loop_t loop{};
  
  // TCP 服务器对象
  c10::intrusive_ptr<UvTcpServer> tcpServer;

  // 保存 TCP 数据的哈希映射，从 key 到值的字节向量
  std::unordered_map<std::string, std::vector<uint8_t>> tcpStore_;
  
  // 正在等待特定 key 的 UvClient 列表的哈希映射
  std::unordered_map<std::string, std::vector<c10::intrusive_ptr<UvHandle>>> waitingSockets_;
  
  // 正在等待特定 key 数量的 UvHandle 列表的哈希映射
  std::unordered_map<c10::intrusive_ptr<UvHandle>, size_t> keysAwaited_;
  
  // 当前连接的客户端集合
  std::unordered_set<c10::intrusive_ptr<UvHandle>> clients_;
  
  // 杂项客户端集合
  std::unordered_set<c10::intrusive_ptr<UvHandle>> miscellaneousClients_;
  
  // 监听的端口号
  int port_;

  // 静态方法：从 UV 句柄获取 LibUVStoreDaemon 实例
  static LibUVStoreDaemon& from_uv(uv_handle_t* stream) {
    return *(LibUVStoreDaemon*)uv_handle_get_data(stream);
  }

  // 处理新连接事件的静态方法
  static void on_new_connection(uv_stream_t* server, int status) {
    from_uv((uv_handle_t*)server).onConnect(status);
  }

  // 处理退出请求的静态方法
  static void on_exit_request(uv_async_t* handle) {
    from_uv((uv_handle_t*)handle).onExitRequest();
  }

  // 处理连接事件的方法
  void onConnect(int status);
  
  // 处理退出请求的方法
  void onExitRequest();
  
  // 唤醒等待特定 key 的客户端的方法
  void wakeupWaitingClients(const std::string& key);

  // 静态方法：打印活动句柄的数量
  // static void print_active_handles(uv_handle_t* handle, void* arg);
};

// UvClient 类定义，继承自 UvTcpSocket
class UvClient : public UvTcpSocket {
  // 分块流对象
  ChunkedStream stream;
  
  // 指向 LibUVStoreDaemon 实例的指针
  LibUVStoreDaemon* store;

 protected:
  // 处理缓冲区的方法，重写自 UvTcpSocket
  void processBuf(const uv_buf_t* buf, size_t nread) override {
    auto tmp = *buf;
    tmp.len = nread;
    stream.append(tmp);
    // 进入循环，持续处理客户端发送的命令，直到读取失败或遇到特定命令
    while (true) {
      // 重置数据流以准备读取新命令
      stream.reset();
      // 初始化命令变量为无效值
      uint8_t command = -1;
      // 从数据流中读取一个字节的命令，如果读取失败则退出循环
      if (!stream.read1(command))
        break;
      // 如果客户端属于杂项客户端，则执行特殊处理
      if (store->isMiscellaneousClient(iptr())) {
        // 如果命令不是验证命令，直接返回
        if ((QueryType)command != QueryType::VALIDATE)
          return;
        // 解析并处理验证命令，如果处理失败则返回
        if (!parse_validate_command())
          return;
      } else {
        // 对于其他类型的客户端命令，根据命令类型进行不同的处理
        switch ((QueryType)command) {
          // 处理 SET 命令
          case QueryType::SET:
            if (!parse_set_command())
              return;
            break;
          // 处理 COMPARE_SET 命令
          case QueryType::COMPARE_SET:
            if (!parse_compare_set_command())
              return;
            break;
          // 处理 GET 命令
          case QueryType::GET:
            if (!parse_get_command())
              return;
            break;
          // 处理 ADD 命令
          case QueryType::ADD:
            if (!parse_add_command())
              return;
            break;
          // 处理 CHECK 命令
          case QueryType::CHECK:
            if (!parse_check_command())
              return;
            break;
          // 处理 WAIT 命令
          case QueryType::WAIT:
            if (!parse_wait_command())
              return;
            break;
          // 处理 GETNUMKEYS 命令
          case QueryType::GETNUMKEYS:
            if (!parse_getnumkeys_command())
              return;
            break;
          // 处理 DELETE_KEY 命令
          case QueryType::DELETE_KEY:
            if (!parse_delete_key_command())
              return;
            break;
          // 处理 APPEND 命令
          case QueryType::APPEND:
            if (!parse_append_command())
              return;
            break;
          // 处理 MULTI_GET 命令
          case QueryType::MULTI_GET:
            if (!parse_multi_get_command())
              return;
            break;
          // 处理 MULTI_SET 命令
          case QueryType::MULTI_SET:
            if (!parse_multi_set_command())
              return;
            break;
          // 处理 CANCEL_WAIT 命令
          case QueryType::CANCEL_WAIT:
            if (!parse_cancel_wait_command())
              return;
            break;
          // 如果收到未知命令，记录调试信息，关闭连接并返回
          default:
            C10D_DEBUG(
                "Client sent invalid command. client:{} command:{}",
                (void*)this,
                (int)command);
            close();
            return;
        }
      }
      // 提交当前命令处理后的数据流状态
      stream.commit();
    }
  }

  // 解析并处理验证命令
  bool parse_validate_command() {
    // 读取验证数值
    uint32_t validateNumber = 0;
    if (!stream.read_value(validateNumber))
      return false;

    // 验证数值与预设的验证魔数比较，如果不匹配则返回失败
    if (validateNumber != c10d::detail::validationMagicNumber)
      return false;
    // 验证通过，返回成功
    return true;
  }

  // 解析并处理 SET 命令
  bool parse_set_command() {
    // 读取键名
    std::string key;
    if (!stream.read_key(key))
      return false;

    // 读取新数据负载
    std::vector<uint8_t> newData;
    if (!stream.read_payload(newData))
      return false;

    // 调用存储接口设置键和新数据
    store->set(key, newData);
    // 返回成功
    return true;
  }

  // 解析并处理 COMPARE_SET 命令
  bool parse_compare_set_command() {
    // 读取键名
    std::string key;
    if (!stream.read_key(key))
      return false;

    // 读取当前值负载
    std::vector<uint8_t> currentValue;
    if (!stream.read_payload(currentValue))
      return false;

    // 读取新值负载
    std::vector<uint8_t> newValue;
    if (!stream.read_payload(newValue))
      return false;

    // 调用存储接口进行比较并设置操作
    auto res = store->compareAndSet(key, currentValue, newValue);
    // 将操作结果写入数据流并发送
    StreamWriter sw(iptr());
    sw.write_vector(res);
    sw.send();
  // 返回 true，表示成功解析命令
  return true;
}

bool parse_get_command() {
  std::string key;
  // 尝试从流中读取键，如果失败则返回 false
  if (!stream.read_key(key))
    return false;

  // 获取存储中对应键的数据
  const auto& data = store->get(key);
  // 创建一个数据写入器，并将数据写入其中
  StreamWriter sw(iptr());
  sw.write_vector(data);
  // 发送数据
  sw.send();
  // 返回 true，表示成功解析并处理命令
  return true;
}

bool parse_add_command() {
  std::string key;
  // 尝试从流中读取键，如果失败则返回 false
  if (!stream.read_key(key))
    return false;

  int64_t addVal = 0;
  // 尝试从流中读取要添加的值，如果失败则返回 false
  if (!stream.read_value(addVal))
    return false;

  // 调用存储对象的 add 方法，添加指定键的值，并获取结果
  addVal = store->add(key, addVal);
  // 创建一个数据写入器，并将结果写入其中
  StreamWriter sw(iptr());
  sw.write_value(addVal);
  // 发送数据
  sw.send();

  // 返回 true，表示成功解析并处理命令
  return true;
}

bool parse_check_command() {
  uint64_t key_count = 0;
  // 尝试从流中读取键的数量，如果失败则返回 false
  if (!stream.read_value(key_count))
    return false;
  // 检查读取的键数量是否超过最大允许数量，如果超过则抛出异常
  TORCH_CHECK(
      key_count <= MAX_KEY_COUNT,
      "Too many keys being waited. ",
      "keys: ",
      key_count,
      ", max: ",
      MAX_KEY_COUNT);

  // 创建一个字符串向量，用于存储读取的键名
  std::vector<std::string> keys(key_count);
  // 依次从流中读取键名，并存储到 keys 向量中
  for (uint64_t i = 0; i < key_count; ++i) {
    if (!stream.read_key(keys[i]))
      return false;
  }

  // 创建一个数据写入器
  StreamWriter sw(iptr());
  // 检查存储中是否存在所有给定的键
  if (store->checkKeys(keys)) {
    sw.write_value(CheckResponseType::READY);
  } else {
    sw.write_value(CheckResponseType::NOT_READY);
  }
  // 发送数据
  sw.send();

  // 返回 true，表示成功解析并处理命令
  return true;
}

bool parse_wait_command() {
  uint64_t key_count = 0;
  // 尝试从流中读取键的数量，如果失败则返回 false
  if (!stream.read_value(key_count)) {
    return false;
  }
  // 检查读取的键数量是否超过最大允许数量，如果超过则抛出异常
  TORCH_CHECK(
      key_count <= MAX_KEY_COUNT,
      "Too many keys being waited. ",
      "keys: ",
      key_count,
      ", max: ",
      MAX_KEY_COUNT);

  // 创建一个字符串向量，用于存储读取的键名
  std::vector<std::string> keys(key_count);
  // 依次从流中读取键名，并存储到 keys 向量中
  for (uint64_t i = 0; i < key_count; ++i) {
    if (!stream.read_key(keys[i]))
      return false;
  }

  // 调用存储对象的 waitKeys 方法，等待所有给定的键就绪
  if (store->waitKeys(keys, iptr())) {
    // 创建一个数据写入器，并向客户端发送停止等待响应
    StreamWriter sw(iptr());
    sw.write1((uint8_t)WaitResponseType::STOP_WAITING);
    sw.send();
  }

  // 返回 true，表示成功解析并处理命令
  return true;
}

bool parse_getnumkeys_command() {
  // 创建一个数据写入器
  StreamWriter sw(iptr());
  // 向客户端发送存储中键的数量
  sw.write_value<int64_t>(store->size());
  // 发送数据
  sw.send();

  // 返回 true，表示成功解析并处理命令
  return true;
}

bool parse_delete_key_command() {
  std::string key;
  // 尝试从流中读取要删除的键，如果失败则返回 false
  if (!stream.read_key(key))
    return false;

  // 调用存储对象的 deleteKey 方法，删除指定键，并获取删除的数量
  auto numDeleted = store->deleteKey(key);
  // 创建一个数据写入器，并将删除的数量发送给客户端
  StreamWriter sw(iptr());
  sw.write_value<int64_t>(numDeleted);
  // 发送数据
  sw.send();

  // 返回 true，表示成功解析并处理命令
  return true;
}

bool parse_append_command() {
  std::string key;
  // 尝试从流中读取要追加数据的键，如果失败则返回 false
  if (!stream.read_key(key)) {
    return false;
  }

  std::vector<uint8_t> data;
  // 尝试从流中读取要追加的数据，如果失败则返回 false
  if (!stream.read_payload(data)) {
    return false;
  }

  // 调用存储对象的 append 方法，将数据追加到指定键的值中
  store->append(key, data);

  // 返回 true，表示成功解析并处理命令
  return true;
}

bool parse_multi_get_command() {
  uint64_t key_count = 0;
  // 尝试从流中读取键的数量，如果失败则返回 false
  if (!stream.read_value(key_count)) {
    return false;
  }
  // 检查读取的键数量是否超过最大允许数量，如果超过则抛出异常
  TORCH_CHECK(
      key_count <= MAX_KEY_COUNT,
      "Too many keys with multi_get. ",
      "keys: ",
      key_count,
      ", max: ",
      MAX_KEY_COUNT);

  // 创建一个数据写入器
  StreamWriter sw(iptr());
    for (const auto _ : c10::irange(key_count)) {
      (void)_; // 抑制未使用变量警告，_ 用于迭代计数
      std::string key;
      if (!stream.read_key(key)) {  // 从流中读取键值，若失败则返回false
        return false;
      }

      sw.write_vector(store->get(key));  // 向 StreamWriter 写入存储中键对应的向量数据
    }
    sw.send();  // 发送 StreamWriter 中缓存的所有数据到对端

    return true;  // 函数执行成功，返回true
  }

  bool parse_multi_set_command() {
    uint64_t key_count = 0;
    if (!stream.read_value(key_count)) {  // 从流中读取键的数量，若失败则返回false
      return false;
    }
    TORCH_CHECK(
        key_count <= MAX_KEY_COUNT,  // 检查键的数量不超过最大限制
        "Too many keys with multi_get. ",  // 错误消息：multi_get 操作中键数量过多
        "keys: ", key_count, ", max: ", MAX_KEY_COUNT);

    for (const auto _ : c10::irange(key_count)) {
      (void)_; // 抑制未使用变量警告，_ 用于迭代计数

      std::string key;
      if (!stream.read_key(key)) {  // 从流中读取键值，若失败则返回false
        return false;
      }

      std::vector<uint8_t> newData;
      if (!stream.read_payload(newData))  // 从流中读取有效载荷数据，若失败则返回false
        return false;
      store->set(key, newData);  // 将键值对存储到 store 中
    }

    return true;  // 函数执行成功，返回true
  }

  bool parse_cancel_wait_command() {
    store->clearClientWaitState(iptr());  // 清除存储中客户端等待状态信息

    StreamWriter sw(iptr());  // 创建一个 StreamWriter 对象，绑定当前客户端
    sw.write1((uint8_t)WaitResponseType::WAIT_CANCELED);  // 向 StreamWriter 写入等待取消响应
    sw.send();  // 发送 StreamWriter 中缓存的所有数据到对端

    return true;  // 函数执行成功，返回true
  }

 public:
  explicit UvClient(uv_loop_t* loop, LibUVStoreDaemon* store)
      : UvTcpSocket(loop), store(store) {}  // 构造函数，初始化成员变量 loop 和 store

  static c10::intrusive_ptr<UvClient> make(
      uv_loop_t* loop,
      LibUVStoreDaemon* store) {
    auto res = c10::make_intrusive<UvClient>(loop, store);  // 创建 UvClient 实例
    res->handleReady();  // 处理客户端就绪状态
    return res;  // 返回创建的 UvClient 实例
  }

  c10::intrusive_ptr<UvClient> iptr() {
    return c10::intrusive_ptr<UvClient>::reclaim_copy(this);  // 返回当前 UvClient 实例的智能指针
  }

 protected:
  void onClose() override {
    store->unregisterClient(iptr());  // 注销存储中的客户端信息
  }
};

// 当客户端连接时的回调函数，接受客户端连接并启动读取操作
void LibUVStoreDaemon::onConnect(int status) {
  // 创建一个新的 UvClient 对象，并注册到当前循环中
  auto client = UvClient::make(&loop, this);
  registerClient(client);

  try {
    // 尝试接受 TCP 连接并启动读取操作
    tcpServer->accept(client);
    client->startRead();
  } catch (std::exception& e) {
    // 如果发生异常，记录警告信息并关闭客户端连接
    C10D_WARNING("Failed to accept client due to {}", e.what());
    client->close();
  }
}

// 处理退出请求的函数，关闭相应的 libuv 句柄并停止事件循环
void LibUVStoreDaemon::onExitRequest() {
  // 输出调试信息，表示存储退出请求
  C10D_DEBUG("Store exit requested\n");
  // 关闭退出句柄对应的 libuv 句柄
  uv_close((uv_handle_t*)&exit_handle, nullptr);
  // 停止 libuv 事件循环
  uv_stop(&loop);
}

// 初始化函数，根据给定的 TCPStoreOptions 进行初始化
void LibUVStoreDaemon::init(const TCPStoreOptions& opts) {
  if (opts.masterListenFd.has_value()) {
    // 如果给定了监听文件描述符，则使用该描述符创建 TCP 服务器对象
    tcpServer = UvTcpServer::makeWithSocket(&loop, *opts.masterListenFd);
  } else {
    try {
      // 否则，尝试使用指定的端口号创建 TCP 服务器对象，启用 IPv6 支持
      tcpServer = UvTcpServer::makeWithPort(&loop, opts.port, /*useIpv6=*/true);
    } catch (std::exception& ex) {
      // 如果创建失败，则记录信息并尝试使用 IPv4 地址
      C10D_INFO(
          "Failed to bind to ipv6 address, trying ipv4. Error: {}", ex.what());
      tcpServer =
          UvTcpServer::makeWithPort(&loop, opts.port, /*useIpv6=*/false);
    }
  }
  // 设置 TCP 服务器连接回调函数为 onConnect 方法
  tcpServer->setOnConnectCallback(
      [this](auto status) { this->onConnect(status); });

  // 记录实际使用的端口号
  port_ = tcpServer->port();
  // 检查实际端口是否与预期的端口匹配
  TORCH_CHECK(
      port_ == opts.port || opts.port == 0, // zero means use any port
      "listen fd ",
      *opts.masterListenFd,
      " is bound to port ",
      port_,
      ", expected to be bound to port ",
      opts.port);
}

// 构造函数，初始化端口号并初始化 libuv 循环
LibUVStoreDaemon::LibUVStoreDaemon(int port) : port_(port) {
  // 初始化 libuv 循环，若失败则抛出异常
  TORCH_CHECK(uv_loop_init(&loop) == 0, "Failed to init uv loop");
  // 初始化 libuv 异步事件句柄，若失败则抛出异常
  TORCH_CHECK(
      uv_async_init(&loop, &exit_handle, LibUVStoreDaemon::on_exit_request) ==
          0,
      "Failed to init uv async event");
  // 设置退出句柄的关联数据为当前对象
  uv_handle_set_data((uv_handle_t*)&exit_handle, this);
}

// 析构函数，清理 libuv 相关资源
LibUVStoreDaemon::~LibUVStoreDaemon() {
  if (!is_running()) {
    // 若事件循环未运行，则直接关闭退出句柄并执行部分事件循环
    uv_close((uv_handle_t*)&exit_handle, nullptr);
    uv_run(&loop, UV_RUN_NOWAIT);
    // 关闭 libuv 循环，若失败则抛出异常
    TORCH_CHECK(uv_loop_close(&loop) == 0, "loop cleanup didn't work");
  } else {
    // 若事件循环运行中，则执行对象的释放操作
    dispose();
  }
}

// 返回当前存储守护进程使用的端口号
uint16_t LibUVStoreDaemon::port() const {
  return port_;
}

// 打印 libuv 活动句柄的回调函数
void LibUVStoreDaemon::print_active_handles(uv_handle_t* handle, void* arg) {
  // 输出调试信息，显示句柄类型及其活动状态和关闭状态
  C10D_DEBUG(
      "UV live handle type {} active:{} is-closing:{}",
      (int)handle->type,
      uv_is_active(handle),
      uv_is_closing(handle));
}

// 启动 libuv 主事件循环
void LibUVStoreDaemon::run() {
  // 设置当前线程的名称
  c10::setThreadName("pt_tcpstore_uv");

  // 输出调试信息，表示 libuv 主循环正在运行
  C10D_DEBUG("Uv main loop running");
  // 运行 libuv 主事件循环，获取返回结果
  int res = uv_run(&loop, UV_RUN_DEFAULT);
  // 若运行结束，输出调试信息
  if (res) {
    C10D_DEBUG("UV main loop done: res:{}", res);
  }

  // 检查是否启用调试模式
  bool debug_enabled =
      c10d::detail::isLogLevelEnabled(c10d::detail::LogLevel::Debug);

  // 若调试模式启用，则输出当前 libuv 活动句柄信息
  if (debug_enabled) {
    C10D_DEBUG("Walking live handles prior to closing clients");
    uv_walk(&loop, LibUVStoreDaemon::print_active_handles, nullptr);
  }

  // 关闭所有客户端连接
  for (const auto& client : clients_) {
    client->close();
  }
  // 关闭 TCP 服务器
  tcpServer->close();

  // 若调试模式启用，则再次输出 libuv 活动句柄信息
  if (debug_enabled) {
    C10D_DEBUG("Walking live handles after closing clients");
    uv_walk(&loop, LibUVStoreDaemon::print_active_handles, nullptr);
  }

  // 进入无限循环等待
  while (true) {
    # 调用 uv_loop_close 函数关闭事件循环，并获取返回结果
    res = uv_loop_close(&loop);
    # 如果返回结果为 0，表示成功关闭事件循环，退出循环
    if (res == 0) {
      break;
    }
    # 如果关闭事件循环失败，记录错误信息，包括返回结果、错误码名称和错误描述
    C10D_INFO(
        "uv_loop_close failed with:{} errn:{} desc:{}",
        res,
        uv_err_name(res),
        uv_strerror(res));
    # 继续运行事件循环直到完成所有未处理事件
    res = uv_run(&loop, UV_RUN_NOWAIT);
    # 如果运行事件循环的返回结果不为 0，说明仍有未处理的事件，等待 500 毫秒
    if (res != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
  # 打印事件循环清理完成的信息
  C10D_INFO("uv_loop cleanup finished.");
}

void LibUVStoreDaemon::stop() {
  // 发送异步退出信号给 libuv 事件循环
  int res = uv_async_send(&exit_handle);
  // 如果发送失败，记录警告日志，并包含错误码、错误名称和描述信息
  if (res) {
    C10D_WARNING(
        "uv_async_send failed with:{} errn:{} desc:{}\n",
        res,
        uv_err_name(res),
        uv_strerror(res));
  }
}

bool LibUVStoreDaemon::isMiscellaneousClient(
    const c10::intrusive_ptr<UvHandle>& client) {
  // 检查 client 是否在 miscellaneousClients_ 中，如果在则移除并返回 true
  if (miscellaneousClients_.find(client) != miscellaneousClients_.end()) {
    miscellaneousClients_.erase(client);
    return true;
  }
  // 否则返回 false
  return false;
}

void LibUVStoreDaemon::registerClient(
    const c10::intrusive_ptr<UvHandle>& client) {
  // 向 clients_ 集合中注册客户端
  clients_.insert(client);
  // 向 miscellaneousClients_ 集合中注册客户端
  miscellaneousClients_.insert(client);
}

void LibUVStoreDaemon::unregisterClient(
    const c10::intrusive_ptr<UvHandle>& client) {
  // 从 clients_ 集合中移除客户端
  clients_.erase(client);
  // 如果客户端也在 miscellaneousClients_ 集合中，则从中移除
  if (miscellaneousClients_.find(client) != miscellaneousClients_.end()) {
    miscellaneousClients_.erase(client);
  }
  // 清除客户端的等待状态
  clearClientWaitState(client);
}

void LibUVStoreDaemon::clearClientWaitState(
    const c10::intrusive_ptr<UvHandle>& client) {
  // 如果 keysAwaited_ 集合中没有这个 client，直接返回
  if (keysAwaited_.find(client) == keysAwaited_.end()) {
    return;
  }
  // 否则，从 keysAwaited_ 集合中移除这个 client
  keysAwaited_.erase(client);
  // 遍历 waitingSockets_ 映射表
  for (auto it = waitingSockets_.begin(); it != waitingSockets_.end();) {
    // 遍历每个映射表中的客户端列表
    for (auto vecIt = it->second.begin(); vecIt != it->second.end();) {
      // 如果找到了当前 client，则从列表中移除
      if (*vecIt == client) {
        vecIt = it->second.erase(vecIt);
      } else {
        ++vecIt;
      }
    }
    // 如果当前映射表的客户端列表为空，则从 waitingSockets_ 中移除
    if (it->second.empty()) {
      it = waitingSockets_.erase(it);
    } else {
      ++it;
    }
  }
}

void LibUVStoreDaemon::set(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  // 设置键为 key 的值为 value
  tcpStore_[key] = value;
  // 在设置完成后唤醒等待中的客户端
  wakeupWaitingClients(key);
}

const std::vector<uint8_t>& LibUVStoreDaemon::compareAndSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& newValue) {
  // 查找键为 key 的值在 tcpStore_ 中的位置
  auto pos = tcpStore_.find(key);
  // 如果键不存在
  if (pos == tcpStore_.end()) {
    // 如果期望值为空，则直接设置新值并返回
    if (expectedValue.empty()) {
      tcpStore_[key] = newValue;
      wakeupWaitingClients(key);
      return newValue;
    } else {
      // 否则返回期望值，表明操作失败
      // TODO: 这段代码路径不理想，因为在键不存在时我们在向调用者返回假数据。应该想出一个更好的解决方案。
      // 在键不存在时返回期望值可能更合理。
      wakeupWaitingClients(key);
      return expectedValue;
    }
  } else {
    // 如果键存在，比较其值和期望值，如果相同则设置新值并返回新值
    if (pos->second == expectedValue) {
      pos->second = newValue;
    }
    wakeupWaitingClients(key);
    return pos->second;
  }
}

const std::vector<uint8_t>& LibUVStoreDaemon::get(const std::string& key) {
  // 静态变量，用于在键不存在时返回一个默认值
  static std::vector<uint8_t> missing_key;
  // 返回键为 key 的值，如果不存在则返回默认值
  return tcpStore_.count(key) ? tcpStore_.at(key) : missing_key;
}

int64_t LibUVStoreDaemon::add(const std::string& key, int64_t addVal) {
  std::vector<uint8_t> oldData;
  // 查找键为 key 的值在 tcpStore_ 中的位置
  auto it = tcpStore_.find(key);
  // 如果找到了键
  if (it != tcpStore_.end()) {
    // 保存旧数据
    oldData = it->second;
    // 将数据转换为字符指针和长度
    auto buf = reinterpret_cast<const char*>(it->second.data());
    auto len = it->second.size();
    // TODO: 未完成的方法
    // 该方法可能用于增加一个 int64_t 值到当前值，但是代码截断了，无法得知实现细节。
    // 将 buf 中的字符串转换为长整型数值，并加到 addVal 上
    addVal += std::stoll(std::string(buf, len));
  }
  // 将 addVal 转换为字符串
  auto addValStr = std::to_string(addVal);
  // 将 addValStr 中的字符转换为字节向量 newData
  std::vector<uint8_t> newData =
      std::vector<uint8_t>(addValStr.begin(), addValStr.end());
  // 将 newData 存储到 tcpStore_ 中，使用 key 作为索引
  tcpStore_[key] = newData;

  // 当执行“add”操作时，唤醒所有等待的客户端
  wakeupWaitingClients(key);

  // 返回 addVal 的值作为操作结果
  return addVal;
}

// 检查给定的键是否都存在于 tcpStore_ 中
bool LibUVStoreDaemon::checkKeys(const std::vector<std::string>& keys) {
  // 使用 std::all_of 算法检查所有 keys 中的键是否都存在于 tcpStore_ 中
  return std::all_of(keys.begin(), keys.end(), [&](const std::string& s) {
    return tcpStore_.count(s) > 0;
  });
}

// 等待所有指定的键存在于 tcpStore_ 中，如果存在则返回 true，否则添加到等待队列并返回 false
bool LibUVStoreDaemon::waitKeys(
    const std::vector<std::string>& keys,
    const c10::intrusive_ptr<UvHandle>& client) {
  // 如果所有键都存在于 tcpStore_ 中，则直接返回 true
  if (checkKeys(keys)) {
    return true;
  }
  // 计算需要等待的键的数量，并将客户端添加到等待队列中
  int numKeysToAwait = 0;
  for (auto& key : keys) {
    // 只计算尚未设置的键的数量
    if (tcpStore_.find(key) == tcpStore_.end()) {
      waitingSockets_[key].push_back(client);
      numKeysToAwait++;
    }
  }
  // 记录每个客户端需要等待的键的数量
  keysAwaited_[client] = numKeysToAwait;
  return false;
}

// 返回 tcpStore_ 中键值对的数量
int64_t LibUVStoreDaemon::size() {
  return tcpStore_.size();
}

// 删除并返回 tcpStore_ 中指定键的条目数量
int64_t LibUVStoreDaemon::deleteKey(const std::string& key) {
  return tcpStore_.erase(key);
}

// 向 tcpStore_ 中指定键追加数据
void LibUVStoreDaemon::append(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  std::vector<uint8_t> oldData;
  // 查找键是否已存在于 tcpStore_ 中
  auto it = tcpStore_.find(key);
  // 如果存在，则在现有数据后追加新数据；否则直接设置新数据
  if (it != tcpStore_.end()) {
    it->second.insert(it->second.end(), value.begin(), value.end());
  } else {
    tcpStore_[key] = value;
  }

  // 因为在追加数据时不应该有等待的客户端，所以直接唤醒等待队列中的客户端
  wakeupWaitingClients(key);
}

// 唤醒等待特定键的客户端
void LibUVStoreDaemon::wakeupWaitingClients(const std::string& key) {
  // 查找等待队列中是否有特定键的客户端
  auto socketsToWait = waitingSockets_.find(key);
  if (socketsToWait != waitingSockets_.end()) {
    // 遍历等待队列中的客户端
    for (const auto& client : socketsToWait->second) {
      // 减少客户端需要等待的键的数量，并在其减少到零时发送停止等待信号
      if (--keysAwaited_[client] == 0) {
        StreamWriter sw(client->iptr());
        sw.write1((uint8_t)WaitResponseType::STOP_WAITING);
        sw.send();
      }
    }
    // 从等待队列中移除已处理的键
    waitingSockets_.erase(socketsToWait);
  }
}

#endif

// 创建基于 LibUV 的 TCP 存储后端的后台线程
std::unique_ptr<BackgroundThread> create_libuv_tcpstore_backend(
    const TCPStoreOptions& opts) {
#ifdef TORCH_USE_LIBUV
  // 创建 LibUVStoreDaemon 的唯一指针，并初始化
  auto res = std::make_unique<LibUVStoreDaemon>(opts.port);
  res->init(opts);
  return res;
#else
  // 如果未启用 LibUV，则抛出错误信息
  TORCH_CHECK(false, "LibUV TCPStore implementation missing");
#endif
}

// 检查是否可用 LibUV TCPStore 后端
bool is_libuv_tcpstore_backend_available() {
#ifdef TORCH_USE_LIBUV
  return true;
#else
  return false;
#endif
}

} // namespace c10d::detail
```