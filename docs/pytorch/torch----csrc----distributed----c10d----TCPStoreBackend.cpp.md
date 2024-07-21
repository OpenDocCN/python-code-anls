# `.\pytorch\torch\csrc\distributed\c10d\TCPStoreBackend.cpp`

```
#include <c10/util/irange.h>
#include <fcntl.h>
#include <algorithm>
#include <array>
#include <system_error>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#include <io.h>
#include <winsock2.h>
#else
#include <poll.h>
#include <unistd.h>
#endif

#include <c10/util/thread_name.h>
#include <torch/csrc/distributed/c10d/TCPStoreBackend.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

#ifdef _WIN32
#include <torch/csrc/distributed/c10d/WinSockUtils.hpp>
#else
#include <torch/csrc/distributed/c10d/UnixSockUtils.hpp>
#endif

#include <torch/csrc/distributed/c10d/socket.h>

namespace c10d::detail {

// Background thread parent class methods
BackgroundThread::BackgroundThread() = default;

BackgroundThread::~BackgroundThread() = default;

// WARNING:
// Since we rely on the subclass for the daemon thread clean-up, we cannot
// destruct our member variables in the destructor. The subclass must call
// dispose() in its own destructor.
void BackgroundThread::dispose() {
  // Stop the run
  stop();
  // Join the thread
  daemonThread_.join();
}

void BackgroundThread::start() {
  // Start the background thread by launching a new thread that runs the 'run' method of the current object
  daemonThread_ = std::thread{&BackgroundThread::run, this};
  // Set the flag to indicate that the thread is running
  is_running_.store(true);
}

// Separate thread that is only launched on master
// TCPStoreMasterDaemon 类的定义，继承自 BackgroundThread 类
class TCPStoreMasterDaemon : public BackgroundThread {
 public:
  // 构造函数，接受一个 Socket 并移动其所有权
  explicit TCPStoreMasterDaemon(Socket&& storeListenSocket);

  // 析构函数，用于清理资源
  ~TCPStoreMasterDaemon() override;

  // 返回监听端口号
  uint16_t port() const override;

 protected:
  // 后台线程运行的主体函数
  void run() override;
  // 停止线程的方法
  void stop() override;

 private:
  // 初始化停止信号
  void initStopSignal();
  // 关闭停止信号
  void closeStopSignal();

  // 查询所有注册的文件描述符
  void queryFds(std::vector<struct pollfd>& fds);
  // 查询指定 socket
  void query(int socket);

  // 清除 socket 的等待状态
  void clearSocketWaitState(int socket);

  // 验证处理程序的有效性
  void validateHandler(int socket);
  // 设置处理程序
  void setHandler(int socket);
  // 比较设置处理程序
  void compareSetHandler(int socket);
  // 添加处理程序
  void addHandler(int socket);
  // 获取处理程序
  void getHandler(int socket) const;
  // 检查处理程序
  void checkHandler(int socket) const;
  // 获取键数量处理程序
  void getNumKeysHandler(int socket) const;
  // 删除处理程序
  void deleteHandler(int socket);
  // 等待处理程序
  void waitHandler(int socket);
  // 追加处理程序
  void appendHandler(int socket);
  // 多键获取处理程序
  void multiGetHandler(int socket);
  // 多键设置处理程序
  void multiSetHandler(int socket);
  // 取消等待处理程序
  void cancelWaitHandler(int socket);
  // 添加杂项 socket
  void addMiscellaneousSocket(int socket);
  // 移除杂项 socket
  void removeMiscellaneousSocket(int socket);
  // 检查是否为杂项 socket
  bool isMiscellaneousSocket(int socket);

  // 检查键是否有效的辅助函数
  bool checkKeys(const std::vector<std::string>& keys) const;
  // 唤醒等待中的客户端的辅助函数，在 setHandler 和 getHandler 中使用
  void wakeupWaitingClients(const std::string& key);
  // 执行设置操作的辅助函数
  void doSet(const std::string& key, const std::vector<uint8_t>& newData);

  // TCP 存储，从键到数据的映射
  std::unordered_map<std::string, std::vector<uint8_t>> tcpStore_;
  // 等待某个键的套接字列表，从键到套接字的映射
  std::unordered_map<std::string, std::vector<int>> waitingSockets_;
  // 套接字等待的键数映射
  std::unordered_map<int, size_t> keysAwaited_;
  // 杂项套接字集合
  std::unordered_set<int> miscellaneousSockets_;

  // 存储监听套接字
  Socket storeListenSocket_;
  // 其他套接字的向量
  std::vector<Socket> sockets_{};
#ifdef _WIN32
  // Windows 平台的超时检查时间
  const std::chrono::milliseconds checkTimeout_ = std::chrono::milliseconds{10};
  // 停止事件句柄
  HANDLE ghStopEvent_{};
#else
  // 控制管道文件描述符数组
  std::array<int, 2> controlPipeFd_{{-1, -1}};
#endif
};

// TCPStoreMasterDaemon 的构造函数实现，移动构造 storeListenSocket 到成员变量中
TCPStoreMasterDaemon::TCPStoreMasterDaemon(Socket&& storeListenSocket)
    : storeListenSocket_{std::move(storeListenSocket)} {
  // 初始化停止信号
  initStopSignal();
}

// TCPStoreMasterDaemon 的析构函数实现
TCPStoreMasterDaemon::~TCPStoreMasterDaemon() {
  // 清理资源
  dispose();
  // 现在可以安全地清理未关闭的套接字
  sockets_.clear();
  // 关闭剩余的控制管道
  closeStopSignal();
}

// 返回监听端口号的实现
std::uint16_t TCPStoreMasterDaemon::port() const {
  return storeListenSocket_.port();
}

#ifdef _WIN32
// Windows 平台下初始化停止信号的实现
void TCPStoreMasterDaemon::initStopSignal() {
  ghStopEvent_ = CreateEvent(NULL, TRUE, FALSE, NULL);
  if (ghStopEvent_ == NULL) {
    TORCH_CHECK(
        false,
        "Failed to create the control pipe to start the "
        "BackgroundThread run");
  }
}

// Windows 平台下关闭停止信号的实现
void TCPStoreMasterDaemon::closeStopSignal() {
  CloseHandle(ghStopEvent_);
}

// Windows 平台下停止线程的实现
void TCPStoreMasterDaemon::stop() {
  SetEvent(ghStopEvent_);
}

#else
void TCPStoreMasterDaemon::initStopSignal() {
  // 创建控制管道，用于与后台线程进行通信
  if (pipe(controlPipeFd_.data()) == -1) {
    // 如果创建管道失败，则抛出错误并显示相应信息
    TORCH_CHECK(
        false,
        "Failed to create the control pipe to start the "
        "BackgroundThread run");
  }
}

void TCPStoreMasterDaemon::closeStopSignal() {
  // 关闭控制管道的所有文件描述符
  for (int fd : controlPipeFd_) {
    if (fd != -1) {
      ::close(fd);
    }
  }
}

void TCPStoreMasterDaemon::stop() {
  // 向控制管道的写入端发送停止信号
  if (controlPipeFd_[1] != -1) {
    ssize_t written_bytes = -1;
    while (true) {
      written_bytes = ::write(controlPipeFd_[1], "\0", 1);
      if (written_bytes < 0) {
        // 如果写入失败且是暂时性错误，则继续尝试写入
        if (errno == EAGAIN) {
          continue;
        }
        // 否则抛出错误并显示相应信息
        TORCH_CHECK(false, "Failed to write the control pipe:", errno);
      }
      break;
    }
    if (written_bytes == 0) {
      // 如果写入的字节数为0，则表示写入失败，抛出相应错误信息
      TORCH_CHECK(false, "Failed to write the control pipe");
    }

    // 关闭管道的写入端
    ::close(controlPipeFd_[1]);
    controlPipeFd_[1] = -1;
  }
}

void TCPStoreMasterDaemon::queryFds(std::vector<struct pollfd>& fds) {
  // 跳过fds[0]和fds[1]的处理说明
  // fds[0]是主监听套接字
  // fds[1]是控制管道的读取文件描述符，在Windows平台下不使用
  for (size_t fdIdx = CONNECT_SOCKET_OFFSET; fdIdx < fds.size(); ++fdIdx) {
    if (fds[fdIdx].revents == 0) {
      // 如果没有事件发生在该套接字上，则继续下一轮循环
      continue;
    }

    // 查询发生事件的套接字
    try {
      query(fds[fdIdx].fd);
    } catch (...) {
      // 处理查询过程中可能发生的异常，一般是接收/发送时发生异常，
      // 表示对端套接字已关闭。如果是正常退出导致的关闭，则存储可以继续执行。
      // 否则，如果是其他异常，则其他连接在使用存储时也会发生异常。
      // 在这里关闭发生异常的连接。
      clearSocketWaitState(fds[fdIdx].fd);

      // 移除发生异常的套接字及其相关信息
      fds.erase(fds.begin() + fdIdx);
      sockets_.erase(sockets_.begin() + fdIdx - CONNECT_SOCKET_OFFSET);
      --fdIdx;
      continue;
    }
  }
}

void TCPStoreMasterDaemon::clearSocketWaitState(int socket) {
  // 清除所有与关闭套接字相关的跟踪状态
  for (auto it = waitingSockets_.begin(); it != waitingSockets_.end();) {
    for (auto vecIt = it->second.begin(); vecIt != it->second.end();) {
      if (*vecIt == socket) {
        // 如果找到了等待的套接字，则从列表中删除
        vecIt = it->second.erase(vecIt);
      } else {
        ++vecIt;
      }
    }
    if (it->second.empty()) {
      // 如果某个key对应的等待列表为空，则删除该key
      it = waitingSockets_.erase(it);
    } else {
      ++it;
    }
  }

  // 清除所有等待关闭的套接字关联的键信息
  for (auto it = keysAwaited_.begin(); it != keysAwaited_.end();) {
    if (it->first == socket) {
      // 如果找到了与关闭套接字相关的键信息，则删除该键
      it = keysAwaited_.erase(it);
    } else {
      ++it;
    }
  }
}

// query communicates with the worker. The format
// of the query is as follows:
// type of query | size of arg1 | arg1 | size of arg2 | arg2 | ...
// or, in the case of wait
// type of query | number of args | size of arg1 | arg1 | ...
void TCPStoreMasterDaemon::query(int socket) {
  // 接收查询类型
  QueryType qt;
  tcputil::recvBytes<QueryType>(socket, &qt, 1);

  // 如果是杂项 socket
  if (isMiscellaneousSocket(socket)) {
    // 移除杂项 socket
    removeMiscellaneousSocket(socket);
    // 如果查询类型为 VALIDATE，则调用验证处理器
    if (qt == QueryType::VALIDATE) {
      validateHandler(socket);
    } else {
      // 真正的杂项客户端：第一个消息不是 VALIDATE
      TORCH_CHECK(
          false, "Miscellaneous client without VALIDATE query is detected");
    }
  } else if (qt == QueryType::SET) {
    // 如果查询类型为 SET，则调用设置处理器
    setHandler(socket);

  } else if (qt == QueryType::COMPARE_SET) {
    // 如果查询类型为 COMPARE_SET，则调用比较设置处理器
    compareSetHandler(socket);

  } else if (qt == QueryType::ADD) {
    // 如果查询类型为 ADD，则调用添加处理器
    addHandler(socket);

  } else if (qt == QueryType::GET) {
    // 如果查询类型为 GET，则调用获取处理器
    getHandler(socket);

  } else if (qt == QueryType::CHECK) {
    // 如果查询类型为 CHECK，则调用检查处理器
    checkHandler(socket);

  } else if (qt == QueryType::WAIT) {
    // 如果查询类型为 WAIT，则调用等待处理器
    waitHandler(socket);

  } else if (qt == QueryType::GETNUMKEYS) {
    // 如果查询类型为 GETNUMKEYS，则调用获取键数处理器
    getNumKeysHandler(socket);

  } else if (qt == QueryType::DELETE_KEY) {
    // 如果查询类型为 DELETE_KEY，则调用删除处理器
    deleteHandler(socket);
  } else if (qt == QueryType::APPEND) {
    // 如果查询类型为 APPEND，则调用追加处理器
    appendHandler(socket);
  } else if (qt == QueryType::MULTI_GET) {
    // 如果查询类型为 MULTI_GET，则调用多重获取处理器
    multiGetHandler(socket);
  } else if (qt == QueryType::MULTI_SET) {
    // 如果查询类型为 MULTI_SET，则调用多重设置处理器
    multiSetHandler(socket);
  } else if (qt == QueryType::CANCEL_WAIT) {
    // 如果查询类型为 CANCEL_WAIT，则调用取消等待处理器
    cancelWaitHandler(socket);
  } else {
    // 如果是未预期的查询类型，抛出异常
    TORCH_CHECK(false, "Unexpected query type");
  }
}

void TCPStoreMasterDaemon::wakeupWaitingClients(const std::string& key) {
  // 查找等待中的 socket 列表
  auto socketsToWait = waitingSockets_.find(key);
  // 如果找到了
  if (socketsToWait != waitingSockets_.end()) {
    // 遍历该 key 对应的所有 socket
    for (int socket : socketsToWait->second) {
      // 如果该 socket 对应的等待数减到 0
      if (--keysAwaited_[socket] == 0) {
        // 发送停止等待响应给 socket
        tcputil::sendValue<WaitResponseType>(
            socket, WaitResponseType::STOP_WAITING);
      }
    }
    // 从等待 socket 的列表中移除这个 key
    waitingSockets_.erase(socketsToWait);
  }
}

void TCPStoreMasterDaemon::doSet(
    const std::string& key,
    const std::vector<uint8_t>& newData) {
  // 更新键对应的数据
  tcpStore_[key] = newData;
  // 在“设置”操作后唤醒所有等待的客户端
  wakeupWaitingClients(key);
}

void TCPStoreMasterDaemon::validateHandler(int socket) {
  // 接收验证数
  uint32_t validateNumber = 0;
  tcputil::recvBytes<uint32_t>(socket, &validateNumber, 1);
  // 如果验证数不等于验证魔数
  if (validateNumber != detail::validationMagicNumber) {
    // 抛出异常：检测到验证不正确的杂项客户端查询
    TORCH_CHECK(
        false,
        "Miscellaneous client with incorrect VALIDATE query is detected");
  }
}

void TCPStoreMasterDaemon::setHandler(int socket) {
  // 接收键
  std::string key = tcputil::recvString(socket);
  // 接收新数据
  std::vector<uint8_t> newData = tcputil::recvVector<uint8_t>(socket);
  // 执行设置操作
  doSet(key, newData);
}

void TCPStoreMasterDaemon::compareSetHandler(int socket) {
  // 接收键
  std::string key = tcputil::recvString(socket);
  // 接收当前值
  std::vector<uint8_t> currentValue = tcputil::recvVector<uint8_t>(socket);
  // 接收新值
  std::vector<uint8_t> newValue = tcputil::recvVector<uint8_t>(socket);

  // 在 TCP 存储中查找键
  auto pos = tcpStore_.find(key);
  // 如果没找到
  if (pos == tcpStore_.end()) {
    // 如果当前值为空，则设置新值
    if (currentValue.empty()) {
      tcpStore_[key] = newValue;
      // 向 socket 发送新值
      tcputil::sendVector<uint8_t>(socket, newValue);
      ```
    } else {
      // 如果条件不满足，执行此代码块
      // TODO: 当键不存在时，当前代码路径不理想，因为我们向调用者“说谎”了。
      // 我们需要想出一个有效的解决方案。
      // 使用 tcputil 发送当前值的字节向量到 socket
      tcputil::sendVector<uint8_t>(socket, currentValue);
    }
  } else {
    // 如果条件满足，执行此代码块
    // 检查键对应的值是否等于当前值
    if (pos->second == currentValue) {
      // 如果相等，将键对应的值更新为新值（移动语义）
      pos->second = std::move(newValue);
    }
    // 使用 tcputil 发送键对应的值的字节向量到 socket
    tcputil::sendVector<uint8_t>(socket, pos->second);
  }
void TCPStoreMasterDaemon::addHandler(int socket) {
    // 接收字符串类型的键值
    std::string key = tcputil::recvString(socket);
    // 接收 int64_t 类型的增量值
    int64_t addVal = tcputil::recvValue<int64_t>(socket);

    // 查找键在 tcpStore_ 中的位置
    auto it = tcpStore_.find(key);
    // 如果键存在
    if (it != tcpStore_.end()) {
        // 将数据转换为 const char* 类型的缓冲区
        auto buf = reinterpret_cast<const char*>(it->second.data());
        // 缓冲区的长度
        auto len = it->second.size();
        // 将缓冲区中的字符串转换为 int64_t 类型，与增量值相加
        addVal += std::stoll(std::string(buf, len));
    }

    // 将增加后的值转换为字符串
    auto addValStr = std::to_string(addVal);
    // 创建新的 uint8_t 类型的数据向量，用于存储新的值
    std::vector<uint8_t> newData =
        std::vector<uint8_t>(addValStr.begin(), addValStr.end());
    // 更新 tcpStore_ 中的键对应的值
    tcpStore_[key] = newData;

    // 向客户端发送新的值
    tcputil::sendValue<int64_t>(socket, addVal);

    // 在执行“add”操作时，唤醒所有等待的客户端
    wakeupWaitingClients(key);
}

void TCPStoreMasterDaemon::getHandler(int socket) const {
    // 接收字符串类型的键值
    std::string key = tcputil::recvString(socket);
    // 获取键对应的数据
    auto data = tcpStore_.at(key);
    // 将数据发送给客户端
    tcputil::sendVector<uint8_t>(socket, data);
}

void TCPStoreMasterDaemon::getNumKeysHandler(int socket) const {
    // 发送 tcpStore_ 中键的数量给客户端
    tcputil::sendValue<int64_t>(socket, tcpStore_.size());
}

void TCPStoreMasterDaemon::deleteHandler(int socket) {
    // 接收字符串类型的键值
    std::string key = tcputil::recvString(socket);
    // 删除 tcpStore_ 中的键值对，并返回删除的数量给客户端
    auto numDeleted = tcpStore_.erase(key);
    tcputil::sendValue<int64_t>(socket, numDeleted);
}

void TCPStoreMasterDaemon::checkHandler(int socket) const {
    // 接收 SizeType 类型的参数数量
    SizeType nargs = 0;
    tcputil::recvBytes<SizeType>(socket, &nargs, 1);
    // 创建存储字符串键的向量
    std::vector<std::string> keys(nargs);
    // 接收所有的字符串键
    for (const auto i : c10::irange(nargs)) {
        keys[i] = tcputil::recvString(socket);
    }

    // 检查所有接收到的键是否存在
    if (checkKeys(keys)) {
        // 如果存在所有键，则向客户端发送 READY 响应
        tcputil::sendValue<CheckResponseType>(socket, CheckResponseType::READY);
    } else {
        // 如果存在未找到的键，则向客户端发送 NOT_READY 响应
        tcputil::sendValue<CheckResponseType>(socket, CheckResponseType::NOT_READY);
    }
}

void TCPStoreMasterDaemon::waitHandler(int socket) {
    // 接收 SizeType 类型的参数数量
    SizeType nargs = 0;
    tcputil::recvBytes<SizeType>(socket, &nargs, 1);
    // 创建存储字符串键的向量
    std::vector<std::string> keys(nargs);
    // 接收所有的字符串键
    for (const auto i : c10::irange(nargs)) {
        keys[i] = tcputil::recvString(socket);
    }

    // 检查所有接收到的键是否存在
    if (checkKeys(keys)) {
        // 如果存在所有键，则向客户端发送 STOP_WAITING 响应
        tcputil::sendValue<WaitResponseType>(
            socket, WaitResponseType::STOP_WAITING);
    } else {
        int numKeysToAwait = 0;
        for (auto& key : keys) {
            // 只计算未设置的键的数量
            if (tcpStore_.find(key) == tcpStore_.end()) {
                // 将等待的套接字添加到等待列表中
                waitingSockets_[key].push_back(socket);
                numKeysToAwait++;
            }
        }
        // 记录套接字等待的键的数量
        keysAwaited_[socket] = numKeysToAwait;
    }
}

void TCPStoreMasterDaemon::appendHandler(int socket) {
    // 接收字符串类型的键值
    std::string key = tcputil::recvString(socket);
    // 接收 uint8_t 类型的数据向量
    std::vector<uint8_t> newData = tcputil::recvVector<uint8_t>(socket);
    // 查找键在 tcpStore_ 中的位置
    auto it = tcpStore_.find(key);
    // 如果键存在
    if (it != tcpStore_.end()) {
        // 在现有数据的末尾插入新数据
        it->second.insert(it->second.end(), newData.begin(), newData.end());
    } else {
        // 如果键不存在，则将新数据插入 tcpStore_ 中
        tcpStore_[key] = newData;
    }
    // 如果执行追加操作，不应该有等待的客户端，所以一切正常
    wakeupWaitingClients(key);
}
void TCPStoreMasterDaemon::multiGetHandler(int socket) {
    // 接收请求的参数个数
    SizeType nargs = 0;
    tcputil::recvBytes<SizeType>(socket, &nargs, 1);
    // 遍历请求中的每一个键，获取对应的数据并发送回客户端
    for (const auto i : c10::irange(nargs)) {
        auto key = tcputil::recvString(socket);
        auto& data = tcpStore_.at(key); // 获取键对应的数据
        tcputil::sendVector<uint8_t>(socket, data, i < (nargs - 1)); // 发送数据到客户端
    }
}

void TCPStoreMasterDaemon::multiSetHandler(int socket) {
    // 接收请求的参数个数
    SizeType nargs = 0;
    tcputil::recvBytes<SizeType>(socket, &nargs, 1);
    // 遍历请求中的每一个键值对，将其存储到 TCP 存储中
    for (auto _ : c10::irange(nargs)) {
        (void)_; // 抑制未使用变量警告
        auto key = tcputil::recvString(socket); // 接收键
        auto value = tcputil::recvVector<uint8_t>(socket); // 接收值
        doSet(key, value); // 调用存储方法存储键值对
    }
}

void TCPStoreMasterDaemon::cancelWaitHandler(int socket) {
    clearSocketWaitState(socket); // 清除套接字等待状态

    // 向客户端发送取消等待的更新消息
    tcputil::sendValue<WaitResponseType>(
        socket, detail::WaitResponseType::WAIT_CANCELED);
}

bool TCPStoreMasterDaemon::checkKeys(
    const std::vector<std::string>& keys) const {
    // 检查给定的键是否都存在于 TCP 存储中
    return std::all_of(keys.begin(), keys.end(), [this](const std::string& s) {
        return tcpStore_.count(s) > 0; // 使用 lambda 表达式检查键是否存在
    });
}

void TCPStoreMasterDaemon::addMiscellaneousSocket(int socket) {
    // 如果套接字不在杂项套接字集合中，则添加进去
    if (miscellaneousSockets_.find(socket) == miscellaneousSockets_.end()) {
        miscellaneousSockets_.insert(socket);
    }
}

void TCPStoreMasterDaemon::removeMiscellaneousSocket(int socket) {
    // 移除指定的杂项套接字
    auto it = miscellaneousSockets_.find(socket);
    if (it != miscellaneousSockets_.end()) {
        miscellaneousSockets_.erase(it);
    }
}

bool TCPStoreMasterDaemon::isMiscellaneousSocket(int socket) {
    // 检查套接字是否在杂项套接字集合中
    return miscellaneousSockets_.find(socket) != miscellaneousSockets_.end();
}

#ifdef _WIN32
void TCPStoreMasterDaemon::run() {
    std::vector<struct pollfd> fds;
    tcputil::addPollfd(fds, storeListenSocket_.handle(), POLLIN);

    // 接收查询
    bool finished = false;
    while (!finished) {
        for (const auto i : c10::irange(sockets_.size())) {
            fds[i].revents = 0; // 重置事件状态
        }

        int res;
        SYSCHECK_ERR_RETURN_NEG1(
            res = WSAPoll(fds.data(), fds.size(), checkTimeout_.count())) // 使用 Windows 下的 WSAPoll 检测事件
        if (res == 0) {
            auto rv = WaitForSingleObject(ghStopEvent_, 0); // 等待事件
            if (rv != WAIT_TIMEOUT) {
                finished = true; // 如果等待超时之外的其他情况，结束循环
                break;
            }
            continue;
        }

        // TCPStore 的监听套接字有事件，并且应该能够接受新连接
        if (fds[0].revents != 0) {
            if (!(fds[0].revents & POLLIN)) {
                C10_THROW_ERROR(
                    DistStoreError,
                    "Unexpected poll revent on the master's listening socket: " +
                        std::to_string(fds[0].revents)); // 抛出异常，不期望的事件
            }
            Socket socket = storeListenSocket_.accept(); // 接受新连接
            int rawSocket = socket.handle(); // 获取原始套接字
            sockets_.emplace_back(std::move(socket)); // 将套接字移入列表
            tcputil::addPollfd(fds, rawSocket, POLLIN); // 将新套接字添加到轮询事件中
            addMiscellaneousSocket(rawSocket); // 添加到杂项套接字集合
        }
        queryFds(fds); // 处理查询的套接字事件
    }
}
#else
void TCPStoreMasterDaemon::run() {
    try {
        c10::setThreadName("pt_tcpstore");
    // 创建一个存储 pollfd 结构体的向量 fds
    std::vector<struct pollfd> fds;
    // 向 fds 中添加 storeListenSocket_ 的文件描述符，并监听 POLLIN 事件
    tcputil::addPollfd(fds, storeListenSocket_.handle(), POLLIN);

    // 尽管我们没有找到任何描述这种情况的文档或文献，但我们注意到在特定情况下，
    // 管道的读端在写端关闭时可能不会接收到 POLLHUP 事件。然而，在相同的情况下，
    // 向管道写入数据会保证读端收到 POLLIN 事件。
    //
    // 为了更可靠地终止，主线程在关闭管道之前会向其写入一个字节，
    // 而后台线程将同时轮询 POLLIN 和 POLLHUP 事件。
    tcputil::addPollfd(fds, controlPipeFd_[0], POLLIN | POLLHUP);

    // 接收查询请求
    bool finished = false;
    while (!finished) {
      // 重置 fds 中每个 pollfd 结构体的 revents 字段
      for (const auto i : c10::irange(sockets_.size())) {
        fds[i].revents = 0;
      }

      // 调用 poll 函数等待事件发生
      SYSCHECK_ERR_RETURN_NEG1(::poll(fds.data(), fds.size(), -1));

      // TCPStore 的监听套接字有事件发生，可以接受新连接
      if (fds[0].revents != 0) {
        // 检查 fds[0] 的 revents 是否有异常的事件发生
        if (fds[0].revents ^ POLLIN) {
          C10_THROW_ERROR(
              DistStoreError,
              "Unexpected poll revent on the master's listening socket: " +
                  std::to_string(fds[0].revents));
        }
        // 接受新连接，并添加到 sockets_ 中
        Socket socket = storeListenSocket_.accept();
        int rawSocket = socket.handle();
        sockets_.emplace_back(std::move(socket));
        // 向 fds 中添加新连接的套接字，并监听 POLLIN 事件
        tcputil::addPollfd(fds, rawSocket, POLLIN);
        // 在获取验证查询之前，所有客户端都是杂项
        addMiscellaneousSocket(rawSocket);
      }

      // 管道接收到事件，告知我们关闭守护进程
      if (fds[1].revents != 0) {
        // 主线程将向管道写入一个字节，然后关闭它，然后加入后台线程
        if (fds[1].revents & ~(POLLIN | POLLHUP)) {
          C10_THROW_ERROR(
              DistStoreError,
              "Unexpected poll revent on the control pipe's reading fd: " +
                  std::to_string(fds[1].revents));
        }
        // 标志守护进程可以结束
        finished = true;
        break;
      }
      // 处理查询文件描述符
      queryFds(fds);
    }
  } catch (const std::exception& ex) {
    // 抛出异常，显示守护进程运行失败
    C10D_ERROR(
        "TCPStoreMasterDaemon::run() failed with exception: ", ex.what());
    throw;
  } catch (...) {
    // 抛出未知异常，显示守护进程运行失败
    C10D_ERROR("TCPStoreMasterDaemon::run() failed with unknown exception");
    throw;
  }
}

#endif

// 创建 TCPStore 后端的工厂函数，接收 TCPStoreOptions 类型的参数 opts
std::unique_ptr<BackgroundThread> create_tcpstore_backend(
    const TCPStoreOptions& opts) {
  // 如果 opts 中有 masterListenFd 的值
  Socket socket = opts.masterListenFd.has_value()
      // 使用已有的文件描述符创建 Socket 对象，并指定端口
      ? Socket::listenFromFd(*opts.masterListenFd, opts.port)
      // 否则新建一个 Socket 对象，并指定端口
      : Socket::listen(opts.port);

  // 返回一个指向 TCPStoreMasterDaemon 对象的独占指针，该对象包含移动后的 socket
  return std::make_unique<TCPStoreMasterDaemon>(std::move(socket));
}

} // namespace c10d::detail
```