# `.\pytorch\torch\csrc\distributed\c10d\control_plane\WorkerServer.cpp`

```
// 包含标准库头文件
#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <tuple>
#include <unordered_map>

// 包含第三方库头文件
#include <ATen/core/interned_strings.h>
#include <c10/util/thread_name.h>
#include <caffe2/utils/threadpool/WorkersPool.h>
#include <torch/csrc/distributed/c10d/control_plane/WorkerServer.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

// 使用 c10d 命名空间下的 control_plane 命名空间
namespace c10d {
namespace control_plane {

// 匿名命名空间，用于定义内部辅助函数和类
namespace {

// RequestImpl 类，实现了 Request 接口
class RequestImpl : public Request {
 public:
  // 构造函数，接收 httplib::Request 对象的引用作为参数
  RequestImpl(const httplib::Request& req) : req_(req) {}

  // 实现接口方法，返回请求的 body 内容
  const std::string& body() override {
    return req_.body;
  }

  // 实现接口方法，返回请求的参数的多重映射
  const std::multimap<std::string, std::string>& params() const override {
    return req_.params;
  }

 private:
  const httplib::Request& req_;  // 保存传入的 httplib::Request 引用
};

// ResponseImpl 类，实现了 Response 接口
class ResponseImpl : public Response {
 public:
  // 构造函数，接收 httplib::Response 对象的引用作为参数
  ResponseImpl(httplib::Response& res) : res_(res) {}

  // 实现接口方法，设置响应的状态码
  void setStatus(int status) override {
    res_.status = status;
  }

  // 实现接口方法，设置响应的内容和内容类型
  void setContent(std::string&& content, const std::string& content_type)
      override {
    res_.set_content(std::move(content), content_type);
  }

 private:
  httplib::Response& res_;  // 保存传入的 httplib::Response 引用
};

// 辅助函数，用于转义 JSON 字符串中的特殊字符
std::string jsonStrEscape(const std::string& str) {
  std::ostringstream ostream;
  for (char ch : str) {
    if (ch == '"') {
      ostream << "\\\"";
    } else if (ch == '\\') {
      ostream << "\\\\";
    } else if (ch == '\b') {
      ostream << "\\b";
    } else if (ch == '\f') {
      ostream << "\\f";
    } else if (ch == '\n') {
      ostream << "\\n";
    } else if (ch == '\r') {
      ostream << "\\r";
    } else if (ch == '\t') {
      ostream << "\\t";
    } else if ('\x00' <= ch && ch <= '\x1f') {
      ostream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch);
    } else {
      ostream << ch;
    }
  }
  return ostream.str();  // 返回转义后的 JSON 字符串
}
} // namespace

// WorkerServer 类的构造函数，接收主机名或文件路径和端口号作为参数
WorkerServer::WorkerServer(const std::string& hostOrFile, int port) {
  // 使用 httplib 库的 Get 方法处理 HTTP GET 请求
  server_.Get("/", [](const httplib::Request& req, httplib::Response& res) {
    // 设置响应内容为 HTML 格式的字符串
    res.set_content(
        R"BODY(<h1>torch.distributed.WorkerServer</h1>
<a href="/handler/">Handler names</a>
)BODY",
        "text/html");
  });
  // 处理 GET 请求，获取所有处理程序的名称列表
  server_.Get(
      "/handler/", [](const httplib::Request& req, httplib::Response& res) {
        // 构建响应体
        std::ostringstream body;
        body << "[";
        bool first = true;
        // 遍历处理程序名称列表
        for (const auto& name : getHandlerNames()) {
          if (!first) {
            body << ",";
          }
          first = false;

          // 将处理程序名称转义并添加到响应体中
          body << "\"" << jsonStrEscape(name) << "\"";
        }
        body << "]";

        // 设置响应内容为 JSON 格式
        res.set_content(body.str(), "application/json");
      });
  // 处理 POST 请求，根据处理程序名称执行相应的处理函数
  server_.Post(
      "/handler/:handler",
      [](const httplib::Request& req, httplib::Response& res) {
        // 获取请求中的处理程序名称参数
        auto handler_name = req.path_params.at("handler");
        HandlerFunc handler;
        try {
          // 根据处理程序名称获取处理函数
          handler = getHandler(handler_name);
        } catch (const std::exception& e) {
          // 处理函数不存在的异常处理
          res.status = 404;
          res.set_content(
              fmt::format("Handler {} not found: {}", handler_name, e.what()),
              "text/plain");
          return;
        }
        // 创建请求和响应的封装对象
        RequestImpl torchReq{req};
        ResponseImpl torchRes{res};

        try {
          // 调用处理函数处理请求
          handler(torchReq, torchRes);
        } catch (const std::exception& e) {
          // 处理函数执行失败的异常处理
          res.status = 500;
          res.set_content(
              fmt::format("Handler {} failed: {}", handler_name, e.what()),
              "text/plain");
          return;
        } catch (...) {
          // 处理函数执行出现未知异常的异常处理
          res.status = 500;
          res.set_content(
              fmt::format(
                  "Handler {} failed with unknown exception", handler_name),
              "text/plain");
          return;
        }
      });

  // 调整保持连接的超时时间，防止服务器过快关闭连接
  server_.set_keep_alive_timeout(1); // second, default is 5
  // 设置最大保持连接次数，超过这个次数后关闭连接
  server_.set_keep_alive_max_count(
      30); // wait max 30 seconds before closing socket

  if (port == -1) {
    // 使用 Unix 套接字进行通信
    server_.set_address_family(AF_UNIX);

    if (std::filesystem::exists(hostOrFile)) {
      // 如果 Unix 套接字文件已存在，则抛出异常
      throw std::runtime_error(fmt::format("{} already exists", hostOrFile));
    }

    // 绑定到指定的 Unix 套接字地址
    C10D_WARNING("Server listening to UNIX {}", hostOrFile);
    if (!server_.bind_to_port(hostOrFile, 80)) {
      // 绑定 Unix 套接字地址失败的异常处理
      throw std::runtime_error(fmt::format("Error binding to {}", hostOrFile));
    }
  } else {
    // 使用 TCP 协议进行通信
    C10D_WARNING("Server listening to TCP {}:{}", hostOrFile, port);
    if (!server_.bind_to_port(hostOrFile, port)) {
      // 绑定 TCP 地址和端口失败的异常处理
      throw std::runtime_error(
          fmt::format("Error binding to {}:{}", hostOrFile, port));
    }
  }

  // 启动服务器线程
  serverThread_ = std::thread([this]() {
    c10::setThreadName("pt_workerserver");

    try {
      // 开始监听客户端连接
      if (!server_.listen_after_bind()) {
        throw std::runtime_error("failed to listen");
      }
    } catch (std::exception& e) {
      // 监听过程中发生异常的异常处理
      C10D_ERROR("Error while running server: {}", e.what());
      throw;
    }
    // 输出服务器退出信息
    C10D_WARNING("Server exited");
  });
}

// 关闭服务器的方法
void WorkerServer::shutdown() {
  // 输出服务器正在关闭的信息
  C10D_WARNING("Server shutting down");
  // 停止服务器的运行
  server_.stop();
  // 等待服务器线程退出
  serverThread_.join();
}
// WorkerServer 类的析构函数定义
WorkerServer::~WorkerServer() {
    // 如果 serverThread_ 线程是可加入的（joinable），表示线程仍在运行
    if (serverThread_.joinable()) {
        // 输出警告消息，提示 WorkerServer 的析构函数被调用但未执行 shutdown()
        C10D_WARNING("WorkerServer destructor called without shutdown");
        // 执行 shutdown() 方法，关闭 WorkerServer
        shutdown();
    }
}

// 结束 control_plane 命名空间
} // namespace control_plane

// 结束 c10d 命名空间
} // namespace c10d
```