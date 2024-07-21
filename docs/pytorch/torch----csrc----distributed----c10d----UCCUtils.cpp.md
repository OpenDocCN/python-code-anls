# `.\pytorch\torch\csrc\distributed\c10d\UCCUtils.cpp`

```py
#ifdef USE_C10D_UCC
// 如果定义了 USE_C10D_UCC 宏，则编译以下代码块

#include <torch/csrc/distributed/c10d/UCCTracing.hpp>
#include <torch/csrc/distributed/c10d/UCCUtils.hpp>

#include <cctype>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace c10d {

namespace {
// 常量定义：用于存储键的常量字符串
constexpr char kTeamRank[] = "teamr";
constexpr char kAllGatherDone[] = "ag_done";
constexpr char kAllGatherFree[] = "ag_free";
} // namespace

// Out-of-band (oob) allgather 函数定义
// sbuf: 发送缓冲区指针, rbuf: 接收缓冲区指针, msglen: 消息长度, coll_info: 收集信息指针, req: 请求指针
ucc_status_t oob_allgather(
    void* sbuf,
    void* rbuf,
    size_t msglen,
    void* coll_info,
    void** req) {
  auto* info = reinterpret_cast<torch_ucc_oob_coll_info_t*>(coll_info);
  // 确保 info 非空
  TORCH_CHECK(info != nullptr);
  // 将发送缓冲区数据转换为 uint8_t 向量
  std::vector<uint8_t> val = std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(sbuf),
      reinterpret_cast<uint8_t*>(sbuf) + msglen);
  try {
    // 将数据存储在 info 对象的存储中，键为 kTeamRank + 当前进程的排名
    info->store->set(info->getKey(kTeamRank + std::to_string(info->rank)), val);
    // 设置 info 对象的接收缓冲区和消息长度
    info->rbuf = rbuf;
    info->msglen = msglen;
    // 设置请求指针
    *req = coll_info;
  } catch (std::exception& ex) {
    // 捕获并记录异常信息
    LOG(ERROR) << "(oob_allgather) Caught exception in Store Operation .. "
               << "[" << ex.what() << "]";
    // 返回错误代码
    return UCC_ERR_NO_MESSAGE;
  }
  // 返回成功状态
  return UCC_OK;
}

// Out-of-band (oob) allgather 测试函数定义
// req: 请求指针
ucc_status_t oob_allgather_test(void* req) {
  auto* info = reinterpret_cast<torch_ucc_oob_coll_info_t*>(req);
  // 确保 info 非空
  TORCH_CHECK(info != nullptr);

  try {
    // 检查所有进程是否完成 allgather 操作
    for (int r = 0; r < info->size; r++) {
      if (!info->store->check({info->getKey(kTeamRank + std::to_string(r))})) {
        return UCC_INPROGRESS;
      }
    }
    // 所有进程完成后，从存储中获取数据并复制到接收缓冲区
    for (int r = 0; r < info->size; r++) {
      std::vector<uint8_t> data =
          info->store->get(info->getKey(kTeamRank + std::to_string(r)));
      memcpy(
          (void*)((ptrdiff_t)info->rbuf + info->msglen * r),
          data.data(),
          info->msglen);
    }
  } catch (std::exception& ex) {
    // 捕获并记录异常信息
    LOG(ERROR) << "(oob_allgather) Caught exception in Store Operation .. "
               << "[" << ex.what() << "]";
    // 返回错误代码
    return UCC_ERR_NO_MESSAGE;
  }
  // 返回成功状态
  return UCC_OK;
}

// Out-of-band (oob) allgather 释放函数定义
// req: 请求指针
ucc_status_t oob_allgather_free(void* req) {
  auto* info = reinterpret_cast<torch_ucc_oob_coll_info_t*>(req);
  // 确保 info 非空
  TORCH_CHECK(info != nullptr);
  try {
    // 增加完成的 allgather 进程数
    int num_done = info->store->add({info->getKey(kAllGatherDone)}, 1);
    // 如果所有进程都完成
    if (num_done == info->size) {
      // 删除完成标记
      info->store->deleteKey(info->getKey(kAllGatherDone));
      // 清除所有进程的数据键，避免竞争条件
      for (const auto r : c10::irange(info->size)) {
        info->store->deleteKey(info->getKey(kTeamRank + std::to_string(r)));
      }
      // 为每个进程信号释放信号
      for (const auto r : c10::irange(info->size)) {
        info->store->add({info->getKey(kAllGatherFree + std::to_string(r))}, 1);
      }
    } else {
      // 等待其他进程释放信号
      info->store->wait(
          {info->getKey(kAllGatherFree + std::to_string(info->rank))});
    }
    // 删除当前进程的释放信号键
    info->store->deleteKey(
        info->getKey(kAllGatherFree + std::to_string(info->rank)));
  } catch (std::exception& ex) {
    // 捕获并记录异常信息
    LOG(ERROR) << "(oob_allgather) Caught exception in Store Operation .. "
               << "[" << ex.what() << "]";
    // 返回错误代码
    return UCC_ERR_NO_MESSAGE;
  }
  // 返回成功状态
  return UCC_OK;
}
    # 记录错误日志并返回错误码
    LOG(ERROR) << "(oob_allgather) Caught exception in Store Operation .. "
               << "[" << ex.what() << "]";
    # 如果捕获到异常，返回无消息错误码
    return UCC_ERR_NO_MESSAGE;
    # 如果没有异常，返回成功码
    }
    return UCC_OK;
// CommUCC 类的构造函数，初始化一个 UCC 通信库的通信对象
CommUCC::CommUCC(
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob,  // 使用共享指针初始化 Out-of-Band 通信信息
    const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger)  // 使用智能指针初始化日志记录器
    : CommBase(logger) {  // 调用基类 CommBase 的构造函数，传入日志记录器

  // 初始化 UCC 库的配置和参数
  ucc_lib_config_h lib_config;
  ucc_context_config_h context_config;
  ucc_lib_params_t lib_params;
  ucc_context_params_t context_params;
  ucc_status_t st;

  // 读取名为 "TORCH" 的 UCC 库配置，存入 lib_config
  TORCH_UCC_CHECK(
      ucc_lib_config_read("TORCH", nullptr, &lib_config),
      "failed to read UCC lib config");

  // 清空 lib_params 结构体，并设置多线程模式
  memset(&lib_params, 0, sizeof(ucc_lib_params_t));
  lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode = UCC_THREAD_MULTIPLE;

  // 初始化 UCC 库，传入 lib_params 和 lib_config，存入 lib
  TORCH_UCC_CHECK(
      ucc_init(&lib_params, lib_config, &lib), "failed to init UCC lib");

  // 释放 lib_config 的资源
  ucc_lib_config_release(lib_config);

  // 查询 lib 的属性，检查多线程模式是否正确初始化
  ucc_lib_attr_t lib_attr;
  lib_attr.mask = UCC_LIB_ATTR_FIELD_THREAD_MODE;
  TORCH_UCC_CHECK(
      ucc_lib_get_attr(lib, &lib_attr), "failed to query for lib attr");

  // 检查 lib_attr 中的线程模式是否为 UCC_THREAD_MULTIPLE
  TORCH_CHECK(
      lib_attr.thread_mode == UCC_THREAD_MULTIPLE,
      "ucc library wasn't initialized with multithreading support, "
      "please check ucc build options");

  // 尝试读取 UCC 上下文配置，如果失败可能会导致死锁
  st = ucc_context_config_read(lib, NULL, &context_config);
  if (st != UCC_OK) {
    // 如果失败，尝试终止 UCC 库并记录错误信息
    TORCH_UCC_CHECK(
        ucc_finalize(lib),
        "failed to finalize UCC library when failing to read UCC context config");
    TORCH_UCC_LOG_ERROR(
        TORCH_UCC_INIT,
        c10::str("failed to read UCC context config: ", ucc_status_string(st)));
    throw std::runtime_error(ucc_status_string(st));
  }

  // 修改上下文配置，设置预估的端点数量为 oob->size
  st = ucc_context_config_modify(
      context_config,
      NULL,
      "ESTIMATED_NUM_EPS",
      std::to_string(oob->size).c_str());
  if (st != UCC_OK) {
    // 如果修改失败，释放 context_config 资源并终止 UCC 库，记录错误信息
    ucc_context_config_release(context_config);
    ucc_finalize(lib);
    TORCH_UCC_LOG_ERROR(
        TORCH_UCC_INIT,
        c10::str(
            "UCC failed to modify UCC context config: ",
            ucc_status_string(st)));
    throw std::runtime_error(ucc_status_string(st));
  }

  // 清空 context_params 结构体，并设置上下文参数
  memset(&context_params, 0, sizeof(ucc_context_params_t));
  context_params.mask =
      UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB;
  context_params.type = UCC_CONTEXT_SHARED;
  context_params.oob.n_oob_eps = oob->size;  // 设置 OOB 端点数量为 oob->size
  context_params.oob.oob_ep = oob->rank;     // 设置当前 OOB 端点的 rank
  context_params.oob.allgather = oob_allgather;  // 设置 allgather 函数指针
  context_params.oob.req_test = oob_allgather_test;  // 设置 req_test 函数指针
  context_params.oob.req_free = oob_allgather_free;  // 设置 req_free 函数指针
  context_params.oob.coll_info = oob.get();  // 设置 OOB 通信的 coll_info 参数

  // 创建 UCC 上下文，传入 lib、context_params 和 context_config，存入 context
  st = ucc_context_create(lib, &context_params, context_config, &context);

  // 释放 context_config 资源
  ucc_context_config_release(context_config);

  // 如果创建上下文失败，尝试终止 UCC 库并记录错误信息
  if (st != UCC_OK) {
    TORCH_UCC_CHECK(
        ucc_finalize(lib),
        "failed to finalize UCC library when failing to creat UCC context");
    TORCH_UCC_LOG_ERROR(
        TORCH_UCC_INIT,
        c10::str("UCC failed to create UCC context: ", ucc_status_string(st)));
    throw std::runtime_error(ucc_status_string(st));
  }
}
void CommUCC::progress() {
  // 调用 UCC 上下文的进展函数来推进 UCC 集合操作
  TORCH_UCC_CHECK(
      ucc_context_progress(context), "failed to progress UCC collective");
}

void CommUCC::free_request(ucc_coll_req_h request) {
  // 调用 UCC 完成函数来释放 UCC 请求对象
  TORCH_UCC_CHECK(
      ucc_collective_finalize(request), "failed to release UCC request");
}

CommUCC::~CommUCC() {
  if (context != nullptr) {
    // 如果上下文对象不为 nullptr，则销毁 UCC 上下文
    TORCH_UCC_CHECK(
        ucc_context_destroy(context), "failed to destroy UCC context");
  }
  if (lib != nullptr) {
    // 如果库对象不为 nullptr，则结束 UCC 库的使用
    TORCH_UCC_CHECK(ucc_finalize(lib), "failed to finalize UCC library");
  }
  context = nullptr;
  lib = nullptr;
}

std::string ProcessGroupUCCLogger::getLogPrefix(torch_ucc_phase_t phase) {
  // 获取日志前缀，根据传入的阶段来确定是否使用本地存储的阶段值
  torch_ucc_phase_t phase_ =
      (local_phase != phase && phase != TORCH_UCC_UNKNOWN) ? phase
                                                           : local_phase;
  return c10::str(log_prefix, "[", ucc_phase_map.at(phase_), "]");
}

void ProcessGroupUCCLogger::setLogPrefix(std::string log_prefix_) {
  // 设置日志前缀为给定的字符串
  log_prefix = log_prefix_;
}

ProcessGroupUCCLogger::ProcessGroupUCCLogger() {
  // 在构造函数中使用默认的日志前缀初始化日志器
  setLogPrefix("[ProcessGroupUCC]");
}

ProcessGroupUCCLogger::ProcessGroupUCCLogger(
    std::string log_prefix,
    torch_ucc_phase_t phase)
    : local_phase(phase) {
  // 在构造函数中使用指定的日志前缀和阶段初始化日志器
  setLogPrefix(log_prefix);
}

} // namespace c10d

#endif // USE_C10D_UCC
```