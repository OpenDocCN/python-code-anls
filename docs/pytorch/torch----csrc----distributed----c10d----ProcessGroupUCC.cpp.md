# `.\pytorch\torch\csrc\distributed\c10d\ProcessGroupUCC.cpp`

```
#ifdef USE_C10D_UCC
// 包含必要的头文件，实现对应的功能模块
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#include <torch/csrc/distributed/c10d/UCCTracing.hpp>
#include <torch/csrc/distributed/c10d/UCCUtils.hpp>
#include <list> // 包含 C++ 标准库中的 list 头文件
#include <memory> // 包含 C++ 标准库中的 memory 头文件
#include <unordered_map> // 包含 C++ 标准库中的 unordered_map 头文件
#include <unordered_set> // 包含 C++ 标准库中的 unordered_set 头文件

namespace c10d {

namespace {
// 创建映射表，将 C10 设备类型映射到 UCC 内存类型
const std::map<c10::DeviceType, ucc_memory_type_t> ucc_mtype_map = {
    {c10::kCPU, UCC_MEMORY_TYPE_HOST},
    {c10::kCUDA, UCC_MEMORY_TYPE_CUDA},
};

// 将 C10 设备类型转换为 UCC 内存类型
ucc_memory_type_t to_ucc_memType(c10::DeviceType _c10_type) {
  if (ucc_mtype_map.find(_c10_type) != ucc_mtype_map.end())
    return ucc_mtype_map.at(_c10_type);
  else
    return UCC_MEMORY_TYPE_UNKNOWN;
}

// 创建映射表，将 Torch 的标量类型映射到 UCC 数据类型
const std::map<at::ScalarType, ucc_datatype_t> ucc_dtype_map = {
    {at::kByte, UCC_DT_UINT8},
    {at::kChar, UCC_DT_INT8},
    {at::kHalf, UCC_DT_FLOAT16},
    {at::kBFloat16, UCC_DT_BFLOAT16},
    {at::kDouble, UCC_DT_FLOAT64},
    {at::kFloat, UCC_DT_FLOAT32},
    {at::kInt, UCC_DT_INT32},
    {at::kLong, UCC_DT_INT64},
    {at::kBool, UCC_DT_UINT8},
};

// 将 Torch 的 Tensor 类型转换为 UCC 数据类型
ucc_datatype_t to_ucc_dType(at::Tensor _tensor) {
  if (_tensor.scalar_type() == at::kBool && _tensor.element_size() != 1) {
    TORCH_CHECK(
        false, "Size of Boolean type larger than 1 is not supported in UCC");
  }
  try {
    return ucc_dtype_map.at(_tensor.scalar_type());
  } catch (const std::out_of_range&) {
    TORCH_CHECK(false, "Not supported data type for UCC");
  }
}

// 创建映射表，将 ReduceOp 枚举类型映射到 UCC 归约操作类型
const std::map<ReduceOp, ucc_reduction_op_t> ucc_op_map = {
    {ReduceOp::SUM, UCC_OP_SUM},
    {ReduceOp::PRODUCT, UCC_OP_PROD},
    {ReduceOp::MIN, UCC_OP_MIN},
    {ReduceOp::MAX, UCC_OP_MAX},
    {ReduceOp::BAND, UCC_OP_BAND},
    {ReduceOp::BOR, UCC_OP_BOR},
    {ReduceOp::BXOR, UCC_OP_BXOR},
    {ReduceOp::AVG, UCC_OP_AVG},
};

// 将 ReduceOp 枚举类型转换为 UCC 归约操作类型
ucc_reduction_op_t to_ucc_reduceOp(
    const ReduceOp _op,
    const at::ScalarType _dt) {
  if (_dt == at::kBool) {
    if (_op == ReduceOp::SUM) {
      // 对于布尔类型，使用位或操作
      return UCC_OP_MAX;
    } else if (_op == ReduceOp::PRODUCT) {
      // 对于布尔类型，使用位与操作
      return UCC_OP_MIN;
    } else if (_op == ReduceOp::AVG) {
      TORCH_CHECK(false, "Cannot use ReduceOp.AVG with boolean inputs");
    }
  }

  try {
    return ucc_op_map.at(_op);
  } catch (const std::out_of_range&) {
    TORCH_CHECK(false, "Not supported ReduceOp for UCC");
  }
}

// 定义包含 UCC 配置选项的结构体
struct torch_ucc_config_t {
  c10::once_flag flag; // C10D 使用的线程安全标志
  std::array<bool, 32> blocking_wait; // 32 个布尔值数组，用于指示是否阻塞等待
  bool enable_comms_logger; // 是否启用通信记录器
  bool use_future; // 是否使用 Future
  bool shared_comm; // 是否共享通信器以节省资源
  bool use_allgatherv; // 是否使用 allgatherv 实现 allgather 操作
  bool enable_health_check; // 是否启用健康检查
} torch_ucc_config;

// 创建环境变量映射表，用于配置 UCC 环境变量
std::unordered_map<std::string, std::string> torch_ucc_envs_map = {
    // TORCH_UCC_BLOCKING_WAIT 语法允许：
    // - TORCH_UCC_BLOCKING_WAIT=none --> 完全禁用阻塞等待
    // 设置环境变量 TORCH_UCC_BLOCKING_WAIT，控制是否启用完全阻塞等待
    // 可选值：
    // - "all"：完全启用阻塞等待
    // - "allreduce,send,recv"：仅在选择的操作上启用阻塞等待
    // 支持的操作有：
    // [allgather, allgather_base, allreduce, alltoall, broadcast,
    //  gather, reduce, reduce_scatter, scatter, send, recv]
    {"TORCH_UCC_BLOCKING_WAIT", "none"},

    // 设置环境变量 TORCH_UCC_USE_FUTURE，启用异步通信支持
    {"TORCH_UCC_USE_FUTURE", "1"},

    // 设置环境变量 TORCH_UCC_PROFILING_ENABLE，禁用通信性能分析
    {"TORCH_UCC_PROFILING_ENABLE", "0"},

    // 设置环境变量 TORCH_UCC_SHARED_COMM，启用共享通信上下文
    {"TORCH_UCC_SHARED_COMM", "1"},

    // 设置环境变量 TORCH_UCC_USE_ALLGATHERV，禁用使用 allgatherv 操作
    {"TORCH_UCC_USE_ALLGATHERV", "0"},

    // 设置环境变量 TORCH_UCC_ENABLE_HEALTH_CHECK，禁用健康检查
    {"TORCH_UCC_ENABLE_HEALTH_CHECK", "0"},

    // 设置环境变量 TORCH_UCC_ENABLE_COMMS_LOGGER，禁用通信日志记录
    {"TORCH_UCC_ENABLE_COMMS_LOGGER", "0"},
};

// 解析阻塞等待操作列表的函数，返回操作类型的向量
std::vector<OpType> parse_blocking_wait(std::string op_list_string) {
  // 静态映射，将字符串映射到对应的操作类型枚举
  const static std::unordered_map<std::string, OpType> str2op = {
      {"allgather", OpType::ALLGATHER},
      {"allgather_base", OpType::_ALLGATHER_BASE},
      {"allreduce", OpType::ALLREDUCE},
      {"alltoall_base", OpType::ALLTOALL_BASE},
      {"broadcast", OpType::BROADCAST},
      {"gather", OpType::GATHER},
      {"reduce", OpType::REDUCE},
      {"reduce_scatter", OpType::REDUCE_SCATTER},
      {"scatter", OpType::SCATTER},
      {"send", OpType::SEND},
      {"recv", OpType::RECV},
  };
  
  // 解析操作列表字符串
  auto op_list = parse_list(op_list_string);
  
  // 如果操作列表只有一个元素且为"none"，返回空向量
  if (op_list == std::vector<std::string>{"none"}) {
    return {};
  }
  
  // 存储解析结果的向量
  std::vector<OpType> result;
  
  // 如果操作列表包含"all"，将所有操作类型加入结果向量
  if (op_list == std::vector<std::string>{"all"}) {
    for (auto entry : str2op) {
      result.push_back(entry.second);
    }
  } else {
    // 否则，根据操作字符串查找对应的操作类型并加入结果向量
    for (auto op_string : op_list) {
      result.push_back(str2op.at(op_string));
    }
  }
  
  return result;
}

// 解析配置文件并更新全局配置对象
void read_config() {
  // 默认配置
  torch_ucc_config.blocking_wait.fill(false);
  torch_ucc_config.use_future = true;
  torch_ucc_config.shared_comm = false;
  torch_ucc_config.use_allgatherv = false;
  torch_ucc_config.enable_health_check = false;
  torch_ucc_config.enable_comms_logger = false;
  
  // 读取所有的 torch_ucc 环境变量并更新映射表
  char* env;
  for (auto& torch_ucc_env : torch_ucc_envs_map) {
    env = std::getenv(torch_ucc_env.first.c_str());
    if (env) {
      torch_ucc_envs_map[torch_ucc_env.first] = std::string(env);
    }
  }
  
  // 从环境变量中获取 TORCH_UCC_BLOCKING_WAIT 对应的字符串，并解析为操作类型并设置为阻塞等待
  auto blocking_wait_str = torch_ucc_envs_map.at("TORCH_UCC_BLOCKING_WAIT");
  for (auto op : parse_blocking_wait(blocking_wait_str)) {
    torch_ucc_config.blocking_wait[(std::uint8_t)op] = true;
  }
  
  // BARRIER 操作始终设置为阻塞等待
  torch_ucc_config.blocking_wait[(std::uint8_t)OpType::BARRIER] = true;
  
  // 从环境变量中获取其他配置项并更新全局配置对象
  torch_ucc_config.use_future =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_USE_FUTURE"));
  torch_ucc_config.shared_comm =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_SHARED_COMM"));
  torch_ucc_config.use_allgatherv =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_USE_ALLGATHERV"));
  torch_ucc_config.enable_health_check =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_ENABLE_HEALTH_CHECK"));
  torch_ucc_config.enable_comms_logger =
      std::stoi(torch_ucc_envs_map.at("TORCH_UCC_ENABLE_COMMS_LOGGER"));
}

// 检查设备是否支持多设备操作
void check_device(c10::Device dev1, c10::Device dev2) {
  if (dev1.is_cuda() && dev2.is_cuda() && dev1 != dev2) {
    throw std::invalid_argument("ProcessGroupUCC multidevice is not supported");
  }
}

// 检查张量是否符合要求
void check_tensor(const std::vector<at::Tensor>& tensors) {
  // 张量数量不为1时抛出异常
  if (tensors.size() != 1) {
    throw std::invalid_argument(
        "ProcessGroupUCC takes 1 tensor. Got " +
        std::to_string(tensors.size()) + ". ");
  }
  // 张量不是连续时抛出异常
  if (!tensors[0].is_contiguous()) {
    throw std::invalid_argument(
        "ProcessGroupUCC input tensor has to be contiguous");
  }
  // 张量为稀疏张量时抛出异常
  if (tensors[0].is_sparse()) {

    throw std::invalid_argument(
        "ProcessGroupUCC input tensor has to be contiguous");
  }
}
    throw std::invalid_argument("ProcessGroupUCC input tensor has to be dense");
  // 抛出异常，指示输入张量必须是密集张量
  }
  // TODO: check cuda case
  // 待办事项：检查 CUDA 情况
}

ProcessGroupUCC::WorkUCC::~WorkUCC() {
#ifdef USE_CUDA
  // 如果 fence 和 ep 都存在，则将 fence 移动到 ep 的事件池中
  if (fence && ep) {
    std::lock_guard<std::mutex> lock(ep->event_pool_mutex);
    ep->event_pool.push(std::move(fence));
  }
#endif
}

void ProcessGroupUCC::WorkUCC::setException() {
  // 如果已经有异常或者 entry_ 为空，则直接返回
  if (exception() || !entry_) {
    return;
  }
  // 将 entry_ 的异常指针设置为当前对象的异常指针
  exception_ = entry_->eptr_;
}

void ProcessGroupUCC::WorkUCC::setAndThrowException() {
  // 设置异常并抛出异常
  setException();
  if (exception()) {
    std::rethrow_exception(exception());
  }
}

bool ProcessGroupUCC::WorkUCC::isCompleted() {
  // 如果 entry_ 为空，则任务已完成
  if (!entry_) {
    return true;
  }
  // 设置异常，并检查任务是否完成，或者任务的状态是否小于等于 0
  // 这里 <= 0 是为了避免列出所有可能的状态码，主线程需要在 UCC 返回成功（== 0）
  // 或者任何错误码（< 0）时解除阻塞
  return exception() || entry_->status_ <= 0;
}

bool ProcessGroupUCC::WorkUCC::isSuccess() const {
  // 如果 entry_ 为空，则任务成功
  if (!entry_) {
    return true;
  }
  // 检查是否没有异常并且任务状态为 0
  return !exception() && entry_->status_ == 0;
}

bool ProcessGroupUCC::WorkUCC::wait(std::chrono::milliseconds /* unused */) {
  // 如果启用了通信日志记录且存在 logger，则记录通信事件
  if (torch_ucc_config.enable_comms_logger && logger_) {
    logger_->trace_generator->recordComms("wait", (uintptr_t)this, rank_);
  }
#ifdef USE_CUDA
  // 如果 fence 存在并且不是阻塞等待类型，则阻塞用户流并抛出异常
  if (fence && !torch_ucc_config.blocking_wait[(int)opType_]) {
    // 阻塞当前 CUDA 流
    setAndThrowException();
    fence->block(at::cuda::getCurrentCUDAStream());
    return true;
  }
#endif
  // 等待任务完成。对于阻塞情况，主线程将在此循环中阻塞，直到进度线程改变该请求的状态。
  // 如果发生超时，UCC 将返回 UCC_ERR_TIMEOUT 作为状态。主线程将在此时抛出异常。
  // 当前 UCC 没有 "abort" 函数。
  while (!isCompleted())
    ;
  setAndThrowException();
  // 如果已设置了结束回调函数，则手动调用，因为进度线程不拥有 WorkUCC
  if (Work::recordFunctionEndCallback_) {
    Work::recordFunctionEndCallback_();
    Work::recordFunctionEndCallback_ = nullptr;
  }
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupUCC::WorkUCC::getFuture() {
  // 返回 future_
  return future_;
}

int ProcessGroupUCC::WorkUCC::sourceRank() const {
  // 如果操作类型不是接收（OpType::RECV）或者接收任意来源（OpType::RECVANYSOURCE），则抛出错误
  if (opType_ != OpType::RECV && opType_ != OpType::RECVANYSOURCE) {
    // 抛出错误
    return Work::sourceRank();
  }
  // 返回来源的排名
  return sourceRank_;
}

std::vector<at::Tensor> ProcessGroupUCC::WorkUCC::result() {
  // 返回输出的引用
  return *outputs_;
}

void ProcessGroupUCC::ProgressEntry::finalize(std::exception_ptr eptr) {
  ucc_status_t status = UCC_OK;

  // 如果请求不为空，则释放请求并获取其状态
  if (request_ != nullptr) {
    status = request_->status;
    comm_->free_request(request_);
  }
  // 如果存在异常指针，则将其赋给 eptr_；否则将状态赋给 status_
  if (eptr) {
    eptr_ = eptr;
  } else {
    status_ = status;
  }
  // 如果存在 future，则根据是否有异常设置其状态
  if (future_) {
    if (eptr) {
      future_->setError(eptr);
    } else {
      future_->markCompleted(
          c10::IValue(data ? data->dst : std::vector<at::Tensor>()));
    }
  }
}

Comm::Comm(
    const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger_,
    # 构造函数，初始化对象时执行的操作
    # 初始化日志记录器
    # 使用 shared_ptr 初始化 oob_ 成员变量
    # 使用给定的 logger_ 参数初始化 logger 成员变量
    # 使用 oob_ 参数初始化 oob 成员变量
    # 使用 oob 和 logger 初始化 ucc_comm 成员变量
    # 根据 is_health_check 参数选择初始化 finalize_phase 成员变量为 TORCH_UCC_HEALTH_CHECK 或 TORCH_UCC_FINALIZE
    # 如果 dev 是 CUDA 设备，则获取其索引并初始化 cuda_device_index 成员变量
    # 否则初始化 cuda_device_index 成员变量为 TORCH_UCC_DEVICE_NOT_SET
    # 初始化 stop_progress_loop 成员变量为 false，表示进度循环未停止
    # 初始化 collective_inprogress 成员变量为 false，表示没有正在进行的集体通信操作
    # 创建一个新的线程，用于执行 Comm 类的 progress_loop 方法，将当前对象（this 指针）作为参数传递
    progress_thread = std::thread(&Comm::progress_loop, this);
#ifdef _GNU_SOURCE
  // 设置进度线程的名称为"ucc-progress"
  pthread_setname_np(progress_thread.native_handle(), "ucc-progress");
#endif
}

Comm::~Comm() {
  // 使用互斥锁保护临界区域，等待进度队列为空且集合操作未在进行中
  std::unique_lock<std::mutex> lock(mutex);
  queue_consume_cv.wait(
      lock, [&] { return progress_queue.empty() && !collective_inprogress; });
  // 设置停止进度循环的标志
  stop_progress_loop = true;
  lock.unlock();
  // 通知所有等待的生产条件变量
  queue_produce_cv.notify_all();
  // 等待进度线程结束
  progress_thread.join();
}

std::shared_ptr<Comm> Comm::get_comm(
    uint32_t& id,
    c10::Device dev,
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob,
    const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger,
    bool is_health_check) {
  // 静态互斥锁和共享指针，用于保护共享资源
  static std::mutex m;
  static std::weak_ptr<Comm> comm;
  static uint32_t comm_id;

  std::lock_guard<std::mutex> lock(m);
  // 设置传入的 id 为当前 comm_id
  id = comm_id;

  // 创建组标识字符串
  std::string group_id = "group_id";
  if (is_health_check) {
    // 如果是健康检查，则在组标识字符串中加入设备类型信息
    group_id = c10::str(dev.type()) + "/" + group_id;
  }

  // 用于存储远程通信 id 的字节数组
  std::vector<uint8_t> remote_comm_id;
  // 删除存储中的键值对
  oob->store->deleteKey(group_id + std::to_string(0));
  if (oob->rank != 0) {
    // 如果不是 rank 为 0 的进程，将当前 id 存储到存储中
    std::vector<uint8_t> val = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(&id),
        reinterpret_cast<uint8_t*>(&id) + sizeof(id));
    oob->store->set(group_id + std::to_string(oob->rank), val);
  } else {
    // 如果是 rank 为 0 的进程，找到所有其他进程的最高 id
    for (int i = 1; i < oob->size; i++) {
      remote_comm_id = oob->store->get(group_id + std::to_string(i));
      oob->store->deleteKey(group_id + std::to_string(i));
      // 找到最高的 id
      id = std::max(id, *(reinterpret_cast<uint32_t*>(remote_comm_id.data())));
    }
    // 将当前 id 存储到存储中
    std::vector<uint8_t> val = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(&id),
        reinterpret_cast<uint8_t*>(&id) + sizeof(id));
    oob->store->set(group_id + std::to_string(oob->rank), val);
  }
  // 获取 rank 为 0 的远程通信 id
  remote_comm_id = oob->store->get(group_id + std::to_string(0));
  oob->comm_id = *(reinterpret_cast<uint32_t*>(remote_comm_id.data()));
  // 准备下一个 comm_id
  comm_id = oob->comm_id + 1;

  if (torch_ucc_config.shared_comm) {
    // 如果使用共享通信对象
    std::shared_ptr<Comm> shared_comm = comm.lock();
    if (!shared_comm) {
      // 如果共享通信对象为空，创建新的共享通信对象
      shared_comm = std::make_shared<Comm>(logger, oob, dev, is_health_check);
      comm = shared_comm;
    } else {
      // 如果共享通信对象不为空，检查是否需要更新 CUDA 设备索引
      if (dev.is_cuda() && !is_health_check) {
        if ((shared_comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) &&
            (shared_comm->cuda_device_index != dev.index())) {
          // 如果初始化时 CUDA 设备索引不一致，抛出异常
          TORCH_UCC_LOG_ERROR(
              is_health_check ? TORCH_UCC_HEALTH_CHECK : TORCH_UCC_INIT,
              "ucc communicator was initialized with different cuda device,"
              "multi device is not supported");
          throw std::invalid_argument(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
        }
        // 更新 CUDA 设备索引
        shared_comm->cuda_device_index = dev.index();
      }
    }
    // 返回共享通信对象
    return shared_comm;
  } else {
    // 如果不使用共享通信对象，创建新的通信对象
    return std::make_shared<Comm>(logger, oob, dev, is_health_check);
  }
}

void Comm::ucc_create_team(
    ucc_team_h& team,
    // 使用给定的 OOB (Out-of-Band) 参数创建一个 UCC 团队
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob) {
      // 定义 UCC 团队参数结构体
      ucc_status_t st;
      ucc_team_params_t team_params;
      // 设置团队参数的掩码，包括端点、端点范围和OOB
      team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE |
          UCC_TEAM_PARAM_FIELD_OOB;
      // 设置OOB参数中的全局收集函数、测试请求函数和释放请求函数
      team_params.oob.allgather = oob_allgather;
      team_params.oob.req_test = oob_allgather_test;
      team_params.oob.req_free = oob_allgather_free;
      // 将 OOBCollInfo 结构体指针赋值给团队参数中的 oob.coll_info
      team_params.oob.coll_info = oob.get();
      // 设置OOB参数中的端点数量和当前进程的OOB端点索引
      team_params.oob.n_oob_eps = oob->size;
      team_params.oob.oob_ep = oob->rank;
      // 设置团队的本地端点和端点范围为连续的
      team_params.ep = oob->rank;
      team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
      // 发起 UCC 团队创建操作，并将结果存储在 team 中
      TORCH_UCC_CHECK(
          ucc_team_create_post(&ucc_comm.context, 1, &team_params, &team),
          "failed to post team create");
      // 循环检查团队创建的状态，直到团队创建完成
      do {
        st = ucc_team_create_test(team);
        // 推进 UCC 上下文的进度，处理异步操作
        ucc_context_progress(ucc_comm.context);
      } while (st == UCC_INPROGRESS);
      // 检查团队创建的最终状态
      TORCH_UCC_CHECK(st, "failed to create UCC team");
void Comm::progress_loop() {
  // 获取互斥锁，确保线程安全访问共享资源
  std::unique_lock<std::mutex> lock(mutex);
#ifdef USE_CUDA
  // 如果使用 CUDA，标记设备未设置
  bool device_set = false;
#endif
  // 循环直到停止进度循环
  while (!stop_progress_loop) {
    // 如果进度队列为空，线程等待通知
    if (progress_queue.empty()) {
      queue_produce_cv.wait(lock);
      continue;
    }
    // 标记当前有集体操作正在进行
    collective_inprogress = true;
    // 取出队列中的第一个工作任务
    auto work = progress_queue.front();
    progress_queue.pop_front();
    // 解锁互斥锁，允许其他线程访问
    lock.unlock();
#ifdef USE_CUDA
    // 如果使用 CUDA，执行以下代码
    // 如果设备未设置，并且 cuda_device_index 不等于 TORCH_UCC_DEVICE_NOT_SET
    if ((!device_set) && (cuda_device_index != TORCH_UCC_DEVICE_NOT_SET)) {
      // 设置当前 CUDA 设备为 cuda_device_index 指定的设备
      c10::cuda::set_device(cuda_device_index);
      
      // 获取当前 CUDA 上下文
      CUcontext pctx = nullptr;
      at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx);
      
      // 如果当前上下文为空
      if (C10_UNLIKELY(!pctx)) {
        // 增加 cuda_device_index 的主要上下文的引用计数
        at::globalContext().getNVRTC().cuDevicePrimaryCtxRetain(&pctx, cuda_device_index);
        // 将 pctx 设置为当前上下文
        at::globalContext().getNVRTC().cuCtxSetCurrent(pctx);
      }
      
      // 标记设备已设置
      device_set = true;
    }
#endif
    // 声明一个异常指针
    std::exception_ptr eptr;
    // 尝试执行以下代码块，捕获可能抛出的异常
    try {
      // 当请求的状态大于0时循环执行
      while (work->request_->status > 0) {
        // 调用 UCC 通信进度函数
        ucc_comm.progress();
      }
      // 如果请求的状态小于0
      if (work->request_->status < 0) {
        // 创建一个指向运行时错误的异常指针，异常信息为 UCC 状态字符串
        eptr = std::make_exception_ptr(
            std::runtime_error(ucc_status_string(work->request_->status)));
        // 构建错误日志信息，描述通信进度失败，包含具体的操作类型或ID
        std::string err_log = c10::str(
            "Failed to progress communication", // TODO: report exact op type or
                                                // id?
            ucc_status_string(work->request_->status));
        // 记录错误日志
        TORCH_UCC_LOG_ERROR(TORCH_UCC_COLL_PROGRESS, err_log);
      }
    } catch (...) {
      // 捕获任何异常，并将异常指针指向当前异常
      eptr = std::current_exception();
    }
    // 完成工作的清理工作，传入异常指针
    work->finalize(eptr);
    // 将 work 指针置空
    work = nullptr;
    // 将 collective_inprogress 标志置为 false
    collective_inprogress = false;
    // 通知等待在 queue_consume_cv 条件变量上的一个线程
    queue_consume_cv.notify_one();
    // 锁定互斥锁
    lock.lock();
  }
}

ProcessGroupUCC::ProcessGroupUCC(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    std::chrono::duration<float> timeout)
    : Backend(rank, size), timeout_(timeout) {
  // 调用一次性初始化函数，读取配置信息
  c10::call_once(torch_ucc_config.flag, read_config);
  // 创建一个 OOB 通信信息的共享指针对象
  oob = std::make_shared<torch_ucc_oob_coll_info_t>();
  // 设置 oob 结构中的排名、大小和存储指针
  oob->rank = rank;
  oob->size = size;
  oob->store = store;
  // 初始化 comm 和 cuda_ee 指针为 nullptr
  comm = nullptr;
  cuda_ee = nullptr;
  // 静态变量，用于分配唯一的通信组ID
  static uint32_t id = 0;
  // 递增生成一个新的通信组ID
  uint32_t pg_id = id++;

  // 创建 ProcessGroupUCCLogger 对象作为日志记录器
  logger = c10::make_intrusive<ProcessGroupUCCLogger>(
      c10::str("[Rank ", rank_, "]", "[ProcessGroupUCC-", pg_id, "]"),
      TORCH_UCC_INIT);
  // 记录初始化信息，包括进程数量和超时时间
  TORCH_UCC_LOG_INFO(
      TORCH_UCC_INIT,
      c10::str(
          "Created ProcessGroupUCC with ",
          size,
          " ranks, with timeout ",
          timeout_.count(),
          " secs"));
  // 构建环境变量的描述信息字符串
  std::string envs = "";
  for (auto& torch_ucc_env : torch_ucc_envs_map) {
    envs += ("\n\t" + torch_ucc_env.first + "=" + torch_ucc_env.second);
  }
  // 记录成功读取和设置的环境变量信息
  TORCH_UCC_LOG_INFO(
      TORCH_UCC_INIT,
      c10::str(
          "Successfully read and set ProcessGroupUCC env. variables as followings",
          envs));

  // 如果启用健康检查功能，则运行健康检查
  if (torch_ucc_config.enable_health_check) {
    // 运行健康检查函数，检查 UCC/UCX 相关问题
    runHealthCheck();
  }
  // 如果启用通信日志记录功能，则初始化通信追踪器
  if (torch_ucc_config.enable_comms_logger) {
    logger->initCommsTracer();
  }
}

ProcessGroupUCC::~ProcessGroupUCC() {
  // 如果启用通信日志记录功能，则刷新通信日志
  if (torch_ucc_config.enable_comms_logger) {
    logger->flushComms(this->getRank(), this->getSize());
  }
  // 如果 comm 对象存在，则进行销毁操作
  if (comm) {
    // 设置日志阶段为 TORCH_UCC_FINALIZE
    logger->setPhase(TORCH_UCC_FINALIZE);
    // 销毁 UCC 通信团队
    comm->ucc_destroy_team(team);
    // 记录成功销毁 UCC 库的日志信息
    TORCH_UCC_LOG_INFO(
        TORCH_UCC_FINALIZE, "Successfully destroyed UCC library");
    try {
      // 如果 cuda_ee 对象存在，则依次销毁
      if (cuda_ee) {
        ucc_ee_destroy(cuda_ee);
        ucc_ee_destroy(cuda_ee_p2p[0]);
        ucc_ee_destroy(cuda_ee_p2p[1]);
      }
      // 捕获并处理可能的异常
    } catch (std::exception& ex) {
      // 捕获标准异常并处理
      TORCH_UCC_LOG_INFO(
          // 记录信息级别日志，用于通信相关组件
          TORCH_UCC_FINALIZE,
          // 信息来源于 TORCH_UCC_FINALIZE
          c10::str(
              // 使用 c10 库中的字符串操作，构造日志信息
              "(~ProcessGroupUCC) Caught error in Store Operation .. ",
              // 捕获到的错误信息的开头
              "[",
              // 开始一个新的日志条目
              ex.what(),
              // 插入捕获到的异常的描述信息
              "]"));
    }
    // 将 comm 置为 nullptr，结束通信
    comm = nullptr;
#ifdef USE_CUDA
// 如果启用了 CUDA，则定义一个函数，根据输入的排名返回相应的 CUDA 设备。
c10::Device getCUDADeviceForRank(int rank) {
  // 检查排名是否有效
  TORCH_CHECK(rank >= 0, "Invalid rank ", rank);
  // 获取当前系统上的 GPU 数量
  auto numGPUs = at::cuda::getNumGPUs();
  // 计算给定排名对应的设备索引
  auto deviceIdx = static_cast<c10::DeviceIndex>(rank % numGPUs);
  // 返回相应的 CUDA 设备
  return c10::Device(c10::DeviceType::CUDA, deviceIdx);
}
#endif

void ProcessGroupUCC::runHealthCheck() {
  // 在单独的线程中运行健康检查，并在条件变量上等待以处理超时。
  // 此设计允许处理潜在的阻塞情况。

  // 当 size_ 为 1 时，无需进行任何通信。
  if (size_ == 1)
    return;

  // 定义用于健康检查的数据结构
  struct HealthCheckData {
    std::mutex healthCheckMutex;                // 健康检查互斥量
    std::condition_variable healthCheckCv;      // 健康检查条件变量
    bool uccHealthCheckSuccess = false;         // UCC 健康检查是否成功的标志
    std::exception_ptr healthCheckException;    // 健康检查异常指针
  } healthCheckData;

  // 创建一个线程执行健康检查
  auto t = std::thread([&healthCheckData, this]() {
    std::list<c10::Device> devices{c10::kCPU};  // 创建设备列表，初始包含 CPU

#ifdef USE_CUDA
    c10::cuda::OptionalCUDAGuard gpuGuard;      // 可选的 CUDA 设备保护器
    // 如果 CUDA 可用，则添加 CUDA 设备到设备列表中
    if (at::cuda::is_available()) {
      devices.emplace_front(getCUDADeviceForRank(rank_));
    }
#endif

    // 遍历设备列表执行健康检查
    for (auto device : devices) {
      bool is_last_device = (device == devices.back());  // 判断是否为最后一个设备
      try {
        auto oob = std::make_shared<torch_ucc_oob_coll_info_t>();  // 创建共享指针 oob
        // 设置 oob 中的信息
        oob->rank = this->oob->rank;
        oob->size = this->oob->size;
        oob->store = this->oob->store;
        ucc_team_h team = nullptr;
        uint32_t comm_id;

#ifdef USE_CUDA
        // 如果当前设备是 CUDA 设备，则设置 CUDA 设备保护器
        if (device.is_cuda()) {
          gpuGuard.set_index(device.index());
        }
#endif

        // 获取通信对象 comm
        auto comm = Comm::get_comm(comm_id, device, oob, logger, true);
        // 使用 comm 创建 UCC team
        comm->ucc_create_team(team, oob);
        // 销毁 UCC team
        comm->ucc_destroy_team(team);
        // 记录 UCC 健康检查成功信息
        TORCH_UCC_LOG_INFO(
            TORCH_UCC_HEALTH_CHECK,
            c10::str(
                "UCC library health check succeed for device ",
                c10::DeviceTypeName(device.type())));
        
        // 如果是最后一个设备，标记 UCC 健康检查完成
        if (is_last_device) {
          std::lock_guard<std::mutex> lk(healthCheckData.healthCheckMutex);
          healthCheckData.uccHealthCheckSuccess = true;
        }

        // 清空 comm 和 oob 对象
        comm = nullptr;
        oob = nullptr;

        // 通知主线程健康检查完成
        if (is_last_device) {
          healthCheckData.healthCheckCv.notify_one();
        }
      } catch (const std::exception&) {
        // 捕获异常并保存异常指针
        healthCheckData.healthCheckException = std::current_exception();
        // 通知等待中的主线程报告异常
        healthCheckData.healthCheckCv.notify_one();
      } // 未知异常将导致程序终止
    }
  });
  // 等待健康检查线程结束
  t.join();
}
    }
  });
  // We don't need to join the thread, just need to verify health check via the
  // CV. Hence we detach the thread here.
  t.detach(); // NOLINT
  // 记录健康检查的等待时间
  TORCH_UCC_LOG_INFO(
      TORCH_UCC_HEALTH_CHECK,
      c10::str(
          "will wait up to ",
          timeout_.count(),
          " msec for UCC health check to complete."));
  // 创建互斥锁，并等待健康检查完成或超时
  std::unique_lock<std::mutex> lock(healthCheckData.healthCheckMutex);
  healthCheckData.healthCheckCv.wait_for(lock, timeout_, [&healthCheckData]() {
    return healthCheckData.uccHealthCheckSuccess;
  });

  // 如果健康检查发生异常，重新抛出异常
  if (healthCheckData.healthCheckException) {
    std::rethrow_exception(healthCheckData.healthCheckException);
  }
  // 如果没有异常，但健康检查失败，则可能是超时或挂起
  // 检查是否健康检查成功，否则抛出异常
  TORCH_CHECK(
      healthCheckData.uccHealthCheckSuccess,
      "ProcessGroupUCC: Health check failure: Failed to initialize UCC on rank ",
      rank_);
// 设置超时时间相关的参数
void ProcessGroupUCC::set_timeout(ucc_coll_args_t& args) {
  // 启用超时标志位
  args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
  args.flags |= UCC_COLL_ARGS_FLAG_TIMEOUT;
  // 将超时时间设置为已保存的超时计数值
  args.timeout = timeout_.count();
}

#ifdef USE_CUDA
// 获取一个 CUDA 事件对象，如果池中有空闲的则从池中取出，否则创建新的
std::unique_ptr<at::cuda::CUDAEvent> ProcessGroupUCC::getPooledEvent() {
  std::unique_ptr<at::cuda::CUDAEvent> ev;
  // 加锁保证线程安全地访问事件池
  std::lock_guard<std::mutex> lock(ep.event_pool_mutex);
  if (ep.event_pool.empty()) {
    // 如果池中没有事件，则创建一个新的 CUDA 事件对象
    ev = std::make_unique<at::cuda::CUDAEvent>();
  } else {
    // 如果池中有空闲事件，则从池中取出一个事件
    ev = std::move(ep.event_pool.front());
    ep.event_pool.pop();
  }
  return ev;
}
#endif

// 完成集体通信后的后处理过程，返回一个工作对象指针
template <typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupUCC::collective_post(
    OpType opType,
    PreProcess preproc,
    PostProcess postproc,
    ucc_coll_args_t& coll,
    std::unique_ptr<ProcessGroupUCC::WorkData> data,
    c10::Device dev,
    std::vector<at::Tensor>& inputTensors,
    std::vector<at::Tensor>& outputTensors,
    const char* prof_title) {
  // 增加序列号，用于标识本次集体操作
  seq_++;
  // 设置超时参数
  set_timeout(coll);
  // 创建一个新的集体操作工作对象
  auto work = c10::make_intrusive<ProcessGroupUCC::WorkUCC>(
      opType, seq_, prof_title, inputTensors, logger);

  // 如果是接收操作，设置接收方的排名
  if (opType == OpType::RECV) {
    work->sourceRank_ = coll.root;
  }

  // 记录通信轨迹，用于性能分析
  RECORD_COMMS_TRACE(
      logger->trace_generator,
      work,
      opType,
      this->getRank(),
      this->getSize(),
      inputTensors,
      outputTensors);

  // 存储输出张量的引用，用于后续结果处理
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputTensors);

  // 根据设备类型进行不同处理
  switch (dev.type()) {
    case c10::DeviceType::CPU: {
      // 如果配置要求使用 future 对象，则创建 future 对象
      if (torch_ucc_config.use_future) {
        work->future_ = c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
      }
      // 执行预处理操作
      preproc();
      // 将通信任务加入队列
      comm->enqueue_collective(std::move(data), work, coll, team);
      // 执行后处理操作
      postproc();
      // 返回工作对象指针
      return work;
    }
#ifdef USE_CUDA
    # 如果设备类型为 CUDA
    case c10::DeviceType::CUDA: {
      # 获取一个池化的 CUDA 事件
      auto cuda_ev = getPooledEvent();
      # 操作流和事件处理器的指针
      at::cuda::CUDAStream* op_stream;
      ucc_ee_h* op_ee;
      # 根据操作类型选择对应的流和事件处理器
      if (opType == OpType::SEND) {
        op_stream = stream_p2p[0].get();
        op_ee = &cuda_ee_p2p[0];
      } else if (opType == OpType::RECV) {
        op_stream = stream_p2p[1].get();
        op_ee = &cuda_ee_p2p[1];
      } else {
        op_stream = stream.get();
        op_ee = &cuda_ee;
      }

      # 记录当前 CUDA 流的状态到 CUDA 事件
      cuda_ev->record(at::cuda::getCurrentCUDAStream(dev.index()));
      # 阻塞当前操作流直至 CUDA 事件完成
      cuda_ev->block(*op_stream);
      # 设置当前操作流为活动 CUDA 流
      at::cuda::CUDAStreamGuard guard(*op_stream);
      # 预处理操作
      preproc();
      # 将数据、任务、集体操作、团队以及事件处理器加入 CUDA 集体通信队列
      comm->enqueue_cuda_collective(std::move(data), work, coll, team, *op_ee);
      # 后处理操作
      postproc();
      # 记录操作流的状态到 CUDA 事件
      cuda_ev->record(*op_stream);
      # 将 CUDA 事件设置为工作的 fence
      work->fence = std::move(cuda_ev);
      # 将工作指向当前处理器
      work->ep = &ep;
      
      # 如果启用了 Torch UCC 的未来功能
      if (torch_ucc_config.use_future) {
        # 在多 CUDA 流中保护操作流
        c10::cuda::CUDAMultiStreamGuard streamGuard(*op_stream);
        # 创建包含给定设备的 IVT 值类型的未来
        std::vector<c10::Device> devList{dev};
        work->future_ = c10::make_intrusive<at::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()), devList);
        # 添加一个回调函数，在未来完成时运行函数结束回调
        if (work->recordFunctionEndCallback_) {
          work->future_->addCallback([work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          });
        }

        # 将输出张量标记为已完成状态
        work->future_->markCompleted(c10::IValue(outputTensors));
      }
      # 返回处理后的工作对象
      return work;
    }
#else // #ifdef USE_CUDA
    // 如果未定义 USE_CUDA，则执行以下代码块
    default: {
      // 如果不支持的设备类型，记录错误日志并抛出异常
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST, c10::str("unsupported device type ", dev.str()));
      throw std::invalid_argument(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
    }
  }
}

// 执行 allgather 操作的函数，返回一个指向 Work 对象的智能指针
c10::intrusive_ptr<Work> ProcessGroupUCC::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {
  auto& tensor = inputTensors[0];
  // 检查输入张量与输出张量的设备是否一致
  check_device(tensor.device(), outputTensors[0][0].device());
  // 初始化通信环境，根据输入张量的设备类型
  initComm(tensor.device());

  // 如果张量在 CPU 设备上，或者使用了 allgatherv 的配置选项
  if (tensor.device().is_cpu() || torch_ucc_config.use_allgatherv) {
    // 创建 allgatherv 操作所需的数据对象
    AllgathervWorkData* data = new AllgathervWorkData(size_);
    for (int i = 0; i < size_; i++) {
      // 设置每个接收者的接收长度为张量大小的字节数
      data->recv_lengths[i] = tensor.element_size() * tensor.numel();
      // 设置每个接收者的接收偏移量为输出张量中对应张量的数据指针地址
      data->recv_offsets[i] = (uint64_t)outputTensors[0][i].data_ptr();
    }
    // 设置 UCC 的集合操作参数
    ucc_coll_args_t coll;
    coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll.flags =
        UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
    coll.coll_type = UCC_COLL_TYPE_ALLGATHERV;
    coll.src.info.buffer = tensor.data_ptr();
    coll.src.info.count = tensor.element_size() * tensor.numel();
    coll.src.info.datatype = UCC_DT_UINT8;
    coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
    coll.dst.info_v.buffer = nullptr;
    coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
    coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
    coll.dst.info_v.datatype = UCC_DT_UINT8;
    coll.dst.info_v.mem_type =
        to_ucc_memType(outputTensors[0][0].device().type());
    SAVE_TENSORS(inputTensors, data->src);
    SAVE_TENSORS(outputTensors[0], data->dst);

    // 返回一个异步执行的 allgather 操作的智能指针
    return collective_post(
        OpType::ALLGATHER,
        []() {},
        []() {},
        coll,
        std::unique_ptr<WorkData>(data),
        tensor.device(),
        inputTensors,
        outputTensors[0],
        "ucc:all_gather");
  } else {
    // 如果不在 CPU 设备上，且未使用 allgatherv 的配置选项，则执行以下代码块

    // 创建通用工作数据对象
    WorkData* data = new WorkData();
    std::vector<at::Tensor> flat_output(outputTensors.size());
    for (size_t i = 0; i < outputTensors.size(); i++) {
      // 检查输出张量列表的有效性，确保张量数量与参与者数量对应
      TORCH_CHECK(
          outputTensors[i].size() == outputTensors.size() * size_,
          "Tensor output list is not valid for the number of participants");
      // 创建扁平化的输出张量列表
      flat_output[i] = c10d::newLikeFlat(outputTensors, i);
    }
    // 保存扁平化的输出张量列表到工作数据对象中
    SAVE_TENSORS(flat_output, data->flat);

    // 设置 UCC 的集合操作参数
    ucc_coll_args_t coll;
    coll.mask = 0;
    coll.flags = 0;
    coll.coll_type = UCC_COLL_TYPE_ALLGATHER;
    coll.src.info.buffer = tensor.data_ptr();
    coll.src.info.count = tensor.numel();
    coll.src.info.datatype = to_ucc_dType(tensor);
    coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
    coll.dst.info.buffer = flat_output[0].data_ptr();
    coll.dst.info.count = flat_output[0].numel();
    coll.dst.info.datatype = to_ucc_dType(flat_output[0]);
    coll.dst.info.mem_type =
        to_ucc_memType(outputTensors[0][0].device().type());
    # 定义一个名为 copy_from_flat 的 lambda 函数，使用捕获列表 [&] 捕获所有外部变量
    auto copy_from_flat = [&] {
        # 声明并初始化一个布尔变量 asyncCopy，赋值为 false
        bool asyncCopy = false;
#ifdef USE_CUDA
      // 检查第一个输出张量是否在 CUDA 设备上
      bool isCuda = outputTensors[0][0].device().is_cuda();
      ;
#endif
      // 遍历所有输出张量
      for (size_t i = 0; i < outputTensors.size(); i++) {
        // 获取当前输入张量的元素总数
        auto inumel = inputTensors[i].numel();
        // 遍历当前输出张量中的所有张量
        for (size_t j = 0; j < outputTensors[i].size(); j++) {
          // 检查当前输出张量的元素总数是否与输入张量相同
          TORCH_CHECK(
              (outputTensors[i][j].numel() == inumel),
              "Tensor operand counts must be same");
#ifdef USE_CUDA
          // 如果是 CUDA 环境，并且 isCuda 为 true
          if (isCuda) {
            // 记录异步流中的 CUDA 内存操作
            c10::cuda::CUDACachingAllocator::recordStream(
                outputTensors[i][j].storage().data_ptr(), (*stream));
            // 设置异步拷贝标志为 true
            asyncCopy = true;
          }
#endif
          // 使用异步拷贝将 flat_output 中的数据复制到当前输出张量中
          outputTensors[i][j].copy_(flat_output[i][j], asyncCopy);
        }
      }
    };
    // 返回集合操作的后处理结果
    return collective_post(
        OpType::ALLGATHER,
        []() {},  // 空 Lambda 函数
        copy_from_flat,
        coll,
        std::unique_ptr<WorkData>(data),
        tensor.device(),
        inputTensors,
        outputTensors[0],
        "ucc:all_gather");
  }
}

// 执行基础的 allgather 操作，并返回工作指针
c10::intrusive_ptr<Work> ProcessGroupUCC::_allgather_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const AllgatherOptions& opts) {
  // 检查输出张量是否合法
  check_tensor({outputTensor});
  // 检查输入张量是否合法
  check_tensor({inputTensor});
  // 初始化通信机制，根据输出张量所在设备
  initComm(outputTensor.device());

  // 创建工作数据对象
  WorkData* data = new WorkData();

  // 设置 UCC 收集操作参数
  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_ALLGATHER;
  coll.src.info.buffer = inputTensor.data_ptr();
  coll.src.info.count = inputTensor.numel();
  coll.src.info.datatype = ucc_dtype_map.at(inputTensor.scalar_type());
  coll.src.info.mem_type = to_ucc_memType(inputTensor.device().type());
  coll.dst.info.buffer = outputTensor.data_ptr();
  coll.dst.info.count = outputTensor.numel();
  coll.dst.info.datatype = ucc_dtype_map.at(outputTensor.scalar_type());
  coll.dst.info.mem_type = to_ucc_memType(outputTensor.device().type());

  // 构建输入和输出张量的向量
  std::vector<at::Tensor> inputTensors = {inputTensor};
  std::vector<at::Tensor> outputTensors = {outputTensor};
  // 将输入和输出张量保存到工作数据中
  SAVE_TENSORS(inputTensors, data->src);
  SAVE_TENSORS(outputTensors, data->dst);

  // 执行集合操作后，返回工作指针
  return collective_post(
      OpType::_ALLGATHER_BASE,
      []() {},  // 空 Lambda 函数
      []() {},  // 空 Lambda 函数
      coll,
      std::unique_ptr<WorkData>(data),
      outputTensor.device(),
      inputTensors,
      outputTensors,
      "ucc:allgather_base");
}
    // 定义一个函数，执行分布式的 all_reduce 操作，输入参数包括张量和选项
    const AllreduceOptions& opts) {
      // 检查输入的张量列表是否合法
      check_tensor(tensors);
      // 获取第一个张量作为操作的主张量
      auto& tensor = tensors[0];
      // 根据张量的设备初始化通信环境
      initComm(tensor.device());
      // 分配一个新的工作数据对象
      WorkData* data = new WorkData();
    
      // 创建 UCC 收集操作的参数对象
      ucc_coll_args_t coll;
      coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
      coll.flags = UCC_COLL_ARGS_FLAG_IN_PLACE; // 设置为原地操作标志
      coll.coll_type = UCC_COLL_TYPE_ALLREDUCE; // 指定收集操作的类型为 allreduce
      coll.op = to_ucc_reduceOp(opts.reduceOp, tensor.scalar_type()); // 设置 UCC 的 reduce 操作类型
      coll.src.info.buffer = nullptr; // 源数据缓冲区为空，因为使用了原地操作
      coll.src.info.count = tensor.numel(); // 设置源数据的元素数量
      coll.src.info.datatype = to_ucc_dType(tensor); // 设置源数据的数据类型
      coll.src.info.mem_type = to_ucc_memType(tensor.device().type()); // 设置源数据的内存类型
      coll.dst.info.buffer = tensor.data_ptr(); // 设置目标数据的缓冲区指针为张量的数据指针
      coll.dst.info.count = tensor.numel(); // 设置目标数据的元素数量
      coll.dst.info.datatype = to_ucc_dType(tensor); // 设置目标数据的数据类型
      coll.dst.info.mem_type = to_ucc_memType(tensor.device().type()); // 设置目标数据的内存类型
    
      // 将张量列表保存到工作数据对象中的目标张量列表
      SAVE_TENSORS(tensors, data->dst);
    
      // 执行集体通信的后处理，提交 allreduce 操作
      return collective_post(
          OpType::ALLREDUCE, // 操作类型为 allreduce
          []() {}, // 空 lambda 函数，不执行前处理操作
          []() {}, // 空 lambda 函数，不执行后处理操作
          coll, // 使用预先配置好的 UCC 收集操作参数
          std::unique_ptr<WorkData>(data), // 传递工作数据对象的唯一指针
          tensor.device(), // 使用张量的设备作为操作的设备
          tensors, // 传递原始的张量列表
          tensors, // 传递原始的张量列表
          "ucc:all_reduce"); // 提供操作的描述信息
    }
}

c10::intrusive_ptr<Work> ProcessGroupUCC::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  // 抛出异常，因为 ProcessGroupUCC 不支持 allreduce_coalesced 操作
  throw std::invalid_argument(
      "ProcessGroupUCC does not support allreduce_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& /* unused */) {
  auto device = outputTensors[0].device();
  // 确保所有张量位于相同的设备上
  for (const auto r : c10::irange(outputTensors.size())) {
    TORCH_CHECK(
        device == outputTensors[r].device() &&
            device == inputTensors[r].device(),
        "Tensors must be on the same device")
  }

  // 初始化通信相关设置
  initComm(device);
  
  // 定义 UCC 收集操作的参数
  ucc_coll_args_t coll;
  AlltoallWorkData* data;
  // 创建 AlltoallWorkData 对象，指定 size_
  data = new AlltoallWorkData(size_);

  /* 避免扁平化张量，使用 alltoallv 实现 Alltoall 操作，具体步骤如下：
     1. 直接将每个张量的地址存储在位移中，缓冲区保持为 nullptr（即 0）
     2. 将数据类型转换为 UINT8，其大小始终为 1 字节，避免在 UCC 层中计算错误的大小
     3. 发起 Alltoallv 操作
  */
  for (const auto i : c10::irange(size_)) {
    data->send_lengths[i] =
        (uint64_t)(inputTensors[i].element_size() * inputTensors[i].numel());
    data->send_offsets[i] = (uint64_t)inputTensors[i].data_ptr();
    data->recv_lengths[i] =
        (uint64_t)(outputTensors[i].element_size() * outputTensors[i].numel());
    data->recv_offsets[i] = (uint64_t)outputTensors[i].data_ptr();
  }

  // 设置 UCC 收集参数的标志和类型
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags =
      UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  coll.coll_type = UCC_COLL_TYPE_ALLTOALLV;
  coll.src.info_v.buffer = 0;
  coll.src.info_v.counts = (ucc_count_t*)data->send_lengths.data();
  coll.src.info_v.displacements = (ucc_aint_t*)data->send_offsets.data();
  coll.src.info_v.datatype = UCC_DT_UINT8;
  coll.src.info_v.mem_type = to_ucc_memType(inputTensors[0].device().type());
  coll.dst.info_v.buffer = 0;
  coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
  coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
  coll.dst.info_v.datatype = UCC_DT_UINT8;
  coll.dst.info_v.mem_type = to_ucc_memType(outputTensors[0].device().type());

  // 保存输入和输出张量信息
  SAVE_TENSORS(inputTensors, data->src);
  SAVE_TENSORS(outputTensors, data->dst);

  // 返回集体通信的异步操作指针
  return collective_post(
      OpType::ALLTOALL,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      device,
      inputTensors,
      outputTensors,
      "ucc:alltoall");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    // 定义函数，声明参数为 AllToAllOptions 的引用（未使用）
    const AllToAllOptions& /* unused */) {
      // 检查输入输出张量所在设备是否一致
      check_device(inputTensor.device(), outputTensor.device());
      // 初始化通信，根据输入张量所在设备
      initComm(inputTensor.device());
      // 定义 UCC 通信的参数结构体和 AlltoallWorkData 对象
      ucc_coll_args_t coll;
      AlltoallWorkData* data;
    
      // 如果未指定输入和输出的分割大小
      if ((outputSplitSizes.size() == 0) && (inputSplitSizes.size() == 0)) {
        // 创建一个没有分片数据的 AlltoallWorkData 对象
        data = new AlltoallWorkData(0);
        // 检查张量的第 0 维是否能均匀地分配给所有处理组
        TORCH_CHECK(
            (outputTensor.size(0) % size_ == 0) &&
                (inputTensor.size(0) % size_ == 0),
            "Tensor's dim 0 does not divide equally across group size");
        // 初始化通信集合参数的字段
        coll.mask = 0;
        coll.flags = 0;
        coll.coll_type = UCC_COLL_TYPE_ALLTOALL;
        coll.src.info.buffer = inputTensor.data_ptr();
        coll.src.info.count = inputTensor.element_size() * inputTensor.numel();
        coll.src.info.datatype = UCC_DT_UINT8;
        coll.src.info.mem_type = to_ucc_memType(inputTensor.device().type());
        coll.dst.info.buffer = outputTensor.data_ptr();
        coll.dst.info.count = outputTensor.element_size() * outputTensor.numel();
        coll.dst.info.datatype = UCC_DT_UINT8;
        coll.dst.info.mem_type = to_ucc_memType(outputTensor.device().type());
        coll.flags = 0; // 重置标志位为默认值
      } else {
        // 否则，根据处理组大小创建 AlltoallWorkData 对象
        data = new AlltoallWorkData(size_);
        // 检查输入和输出的分割大小是否有效
        c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
        c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
        // 计算输出张量分片的长度和偏移量
        computeLengthsAndOffsets(
            outputSplitSizes,
            outputTensor,
            &data->recv_lengths,
            &data->recv_offsets);
        // 计算输入张量分片的长度和偏移量
        computeLengthsAndOffsets(
            inputSplitSizes, inputTensor, &data->send_lengths, &data->send_offsets);
        // 设置通信集合参数的字段
        coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        coll.coll_type = UCC_COLL_TYPE_ALLTOALLV;
        coll.src.info_v.buffer = inputTensor.data_ptr();
        coll.src.info_v.counts = (ucc_count_t*)data->send_lengths.data();
        coll.src.info_v.displacements = (ucc_aint_t*)data->send_offsets.data();
        coll.src.info_v.datatype = to_ucc_dType(inputTensor);
        coll.src.info_v.mem_type = to_ucc_memType(inputTensor.device().type());
        coll.dst.info_v.buffer = outputTensor.data_ptr();
        coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
        coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
        coll.dst.info_v.datatype = to_ucc_dType(outputTensor);
        coll.dst.info_v.mem_type = to_ucc_memType(outputTensor.device().type());
        // 设置通信集合参数的标志位，指示使用连续的源和目标缓冲区，
        // 以及使用 64 位计数和偏移量
        coll.flags = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
            UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER | UCC_COLL_ARGS_FLAG_COUNT_64BIT |
            UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
    
        // 如果启用了通信记录器，则记录可选信息
        if (torch_ucc_config.enable_comms_logger) {
          logger->trace_generator->recordOptionalInfo(
              outputSplitSizes, inputSplitSizes);
    }
  }
  // 创建输入张量的向量，其中包含一个输入张量
  std::vector<at::Tensor> inputTensors = {inputTensor};
  // 创建输出张量的向量，其中包含一个输出张量
  std::vector<at::Tensor> outputTensors = {outputTensor};
  // 将输入张量保存到给定的数据源
  SAVE_TENSORS(inputTensors, data->src);
  // 将输出张量保存到给定的数据目标
  SAVE_TENSORS(outputTensors, data->dst);

  // 执行集体通信操作，类型为 ALLTOALL_BASE
  return collective_post(
      OpType::ALLTOALL_BASE,     // 操作类型为 ALLTOALL_BASE
      []() {},                   // 空的 Lambda 函数，用于执行前处理操作
      []() {},                   // 空的 Lambda 函数，用于执行后处理操作
      coll,                      // 通信收集器对象
      std::unique_ptr<WorkData>(data),  // 包含工作数据的独特指针
      inputTensor.device(),      // 输入张量所在的设备
      inputTensors,              // 输入张量的向量
      outputTensors,             // 输出张量的向量
      "ucc:alltoall");           // 指定通信操作的标识符
}

c10::intrusive_ptr<Work> ProcessGroupUCC::barrier(const BarrierOptions& opts) {
  // 默认设备为 CPU
  c10::Device device = c10::Device(c10::DeviceType::CPU);
#ifdef USE_CUDA
  // 获取 CUDA 设备数量
  auto numGPUs = c10::cuda::device_count();
  // 如果用户指定了设备 IDs，则使用第一个设备
  if (!opts.device_ids.empty()) {
    device = c10::Device(c10::DeviceType::CUDA, opts.device_ids.front());
  } else if (comm && comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) {
    // 如果通信对象已设置了 CUDA 设备索引，则使用该索引
    device = c10::Device(c10::DeviceType::CUDA, comm->cuda_device_index);
  } else if (numGPUs > 0) {
    // 获取当前 CUDA 设备索引
    int8_t deviceIdx = static_cast<int8_t>(c10::cuda::current_device());
    // 如果当前设备索引为 0，根据当前进程的 rank 选择一个 GPU 设备
    if (0 == (int)deviceIdx) {
      deviceIdx = static_cast<int8_t>(this->getRank() % numGPUs);
    }
    // 日志记录关于 GPU 设备选择的信息
    TORCH_UCC_LOG_INFO(
        TORCH_UCC_COLL_POST,
        c10::str(
            "post barrier before specifying any GPU while there are ",
            numGPUs,
            " GPUs available. ",
            "Not clear if GPU barrier is required, using GPU ",
            (int)deviceIdx,
            " to perform barrier. ",
            "Specify device_ids option in barrier() to force ",
            "use of a particular device"));
    // 使用选择的 GPU 设备
    device = c10::Device(c10::DeviceType::CUDA, deviceIdx);
  }
#endif
  // 初始化通信对象
  initComm(device);

  // 设置 UCC 的集合参数为屏障类型
  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_BARRIER;
  // 创建一个空的 Tensor 列表作为占位符
  auto dummy_tensor = std::vector<at::Tensor>();
  // 调用 collective_post 函数发起屏障操作
  return collective_post(
      OpType::BARRIER,
      []() {},  // 空 lambda 函数
      []() {},  // 空 lambda 函数
      coll,
      nullptr,
      device,
      dummy_tensor,
      dummy_tensor,
      "ucc:barrier");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  // 检查传入的 Tensor 是否有效
  check_tensor(tensors);
  // 获取第一个 Tensor，并初始化通信对象
  auto& tensor = tensors[0];
  initComm(tensor.device());
  // 创建一个工作数据对象
  WorkData* data = new WorkData();

  // 设置 UCC 的集合参数为广播类型
  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_BCAST;
  coll.src.info.buffer = tensor.data_ptr();
  coll.src.info.count = tensor.numel();
  coll.src.info.datatype = to_ucc_dType(tensor);
  coll.src.info.mem_type = to_ucc_memType(tensor.device().type());
  coll.root = opts.rootRank;
  // 保存 Tensor 到目标列表中
  SAVE_TENSORS(tensors, data->dst);

  // 如果启用通信日志记录，则记录相关信息
  if (torch_ucc_config.enable_comms_logger) {
    logger->trace_generator->recordOptionalInfo(opts.rootRank);
  }

  // 调用 collective_post 函数发起广播操作
  return collective_post(
      OpType::BROADCAST,
      []() {},  // 空 lambda 函数
      []() {},  // 空 lambda 函数
      coll,
      std::unique_ptr<WorkData>(data),
      tensor.device(),
      tensors,
      tensors,
      "ucc:broadcast");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  // 创建一个空的张量向量用于存储输出结果
  std::vector<at::Tensor> outputs;
  // 获取输入张量的引用
  auto& input = inputTensors[0];
  // 初始化通信设备
  initComm(input.device());

  // 创建一个新的 AllgathervWorkData 对象，大小为 size_
  AllgathervWorkData* data = new AllgathervWorkData(size_);
  // 创建 UCC 集合参数对象 coll
  ucc_coll_args_t coll;
  // 指定根进程的排名
  coll.root = opts.rootRank;
  // 指定 UCC_COLL_ARGS_FIELD_FLAGS 为 mask
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  // 设置标志位，指明 count 和 displacements 使用 64 位
  coll.flags =
      UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  // 指定集合类型为 GATHERV
  coll.coll_type = UCC_COLL_TYPE_GATHERV;

  /* for non-root ranks, only src is valid */
  // 对于非根进程，只有 src 是有效的
  coll.src.info.buffer = input.data_ptr();
  // 设置 src 的数据量，以字节为单位
  coll.src.info.count = (uint64_t)(input.element_size() * input.numel());
  // 设置数据类型为 UCC_DT_UINT8
  coll.src.info.datatype = UCC_DT_UINT8;
  // 根据输入张量的设备类型，设置内存类型
  coll.src.info.mem_type = to_ucc_memType(input.device().type());

  // 如果当前进程是根进程
  if (getRank() == opts.rootRank) {
    // 检查输出张量列表是否只有一个元素
    if (outputTensors.size() != 1) {
      // 如果不是，输出错误日志
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST,
          c10::str(
              "gather requires a single-element output list containing a list with ",
              getSize(),
              " tensors."));
    } else if (outputTensors[0].size() != static_cast<size_t>(getSize())) {
      // 检查输出张量的大小是否与进程组大小相匹配
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST,
          c10::str(
              "Incorrect output list size ",
              outputTensors[0].size(),
              ". Output list size should be ",
              getSize(),
              ", same as size of the process group."));
    }
    // 将输出张量列表设置为输出
    outputs = outputTensors[0];

    // 遍历所有进程
    for (int i = 0; i < size_; i++) {
      // 设置每个接收长度为对应输出张量的数据量，以字节为单位
      data->recv_lengths[i] =
          (uint64_t)(outputs[i].element_size() * outputs[i].numel());
      // 设置每个接收偏移为对应输出张量的数据指针
      data->recv_offsets[i] = (uint64_t)outputs[i].data_ptr();
    }
    /* use gatherv and store non-contiguous addresses in displacements to avoid
     * flatten outputTensors */
    // 使用 gatherv 并在 displacements 中存储非连续地址，以避免扁平化输出张量
    coll.dst.info_v.buffer = nullptr;
    // 设置 dst 信息的 counts 为接收长度数组的数据
    coll.dst.info_v.counts = (ucc_count_t*)data->recv_lengths.data();
    // 设置 dst 信息的 displacements 为接收偏移数组的数据
    coll.dst.info_v.displacements = (ucc_aint_t*)data->recv_offsets.data();
    // 设置数据类型为 UCC_DT_UINT8
    coll.dst.info_v.datatype = UCC_DT_UINT8;
    // 根据输出张量的设备类型，设置内存类型
    coll.dst.info_v.mem_type = to_ucc_memType(outputs[0].device().type());

    // 保存输出张量到 data 的 dst
    SAVE_TENSORS(outputs, data->dst);
  } else {
    // 对于非根进程，outputTensors 应该是一个空列表
    if (outputTensors.size() != 0) {
      // 如果不是空列表，输出错误日志
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST, "requires empty output on non-root");
    }
    // 将 outputs 设置为空向量
    outputs = {};
    // 向 outputs 添加一个空张量，以便将来使用
    outputs.emplace_back();
  }

  // 保存输入张量到 data 的 src
  SAVE_TENSORS(inputTensors, data->src);

  // 执行集合操作的后处理，使用 OpType::GATHER 类型，无操作回调，无错误回调，使用 coll 参数
  // 使用 data 的唯一指针，输入张量的设备类型，输入张量列表，输出张量列表，指定标签为 "ucc:gather"
  return collective_post(
      OpType::GATHER,
      []() {},
      []() {},
      coll,
      std::unique_ptr<WorkData>(data),
      input.device(),
      inputTensors,
      outputs,
      "ucc:gather");
}

c10::intrusive_ptr<Work> ProcessGroupUCC::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  // 检查输入张量列表的有效性
  check_tensor(tensors);
  // 获取第一个张量作为操作的基准张量
  auto& tensor = tensors[0];
  // 初始化与张量设备相关的通信环境
  initComm(tensor.device());
  // 创建一个新的工作数据对象
  WorkData* data = new WorkData();

  // 设置集合通信的参数结构体
  ucc_coll_args_t coll;
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
  coll.coll_type = UCC_COLL_TYPE_REDUCE;
  coll.op = ucc_op_map.at(opts.reduceOp); // 根据选项映射操作类型
  coll.root = opts.rootRank; // 设置根节点的排名
  coll.src.info.buffer = tensor.data_ptr(); // 源数据缓冲区指针
  coll.src.info.count = tensor.numel(); // 元素数量
  coll.src.info.datatype = ucc_dtype_map.at(tensor.scalar_type()); // 数据类型
  coll.src.info.mem_type = to_ucc_memType(tensor.device().type()); // 内存类型
  coll.dst.info.buffer = tensor.data_ptr(); // 目标数据缓冲区指针（对于原地操作）
  coll.dst.info.count = tensor.numel(); // 元素数量
  coll.dst.info.datatype = ucc_dtype_map.at(tensor.scalar_type()); // 数据类型
  coll.dst.info.mem_type = to_ucc_memType(tensor.device().type()); // 内存类型
  SAVE_TENSORS(tensors, data->dst); // 保存张量到工作数据的目标列表
  // 执行集合通信的后处理，并返回相应的工作对象
  return collective_post(
      OpType::REDUCE,
      []() {}, // 空操作，无需处理回调
      []() {}, // 空操作，无需处理回调
      coll,
      std::unique_ptr<WorkData>(data), // 包含工作数据的独占指针
      tensor.device(),
      tensors,
      tensors,
      "ucc:reduce"); // 操作类型标识字符串
}

c10::intrusive_ptr<Work> ProcessGroupUCC::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  // 检查输出张量列表和输入张量列表的匹配性
  TORCH_CHECK(
      (outputTensors.size() == inputTensors.size()),
      "Tensor input/output list for reduce_scatter must have same size");
  // 检查输出张量列表的有效性
  check_tensor(outputTensors);
  // 检查输入张量的设备是否与输出张量的设备相匹配
  check_device(inputTensors[0][0].device(), outputTensors[0].device());
  // 初始化与输入张量设备相关的通信环境
  initComm(inputTensors[0][0].device());
  // 创建一个新的工作数据对象
  auto data = std::make_unique<WorkData>();
  // 创建扁平化的输入张量列表
  std::vector<at::Tensor> flat_input(inputTensors.size());
  for (size_t i = 0; i < inputTensors.size(); i++) {
    // 检查输入张量列表的有效性与参与者数量的匹配性
    TORCH_CHECK(
        inputTensors[i].size() == inputTensors.size() * size_,
        "Tensor input list is not valid for the number of participants");
    // 根据索引创建与输入张量列表相似的扁平化张量
    flat_input[i] = c10d::newLikeFlat(inputTensors, i);
  }
  // 将扁平化的输入张量列表保存到工作数据的源列表中
  SAVE_TENSORS(flat_input, data->flat);
  // 检查扁平化输入张量列表的有效性
  check_tensor(flat_input);
  // 设置集合通信的参数结构体
  ucc_coll_args_t coll;
  coll.mask = 0;
  coll.flags = 0;
  coll.coll_type = UCC_COLL_TYPE_REDUCE_SCATTER;
  coll.op = to_ucc_reduceOp(opts.reduceOp, flat_input[0].scalar_type());

  coll.src.info.buffer = flat_input[0].data_ptr();
  coll.src.info.count = flat_input[0].numel();
  coll.src.info.datatype = to_ucc_dType(flat_input[0]);
  coll.src.info.mem_type = to_ucc_memType(flat_input[0].device().type());
  coll.dst.info.buffer = outputTensors[0].data_ptr();
  coll.dst.info.count = outputTensors[0].numel();
  coll.dst.info.datatype = to_ucc_dType(outputTensors[0]);
  coll.dst.info.mem_type = to_ucc_memType(outputTensors[0].device().type());

  // 将输入张量列表和输出张量列表保存到工作数据的源列表和目标列表中
  SAVE_TENSORS(inputTensors[0], data->src);
  SAVE_TENSORS(outputTensors, data->dst);

  // 定义一个闭包，用于将数据复制到扁平化的输入张量列表中
  auto copy_to_flat = [&] {
    bool asyncCopy = false;
    auto isize = inputTensors.size();
#ifdef USE_CUDA
    bool isCuda = inputTensors[0][0].device().is_cuda();
#endif
    # 遍历输出张量列表中的每一个张量
    for (size_t i = 0; i < isize; i++) {
      # 获取当前输出张量的元素数量
      auto onumel = outputTensors[i].numel();
      # 遍历当前输入张量的维度大小
      for (size_t j = 0; j < inputTensors[i].size(); j++) {
        # 检查当前输入张量和输出张量的元素数量是否相等
        TORCH_CHECK(
            (inputTensors[i][j].numel() == onumel),
            "Tensor operand counts must be same");
#ifdef USE_CUDA
        // 如果使用 CUDA，并且当前操作涉及到 CUDA，则记录当前张量的数据指针和流，用于异步拷贝
        if (isCuda) {
          c10::cuda::CUDACachingAllocator::recordStream(
              inputTensors[i][j].storage().data_ptr(), (*stream));
          asyncCopy = true;
        }
#endif
        // 将输入张量的数据拷贝到扁平化的输入张量中，根据 asyncCopy 是否为真进行异步拷贝
        flat_input[i][j].copy_(inputTensors[i][j], asyncCopy);
      }
    }
  };

  // 调用 collective_post 函数进行集体通信操作，类型为 REDUCE_SCATTER
  return collective_post(
      OpType::REDUCE_SCATTER,
      // 函数对象，将输入张量复制到扁平化输入张量中
      copy_to_flat,
      // 空的完成回调函数
      []() {},
      coll,  // UCC 通信参数
      std::move(data),  // 移动数据对象
      inputTensors[0][0].device(),  // 输入张量的设备类型
      inputTensors[0],  // 输入张量列表
      outputTensors,  // 输出张量列表
      "ucc:reduce_scatter");  // 操作类型描述
}

c10::intrusive_ptr<Work> ProcessGroupUCC::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  auto& tensor = outputTensors[0];
  // 初始化与张量设备对应的通信环境
  initComm(tensor.device());

  // 创建 ScattervWorkData 对象来存储散射操作所需的数据
  ScattervWorkData* data = new ScattervWorkData(size_);
  ucc_coll_args_t coll;
  coll.root = opts.rootRank;  // 散射根节点的排名
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  // 设置通信标志位，包括使用 64 位计数和位移
  coll.flags =
      UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  coll.coll_type = UCC_COLL_TYPE_SCATTERV;  // 指定通信类型为散射v

  if (getRank() == opts.rootRank) {
    // 如果当前进程是根节点进程

    /* src is only valid at non-root rank */
    // 在非根节点进程，src 是无效的

    // 检查输入张量列表的长度是否为 1
    if (inputTensors.size() != 1) {
      // 报错：gather 需要一个包含 getSize() 个张量的单元素输出列表
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST,
          c10::str(
              "gather requires a single-element output list containing a list with ",
              getSize(),
              " tensors."));
    } else if (inputTensors[0].size() != static_cast<size_t>(getSize())) {
      // 报错：输出列表的大小不正确，应与进程组的大小相同
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST,
          c10::str(
              "Incorrect output list size ",
              inputTensors[0].size(),
              ". Output list size should be ",
              getSize(),
              ", same as size of the process group."));
    }

    // 遍历所有进程，设置发送长度和偏移量
    for (int i = 0; i < size_; i++) {
      data->send_lengths[i] = (uint64_t)tensor.element_size() * tensor.numel();
      data->send_offsets[i] = (uint64_t)inputTensors[0][i].data_ptr();
    }

    /* use scatter and store non-contiguous addresses in displacements to avoid
     * flatten inputTensors */
    // 使用散射操作，并将非连续地址存储在位移中，以避免扁平化输入张量
    coll.src.info_v.buffer = nullptr;
    coll.src.info_v.counts = (ucc_count_t*)data->send_lengths.data();
    coll.src.info_v.displacements = (ucc_aint_t*)data->send_offsets.data();
    coll.src.info_v.datatype = UCC_DT_UINT8;
    coll.src.info_v.mem_type =
        to_ucc_memType(inputTensors[0][0].device().type());

    // 保存输入张量到数据对象的 src 字段
    SAVE_TENSORS(inputTensors[0], data->src);
  } else {
    // 对于非根节点进程，输入张量应为空列表
    if (inputTensors.size() != 0) {
      // 报错：在非根节点上需要空的输出
      TORCH_UCC_LOG_ERROR(
          TORCH_UCC_COLL_POST, "requires empty output on non-root");
      //
  }
}

coll.dst.info.buffer = tensor.data_ptr();
// 将张量的数据指针赋给目标信息结构体的缓冲区字段
coll.dst.info.count = (uint64_t)tensor.element_size() * tensor.numel();
// 计算张量数据的总字节数，并赋给目标信息结构体的计数字段
coll.dst.info.datatype = UCC_DT_UINT8;
// 设置目标信息结构体的数据类型为无符号8位整数
coll.dst.info.mem_type = to_ucc_memType(tensor.device().type());
// 将张量的设备类型转换为 UCC 内存类型，并赋给目标信息结构体的内存类型字段
SAVE_TENSORS(outputTensors, data->dst);
// 保存输出张量到数据结构体的目标字段

return collective_post(
    OpType::SCATTER,
    []() {},
    []() {},
    coll,
    std::unique_ptr<WorkData>(data),
    tensor.device(),
    (getRank() == opts.rootRank) ? inputTensors[0] : outputTensors,
    // 如果当前进程是根进程，使用输入张量作为收集操作的输入；否则使用输出张量
    outputTensors,
    // 输出张量作为收集操作的输出
    "ucc:scatter");
// 执行收集操作，将结果返回
}

c10::intrusive_ptr<Work> ProcessGroupUCC::send(
    std::vector<at::Tensor>& tensors,  // 接收一个张量向量作为参数
    int dstRank,  // 目标排名，表示接收方的排名
    int tag) {  // 标签，用于区分不同的通信操作
  check_tensor(tensors);  // 检查张量向量的有效性
  auto& tensor = tensors[0];  // 获取张量向量中的第一个张量
  initComm(tensor.device());  // 根据张量所在的设备初始化通信环境

  WorkData* data = new WorkData();  // 创建工作数据对象
  ucc_coll_args_t coll;  // 定义 UCC 收集通信参数结构体
  coll.tag = tag;  // 设置通信操作的标签
  coll.mask = UCC_COLL_ARGS_FIELD_ACTIVE_SET | UCC_COLL_ARGS_FIELD_TAG;  // 指定参数掩码
  coll.flags = 0;  // 标志位初始化为0
  coll.coll_type = UCC_COLL_TYPE_BCAST;  // 设置通信类型为广播
  coll.src.info.buffer = tensor.data_ptr();  // 设置通信源的缓冲区
  coll.src.info.count = tensor.numel();  // 设置通信源数据的元素个数
  coll.src.info.datatype = to_ucc_dType(tensor);  // 将张量数据类型转换为 UCC 支持的数据类型
  coll.src.info.mem_type = to_ucc_memType(tensor.device().type());  // 将设备类型转换为 UCC 支持的内存类型
  coll.root = getRank();  // 设置通信的根节点排名

  coll.active_set.size = 2;  // 设置活动通信集的大小为2
  coll.active_set.start = getRank();  // 设置活动通信集的起始排名
  coll.active_set.stride = dstRank - getRank();  // 设置活动通信集的步长，用于指定通信的接收方排名
  SAVE_TENSORS(tensors, data->dst);  // 保存张量向量到工作数据对象的目标属性

  return collective_post(
      OpType::SEND,  // 发送操作的类型
      []() {},  // 空的 Lambda 函数，用作通信完成后的回调
      []() {},  // 空的 Lambda 函数，用作通信完成后的回调
      coll,  // 使用预定义的通信参数结构体
      std::unique_ptr<WorkData>(data),  // 使用工作数据对象创建独占的智能指针
      tensor.device(),  // 获取张量所在的设备
      tensors,  // 张量向量的引用
      tensors,  // 张量向量的引用
      "ucc:send");  // 用于日志记录的通信操作名称
}

c10::intrusive_ptr<Work> ProcessGroupUCC::recv(
    std::vector<at::Tensor>& tensors,  // 接收一个张量向量作为参数
    int srcRank,  // 源排名，表示发送方的排名
    int tag) {  // 标签，用于区分不同的通信操作
  check_tensor(tensors);  // 检查张量向量的有效性
  auto& tensor = tensors[0];  // 获取张量向量中的第一个张量
  initComm(tensor.device());  // 根据张量所在的设备初始化通信环境

  WorkData* data = new WorkData();  // 创建工作数据对象
  ucc_coll_args_t coll;  // 定义 UCC 收集通信参数结构体
  coll.tag = tag;  // 设置通信操作的标签
  coll.mask = UCC_COLL_ARGS_FIELD_ACTIVE_SET | UCC_COLL_ARGS_FIELD_TAG;  // 指定参数掩码
  coll.flags = 0;  // 标志位初始化为0
  coll.coll_type = UCC_COLL_TYPE_BCAST;  // 设置通信类型为广播
  coll.src.info.buffer = tensor.data_ptr();  // 设置通信源的缓冲区
  coll.src.info.count = tensor.numel();  // 设置通信源数据的元素个数
  coll.src.info.datatype = to_ucc_dType(tensor);  // 将张量数据类型转换为 UCC 支持的数据类型
  coll.src.info.mem_type = to_ucc_memType(tensor.device().type());  // 将设备类型转换为 UCC 支持的内存类型
  coll.root = srcRank;  // 设置通信的根节点排名

  coll.active_set.size = 2;  // 设置活动通信集的大小为2
  coll.active_set.start = srcRank;  // 设置活动通信集的起始排名
  coll.active_set.stride = getRank() - srcRank;  // 设置活动通信集的步长，用于指定通信的发送方排名
  SAVE_TENSORS(tensors, data->dst);  // 保存张量向量到工作数据对象的目标属性

  return collective_post(
      OpType::RECV,  // 接收操作的类型
      []() {},  // 空的 Lambda 函数，用作通信完成后的回调
      []() {},  // 空的 Lambda 函数，用作通信完成后的回调
      coll,  // 使用预定义的通信参数结构体
      std::unique_ptr<WorkData>(data),  // 使用工作数据对象创建独占的智能指针
      tensor.device(),  // 获取张量所在的设备
      tensors,  // 张量向量的引用
      tensors,  // 张量向量的引用
      "ucc:recv");  // 用于日志记录的通信操作名称
}

void ProcessGroupUCC::setSequenceNumberForGroup() {
  // 空实现，未定义函数
}

uint64_t ProcessGroupUCC::getSequenceNumberForGroup() {
  return seq_;  // 返回序列号
}

c10::intrusive_ptr<Backend> ProcessGroupUCC::createProcessGroupUCC(
    const c10::intrusive_ptr<::c10d::Store>& store,  // 共享存储指针
    int rank,  // 进程排名
    int size,  // 进程组大小
    const std::chrono::duration<float>& timeout) {  // 超时持续时间
  return c10::make_intrusive<ProcessGroupUCC>(store, rank, size, timeout);  // 创建 UCC 进程组实例
}

void ProcessGroupUCC::initComm(c10::Device dev) {
  if (!comm) {  // 如果通信对象未初始化
#ifdef USE_CUDA
    if (dev.is_cuda()) {  // 如果设备是 CUDA 设备
      c10::cuda::set_device(dev.index());  // 设置 CUDA 设备
    }
#endif
    comm = Comm::get_comm(comm_id, dev, oob, logger);  // 获取通信对象
    TORCH_UCC_LOG_INFO(TORCH_UCC_INIT, "Successfully initialized UCX library");  // 记录初始化成功的日志
    comm->ucc_create_team(team, oob);  // 创建 UCC 团队
    TORCH_UCC_LOG_INFO(TORCH_UCC_INIT, "Successfully initialized UCC library");  // 记录初始化成功的日志
    logger->setPhase(TORCH_UCC_READY);  // 设置日志记录阶段为准备完成
  } else {
    # 检查当前设备是否为 CUDA 设备
    if (dev.is_cuda()) {
        # 如果通信器已经设置了 CUDA 设备索引，并且与当前设备索引不同
        if ((comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) &&
            (comm->cuda_device_index != dev.index())) {
            # 记录错误日志，说明 UCC 通信器已使用不同的 CUDA 设备进行初始化，不支持多设备
            TORCH_UCC_LOG_ERROR(
                TORCH_UCC_INIT,
                "ucc communicator was initialized with different cuda device,"
                "multi device is not supported");
            # 抛出异常，指示不支持的错误状态
            throw std::invalid_argument(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
        }
        # 将当前设备索引设置为通信器的 CUDA 设备索引
        comm->cuda_device_index = dev.index();
    }
#ifdef USE_CUDA
  // 如果定义了 USE_CUDA 宏，则执行以下代码块

  // 创建 UCC 执行引擎，仅当设备是 CUDA 设备时才创建
  if (!cuda_ee && dev.is_cuda()) {
    // 创建一个 CUDA 流
    stream = std::make_unique<at::cuda::CUDAStream>(
        at::cuda::getStreamFromPool(true, dev.index()));
    // 设置 UCC 执行引擎的参数
    ucc_ee_params_t params;
    params.ee_type = UCC_EE_CUDA_STREAM;
    params.ee_context = (void*)stream->stream();
    params.ee_context_size = sizeof(cudaStream_t);
    // 创建 UCC 执行引擎
    TORCH_UCC_CHECK(
        ucc_ee_create(team, &params, &cuda_ee),
        "failed to create UCC execution engine");
    
    // 创建两个用于 P2P 通信的 CUDA 流
    for (int i = 0; i < 2; i++) {
      stream_p2p[i] = std::make_unique<at::cuda::CUDAStream>(
          at::cuda::getStreamFromPool(true, dev.index()));
      // 设置 UCC P2P 执行引擎的参数
      ucc_ee_params_t params;
      params.ee_type = UCC_EE_CUDA_STREAM;
      params.ee_context = (void*)stream_p2p[i]->stream();
      params.ee_context_size = sizeof(cudaStream_t);
      // 创建 UCC P2P 执行引擎
      TORCH_UCC_CHECK(
          ucc_ee_create(team, &params, &cuda_ee_p2p[i]),
          "failed to create UCC P2P execution engine");
    }
  }
#endif
}

} // namespace c10d

#endif // USE_C10D_UCC
```