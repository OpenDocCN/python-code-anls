# `.\pytorch\torch\csrc\cuda\python_nccl.cpp`

```py
// 引入 torch/csrc/cuda/python_nccl.h 文件

#include <torch/csrc/cuda/python_nccl.h>

// 引入 ATen 库的 functional.h 文件
#include <ATen/core/functional.h>
// 引入 pybind11 库
#include <pybind11/pybind11.h>
// 引入 torch/csrc 下的头文件
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/Types.h>
#include <torch/csrc/cuda/THCP.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/utils/pybind.h>

// 引入 c10 库的 CUDA 相关头文件
#include <c10/cuda/CUDAGuard.h>
// 引入 c10 库的 irange.h 文件
#include <c10/util/irange.h>

// 使用 at 命名空间
using namespace at;
// 使用 torch 命名空间
using namespace torch;
// 使用 torch::cuda::nccl 命名空间
using namespace torch::cuda::nccl;
// 使用 torch::cuda::nccl::detail 命名空间
using namespace torch::cuda::nccl::detail;

// 定义静态常量 COMM_CAPSULE_NAME
static const char* COMM_CAPSULE_NAME = "torch.cuda.nccl.Communicator";

// 定义函数 THCPModule_nccl_version，返回 nccl 的版本号作为 Python 的长整型对象
PyObject* THCPModule_nccl_version(PyObject* self, PyObject* args) {
  return PyLong_FromUnsignedLongLong(version());
}

// 定义函数 THCPModule_nccl_version_suffix，返回 nccl 的版本后缀作为 Python 的字节对象
PyObject* THCPModule_nccl_version_suffix(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  return PyBytes_FromString(version_suffix());
  END_HANDLE_TH_ERRORS
}

// 定义函数 THCPModule_nccl_unique_id，获取 nccl 唯一标识符并返回作为 Python 的字节对象
PyObject* THCPModule_nccl_unique_id(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  ncclUniqueId id;
  get_unique_id(id);
  return PyBytes_FromStringAndSize((char*)&id, NCCL_UNIQUE_ID_BYTES);
  END_HANDLE_TH_ERRORS
}

// 定义静态函数 unpack_nccl_comm，从 Python 的胶囊对象中提取 nccl 通信句柄并返回
static ncclComm_t unpack_nccl_comm(PyObject* capsule) {
  ncclComm_t comm =
      (ncclComm_t)PyCapsule_GetPointer(capsule, COMM_CAPSULE_NAME);
  if (!comm)
    throw python_error();
  return comm;
}

// 定义静态函数 destroy_nccl_comm，销毁 nccl 通信句柄对应的 Python 胶囊对象
static void destroy_nccl_comm(PyObject* capsule) {
  HANDLE_TH_ERRORS
  ncclComm_t comm = unpack_nccl_comm(capsule);
  {
    // 释放全局解释器锁并销毁通信句柄
    pybind11::gil_scoped_release no_gil;
    comm_destroy(comm);
  }
  END_HANDLE_TH_ERRORS_RET()
}

// 定义静态函数 unpack_streams，从 Python 对象中解析 CUDA 流列表并返回为可选的 CUDA 流向量
static std::vector<std::optional<at::cuda::CUDAStream>> unpack_streams(
    PyObject* obj,
    size_t size) {
  // 如果对象为 Py_None，则返回大小为 size 的空 CUDA 流向量
  if (obj == Py_None) {
    return std::vector<std::optional<at::cuda::CUDAStream>>(size, c10::nullopt);
  }
  // 否则将对象解析为 CUDA 流列表并检查其大小是否与给定 size 相等
  auto streams = THPUtils_PySequence_to_CUDAStreamList(obj);
  if (streams.size() != size) {
    throw std::runtime_error(
        "number of streams is not equal to number of inputs");
  }
  return streams;
}

// 内联函数声明，从 Python 对象中提取 Tensor
static inline at::Tensor extract_tensor(PyObject* obj);
// 内联函数声明，从 Python 对象中提取 Tensor 向量
static inline std::vector<at::Tensor> extract_tensors(PyObject* obj);

// 定义静态函数 unpack_comms，从 Python 对象中解析 nccl 通信句柄列表并返回
static std::vector<ncclComm_t> unpack_comms(PyObject* obj, size_t size) {
  // 如果对象为 Py_None，则返回空的 nccl 通信句柄向量
  if (obj == Py_None) {
    return std::vector<ncclComm_t>();
  }
  std::vector<ncclComm_t> comms;
  // 如果对象为 Python 胶囊对象，则直接提取 nccl 通信句柄
  if (PyCapsule_CheckExact(obj)) {
    comms = {unpack_nccl_comm(obj)};
  } else {
    // 否则将对象解析为序列对象并逐一提取其中的 nccl 通信句柄
    auto seq = THPObjectPtr(PySequence_Fast(obj, "comm is not a sequence"));
    if (!seq)
      throw python_error();
    auto size = PySequence_Fast_GET_SIZE(seq.get());
    comms = std::vector<ncclComm_t>(size);
    for (const auto i : c10::irange(size)) {
      comms[i] = unpack_nccl_comm(PySequence_Fast_GET_ITEM(seq.get(), i));
    }
  }
  // 检查提取到的通信句柄数量是否与指定的 size 相等
  if (comms.size() != size) {
    throw std::runtime_error(
        "number of communicators is not equal to number of inputs");
  }
  return comms;
}
PyObject* THCPModule_nccl_init_rank(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 初始化变量
  int nranks = 0;  // 接收的总进程数
  const char* id = nullptr;  // NCCL 唯一标识符的指针
  Py_ssize_t id_len = 0;  // NCCL 唯一标识符的长度
  int rank = 0;  // 当前进程的排名

  // 解析输入参数
  if (!PyArg_ParseTuple(
          args, "is#i:nccl_init_rank", &nranks, &id, &id_len, &rank)) {
    return nullptr;
  }
  
  // 检查 NCCL 唯一标识符长度是否正确
  TORCH_CHECK(
      id_len == NCCL_UNIQUE_ID_BYTES,
      "invalid unqiue_id (expected ",
      NCCL_UNIQUE_ID_BYTES,
      " bytes, got ",
      id_len,
      ")");

  // 创建 NCCL 通信标识符对象
  ncclUniqueId commId;
  memcpy(&commId, id, NCCL_UNIQUE_ID_BYTES);
  ncclComm_t comm = nullptr;

  // 使用 GIL 释放，调用底层函数初始化 NCCL 通信
  {
    pybind11::gil_scoped_release no_gil;
    comm = comm_init_rank(nranks, commId, rank);
  }

  // 返回 Python 封装后的通信对象
  return PyCapsule_New(comm, COMM_CAPSULE_NAME, &destroy_nccl_comm);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_reduce(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 初始化变量
  PyObject *_inputs = nullptr, *_output = nullptr, *_streams = nullptr,
           *_comms = nullptr;
  int root = 0, op = 0;

  // 解析输入参数
  if (!PyArg_ParseTuple(
          args, "OOiiOO", &_inputs, &_output, &root, &op, &_streams, &_comms)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_reduce",
        1,
        "(sequence[Tensor] inputs, Tensor output, int root,"
        " int op, sequence[torch.cuda.Stream or None]");
    return nullptr;
  }

  // 提取输入参数中的张量
  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  auto output = extract_tensor(_output);
  std::vector<std::optional<at::cuda::CUDAStream>> streams =
      unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  // 使用 GIL 释放，调用底层函数进行 NCCL reduce 操作
  {
    pybind11::gil_scoped_release no_gil;
    torch::cuda::nccl::reduce(inputs, output, root, op, streams, user_comms);
  }

  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_all_reduce(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  // 初始化变量
  PyObject *_inputs = nullptr, *_outputs = nullptr, *_streams = nullptr,
           *_comms = nullptr;
  int op = 0;

  // 解析输入参数
  if (!PyArg_ParseTuple(
          args, "OOiOO", &_inputs, &_outputs, &op, &_streams, &_comms)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_all_reduce",
        1,
        "(sequence[Tensor] inputs, sequence[Tensor] outputs, int op,"
        " sequence[torch.cuda.Stream] streams,"
        " sequence[torch.cuda.nccl.Communicator] comms)");
    return nullptr;
  }

  // 提取输入参数中的张量和流对象
  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  std::vector<at::Tensor> outputs = extract_tensors(_outputs);
  auto streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  // 使用 GIL 释放，调用底层函数进行 NCCL all_reduce 操作
  {
    pybind11::gil_scoped_release no_gil;
    all_reduce(inputs, outputs, op, streams, user_comms);
  }

  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
PyObject* THCPModule_nccl_broadcast(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs = nullptr, *_streams = nullptr, *_comms = nullptr;
  int root = 0;

  // 解析传入的参数，期望参数格式为 (sequence[Tensor] inputs, int root, sequence[torch.cuda.Stream] streams, sequence[torch.cuda.nccl.Communicator] comms)
  if (!PyArg_ParseTuple(args, "OiOO", &_inputs, &root, &_streams, &_comms)) {
    // 参数解析失败时，抛出异常并返回空指针
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_broadcast",
        1,
        "(sequence[Tensor] inputs, int root"
        " sequence[torch.cuda.Stream] streams,"
        " sequence[torch.cuda.nccl.Communicator] comms)");
    return nullptr;
  }

  // 从输入的 PyObject 中提取 Tensor 列表
  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  // 检查指定的 root 是否在有效范围内
  TORCH_CHECK(root >= 0 && (size_t)root < inputs.size(), "invalid root");
  // 解析 streams 参数，获取对应数量的 CUDA 流列表
  auto streams = unpack_streams(_streams, inputs.size());
  // 解析 comms 参数，获取对应数量的 NCCL 通信器列表
  auto user_comms = unpack_comms(_comms, inputs.size());

  {
    // 释放 GIL，允许在 Python 解释器之外执行 CUDA NCCL 操作
    pybind11::gil_scoped_release no_gil;
    // 调用 CUDA NCCL 广播函数，广播输入张量 inputs 到所有节点
    torch::cuda::nccl::broadcast(inputs, streams, user_comms);
  }

  // 返回 Python 中的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_all_gather(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs = nullptr, *_outputs = nullptr, *_streams = nullptr,
           *_comms = nullptr;

  // 解析传入的参数，期望参数格式为 (sequence[Tensor] inputs, sequence[Tensor] outputs, sequence[torch.cuda.Stream] streams, sequence[torch.cuda.nccl.Communicator] comms)
  if (!PyArg_ParseTuple(
          args, "OOOO", &_inputs, &_outputs, &_streams, &_comms)) {
    // 参数解析失败时，抛出异常并返回空指针
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_all_gather",
        1,
        "(sequence[Tensor] inputs, sequence[Tensor] outputs"
        " sequence[torch.cuda.Stream] streams,"
        " sequence[torch.cuda.nccl.Communicator] comms)");
    return nullptr;
  }

  // 从输入的 PyObject 中提取输入和输出 Tensor 列表
  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  std::vector<at::Tensor> outputs = extract_tensors(_outputs);
  // 解析 streams 参数，获取对应数量的 CUDA 流列表
  auto streams = unpack_streams(_streams, inputs.size());
  // 解析 comms 参数，获取对应数量的 NCCL 通信器列表
  auto user_comms = unpack_comms(_comms, inputs.size());

  {
    // 释放 GIL，允许在 Python 解释器之外执行 CUDA NCCL 操作
    pybind11::gil_scoped_release no_gil;
    // 调用 CUDA NCCL 全局收集函数，将输入张量 inputs 收集到输出张量 outputs 中
    all_gather(inputs, outputs, streams, user_comms);
  }

  // 返回 Python 中的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_reduce_scatter(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs = nullptr, *_outputs = nullptr, *_streams = nullptr,
           *_comms = nullptr;
  int op = 0;

  // 解析传入的参数，期望参数格式为 (sequence[Tensor] inputs, sequence[Tensor] outputs, int op, sequence[torch.cuda.Stream] streams, sequence[torch.cuda.nccl.Communicator] comms)
  if (!PyArg_ParseTuple(
          args, "OOiOO", &_inputs, &_outputs, &op, &_streams, &_comms)) {
    // 参数解析失败时，抛出异常并返回空指针
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_reduce_scatter",
        1,
        "(sequence[Tensor] inputs, sequence[Tensor] outputs, int op"
        " sequence[torch.cuda.Stream] streams,"
        " sequence[torch.cuda.nccl.Communicator] comms)");
    return nullptr;
  }

  // 从输入的 PyObject 中提取输入和输出 Tensor 列表
  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  std::vector<at::Tensor> outputs = extract_tensors(_outputs);
  // 解析 streams 参数，获取对应数量的 CUDA 流列表
  auto streams = unpack_streams(_streams, inputs.size());
  // 解析 comms 参数，获取对应数量的 NCCL 通信器列表
  auto user_comms = unpack_comms(_comms, inputs.size());

  {
    // 释放 GIL，允许在 Python 解释器之外执行 CUDA NCCL 操作
    pybind11::gil_scoped_release no_gil;
    // 调用 CUDA NCCL 按元素减少散步函数，将输入张量 inputs 按照操作 op 进行减少散步到输出张量 outputs 中
    reduce_scatter(inputs, outputs, op, streams, user_comms);
  }

  // 返回 Python 中的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
// 从给定的 Python 对象中提取一个 Tensor 对象
static inline at::Tensor extract_tensor(PyObject* obj) {
  // 检查传入的对象是否为 Tensor 类型，否则抛出错误，显示实际类型
  TORCH_CHECK_TYPE(
      THPVariable_Check(obj),
      "expected Tensor (got ",
      Py_TYPE(obj)->tp_name,
      ")");
  // 将 PyObject 转换为 Tensor 对象并返回
  return THPVariable_Unpack(obj);
}

// 从给定的 Python 对象中提取多个 Tensor 对象并存储在 vector 中返回
static inline std::vector<at::Tensor> extract_tensors(PyObject* obj) {
  // 将 Python 对象转换为快速序列对象，如果失败则抛出异常
  auto seq = THPObjectPtr(PySequence_Fast(obj, "expected a sequence"));
  if (!seq)
    throw python_error();

  // 获取序列的长度
  const Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
  // 准备一个存储 Tensor 的 vector，预留足够的空间以提升性能
  std::vector<at::Tensor> list;
  if (length >= 0) {
    list.reserve(length);
  }
  // 遍历序列中的每个元素
  for (Py_ssize_t i = 0; i < length; i++) {
    // 获取序列中的第 i 个元素
    PyObject* item = PySequence_Fast_GET_ITEM(seq.get(), i);
    // 检查元素是否为 Tensor 类型，否则抛出错误，显示实际类型
    TORCH_CHECK_TYPE(
        THPVariable_Check(item),
        "expected Tensor at ",
        i,
        " (got ",
        Py_TYPE(item)->tp_name,
        ")");
    // 将 PyObject 转换为 Tensor 并添加到 vector 中
    list.emplace_back(THPVariable_Unpack(item));
  }
  // 返回存储了所有 Tensor 的 vector
  return list;
}
```