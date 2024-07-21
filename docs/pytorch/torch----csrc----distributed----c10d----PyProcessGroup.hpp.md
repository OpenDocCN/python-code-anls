# `.\pytorch\torch\csrc\distributed\c10d\PyProcessGroup.hpp`

```
#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

namespace c10d {

// PyProcessGroup is a pybind11 trampoline class to allow a Python
// class to inherit from torch.distributed.ProcessGroup
class PyProcessGroup : public ProcessGroup {
 public:
  // PyWork is a pybind11 trampoline class to allow a Python
  // class to inherit from torch.distributed.Work
  class PyWork : public Work {
   public:
    // Default constructor for PyWork
    PyWork() = default;

    // Override of wait function to add Python bindings
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
      PYBIND11_OVERRIDE(
          bool, /* Return type */
          Work, /* Parent class */
          wait, /* Name of function in C++ */
          timeout); /* Timeout argument */
    }

    // Override of getFuture function to add Python bindings
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
      // Acquire Python's Global Interpreter Lock (GIL)
      pybind11::gil_scoped_acquire gil;
      // Retrieve Python override of get_future method
      auto override =
          pybind11::get_override(static_cast<const Work*>(this), "get_future");

      if (override) {
        // Call Python override to get PythonFutureWrapper object
        py::object o = override();
        auto futWrapper =
            o.cast<std::shared_ptr<torch::jit::PythonFutureWrapper>>();
        // Return the future stored in PythonFutureWrapper
        return futWrapper->fut;
      }

      // If no Python override found, fallback to base class implementation
      return Work::getFuture();
    }
  };

  // Using statement for inheriting constructors from ProcessGroup
  using ProcessGroup::ProcessGroup;

  // Override of getBackendName to add Python bindings
  const std::string getBackendName() const override {
    PYBIND11_OVERRIDE_PURE(
        std::string, /* Return type */
        ProcessGroup, /* Parent class */
        getBackendName, /* Name of function in C++ */
    );
  }

  // Override of allgather function to add Python bindings
  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        allgather, /* Name of function in C++ */
        outputTensors, /* Output tensors to gather into */
        inputTensors, /* Input tensors to gather from */
        opts); /* Options for allgather operation */
  }

  // Override of allgather_into_tensor_coalesced function to add Python bindings
  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* Return type */
        ProcessGroup, /* Parent class */
        allgather_into_tensor_coalesced, /* Name of function in C++ */
        outputTensors, /* Output tensors to coalesce into */
        inputTensors, /* Input tensors to coalesce from */
        opts); /* Options for allgather into tensor coalesced operation */
  }

  // Override of allreduce function to add Python bindings
  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* 返回类型 */
        ProcessGroup, /* 父类 */
        allreduce, /* C++ 中函数的名称 */
        tensors, /* 函数的参数：张量列表 */
        opts); /* 函数的参数：选项 */
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* 返回类型 */
        ProcessGroup, /* 父类 */
        allreduce_coalesced, /* C++ 中函数的名称 */
        tensors, /* 函数的参数：张量列表 */
        opts); /* 函数的参数：选项 */
  }

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* 返回类型 */
        ProcessGroup, /* 父类 */
        alltoall_base, /* C++ 中函数的名称 */
        outputBuffer, /* 函数的参数：输出缓冲区 */
        inputBuffer, /* 函数的参数：输入缓冲区 */
        outputSplitSizes, /* 函数的参数：输出分割大小 */
        inputSplitSizes, /* 函数的参数：输入分割大小 */
        opts); /* 函数的参数：选项 */
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* 返回类型 */
        ProcessGroup, /* 父类 */
        barrier, /* C++ 中函数的名称 */
        opts); /* 函数的参数：选项 */
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* 返回类型 */
        ProcessGroup, /* 父类 */
        broadcast, /* C++ 中函数的名称 */
        tensors, /* 函数的参数：张量列表 */
        opts); /* 函数的参数：选项 */
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* 返回类型 */
        ProcessGroup, /* 父类 */
        reduce_scatter, /* C++ 中函数的名称 */
        outputTensors, /* 函数的参数：输出张量列表 */
        inputTensors, /* 函数的参数：输入张量列表的列表 */
        opts); /* 函数的参数：选项 */
  }

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* 返回类型 */
        ProcessGroup, /* 父类 */
        reduce_scatter_tensor_coalesced, /* C++ 中函数的名称 */
        outputTensors, /* 函数的参数：输出张量列表 */
        inputTensors, /* 函数的参数：输入张量列表 */
        opts); /* 函数的参数：选项 */
  }

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* 返回类型 */
        ProcessGroup, /* 父类 */
        send, /* C++ 中函数的名称 */
        tensors, /* 函数的参数：张量列表 */
        dstRank, /* 函数的参数：目标排名 */
        tag); /* 函数的参数：标签 */
  }
    // 使用 PYBIND11_OVERRIDE 宏来覆盖基类的虚函数 send，实现多态性
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* 返回类型 */
        ProcessGroup,             /* 父类 */
        send,                     /* C++ 中函数的名称 */
        tensors,                  /* 函数参数 tensors */
        dstRank,                  /* 函数参数 dstRank */
        tag);                     /* 函数参数 tag */
  }

  // 实现 ProcessGroup 类中的虚函数 recv，接收来自指定 srcRank 的消息
  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors, /* 接收消息的张量 */
      int srcRank,                      /* 消息发送者的排名 */
      int tag) override {               /* 消息标签 */
    // 使用 PYBIND11_OVERRIDE 宏来覆盖基类的虚函数 recv，实现多态性
    PYBIND11_OVERRIDE(
        c10::intrusive_ptr<Work>, /* 返回类型 */
        ProcessGroup,             /* 父类 */
        recv,                     /* C++ 中函数的名称 */
        tensors,                  /* 函数参数 tensors */
        srcRank,                  /* 函数参数 srcRank */
        tag);                     /* 函数参数 tag */
  }


这些注释提供了对每个代码行的详细解释，包括了宏的使用目的和每个参数的含义。
};

// 定义了一个名为 PythonOnCompletionHook 的类，用于包装 Python 对象，并在析构函数中获取 Python GIL（全局解释器锁）
class TORCH_PYTHON_API PythonOnCompletionHook {
 public:
  // 构造函数，接受一个 py::object 对象作为参数，并将其移动到成员变量 hook_ 中
  PythonOnCompletionHook(py::object hook) : hook_(std::move(hook)) {}

  // 析构函数，在对象销毁时自动调用，用于释放相关资源
  ~PythonOnCompletionHook() {
    // 获取 Python GIL
    py::gil_scoped_acquire ag;
    // 减少 hook_ 对象的引用计数
    hook_.dec_ref();
    // 显式将 hook_ 设置为 nullptr，防止 py::object 的析构函数重复减少 PyObject 的引用计数
    // 参见 python_ivalue.h 中的注释 [Destructing py::object]
    hook_.ptr() = nullptr;
  }

  // 函数调用运算符的重载，接受一个 std::shared_ptr<WorkInfo> 参数
  void operator()(const std::shared_ptr<WorkInfo>& workInfo) const {
    // 异常指针，用于捕获异常
    std::exception_ptr eptr;
    {
      // 获取 Python GIL
      py::gil_scoped_acquire acquire;
      try {
        // 调用 hook_ 对象，传入 workInfo 参数
        hook_(workInfo);
      } catch (py::error_already_set& e) {
        // 捕获 py::error_already_set 异常，需要特殊处理以确保在没有 GIL 时不会析构异常对象
        eptr = std::make_exception_ptr(std::runtime_error(e.what()));
        e.restore();  // 恢复异常状态
        PyErr_Clear();  // 清除当前异常状态
      } catch (std::exception& e) {
        // 捕获其他 C++ 异常
        eptr = std::current_exception();
      }
    }
    // 到达此处表示已经没有 Python 相关的操作，可以抛出异常供后续处理
    if (eptr)
      std::rethrow_exception(eptr);
  }

 private:
  py::object hook_;  // 成员变量，保存传入的 Python 对象
};

// 命名空间 c10d 结束
} // namespace c10d
```