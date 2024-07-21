# `.\pytorch\torch\csrc\lazy\python\init.cpp`

```
#include <torch/csrc/lazy/python/init.h> // 引入 Torch 懒执行模块的 Python 初始化头文件

#include <ATen/FunctionalTensorWrapper.h> // 引入 ATen 的功能性张量包装头文件
#include <c10/core/Device.h> // 引入 c10 核心设备头文件
#include <torch/csrc/jit/python/pybind.h> // 引入 Torch JIT Python 绑定头文件
#include <torch/csrc/lazy/backend/backend_device.h> // 引入 Torch 懒执行后端设备头文件
#include <torch/csrc/lazy/backend/backend_interface.h> // 引入 Torch 懒执行后端接口头文件
#include <torch/csrc/lazy/core/config.h> // 引入 Torch 懒执行核心配置头文件
#include <torch/csrc/lazy/core/debug_util.h> // 引入 Torch 懒执行核心调试工具头文件
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h> // 引入 Torch 懒执行核心内部操作头文件
#include <torch/csrc/lazy/core/ir_dump_util.h> // 引入 Torch 懒执行核心 IR dump 工具头文件
#include <torch/csrc/lazy/core/lazy_graph_executor.h> // 引入 Torch 懒执行核心懒执行图执行器头文件
#include <torch/csrc/lazy/core/metrics.h> // 引入 Torch 懒执行核心度量头文件
#include <torch/csrc/lazy/core/trie.h> // 引入 Torch 懒执行核心 Trie 树头文件
#include <torch/csrc/lazy/python/python_util.h> // 引入 Torch 懒执行 Python 工具头文件
#if !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h> // 引入 Torch 懒执行 TS 后端实现头文件
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h> // 引入 Torch 懒执行 TS 降阶上下文头文件
#endif // FBCODE_CAFFE2 || OVRSOURCE
#include <string> // 引入字符串标准库头文件
#include <vector> // 引入向量标准库头文件

namespace torch {
namespace lazy {

// TODO(whc) backend 'device' related APIs are not very clear, this code could
// be simplified but it should probably be done together with
// designing/refactoring the overall approach to get/set of default eager/lazy
// device types

// 根据设备字符串获取后端设备对象或当前设备对象
torch::lazy::BackendDevice GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    // 获取默认设备类型
    getBackend()->GetDefaultDeviceType();
    return torch::lazy::BackendDevice(); // 返回空的后端设备对象
  }
  return torch::lazy::atenDeviceToBackendDevice(c10::Device(device_str)); // 将 ATen 设备转换为后端设备对象
}

// 获取张量的唯一 ID
std::ptrdiff_t GetTensorId(const at::Tensor& tensor) {
  // 尝试获取懒执行张量指针
  torch::lazy::LazyTensorPtr lazy_tensor = torch::lazy::TryGetLtcTensor(tensor);
  return lazy_tensor->GetUniqueId(); // 返回懒执行张量的唯一 ID
}

// 获取张量的 IR dump
std::string GetTensorsDump(
    const std::vector<at::Tensor>& tensors,
    const std::function<std::string(c10::ArrayRef<const torch::lazy::Node*>)>&
        coverter) {
  std::vector<const torch::lazy::Node*> nodes; // 存储懒执行节点指针的向量
  std::vector<torch::lazy::Value> values; // 存储懒执行值的向量
  for (auto& tensor : tensors) {
    auto inner = at::functionalization::impl::from_functional_tensor(tensor); // 从功能性张量获取内部张量
    torch::lazy::LazyTensorPtr lazy_tensor =
        torch::lazy::TryGetLtcTensor(inner); // 尝试获取懒执行张量指针
    values.push_back(lazy_tensor->GetIrValue()); // 获取懒执行值并存储到 values 向量中
    nodes.push_back(values.back().node.get()); // 将懒执行节点指针存储到 nodes 向量中
  }
  return coverter(nodes); // 使用转换器函数处理 nodes 向量并返回结果
}

// 获取所有或部分懒执行张量指针
std::vector<torch::lazy::LazyTensorPtr> GetLtcTensors(
    const std::vector<at::Tensor>& tensors,
    bool want_all) {
  std::vector<torch::lazy::LazyTensorPtr> lazy_tensors; // 存储懒执行张量指针的向量
  lazy_tensors.reserve(tensors.size()); // 预留空间以存储与张量数目相同的懒执行张量指针
  if (want_all) {
    for (auto& tensor : tensors) {
      lazy_tensors.push_back(torch::lazy::TryGetLtcTensor(tensor)); // 获取所有张量的懒执行张量指针并存储到 lazy_tensors 向量中
    }
  } else {
    for (auto& tensor : tensors) {
      auto lazy_tensor = torch::lazy::TryGetLtcTensor(tensor); // 尝试获取懒执行张量指针
      if (lazy_tensor) {
        lazy_tensors.push_back(lazy_tensor); // 如果获取成功，则存储到 lazy_tensors 向量中
      }
    }
  }
  return lazy_tensors; // 返回存储懒执行张量指针的向量
}

} // namespace lazy
} // namespace torch
// 获取张量的后端计算图形式
std::string GetTensorsBackendGraph(const std::vector<at::Tensor>& tensors) {
  // 获取懒惰张量的指针列表，仅获取需要的张量
  std::vector<torch::lazy::LazyTensorPtr> lazy_tensors =
      GetLtcTensors(tensors, /*want_all=*/false);
  // 调用懒惰图执行器的方法，导出后端计算图形式
  return torch::lazy::LazyGraphExecutor::Get()->DumpBackendComputation(
      lazy_tensors);
}

// 同步张量
void SyncTensors(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices,
    bool wait,
    bool sync_ltc_data) {
  // 获取懒惰张量的指针列表，仅获取需要的张量
  std::vector<torch::lazy::LazyTensorPtr> lazy_tensors =
      GetLtcTensors(tensors, /*want_all=*/false);
  // 调用懒惰图执行器的方法，同步张量图
  torch::lazy::LazyGraphExecutor::Get()->SyncTensorsGraph(
      &lazy_tensors, devices, wait, sync_ltc_data);
}

// 初始化延迟绑定
void initLazyBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto lazy = m.def_submodule("_lazy");
  auto lazy_ts_backend = m.def_submodule("_lazy_ts_backend");

  lazy.def(
      "_mark_step",
      // TODO(whc) this API should probably change from vector<string> to
      // vector<c10::device> but in a separate PR
      // 标记步骤函数，应考虑从 vector<string> 改为 vector<c10::device> 的变更
      [](const std::string& device_str,
         const std::vector<std::string>& devices,
         bool wait) {
        pybind11::gil_scoped_release no_gil;
        // 获取设备或当前设备的后端
        auto backend_device = GetDeviceOrCurrent(device_str);
        // 调用懒惰图执行器的方法，同步活跃张量图
        torch::lazy::LazyGraphExecutor::Get()->SyncLiveTensorsGraph(
            &backend_device, devices, wait);
        // 调用懒惰图执行器的方法，标记步骤
        torch::lazy::LazyGraphExecutor::Get()->MarkStep(backend_device);
      },
      py::arg("device") = "",
      py::arg("devices"),
      py::arg("wait") = true);

  lazy.def(
      "_wait_device_ops",
      // 等待设备操作函数
      [](const std::vector<std::string>& devices) {
        pybind11::gil_scoped_release no_gil;
        // TODO: Add support of non-empty devices.
        // TODO: 添加对非空设备的支持
        if (!devices.empty()) {
          LOG(ERROR) << "Non-empty devices are not supported.";
        }
        // 调用懒惰图执行器的方法，等待设备操作
        torch::lazy::LazyGraphExecutor::Get()->WaitDeviceOps({});
      },
      py::arg("devices"));

  lazy.def("_reset_metrics", []() {
    // 重置指标计数器和度量
    torch::lazy::MetricsArena::Get()->ResetCounters();
    torch::lazy::MetricsArena::Get()->ResetMetrics();
  });

  lazy.def("_counter_names", []() {
    // 返回计数器的名称
    return torch::lazy::GetCounterNames();
  });

  lazy.def(
      "_metrics_report", []() {
        // 创建度量报告
        return torch::lazy::CreateMetricReport();
      });

  lazy.def("_counter_value", [](const std::string& name) -> py::object {
    // 获取计数器的值
    torch::lazy::CounterData* data = torch::lazy::GetCounter(name);
    return data != nullptr ? py::cast<int64_t>(data->Value()) : py::none();
  });

  lazy.def("_get_tensor_id", [](const at::Tensor& tensor) {
    // 获取张量的ID
    // TODO: Complete implementation
  });
}
  return GetTensorId(tensor);
});

lazy.def(
    "_get_tensors_text",
    [](const std::vector<at::Tensor>& tensors) -> std::string {
      // 定义一个转换器函数，将节点数组转换为文本表示
      auto coverter = [](c10::ArrayRef<const torch::lazy::Node*> nodes) {
        return torch::lazy::DumpUtil::ToText(nodes);
      };
      // 调用 GetTensorsDump 函数，使用定义的 coverter 函数将张量转储为文本形式
      return GetTensorsDump(tensors, coverter);
    });

lazy.def(
    "_get_tensors_dot",
    [](const std::vector<at::Tensor>& tensors) -> std::string {
      // 定义一个转换器函数，将节点数组转换为 DOT 图形表示
      auto coverter = [](c10::ArrayRef<const torch::lazy::Node*> nodes) {
        return torch::lazy::DumpUtil::ToDot(nodes);
      };
      // 调用 GetTensorsDump 函数，使用定义的 coverter 函数将张量转储为 DOT 图形形式
      return GetTensorsDump(tensors, coverter);
    });

lazy.def(
    "_get_tensors_backend",
    [](const std::vector<at::Tensor>& tensors) -> std::string {
      // 调用 GetTensorsBackendGraph 函数，返回张量的后端图形表示
      return GetTensorsBackendGraph(tensors);
    });

lazy.def("_get_graph_hash", [](const std::vector<at::Tensor>& tensors) {
  // 创建一个空的 LazyTensorPtr 数组来存储转换后的张量
  std::vector<LazyTensorPtr> xtensors;
  xtensors.reserve(tensors.size());
  // 遍历给定的张量数组，尝试获取每个张量的 LTC 张量并添加到 xtensors 中
  for (auto& tensor : tensors) {
    xtensors.emplace_back(TryGetLtcTensor(tensor));
  }
  // 调用 LazyGraphExecutor 的 GetGraphHash 方法获取图形的哈希值
  auto hash = LazyGraphExecutor::Get()->GetGraphHash(xtensors);
  // 将哈希值转换为二进制字符串
  std::string bin((const char*)&hash, sizeof(hash));
  // 返回二进制字符串作为 Python 字节对象
  return py::bytes(bin);
});

lazy.def(
    "_sync_multi",
    [](const std::vector<at::Tensor>& tensors,
       const std::vector<std::string>& devices,
       bool wait,
       bool sync_ltc_data) {
      // 释放全局解释器锁，以便在 C++ 函数执行期间不阻止 Python 线程
      pybind11::gil_scoped_release no_gil;
      // 调用 SyncTensors 函数，同步多个张量到指定设备
      SyncTensors(tensors, devices, wait, sync_ltc_data);
    },
    py::arg("tensors"),  // 参数：张量数组
    py::arg("devices"),  // 参数：设备数组
    py::arg("wait") = true,         // 参数：是否等待，默认为 true
    py::arg("sync_ltc_data") = true  // 参数：是否同步 LTC 数据，默认为 true
);

lazy.def("_get_force_fallback", []() {
  // 返回当前的 LTC 强制回退设置
  return torch::lazy::getLTCForceFallback();
});

lazy.def("_set_force_fallback", [](std::string newval) {
  // 设置 LTC 强制回退选项的新值
  torch::lazy::getLTCForceFallback() = newval;
});

lazy.def("_clear_ir_cache", []() {
  // 清空 IR 缓存
  TrieCache::Get()->Clear();
});

lazy.def("_dump_ir_cache", [](std::string filename) {
  // 将 IR 缓存转储到 DOT 文件
  TrieCache::Get()->DumpToDotFile(filename);
});

lazy.def("_set_reuse_ir", [](bool val) {
  // 设置是否重用 IR 的标志
  FLAGS_torch_lazy_reuse_ir = val;
});

lazy.def("_set_symbolic_shape_mode", [](bool val) {
  // 设置是否启用符号形状模式的标志
  FLAGS_ltc_enable_symbolic_shapes = val;
});

lazy.def("_get_symbolic_shape_mode", []() {
  // 获取当前是否启用符号形状模式的标志
  return FLAGS_ltc_enable_symbolic_shapes;
});

lazy.def("_get_default_device_type", []() {
  // 获取默认设备类型并转换为字符串返回
  return getBackend()->GetDefaultDeviceType()->toString();
});

lazy_ts_backend.def("_init", []() {
#if !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
    // 如果不是在 FBCODE_CAFFE2 或 OVRSOURCE 构建中，则初始化 TorchScript 后端
    torch::lazy::InitTorchScriptBackend();
#else
    // 如果是在 FBCODE 或 OVRSOURCE 构建中，则抛出错误，不支持 TorchScript 后端
    TORCH_CHECK(false, "TorchScript backend not yet supported in FBCODE/OVRSOURCE builds");
#endif // !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
  });

/*
 * 返回 DeviceData 节点的张量 ID 和张量列表。
 * TODO(shunting) 重新审视这个 API 是否适合 XLA
 */
lazy_ts_backend.def(
    "_get_tensors_ts_device_data_node",
    [](const std::vector<at::Tensor>& tensors)
        -> std::pair<std::vector<int64_t>, std::vector<at::IValue>> {
#if !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
      // 获取 tensors 中每个 Tensor 对应的 LTC Tensor，并构建根节点列表
      std::vector<const Node*> roots;
      for (auto& tensor : tensors) {
        auto xtensor = TryGetLtcTensor(tensor);
        roots.push_back(xtensor->GetIrValue().node.get());
      }
      // 计算根节点的后序遍历顺序
      auto post_order = Util::ComputePostOrder(roots);
      std::vector<int64_t> tensor_ids;
      std::vector<at::IValue> ivalues;

      // 用于存储已经处理过的 BackendData 的 handle 集合，用于去重 DeviceData
      std::unordered_set<BackendData::Handle> data_handles_;
      for (auto nodeptr : post_order) {
        // 如果节点操作符是 ltc_device_data
        if (nodeptr->op() == *torch::lazy::ltc_device_data) {
          // 从节点中获取后端数据
          const auto backend_data = getBackend()->GetComputationDataFromNode(nodeptr);

          auto infoptr = backend_data->info();
          auto deviceDataInfoPtr =
              (torch::lazy::LazyGraphExecutor::DeviceDataInfo*)infoptr;
          auto* tsDataPtr = (torch::lazy::TSData*)backend_data.get();

          // 根据 handle 去重 DeviceData
          auto handle = tsDataPtr->GetHandle();
          if (!data_handles_.insert(handle).second) {
            continue;
          }
          tensor_ids.push_back(deviceDataInfoPtr->tensor_id);

          /*
           * 如果 TSData 包含张量，则张量 ID 将唯一标识该张量。
           * 我们使用该张量 ID 在其他地方查找张量，比如在 Python 前向方法参数中。
           *
           * 如果 TSData 包含标量，张量 ID 本身并不重要。我们在后续调用中重用标量值。
           */
          if (tsDataPtr->HasValue()) {
            ivalues.emplace_back(tsDataPtr->data());
          } else {
            TORCH_CHECK(tsDataPtr->scalar.has_value());
            ivalues.emplace_back(tsDataPtr->scalar.value());
          }
        }
      }
      return std::make_pair(tensor_ids, ivalues);
#else
      // 如果是在 FBCODE 构建中，则抛出错误，不支持 TorchScript 后端
      TORCH_CHECK(
          false, "TorchScript backend not yet supported in FBCODE builds");
      return std::make_pair(
          std::vector<int64_t>(), std::vector<at::IValue>());
#endif // !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
    });

// TODO(shunting) 重新审视这部分是否适合 XLA
lazy_ts_backend.def(
    "_run_cached_graph",
    [](const std::string& hash_str,
       const std::vector<at::IValue>& graph_inputs) {
#if !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
        // 检查哈希字符串的长度是否等于哈希类型的大小
        TORCH_CHECK(hash_str.size() == sizeof(hash_t));
        // 将哈希字符串转换为哈希类型并获取对应的计算结果
        hash_t hash = *(hash_t*)(hash_str.c_str());
        auto cachedComputation =
            LazyGraphExecutor::Get()->GetComputationCache()->Get(hash);
        // 检查是否成功获取到计算结果
        TORCH_CHECK(
            cachedComputation,
            "Failed to get computation by hash. Maybe the entry get kicked out of the LRU cache"); // TODO implement a fallback mechanism, or make sure those entries never get kicked out
        // 获取计算对象指针
        auto computationPtr =
            (torch::lazy::TSComputation*)cachedComputation->computation.get();

        // 准备输入数据栈
        std::vector<torch::jit::IValue> stack;
        stack.reserve(graph_inputs.size());
        // 将图输入数据添加到栈中
        for (const auto& arg : graph_inputs) {
          stack.emplace_back(arg);
        }
        // 运行计算图
        computationPtr->graph_executor().run(stack);
        // 准备保存运行结果的容器
        result.reserve(stack.size());
        // 将栈中的每个元素转换为张量并保存到结果中
        for (torch::jit::IValue elem : stack) {
          result.push_back(elem.toTensor());
        }
#else
        // 如果未定义所需宏，抛出错误，指出在 FBCODE 构建中不支持 TorchScript 后端
        TORCH_CHECK(
            false, "TorchScript backend not yet supported in FBCODE builds");
#endif // !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
        // 返回运行结果
        return result;
      });
  lazy_ts_backend.def("_get_latest_computation_graph", []() {
#if !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
    // 获取最新的计算图并返回其字符串表示
    auto computation = LazyGraphExecutor::Get()
                           ->GetComputationCache()
                           ->GetLatest()
                           ->computation;
    // 确保获取到的是 TSComputation 类型的计算对象
    auto ts_computation = dynamic_cast<TSComputation*>(computation.get());
    TORCH_CHECK(ts_computation, "Found non-TSComputation in cache");
    return ts_computation->graph()->toString();
#else
    // 如果未定义所需宏，抛出错误，指出在 FBCODE 构建中不支持 TorchScript 后端
    TORCH_CHECK(
        false, "TorchScript backend not yet supported in FBCODE builds");
    return "";
#endif // !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
  });

  // GetPythonFramesFunction() has not ever worked with torchdeploy/multipy
  // possibly becuase GetPythonFrames resolves to external cpython rather
  // than embedded cpython. So far this problem has only been observed
  // internally, so we will just block it off there.

#if !(defined(USE_DEPLOY))

  // When libtorch_python is loaded, we register the python frame getter
  // otherwise, debug util simply omits python frames
  // 当加载 libtorch_python 时，注册 Python 帧获取器；否则，调试工具将省略 Python 帧
  GetPythonFramesFunction() = GetPythonFrames;

#endif // USE_DEPLOY
}

} // namespace lazy
} // namespace torch
```