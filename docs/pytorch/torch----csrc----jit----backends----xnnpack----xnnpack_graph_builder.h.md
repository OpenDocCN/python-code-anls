# `.\pytorch\torch\csrc\jit\backends\xnnpack\xnnpack_graph_builder.h`

```
// 声明命名空间和类 XNNGraph，这是一个用于处理 XNN 子图的类
namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

// XNNGraph 类的定义
class XNNGraph {
 private:
  // 输出的最小和最大值
  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  // 序列化器对象
  XNNSerializer _serializer;
  // XNN 子图指针
  xnn_subgraph_t _subgraph_ptr;
  // 包含所有中间张量值的无序集合
  std::unordered_set<torch::jit::Value*> _intermediate_tensors;
  // 将张量值映射到 xnnpack id 的映射表
  std::unordered_map<torch::jit::Value*, uint32_t> _val_to_ids;
  // 包含 Torch 输入和输出的向量，必须有序以保留输入/输出的顺序
  std::vector<torch::jit::Value*> _inputs;
  std::vector<torch::jit::Value*> _outputs;

  // 优化和跟踪 TorchScript 图的图传递
  // 主要是为了将图优化为 XNN 子图可以处理的格式
  std::shared_ptr<torch::jit::Graph> optimizeAndTraceGraph(
      std::shared_ptr<torch::jit::Graph> graph,
      std::vector<c10::IValue>& example_inputs);

  // 收集图中所有的中间张量值，跳过所有的 prim 常量
  // 主要目的是为 XNN 子图预先定义张量值
  void gatherTensorValues(std::shared_ptr<torch::jit::Graph>& graph);

  // 收集给定节点中的张量值
  void gatherNodeInputs(torch::jit::Node& node);

  // 辅助函数：确定一个 jit 值是否是图的输入
  bool isGraphInput(torch::jit::Value* val);

  // 辅助函数：确定一个 jit 值是否是图的输出
  bool isGraphOutput(torch::jit::Value* val);

  // 定义图中所有节点的 xnnpack 节点
  void defineAllNodes(std::shared_ptr<torch::jit::Graph>& graph);

  // 定义图中所有使用的 xnnpack 张量值
  void defineAllTensorValues();

  // 遍历图并检查是否有不支持的操作
  void checkOpsToDelegate(std::shared_ptr<torch::jit::Graph>& graph);

 public:
  // 构造函数：初始化 XNN 子图和序列化器，同时初始化 xnnpack 库
  XNNGraph() : _serializer(), _subgraph_ptr(nullptr) {
    // 初始化 xnnpack 库
    xnn_status status = xnn_initialize(/*allocator =*/nullptr);
    TORCH_CHECK(xnn_status_success == status, "Failed to initialize xnnpack");
  }

  // 析构函数：反初始化 xnnpack 库，同时释放 XNN 子图资源
  ~XNNGraph() {
    xnn_deinitialize();
    // 如果子图指针不为空，则删除 XNN 子图
    if (_subgraph_ptr != nullptr) {
      xnn_delete_subgraph(_subgraph_ptr);
    }
  }
  }
}

void buildXNNGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    std::vector<c10::IValue> example_inputs);


// 关闭 buildXNNGraph 函数定义，并指定它将构建一个由 torch::jit::Graph 对象指针 graph 表示的神经网络图，并接受一个 example_inputs 向量作为输入参数
void buildXNNGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    std::vector<c10::IValue> example_inputs);



void runGraphOnInputs(
    std::vector<at::Tensor> tensor_inputs,
    std::vector<at::Tensor> tensor_outputs);


// 定义 runGraphOnInputs 函数，该函数接受两个参数：tensor_inputs 作为输入张量的向量，tensor_outputs 作为输出张量的向量
void runGraphOnInputs(
    std::vector<at::Tensor> tensor_inputs,
    std::vector<at::Tensor> tensor_outputs);



std::string serializedXNNGraph();


// 声明 serializedXNNGraph 函数，它将返回一个字符串，表示序列化的神经网络图
std::string serializedXNNGraph();



std::vector<std::vector<long>> getGraphOutputShapes();


// 声明 getGraphOutputShapes 函数，该函数将返回一个二维长整型向量的向量，表示神经网络图的输出形状
std::vector<std::vector<long>> getGraphOutputShapes();
};

// 结束委托命名空间
} // namespace delegate

// 结束 xnnpack 命名空间
} // namespace xnnpack

// 结束 jit 命名空间
} // namespace jit

// 结束 torch 命名空间
} // namespace torch
```