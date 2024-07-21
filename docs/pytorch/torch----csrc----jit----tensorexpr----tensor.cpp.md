# `.\pytorch\torch\csrc\jit\tensorexpr\tensor.cpp`

```py
// 引入头文件，声明使用了 Torch 的 tensor expression 相关模块
#include <torch/csrc/jit/tensorexpr/tensor.h>

// 引入 C10 库中的日志记录和整数范围遍历工具
#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

// 定义 torch::jit::tensorexpr 命名空间
namespace torch::jit::tensorexpr {

// 实现 Tensor 类中的 constructStmt 方法
StmtPtr Tensor::constructStmt(
    const std::vector<VarPtr>& args,             // 参数列表，包含 VarPtr 类型的参数
    ExprPtr body,                                // 主体表达式，包含 ExprPtr 类型的表达式
    const std::vector<ExprPtr>& reduce_dims,     // 需要进行缩减的维度列表，包含 ExprPtr 类型的表达式
    const std::vector<VarPtr>& reduce_args) const {  // 缩减参数列表，包含 VarPtr 类型的参数

  // 将 args 转换为 indices，类型为 ExprPtr
  std::vector<ExprPtr> indices(args.begin(), args.end());

  // 获取 buf 的维度数 ndim
  size_t ndim = buf()->ndim();
  // 获取 reduce_dims 的维度数 reduce_ndim
  size_t reduce_ndim = reduce_dims.size();
  // 将 body 转换为 ReduceOp 类型的 reduce_op
  auto reduce_op = to<ReduceOp>(body);
  // 如果 reduce_ndim 大于 0，则获取 reduce_op 的累加缓冲区 acc_buf
  auto acc_buf = reduce_ndim > 0 ? reduce_op->getAccBuf() : nullptr;

  // 创建一个 Store 语句 s，将 body 存储到 buf_ 中的 indices 位置
  StmtPtr s = alloc<Store>(buf_, indices, body);

  // 如果 reduce_ndim 大于 0，执行以下操作
  if (reduce_ndim > 0) {
    // 断言 reduce_op 不为 nullptr
    TORCH_INTERNAL_ASSERT(reduce_op != nullptr);
    // 如果 acc_buf 不为 nullptr
    if (acc_buf != nullptr) {
      // 获取 reduce_op 的 reducer 函数
      auto reducer = reduce_op->reducer();
      // 复制一份 args 到 output_args
      std::vector<ExprPtr> output_args(args.begin(), args.end());
      // 创建新的 reduce_op 表达式 new_reduce_op
      ExprPtr new_reduce_op = reducer(
          to<Buf>(acc_buf),
          alloc<Cast>(acc_buf->dtype(), reduce_op->getRiOperand()),
          output_args,
          reduce_args);
      // 设置 new_reduce_op 的数据类型为 acc_buf 的数据类型
      new_reduce_op->set_dtype(acc_buf->dtype());
      // 更新 s 为将 new_reduce_op 存储到 acc_buf 中的 indices 位置
      s = alloc<Store>(to<Buf>(acc_buf), indices, new_reduce_op);
    }
  }

  // 如果 ndim 和 reduce_ndim 均为 0，直接返回 s
  if (ndim == 0 && reduce_ndim == 0) {
    return s;
  }

  // 如果 reduce_ndim 大于 0，执行以下操作
  if (reduce_ndim > 0) {
    // 断言 reduce_op 不为 nullptr
    TORCH_INTERNAL_ASSERT(reduce_op != nullptr);

    // 反向遍历 reduce_dims，依次创建 For 循环语句 s
    for (const auto i : c10::irange(reduce_ndim)) {
      // 根据索引从内向外遍历 reduce_dims，将 dim_index 设置为当前索引
      size_t dim_index = reduce_ndim - i - 1;
      // 获取 reduce_dims 中的 dim
      auto const& dim = reduce_dims[dim_index];
      // 将 s 设置为在 reduce_args[dim_index] 上进行 For 循环，范围为 immLike(dim, 0) 到 dim，内部嵌套 s
      s = alloc<For>(reduce_args[dim_index], immLike(dim, 0), dim, s);
    }
    // 将 s 包装为一个 Block，以便添加前置和后置语句
    s = alloc<Block>(std::vector<StmtPtr>({s}));

    // 初始化缓冲区 init_buf 和表达式 init_expr
    BufPtr init_buf = acc_buf ? to<Buf>(acc_buf) : buf();
    ExprPtr init_expr =
        acc_buf ? to<Buf>(acc_buf)->initializer() : buf()->initializer();
    // 如果 init_expr 不为 nullptr，将其作为初始化语句存储到 s 的前面
    if (init_expr) {
      StorePtr init_stmt = alloc<Store>(init_buf, indices, init_expr);
      to<Block>(s)->prepend_stmt(init_stmt);
    }

    // 如果 acc_buf 不为 nullptr，执行以下操作
    if (acc_buf != nullptr) {
      // 加载 acc_buf 的数据到 load_acc
      LoadPtr load_acc = alloc<Load>(acc_buf, indices);
      // 将 load_acc 转换为 buf() 的数据类型，并存储到 buf() 的 indices 位置
      auto cast = alloc<Cast>(buf()->dtype(), load_acc);
      StorePtr post_stmt = alloc<Store>(buf(), indices, cast);
      // 将 post_stmt 作为后置语句添加到 s 的末尾
      to<Block>(s)->append_stmt(post_stmt);
    }
  }

  // 断言调试模式下的 buf_ 是否是连续的或者特定的内存格式
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      buf_->is_contiguous() ||
      buf_->is_contiguous(at::MemoryFormat::ChannelsLast) ||
      buf_->is_contiguous(at::MemoryFormat::ChannelsLast3d) ||
      buf_->is_channels_last_1d_contiguous());

  // 定义循环顺序的函数 loop_order_fn
  auto loop_order_fn = [&]() {
    std::vector<int32_t> loop_order;
    // 如果 buf_ 是连续的
    if (buf_->is_contiguous()) {
      // 从后向前遍历 args，依次将索引加入 loop_order
      for (int32_t i = args.size() - 1; i >= 0; i--) {
        loop_order.push_back(i);
      }
    } else if (buf_->is_contiguous(c10::MemoryFormat::ChannelsLast)) {
      // 如果 buf_ 是 ChannelsLast 内存格式，设置特定的遍历顺序
      loop_order = {1, 3, 2, 0};
    } else if (buf_->is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
      // 如果 buf_ 是 ChannelsLast3d 内存格式，设置特定的遍历顺序
      loop_order = {1, 4, 3, 2, 0};
    } else {
      // 如果以上条件均不满足，设置默认的遍历顺序
      loop_order = {1, 2, 0};
    }
    # 调用函数 `loop_order_fn()` 并将结果赋给 `loop_order` 变量，该函数返回一个表示循环顺序的集合
    auto loop_order = loop_order_fn();
    
    # 对 `loop_order` 中的每个维度索引 `dim_index` 进行迭代循环
    for (auto dim_index : loop_order) {
        # 获取缓冲区 `buf()` 中的维度 `dim_index` 的引用
        auto const& dim = buf()->dim(dim_index);
        # 使用 `alloc<For>(args[dim_index], immLike(dim, 0), dim, s)` 创建一个 `For` 循环节点，并将其附加到 `s` 上
        s = alloc<For>(args[dim_index], immLike(dim, 0), dim, s);
    }
    
    # 返回最终生成的 `s`，该变量包含了根据 `loop_order` 创建的一系列 `For` 循环节点
    return s;
}



Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    std::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func) {
  // 根据维度创建索引变量
  std::vector<VarHandle> args = create_index_vars(dims);
  // 调用用户提供的函数计算表达式体
  ExprHandle body = body_func(args);
  // 创建缓存对象，用于表示张量的存储
  BufHandle buf = Buf::make(name, dims, body.dtype(), c10::nullopt, strides);
  // 返回表示张量的对象
  return Tensor(buf, args, body);
}



Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func) {
  // 调用具有默认步长的 Compute 函数重载
  return Compute(name, dims, c10::nullopt, body_func);
}



Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    std::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const VarHandle&)>& body_func) {
  // 如果维度不等于1，抛出异常
  if (dims.size() != 1) {
    throw malformed_input("mismatch between body and arg size (1)");
  }
  // 创建索引变量
  std::vector<VarHandle> args = create_index_vars(dims);
  // 调用用户提供的函数计算表达式体
  ExprHandle body = body_func(args[0]);
  // 创建缓存对象，用于表示张量的存储
  BufHandle buf = Buf::make(name, dims, body.dtype(), c10::nullopt, strides);
  // 返回表示张量的对象
  return Tensor(buf, args, body);
}



Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(const VarHandle&)>& body_func) {
  // 调用具有默认步长的 Compute 函数重载
  return Compute(name, dims, c10::nullopt, body_func);
}



Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    std::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const VarHandle&, const VarHandle&)>&
        body_func) {
  // 如果维度不等于2，抛出异常
  if (dims.size() != 2) {
    throw malformed_input("mismatch between body and arg size (2)");
  }
  // 创建索引变量
  std::vector<VarHandle> args = create_index_vars(dims);
  // 调用用户提供的函数计算表达式体
  ExprHandle body = body_func(args[0], args[1]);
  // 创建缓存对象，用于表示张量的存储
  BufHandle buf = Buf::make(name, dims, body.dtype(), c10::nullopt, strides);
  // 返回表示张量的对象
  return Tensor(buf, args, body);
}



Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(const VarHandle&, const VarHandle&)>&
        body_func) {
  // 调用具有默认步长的 Compute 函数重载
  return Compute(name, dims, c10::nullopt, body_func);
}



Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    std::optional<std::vector<ExprHandle>> strides,
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func) {
  // 如果维度不等于3，抛出异常
  if (dims.size() != 3) {
    throw malformed_input("mismatch between body and arg size (3)");
  }
  // 创建索引变量
  std::vector<VarHandle> args = create_index_vars(dims);
  // 调用用户提供的函数计算表达式体
  ExprHandle body = body_func(args[0], args[1], args[2]);
  // 创建缓存对象，用于表示张量的存储
  BufHandle buf = Buf::make(name, dims, body.dtype(), c10::nullopt, strides);
  // 返回表示张量的对象
  return Tensor(buf, args, body);
}



Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func) {
  // 调用具有默认步长的 Compute 函数重载
  return Compute(name, dims, c10::nullopt, body_func);
}
    // 定义一个函数，接受三个 VarHandle 类型的参数，返回一个 ExprHandle 类型的值。
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func)
    {
        // 调用 Compute 函数，传入名称 name、维度 dims、空的 reduction domain（c10::nullopt）以及提供的函数 body_func。
        return Compute(name, dims, c10::nullopt, body_func);
    }
}

// 定义一个名为 Compute 的函数，用于生成张量对象
Tensor Compute(
    const std::string& name, // 张量的名称
    const std::vector<ExprHandle>& dims, // 张量的维度列表
    std::optional<std::vector<ExprHandle>> strides, // 可选的步长列表
    const std::function<ExprHandle( // 处理张量元素的函数对象
        const VarHandle&, // 参数1
        const VarHandle&, // 参数2
        const VarHandle&, // 参数3
        const VarHandle&)>& body_func) { // 参数4，处理函数对象
  if (dims.size() != 4) { // 如果维度列表不等于4，抛出异常
    throw malformed_input("mismatch between body and arg size (4)");
  }
  // 创建索引变量列表
  std::vector<VarHandle> args = create_index_vars(dims);
  // 调用处理函数生成张量体表达式
  ExprHandle body = body_func(args[0], args[1], args[2], args[3]);
  // 创建张量缓冲区对象
  BufHandle buf = Buf::make(name, dims, body.dtype(), c10::nullopt, strides);
  // 返回张量对象
  return Tensor(buf, args, body);
}

// 定义一个名为 Compute 的函数重载，简化步长参数为空时的调用
Tensor Compute(
    const std::string& name, // 张量的名称
    const std::vector<ExprHandle>& dims, // 张量的维度列表
    const std::function<ExprHandle( // 处理张量元素的函数对象
        const VarHandle&, // 参数1
        const VarHandle&, // 参数2
        const VarHandle&, // 参数3
        const VarHandle&)>& body_func) { // 参数4，处理函数对象
  return Compute(name, dims, c10::nullopt, body_func); // 调用带步长参数的 Compute 函数
}

// 定义一个名为 Reduce 的函数，用于生成缩减操作的张量对象
Tensor Reduce(
    const std::string& name, // 张量的名称
    const std::vector<ExprHandle>& dims, // 张量的维度列表
    std::optional<std::vector<ExprHandle>> strides, // 可选的步长列表
    const Reducer& reducer, // 缩减操作对象
    const BufHandle& buffer, // 缓冲区对象
    const std::vector<ExprHandle>& reduce_dims) { // 缩减维度列表
  // 调用重载函数，传递参数列表和 Lambda 表达式以加载缓冲区数据
  return Reduce(
      name,
      dims,
      strides,
      reducer,
      [&](ParameterList& p) { return buffer.load(p); },
      reduce_dims);
}

// 定义一个名为 Reduce 的函数重载，简化步长参数为空时的调用
Tensor Reduce(
    const std::string& name, // 张量的名称
    const std::vector<ExprHandle>& dims, // 张量的维度列表
    const Reducer& reducer, // 缩减操作对象
    const BufHandle& buffer, // 缓冲区对象
    const std::vector<ExprHandle>& reduce_dims) { // 缩减维度列表
  return Reduce(name, dims, c10::nullopt, reducer, buffer, reduce_dims); // 调用带步长参数的 Reduce 函数
}

// 定义一个名为 Reduce 的函数，用于处理张量对象的缩减操作
Tensor Reduce(
    const std::string& name, // 张量的名称
    const std::vector<ExprHandle>& dims, // 张量的维度列表
    std::optional<std::vector<ExprHandle>> strides, // 可选的步长列表
    const Reducer& reducer, // 缩减操作对象
    Tensor tensor, // 张量对象
    const std::vector<ExprHandle>& reduce_dims) { // 缩减维度列表
  // 调用重载函数，传递参数列表和 Lambda 表达式以加载张量数据
  return Reduce(
      name,
      dims,
      strides,
      reducer,
      [&](ParameterList& p) { return tensor.load(p); },
      reduce_dims);
}

// 定义一个名为 Reduce 的函数重载，简化步长参数为空时的调用
Tensor Reduce(
    const std::string& name, // 张量的名称
    const std::vector<ExprHandle>& dims, // 张量的维度列表
    const Reducer& reducer, // 缩减操作对象
    Tensor tensor, // 张量对象
    const std::vector<ExprHandle>& reduce_dims) { // 缩减维度列表
  return Reduce(name, dims, c10::nullopt, reducer, tensor, reduce_dims); // 调用带步长参数的 Reduce 函数
}

} // namespace torch::jit::tensorexpr
```