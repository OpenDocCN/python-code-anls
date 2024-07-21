# `.\pytorch\test\cpp\tensorexpr\test_graph_opt.cpp`

```py
  graph(%x : Float(10, strides=[1], device=cpu),
        %y : Float(20, strides=[1], device=cpu),
        %z : Float(30, strides=[1], device=cpu)):
    %dim : int = prim::Constant[value=0]()  // 创建一个常量节点，值为0，表示张量拼接的维度
    %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)  // 创建一个张量列表，包含输入的三个张量 %x, %y, %z
    %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)  // 使用aten::cat操作，沿着维度0拼接xyz_list中的张量，得到拼接后的张量%cat
    %5 : Float(60, strides=[1], device=cpu) = aten::log(%cat)  // 对拼接后的张量%cat应用aten::log操作，结果存储在%5中
    return (%5))IR";  // 返回操作后的结果张量%5
    // 定义一个名为 `graph` 的函数，接受三个输入参数 x, y, z，类型为 FloatTensor，大小分别为 10, 20, 30，设备为 CPU
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(30, strides=[1], device=cpu)):
      // 设置一个整数常量 %dim，值为 0
      %dim : int = prim::Constant[value=0]()
      // 创建一个包含输入张量 x, y, z 的列表 %xyz_list
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      // 执行 torch 的 cat 操作，将列表 %xyz_list 中的张量按维度 %dim 连接起来，结果存入 %cat 中
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      // 对 %cat 中的元素执行自然对数运算，结果存入 %5 中
      %5 : Float(60, strides=[1], device=cpu) = aten::log(%cat)
      // 对 %5 中的元素执行双曲正切函数运算，结果存入 %6 中
      %6 : Float(60, strides=[1], device=cpu) = aten::tanh(%5)
      // 返回 %6 作为函数的结果
      return (%6))IR";
      
    // 创建一个名为 g 的共享指针，指向 Graph 类的实例
    auto g = std::make_shared<Graph>();
    // 使用 torch::jit::parseIR 将 graph_string 解析为 IR，并存入 g 中
    torch::jit::parseIR(graph_string, g.get());
    // 对解析后的图 g 进行静态分析
    g->lint();
    
    // 创建一个 TensorExprKernel 类的实例 kernel，以 g 作为构造函数的参数
    TensorExprKernel kernel(g);
    
    // 对 kernel.graph() 运行 FileCheck 检查器，验证图中 aten::log 和 aten::tanh 操作已经移动到 aten::cat 的输入位置
    testing::FileCheck()
        .check("aten::log")    // 检查是否存在 aten::log 操作
        ->check("aten::log")   // 检查是否存在第二个 aten::log 操作
        ->check("aten::log")   // 检查是否存在第三个 aten::log 操作
        ->check("aten::tanh")  // 检查是否存在 aten::tanh 操作
        ->check("aten::tanh")  // 检查是否存在第二个 aten::tanh 操作
        ->check("aten::tanh")  // 检查是否存在第三个 aten::tanh 操作
        ->check("aten::cat")   // 检查是否存在 aten::cat 操作
        ->check_not("aten::log")   // 确保不存在额外的 aten::log 操作
        ->check_not("aten::tanh")  // 确保不存在额外的 aten::tanh 操作
        ->run(*kernel.graph()); // 在 kernel 的图上运行检查器
    
    // 创建三个随机张量 x, y, z，数据类型为 Float，分别大小为 10, 20, 30
    auto x = at::rand({10}, at::kFloat);
    auto y = at::rand({20}, at::kFloat);
    auto z = at::rand({30}, at::kFloat);
    // 计算参考结果 ref，即 at::cat({x, y, z}, 0) 后执行 at::log 和 at::tanh 操作
    auto ref = at::tanh(at::log(at::cat({x, y, z}, 0)));
    
    // 创建一个输入张量的 IValue 向量 stack，以便在 kernel 上运行
    std::vector<at::Tensor> inputs = {x, y, z};
    std::vector<IValue> stack = fmap<IValue>(inputs);
    kernel.run(stack); // 在 kernel 上运行，计算结果存入 stack 中
    auto out = stack[0].toTensor(); // 从 stack 中获取输出张量 out
    // 断言输出张量 out 的大小与 ref 相同
    ASSERT_EQ(out.sizes(), ref.sizes());
    // 断言输出张量 out 的数据类型与 ref 相同
    ASSERT_EQ(out.dtype(), ref.dtype());
    // 断言输出张量 out 与 ref 在容差范围内相似
    ASSERT_TRUE(at::allclose(out, ref));
TEST_F(GraphOpt, OptimizeCat3) {
#ifdef TORCH_ENABLE_LLVM
  // 定义图形字符串，包含 IR 表示的计算图
  const auto graph_string = R"IR(
    graph(%a : Float(60, strides=[1], device=cpu),
          %x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      %5 : Float(60, strides=[1], device=cpu) = aten::tanh(%cat)
      %6 : Float(60, strides=[1], device=cpu) = aten::mul(%a, %5)
      return (%6))IR";

  // 创建一个新的计算图对象
  auto g = std::make_shared<Graph>();
  // 将图形字符串解析并加载到计算图中
  torch::jit::parseIR(graph_string, g.get());
  // 对加载后的计算图进行 lint 检查
  g->lint();

  // 创建 TensorExprKernel 对象来优化和处理计算图
  TensorExprKernel kernel(g);

  // 使用 FileCheck 进行检查和验证优化后的计算图
  // 确保 `aten::tanh` 操作被移动到 `aten::cat` 的输入处
  // 确保 `aten::mul` 操作未被移动，因为它不是单张量操作（具有两个张量输入）
  testing::FileCheck()
      .check("aten::tanh")
      ->check("aten::tanh")
      ->check("aten::tanh")
      ->check("aten::cat")
      ->check("aten::mul")
      ->check_not("aten::tanh")
      ->run(*kernel.graph());

  // 创建输入张量 a, x, y, z，随机初始化为 Float 类型
  auto a = at::rand({60}, at::kFloat);
  auto x = at::rand({10}, at::kFloat);
  auto y = at::rand({20}, at::kFloat);
  auto z = at::rand({30}, at::kFloat);
  // 计算预期的参考输出 ref，使用 at::tanh 和 at::cat 运算
  auto ref = at::tanh(at::cat({x, y, z}, 0)) * a;

  // 创建输入张量的向量和值的向量，用于运行优化后的计算图
  std::vector<at::Tensor> inputs = {a, x, y, z};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  // 运行优化后的计算图
  kernel.run(stack);
  // 获取运行后的输出张量
  auto out = stack[0].toTensor();
  // 断言输出张量的形状与参考输出的形状相同
  ASSERT_EQ(out.sizes(), ref.sizes());
  // 断言输出张量的数据类型与参考输出的数据类型相同
  ASSERT_EQ(out.dtype(), ref.dtype());
  // 断言输出张量与参考输出在数值上接近
  ASSERT_TRUE(at::allclose(out, ref));
#endif
}



TEST_F(GraphOpt, OptimizeCatWithTypePromotionInUser) {
#ifdef TORCH_ENABLE_LLVM
  // 定义图形字符串，包含 IR 表示的计算图
  const auto graph_string = R"IR(
    // 定义一个 TorchScript 图表达式，包含三个输入参数 x, y, z，均为整数张量类型
    graph(%x : Int(10, strides=[1], device=cpu),
          %y : Int(20, strides=[1], device=cpu),
          %z : Int(30, strides=[1], device=cpu)):
      // 定义一个常量维度值为 0 的节点
      %dim : int = prim::Constant[value=0]()
      // 创建一个张量列表，包含参数 x, y, z
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      // 执行 torch 中的 cat 操作，沿指定维度 dim 连接张量列表 xyz_list
      %cat : Int(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      // 执行 torch 中的 tanh 操作，对 cat 的结果应用双曲正切函数
      %5 : Float(60, strides=[1], device=cpu) = aten::tanh(%cat)
      // 返回 tanh 操作的结果作为输出
      return (%5))IR";
    
    // 创建一个 TorchScript 图对象 g，并解析给定的图表达式 graph_string
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());
    // 对解析得到的图进行静态分析检查
    g->lint();
    
    // 使用 TensorExprKernel 封装图表达式 kernel
    TensorExprKernel kernel(g);
    
    // 进行测试验证，确保 `aten::tanh` 操作被正确移动到 `aten::cat` 的输入端
    // 并且输入到 `cat` 操作的标量类型应为 `Float`，因为它们是 `tanh` 的结果，会进行类型提升
    testing::FileCheck()
        .check("aten::tanh")  // 检查是否存在 `aten::tanh`
        ->check("aten::tanh")  // 再次检查 `aten::tanh`
        ->check("aten::tanh")  // 第三次检查 `aten::tanh`
        ->check("aten::cat")   // 检查是否存在 `aten::cat`
        ->check_not("aten::tanh")  // 确保 `aten::tanh` 不再存在
        ->run(*kernel.graph());  // 在 kernel 对象封装的图上运行检查
    
    // 生成随机整数张量 x, y, z，使用整数类型最大限制，分别包含 10, 20, 30 个元素
    auto x = at::randint(std::numeric_limits<int>::max(), {10}, at::kInt);
    auto y = at::randint(std::numeric_limits<int>::max(), {20}, at::kInt);
    auto z = at::randint(std::numeric_limits<int>::max(), {30}, at::kInt);
    // 生成参考结果，对连接后的张量列表 {x, y, z} 沿维度 0 执行 `cat` 和 `tanh`
    auto ref = at::tanh(at::cat({x, y, z}, 0));
    
    // 准备输入列表，包含张量 x, y, z
    std::vector<at::Tensor> inputs = {x, y, z};
    // 使用 fmap 将输入列表转换为 IValue 类型的栈
    std::vector<IValue> stack = fmap<IValue>(inputs);
    // 在 kernel 对象上运行图表达式，传入栈作为输入
    kernel.run(stack);
    // 从运行后的栈中取出结果张量 out
    auto out = stack[0].toTensor();
    // 断言输出张量的尺寸与参考结果相同
    ASSERT_EQ(out.sizes(), ref.sizes());
    // 断言输出张量的数据类型与参考结果相同
    ASSERT_EQ(out.dtype(), ref.dtype());
    // 断言输出张量与参考结果在数值上相近
    ASSERT_TRUE(at::allclose(out, ref));
TEST_F(GraphOpt, OptimizeCatWithTypePromotionInCat) {
#ifdef TORCH_ENABLE_LLVM
  // 定义包含图形的字符串表示，描述了一个计算图
  const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Double(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Double(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      %5 : Double(60, strides=[1], device=cpu) = aten::log(%cat)
      return (%5))IR";
  // 创建一个新的图对象
  auto g = std::make_shared<Graph>();
  // 将字符串表示的IR解析到图中
  torch::jit::parseIR(graph_string, g.get());
  // 对图进行静态分析
  g->lint();

  // 创建一个TensorExprKernel对象，用于表示图的张量表达式内核
  TensorExprKernel kernel(g);

  // 检查：由于`aten::cat`操作执行了类型提升，不应有任何转换发生
  testing::FileCheck()
      .check("aten::cat")       // 检查图中是否包含`aten::cat`操作
      ->check("aten::log")      // 检查图中是否包含`aten::log`操作
      ->check_not("aten::cat")  // 确保第二次检查时`aten::cat`操作不再出现
      ->check_not("aten::log")  // 确保第二次检查时`aten::log`操作不再出现
      ->run(*kernel.graph());   // 在图的张量表达式内核上运行检查
#endif
}

TEST_F(GraphOpt, OptimizeCatNoSingleTensorElementwiseOp) {
#ifdef TORCH_ENABLE_LLVM
  // 定义包含图形的字符串表示，描述了一个计算图
  const auto graph_string = R"IR(
    graph(%0 : Float(60, strides=[1], device=cpu),
          %x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      %5 : Float(60, strides=[1], device=cpu) = aten::mul(%0, %cat)
      return (%5))IR";
  // 创建一个新的图对象
  auto g = std::make_shared<Graph>();
  // 将字符串表示的IR解析到图中
  torch::jit::parseIR(graph_string, g.get());
  // 对图进行静态分析
  g->lint();

  // 创建一个TensorExprKernel对象，用于表示图的张量表达式内核
  TensorExprKernel kernel(g);

  // 检查：由于`aten::cat`操作的消费者不是单一张量的逐元素操作，不应有任何转换发生
  testing::FileCheck()
      .check("aten::cat")       // 检查图中是否包含`aten::cat`操作
      ->check("aten::mul")      // 检查图中是否包含`aten::mul`操作
      ->check_not("aten::cat")  // 确保第二次检查时`aten::cat`操作不再出现
      ->check_not("aten::mul")  // 确保第二次检查时`aten::mul`操作不再出现
      ->run(*kernel.graph());   // 在图的张量表达式内核上运行检查
#endif
}
    // 定义一个 TorchScript IR 图形表示，描述了一个计算图
    graph(%0 : Float(60, strides=[1], device=cpu),
          %1 : Float(60, strides=[1], device=cpu),
          %x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(30, strides=[1], device=cpu)):
      // 定义一个整数常量 one 值为 1
      %one : int = prim::Constant[value=1]()
      // 定义一个整数常量 dim 值为 0
      %dim : int = prim::Constant[value=0]()
      // 创建一个 Tensor 列表 xyz_list，包含张量 %x, %y, %z
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      // 使用 aten::cat 函数沿指定维度 dim 连接张量列表 xyz_list
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      // 使用 aten::mul 函数对张量 %0 和 %cat 进行逐元素相乘
      %5 : Float(60, strides=[1], device=cpu) = aten::mul(%0, %cat)
      // 使用 aten::add 函数对 %5 和 %1 进行逐元素相加，并加上常量 %one
      %6 : Float(60, strides=[1], device=cpu) = aten::add(%5, %1, %one)
      // 返回计算结果 %6
      return (%6))IR";
  // 创建一个新的 TorchScript 图形对象 g，解析给定的 IR 字符串并添加到 g 中
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  // 对图形 g 进行 lint 检查
  g->lint();

  // 创建一个 TensorExprKernel 对象 kernel，基于给定的图形 g
  TensorExprKernel kernel(g);

  // 预期不会进行任何转换，因为 cat 的消费者不是单个张量的逐元素操作
  // 使用 FileCheck 进行测试，验证是否存在特定的 aten 操作，并确保不存在其他操作
  testing::FileCheck()
      .check("aten::cat")
      ->check("aten::mul")
      ->check("aten::add")
      ->check_not("aten::cat")
      ->check_not("aten::mul")
      ->check_not("aten::add")
      ->run(*kernel.graph());
#endif
}

// 在 GraphOpt 测试框架中定义一个测试用例 AOTGraphPrepPasses
TEST_F(GraphOpt, AOTGraphPrepPasses) {
  // 定义一个包含 IR 的字符串，描述了一个计算图
  const auto graph_string = R"IR(
    graph(%x, %y, %z, %i : int):
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      return (%xyz_list, %i))IR";
  
  // 创建一个共享指针指向 Graph 对象
  auto g = std::make_shared<Graph>();
  
  // 使用 Torch 的 JIT 解析上述 IR 字符串，并将解析结果存储在 g 中
  torch::jit::parseIR(graph_string, g.get());

  // 调用函数 removeGraphOutput，移除计算图 g 的第一个输出
  removeGraphOutput(g, 1);
  
  // 调用函数 replaceListOutputWithTuple，将计算图 g 中列表输出替换为元组输出
  replaceListOutputWithTuple(g);
  
  // 调用函数 LowerAllTuples，对计算图 g 中的所有元组进行降级处理
  LowerAllTuples(g);

  // 使用 FileCheck 进行断言，验证计算图 g 是否包含 "return (%x, %y, %z)" 的语句
  testing::FileCheck().check("return (%x, %y, %z)")->run(*g);
}

// 声明结束 GraphOpt 测试框架
} // namespace jit
} // namespace torch
```