# `.\pytorch\torch\csrc\jit\runtime\static\native_ops.cpp`

```py
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <c10/util/ssize.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/mobile/promoted_prim_ops.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

// 命名空间，用于封装实现细节
namespace {
    // 创建 borrow 类型的 IValue 的工具函数
    constexpr auto createBorrowedIValue = c10::MaybeOwnedTraits<c10::IValue>::createBorrow;
} // namespace

// torch::jit 命名空间下的函数和类
namespace torch::jit {

// 匿名命名空间，用于封装实现细节
namespace {

// 将 ProcessedNode 节点的输入打包成 IValue 数组
std::vector<IValue> boxInputs(const ProcessedNode& pnode) {
  std::vector<IValue> result;
  // 遍历节点的输入，将其存入结果数组
  for (const auto i : c10::irange(pnode.num_inputs())) {
    result.push_back(pnode.Input(i));
  }
  return result;
}

} // namespace

// 定义一个注册表，用于注册原生操作函数符
C10_DEFINE_REGISTRY(SRNativeOperatorRegistry, SROperatorFunctor);

// 检查给定操作是否已经注册
bool nativeOpIsRegistered(const c10::Symbol& op_name) {
  const std::string name(op_name.toQualString());
  return SRNativeOperatorRegistry()->Has(name);
}

// 获取给定节点的原生操作函数符
SROperator getNativeOperation(Node* n) {
  auto op_name = n->kind().toQualString();
  if (SRNativeOperatorRegistry()->Has(op_name)) {
    return SRNativeOperatorRegistry()->Create(op_name)->Generate(n);
  }
  return nullptr;
}

// 注册 prim::TupleConstruct 操作的函数符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::TupleConstruct,
    prim_TupleConstruct,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::TupleConstruct)) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        // 准备输入数据
        auto stack = boxInputs(*p_node);
        // 执行操作
        auto* node = p_node->node();
        const auto& type = node->output()->type()->expect<TupleType>();
        if (type->name().has_value()) {
          namedTupleConstruct(stack, type, node->inputs().size());
        } else {
          tupleConstruct(stack, node->inputs().size());
        }
        // 将输出放回到节点中
        p_node->Output(0) = std::move(stack[0]);
      };
    });

// 注册 prim::TupleUnpack 操作的函数符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::TupleUnpack,
    prim_TupleUnpack,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::TupleUnpack)) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& elems = p_node->Input(0).toTupleRef().elements();
        const size_t num_outputs = p_node->outputs().size();
        TORCH_CHECK(
            num_outputs == elems.size(),
            "Number of outputs must match number of tuple elements.")
        for (size_t i = 0; i < num_outputs; ++i) {
          p_node->Output(i) = elems[i];
        }
      };
    });

// 注册 prim::DictConstruct 操作的函数符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::DictConstruct,
    prim_DictConstruct,
    // 待补充...


**注释：**
    // 定义一个匿名函数，接受一个 Node* 参数并返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否为 prim::DictConstruct 类型，如果不是则返回空指针
      if (!sr_schema_check_kind(n, prim::DictConstruct)) {
        return nullptr;
      }
      // 获取节点 n 输出的类型，并期望其为 DictType 类型
      auto dict_type = n->output()->type()->expect<DictType>();
      // 获取节点 n 的输入数量
      const auto num_inputs = n->inputs().size();
      // 使用 TORCH_DCHECK_EQ 断言确认输入数量为偶数
      TORCH_DCHECK_EQ(num_inputs % 2, 0);
      // 返回一个 lambda 表达式，捕获 dict_type, num_inputs, 以及计算得到的 dict_size
      return [dict_type = std::move(dict_type),
              num_inputs,
              dict_size = num_inputs / 2](ProcessedNode* p_node) {
        // 创建一个 GenericDict 对象 result，用于存储键值对
        auto result = c10::impl::GenericDict(
            dict_type->containedType(0), dict_type->containedType(1));
        // 预留足够的空间以容纳 dict_size 个元素
        result.reserve(dict_size);
        // 遍历输入的键值对，每两个输入构成一个键值对插入到 result 中
        for (size_t i = 0; i < num_inputs; i += 2) {
          const auto& key = p_node->Input(i);
          const auto& value = p_node->Input(i + 1);
          result.insert_or_assign(key, value);  // 插入或更新键值对
        }
        // 将 result 分配给 p_node 的第一个输出
        p_node->Output(0) = result;
      };
    });
// 注册本地操作符函数 static_runtime::dict_unpack
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    static_runtime::dict_unpack,
    static_runtime_dict_unpack,
    [](Node* n) -> SROperator {
      // 检查节点 n 的模式是否符合 "static_runtime::dict_unpack(...) -> ..."
      if (!sr_schema_check(n, "static_runtime::dict_unpack(...) -> ...")) {
        return nullptr;
      }
      // 返回一个 lambda 函数，处理节点的输入和输出
      return [](ProcessedNode* p_node) {
        // 断言：输入数量减去 1 应等于输出数量
        DCHECK(
            static_cast<size_t>(p_node->num_inputs() - 1) ==
            p_node->outputs().size());
        // 获取第一个输入，将其视为通用字典类型
        auto dict = p_node->Input(0).toGenericDict();
        const auto num_inputs = p_node->num_inputs();
        // 遍历输入节点，从字典中查找对应的键值对，并创建 BorrowedIValue
        for (size_t i = 1; i < num_inputs; ++i) {
          const auto& key = p_node->Input(i);
          auto value = dict.find(key);
          // 检查字典中是否存在对应的键
          TORCH_CHECK(value != dict.end(), "Key not in dict: ", key);
          // 将 BorrowedIValue 存入节点的输出中
          p_node->Output(i - 1) = createBorrowedIValue(value->value());
        }
      };
    });

// 注册本地操作符函数 aten::__getitem__
REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::__getitem__, aten_getitem, [](Node* n) -> SROperator {
  // 检查节点 n 是否符合指定的多个模式
  if (!sr_schema_check(
          n,
          // TODO: "aten::__getitem__.str(str s, int index) -> str",
          "aten::__getitem__.t(t[](a) list, int idx) -> t(*)",
          "aten::__getitem__.Dict_str(Dict(str, t) self, str key) -> t(*)",
          "aten::__getitem__.Dict_int(Dict(int, t) self, int key) -> t(*)",
          "aten::__getitem__.Dict_bool(Dict(bool, t) self, bool key) -> t(*)",
          "aten::__getitem__.Dict_float(Dict(float, t) self, float key) -> t(*)",
          "aten::__getitem__.Dict_complex(Dict(complex, t) self, complex key) -> t(*)",
          "aten::__getitem__.Dict_Tensor(Dict(Tensor, t) self, Tensor key) -> t(*)")) {
    return nullptr;
  }

  // 检查节点的输入数量是否为 2
  if (n->inputs().size() != 2) {
    return nullptr;
  }

  // 如果输入的第一个节点是字典类型
  if (n->input(0)->type()->castRaw<DictType>()) {
    // 返回一个 lambda 函数，处理节点的输入和输出
    return [](ProcessedNode* p_node) {
      auto dict = p_node->Input(0).toGenericDict();
      const auto& key = p_node->Input(1);
      auto value = dict.find(key);
      // 检查字典中是否存在对应的键
      TORCH_CHECK(value != dict.end(), "Key not in dict: ", key);
      // 将找到的值存入节点的输出中
      p_node->Output(0) = value->value();
    };
  } else if (n->input(0)->type()->castRaw<ListType>()) {
    // 如果输入的第一个节点是列表类型
    // 返回一个 lambda 函数，处理节点的输入和输出
    return [](ProcessedNode* p_node) {
      const auto& list = p_node->Input(0).toList();
      auto idx = p_node->Input(1).toInt();
      // 获取列表中指定索引的元素，并存入节点的输出中
      p_node->Output(0) = getItem(list, idx);
    };
  }

  // TODO(T98581096): 使 __getitem__ 能够适用于其他容器类型
  // 暂时返回 nullptr，表示不支持的情况
  return nullptr;
});

// 注册本地操作符函数 prim::ListConstruct
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::ListConstruct,
    prim_ListConstruct,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否是 prim::ListConstruct 类型
      if (!sr_schema_check_kind(n, prim::ListConstruct)) {
        return nullptr;
      }
      // 返回一个 lambda 函数，处理节点的输入和输出
      return [](ProcessedNode* p_node) {
        // 准备输入
        auto stack = boxInputs(*p_node);
        // 运行操作，构造列表
        listConstruct(
            stack,
            p_node->node()->output()->type()->expectRef<ListType>(),
            p_node->num_inputs());
        // 将输出放回
        p_node->Output(0) = std::move(stack[0]);
      };
    });

// 注册本地操作符函数 prim::ListUnpack
    prim_ListUnpack,
    // 声明一个 Lambda 函数，接受一个 Node* 参数，返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否符合 prim::ListUnpack 类型，如果不符合则返回空指针
      if (!sr_schema_check_kind(n, prim::ListUnpack)) {
        return nullptr;
      }
      // 获取节点 n 的输出数量
      const auto num_outputs = n->outputs().size();
      // 返回一个 Lambda 函数，这个函数接受一个 ProcessedNode* 参数
      return [num_outputs](ProcessedNode* p_node) {
        // 获取 p_node 的第一个输入作为列表的引用
        const auto list = p_node->Input(0).toListRef();
        // 检查列表的长度是否等于 num_outputs，如果不等则抛出错误信息
        TORCH_CHECK(
            list.size() == num_outputs,
            "Expected ",
            num_outputs,
            " elements in list but got ",
            list.size());
        // 遍历 num_outputs 范围内的索引 i
        for (const auto i : c10::irange(num_outputs)) {
          // 将列表中的第 i 个元素赋值给 p_node 的第 i 个输出
          p_node->Output(i) = list[i];
        }
      };
    });
# 注册自定义的本地操作符函数，实现对应的功能
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::append,  # 注册 aten::append 操作符
    aten_append,   # 定义本地函数 aten_append
    [](Node* n) -> SROperator {  # Lambda 函数，输入为节点指针，输出为 SROperator
      # 检查节点 n 是否符合特定的 schema
      if (!sr_schema_check(
              n, "aten::append.t(t[](a!) self, t(c -> *) el) -> t[](a!)")) {
        return nullptr;  # 如果不符合条件，返回空指针
      }
      return [](ProcessedNode* p_node) {  # Lambda 函数，处理已处理的节点 p_node
        auto list = p_node->Input(0).toList();  # 获取输入节点的列表
        list.push_back(p_node->Input(1));  # 将第二个输入节点添加到列表的末尾
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::list,  # 注册 aten::list 操作符
    aten_list,   # 定义本地函数 aten_list
    [](Node* n) -> SROperator {  # Lambda 函数，输入为节点指针，输出为 SROperator
      // 检查节点 n 是否匹配特定的 Torch 脚本 schema
      if (n->matches(torch::schema("aten::list(str t) -> str[]"))) {
        return [](ProcessedNode* p_node) {  # Lambda 函数，处理已处理的节点 p_node
          const auto str = p_node->Input(0).toStringRef();  # 获取输入节点的字符串表示
          c10::List<std::string> chars;  # 创建一个字符串列表
          chars.reserve(str.size());  # 预留足够的空间
          for (auto c : str) {  # 遍历字符串中的每个字符
            chars.emplace_back(1, c);  # 将每个字符作为单个字符串元素添加到列表中
          }
          p_node->Output(0) = std::move(chars);  # 将列表移动到输出节点的第一个输出位置
        };
      }

      // 检查节点 n 是否匹配特定的 Torch 脚本 schema
      if (n->matches(torch::schema("aten::list.t(t[] l) -> t[]"))) {
        return [](ProcessedNode* p_node) {  # Lambda 函数，处理已处理的节点 p_node
          const auto input = p_node->Input(0).toList();  # 获取输入节点的列表表示
          p_node->Output(0) = input.copy();  # 复制输入列表并将其移动到输出节点的第一个输出位置
        };
      }

      LogAndDumpSchema(n);  # 记录和转储未匹配的 schema
      return nullptr;  # 返回空指针表示未能处理该节点
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::numel,  # 注册 aten::numel 操作符
    aten_numel,   # 定义本地函数 aten_numel
    [](Node* n) -> SROperator {  # Lambda 函数，输入为节点指针，输出为 SROperator
      # 检查节点 n 是否符合特定的 schema
      if (!sr_schema_check(n, "aten::numel(Tensor self) -> int")) {
        return nullptr;  # 如果不符合条件，返回空指针
      }
      return [](ProcessedNode* p_node) {  # Lambda 函数，处理已处理的节点 p_node
        const auto& arg = p_node->Input(0).toTensor();  # 获取输入节点的张量表示
        p_node->Output(0) = arg.numel();  # 将张量的元素数量作为输出节点的第一个输出位置
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::cpu,  # 注册 aten::cpu 操作符
    aten_cpu,   # 定义本地函数 aten_cpu
    [](Node* n) -> SROperator {  # Lambda 函数，输入为节点指针，输出为 SROperator
      # 检查节点 n 是否符合特定的 schema
      if (!sr_schema_check(n, "aten::cpu(Tensor self) -> Tensor")) {
        return nullptr;  # 如果不符合条件，返回空指针
      }
      return [](ProcessedNode* p_node) {  # Lambda 函数，处理已处理的节点 p_node
        const auto& arg = p_node->Input(0).toTensor();  # 获取输入节点的张量表示
        p_node->Output(0) = arg.cpu();  # 将张量移动到 CPU 并作为输出节点的第一个输出位置
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::__range_length,  # 注册 aten::__range_length 操作符
    aten_range_length,     # 定义本地函数 aten_range_length
    [](Node* n) -> SROperator {  # Lambda 函数，输入为节点指针，输出为 SROperator
      # 检查节点 n 是否符合特定的 schema
      if (!sr_schema_check(
              n, "aten::__range_length(int lo, int hi, int step) -> int")) {
        return nullptr;  # 如果不符合条件，返回空指针
      }
      return [](ProcessedNode* p_node) {  # Lambda 函数，处理已处理的节点 p_node
        auto lo = p_node->Input(0).toInt();  # 获取输入节点的整数表示
        auto hi = p_node->Input(1).toInt();  # 获取输入节点的整数表示
        auto step = p_node->Input(2).toInt();  # 获取输入节点的整数表示

        // 运行时处理 step == 0 的错误情况
        if (step == 0) {
          throw std::runtime_error("range() arg 3 must not be zero");
        }

        // 根据步长的正负和起止值计算范围长度
        if (step > 0 && lo < hi) {
          p_node->Output(0) = 1 + (hi - 1 - lo) / step;  # 计算范围长度并作为输出节点的第一个输出位置
        } else if (step < 0 && lo > hi) {
          p_node->Output(0) = 1 + (lo - 1 - hi) / (0 - step);  # 计算范围长度并作为输出节点的第一个输出位置
        } else {
          p_node->Output(0) = 0;  # 若范围无效则输出 0
        }
      };
    });
REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::index_put, aten_index_put, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的索引更新操作的模式
  if (n->matches(torch::schema(
          "aten::index_put(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor")) ||
      n->matches(torch::schema(
          "aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"))) {
    // 如果匹配成功，则返回一个lambda函数
    return [](ProcessedNode* p_node) {
      // 从处理后的节点中获取输入参数
      const auto& self = p_node->Input(0).toTensor();
      const auto& indices =
          at::native::toListOfOptionalTensors(p_node->Input(1).toListRef());
      const auto& values = p_node->Input(2).toTensor();
      const auto accumulate = p_node->Input(3).toBool();
      // 执行索引更新操作，并将结果存储到输出中
      p_node->Output(0) =
          at::native::index_put(self, indices, values, accumulate);
    };
  }

  // 如果节点不匹配任何模式，则记录并转储节点的模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::item,
    aten_item,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配获取Tensor标量值的模式
      if (!sr_schema_check(n, "aten::item(Tensor self) -> Scalar")) {
        return nullptr;
      }
      // 如果匹配成功，则返回一个lambda函数
      return [](ProcessedNode* p_node) {
        // 从处理后的节点中获取输入参数
        const auto& self = p_node->Input(0).toTensor();
        // 执行获取Tensor标量值的操作，并将结果存储到输出中
        p_node->Output(0) = at::native::item(self);
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::GetAttr,
    prim_GetAttr,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配获取属性的模式
      if (!sr_schema_check_kind(n, prim::GetAttr)) {
        return nullptr;
      }
      // 如果匹配成功，则返回一个lambda函数
      return [](ProcessedNode* p_node) {
        // 从处理后的节点中获取输入参数
        auto& module = p_node->Input(0).toObjectRef();
        Node* node = p_node->node();
        const auto& type = node->input()->type()->expectRef<ClassType>();
        const auto& field = node->s(attr::name);
        const auto slot = type.getAttributeSlot(field);
        // 获取模块的指定属性并存储到输出中
        p_node->Output(0) = module.getSlot(slot);
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::SetAttr,
    prim_SetAttr,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配设置属性的模式
      if (!sr_schema_check_kind(n, prim::SetAttr)) {
        return nullptr;
      }
      // 如果匹配成功，则返回一个lambda函数
      return [](ProcessedNode* p_node) {
        // 从处理后的节点中获取输入参数
        auto& module = p_node->Input(0).toObjectRef();
        Node* node = p_node->node();
        const auto& type = node->inputs()[0]->type()->expectRef<ClassType>();
        const auto& field = node->s(attr::name);
        const auto slot = type.getAttributeSlot(field);
        // 设置模块的指定属性值
        module.setSlot(slot, p_node->Input(1));
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::transpose,
    aten_transpose,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配转置操作的模式
      if (!n->matches(torch::schema(
              "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)"))) {
        // 如果节点不匹配模式，则记录并转储节点的模式信息
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 如果匹配成功，则返回一个lambda函数
      return [](ProcessedNode* p_node) {
        // 从处理后的节点中获取输入参数
        const auto& in0_t = p_node->Input(0).toTensor();
        const auto in1_i = p_node->Input(1).toInt();
        const auto in2_i = p_node->Input(2).toInt();
        // 执行Tensor转置操作，并将结果存储到输出中
        p_node->Output(0) = at::native::transpose(in0_t, in1_i, in2_i);
      };
    });
REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::flatten, aten_flatten, [](Node* n) -> SROperator {
  // 检查节点是否匹配 flatten 操作的输入模式，如果不匹配则记录日志并返回空指针
  if (!n->matches(torch::schema(
          "aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 匹配成功，返回一个 lambda 表达式，该表达式将节点处理为展平操作
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor(); // 获取输入张量
    const auto in1_i = p_node->Input(1).toInt();    // 获取 start_dim 参数
    const auto in2_i = p_node->Input(2).toInt();    // 获取 end_dim 参数
    p_node->Output(0) = at::native::flatten(in0_t, in1_i, in2_i); // 执行展平操作并将结果写入输出
  };
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::permute,
    aten_permute,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配 permute 操作的输入模式，如果不匹配则记录日志并返回空指针
      if (!n->matches(torch::schema(
              "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 匹配成功，返回一个 lambda 表达式，该表达式将节点处理为排列操作
      return [](ProcessedNode* p_node) {
        const auto& in0_t = p_node->Input(0).toTensor();   // 获取输入张量
        const auto in1_iv = p_node->Input(1).toDimVector(); // 获取 dims 参数
        p_node->Output(0) = at::native::permute(in0_t, in1_iv); // 执行排列操作并将结果写入输出
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::reshape,
    aten_reshape,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配 reshape 操作的输入模式，如果不匹配则记录日志并返回空指针
      if (!n->matches(torch::schema(
              "aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 匹配成功，返回一个 lambda 表达式，该表达式将节点处理为重塑操作
      return [](ProcessedNode* p_node) {
        const auto& in0_t = p_node->Input(0).toTensor();   // 获取输入张量
        const auto in1_iv = p_node->Input(1).toDimVector(); // 获取 shape 参数
        p_node->Output(0) = at::native::reshape(in0_t, in1_iv); // 执行重塑操作并将结果写入输出
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::slice, aten_slice, [](Node* n) -> SROperator {
  // 检查节点是否匹配 slice 操作的输入模式，如果不匹配则记录日志并返回空指针
  if (!n->matches(torch::schema(
          "aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=0, int? end=9223372036854775807, int step=1) -> Tensor(a)"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 匹配成功，返回一个 lambda 表达式，该表达式将节点处理为切片操作
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();    // 获取输入张量
    const auto in1_i = p_node->Input(1).toInt();        // 获取 dim 参数
    const auto in2_i = p_node->Input(2).toOptional<int64_t>(); // 获取 start 参数
    const auto in3_i = p_node->Input(3).toOptional<int64_t>(); // 获取 end 参数
    const auto in4_i = p_node->Input(4).toInt();        // 获取 step 参数
    p_node->Output(0) = at::native::slice(in0_t, in1_i, in2_i, in3_i, in4_i); // 执行切片操作并将结果写入输出
  };
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::narrow, aten_narrow, [](Node* n) -> SROperator {
  // 检查节点是否匹配 narrow 操作的输入模式，如果不匹配则记录日志并返回空指针
  if (!n->matches(torch::schema(
          "aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)")) &&
      !n->matches(torch::schema(
          "aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  // 匹配成功，返回一个 lambda 表达式，该表达式将节点处理为收窄操作
  return [](ProcessedNode* p_node) {
    const auto& self = p_node->Input(0).toTensor();   // 获取 self 张量
    const auto dim = p_node->Input(1).toInt();        // 获取 dim 参数
    int64_t start = 0;
    if (p_node->Input(2).isScalar()) {
      start = p_node->Input(2).toInt();               // 获取 start 参数
    }
    } else {
      // 如果条件不满足，则从节点的第二个输入获取张量 t
      auto& t = p_node->Input(2).toTensor();
      // 获取 t 的整数值，作为起始索引
      start = t.item<int64_t>();
    }
    // 获取长度，使用 p_node 的第三个输入转换为整数
    const auto length = p_node->Input(3).toInt(); // length
    // 检查当前张量 self 的维度是否大于 0
    TORCH_CHECK(
        self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
    // 获取当前维度 dim 的大小
    auto cur_size = self.sizes()[dim];
    // 如果起始索引 start 不等于当前维度大小且小于 0
    if (start != cur_size && start < 0) { // start being the end is valid, but
                                          // not a valid dim specification.
      // 使用 maybe_wrap_dim 函数处理 start，确保在合法范围内
      start = at::maybe_wrap_dim(start, cur_size);
    }
    // 检查起始索引和长度是否在有效范围内
    TORCH_CHECK(
        length >= 0 && start <= cur_size - length,
        "start (",
        start,
        ") + length (",
        length,
        ") exceeds dimension size (",
        cur_size,
        ").");
    // 在 p_node 的第一个输出位置设置调用 slice 函数的结果
    p_node->Output(0) = at::native::slice(self, dim, start, start + length, 1);
  };
});

// 注册本地操作函数，处理aten::to操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::to, aten_to, [](Node* n) -> SROperator {
  // 检查节点是否匹配指定的aten::to.other模式
  if (n->matches(torch::schema(
          "aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"))) {
    // 返回lambda函数，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 提取输入参数
      const auto& in0_t = p_node->Input(0).toTensor();
      const auto& in1_t = p_node->Input(1).toTensor();
      const auto in2_i = p_node->Input(2).toBool();
      const auto in3_i = p_node->Input(3).toBool();
      const auto in4_o = p_node->Input(4).toOptional<at::MemoryFormat>();
      // 调用at::native::to函数，将结果赋给输出
      p_node->Output(0) = at::native::to(in0_t, in1_t, in2_i, in3_i, in4_o);
    };
  }
  // 检查节点是否匹配指定的aten::to.dtype模式
  if (n->matches(torch::schema(
          "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"))) {
    // 返回lambda函数，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 提取输入参数
      const auto& in0_t = p_node->Input(0).toTensor();
      const auto in1_i = p_node->Input(1).toScalarType();
      const auto in2_i = p_node->Input(2).toBool();
      const auto in3_i = p_node->Input(3).toBool();
      const auto in4_o = p_node->Input(4).toOptional<at::MemoryFormat>();
      // 调用at::native::to函数，将结果赋给输出
      p_node->Output(0) = at::native::to(in0_t, in1_i, in2_i, in3_i, in4_o);
    };
  }
  // 检查节点是否匹配指定的aten::to.prim_dtype模式
  if (n->matches(torch::schema(
          "aten::to.prim_dtype(Tensor(a) self, int? dtype, bool non_blocking=False, bool copy=False) -> Tensor(a|b)"))) {
    // 返回lambda函数，处理匹配的节点
    return [](ProcessedNode* p_node) {
      // 提取输入参数
      const auto& in0_t = p_node->Input(0).toTensor();
      const auto in1_i = p_node->Input(1).toOptional<at::ScalarType>();
      const auto in2_i = p_node->Input(2).toBool();
      const auto in3_i = p_node->Input(3).toBool();
      // 模拟JIT解释器的行为，如果dtype和copy都未设置，则返回自身
      if (!in1_i && !in3_i) {
        p_node->Output(0) = in0_t;
      } else {
        // 否则，确保dtype不为空，调用at::native::to函数，将结果赋给输出
        TORCH_CHECK(
            in1_i,
            "dytpe cannot be None when copy is True for aten::to.prim_dtype");
        p_node->Output(0) = at::native::to(in0_t, *in1_i, in2_i, in3_i);
      }
    };
  }
  // 若以上所有情况都不匹配，则记录并输出模式信息
  LogAndDumpSchema(n);
  return nullptr;
});

// 注册本地操作函数，处理aten::detach操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::detach,
    aten_detach,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配指定的aten::detach模式
      if (!n->matches(
              torch::schema("aten::detach(Tensor(a) self) -> Tensor(a)"))) {
        // 若不匹配，则记录并返回空指针
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 返回lambda函数，处理匹配的节点
      return [](ProcessedNode* p_node) {
        // 提取输入参数，调用at::native::alias函数，并将结果赋给输出
        const auto& in0_t = p_node->Input(0).toTensor();
        p_node->Output(0) = at::native::alias(in0_t);
      };
    });

// 注册本地操作函数，处理aten::expand_as操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::expand_as,
    aten_expand_as,
    // 匿名函数，接受一个指向 Node 类型的指针 n，返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 如果节点 n 不匹配特定的 Torch 脚本
      if (!n->matches(torch::schema(
              "aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)"))) {
        // 记录并且输出节点的模式信息
        LogAndDumpSchema(n);
        // 返回空指针
        return nullptr;
      }
      // 匿名函数，接受一个 ProcessedNode* 类型的参数 p_node，返回 void
      return [](ProcessedNode* p_node) {
        // 获取输入的第一个张量 self
        const auto& self = p_node->Input(0).toTensor();
        // 获取输入的第二个张量 other
        const auto& other = p_node->Input(1).toTensor();
        // 将 self 张量根据 other 张量的尺寸进行扩展，并将结果保存到输出的第一个位置
        p_node->Output(0) = self.expand(other.sizes());
      };
    });
// 注册本地运算符函数，处理 prim::isinstance 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::isinstance,
    prim_isinstance,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 脚本模式
      if (!n->matches(
              torch::schema("prim::isinstance(Any to_check) -> bool"))) {
        // 如果不匹配，记录日志并且返回空指针
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 返回一个 lambda 函数，处理节点的输出
      return [](ProcessedNode* p_node) {
        auto input_type = p_node->Input(0).type();

        auto* node = p_node->node();
        // 获取操作符节点的类型候选项
        const std::vector<TypePtr>& candidates = node->tys(attr::types);
        // 遍历候选项，检查输入类型是否是候选类型的子类型
        for (const auto& candidate_type : candidates) {
          if (input_type->isSubtypeOf(*candidate_type)) {
            // 设置节点的输出为 true 并返回
            p_node->Output(0) = true;
            return;
          }
        }

        // 如果没有匹配的类型，设置节点的输出为 false
        p_node->Output(0) = false;
      };
    });

// 注册本地运算符函数，处理 prim::TypeCheck 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::TypeCheck,
    prim_TypeCheck,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 prim::TypeCheck 类型的模式
      if (!sr_schema_check_kind(n, prim::TypeCheck)) {
        // 如果不符合，返回空指针
        return nullptr;
      }
      // 返回一个 lambda 函数，处理节点的输出
      return [](ProcessedNode* p_node) {
        auto* node = p_node->node();
        const size_t num_inputs = node->inputs().size();
        // 内部断言，确保输入输出数量正确
        TORCH_INTERNAL_ASSERT(
            num_inputs && num_inputs + 1 == node->outputs().size());

        // 获取期望的输入类型
        const auto& expected_types = node->tys(attr::types);

        // 将输入直接复制到输出
        for (size_t i = 0; i < num_inputs; i++) {
          p_node->Output(i) = p_node->Input(i);
        }

        // 检查每个输入是否符合其期望的类型
        for (size_t i = 0; i < num_inputs; i++) {
          auto& input_tensor = p_node->Input(i).toTensor();
          auto* expected_type = expected_types[i]->castRaw<TensorType>();
          if (input_tensor.defined() &&
              !expected_type->matchTensor(input_tensor)) {
            // 如果类型不匹配，设置最后一个输出为 false 并返回
            p_node->Output(num_inputs) = false;
            return;
          }
        }

        // 所有类型匹配，设置最后一个输出为 true
        p_node->Output(num_inputs) = true;
      };
    });

// See [Borrowed IValue Outputs]
// 注册本地运算符函数，处理 static_runtime::VarTupleUnpack 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    static_runtime::VarTupleUnpack,
    static_runtime_VarTupleUnpack,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 static_runtime::VarTupleUnpack 操作符的模式
      if (!sr_schema_check(n, "static_runtime::VarTupleUnpack(...) -> ...")) {
        // 如果不符合，返回空指针
        return nullptr;
      }
      // 返回一个 lambda 函数，处理节点的输出
      return [](ProcessedNode* pnode) {
        size_t output_idx = 0;
        // 遍历节点的输入
        for (const auto idx : c10::irange(pnode->num_inputs())) {
          const auto& tuple = pnode->Input(idx);
          // 遍历元组的元素，并将每个元素转换为 BorrowedIValue 输出
          for (auto& elem : tuple.toTupleRef().elements()) {
            pnode->Output(output_idx) = createBorrowedIValue(elem);
            ++output_idx;
          }
        }
      };
    });

// 注册本地运算符函数，处理 aten::view 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::view,
    aten_view,
    [](Node* n) -> SROperator {
      // 检查节点是否匹配特定的 Torch 脚本模式
      if (!n->matches(torch::schema(
              "aten::view(Tensor(a) self, int[] size) -> (Tensor(a))"))) {
        // 如果不匹配，记录日志并且返回空指针
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 返回一个 lambda 函数，处理节点的输出
      return [](ProcessedNode* p_node) {
        // 获取输入张量和视图大小
        const auto& input = p_node->Input(0).toTensor();
        const auto size = p_node->Input(1).toIntList();
        // 执行张量的视图操作，并将结果设置为节点的输出
        p_node->Output(0) = at::native::view(input, size.vec());
      };
    });

// 注册本地运算符函数，处理 aten::size 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::size,
    aten_size,
    [](Node* n) -> SROperator {
      // 匿名函数，接受一个 Node* 参数，返回一个 SROperator 对象
      if (n->matches(
              torch::schema("aten::size(Tensor self, int dim) -> int"))) {
        // 如果 Node 对象匹配给定的 schema
        return [](ProcessedNode* p_node) {
          // 返回另一个匿名函数，接受一个 ProcessedNode* 参数
          const auto& input = p_node->Input(0).toTensor();
          // 获取第一个输入作为 Tensor 类型的引用
          auto dim = p_node->Input(1).toInt();
          // 获取第二个输入并转换为整数类型
          const auto ndim = input.dim();
          // 获取输入 Tensor 的维度数量
    
          if (dim < 0 || dim >= ndim) {
            // 如果指定的维度小于0或者超过最大维度数
            dim = c10::maybe_wrap_dim(dim, ndim);
            // 使用 maybe_wrap_dim 方法将维度包装为有效的值
          }
          p_node->Output(0) = input.sizes()[dim];
          // 设置输出节点的第一个输出为输入 Tensor 在指定维度上的大小
        };
      }
      if (n->matches(torch::schema("aten::size(Tensor self) -> int[]"))) {
        // 如果 Node 对象匹配给定的 schema
        return [](ProcessedNode* p_node) {
          // 返回另一个匿名函数，接受一个 ProcessedNode* 参数
          const auto& input = p_node->Input(0).toTensor();
          // 获取第一个输入作为 Tensor 类型的引用
          p_node->Output(0) = input.sizes();
          // 设置输出节点的第一个输出为输入 Tensor 的大小数组
        };
      }
      LogAndDumpSchema(n);
      // 如果以上两个条件都不匹配，则记录并转储当前 Node 的 schema
      return nullptr;
      // 返回空指针
    });
    // 匿名函数的定义结束
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::squeeze,  // 注册 squeeze 操作符的 Functor
    aten_squeeze,   // 定义该操作符的内部名称
    [](Node* n) -> SROperator {  // Lambda 函数，接受一个 Node* 参数，返回一个 SROperator 对象
      if (!n->matches(torch::schema(
              "aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)"))) {  // 检查节点 n 是否匹配 squeeze 操作的特定模式
        LogAndDumpSchema(n);  // 若不匹配，记录并输出节点的模式信息
        return nullptr;  // 返回空指针表示注册失败
      }

      return [](ProcessedNode* p_node) {  // 若匹配，返回一个处理节点的 Lambda 函数
        const auto& self = p_node->Input(0).toTensor();  // 获取输入节点的第一个参数，并转换为 Tensor 类型
        const auto dim = p_node->Input(1).toInt();  // 获取输入节点的第二个参数，并转换为整数类型
        p_node->Output(0) = at::native::squeeze(self, dim);  // 调用 squeeze 函数处理输入参数，并将结果写入到输出节点的第一个参数位置
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::split,  // 注册 split 操作符的 Functor
    aten_split,   // 定义该操作符的内部名称
    [](Node* n) -> SROperator {  // Lambda 函数，接受一个 Node* 参数，返回一个 SROperator 对象
      if (n->matches(torch::schema(
              "aten::split(Tensor(a -> *) self, int split_size, int dim=0) -> Tensor(a)[]"))) {  // 检查节点 n 是否匹配 split 操作的特定模式
        return [](ProcessedNode* p_node) {  // 若匹配，返回一个处理节点的 Lambda 函数
          const auto& self = p_node->Input(0).toTensor();  // 获取输入节点的第一个参数，并转换为 Tensor 类型
          const auto split_size = p_node->Input(1).toInt();  // 获取输入节点的第二个参数，并转换为整数类型
          const auto dim = p_node->Input(2).toInt();  // 获取输入节点的第三个参数，并转换为整数类型
          p_node->Output(0) = at::native::split(self, split_size, dim);  // 调用 split 函数处理输入参数，并将结果写入到输出节点的第一个参数位置
        };
      }

      if (n->matches(torch::schema(
              "aten::split(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> (Tensor[])"))) {  // 检查节点 n 是否匹配 split_with_sizes 操作的特定模式
        return [](ProcessedNode* p_node) {  // 若匹配，返回一个处理节点的 Lambda 函数
          const auto& self = p_node->Input(0).toTensor();  // 获取输入节点的第一个参数，并转换为 Tensor 类型
          const auto& split_sizes = p_node->Input(1).toIntList();  // 获取输入节点的第二个参数，并转换为整数列表类型
          const auto dim = p_node->Input(2).toInt();  // 获取输入节点的第三个参数，并转换为整数类型
          p_node->Output(0) = at::native::split_with_sizes(self, split_sizes.vec(), dim);  // 调用 split_with_sizes 函数处理输入参数，并将结果写入到输出节点的第一个参数位置
        };
      }

      LogAndDumpSchema(n);  // 若都不匹配，则记录并输出节点的模式信息
      return nullptr;  // 返回空指针表示注册失败
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::split_with_sizes,  // 注册 split_with_sizes 操作符的 Functor
    aten_split_with_sizes,   // 定义该操作符的内部名称
    [](Node* n) -> SROperator {  // Lambda 函数，接受一个 Node* 参数，返回一个 SROperator 对象
      if (!n->matches(torch::schema(
              "aten::split_with_sizes(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> Tensor(a)[]")) &&  // 检查节点 n 是否匹配 split_with_sizes 操作的第一种特定模式
          !n->matches(torch::schema(
              "aten::split_with_sizes(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> (Tensor[])"))) {  // 检查节点 n 是否匹配 split_with_sizes 操作的第二种特定模式
        LogAndDumpSchema(n);  // 若都不匹配，则记录并输出节点的模式信息
        return nullptr;  // 返回空指针表示注册失败
      }
      return [](ProcessedNode* p_node) {  // 若匹配，则返回一个处理节点的 Lambda 函数
        const auto& self = p_node->Input(0).toTensor();  // 获取输入节点的第一个参数，并转换为 Tensor 类型
        const auto& split_sizes = p_node->Input(1).toIntList();  // 获取输入节点的第二个参数，并转换为整数列表类型
        const auto dim = p_node->Input(2).toInt();  // 获取输入节点的第三个参数，并转换为整数类型
        p_node->Output(0) =
            at::native::split_with_sizes(self, split_sizes.vec(), dim);  // 调用 split_with_sizes 函数处理输入参数，并将结果写入到输出节点的第一个参数位置
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    static_runtime::select_tensor,  // 注册 select_tensor 操作符的 Functor
    aten_select_tensor,
    // 定义一个 lambda 函数，接受一个 Node* 类型参数 n，返回一个 SROperator 类型对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否符合指定的静态运行时模式
      if (!sr_schema_check(
              n,
              "static_runtime::select_tensor(Tensor(a) a, Tensor(b) b, bool use_b) -> Tensor(a|b)")) {
        // 如果不符合模式要求，返回空指针
        return nullptr;
      }
      // 返回另一个 lambda 函数，接受一个 ProcessedNode* 类型参数 p_node
      return [](ProcessedNode* p_node) {
        // 检查第三个输入是否为复制操作
        const auto did_copy = p_node->Input(2).toBool();
        // 断言第一个输入是 Tensor 类型
        DCHECK(p_node->Input(0).isTensor());
        // 如果进行了复制，则第二个输入也必须是 Tensor 类型
        DCHECK(!did_copy || p_node->Input(1).isTensor());
        // 根据是否复制，选择要赋值的 IValue 对象
        const IValue& assignFrom =
            did_copy ? p_node->Input(1) : p_node->Input(0);
        // 确保赋值来源与输出不是同一个对象
        TORCH_DCHECK_NE(&assignFrom, &p_node->Output(0));
        // 确保输出是空的，MemoryPlanner 应该已经清理过了
        DCHECK(p_node->Output(0).isNone());
        // 将输出设置为一个新的 IValue 对象，使用 borrow 方式创建 TensorBase
        p_node->Output(0) =
            IValue(c10::MaybeOwnedTraits<at::TensorBase>::createBorrow(
                assignFrom.toTensor()));
      };
    });
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::mul,
    aten_mul,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配给定的乘法操作的模式
      if (!n->matches(
              torch::schema("aten::mul.left_t(t[] l, int n) -> (t[])"))) {
        // 如果不匹配，则记录并输出节点的模式信息，并返回空指针
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 如果匹配，则返回一个 lambda 函数处理节点
      return [](ProcessedNode* pnode) {
        // 从处理后的节点中获取输入的列表和整数
        const auto& list = pnode->Input(0).toList();
        const auto n = pnode->Input(1).toInt();

        // 获取列表元素的类型
        auto list_type = list.elementType();
        // 创建一个新的泛型列表对象
        auto ret = c10::impl::GenericList(list_type);
        // 预先分配足够的空间以容纳结果列表的元素
        ret.reserve(list.size() * n);
        // 执行循环，将列表中的每个元素复制 n 次添加到结果列表中
        for (const auto i : c10::irange(n)) {
          (void)i; // 用于避免编译器未使用警告
          for (const auto& ival : list) {
            ret.push_back(ival);
          }
        }
        // 将结果列表设置为处理后的节点的输出
        pnode->Output(0) = ret;
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::sub,
    aten_sub,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配给定的减法操作的模式
      if (!n->matches(torch::schema("aten::sub.int(int a, int b) -> (int)"))) {
        // 如果不匹配，则记录并输出节点的模式信息，并返回空指针
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 如果匹配，则返回一个 lambda 函数处理节点
      return [](ProcessedNode* pnode) {
        // 从处理后的节点中获取输入的两个整数，并计算它们的差
        const auto a = pnode->Input(0).toInt();
        const auto b = pnode->Input(1).toInt();
        // 将计算结果设置为处理后的节点的输出
        pnode->Output(0) = a - b;
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::add,
    aten_add,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配给定的加法操作的模式
      if (n->matches(torch::schema("aten::add.t(t[] a, t[] b) -> (t[])"))) {
        // 如果匹配列表加法操作的模式，则返回一个 lambda 函数处理节点
        return [](ProcessedNode* pnode) {
          // 从处理后的节点中获取输入的两个列表，并将它们连接成一个新的列表
          const auto& a = pnode->Input(0).toList();
          const auto& b = pnode->Input(1).toList();
          auto ret = a.copy(); // 复制列表 a 的内容到新列表 ret
          ret.append(b); // 将列表 b 的内容追加到 ret 中
          // 将结果列表设置为处理后的节点的输出
          pnode->Output(0) = ret;
        };
      }

      // 如果不匹配列表加法操作的模式，则检查是否匹配整数加法操作的模式
      if (n->matches(torch::schema("aten::add.int(int a, int b) -> (int)"))) {
        // 如果匹配整数加法操作的模式，则返回一个 lambda 函数处理节点
        return [](ProcessedNode* pnode) {
          // 从处理后的节点中获取输入的两个整数，并计算它们的和
          const auto a = pnode->Input(0).toInt();
          const auto b = pnode->Input(1).toInt();
          // 将计算结果设置为处理后的节点的输出
          pnode->Output(0) = a + b;
        };
      }

      // 如果既不匹配列表加法操作的模式也不匹配整数加法操作的模式，则记录并输出节点的模式信息，并返回空指针
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::tensor_split,
    aten_tensor_split,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配给定的 tensor_split.indices 操作的模式
      if (n->matches(torch::schema(
              "aten::tensor_split.indices(Tensor(a -> *) self, int[] indices, int dim=0) -> Tensor(a)[]"))) {
        // 如果匹配，则返回一个 lambda 函数处理节点
        return [](ProcessedNode* pnode) {
          // 从处理后的节点中获取输入的张量、索引数组和维度，并调用相应的原生函数进行处理
          const auto& a = pnode->Input(0).toTensor();
          const auto& b = pnode->Input(1).toIntVector();
          const auto c = pnode->Input(2).toInt();
          // 将处理结果设置为处理后的节点的输出
          pnode->Output(0) = at::native::tensor_split(a, b, c);
        };
      }

      // 检查节点 n 是否匹配给定的 tensor_split.sections 操作的模式
      if (n->matches(torch::schema(
              "aten::tensor_split.sections(Tensor(a -> *) self, int sections, int dim=0) -> Tensor(a)[]"))) {
        // 如果匹配，则返回一个 lambda 函数处理节点
        return [](ProcessedNode* pnode) {
          // 从处理后的节点中获取输入的张量、SymInt 和维度，并调用相应的原生函数进行处理
          const auto& a = pnode->Input(0).toTensor();
          const auto b = pnode->Input(1).toSymInt();
          const auto c = pnode->Input(2).toInt();
          // 将处理结果设置为处理后的节点的输出
          pnode->Output(0) = at::native::tensor_split_sections_symint(a, b, c);
        };
      }

      // 如果既不匹配 tensor_split.indices 操作的模式也不匹配 tensor_split.sections 操作的模式，则记录并输出节点的模式信息，并返回空指针
      LogAndDumpSchema(n);
      return nullptr;
    });
    };
  }



// 如果节点匹配特定的 Torch 脚本模式
if (n->matches(torch::schema(
        "aten::tensor_split.tensor_indices_or_sections(Tensor(a -> *) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]"))) {
  // 返回一个 Lambda 表达式
  return [](ProcessedNode* pnode) {
    // 获取输入节点的第一个（自变量）作为 Tensor 对象
    const auto& a = pnode->Input(0).toTensor();
    // 获取输入节点的第二个（自变量）作为 Tensor 对象
    const auto& b = pnode->Input(1).toTensor();
    // 获取输入节点的第三个（自变量）作为整数
    const auto c = pnode->Input(2).toInt();
    // 设置输出节点的第一个输出为 tensor_split 函数的结果
    pnode->Output(0) = at::native::tensor_split(a, b, c);
  };
}
// 如果不匹配特定 Torch 脚本模式，则记录日志并转储模式
LogAndDumpSchema(n);
// 返回空指针表示未找到匹配的处理函数
return nullptr;
});

// 注册自定义操作符的函数对象，处理 torch 的 aten::Int 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::Int,
    aten_Int,
    [](Node* n) -> SROperator {
      // 检查节点是否符合特定的 torch 模式，如果不符合则记录日志并返回空指针
      if (!n->matches(torch::schema("aten::Int(Tensor a) -> int"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      // 返回一个 lambda 表达式，用于处理节点的计算
      return [](ProcessedNode* pnode) {
        // 获取节点的输入张量
        const auto& input = pnode->Input(0).toTensor();
        // 将张量的标量值转换为整数，并设置为节点的输出
        pnode->Output(0) = at::native::item(input).toInt();
      };
    });

// 查看 [Create owned refs for special values] 注册的自定义操作符函数对象
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    static_runtime::create_owned_ref,
    static_runtime_create_owned_ref,
    [](Node* n) -> SROperator {
      // 检查节点是否符合特定的 SR 模式，如果不符合则返回空指针
      if (!sr_schema_check(n, "static_runtime::create_owned_ref(...) -> ...")) {
        return nullptr;
      }
      // 返回一个 lambda 表达式，将节点的输入直接设置为输出
      return [](ProcessedNode* p_node) { p_node->Output(0) = p_node->Input(0); };
    });

namespace {
// 判断一个基本块是否没有输出
bool outputsEmpty(const Block* block) {
  return block->outputs().size() == 1 && block->outputs().at(0)->mustBeNone();
}

// 判断一个基本块是否为空
bool blockEmpty(const Block* block) {
  return block->nodes().begin() == block->nodes().end();
}

// 枚举类型，表示基本块的运行计划
enum class BlockRunPlan : int8_t {
  kRunOnlyTrueBlock,    // 仅运行真块
  kRunOnlyFalseBlock,   // 仅运行假块
  kRunBothBlocks,       // 同时运行两个块
  kRunNeitherBlock,     // 不运行任何块
};
} // namespace

// 注册 prim::If 操作符的函数对象
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::If,
    prim_If,
    });

namespace {

// 收集循环子块的输入值
std::vector<IValue> collectLoopSubBlockInputs(const ProcessedNode& p_node) {
  const auto num_inputs = p_node.num_inputs();
  TORCH_DCHECK_GE(num_inputs, 2);
  // 循环节点的前两个输入是最大迭代次数和初始条件，这里不收集它们，因为它们不是子块的输入
  const auto num_args = num_inputs - 2;

  std::vector<IValue> result;
  result.reserve(num_args + 1);
  // 循环子块的第一个参数总是循环计数器，初始值为零
  result.emplace_back(0);

  for (const auto i : c10::irange(num_args)) {
    // 收集循环子块的其余输入参数
    result.push_back(p_node.Input(2 + i));
  }

  return result;
}

} // namespace

namespace {
/*
  ForkedSubgraphSRLauncher 负责在静态运行时的新实例上执行分叉子图。
  一旦执行完成，将标记 future 完成，以指示 aten::wait() 继续执行。
*/
class TORCH_API ForkedSubgraphSRLauncher {
 public:
  ForkedSubgraphSRLauncher(
      std::shared_ptr<StaticModule> smodule,
      std::vector<IValue> args,
      c10::intrusive_ptr<Future> future,
      TaskLauncher launcher)
      : smodule_(std::move(smodule)),
        args_(std::move(args)),
        future_(std::move(future)),
        launcher_(std::move(launcher)) {}

  // 执行函数调用操作符
  void operator()() {
    try {
      // 创建静态运行时对象并异步运行子图
      StaticRuntime runtime(*smodule_);
      auto future_subgraph = runtime.runAsync(args_, {}, launcher_);
      // 等待子图执行完成并抛出异常（如果有）
      future_subgraph->waitAndThrow();
      // 标记 future 完成，将子图的值设置为 future 的值
      future_->markCompleted(future_subgraph->value());
    } catch (const std::exception& e) {
      // 如果出现异常，设置 future 的错误状态
      future_->setErrorIfNeeded(
          std::make_exception_ptr(c10::ivalue::Future::FutureError(e.what())));
    }

  } // end of operator()

}; // end of class ForkedSubgraphSRLauncher

} // end of namespace
    }
  }



  // 结束了两个嵌套的代码块
  // 这里标志着一个类的私有部分的结束



 private:
  std::shared_ptr<StaticModule> smodule_;
  // 一个共享指针，指向StaticModule类型的对象，用于管理模块的静态数据

  std::vector<IValue> args_;
  // 一个存储IValue类型对象的向量，用于保存参数

  c10::intrusive_ptr<Future> future_;
  // 一个c10库中的intrusive_ptr智能指针，指向Future类型的对象

  torch::jit::TaskLauncher launcher_;
  // torch::jit命名空间下的TaskLauncher类型对象，用于启动任务



  // 这段代码声明了一个类的私有成员变量，分别是smodule_、args_、future_和launcher_
  // 它们分别用于管理静态模块、存储参数、处理未来结果和启动任务
};

/*
  helper function to create a future on return type
  of the graph outputs. This function is utilized by
  prim::fork and aten::wait operations for async
  execution of subgraphs
*/
c10::intrusive_ptr<Future> createFutureTypeFromGraphOutput(
    std::shared_ptr<torch::jit::Graph> graph) {
  TypePtr return_type_;
  // 如果图的输出只有一个，返回类型为该输出的类型
  if (graph->outputs().size() == 1) {
    return_type_ = graph->outputs().at(0)->type();
  } else {
    // 如果图的输出有多个，创建一个元组类型来包含所有输出类型
    return_type_ = TupleType::create(
        fmap(graph->outputs(), [](const Value* v) { return v->type(); }));
  }
  // 创建一个具有指定返回类型的 Future 对象
  c10::intrusive_ptr<Future> future = c10::make_intrusive<Future>(return_type_);
  return future;
}
} // namespace

/*
  prim::fork forks the execution of a subgraph. It returns a future on which
  the corresponding aten::wait op waits until future is marked complete
  Current implementation creates a instance of StaticModule uses it to
  create StaticRuntime instances on the fly during runtime to handle the
  execution of forked subgraph. Async execution is handled by
  aten::ParallelThreadPoolNative threadpool.
*/
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::fork,
    prim_Fork,
    [](Node* node) -> SROperator {
      // 检查节点是否为 prim::fork 类型，如果不是则返回空指针
      if (!sr_schema_check_kind(node, prim::fork)) {
        return nullptr;
      }
      // 获取 fork 操作的子图
      auto forkedGraph = node->g(attr::Subgraph);
      // 内联化子图
      Inline(*forkedGraph);
      // 获取节点的静态运行时元数据
      auto sr_metadata = node->ival(getStaticRuntimeMetadataSymbol())
                             .toCustomClass<StaticRuntimeMetadata>();
      // 使用子图和静态运行时选项创建 StaticModule 实例
      auto smodule =
          std::make_shared<StaticModule>(forkedGraph, sr_metadata->get_opts());

      return [forkedGraph = std::move(forkedGraph),
              smodule = std::move(smodule)](ProcessedNode* p_node) {
        std::vector<IValue> args;
        args.reserve(p_node->num_inputs());
        // 将处理节点的输入添加到参数列表中
        for (const auto i : c10::irange(p_node->num_inputs())) {
          args.push_back(p_node->Input(i));
        }

        // 创建一个与子图输出类型对应的 Future
        c10::intrusive_ptr<Future> future =
            createFutureTypeFromGraphOutput(forkedGraph);
        // 将 Future 分配给处理节点的输出
        p_node->Output(0) = future;

        auto* metadata = p_node->metadata();
        DCHECK(metadata);
        auto* launcher = metadata->launcher();
        DCHECK(launcher);
        // 使用 StaticModule 和参数列表创建 ForkedSubgraphSRLauncher 实例
        ForkedSubgraphSRLauncher runtime_launcher(
            smodule, args, future, *launcher);
        // 调用 launcher 来执行运行时启动器
        (*launcher)(std::move(runtime_launcher));
      };
    });
/*
  aten::wait waits on the future (present in corresponding fork)
  to be executed. Once the execution is complete, the future is marked
  completed and wait execution continues.
*/
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::wait,
    aten_Wait,
    [](Node* node) -> SROperator {
      // aten::wait 操作等待对应的 Future 执行完成
      return [node](ProcessedNode* p_node) {
        // 将处理节点的输入（即待等待的 Future）标记为完成
        p_node->Output(0)->markCompleted();
      };
    });
    [](Node* n) -> SROperator {
        // Lambda function that takes a Node pointer `n` and returns an SROperator
        if (!sr_schema_check(n, "aten::wait(Future(t) self) -> t")) {
            // Check if the schema of node `n` matches "aten::wait(Future(t) self) -> t"
            return nullptr;
        }
        // If schema check passed, return a lambda function
        return [](ProcessedNode* p_node) {
            TORCH_INTERNAL_ASSERT(p_node->Input(0).isFuture());
            // Assert that the input at index 0 of `p_node` is a future
            auto future = p_node->Input(0).toFuture();
    
            // blocking call: waiting for the future to be completed
            future->waitAndThrow();
    
            TORCH_INTERNAL_ASSERT(future->completed());
            // Assert that the future has completed
            TORCH_INTERNAL_ASSERT(!future->hasError());
            // Assert that the future has no error
            TORCH_INTERNAL_ASSERT(future->hasValue());
            // Assert that the future has a value
    
            // If the future's value is not a tuple, assign it directly to the output
            if (!future->value().isTuple()) {
                p_node->Output(0) = future->value();
                return;
            }
            // If the future's value is a tuple, unpack its elements into `p_node`'s outputs
            auto& elems = future->value().toTupleRef().elements();
            TORCH_DCHECK_EQ(elems.size(), p_node->num_outputs());
            // Double check that the number of elements in the tuple matches `p_node`'s expected outputs
            for (const auto i : c10::irange(elems.size())) {
                p_node->Output(i) = elems[i];
            }
        };
    });
    // End of lambda function definition
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::Loop,  // 注册 Loop 操作符的操作函数
    prim_Loop,   // 将 prim::Loop 映射到 prim_Loop 函数
    [](Node* n) -> SROperator {  // Lambda 函数，接收 Node* 参数，返回 SROperator
      if (!sr_schema_check_kind(n, prim::Loop)) {  // 检查节点 n 的类型是否为 prim::Loop
        return nullptr;  // 如果不是，返回空指针
      }
      return [](ProcessedNode* p_node) {  // 返回另一个 Lambda 函数，接收 ProcessedNode* 参数
        const auto max_trip_count = p_node->Input(0).toInt();  // 从节点的第一个输入中获取最大迭代次数
        auto condition = p_node->Input(1).toBool();  // 从节点的第二个输入中获取循环条件

        auto* metadata = p_node->metadata();  // 获取节点的元数据指针
        DCHECK(metadata);  // 断言元数据不为空
        auto& block_runners = metadata->block_runners();  // 获取元数据中的 block_runners
        TORCH_DCHECK_EQ(block_runners.size(), 1);  // 断言 block_runners 的大小为 1
        auto& runner = block_runners[0];  // 获取第一个 block_runner

        auto args = collectLoopSubBlockInputs(*p_node);  // 收集循环子块的输入参数
        int64_t loop_count = 0;  // 初始化循环计数器

        while (condition && loop_count < max_trip_count) {  // 循环直到条件不满足或达到最大迭代次数
          auto output = runner(args);  // 运行 block_runner，并获取输出

          if (output.isTuple()) {  // 如果输出是元组
            auto& elems = output.toTupleRef().elements();  // 获取元组的元素引用
            DCHECK(elems.size() == args.size());  // 断言元素数量与参数数量相同
            for (const auto i : c10::irange(1, args.size())) {  // 遍历元素（从1开始，跳过第一个参数）
              args[i] = elems[i];  // 更新参数
            }
            condition = elems[0].toBool();  // 更新循环条件
          } else {
            condition = output.toBool();  // 如果输出不是元组，更新循环条件
          }
          args[0] = ++loop_count;  // 更新循环计数器并作为参数的第一个元素
        }

        const auto num_outputs = p_node->num_outputs();  // 获取节点的输出数量
        TORCH_DCHECK_EQ(args.size(), num_outputs + 1);  // 断言参数数量为输出数量加一
        for (const auto i : c10::irange(num_outputs)) {  // 遍历输出数量
          p_node->Output(i) = std::move(args[i + 1]);  // 将参数的值移动到节点的输出中
        }
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::CreateObject,  // 注册 CreateObject 操作符的操作函数
    prim_CreateObject,   // 将 prim::CreateObject 映射到 prim_CreateObject 函数
    [](Node* node) -> SROperator {  // Lambda 函数，接收 Node* 参数，返回 SROperator
      if (!sr_schema_check_kind(node, prim::CreateObject)) {  // 检查节点 node 的类型是否为 prim::CreateObject
        return nullptr;  // 如果不是，返回空指针
      }
      auto class_type = node->output()->type()->expect<ClassType>();  // 获取节点输出的类类型
      return [class_type = std::move(class_type)](ProcessedNode* pnode) {  // 返回 Lambda 函数，捕获类类型
        pnode->Output(0) = c10::ivalue::Object::create(  // 在节点的输出中创建一个对象
            c10::StrongTypePtr(class_type->compilation_unit(), class_type),  // 使用类类型的强类型指针
            class_type->numAttributes());  // 设置对象的属性数量
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::TupleIndex,  // 注册 TupleIndex 操作符的操作函数
    prim_TupleIndex,   // 将 prim::TupleIndex 映射到 prim_TupleIndex 函数
    [](Node* n) -> SROperator {  // Lambda 函数，接收 Node* 参数，返回 SROperator
      if (!sr_schema_check_kind(n, prim::TupleIndex)) {  // 检查节点 n 的类型是否为 prim::TupleIndex
        return nullptr;  // 如果不是，返回空指针
      }
      return [](ProcessedNode* pnode) {  // 返回 Lambda 函数，接收 ProcessedNode* 参数
        const auto& elems = pnode->Input(0).toTupleRef().elements();  // 获取输入的元组元素
        using c10::ssize;  // 使用 ssize
        const auto num_elems = ssize(elems);  // 获取元组元素数量
        const auto idx = pnode->Input(1).toInt();  // 获取索引值
        const auto norm_idx = normalizeIndex(idx, num_elems);  // 规范化索引值
        if (norm_idx < 0 || norm_idx >= num_elems) {  // 如果规范化后的索引超出范围
          throw std::out_of_range("Tuple index out of range");  // 抛出索引超出范围的异常
        }
        pnode->Output(0) = elems[norm_idx];  // 将元素赋给节点的输出
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::RaiseException,
    prim_RaiseException,
    [](Node* n) -> SROperator {
      // 匿名函数，接收一个指向Node的指针参数n，并返回一个SROperator
      if (!sr_schema_check_kind(n, prim::RaiseException)) {
        // 如果节点n不是RaiseException类型，返回空指针
        return nullptr;
      }
      // 如果节点n是RaiseException类型，返回另一个匿名函数
      return [](ProcessedNode* pnode) {
        // 匿名函数接收一个ProcessedNode类型的指针参数pnode
        const auto& message = pnode->Input(0).toStringRef();
        // 获取pnode的第一个输入，并将其转换为引用类型的字符串
        throw std::runtime_error(message);
        // 抛出std::runtime_error异常，异常消息为message
      };
    });
# 注册原生操作符的函数，处理 prim::Uninitialized 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::Uninitialized,
    prim_Uninitialized,
    [](Node* n) -> SROperator {
      # 检查节点 n 是否符合 prim::Uninitialized 的模式
      if (!sr_schema_check_kind(n, prim::Uninitialized)) {
        return nullptr;
      }
      # 返回一个 lambda 表达式，该表达式在处理节点时设置输出为未初始化状态
      return [](ProcessedNode* pnode) {
        pnode->Output(0) = IValue::uninitialized();
      };
    });

# 注册原生操作符的函数，处理 aten::format 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::format,
    aten_format,
    [](Node* n) -> SROperator {
      # 检查节点 n 是否符合指定的 aten::format 模式
      if (!sr_schema_check(n, "aten::format(str self, ...) -> str")) {
        return nullptr;
      }
      # 确保节点的输入非空
      TORCH_CHECK(!n->inputs().empty());
      # 返回一个 lambda 表达式，该表达式在处理节点时进行格式化操作
      return [](ProcessedNode* pnode) {
        const auto num_inputs = pnode->num_inputs();
        auto stack = boxInputs(*pnode);
        format(stack, num_inputs);
        TORCH_DCHECK_EQ(stack.size(), 1);
        pnode->Output(0) = std::move(stack[0]);
      };
    });

# 注册原生操作符的函数，处理 prim::device 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::device,
    prim_device,
    [](Node* n) -> SROperator {
      # 检查节点 n 是否符合 prim::device 的模式
      if (!sr_schema_check(n, "prim::device(Tensor a) -> Device")) {
        return nullptr;
      }
      # 返回一个 lambda 表达式，该表达式在处理节点时获取输入 Tensor 的设备信息并输出
      return [](ProcessedNode* pnode) {
        const auto& input = pnode->Input(0).toTensor();
        pnode->Output(0) = input.device();
      };
    });

# 注册原生操作符的函数，处理 prim::dtype 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::dtype,
    prim_dtype,
    [](Node* n) -> SROperator {
      # 检查节点 n 是否符合 prim::dtype 的模式
      if (!sr_schema_check_kind(n, prim::dtype)) {
        return nullptr;
      }
      # 返回一个 lambda 表达式，该表达式在处理节点时获取输入 Tensor 的数据类型并输出
      return [](ProcessedNode* pnode) {
        const auto& input = pnode->Input(0).toTensor();
        pnode->Output(0) = static_cast<int64_t>(input.scalar_type());
      };
    });

# 注册原生操作符的函数，处理 aten::dim 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::dim,
    aten_dim,
    [](Node* n) -> SROperator {
      # 检查节点 n 是否符合指定的 aten::dim 模式
      if (!sr_schema_check(n, "aten::dim(Tensor self) -> int")) {
        return nullptr;
      }
      # 返回一个 lambda 表达式，该表达式在处理节点时获取输入 Tensor 的维度并输出
      return [](ProcessedNode* pnode) {
        const auto& input = pnode->Input(0).toTensor();
        pnode->Output(0) = input.dim();
      };
    });

# 注册原生操作符的函数，处理 aten::__not__ 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::__not__,
    aten_not,
    [](Node* n) -> SROperator {
      # 检查节点 n 是否符合指定的 aten::__not__ 模式
      if (!sr_schema_check(n, "aten::__not__(bool self) -> bool")) {
        return nullptr;
      }
      # 返回一个 lambda 表达式，该表达式在处理节点时对布尔值取反并输出
      return [](ProcessedNode* pnode) {
        auto input = pnode->Input(0).toBool();
        pnode->Output(0) = !input;
      };
    });

# 注册原生操作符的函数，处理 aten::Bool 操作符
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::Bool,
    aten_Bool,
    // 匿名函数，接受一个 Node* 参数并返回一个 SROperator 对象
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配 "aten::Bool.Tensor(Tensor a) -> bool" 模式
      if (n->matches(torch::schema("aten::Bool.Tensor(Tensor a) -> bool"))) {
        // 如果匹配，返回一个处理函数，将输入的第一个张量转换为非零布尔值
        return [](ProcessedNode* pnode) {
          const auto& input = pnode->Input(0).toTensor();  // 获取输入张量
          pnode->Output(0) = at::native::is_nonzero(input);  // 计算非零布尔值并存储到输出中
        };
      }
      // 检查节点 n 是否匹配 "aten::Bool.int(int a) -> bool" 模式
      if (n->matches(torch::schema("aten::Bool.int(int a) -> bool"))) {
        // 如果匹配，返回一个处理函数，将输入的整数转换为布尔值
        return [](ProcessedNode* pnode) {
          const auto input = pnode->Input(0).toInt();  // 获取输入整数
          pnode->Output(0) = static_cast<bool>(input);  // 转换为布尔值并存储到输出中
        };
      }
      // 检查节点 n 是否匹配 "aten::Bool.float(float a) -> bool" 模式
      if (n->matches(torch::schema("aten::Bool.float(float a) -> bool"))) {
        // 如果匹配，返回一个处理函数，将输入的浮点数转换为布尔值
        return [](ProcessedNode* pnode) {
          const auto input = pnode->Input(0).toDouble();  // 获取输入浮点数
          pnode->Output(0) = static_cast<bool>(input);  // 转换为布尔值并存储到输出中
        };
      }
      // 如果以上条件都不匹配，记录并转储节点的模式信息
      LogAndDumpSchema(n);
      // 返回空指针，表示未找到匹配的处理函数
      return nullptr;
    });
// 注册自定义的原生操作符处理函数，用于判断张量是否在 CUDA 设备上
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::is_cuda,
    prim_is_cuda,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 "prim::is_cuda(Tensor a) -> bool" 的模式
      if (!sr_schema_check(n, "prim::is_cuda(Tensor a) -> bool")) {
        return nullptr;
      }
      // 返回处理节点的函数，获取输入张量并输出其是否在 CUDA 上的布尔值
      return [](ProcessedNode* pnode) {
        const auto& input = pnode->Input(0).toTensor();
        pnode->Output(0) = input.is_cuda();
      };
    });

// 注册自定义的原生操作符处理函数，用于转换张量为列表形式
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::tolist,
    prim_tolist,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 prim::tolist 的模式
      if (!sr_schema_check_kind(n, prim::tolist)) {
        return nullptr;
      }
      // 返回处理节点的函数，将输入张量、维度和元素类型转换为列表形式
      return [](ProcessedNode* pnode) {
        const auto& input = pnode->Input(0).toTensor();
        const auto dim = pnode->Input(1).toInt();
        const auto elem_type = pnode->Input(2).toInt();
        std::vector<IValue> stack{input, dim, elem_type};
        toList(stack); // 将 stack 中的元素转换为列表形式
        TORCH_DCHECK_EQ(stack.size(), 1); // 断言 stack 的大小为 1
        pnode->Output(0) = std::move(stack[0]); // 输出转换后的列表形式
      };
    });

// 注册自定义的原生操作符处理函数，处理 IfThenElse 操作
// 见 [Borrowed IValue Outputs] 注释
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::IfThenElse,
    prim_IfThenElse,
    [](Node* n) -> SROperator {
      // 检查节点是否符合 prim::IfThenElse 的模式
      if (!sr_schema_check_kind(n, prim::IfThenElse)) {
        return nullptr;
      }
      // 返回处理节点的函数，根据条件选择输出的 Borrowed IValue
      return [](ProcessedNode* pnode) {
        const auto condition = pnode->Input(0).toBool();
        pnode->Output(0) = condition ? createBorrowedIValue(pnode->Input(1))
                                     : createBorrowedIValue(pnode->Input(2));
      };
    });

// 注册自定义的原生操作符处理函数，用于获取张量的长度
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::len,
    aten_len,
    [](Node* n) -> SROperator {
      // 检查节点 n 是否匹配 aten::len.t 或 aten::len.any 的模式，返回一个 lambda 表达式
      if (n->matches(torch::schema("aten::len.t(t[] a) -> int")) ||
          n->matches(torch::schema("aten::len.any(Any[] a) -> int"))) {
        // 返回一个 lambda 表达式，计算输入列表的长度并输出到处理节点的第一个输出端口
        return [](ProcessedNode* pnode) {
          const auto list = pnode->Input(0).toListRef();
          const int64_t size = list.size();
          pnode->Output(0) = size;
        };
      }
      // 检查节点 n 是否匹配 aten::len.Tensor 的模式，返回一个 lambda 表达式
      if (n->matches(torch::schema("aten::len.Tensor(Tensor t) -> int"))) {
        // 返回一个 lambda 表达式，计算张量的第一维度大小并输出到处理节点的第一个输出端口
        return [](ProcessedNode* pnode) {
          const auto& t = pnode->Input(0).toTensor();
          TORCH_CHECK(t.dim() > 0);  // 检查张量的维度是否大于0
          pnode->Output(0) = t.sizes()[0];  // 将第一维度大小输出到处理节点的第一个输出端口
        };
      }
      // 检查节点 n 是否匹配 aten::len.str 的模式，返回一个 lambda 表达式
      if (n->matches(torch::schema("aten::len.str(str s) -> int"))) {
        // 返回一个 lambda 表达式，计算字符串的长度并输出到处理节点的第一个输出端口
        return [](ProcessedNode* pnode) {
          const auto& string = pnode->Input(0).toStringRef();
          pnode->Output(0) = static_cast<int64_t>(string.size());  // 将字符串长度输出到处理节点的第一个输出端口
        };
      }
      // 检查节点 n 是否匹配各种字典类型的模式，返回一个 lambda 表达式
      if (n->matches(
              torch::schema("aten::len.Dict_str(Dict(str, t) self) -> int")) ||
          n->matches(
              torch::schema("aten::len.Dict_int(Dict(int, t) self) -> int")) ||
          n->matches(torch::schema(
              "aten::len.Dict_bool(Dict(bool, t) self) -> int")) ||
          n->matches(torch::schema(
              "aten::len.Dict_float(Dict(float, t) self) -> int")) ||
          n->matches(torch::schema(
              "aten::len.Dict_complex(Dict(complex, t) self) -> int")) ||
          n->matches(torch::schema(
              "aten::len.Dict_Tensor(Dict(Tensor, t) self) -> int"))) {
        // 返回一个 lambda 表达式，计算字典的大小并输出到处理节点的第一个输出端口
        return [](ProcessedNode* pnode) {
          const auto& dict = pnode->Input(0).toGenericDict();
          pnode->Output(0) = static_cast<int64_t>(dict.size());  // 将字典的大小输出到处理节点的第一个输出端口
        };
      }
      // 如果节点 n 没有匹配任何已知模式，记录并输出节点的模式信息
      LogAndDumpSchema(n);
      // 返回空指针，表示没有适合的操作符
      return nullptr;
    });
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::IntImplicit,  // 注册名为 aten::IntImplicit 的原生运算符处理函数
    aten_IntImplicit,   // 使用 aten_IntImplicit 作为注册函数的名称
    [](Node* n) -> SROperator {  // Lambda 表达式，接受一个 Node 指针 n，返回一个 SROperator
      if (!n->matches(torch::schema("aten::IntImplicit(Tensor a) -> int"))) {
        LogAndDumpSchema(n);  // 如果节点 n 的模式匹配失败，则记录并输出其模式信息
        return nullptr;        // 返回空指针
      }
      return [](ProcessedNode* pnode) {  // 返回另一个 Lambda 表达式，处理已处理的节点 pnode
        const auto& tensor = pnode->Input(0).toTensor();  // 获取输入的第一个张量 tensor
        // JIT 在需要梯度时会进行检查，但在这里我们跳过，因为 SR 只用于推断
        if (!tensor.sizes().empty()) {  // 如果张量 tensor 的维度不为空
          throw std::runtime_error(
              "Cannot convert a tensor of dimension > 0 to scalar");  // 抛出运行时错误
        }
        if (!isIntegralType(tensor.scalar_type(), /*includeBool=*/false)) {  // 如果 tensor 的数据类型不是整数类型（不包括布尔类型）
          std::stringstream ss;
          ss << "Cannot input a tensor of type " << tensor.scalar_type()
             << " as an integral argument";  // 构建错误消息
          throw std::runtime_error(ss.str());  // 抛出运行时错误
        }
        pnode->Output(0) = at::native::item(tensor).toInt();  // 将处理节点的输出设置为 tensor 的整数值
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::select,  // 注册名为 aten::select 的原生运算符处理函数
    aten_select,   // 使用 aten_select 作为注册函数的名称
    [](Node* n) -> SROperator {  // Lambda 表达式，接受一个 Node 指针 n，返回一个 SROperator
      if (!n->matches(torch::schema(
              "aten::select(Tensor(a) self, int dim, int index) -> Tensor(a)"))) {
        LogAndDumpSchema(n);  // 如果节点 n 的模式匹配失败，则记录并输出其模式信息
        return nullptr;        // 返回空指针
      }
      return [](ProcessedNode* pnode) {  // 返回另一个 Lambda 表达式，处理已处理的节点 pnode
        const auto& self = pnode->Input(0).toTensor();  // 获取输入的第一个张量 self
        const auto dim = pnode->Input(1).toInt();       // 获取输入的第二个参数 dim，并转换为整数
        const auto index = pnode->Input(2).toInt();     // 获取输入的第三个参数 index，并转换为整数
        pnode->Output(0) = at::native::select(self, dim, index);  // 将处理节点的输出设置为 select 操作的结果
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::reshape_as,  // 注册名为 aten::reshape_as 的原生运算符处理函数
    aten_reshape_as,   // 使用 aten_reshape_as 作为注册函数的名称
    [](Node* n) -> SROperator {  // Lambda 表达式，接受一个 Node 指针 n，返回一个 SROperator
      if (!n->matches(torch::schema(
              "aten::reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)"))) {
        LogAndDumpSchema(n);  // 如果节点 n 的模式匹配失败，则记录并输出其模式信息
        return nullptr;        // 返回空指针
      }
      return [](ProcessedNode* pnode) {  // 返回另一个 Lambda 表达式，处理已处理的节点 pnode
        const auto& self = pnode->Input(0).toTensor();   // 获取输入的第一个张量 self
        const auto& other = pnode->Input(1).toTensor();  // 获取输入的第二个张量 other
        pnode->Output(0) = at::native::reshape(self, other.sizes());  // 将处理节点的输出设置为将 self 重塑成与 other 相同形状的张量
      };
    });

} // namespace torch::jit  // 结束 torch::jit 命名空间
```