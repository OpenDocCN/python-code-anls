# `.\pytorch\torch\csrc\jit\passes\value_refinement_utils.cpp`

```
// 引入头文件 <c10/util/irange.h> 和 <torch/csrc/jit/passes/value_refinement_utils.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/passes/value_refinement_utils.h>

// 命名空间声明：torch::jit
namespace torch {
namespace jit {

// [value refinement algorithm]

// 当进行像 `cond = len(x) == 4` 或 `cond = len(x) != 4` 这样的比较时，
// `cond` 值携带关于 `x` 长度的信息（精炼）。
// 当 `cond` 用作 if 语句的条件时，它携带的 true 值的信息可以插入到 true 分支中，
// false 值的信息同样适用于 false 分支。
// 对于像 `y = len(x) if len(x) == 1 else 1` 这样的情况，在 true 分支中我们可以
// 用 1 替换 len(x)，因为 `len(x) == 1` 的 true 精炼将存在于 true 分支中。
// 另外，我们可以优化类似于：
// if len(x) != 4:
//    raise Exception(...)
// return len(x)
// 因为 true 分支总是抛出异常，false 分支中存在的任何精炼将存在于 if 节点所有者的块中。
// 我们还可以通过取两个不同布尔值在 if 节点连接处的精炼的交集来合并通过 if 节点的两个布尔值携带的精炼。
// if cond:
//    z = len(x) == 4 and len(y) == 5
// else:
//    z = len(x) == 4
// 这里，z 的 true 值将精炼 len(x) 为 4，但不会影响 len(y)。
// 如果代码写成：
// if cond:
//    z = len(x) == 4 and len(y) == 5
// else:
//    z = False
// 那么 z 的 true 值将精炼 x 和 y，因为如果 z 是 true，它必须来自 true 分支。
// 使用 `and` 或 `or` 的代码将类似地转换成这样的形式。此外，`cond` 上存在的任何 True 精炼也可以与 if 节点的 True 输出值关联。

// 精炼的交集是那些在两个精炼中都存在且精炼为相同长度的 Value*
// 例如：
// if cond:
//    x = len(a) == 4 and len(b) == 5
// else:
//    x = len(a) == 4
// 对于节点的输出 x，我们取每个块输出上存储的精炼的交集，结果将只包含 len(a) == 4 的精炼。
ListRefinement intersectRefinements(
    const ListRefinement& ref1, // 输入精炼 ref1
    const ListRefinement& ref2) { // 输入精炼 ref2
  ListRefinement out; // 输出精炼 out
  for (const auto& pair : ref1) { // 遍历 ref1 中的每对键值对
    auto val2 = ref2.find(pair.first); // 在 ref2 中查找键为 pair.first 的值
    if (val2 != ref2.end() && val2->second == pair.second) { // 如果在 ref2 中找到了对应的键值且值相等
      out[pair.first] = pair.second; // 将该键值对添加到输出精炼 out 中
    }
  }
  return out; // 返回输出精炼 out
}

// 要进行并集操作，只需从两个输入中获取所有精炼即可。
// 我们不需要担心长度精炼的冲突，因为像 `if len(x) == 4 and len(x) == 5` 这样的路径永远不会被执行。
// 例如：
// if len(a) == 5:
//     x = len(b) == 4
// else:
//     x = False
// 对于输出 x 的值，如果条件为 true，那么在 true 分支中存在的精炼也必须是 true，
// 因此我们取 `len(a) == 5` 和 `len(b) == 4` 的并集，并将它们分配给输出 x 值的 true 精炼。
// 这在 `and` 或 `or` 布尔表达式的展开中是非常常见的模式。
// 合并两个 ListRefinement 对象，并返回结果
ListRefinement unionRefinements(
    const ListRefinement& ref1,
    const ListRefinement& ref2) {
  // 将 ref1 复制给 out
  ListRefinement out = ref1;
  // 将 ref2 中的所有元素插入到 out 中
  out.insert(ref2.begin(), ref2.end());
  // 返回合并后的结果
  return out;
}

// 处理 if 节点的细化操作，更新当前块及相关数据的细化信息
void joinIfRefinements(
    Node* if_node,
    std::unordered_set<Block*>& throwing_blocks,
    ListRefinement& curr_block_refinements,
    ListRefinement& true_block_refinements,
    ListRefinement& false_block_refinements,
    std::unordered_map<Value*, BooleanRefinementMapping>&
        boolean_value_refinements) {
  // 获取 if_node 的 IfView
  IfView if_n(if_node);
  // 获取拥有当前 if 节点的块
  Block* b = if_node->owningBlock();

  // 检查真假分支块是否会抛出异常
  bool true_block_throws = throwing_blocks.count(if_n.thenBlock());
  bool false_block_throws = throwing_blocks.count(if_n.elseBlock());

  // 如果真假分支块有一个抛出异常，则处理其细化信息
  if (true_block_throws || false_block_throws) {
    // 如果真假分支均抛出异常，将当前块加入抛出异常的块集合并返回
    if (true_block_throws && false_block_throws) {
      throwing_blocks.insert(b);
      return;
    }
    // 根据抛出异常的情况，更新当前块的细化信息
    if (true_block_throws) {
      curr_block_refinements.insert(
          false_block_refinements.begin(), false_block_refinements.end());
    } else {
      curr_block_refinements.insert(
          true_block_refinements.begin(), true_block_refinements.end());
    }
    // 获取非抛出异常的块，并更新布尔输出的细化映射信息
    Block* non_throwing_block =
        true_block_throws ? if_node->blocks().at(1) : if_node->blocks().at(0);
    for (const auto i : c10::irange(if_n.outputs().size())) {
      if (boolean_value_refinements.count(
              non_throwing_block->outputs().at(i))) {
        boolean_value_refinements[if_node->outputs().at(i)] =
            boolean_value_refinements[non_throwing_block->outputs().at(i)];
      }
    }
    return;
  }

  // 处理布尔类型的输出值细化信息
  for (const auto i : c10::irange(if_n.outputs().size())) {
    // 如果输出值不是布尔类型，则直接返回
    if (!(if_n.outputs().at(i)->type() == BoolType::get())) {
      return;
    }
    // 获取 if 节点的真假输出值
    Value* true_v = if_n.thenOutputs().at(i);
    Value* false_v = if_n.elseOutputs().at(i);

    // 如果真假输出值都没有细化信息，并且也不是常量布尔值，则返回
    if (!boolean_value_refinements.count(true_v) &&
        !boolean_value_refinements.count(false_v) &&
        !constant_as<bool>(true_v) && !constant_as<bool>(false_v)) {
      return;
    }

    // 如果其中一个块有常量布尔输出值，根据情况更新细化信息
    // 如果两个块都没有常量布尔输出值，则取布尔输出的细化交集
    // 这部分处理说明了如何处理布尔输出的细化信息
  }
}
    // 判断常量 true_v 是否能够转换为布尔类型
    if (auto maybe_bool = constant_as<bool>(true_v)) {
      // 如果能转换并且结果为 true
      if (*maybe_bool) {
        // 使用 false_v 的假修饰与 false_block_refinements 的联合修饰来创建假修饰映射
        out = BooleanRefinementMapping::FalseRefinements(unionRefinements(
            boolean_value_refinements[false_v].false_refine(),
            false_block_refinements));
      } else {
        // 否则，使用 false_v 的真修饰与 false_block_refinements 的联合修饰来创建真修饰映射
        out = BooleanRefinementMapping::TrueRefinements(unionRefinements(
            boolean_value_refinements[false_v].true_refine(),
            false_block_refinements));
      }
    } else if (auto maybe_bool = constant_as<bool>(false_v)) {
      // 如果常量 false_v 能够转换为布尔类型
      if (*maybe_bool) {
        // 使用 true_v 的假修饰与 true_block_refinements 的联合修饰来创建假修饰映射
        out = BooleanRefinementMapping::FalseRefinements(unionRefinements(
            boolean_value_refinements[true_v].false_refine(),
            true_block_refinements));
      } else {
        // 否则，使用 true_v 的真修饰与 true_block_refinements 的联合修饰来创建真修饰映射
        out = BooleanRefinementMapping::TrueRefinements(unionRefinements(
            boolean_value_refinements[true_v].true_refine(),
            true_block_refinements));
      }
    } else if (
        // 如果 true_v 和 false_v 在布尔值修饰映射中均有定义
        boolean_value_refinements.count(true_v) &&
        boolean_value_refinements.count(false_v)) {
      // 对 true_v 和 false_v 的布尔值修饰映射进行交集操作，并赋值给 out
      out = boolean_value_refinements[true_v].intersectBooleanRefinementMapping(
          boolean_value_refinements[false_v]);
    }
    // 将计算得到的 out 映射赋值给 if_n.outputs().at(i) 的布尔值修饰映射
    boolean_value_refinements[if_n.outputs().at(i)] = out;
// 在命名空间"jit"中定义一个函数handleCommonRefinentOperators，用于处理常见的操作符节点
bool handleCommonRefinentOperators(
    Node* n,  // 参数n表示当前处理的节点
    std::unordered_set<Block*>& throwing_blocks,  // 引发异常的块的集合
    std::unordered_map<Value*, BooleanRefinementMapping>& info  // 存储值和布尔精化映射的字典
) {
  if (n->kind() == prim::RaiseException) {  // 如果节点的操作符类型是prim::RaiseException
    throwing_blocks.insert(n->owningBlock());  // 将节点所属的块加入到引发异常的块集合中
    return true;  // 返回true，表示操作成功处理
  }
  if (n->kind() == aten::__not__ &&  // 如果节点的操作符类型是aten::__not__，且
      n->inputs().at(0)->type()->cast<BoolType>()) {  // 输入的第一个参数是布尔类型
    // __not__(inp) -> reverse refinements
    if (info.count(n->input())) {  // 如果info字典中包含节点n的输入
      auto& input_ref = info[n->input()];  // 获取节点n的输入对应的布尔精化映射
      // 将节点n的输出设置为反向精化映射
      info[n->output()] = BooleanRefinementMapping(
          input_ref.false_refine(), input_ref.true_refine());
    }
    return true;  // 返回true，表示操作成功处理
  }
  if (n->matches("aten::eq(bool a, bool b) -> bool") ||  // 如果节点匹配等于操作符或者不等于操作符
      (n->matches("aten::ne(bool a, bool b) -> bool"))) {
    for (size_t const_index : {0, 1}) {  // 遍历常量索引 {0, 1}
      if (n->input(const_index)->node()->kind() != prim::Constant) {
        continue;  // 如果输入的常量不是prim::Constant类型，则继续下一次循环
      }
      auto const_input = constant_as<bool>(n->input(const_index)).value();  // 获取布尔常量的值
      auto non_const_input = n->input(1 - const_index);  // 获取非常量输入
      if (!info.count(non_const_input)) {
        continue;  // 如果info字典中不包含非常量输入，则继续下一次循环
      }
      // 根据常量输入值设置输出的布尔精化映射
      auto& input_ref = info[non_const_input];
      if ((!const_input && n->kind() == aten::eq) ||
          (const_input && n->kind() == aten::ne)) {
        // value == False / value != True -> equivalent to __not__ value
        // value == True / value != False -> equivalent to value
        info[n->output()] = BooleanRefinementMapping(
            input_ref.false_refine(), input_ref.true_refine());
      } else {
        info[n->output()] = BooleanRefinementMapping(
            input_ref.true_refine(), input_ref.false_refine());
      }
    }
    return true;  // 返回true，表示操作成功处理
  }
  return false;  // 如果未匹配到任何处理操作，则返回false
}

// 结束命名空间"torch"
} // namespace torch
// 结束命名空间"jit"
} // namespace jit
```