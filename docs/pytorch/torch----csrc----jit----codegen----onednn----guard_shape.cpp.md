# `.\pytorch\torch\csrc\jit\codegen\onednn\guard_shape.cpp`

```
//! [ Note -- prepareFusionGroupAndGuardOutputs implementation ]
//! shamelessly copying code from NNC (tensorexpr_fuser)  with very little
//! modification, original code at:
//! `torch/csrc/jit/passes/tensorexpr_fuser.cpp:prepareFusionGroupAndGuardOutputs`
//!
//! We have the assumption that LLGA does not have operators
//! depending on the content of the tensor.
void prepareFusionGroupAndGuardOutputs(Block* block) {
  // 初始化一个空的节点指针向量，用于存储找到的融合组节点
  std::vector<Node*> fusion_groups;
  // 遍历当前块中的所有节点
  for (Node* n : block->nodes()) {
    // 遍历每个节点的子块
    for (Block* b : n->blocks()) {
      // 递归调用准备函数以处理子块
      prepareFusionGroupAndGuardOutputs(b);
    }
    // 如果当前节点的类型是 prim::oneDNNFusionGroup
    if (n->kind() == prim::oneDNNFusionGroup) {
      // 将找到的融合组节点加入到融合组节点向量中
      fusion_groups.push_back(n);
    }
  }
  // 对每个找到的融合组节点执行类型守卫插入操作
  for (Node* fusion_group : fusion_groups) {
    // TODO: 添加进一步的优化传递以移除仅在大小中使用的输出，
    // 参考 `torch/csrc/jit/passes/tensorexpr_fuser.cpp:removeOutputsUsedOnlyInSize`
    // removeOutputsUsedOnlyInSize(fusion_group);
    // 在融合组节点中插入类型守卫，保证其类型为 TensorTypePtr，守卫类型为 prim::oneDNNFusionGuard
    insertTypeGuard(
        fusion_group,
        [](const TensorTypePtr& t) { return t; },
        prim::oneDNNFusionGuard);
  }
}
```