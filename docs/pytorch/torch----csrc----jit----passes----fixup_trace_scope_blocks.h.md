# `.\pytorch\torch\csrc\jit\passes\fixup_trace_scope_blocks.h`

```py
// 一旦跟踪完成，我们得到一个结构不完整的图形，其中插入了块。
// 示例：
//
// graph(%self : ClassType<Module>,
//       %input.1 : Float(3, 4)):
//   %1 : ClassType<Module> = prim::GetAttr[name="relu1"](%self)
//   %2 : ClassType<Module> = prim::GetAttr[name="relu2"](%self)
//   %3 : ClassType<Module> = prim::GetAttr[name="rrr"](%2)
//    = prim::TracedModuleForward[scope="__module.relu1"]()
//     block0():
//       %input : Float(3, 4) = aten::relu(%input.1),
//       -> ()
//    = prim::TracedModuleForward[scope="__module.relu2"](),
//     block0():
//        = prim::TracedModuleForward[scope="__module.relu2.rrr"](),
//         block0():
//           %6 : Float(3, 4) = aten::relu(%input),
//           -> ()
//       -> ()
//   return (%6)
//
// 在这个过程中，我们执行以下操作：
//   1) 将值定义提升到尽可能高的作用域，以确保它们支配所有使用。例如，上述图中的 `input` 需要提升到顶层块，以便其在第二个 `relu` 运算符中的使用被支配。
//   2) 对块进行 Lambda 提升。这确保了每个作用域内使用的所有值都被其定义捕获。
//   3) 将作用域块转换为其相应模块上的方法，并将 TracedModuleForward 节点转换为对这些方法的 CallMethod 节点。
//
// 然后，我们将得到一个结构完整、包含正确方法调用的图形。
TORCH_API void FixupTraceScopeBlocks(
    std::shared_ptr<Graph>& graph,
    Module* self);
```