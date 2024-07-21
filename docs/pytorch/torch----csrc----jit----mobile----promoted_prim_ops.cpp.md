# `.\pytorch\torch\csrc\jit\mobile\promoted_prim_ops.cpp`

```
void tupleIndex(Stack& stack) {
    // 从堆栈中弹出索引值，转换为 int64_t 类型
    int64_t index = pop(stack).toInt();
    // 从堆栈中弹出一个元组对象
    auto tuple = pop(stack).toTuple();
    // 标准化索引值，确保在有效范围内
    auto norm_index = normalizeIndex(index, tuple->elements().size());
    // 如果标准化后的索引超出了元组的范围，抛出异常
    if (norm_index < 0 ||
        norm_index >= static_cast<int64_t>(tuple->elements().size())) {
      throw std::out_of_range("Tuple list index out of range");
    }
    // 将元组中指定索引的元素推送回堆栈
    stack.emplace_back(tuple->elements()[norm_index]);
}

void raiseException(Stack& stack) {
    // 抛出异常，异常信息为堆栈顶部元素的字符串表示
    throw JITException(pop(stack).toStringRef());
}

void raiseExceptionWithMessage(Stack& stack) {
    // 从堆栈中弹出可选的字符串类名
    std::optional<std::string> qualified_class_name =
        pop(stack).toOptional<std::string>();
    // 从堆栈中弹出字符串消息
    std::string message;
    pop(stack, message);
    // 抛出带有消息和可选类名的异常
    throw JITException(message, qualified_class_name);
}

void is(Stack& stack) {
    // 从堆栈中弹出两个对象，并检查第一个对象是否与第二个对象相同
    IValue self, obj;
    pop(stack, self, obj);
    push(stack, self.is(obj));
}

void unInitialized(Stack& stack) {
    // 推送一个未初始化的 IValue 对象到堆栈
    push(stack, IValue::uninitialized());
}

void isNot(Stack& stack) {
    // 从堆栈中弹出两个对象，并检查第一个对象是否与第二个对象不同
    IValue self, obj;
    pop(stack, self, obj);
    push(stack, !self.is(obj));
}

void aten_format(Stack& stack) {
    // 从堆栈中弹出一个整数，调用 format 函数处理堆栈中的输入
    size_t num_inputs = pop(stack).toInt();
    format(stack, num_inputs);
}

void size(Stack& stack) {
    // 从堆栈中弹出一个张量对象，并推送其尺寸（sizes）的向量到堆栈
    auto t = std::move(pop(stack)).toTensor();
    pack(stack, t.sizes().vec());
}

void sym_size(Stack& stack) {
    // 从堆栈中弹出一个张量对象，并推送其符号尺寸（sym_sizes）的向量到堆栈
    auto t = std::move(pop(stack)).toTensor();
    pack(stack, t.sym_sizes().vec());
}

void sym_size_int(Stack& stack) {
    // 从堆栈中弹出一个整数作为维度索引，从堆栈中弹出一个张量对象，并推送指定维度的符号尺寸到堆栈
    auto dim = pop(stack).toInt();
    auto t = pop(stack).toTensor();
    push(stack, t.sym_sizes()[dim]);
}

void sym_stride_int(Stack& stack) {
    // 从堆栈中弹出一个整数作为维度索引，从堆栈中弹出一个张量对象，并推送指定维度的符号步幅到堆栈
    auto dim = pop(stack).toInt();
    auto t = pop(stack).toTensor();
    push(stack, t.sym_strides()[dim]);
}

void sym_numel(Stack& stack) {
    // 从堆栈中弹出一个张量对象，并推送其符号元素数（sym_numel）到堆栈
    auto t = std::move(pop(stack)).toTensor();
    push(stack, t.sym_numel());
}

void sym_storage_offset(Stack& stack) {
    // 从堆栈中弹出一个张量对象，并推送其符号存储偏移（sym_storage_offset）到堆栈
    auto t = std::move(pop(stack)).toTensor();
    push(stack, t.sym_storage_offset());
}

void sym_stride(Stack& stack) {
    // 从堆栈中弹出一个张量对象，并推送其符号步幅（sym_strides）的向量到堆栈
    auto t = std::move(pop(stack)).toTensor();
    pack(stack, t.sym_strides().vec());
}

void device(Stack& stack) {
    // 从堆栈中弹出一个张量对象，并推送其设备信息到堆栈
    push(stack, pop(stack).toTensor().device());
}

void device_with_index(Stack& stack) {
    // 从堆栈中弹出一个字符串类型和一个整数，并推送相应的设备信息到堆栈
    std::string type = pop(stack).toStringRef();
    int index = pop(stack).toInt();
    std::string device_str = type + ":" + std::to_string(index);
    auto device = c10::Device(device_str);
    push(stack, device);
}

void dtype(Stack& stack) {
    // 从堆栈中弹出一个张量对象，并推送其数据类型到堆栈
    at::Tensor a;
    pop(stack, a);
    push(stack, static_cast<int64_t>(a.scalar_type()));
}

void layout(Stack& stack) {
    // 从堆栈中弹出一个张量对象，并推送其布局信息到堆栈
    push(stack, pop(stack).toTensor().layout());
}
void toPrimDType(Stack& stack) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool non_blocking;  // 声明变量 non_blocking，用于指示是否非阻塞
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool copy;  // 声明变量 copy，用于指示是否复制数据
  pop(stack, non_blocking, copy);  // 从堆栈中弹出 non_blocking 和 copy 变量的值
  std::optional<at::ScalarType> scalarType =  // 从堆栈中弹出一个可选的标量类型
      pop(stack).toOptional<at::ScalarType>();
  std::optional<c10::Device> device = c10::nullopt;  // 初始化一个空的设备类型
  at::Tensor self = pop(stack).toTensor();  // 从堆栈中弹出一个张量对象
  push(stack, to_dispatch(self, device, scalarType, non_blocking, copy));  // 将处理后的结果推入堆栈中
}

void dim(Stack& stack) {
  at::Tensor arg = pop(stack).toTensor();  // 从堆栈中弹出一个张量对象
  push(stack, arg.dim());  // 将张量对象的维度数推入堆栈中
}

void _not(Stack& stack) {
  push(stack, !pop(stack).toBool());  // 将布尔值的逻辑非结果推入堆栈中
}

void boolTensor(Stack& stack) {
  at::Tensor a;  // 声明一个张量对象
  pop(stack, a);  // 从堆栈中弹出一个张量对象赋值给 a
  push(stack, at::native::is_nonzero(a));  // 将张量 a 是否非零的结果推入堆栈中
}

void toList(Stack& stack) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int elem_ty_val;  // 声明变量 elem_ty_val，表示元素类型的数值
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int dim_val;  // 声明变量 dim_val，表示列表的维度
  at::Tensor t;  // 声明一个张量对象 t

  pop(stack, elem_ty_val);  // 从堆栈中弹出 elem_ty_val，表示元素类型的数值
  pop(stack, dim_val);  // 从堆栈中弹出 dim_val，表示列表的维度
  pop(stack, t);  // 从堆栈中弹出一个张量对象赋值给 t

  // 如果张量不在 CPU 上，则将其转移到 CPU 上
  if (!t.device().is_cpu()) {
    t = t.cpu();
  }

  // 根据 elem_ty_val 和 dim_val 重建输出类型
  at::TypePtr out_ty;
  if (elem_ty_val == 0) {
    out_ty = at::IntType::get();
  } else if (elem_ty_val == 1) {
    out_ty = at::FloatType::get();
  } else if (elem_ty_val == 2) {
    out_ty = at::BoolType::get();
  } else if (elem_ty_val == 3) {
    out_ty = at::ComplexType::get();
  } else {
    TORCH_CHECK(
        false,
        "Unsupported element type for tolist; only int, float, complex and bool are supported");
  }

  // 检查张量的数据类型是否与注释的类型匹配
  // 对于 float/complex 类型，允许张量数据类型为 float/complex，将会在后续转换为 double/c10::complex<double>
  TORCH_CHECK(
      (out_ty == at::FloatType::get() && t.is_floating_point()) ||
          (out_ty == at::ComplexType::get() && t.is_complex()) ||
          tryScalarTypeFromJitType(*out_ty) == t.scalar_type(),
      "Output annotation element type and runtime tensor element type must match for tolist(): ",
      *tryScalarTypeFromJitType(*out_ty),
      " vs ",
      t.scalar_type());

  // 检查张量的维度是否与注释的维度匹配
  TORCH_CHECK(
      dim_val == t.dim(),
      "Output annotation list dimension and runtime tensor dimension must match for tolist()");

  // 将 out_ty 包装在 ListType 中 dim_val 次
  for (const auto i : c10::irange(dim_val)) {
    (void)i; // 抑制未使用变量警告
    # 使用 `at::ListType::create` 方法创建一个新的列表类型对象 `out_ty`，并将其赋值给 `out_ty`
    out_ty = at::ListType::create(out_ty);
  }

  # 获取张量 `t` 的维度并存储在 `dim` 中
  int64_t dim = t.dim();
  # 获取张量 `t` 的大小（shape）并存储在 `sizes` 中
  auto sizes = t.sizes();
  # 获取张量 `t` 的步幅（strides）并存储在 `strides` 中
  auto strides = t.strides();
  # 获取张量 `t` 的元素大小（字节数）并存储在 `element_size` 中
  size_t element_size = t.element_size();
  # 获取张量 `t` 的数据指针，并转换为 `char*` 类型存储在 `data` 中
  char* data = static_cast<char*>(t.data_ptr());
  # 调用 `tensorToListRecursive` 函数，将张量 `t` 的数据指针 `data`、维度 `dim`、列表类型 `out_ty`、
  # 标量类型 `t.scalar_type()`、大小 `sizes`、步幅 `strides`、元素大小 `element_size` 作为参数，
  # 并获得递归处理后的结果，存储在 `result` 中
  auto result = tensorToListRecursive(
      data, 0, dim, out_ty, t.scalar_type(), sizes, strides, element_size);
  # 将 `result` 移动到堆栈 `stack` 中
  push(stack, std::move(result));
} // namespace jit
} // namespace torch
```