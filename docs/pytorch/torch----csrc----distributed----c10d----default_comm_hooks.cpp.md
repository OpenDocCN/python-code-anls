# `.\pytorch\torch\csrc\distributed\c10d\default_comm_hooks.cpp`

```
namespace c10d {

# 进入 c10d 命名空间，该命名空间包含了分布式相关的 C10 库功能。


c10::intrusive_ptr<c10::ivalue::Future> AllReduceCommHook::runHook(
    GradBucket& bucket) {

# 定义了 `AllReduceCommHook` 类中的 `runHook` 方法，接受一个 `GradBucket` 类型的引用参数，并返回一个 `c10::intrusive_ptr<c10::ivalue::Future>` 类型的指针。


std::vector<at::Tensor> tensors = {bucket.getBufferRef()};

# 创建了一个名为 `tensors` 的 `std::vector`，其中包含了 `bucket` 对象中的缓冲区引用作为唯一元素。


// Apply the division first to avoid overflow, especially for FP16.
tensors[0] /= state_->getSize();

# 对 `tensors` 中的第一个张量执行除法操作，以避免溢出，特别是对于 FP16 类型的张量。


return state_->allreduce(tensors)->getFuture();

# 调用 `state_` 对象的 `allreduce` 方法，对 `tensors` 中的张量进行全局归约操作，并返回一个 `Future` 对象。


}

c10::intrusive_ptr<c10::ivalue::Future> FP16CompressCommHook::runHook(
    GradBucket& bucket) {

# 定义了 `FP16CompressCommHook` 类中的 `runHook` 方法，接受一个 `GradBucket` 类型的引用参数，并返回一个 `c10::intrusive_ptr<c10::ivalue::Future>` 类型的指针。


auto compressed_tensor = bucket.getBufferRef().to(torch::kFloat16);

# 将 `bucket` 对象的缓冲区引用转换为 `torch::kFloat16` 类型，得到压缩后的张量 `compressed_tensor`。


// Apply the division first to avoid overflow.
compressed_tensor /= state_->getSize();

# 对 `compressed_tensor` 执行除法操作，以避免溢出。


std::vector<at::Tensor> tensors = {compressed_tensor};

# 创建一个名为 `tensors` 的 `std::vector`，其中包含了 `compressed_tensor` 作为唯一元素。


auto allreduce_fut = state_->allreduce(tensors)->getFuture();

# 调用 `state_` 对象的 `allreduce` 方法，对 `tensors` 中的张量进行全局归约操作，并获取其 `Future` 对象。


auto decompressed_tensor = bucket.getBufferRef();
auto decompress = [decompressed_tensor](c10::ivalue::Future& allreduce_fut) {
    auto result = allreduce_fut.value();
    TORCH_INTERNAL_ASSERT(
        result.isTensorList(),
        "ProcessGroup::allreduce should return TensorList");

    auto reduce_tensor = result.toTensorVector()[0];
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        reduce_tensor.scalar_type() == at::ScalarType::Half,
        "Expected reduced tensor to be fp16 in FP16CompressHook, but got type ",
        reduce_tensor.scalar_type());
    decompressed_tensor.copy_(reduce_tensor);
    return c10::IValue(decompressed_tensor);
};

# 定义了一个名为 `decompress` 的 Lambda 函数，该函数接受一个 `allreduce_fut` 参数，并执行以下操作：
## - 获取 `allreduce_fut` 的值，并断言其为 `TensorList` 类型。
## - 获取减少后的张量 `reduce_tensor`，并断言其类型为 `Half` 类型。
## - 将 `reduce_tensor` 的值复制到 `decompressed_tensor` 中，并返回一个 `c10::IValue` 对象。


return allreduce_fut->then(decompress, allreduce_fut->elementType());

# 返回 `allreduce_fut` 的 `then` 方法调用结果，将 `decompress` 函数注册为 `allreduce_fut` 的回调函数，并指定其元素类型。


}

c10::intrusive_ptr<c10::ivalue::Future> _AllReduceBySumCommHook::runHook(
    GradBucket& bucket) {

# 定义了 `_AllReduceBySumCommHook` 类中的 `runHook` 方法，接受一个 `GradBucket` 类型的引用参数，并返回一个 `c10::intrusive_ptr<c10::ivalue::Future>` 类型的指针。


std::vector<at::Tensor> tensors = {bucket.getBufferRef()};

# 创建一个名为 `tensors` 的 `std::vector`，其中包含了 `bucket` 对象的缓冲区引用作为唯一元素。


#ifdef IS_NCCLX

# 如果定义了 `IS_NCCLX` 宏，则执行以下代码块。


// case with sparse_metadata_ set and using indices from there
if (bucket.getSparseGradIndices().has_value()) {
    AllreduceOptions opts = AllreduceOptions();
    opts.sparseIndices = bucket.getSparseGradIndices().value();
    return state_->allreduce(tensors, opts)->getFuture();
}

# 如果 `bucket` 中包含稀疏梯度索引，则创建 `AllreduceOptions` 对象 `opts`，设置其稀疏索引选项，并调用 `state_` 对象的 `allreduce` 方法，使用这些选项对 `tensors` 中的张量进行全局归约操作，并返回其 `Future` 对象。


#else

# 如果未定义 `IS_NCCLX` 宏，则执行以下代码块。


return state_->allreduce(tensors)->getFuture();

# 直接调用 `state_` 对象的 `allreduce` 方法，对 `tensors` 中的张量进行全局归约操作，并返回其 `Future` 对象。


#endif

# 结束条件编译块。


}

} // namespace c10d

# 结束 `c10d` 命名空间。
```