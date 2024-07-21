# `.\pytorch\torch\csrc\cuda\utils.cpp`

```
#ifdef USE_CUDA
// 当宏定义 USE_CUDA 被定义时编译以下内容

// 将 Python 对象转换为 CUDAStream 的可选列表
std::vector<std::optional<at::cuda::CUDAStream>>
THPUtils_PySequence_to_CUDAStreamList(PyObject* obj) {
  // 检查输入对象是否为序列
  if (!PySequence_Check(obj)) {
    throw std::runtime_error(
        "Expected a sequence in THPUtils_PySequence_to_CUDAStreamList");
  }
  // 快速获取序列对象的引用
  THPObjectPtr seq = THPObjectPtr(PySequence_Fast(obj, nullptr));
  // 如果获取失败，抛出异常
  if (seq.get() == nullptr) {
    throw std::runtime_error(
        "expected PySequence, but got " + std::string(THPUtils_typename(obj)));
  }

  // 创建存储 CUDAStream 可选对象的向量
  std::vector<std::optional<at::cuda::CUDAStream>> streams;
  // 获取序列的长度
  Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
  // 遍历序列中的每一项
  for (Py_ssize_t i = 0; i < length; i++) {
    // 获取序列中的当前项
    PyObject* stream = PySequence_Fast_GET_ITEM(seq.get(), i);

    // 检查当前项是否为 THCPStreamClass 的实例
    if (PyObject_IsInstance(stream, THCPStreamClass)) {
      // 使用 reinterpret_cast 进行类型转换，创建 CUDAStream 对象并添加到向量中
      streams.emplace_back(at::cuda::CUDAStream::unpack3(
          (reinterpret_cast<THCPStream*>(stream))->stream_id,
          (reinterpret_cast<THCPStream*>(stream))->device_index,
          static_cast<c10::DeviceType>(
              (reinterpret_cast<THCPStream*>(stream))->device_type)));
    } else if (stream == Py_None) {
      // 如果当前项是 None，则添加一个空的 std::optional 到向量中
      streams.emplace_back();
    } else {
      // 如果遇到未知数据类型，则抛出异常
      // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
      std::runtime_error(
          "Unknown data type found in stream list. Need torch.cuda.Stream or None");
    }
  }
  // 返回存储 CUDAStream 可选对象的向量
  return streams;
}
#endif
```