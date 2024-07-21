# `.\pytorch\torch\csrc\jit\python\python_list.cpp`

```py
namespace torch::jit {

// 实现 ScriptListIterator 类的 next() 方法，用于获取迭代器的下一个元素
IValue ScriptListIterator::next() {
  // 如果迭代器已经到达末尾，则抛出停止迭代的异常
  if (iter_ == end_) {
    throw py::stop_iteration();
  }

  // 获取当前迭代器指向的元素
  IValue result = *iter_;

  // 将迭代器向前移动到下一个元素
  iter_++;

  // 返回获取的元素值
  return result;
}

// 返回迭代器是否已经完成迭代所有元素
bool ScriptListIterator::done() const {
  return iter_ == end_;
}

namespace {
// 将 ScriptList 转换为 py::list 类型的函数
py::list scriptListToPyList(const ScriptList& src) {
  // 创建一个与 ScriptList 长度相同的 py::list 对象
  py::list out(src.len());
  // 获取 ScriptList 的迭代器
  auto iter = src.iter();

  size_t i = 0;
  // 遍历 ScriptList 中的每个元素
  while (!iter.done()) {
    // 获取迭代器的下一个元素
    auto val = iter.next();

    // TODO: 处理嵌套的字典结构。

    // 如果当前元素是列表，则递归调用 scriptListToPyList 处理
    if (val.isList()) {
      out[i] = scriptListToPyList(val);
    } else {
      // 否则，将当前元素转换为 PyObject 并存储在 py::list 中
      out[i] = toPyObject(val);
    }
    ++i;
  }

  // 返回转换后的 py::list 对象
  return out;
}
} // namespace

} // namespace torch::jit
```