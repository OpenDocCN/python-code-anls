# `.\pytorch\torch\csrc\jit\python\python_dict.cpp`

```py
// 定义命名空间 torch::jit
namespace torch::jit {

// 实现 ScriptDictIterator 类的 next() 方法
IValue ScriptDictIterator::next() {
  // 检查迭代器是否已经到达结尾
  if (iter_ == end_) {
    // 如果是，则抛出停止迭代异常
    throw py::stop_iteration();
  }

  // 由于这是 .items() 的迭代器，当前的键和值应作为元组返回
  IValue result = c10::ivalue::Tuple::create({iter_->key(), iter_->value()});

  // 推进迭代器以备下一次迭代
  iter_++;

  // 返回组装好的结果
  return result;
}

// 实现 ScriptDictKeyIterator 类的 next() 方法
IValue ScriptDictKeyIterator::next() {
  // 检查迭代器是否已经到达结尾
  if (iter_ == end_) {
    // 如果是，则抛出停止迭代异常
    throw py::stop_iteration();
  }

  // 由于这是 .keys() 和 __iter__() 的迭代器，只返回键值
  IValue result = iter_->key();

  // 推进迭代器以备下一次迭代
  iter_++;

  // 返回键值
  return result;
}

} // namespace torch::jit


这段代码实现了两个类（`ScriptDictIterator` 和 `ScriptDictKeyIterator`）的迭代器方法 `next()`。每个方法的作用是从字典迭代器中获取下一个元素，并返回相应的键值对或键。
```