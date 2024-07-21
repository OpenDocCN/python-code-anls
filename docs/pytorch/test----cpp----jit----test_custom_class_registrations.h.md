# `.\pytorch\test\cpp\jit\test_custom_class_registrations.h`

```py
#include <torch/custom_class.h>  // 包含 Torch 自定义类的头文件
#include <torch/script.h>        // 包含 Torch 脚本的头文件

namespace torch {
namespace jit {

struct ScalarTypeClass : public torch::CustomClassHolder {
  ScalarTypeClass(at::ScalarType s) : scalar_type_(s) {}  // 初始化 ScalarTypeClass 结构体，传入标量类型

  at::ScalarType scalar_type_;  // 成员变量，保存标量类型
};

template <class T>
struct MyStackClass : torch::CustomClassHolder {
  std::vector<T> stack_;  // 使用模板类型 T 创建的向量 stack_

  MyStackClass(std::vector<T> init) : stack_(init.begin(), init.end()) {}  // 初始化 MyStackClass 对象，传入初始向量

  void push(T x) {  // 将元素 x 推入栈中
    stack_.push_back(x);
  }

  T pop() {  // 弹出栈顶元素并返回
    auto val = stack_.back();
    stack_.pop_back();
    return val;
  }

  c10::intrusive_ptr<MyStackClass> clone() const {  // 克隆当前对象并返回新的共享指针
    return c10::make_intrusive<MyStackClass>(stack_);
  }

  void merge(const c10::intrusive_ptr<MyStackClass>& c) {  // 合并另一个 MyStackClass 对象的元素到当前栈中
    for (auto& elem : c->stack_) {
      push(elem);
    }
  }

  std::tuple<double, int64_t> return_a_tuple() const {  // 返回包含 double 和 int64_t 类型的元组
    return std::make_tuple(1337.0f, 123);
  }
};

} // namespace jit
} // namespace torch
```