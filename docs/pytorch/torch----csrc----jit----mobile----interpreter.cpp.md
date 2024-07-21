# `.\pytorch\torch\csrc\jit\mobile\interpreter.cpp`

```
#include <torch/csrc/jit/mobile/interpreter.h>  // 引入移动端解释器的头文件

#include <ATen/core/class_type.h>  // 引入 ATen 类型系统的类类型头文件
#include <ATen/core/dynamic_type.h>  // 引入 ATen 动态类型的头文件
#include <ATen/core/function.h>  // 引入 ATen 函数相关头文件
#include <ATen/core/jit_type.h>  // 引入 ATen JIT 类型的头文件
#include <ATen/core/operator_name.h>  // 引入 ATen 运算符名称的头文件
#include <ATen/record_function.h>  // 引入 ATen 记录函数的头文件
#include <c10/util/Exception.h>  // 引入 C10 异常处理的头文件
#include <c10/util/irange.h>  // 引入 C10 数字范围的头文件
#include <torch/csrc/jit/backends/backend_exception.h>  // 引入 JIT 后端异常处理的头文件
#include <torch/csrc/jit/mobile/function.h>  // 引入移动端函数定义的头文件
#include <torch/csrc/jit/mobile/observer.h>  // 引入移动端观察器的头文件
#include <torch/csrc/jit/mobile/promoted_prim_ops.h>  // 引入移动端推广原始操作的头文件
#include <torch/csrc/jit/runtime/jit_exception.h>  // 引入 JIT 运行时异常处理的头文件
#include <torch/csrc/jit/runtime/vararg_functions.h>  // 引入 JIT 可变参数函数的头文件

namespace torch {
namespace jit {
char const* toString(OpCode op);  // 定义函数声明，将操作码转换为字符串形式
std::ostream& operator<<(std::ostream& out, Instruction inst);  // 定义流输出操作符重载，用于打印指令

namespace mobile {
InterpreterState::InterpreterState(const Code& code) {  // 解释器状态构造函数的实现
  enterFrame(code);  // 调用 enterFrame 方法进入执行帧
}

namespace {
static thread_local std::vector<DebugHandle> exception_debug_handles_;  // 声明线程局部静态变量，存储异常调试句柄

void createObject(Stack& stack, const at::ClassTypePtr& type) {  // 创建对象函数的实现
  auto userObj = c10::ivalue::Object::create(  // 创建用户对象实例
      c10::StrongTypePtr(type->compilation_unit(), type),  // 使用类型的强类型指针创建
      type->numAttributes());  // 指定对象的属性数量
  push(stack, std::move(userObj));  // 将创建的对象压入栈中
}

void isinstance(Stack& stack, at::ArrayRef<at::TypePtr> types) {  // 类型判断函数的实现
  at::TypePtr ty = pop(stack).type<c10::DynamicType>();  // 从栈中弹出对象类型
  for (const at::TypePtr& candidate : types) {  // 遍历候选类型列表
    if (ty->isSubtypeOf(*candidate)) {  // 判断对象类型是否是候选类型的子类型
      push(stack, true);  // 将判断结果压入栈中，表示对象类型是候选类型的子类型
      return;
    }
  }
  push(stack, false);  // 将判断结果压入栈中，表示对象类型不是任何候选类型的子类型
}
} // namespace

using namespace at;

const std::vector<DebugHandle>& getInterpretersExceptionDebugHandles() {  // 获取解释器异常调试句柄的函数实现
  return exception_debug_handles_;  // 返回异常调试句柄向量的引用
}

void InterpreterState::enterFrame(const Code& code) {  // 进入执行帧的方法实现
  frames_.emplace_back(code);  // 将指定代码块作为新的帧加入帧堆栈
  registers_.resize(registers_.size() + code.register_size_);  // 调整寄存器大小以适应新的代码块
}

void InterpreterState::leaveFrame() {  // 离开执行帧的方法实现
  registers_.resize(  // 调整寄存器大小，以移除当前帧的寄存器空间
      registers_.size() - frames_.back().getCode().register_size_);
  frames_.pop_back();  // 弹出当前帧
}

void InterpreterState::saveExceptionDebugHandles() {  // 保存异常调试句柄的方法实现
  std::vector<DebugHandle> exception_debug_handles;  // 声明异常调试句柄向量
  for (auto frame = frames_.crbegin(); frame != frames_.crend(); frame++) {  // 反向遍历帧堆栈
    size_t pc = frame->getPC() - (frame != frames_.crbegin() ? 1 : 0);  // 获取当前帧的程序计数器位置
    if (auto handle = frame->getDebugHandle(pc)) {  // 获取当前位置的调试句柄
      exception_debug_handles.push_back(*handle);  // 将调试句柄加入向量
    } else {
      exception_debug_handles.push_back(-1);  // 若无调试句柄，加入无效值
    }
  }
  exception_debug_handles_ = std::move(exception_debug_handles);  // 移动异常调试句柄向量到线程局部静态变量
}

void InterpreterState::callFunction(torch::jit::Function& f, Stack& stack) {  // 调用函数的方法实现
  bool newFrame =
      f.call(stack, [&](const mobile::Code& code) { enterFrame(code); });  // 调用函数对象并进入新的执行帧
  (frames_.rbegin() + (newFrame ? 1 : 0))->step();  // 如果创建了新帧，则执行新帧的步进操作
}

bool InterpreterState::run(Stack& stack) {  // 运行解释器的方法实现
  while (true) {  // 进入主循环
    } catch (c10::BackendRuntimeException& e) {  // 捕获后端运行时异常
      saveExceptionDebugHandles();  // 保存异常调试句柄
      TORCH_RETHROW(e);  // 重新抛出异常
    } catch (c10::Error& error) {  // 捕获 C10 错误
      // 捕获并重新抛出错误的原因是为了设置后续查询的异常程序计数器位置
      saveExceptionDebugHandles();  // 保存异常调试句柄
      TORCH_RETHROW(error);  // 重新抛出错误
  } catch (...) {
    // 捕获所有异常，保存调试处理信息，然后重新抛出异常
    saveExceptionDebugHandles();
    throw;
  }
  // 以下代码段被注释掉，本意是遍历栈中的每个元素
  // 如果元素是张量，输出其尺寸；否则直接输出元素值
  //  for (auto val : stack) {
  //    if (val.isTensor()) {
  //      std::cout << val.toTensor().sizes() << std::endl;
  //    } else {
  //      std::cout << val << std::endl;
  //    }
  //  }
  // 返回假，表示函数执行未达到期望条件
  return false;
}

// 返回对应寄存器的引用
IValue& InterpreterState::reg(size_t reg) {
  // 检查寄存器索引是否有效
  TORCH_CHECK(
      reg > 0 && reg <= registers_.size(), "Invalid register index: ", reg);
  // 返回寄存器的引用
  return *(registers_.end() - reg);
}

// 命名空间 mobile 结束
} // namespace mobile

// 命名空间 jit 结束
} // namespace jit

// 命名空间 torch 结束
} // namespace torch
```