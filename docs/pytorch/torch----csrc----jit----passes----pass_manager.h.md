# `.\pytorch\torch\csrc\jit\passes\pass_manager.h`

```
#pragma once

#include <torch/csrc/jit/ir/ir.h>

/* `getCustomPrePasses()` 返回一个包含所有将在微分之后但融合之前执行的传递的向量。
 * 这是编译器后端插入传递的事实位置。
 *
 * `getCustomPostPasses()` 返回一个包含所有将在微分之后并且在融合之后（如果有）执行的传递的向量。
 * 这是需要融合清理传递的位置（如果需要的话）。
 *
 * 可以通过在编译单元中创建全局 `Register{Pre,Post}Pass r(Pass)` 变量来静态注册传递。
 *
 * pass_manager.h 使用 Meyer's 单例模式来存储修改 IR 图的传递的向量。
 */

namespace torch {
namespace jit {

// A pass modifies a Graph in place.
using GraphPass = std::function<void(std::shared_ptr<Graph>&)>;

// Since Passes are std::functions, we associate a UUID to each pass, this way
// if we want to deregister a pass, we have something to reference it by.
using GraphPassNameType = unsigned int;

// Graph pass entries have a name associated with them
using GraphPassEntry = std::pair<GraphPass, GraphPassNameType>;

// 返回当前注册的后传递。传递以静态向量存储。
TORCH_API std::vector<std::pair<GraphPass, GraphPassNameType>>&
getCustomPostPasses();

// 返回当前注册的前传递。传递以静态向量存储。
TORCH_API std::vector<std::pair<GraphPass, GraphPassNameType>>&
getCustomPrePasses();

// 注册后传递并返回与之关联的名称。
TORCH_API GraphPassNameType registerPostPass(GraphPass p);

// 注册前传递并返回与之关联的名称。
TORCH_API GraphPassNameType registerPrePass(GraphPass p);

// 通过名称查找传递，并从注册传递中移除。
TORCH_API void clearPostPass(GraphPassNameType p);

// 通过名称查找传递，并从注册传递中移除。
TORCH_API void clearPrePass(GraphPassNameType p);

// 移除所有注册的后传递。
TORCH_API void clearAllPostPasses();

// 移除所有注册的前传递。
TORCH_API void clearAllPrePasses();

// LEGACY CALL
struct TORCH_API RegisterPostPass {
  RegisterPostPass(GraphPass p);
};

using RegisterPass = RegisterPostPass;

/*
 * PassManager 是对上述注册/清除后传递函数的封装。它会注册 "registerPass" 中提供的传递，
 * 并保存它的关联名称，以便稍后调用 clearPass 时可以删除使用的传递。
 *
 * PassManager 是模板化的，因为我们想基于特定 GraphPass 的静态变量。当从 PassManager 派生时，
 * 模板参数应该发送您派生类的派生类型，就像您为奇特递归模板模式所做的那样。
 * 此模板参数实际上并未使用，只是为了防止静态成员在派生类型之间共享。
 */
template <typename DerivedType>
struct C10_EXPORT PassManager {
 private:
  // We want this class to be abstract because it's
  // 该类希望是抽象类，因此定义了纯虚函数
  virtual void abstract() = 0;

 protected:
  /*
   * isRegistered() will return if a pass has been registered
   * isRegistered(true) will change the value of the internal static bool
   *
   * There's an internal static bool to this function to keep track of the
   * state, this is so when functions are derived from this class, they don't
   * have to worry about initializing the static members.
   */
  // 判断是否已注册 pass，并根据 flip_bit 参数更新静态变量 val
  static bool isRegistered(bool flip_bit = false) {
    static bool val = false;
    if (flip_bit)
      val = !val;
    return val;
  }

  /*
   * name() will return the name of the registered pass
   * name(pass_name, true) will set the name of the pass
   * Similarly to isRegistered we use an internal static variable to hold the
   * name.
   */
  // 获取或设置注册 pass 的名称，并使用静态变量 pass_id 保存名称
  static GraphPassNameType passID(
      GraphPassNameType PassID = 0,
      bool set = false) {
    static GraphPassNameType pass_id = 0;
    if (set)
      pass_id = PassID;
    return pass_id;
  }

 public:
  // registerPass(pass) will register the pass provided and set the
  // name/isRegistered functions appropriately, it returns a bool value
  // indicating whether the given pass is already registered previously.
  // 注册传入的 pass，并根据是否已注册决定是否修改名称和状态，并返回是否已经注册的布尔值
  static bool registerPass(GraphPass p) {
    if (!isRegistered()) {
      // If we don't already have a registered pass, register pass
      // hold on to its name, change isRegistered to true
      // 如果没有已注册的 pass，则注册 pass，保存其名称，并将 isRegistered 置为 true
      passID(registerPostPass(std::move(p)), true);
      isRegistered(true);
      return false;
    }
    return true;
  }

  // Calls ClearPostPass(passID())
  // 调用 ClearPostPass(passID()) 方法
  static void clearPass() {
    // If the pass is registered, clear it and change isRegistered to false.
    // 如果 pass 已注册，则清除它并将 isRegistered 置为 false
    if (isRegistered()) {
      clearPostPass(passID());
      isRegistered(true);
    }
  }

  // clang-tidy requires virtual destructor;
  // 虚析构函数，用于满足 clang-tidy 的要求
  virtual ~PassManager() = default;
};

} // namespace jit
} // namespace torch
```