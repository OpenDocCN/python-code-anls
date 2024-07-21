# `.\pytorch\torch\csrc\profiler\unwind\unwinder.h`

```py
#pragma once
// 包含必要的头文件以进行调用栈展开操作
#include <torch/csrc/profiler/unwind/action.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <cstdint> // 用于整数类型的定义
#include <limits> // 包含 std::numeric_limits 用于获取类型的极限值

namespace torch::unwind {

// 定义用于存储当前展开状态的结构体
struct UnwindState {
  int64_t rip, rbp, rsp; // 存储指令指针、基址指针和栈指针的当前值
};

// 定义用于执行调用栈展开操作的类
struct Unwinder {
  // 构造函数，接受三个动作参数来初始化展开器
  Unwinder(Action rsp, Action rip, Action rbp)
      : kind_(rip.kind == A_UNDEFINED ? END : STANDARD), // 根据 rip 的类型确定 kind 的值
        reg_(rsp.reg), // 存储寄存器编号
        off_(rsp.data), // 存储偏移量
        rip_off_(rip.data), // 存储 rip 的偏移量
        rbp_off_(
            rbp.kind == A_UNDEFINED ? std::numeric_limits<int64_t>::max() // 如果 rbp 未定义，则设置为 int64_t 类型的最大值
                                    : rbp.data), // 否则存储 rbp 的偏移量
        deref_(rsp.kind == A_REG_PLUS_DATA_DEREF) { // 根据 rsp 的类型设置是否进行解引用
    check(rsp.reg == D_RSP || rsp.reg == D_RBP); // 检查 rsp.reg 是否是 D_RSP 或 D_RBP
    check(rip.kind == A_UNDEFINED || rip.kind == A_LOAD_CFA_OFFSET); // 检查 rip.kind 是否未定义或者是 A_LOAD_CFA_OFFSET
    if (rsp.kind == A_REG_PLUS_DATA) {
      check(rbp.kind == A_LOAD_CFA_OFFSET || rbp.kind == A_UNDEFINED); // 如果 rsp.kind 是 A_REG_PLUS_DATA，则检查 rbp.kind 的合法性
    } else if (rsp.kind == A_REG_PLUS_DATA_DEREF) {
      if (rbp.kind == A_REG_PLUS_DATA_DEREF) {
        check(rbp.reg == rsp.reg); // 如果 rsp.kind 是 A_REG_PLUS_DATA_DEREF，则检查 rbp.reg 是否等于 rsp.reg
        rbp_off_ -= rsp.data; // 计算 rbp_off_
      } else {
        check(rbp.kind == A_UNDEFINED); // 否则检查 rbp.kind 是否未定义
      }
    } else {
      check(false); // 如果 rsp.kind 不符合以上任何情况，则抛出异常
    }
  }
  
  // 辅助函数，用于检查条件，若条件不满足则抛出 UnwindError 异常
  void check(bool cond) {
    if (!cond) {
      throw UnwindError("Unwinding actions do not follow supported patterns");
    }
  }
  
  // 返回展开器是否为终止状态
  bool terminator() const {
    return kind_ != STANDARD;
  }
  
  // 返回展开器是否处于未知状态
  bool isUnknown() const {
    return kind_ == UNKNOWN;
  }
  
  // 创建一个表示当前实现不支持的展开器
  static Unwinder unknown() {
    return Unwinder();
  }
  
  // 执行展开操作，返回更新后的 UnwindState
  UnwindState run(const UnwindState& cur) const {
    UnwindState r = cur; // 创建一个新的 UnwindState，初始化为当前状态
    r.rsp = (reg_ == D_RSP ? cur.rsp : cur.rbp) + off_; // 根据 reg_ 的值更新 rsp
    r.rbp = rbp_off_ == std::numeric_limits<int64_t>::max()
        ? cur.rbp
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        : *(int64_t*)(r.rsp + rbp_off_); // 根据 rbp_off_ 的值更新 rbp
    
    if (deref_) {
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      r.rsp = *(int64_t*)r.rsp; // 如果 deref_ 为 true，则解引用 rsp
    }
    
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    r.rip = *(int64_t*)(r.rsp + rip_off_); // 根据 rip_off_ 的值更新 rip
    
    return r; // 返回更新后的 UnwindState
  }

 private:
  Unwinder() : kind_(UNKNOWN), reg_(0), off_(0), rip_off_(0), rbp_off_(0) {} // 私有构造函数，用于创建未知状态的展开器
  enum Kind { STANDARD, END, UNKNOWN } kind_; // 枚举类型 Kind，表示展开器的状态
  uint32_t reg_; // 存储寄存器编号
  int64_t off_; // 存储偏移量
  int64_t rip_off_; // 存储 rip 的偏移量
  int64_t rbp_off_; // 存储 rbp 的偏移量
  bool deref_{false}; // 标志是否进行解引用
};

} // namespace torch::unwind
```