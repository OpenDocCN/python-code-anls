# `.\pytorch\c10\core\SymFloat.h`

```
#pragma once

#include <c10/core/SymBool.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <limits>
#include <ostream>
#include <utility>

namespace c10 {

// NB: this is actually double precision; we're using the Python naming here
class C10_API SymFloat {
 public:
  /*implicit*/ SymFloat(double d) : data_(d){};
  // Construct SymFloat from SymNode, initializing with NaN if not float
  SymFloat(SymNode ptr)
      : data_(std::numeric_limits<double>::quiet_NaN()), ptr_(std::move(ptr)) {
    TORCH_CHECK(ptr_->is_float());
  };
  // Default constructor initializes with 0.0
  SymFloat() : data_(0.0) {}

  // Return the underlying SymNodeImpl pointer without ownership transfer
  SymNodeImpl* toSymNodeImplUnowned() const {
    return ptr_.get();
  }

  // Release ownership of SymNodeImpl from SymFloat
  SymNodeImpl* release() && {
    return std::move(ptr_).release();
  }

  // Convert SymFloat to SymNode
  SymNode toSymNodeImpl() const;

  // Wrap SymFloat with base SymNode, ensuring a SymNode is returned
  SymNode wrap_node(const SymNode& base) const;

  // Return the floating point value if not symbolic
  double expect_float() const {
    TORCH_CHECK(!is_symbolic());
    return data_;
  }

  // Arithmetic operations for SymFloat
  SymFloat operator+(const SymFloat&) const;
  SymFloat operator-(const SymFloat&) const;
  SymFloat operator*(const SymFloat&) const;
  SymFloat operator/(const SymFloat&) const;

  // Symbolic comparison operations for SymFloat
  SymBool sym_eq(const SymFloat&) const;
  SymBool sym_ne(const SymFloat&) const;
  SymBool sym_lt(const SymFloat&) const;
  SymBool sym_le(const SymFloat&) const;
  SymBool sym_gt(const SymFloat&) const;
  SymBool sym_ge(const SymFloat&) const;

  // Comparison operators for SymFloat, using guard_bool for assertion
  bool operator==(const SymFloat& o) const {
    return sym_eq(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator!=(const SymFloat& o) const {
    return sym_ne(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator<(const SymFloat& o) const {
    return sym_lt(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator<=(const SymFloat& o) const {
    return sym_le(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator>(const SymFloat& o) const {
    return sym_gt(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator>=(const SymFloat& o) const {
    return sym_ge(o).guard_bool(__FILE__, __LINE__);
  }
};

} // namespace c10
    // 返回一个符号引用 `o` 的守护布尔值。此方法根据当前文件和行号作为诊断信息。
    return sym_ge(o).guard_bool(__FILE__, __LINE__);
    }
    
    SymFloat min(const SymFloat& sci) const;
    SymFloat max(const SymFloat& sci) const;
    
    // 需要指导在哪里放置这段代码
    SymFloat sqrt() const;
    
    // 插入一个保护，使浮点数变为其具体值，然后返回该值。即使浮点数是符号化的，
    // 只要我们知道其基础值，这个操作总是有效的。不要无脑地在所有地方使用此方法；
    // 这可能会导致 PyTorch 程序的过度专门化。
    //
    // 应该以 guard_float(__FILE__, __LINE__) 的形式调用它。文件名和行号可用于
    // 诊断过度专门化问题。
    double guard_float(const char* file, int64_t line) const;
    
    bool has_hint() const;
    
    // 注意：在头文件中保持这个定义很重要，因为我们期望在移动构建中折叠 `if` 检查，
    // 其中 `is_symbolic` 始终为 false。
    C10_ALWAYS_INLINE bool is_symbolic() const {
        return ptr_;
    }
    
    double as_float_unchecked() const {
        return data_;
    }
    
    private:
    // TODO: optimize to union
    // 双重存储：优化为联合体的结构
    double data_;
    SymNode ptr_;
};

// C10_API 是一个宏，用于声明符号浮点类型的输出流操作符重载
// std::ostream& 是输出流的引用，os 是要输出到的流对象，const SymFloat& s 是要输出的 SymFloat 对象的引用
std::ostream& operator<<(std::ostream& os, const SymFloat& s);
// 结束 c10 命名空间的声明
} // namespace c10
```