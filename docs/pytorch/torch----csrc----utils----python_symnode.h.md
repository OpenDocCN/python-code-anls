# `.\pytorch\torch\csrc\utils\python_symnode.h`

```py
#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/core/SymNodeImpl.h>

#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {

// 声明获取符号整数类、符号浮点数类和符号布尔值类的函数
TORCH_PYTHON_API py::handle get_symint_class();
TORCH_PYTHON_API py::handle get_symfloat_class();
TORCH_PYTHON_API py::handle get_symbool_class();

// 注意事项：这些函数不能在torch尚未设置之前调用，否则会出错
// 备选设计是让torch向我们注册对象
inline bool is_symint(py::handle obj) {
  // 检查对象是否为符号整数类的实例
  return py::isinstance(obj, get_symint_class());
}
inline bool is_symfloat(py::handle obj) {
  // 检查对象是否为符号浮点数类的实例
  return py::isinstance(obj, get_symfloat_class());
}
inline bool is_symbool(py::handle obj) {
  // 检查对象是否为符号布尔值类的实例
  return py::isinstance(obj, get_symbool_class());
}

namespace impl {

// 该c10::SymNodeImpl类简单地作为Python对象的适配器，实现了API。
// Python对象是真实数据的来源，这里只是一个适配器，让C++调用可以访问这些对象。
class PythonSymNodeImpl : public c10::SymNodeImpl {
 public:
  PythonSymNodeImpl(py::object pyobj) : c10::SymNodeImpl() {
    // 使用Python对象初始化，使用SafePyObject确保安全地管理Python对象
    pyobj_ = std::make_shared<c10::SafePyObject>(
        pyobj.release().ptr(), getPyInterpreter());
  };

  // 重写wrap_int方法，将传入的整数包装成符号节点对象
  c10::SymNode wrap_int(int64_t num) override {
    py::gil_scoped_acquire acquire;
    // 调用Python对象的wrap_int方法
    auto r = getPyObj().attr("wrap_int")(num);
    // 创建新的PythonSymNodeImpl对象，返回符号节点
    return c10::make_intrusive<PythonSymNodeImpl>(std::move(r));
  }

  // 重写wrap_float方法，将传入的浮点数包装成符号节点对象
  c10::SymNode wrap_float(double num) override {
    py::gil_scoped_acquire acquire;
    // 调用Python对象的wrap_float方法
    auto r = getPyObj().attr("wrap_float")(num);
    // 创建新的PythonSymNodeImpl对象，返回符号节点
    return c10::make_intrusive<PythonSymNodeImpl>(std::move(r));
  }

  // 重写wrap_bool方法，将传入的布尔值包装成符号节点对象
  c10::SymNode wrap_bool(bool num) override {
    py::gil_scoped_acquire acquire;
    // 调用Python对象的wrap_bool方法
    auto r = getPyObj().attr("wrap_bool")(num);
    // 创建新的PythonSymNodeImpl对象，返回符号节点
    return c10::make_intrusive<PythonSymNodeImpl>(std::move(r));
  }

  // 定义一组宏，用于生成符号节点方法的实现
#define TORCH_SYMNODE_SIZES_STRIDES(n)                                        \
  c10::SymNode n(                                                             \
      c10::ArrayRef<c10::SymNode> sizes, c10::ArrayRef<c10::SymNode> strides) \
      override {                                                              \
    py::gil_scoped_acquire acquire;                                           \
    // 调用Python对象的相应方法，传入sizes和strides参数
    auto r = getPyObj().attr(#n)(sizes, strides);                             \
    // 创建新的PythonSymNodeImpl对象，返回符号节点
    return c10::make_intrusive<PythonSymNodeImpl>(std::move(r));              \
  }

  // 生成多个符号节点方法的实现，如is_contiguous等
  TORCH_SYMNODE_SIZES_STRIDES(is_contiguous)
  TORCH_SYMNODE_SIZES_STRIDES(is_channels_last_contiguous_2d)
  TORCH_SYMNODE_SIZES_STRIDES(is_channels_last_contiguous_3d)
  TORCH_SYMNODE_SIZES_STRIDES(is_channels_last_strides_2d)
  TORCH_SYMNODE_SIZES_STRIDES(is_channels_last_strides_3d)
  TORCH_SYMNODE_SIZES_STRIDES(is_non_overlapping_and_dense)
  
  // 取消宏定义
#undef TORCH_SYMNODE_SIZES_STRIDES

  // 重写bool_方法，获取Python对象的布尔值表示
  bool bool_() override {
    py::gil_scoped_acquire acquire;
    // 调用Python对象的bool_方法
    auto r = getPyObj().attr("bool_")();
    // 返回布尔值结果
    return r.cast<bool>();
  }

  // 更多的方法实现可以在此添加
};
} // namespace impl
} // namespace torch
    // 调用getPyObj()函数获取Python对象，然后调用其attr方法来执行bool_属性，并检查其返回值是否为True
    return getPyObj().attr("bool_")().is(py::handle(Py_True));
  }

  // 检查Python对象是否具有is_int方法，并调用该方法，返回其返回值是否为True
  bool is_int() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("is_int")().is(py::handle(Py_True));
  }

  // 检查Python对象是否具有is_float方法，并调用该方法，返回其返回值是否为True
  bool is_float() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("is_float")().is(py::handle(Py_True));
  }

  // 检查Python对象是否具有is_bool方法，并调用该方法，返回其返回值是否为True
  bool is_bool() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("is_bool")().is(py::handle(Py_True));
  }

  // 检查Python对象是否具有is_nested_int方法，并调用该方法，返回其返回值是否为True
  bool is_nested_int() const override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("is_nested_int")().is(py::handle(Py_True));
  }

  // 检查Python对象是否具有has_hint方法，并调用该方法，返回其返回值是否为True
  bool has_hint() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("has_hint")().is(py::handle(Py_True));
  }

  // 调用Python对象的guard_int方法，传入file和line参数，并将其返回值转换为int64_t类型后返回
  int64_t guard_int(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("guard_int")(file, line).cast<int64_t>();
  }

  // 调用Python对象的guard_float方法，传入file和line参数，并将其返回值转换为double类型后返回
  double guard_float(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("guard_float")(file, line).cast<double>();
  }

  // 调用Python对象的guard_bool方法，传入file和line参数，并将其返回值转换为bool类型后返回
  bool guard_bool(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("guard_bool")(file, line).cast<bool>();
  }

  // 调用Python对象的expect_true方法，传入file和line参数，并将其返回值转换为bool类型后返回
  bool expect_true(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("expect_true")(file, line).cast<bool>();
  }

  // 调用Python对象的expect_size方法，传入file和line参数，并将其返回值转换为bool类型后返回
  bool expect_size(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("expect_size")(file, line).cast<bool>();
  }

  // 调用Python对象的guard_size_oblivious方法，传入file和line参数，并将其返回值转换为bool类型后返回
  bool guard_size_oblivious(const char* file, int64_t line) override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("guard_size_oblivious")(file, line).cast<bool>();
  }

  // 调用Python对象的int_方法，并将其返回值转换为int64_t类型后返回
  int64_t int_() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("int_")().cast<int64_t>();
  }

  // 调用Python对象的maybe_as_int方法，检查其返回值是否为None，如果是则返回std::nullopt，否则转换为int64_t类型后返回
  std::optional<int64_t> maybe_as_int() override {
    py::gil_scoped_acquire acquire;
    const auto& r = getPyObj().attr("maybe_as_int")();
    if (r.is_none()) {
      return c10::nullopt;
    } else {
      return r.cast<int64_t>();
    }
  }

  // 调用Python对象的str方法，并将其返回值转换为std::string类型后返回
  std::string str() override {
    py::gil_scoped_acquire acquire;
    return getPyObj().attr("str")().cast<std::string>();
  }

  // 调用Python对象的dispatch_sym_ite_方法，传入fname、other和third参数，并返回其处理后的c10::SymNode对象
  c10::SymNode dispatch_sym_ite_(
      const char* fname,
      const c10::SymNode& other,
      const c10::SymNode& third) {
    // 将other和third转换为PythonSymNodeImpl类型
    auto pother = dynamic_cast<PythonSymNodeImpl*>(other.get());
    auto pthird = dynamic_cast<PythonSymNodeImpl*>(third.get());
    // 断言转换成功
    TORCH_CHECK(pother);
    TORCH_CHECK(pthird);
    py::gil_scoped_acquire acquire;
    // 调用Python对象的fname方法，传入pother和pthird对象，并返回处理后的Python对象
    auto r = getPyObj().attr(fname)(pother->getPyObj(), pthird->getPyObj());
    // 将Python对象的返回值包装为PythonSymNodeImpl类型的c10::SymNode对象并返回
    return c10::make_intrusive<PythonSymNodeImpl>(r);
  }

  // 调用Python对象的dispatch_common_方法，传入fname和other参数，并返回其处理后的c10::SymNode对象
  c10::SymNode dispatch_common_(const char* fname, const c10::SymNode& other) {
    // 将other转换为PythonSymNodeImpl类型
    auto pother = dynamic_cast<PythonSymNodeImpl*>(other.get());
    // 断言转换成功
    TORCH_CHECK(pother);
    py::gil_scoped_acquire acquire;
    // 调用Python对象的fname方法，传入pother对象，并返回处理后的Python对象
    auto r = getPyObj().attr(fname)(pother->getPyObj());
    // 将Python对象的返回值包装为PythonSymNodeImpl类型的c10::SymNode对象并返回
    return c10::make_intrusive<PythonSymNodeImpl>(r);
  }
  // 调用 Python 对象的特定方法，并返回处理结果作为 PythonSymNodeImpl 实例
  auto r = getPyObj().attr(fname)(pother->getPyObj());
  // 返回使用 Python 返回值创建的 PythonSymNodeImpl 实例
  return c10::make_intrusive<PythonSymNodeImpl>(r);
}

// 调用 Python 对象的特定方法，不带参数，并返回处理结果作为 PythonSymNodeImpl 实例
c10::SymNode dispatch_common_(const char* fname) {
  // 获取全局解释器锁，确保线程安全
  py::gil_scoped_acquire acquire;
  // 调用 Python 对象的指定方法，并返回处理结果作为 PythonSymNodeImpl 实例
  auto r = getPyObj().attr(fname)();
  // 返回使用 Python 返回值创建的 PythonSymNodeImpl 实例
  return c10::make_intrusive<PythonSymNodeImpl>(r);
}

// 以下所有函数皆为数学运算函数的实现，使用 dispatch_common_ 函数来调用对应的 Python 方法并返回结果

// 实现加法运算，调用 dispatch_common_ 函数处理
c10::SymNode add(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现减法运算，调用 dispatch_common_ 函数处理
c10::SymNode sub(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现乘法运算，调用 dispatch_common_ 函数处理
c10::SymNode mul(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现真除法运算，调用 dispatch_common_ 函数处理
c10::SymNode truediv(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现浮点数真除法运算，调用 dispatch_common_ 函数处理
c10::SymNode float_truediv(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现整数真除法运算，调用 dispatch_common_ 函数处理
c10::SymNode int_truediv(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现乘方运算，调用 dispatch_common_ 函数处理
c10::SymNode pow(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现浮点数乘方运算，调用 dispatch_common_ 函数处理
c10::SymNode float_pow(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现以自然数为底的乘方运算，调用 dispatch_common_ 函数处理
c10::SymNode pow_by_natural(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现整数地板除法运算，调用 dispatch_common_ 函数处理
c10::SymNode floordiv(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现浮点数地板除法运算，调用 dispatch_common_ 函数处理
c10::SymNode int_floordiv(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现取模运算，调用 dispatch_common_ 函数处理
c10::SymNode mod(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现等于比较运算，调用 dispatch_common_ 函数处理
c10::SymNode eq(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现不等于比较运算，调用 dispatch_common_ 函数处理
c10::SymNode ne(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现大于比较运算，调用 dispatch_common_ 函数处理
c10::SymNode gt(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现小于比较运算，调用 dispatch_common_ 函数处理
c10::SymNode lt(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现小于等于比较运算，调用 dispatch_common_ 函数处理
c10::SymNode le(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现大于等于比较运算，调用 dispatch_common_ 函数处理
c10::SymNode ge(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现符号取最小值运算，调用 dispatch_common_ 函数处理
c10::SymNode sym_min(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现符号取最大值运算，调用 dispatch_common_ 函数处理
c10::SymNode sym_max(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现符号逻辑与运算，调用 dispatch_common_ 函数处理
c10::SymNode sym_and(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现符号逻辑或运算，调用 dispatch_common_ 函数处理
c10::SymNode sym_or(const c10::SymNode& other) override {
  return dispatch_common_(__func__, other);
}

// 实现符号条件三元运算，调用 dispatch_sym_ite_ 函数处理
c10::SymNode sym_ite(const c10::SymNode& other, const c10::SymNode& third)
    override {
  return dispatch_sym_ite_(__func__, other, third);
}

// 实现符号逻辑非运算，调用 dispatch_common_ 函数处理
c10::SymNode sym_not() override {
  // 调用 dispatch_common_ 函数，并将结果返回
  return dispatch_common_(__func__);
}

c10::SymNode ceil() override {
  // 调用 dispatch_common_ 函数，并将结果返回
  return dispatch_common_(__func__);
}

c10::SymNode floor() override {
  // 调用 dispatch_common_ 函数，并将结果返回
  return dispatch_common_(__func__);
}

c10::SymNode neg() override {
  // 调用 dispatch_common_ 函数，并将结果返回
  return dispatch_common_(__func__);
}

c10::SymNode clone() override {
  // 调用 dispatch_common_ 函数，并将结果返回
  return dispatch_common_(__func__);
}

c10::SymNode sym_float() override {
  // 调用 dispatch_common_ 函数，并将结果返回
  return dispatch_common_(__func__);
}

py::handle getPyObj() const {
  // 使用 pyobj_ 指针获取 Python 对象，并返回其 py::handle
  return py::handle(pyobj_->ptr(getPyInterpreter()));
}
std::shared_ptr<c10::SafePyObject> pyobj_ = nullptr;
};

} // namespace impl
} // namespace torch
```