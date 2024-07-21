# `.\pytorch\torch\csrc\jit\runtime\register_ops_utils.cpp`

```
#include <ATen/CPUGeneratorImpl.h>
// 包含 CPUGeneratorImpl.h 头文件，提供 CPU 生成器的实现

// TODO(antoniojkim): Add CUDA support for make_generator_for_device
// #ifdef USE_CUDA
// #include <ATen/cuda/CUDAGeneratorImpl.h>
// #endif

#ifdef USE_MPS
#include <ATen/mps/MPSGeneratorImpl.h>
#endif
// 如果定义了 USE_MPS 宏，则包含 MPSGeneratorImpl.h 头文件，提供 MPS 生成器的实现

#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/csrc/jit/runtime/slice_indices_adjust.h>
#include <limits>

#include <c10/util/irange.h>
// 包含 irange.h 头文件，提供 c10 命名空间下的范围工具

namespace torch::jit {

template <>
c10::impl::GenericList make_result_list<IValue>(const TypePtr& elemType) {
  return c10::impl::GenericList(elemType);
}
// 实例化 make_result_list 模板函数，返回一个根据 elemType 类型指针构造的泛型列表

template <>
void listIndex<at::Tensor>(Stack& stack) {
  at::Tensor elem = pop(stack).to<at::Tensor>();
  c10::List<at::Tensor> list = pop(stack).to<c10::List<at::Tensor>>();

  auto pos =
      std::find_if(list.begin(), list.end(), [elem](const at::Tensor& b) {
        const auto cmp_result = elem.eq(b);
        return at::native::is_nonzero(cmp_result);
      });
  // 在列表中查找元素 elem 的位置，并将其索引推入堆栈

  if (pos != list.end()) {
    push(stack, static_cast<int64_t>(std::distance(list.begin(), pos)));
  } else {
    AT_ERROR("'", elem, "' is not in list");
  }
  // 如果找到了元素 elem，则将其索引推入堆栈；否则，抛出错误信息
}

template <>
void listCount<at::Tensor>(Stack& stack) {
  at::Tensor elem = pop(stack).to<at::Tensor>();
  c10::List<at::Tensor> list = pop(stack).to<c10::List<at::Tensor>>();

  const int64_t count =
      std::count_if(list.begin(), list.end(), [&](const at::Tensor& b) {
        const auto cmp_result = elem.eq(b);
        return at::native::is_nonzero(cmp_result);
      });
  // 计算列表中与元素 elem 相等的元素数量，并将结果推入堆栈
  push(stack, count);
}

template <>
void listEq<at::Tensor>(Stack& stack) {
  c10::List<at::Tensor> b = pop(stack).to<c10::List<at::Tensor>>();
  c10::List<at::Tensor> a = pop(stack).to<c10::List<at::Tensor>>();
  push(stack, tensor_list_equal(a, b));
}
// 比较两个列表 a 和 b 是否相等，并将比较结果推入堆栈

template <>
void listNe<at::Tensor>(Stack& stack) {
  c10::List<at::Tensor> b = pop(stack).to<c10::List<at::Tensor>>();
  c10::List<at::Tensor> a = pop(stack).to<c10::List<at::Tensor>>();
  push(stack, !tensor_list_equal(a, b));
}
// 比较两个列表 a 和 b 是否不相等，并将比较结果推入堆栈

template <>
void listSort<at::Tensor>(Stack& stack) {
  bool reverse = pop(stack).toBool();
  c10::List<at::Tensor> list = pop(stack).toTensorList();
  std::sort(
      list.begin(),
      list.end(),
      [reverse](const at::Tensor& a, const at::Tensor& b) -> bool {
        // "strict weak ordering" issue - see other sort
        if (a.getIntrusivePtr() == b.getIntrusivePtr()) {
          return false;
        }
        return (at::native::is_nonzero(a.lt(b))) ^ reverse;
      });
}
// 对列表中的张量进行排序，可选择是否逆序排序，并将结果更新到原列表中

template <>
void listCopyAndSort<at::Tensor>(Stack& stack) {
  c10::List<at::Tensor> list = pop(stack).toTensorList();
  auto list_copied = list.copy();
  std::sort(
      list_copied.begin(),
      list_copied.end(),
      [](const at::Tensor& a, const at::Tensor& b) {
        return at::native::is_nonzero(a.lt(b));
      });
  push(stack, list_copied);
}
// 复制并对列表中的张量进行排序，并将排序后的列表推入堆栈

template <>
// 模板特化声明
void listRemove<at::Tensor>(Stack& stack) {
  // 从堆栈中弹出一个张量作为要移除的元素
  at::Tensor elem = pop(stack).to<at::Tensor>();
  // 从堆栈中弹出一个张量列表
  c10::List<at::Tensor> list = pop(stack).to<c10::List<at::Tensor>>();

  // 在列表中查找第一个与 elem 相等的张量
  auto pos = std::find_if(list.begin(), list.end(), [&](const at::Tensor& b) {
    // 比较 elem 和当前张量 b 的内容
    const auto cmp_result = elem.eq(b);
    // 返回是否比较结果为非零的布尔值
    return at::native::is_nonzero(cmp_result);
  });

  // 如果找到了匹配的张量，则从列表中移除它
  if (pos != list.end()) {
    list.erase(pos);
  } else {
    // 如果未找到匹配的张量，则抛出异常
    AT_ERROR("list.remove(x): x not in list");
  }
}

void checkImplicitTensorToNum(const at::Tensor& t, bool toInt) {
  // 如果张量 t 需要梯度，则抛出异常
  if (t.requires_grad()) {
    throw std::runtime_error(
        "Cannot input a tensor that requires grad as a scalar argument");
  }
  // 如果张量 t 的维度不为空，则抛出异常
  if (!t.sizes().empty()) {
    throw std::runtime_error(
        "Cannot input a tensor of dimension other than 0 as a scalar argument");
  }
  // 如果需要将张量 t 转换为整数，并且张量类型不是整数类型，则抛出异常
  if (toInt && !isIntegralType(t.scalar_type(), /*includeBool=*/false)) {
    std::stringstream ss;
    ss << "Cannot input a tensor of type " << t.scalar_type()
       << " as an integral argument";
    throw std::runtime_error(ss.str());
  }
}

void checkDoubleInRange(double a) {
  // 如果浮点数 a 是 NaN、无穷大，或者超出整型范围，则抛出异常
  if (std::isnan(a) || std::isinf(a) ||
      a > double(std::numeric_limits<int64_t>::max()) ||
      a < double(std::numeric_limits<int64_t>::min())) {
    throw c10::Error(
        "Cannot convert float " + std::to_string(a) + " to integer");
    return;
  }
}

int64_t partProduct(int n, int m) {
  // 如果 m 小于等于 n+1，则返回 n 强制类型转换为 int64_t
  if (m <= (n + 1))
    return (int64_t)n;
  // 如果 m 等于 n+2，则返回 n 乘以 m 强制类型转换为 int64_t
  if (m == (n + 2))
    return (int64_t)n * m;
  // 计算 n 和 m 之间的中间值 k，使用溢出安全的方式
  auto k = n + (m - n) / 2; // Overflow-safe midpoint
  // 如果 k 不是奇数，则将 k 减去 1
  if ((k & 1) != 1)
    k = k - 1;
  // 返回递归调用的部分乘积结果
  return partProduct(n, k) * partProduct(k + 2, m);
}

void loop(int n, int64_t& p, int64_t& r) {
  // 如果 n 小于等于 2，则直接返回
  if (n <= 2)
    return;
  // 递归调用自身，将 n 除以 2，并更新 p 和 r
  loop(n / 2, p, r);
  // 更新 p，将 p 乘以部分乘积的结果
  p = p * partProduct(n / 2 + 1 + ((n / 2) & 1), n - 1 + (n & 1));
  // 更新 r，将 r 乘以 p 的值
  r = r * p;
}

int nminussumofbits(int v) {
  // 将整数 v 转换为长整型 w
  long w = (long)v;
  // 使用位运算计算 v 的位反转和位求和的结果
  w -= (0xaaaaaaaa & w) >> 1; // NOLINT
  w = (w & 0x33333333) + ((w >> 2) & 0x33333333); // NOLINT
  w = (w + (w >> 4)) & 0x0f0f0f0f; // NOLINT
  w += w >> 8; // NOLINT
  w += w >> 16; // NOLINT
  // 返回 v 减去 w 的低八位值
  return v - (int)(w & 0xff); // NOLINT
}

int64_t factorial(int n) {
  // 如果 n 小于 0，则抛出异常
  if (n < 0) {
    throw std::runtime_error("factorial() not defined for negative values");
  }
  // 初始化变量 p 和 r，调用 loop 函数计算阶乘结果
  int64_t p = 1, r = 1;
  loop(n, p, r);
  // 返回 r 左移 n 的负位数之和
  return r << nminussumofbits(n);
}

double degrees(double x) {
  // 将角度 x 转换为弧度
  return x * radToDeg;
}
double radians(double x) {
  // 将弧度 x 转换为角度
  return x * degToRad;
}

void listAppend(Stack& stack) {
  // 从堆栈中弹出一个值作为要添加到列表的元素
  IValue el = pop(stack).to<IValue>();
  // 从堆栈中弹出一个列表
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  // 将元素 el 添加到列表末尾
  list.push_back(std::move(el));
  // 将更新后的列表重新推入堆栈
  push(stack, std::move(list));
}

void listReverse(Stack& stack) {
  // 从堆栈中弹出一个列表
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  // 将列表中的元素进行反转
  std::reverse(list.begin(), list.end());
}
void listPopImpl(Stack& stack, const char* empty_message) {
  // 从栈中弹出一个整数，作为要操作的列表的索引
  int64_t idx = pop(stack).to<int64_t>();
  // 从栈中弹出一个列表对象
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  // 获取列表的当前大小
  const int64_t list_size = list.size();
  // 规范化索引，确保在有效范围内
  const int64_t normalized_idx = normalizeIndex(idx, list_size);

  // 如果列表为空，则抛出错误信息
  if (list_size == 0) {
    AT_ERROR(empty_message);
  }

  // 将指定索引位置的元素推入栈中
  push(stack, getItem(list, idx));
  // 删除列表中规范化后的索引位置的元素
  list.erase(list.begin() + normalized_idx);
}

void listPop(Stack& stack) {
  // 调用 listPopImpl 函数，并传入错误消息字符串
  return listPopImpl(stack, "pop from empty list");
}

void listClear(Stack& stack) {
  // 从栈中弹出一个列表对象，并清空该列表
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();
  list.clear();
}

void listDelete(Stack& stack) {
  // 调用 listPopImpl 函数，并传入错误消息字符串，删除列表中指定索引位置的元素
  listPopImpl(stack, "pop index out of range");
  // 从栈中弹出一个元素，但不使用它
  pop(stack);
}

void listInsert(Stack& stack) {
  // 从栈中弹出一个元素、一个整数作为索引和一个列表对象
  IValue elem = pop(stack).to<IValue>();
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  // 获取列表的当前大小和规范化索引
  const int64_t list_size = list.size();
  const int64_t normalized_idx = normalizeIndex(idx, list_size);

  // 如果规范化后的索引超出了列表范围，则根据情况在列表开头或结尾插入元素
  if (normalized_idx < 0 || normalized_idx >= list_size) {
    if (normalized_idx < 0) {
      list.insert(list.begin(), elem);
    } else {
      list.push_back(elem);
    }
  } else {
    // 在规范化后的索引位置插入元素
    list.insert(list.begin() + normalized_idx, elem);
  }
}

void listExtend(Stack& stack) {
  // 从栈中弹出两个列表对象，将第二个列表中的所有元素扩展到第一个列表中
  c10::List<IValue> b = pop(stack).to<c10::List<IValue>>();
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();

  a.reserve(a.size() + b.size()); // 预留足够的空间以容纳扩展后的列表元素
  for (const auto i : c10::irange(b.size())) {
    a.push_back(b.get(i)); // 逐个将第二个列表中的元素添加到第一个列表中
  }
}

void listCopy(Stack& stack) {
  // 从栈中弹出一个列表对象，并将其复制一份推入栈中
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();
  push(stack, list.copy());
}

void listSelect(Stack& stack) {
  // 从栈中弹出一个整数作为索引和一个列表对象，将列表中指定索引位置的元素推入栈中
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  push(stack, getItem(list, idx));
}

void listLen(Stack& stack) {
  // 从栈中弹出一个列表对象，并将其大小推入栈中
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();

  const int64_t size = a.size(); // 获取列表的大小
  push(stack, size); // 将列表的大小推入栈中
}

void listList(Stack& stack) {
  // 从栈中弹出一个列表对象，并将其复制一份推入栈中
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();
  push(stack, a.copy());
}

void listAdd(Stack& stack) {
  // 从栈中弹出两个列表对象，将它们合并为一个新的列表对象，并推入栈中
  c10::List<IValue> b = pop(stack).to<c10::List<IValue>>();
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();

  // 创建一个结果列表对象，以第一个列表的元素类型为准
  c10::List<IValue> ret = make_result_list<IValue>(a.elementType());

  // 如果第一个列表只被一个引用使用，则可以直接移动它的内容
  if (a.use_count() == 1) {
    ret = std::move(a);
  } else {
    ret = a.copy(); // 否则，复制第一个列表的内容到结果列表中
  }

  ret.append(std::move(b)); // 将第二个列表的元素追加到结果列表的末尾

  push(stack, std::move(ret)); // 将结果列表推入栈中
}

void listInplaceAdd(Stack& stack) {
  // 从栈中弹出两个列表对象，将第二个列表的所有元素追加到第一个列表中，并推入栈中
  c10::List<IValue> b = pop(stack).to<c10::List<IValue>>();
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();
  a.append(std::move(b)); // 将第二个列表的元素追加到第一个列表的末尾
  push(stack, std::move(a)); // 将修改后的第一个列表推入栈中
}

void listMulIntLeftInPlace(Stack& stack) {
  // 从栈中弹出一个整数和一个列表对象，根据整数值对列表进行复制或清空操作
  int64_t n = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();
  if (n <= 0) {
    list.clear(); // 如果整数值小于等于零，则清空列表
  } else if (n > 1) {
    size_t list_size = list.size();
    // 复制列表元素 n-1 次，即总共复制 n 份
    for (int i = 1; i < n; ++i) {
      list.reserve(list.size() + list_size);
      for (const auto j : c10::irange(list_size)) {
        list.push_back(list.get(j));
      }
    }
  }
  // 将修改后的列表推入栈中
  push(stack, std::move(list));
}
    // 对于范围从1到n的每个值i进行循环
    for (const auto i : c10::irange(1, n)) {
      // 忽略未使用变量的警告，将i声明为void类型
      (void)i; // Suppress unused variable warning
      // 对于列表大小的每个值j进行循环
      for (const auto j : c10::irange(list_size)) {
        // 向列表末尾添加当前索引处的列表元素，将其推送到列表中
        list.push_back(list.get(j));
      }
    }
  }

  // 将列表作为右值推送到堆栈中
  push(stack, std::move(list));
}

// 函数：listMulIntLeft
void listMulIntLeft(Stack& stack) {
  // 从栈中取出整数 n 和列表 list
  int64_t n = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  // 创建一个与 list 元素类型相同的空列表 ret
  c10::List<IValue> ret = make_result_list<IValue>(list.elementType());
  // 计算结果列表 ret 的预留空间大小为 list.size() * n
  const auto size = list.size() * n;
  ret.reserve(size);

  // 嵌套循环，将列表 list 的每个元素复制 n 次添加到 ret 中
  for (const auto i : c10::irange(n)) {
    (void)i; // 抑制未使用变量警告
    for (IValue e : list) {
      ret.push_back(std::move(e));
    }
  }

  // 将结果列表 ret 推回栈中
  push(stack, std::move(ret));
}

// 函数：listMulIntRight
void listMulIntRight(Stack& stack) {
  // 从栈中取出列表 list 和整数 n
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();
  int64_t n = pop(stack).to<int64_t>();

  // 创建一个与 list 元素类型相同的空列表 ret
  c10::List<IValue> ret = make_result_list<IValue>(list.elementType());
  // 计算结果列表 ret 的预留空间大小为 list.size() * n
  const auto size = list.size() * n;
  ret.reserve(size);

  // 嵌套循环，将列表 list 的每个元素复制 n 次添加到 ret 中
  for (const auto i : c10::irange(n)) {
    (void)i; // 抑制未使用变量警告
    for (IValue e : list) {
      ret.push_back(std::move(e));
    }
  }

  // 将结果列表 ret 推回栈中
  push(stack, std::move(ret));
}

// 函数：listSlice
void listSlice(Stack& stack) {
  // 从栈中取出 start_val、end_val 和 step_val
  auto step_val = pop(stack);
  auto end_val = pop(stack);
  auto start_val = pop(stack);

  // 默认情况下，start 和 end 均为 None，按照 Python 约定转换为 INT64_MAX。
  // 如果未给定 step，则默认为 1。
  int64_t step = step_val.isInt() ? step_val.to<int64_t>() : 1;
  int64_t end = end_val.isInt() ? end_val.to<int64_t>()
                                : std::numeric_limits<int64_t>::max();
  int64_t start = start_val.isInt() ? start_val.to<int64_t>()
                                    : std::numeric_limits<int64_t>::max();

  // 从栈中取出列表 list
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  // 计算列表的大小
  const int64_t list_size = list.size();

  // 创建一个与 list 元素类型相同的空列表 sliced_list
  c10::List<IValue> sliced_list = make_result_list<IValue>(list.elementType());
  // 调整 start、end 和 step 后，计算 sliced_list 的预留空间大小
  const int64_t num_values =
      slice_indices_adjust(list_size, &start, &end, step);
  sliced_list.reserve(num_values);

  // 根据调整后的索引从 list 中获取元素填充 sliced_list
  int i = start;
  for (const auto j : c10::irange(num_values)) {
    (void)j; // 抑制未使用变量警告
    sliced_list.push_back(list.get(i));
    i += step;
  }

  // 将结果列表 sliced_list 推回栈中
  push(stack, std::move(sliced_list));
}

// 函数：listSetItem
void listSetItem(Stack& stack) {
  // 从栈中取出 value、idx 和 list
  IValue value = pop(stack).to<IValue>();
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  // 将 value 插入到 list 的指定索引 idx 处
  setItem(list, idx, std::move(value));

  // 将修改后的 list 推回栈中
  push(stack, std::move(list));
}

// 函数：make_generator_for_device
at::Generator make_generator_for_device(
    c10::Device device,
    std::optional<int64_t> seed) {
  // 如果设备为 CPU
  if (device.is_cpu()) {
    // 如果提供了种子值，则创建一个对应的 CPU 生成器
    if (seed.has_value()) {
      return at::detail::createCPUGenerator(seed.value());
    } else {
      // 否则，创建一个默认的 CPU 生成器
      return at::detail::createCPUGenerator();
    }
// TODO(antoniojkim): 启用对 CUDA 设备的支持
//                    实现以下部分在 rocm 构建期间会导致问题
// #ifdef USE_CUDA
//   } else if (device.is_cuda()) {
//     // 如果设备为 CUDA，则创建对应的 CUDA 生成器
//     auto generator = at::cuda::detail::createCUDAGenerator(device.index());
//     // 如果提供了种子值，则设置当前的种子值
//     if (seed.has_value()) {
//       generator.set_current_seed(seed.value());
//     }
//     return generator;
#ifdef USE_MPS
  // 如果编译器定义了 USE_MPS 宏，则执行以下代码块
  } else if (device.is_mps()) {
    // 如果设备是 MPS（Multi-Process Service），则执行以下代码块
    if (seed.has_value()) {
      // 如果 seed 有值，则使用该值创建 MPS 生成器
      return at::mps::detail::createMPSGenerator(seed.value());
    } else {
      // 如果 seed 没有值，则创建不带 seed 的 MPS 生成器
      return at::mps::detail::createMPSGenerator();
    }
#endif
  } else {
    // 如果以上条件都不满足，则抛出错误，指出不支持当前设备
    AT_ERROR(
        "Unsupported device for at::make_generator_for_device found: ",
        device.str());
  }
}

} // namespace torch::jit
```