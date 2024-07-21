# `.\pytorch\aten\src\ATen\test\scalar_tensor_test.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/ATen.h>  // 引入 PyTorch 的 ATen 头文件
#include <ATen/Utils.h>  // 引入 PyTorch 的 ATen 工具头文件
#include <c10/util/accumulate.h>  // 引入 PyTorch 的 c10 库中的 accumulate 头文件

#include <algorithm>  // 引入算法标准库头文件，用于 STL 算法
#include <iostream>  // 引入输入输出流库头文件
#include <numeric>  // 引入数值算法库头文件，用于一些数值计算

using namespace at;  // 使用 at 命名空间

#define TRY_CATCH_ELSE(fn, catc, els)                          \
  {                                                            \
    /* 避免在 els 代码抛出异常时错误地通过 */ \
    bool _passed = false;                                      \
    try {                                                      \
      fn;                                                      \
      _passed = true;                                          \
      els;                                                     \
    } catch (std::exception & e) {                             \
      ASSERT_FALSE(_passed);                                   \
      catc;                                                    \
    }                                                          \
  }

void require_equal_size_dim(const Tensor &lhs, const Tensor &rhs) {
  ASSERT_EQ(lhs.dim(), rhs.dim());  // 断言左右张量的维度相等
  ASSERT_TRUE(lhs.sizes().equals(rhs.sizes()));  // 断言左右张量的尺寸大小相等
}

bool should_expand(const IntArrayRef &from_size, const IntArrayRef &to_size) {
  if (from_size.size() > to_size.size()) {  // 如果源尺寸维度大于目标尺寸维度
    return false;  // 返回不扩展
  }
  for (auto from_dim_it = from_size.rbegin(); from_dim_it != from_size.rend();
       ++from_dim_it) {
    for (auto to_dim_it = to_size.rbegin(); to_dim_it != to_size.rend();
         ++to_dim_it) {
      if (*from_dim_it != 1 && *from_dim_it != *to_dim_it) {
        return false;  // 如果源尺寸维度不为1且与目标尺寸维度不相等，返回不扩展
      }
    }
  }
  return true;  // 否则返回可以扩展
}

void test(DeprecatedTypeProperties &T) {
  std::vector<std::vector<int64_t>> sizes = {{}, {0}, {1}, {1, 1}, {2}};

  // single-tensor/size tests
  for (auto s = sizes.begin(); s != sizes.end(); ++s) {
    // 验证张量的维度、尺寸、步长等是否与请求匹配
    auto t = ones(*s, T);  // 创建一个由1组成的张量，尺寸为*s，类型为T
    ASSERT_EQ((size_t)t.dim(), s->size());  // 断言张量的维度等于请求的维度大小
    ASSERT_EQ((size_t)t.ndimension(), s->size());  // 断言张量的维度等于请求的维度大小
    ASSERT_TRUE(t.sizes().equals(*s));  // 断言张量的尺寸与请求的尺寸相等
    ASSERT_EQ(t.strides().size(), s->size());  // 断言张量的步长数等于请求的维度大小
    const auto numel = c10::multiply_integers(s->begin(), s->end());
    ASSERT_EQ(t.numel(), numel);  // 断言张量的元素个数等于请求的元素个数
    // 验证能否输出张量
    std::stringstream ss;
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_NO_THROW(ss << t << std::endl);

    // set_
    auto t2 = ones(*s, T);  // 创建一个由1组成的张量，尺寸为*s，类型为T
    t2.set_();  // 将张量t2设置为空张量
    require_equal_size_dim(t2, ones({0}, T));  // 要求空张量t2与尺寸为{0}的全1张量相等

    // unsqueeze
    ASSERT_EQ(t.unsqueeze(0).dim(), t.dim() + 1);  // 断言在指定维度(0)上挤压后的张量维度增加1

    // unsqueeze_
    {
      auto t2 = ones(*s, T);  // 创建一个由1组成的张量，尺寸为*s，类型为T
      auto r = t2.unsqueeze_(0);  // 在维度(0)上原地挤压张量t2
      ASSERT_EQ(r.dim(), t.dim() + 1);  // 断言原地挤压后的张量维度增加1
    }

    // squeeze (with dimension argument)
    if (t.dim() == 0 || t.sizes()[0] == 1) {
      ASSERT_EQ(t.squeeze(0).dim(), std::max<int64_t>(t.dim() - 1, 0));  // 如果张量维度为0或者第一维大小为1，挤压后的维度减1
    } else {
      // 在 PyTorch 中，试图挤压一个非大小为1的维度是一个无操作；而在 NumPy 中会报错
      ASSERT_EQ(t.squeeze(0).dim(), t.dim());  // 断言挤压后的维度与原维度相等
    }
    // squeeze (with no dimension argument)
    {
      // 创建一个向量，用于存储除了大小为1之外的所有维度
      std::vector<int64_t> size_without_ones;
      // 遍历张量 *s 的大小
      for (auto size : *s) {
        // 如果大小不为1，则将其加入 size_without_ones 向量中
        if (size != 1) {
          size_without_ones.push_back(size);
        }
      }
      // 执行 squeeze 操作，返回结果存储在 result 中
      auto result = t.squeeze();
      // 断言 result 的大小与 size_without_ones 生成的全1张量的大小相等
      require_equal_size_dim(result, ones(size_without_ones, T));
    }

    {
      // squeeze_ (with dimension argument)
      // 创建一个全1张量 t2，其大小与 *s 和类型 T 相关
      auto t2 = ones(*s, T);
      // 如果 t2 的维度为0或第一个维度大小为1
      if (t2.dim() == 0 || t2.sizes()[0] == 1) {
        // 对 t2 执行 squeeze_ 操作，并断言结果的维度与 t 的维度减1后的最大值相等
        ASSERT_EQ(t2.squeeze_(0).dim(), std::max<int64_t>(t.dim() - 1, 0));
      } else {
        // 在 PyTorch 中，尝试对大小不为1的维度执行 squeeze 是无效操作；在 NumPy 中会报错
        ASSERT_EQ(t2.squeeze_(0).dim(), t.dim());
      }
    }

    // squeeze_ (with no dimension argument)
    {
      // 创建一个全1张量 t2，其大小与 *s 和类型 T 相关
      auto t2 = ones(*s, T);
      // 创建一个向量，用于存储除了大小为1之外的所有维度
      std::vector<int64_t> size_without_ones;
      // 遍历张量 *s 的大小
      for (auto size : *s) {
        // 如果大小不为1，则将其加入 size_without_ones 向量中
        if (size != 1) {
          size_without_ones.push_back(size);
        }
      }
      // 执行 squeeze_ 操作，返回结果存储在 r 中
      auto r = t2.squeeze_();
      // 断言 t2 的大小与 size_without_ones 生成的全1张量的大小相等
      require_equal_size_dim(t2, ones(size_without_ones, T));
    }

    // reduce (with dimension argument and with 1 return argument)
    // 如果张量 t 的元素数不为0
    if (t.numel() != 0) {
      // 断言对 t 沿第0维度求和后的张量的维度与 t 的维度减1后的最大值相等
      ASSERT_EQ(t.sum(0).dim(), std::max<int64_t>(t.dim() - 1, 0));
    } else {
      // 如果 t 的元素数为0，则断言 t 沿第0维度求和后与全0张量相等
      ASSERT_TRUE(t.sum(0).equal(at::zeros({}, T)));
    }

    // reduce (with dimension argument and with 2 return arguments)
    // 如果张量 t 的元素数不为0
    if (t.numel() != 0) {
      // 计算 t 沿第0维度的最小值
      auto ret = t.min(0);
      // 断言返回的第一个张量的维度与 t 的维度减1后的最大值相等
      ASSERT_EQ(std::get<0>(ret).dim(), std::max<int64_t>(t.dim() - 1, 0));
      // 断言返回的第二个张量的维度与 t 的维度减1后的最大值相等
      ASSERT_EQ(std::get<1>(ret).dim(), std::max<int64_t>(t.dim() - 1, 0));
    } else {
      // 如果 t 的元素数为0，则断言调用 t.min(0) 会抛出异常
      ASSERT_ANY_THROW(t.min(0));
    }

    // simple indexing
    // 如果张量 t 的维度大于0且元素数不为0
    if (t.dim() > 0 && t.numel() != 0) {
      // 断言访问 t 的第一个元素后的张量的维度与 t 的维度减1后的最大值相等
      ASSERT_EQ(t[0].dim(), std::max<int64_t>(t.dim() - 1, 0));
    } else {
      // 如果 t 的维度为0或元素数为0，则断言访问 t 的第一个元素会抛出异常
      ASSERT_ANY_THROW(t[0]);
    }

    // fill_ (argument to fill_ can only be a 0-dim tensor)
    // 尝试使用 t.sum(0) 的结果来填充张量 t，断言 t 的维度大于1或不大于1
    TRY_CATCH_ELSE(
        t.fill_(t.sum(0)), ASSERT_GT(t.dim(), 1), ASSERT_LE(t.dim(), 1));
  }

  // 遍历 sizes 的迭代器
  for (auto lhs_it = sizes.begin(); lhs_it != sizes.end(); ++lhs_it) {
    // NOLINTNEXTLINE(modernize-loop-convert)
    // 略过现代化循环转换的 LINT 错误
  }
}
}

TEST(TestScalarTensor, TestScalarTensorCPU) {
  // 设置随机种子为123
  manual_seed(123);
  // 调用测试函数，使用 CPU 上的浮点数张量
  test(CPU(kFloat));
}

TEST(TestScalarTensor, TestScalarTensorCUDA) {
  // 设置随机种子为123
  manual_seed(123);

  // 检查是否支持 CUDA
  if (at::hasCUDA()) {
    // 如果支持 CUDA，则调用测试函数，使用 CUDA 上的浮点数张量
    test(CUDA(kFloat));
  }
}

TEST(TestScalarTensor, TestScalarTensorMPS) {
  // 设置随机种子为123
  manual_seed(123);

  // 检查是否支持 MPS
  if (at::hasMPS()) {
    // 如果支持 MPS，则调用测试函数，使用 MPS 上的浮点数张量
    test(MPS(kFloat));
  }
}
```