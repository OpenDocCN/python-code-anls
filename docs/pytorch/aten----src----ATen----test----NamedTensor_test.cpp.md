# `.\pytorch\aten\src\ATen\test\NamedTensor_test.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 的头文件，用于单元测试

#include <ATen/ATen.h>  // 包含 PyTorch 的 ATen 头文件
#include <ATen/NamedTensorUtils.h>  // 包含命名张量相关的工具函数
#include <ATen/TensorNames.h>  // 包含张量命名相关的功能
#include <c10/util/Exception.h>  // 包含 C10 库中的异常处理
#include <c10/util/irange.h>  // 包含 C10 库中用于迭代范围的函数

using at::Dimname;  // 使用 ATen 命名空间中的 Dimname 类
using at::DimnameList;  // 使用 ATen 命名空间中的 DimnameList 类
using at::Symbol;  // 使用 ATen 命名空间中的 Symbol 类
using at::namedinference::TensorName;  // 使用 ATen 命名空间中命名推断相关的 TensorName 类
using at::namedinference::TensorNames;  // 使用 ATen 命名空间中命名推断相关的 TensorNames 类

// 从字符串创建 Dimname 对象的静态函数
static Dimname dimnameFromString(const std::string& str) {
  return Dimname::fromSymbol(Symbol::dimname(str));
}

// 命名为 NamedTensorTest 的测试套件，测试张量是否具有命名维度
TEST(NamedTensorTest, isNamed) {
  auto tensor = at::zeros({3, 2, 5, 7});  // 创建一个全零张量
  ASSERT_FALSE(tensor.has_names());  // 断言张量没有命名

  tensor = at::zeros({3, 2, 5, 7});  // 重新创建一个全零张量
  ASSERT_FALSE(tensor.has_names());  // 断言张量没有命名

  tensor = at::zeros({3, 2, 5, 7});  // 重新创建一个全零张量
  auto N = dimnameFromString("N");  // 创建命名维度对象 N
  auto C = dimnameFromString("C");  // 创建命名维度对象 C
  auto H = dimnameFromString("H");  // 创建命名维度对象 H
  auto W = dimnameFromString("W");  // 创建命名维度对象 W
  std::vector<Dimname> names = { N, C, H, W };  // 创建包含命名维度的向量
  at::internal_set_names_inplace(tensor, names);  // 在张量中就地设置命名维度
  ASSERT_TRUE(tensor.has_names());  // 断言张量现在具有命名
}

// 比较两个 DimnameList 是否相等的静态函数
static bool dimnames_equal(at::DimnameList names, at::DimnameList other) {
  if (names.size() != other.size()) {  // 如果两个列表大小不同，返回 false
    return false;
  }
  for (const auto i : c10::irange(names.size())) {  // 遍历列表中的每个元素
    const auto& name = names[i];  // 获取当前列表中的名称
    const auto& other_name = other[i];  // 获取另一个列表中的名称
    if (name.type() != other_name.type() || name.symbol() != other_name.symbol()) {  // 如果名称类型或符号不相同，返回 false
      return false;
    }
  }
  return true;  // 列表相等，返回 true
}

// 命名为 NamedTensorTest 的测试套件，测试在张量上附加元数据
TEST(NamedTensorTest, attachMetadata) {
  auto tensor = at::zeros({3, 2, 5, 7});  // 创建一个全零张量
  auto N = dimnameFromString("N");  // 创建命名维度对象 N
  auto C = dimnameFromString("C");  // 创建命名维度对象 C
  auto H = dimnameFromString("H");  // 创建命名维度对象 H
  auto W = dimnameFromString("W");  // 创建命名维度对象 W
  std::vector<Dimname> names = { N, C, H, W };  // 创建包含命名维度的向量

  at::internal_set_names_inplace(tensor, names);  // 在张量中就地设置命名维度

  const auto retrieved_meta = tensor.get_named_tensor_meta();  // 获取张量的命名张量元数据
  ASSERT_TRUE(dimnames_equal(retrieved_meta->names(), names));  // 断言张量的命名维度与预期相同

  // 测试删除元数据
  tensor.unsafeGetTensorImpl()->set_named_tensor_meta(nullptr);  // 设置张量的命名张量元数据为空指针
  ASSERT_FALSE(tensor.has_names());  // 断言张量没有命名
}

// 命名为 NamedTensorTest 的测试套件，测试在张量上就地设置命名维度
TEST(NamedTensorTest, internalSetNamesInplace) {
  auto tensor = at::zeros({3, 2, 5, 7});  // 创建一个全零张量
  auto N = dimnameFromString("N");  // 创建命名维度对象 N
  auto C = dimnameFromString("C");  // 创建命名维度对象 C
  auto H = dimnameFromString("H");  // 创建命名维度对象 H
  auto W = dimnameFromString("W");  // 创建命名维度对象 W
  std::vector<Dimname> names = { N, C, H, W };  // 创建包含命名维度的向量
  ASSERT_FALSE(tensor.has_names());  // 断言张量没有命名

  // 设置命名维度
  at::internal_set_names_inplace(tensor, names);
  const auto retrieved_names = tensor.opt_names().value();  // 获取张量的可选命名维度
  ASSERT_TRUE(dimnames_equal(retrieved_names, names));  // 断言设置的命名维度与预期相同

  // 删除命名维度
  at::internal_set_names_inplace(tensor, at::nullopt);  // 在张量上清除命名维度
  ASSERT_TRUE(tensor.get_named_tensor_meta() == nullptr);  // 断言张量的命名张量元数据为空
  ASSERT_TRUE(tensor.opt_names() == at::nullopt);  // 断言张量没有命名维度
}
TEST(NamedTensorTest, empty) {
  // 创建 Dimname 对象 N, C, H, W 分别对应维度名 "N", "C", "H", "W"
  auto N = Dimname::fromSymbol(Symbol::dimname("N"));
  auto C = Dimname::fromSymbol(Symbol::dimname("C"));
  auto H = Dimname::fromSymbol(Symbol::dimname("H"));
  auto W = Dimname::fromSymbol(Symbol::dimname("W"));
  // 创建 Dimname 向量 names，包含 N, C, H, W 四个维度名
  std::vector<Dimname> names = { N, C, H, W };

  // 创建一个空的 Tensor 对象 tensor
  auto tensor = at::empty({});
  // 断言 tensor 没有命名维度
  ASSERT_EQ(tensor.opt_names(), at::nullopt);

  // 重新创建一个空的 Tensor 对象 tensor
  tensor = at::empty({1, 2, 3});
  // 再次断言 tensor 没有命名维度
  ASSERT_EQ(tensor.opt_names(), at::nullopt);

  // 创建一个带有指定维度名 names 的 Tensor 对象 tensor
  tensor = at::empty({1, 2, 3, 4}, names);
  // 断言 tensor 的命名维度与 names 相同
  ASSERT_TRUE(dimnames_equal(tensor.opt_names().value(), names));

  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言创建带有指定维度名 names 的 Tensor 对象时会抛出 c10::Error 异常
  ASSERT_THROW(at::empty({1, 2, 3}, names), c10::Error);
}

TEST(NamedTensorTest, dimnameToPosition) {
  // 创建 Dimname 对象 N, C, H, W 分别对应维度名 "N", "C", "H", "W"
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  // 创建 Dimname 向量 names，包含 N, C, H, W 四个维度名
  std::vector<Dimname> names = { N, C, H, W };

  // 创建一个空的 Tensor 对象 tensor
  tensor = at::empty({1, 1, 1});
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言在不含有指定维度名的情况下，调用 dimname_to_position 函数会抛出 c10::Error 异常
  ASSERT_THROW(dimname_to_position(tensor, N), c10::Error);

  // 创建一个带有指定维度名 names 的 Tensor 对象 tensor
  tensor = at::empty({1, 1, 1, 1}, names);
  // 断言调用 dimname_to_position 函数返回维度名 H 在 names 中的位置，即 2
  ASSERT_EQ(dimname_to_position(tensor, H), 2);
}

static std::vector<Dimname> tensornames_unify_from_right(
    DimnameList names,
    DimnameList other_names) {
  // 创建 TensorNames 对象 names_wrapper 和 other_wrapper 分别包装 names 和 other_names
  auto names_wrapper = at::namedinference::TensorNames(names);
  auto other_wrapper = at::namedinference::TensorNames(other_names);
  // 调用 TensorNames::unifyFromRightInplace 方法，将 names 和 other_names 合并并返回结果
  return names_wrapper.unifyFromRightInplace(other_wrapper).toDimnameVec();
}

static void check_unify(
    DimnameList names,
    DimnameList other_names,
    DimnameList expected) {
  // 检查使用 legacy at::unify_from_right 方法合并 names 和 other_names 的结果
  const auto result = at::unify_from_right(names, other_names);
  // 断言合并结果与预期结果 expected 相同
  ASSERT_TRUE(dimnames_equal(result, expected));

  // 检查使用 TensorNames::unifyFromRight 方法合并 names 和 other_names 的结果
  // 在未来，at::unify_from_right 和 TensorNames::unifyFromRight 可能会合并
  // 目前分别测试它们的结果
  const auto also_result = tensornames_unify_from_right(names, other_names);
  // 断言合并结果与预期结果 expected 相同
  ASSERT_TRUE(dimnames_equal(also_result, expected));
}

static void check_unify_error(DimnameList names, DimnameList other_names) {
  // 在未来，at::unify_from_right 和 TensorNames::unifyFromRight 可能会合并
  // 目前分别测试它们的异常情况
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言合并 names 和 other_names 会抛出 c10::Error 异常
  ASSERT_THROW(at::unify_from_right(names, other_names), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言调用 tensornames_unify_from_right 函数合并 names 和 other_names 会抛出 c10::Error 异常
  ASSERT_THROW(tensornames_unify_from_right(names, other_names), c10::Error);
}
TEST(NamedTensorTest, unifyFromRight) {
  auto N = dimnameFromString("N");  // 创建名为 "N" 的 Dimname 对象
  auto C = dimnameFromString("C");  // 创建名为 "C" 的 Dimname 对象
  auto H = dimnameFromString("H");  // 创建名为 "H" 的 Dimname 对象
  auto W = dimnameFromString("W");  // 创建名为 "W" 的 Dimname 对象
  auto None = dimnameFromString("*");  // 创建名为 "*" 的 Dimname 对象

  std::vector<Dimname> names = { N, C };  // 创建包含 N 和 C 的 Dimname 向量

  // 测试维度名统一化函数的不同情况
  check_unify({ N, C, H, W }, { N, C, H, W }, { N, C, H, W });
  check_unify({ W }, { C, H, W }, { C, H, W });
  check_unify({ None, W }, { C, H, W }, { C, H, W });
  check_unify({ None, None, H, None }, { C, None, W }, { None, C, H, W });

  // 测试维度名统一化函数的错误情况
  check_unify_error({ W, H }, { W, C });
  check_unify_error({ W, H }, { C, H });
  check_unify_error({ None, H }, { H, None });
  check_unify_error({ H, None, C }, { H });
}

TEST(NamedTensorTest, alias) {
  // 在 Python 中未公开 tensor.alias 方法，这里测试其名称传播
  auto N = dimnameFromString("N");  // 创建名为 "N" 的 Dimname 对象
  auto C = dimnameFromString("C");  // 创建名为 "C" 的 Dimname 对象
  std::vector<Dimname> names = { N, C };  // 创建包含 N 和 C 的 Dimname 向量

  // 创建一个空张量并创建其别名
  auto tensor = at::empty({2, 3}, std::vector<Dimname>{ N, C });
  auto aliased = tensor.alias();
  ASSERT_TRUE(dimnames_equal(tensor.opt_names().value(), aliased.opt_names().value()));
}

TEST(NamedTensorTest, NoNamesGuard) {
  auto N = dimnameFromString("N");  // 创建名为 "N" 的 Dimname 对象
  auto C = dimnameFromString("C");  // 创建名为 "C" 的 Dimname 对象
  std::vector<Dimname> names = { N, C };  // 创建包含 N 和 C 的 Dimname 向量

  // 创建一个空张量并使用命名模式
  auto tensor = at::empty({2, 3}, names);
  ASSERT_TRUE(at::NamesMode::is_enabled());
  {
    // 禁用命名模式的临时保护区域
    at::NoNamesGuard guard;
    ASSERT_FALSE(at::NamesMode::is_enabled());
    ASSERT_FALSE(tensor.opt_names());
    ASSERT_FALSE(at::impl::get_opt_names(tensor.unsafeGetTensorImpl()));
  }
  ASSERT_TRUE(at::NamesMode::is_enabled());
}

static std::vector<Dimname> nchw() {
  auto N = dimnameFromString("N");  // 创建名为 "N" 的 Dimname 对象
  auto C = dimnameFromString("C");  // 创建名为 "C" 的 Dimname 对象
  auto H = dimnameFromString("H");  // 创建名为 "H" 的 Dimname 对象
  auto W = dimnameFromString("W");  // 创建名为 "W" 的 Dimname 对象
  return { N, C, H, W };  // 返回包含 N, C, H, W 的 Dimname 向量
}

TEST(NamedTensorTest, TensorNamePrint) {
  auto names = nchw();  // 获取 NCHW 维度名向量
  {
    auto N = TensorName(names, 0);  // 创建索引为 0 的 "N" 的 TensorName 对象
    ASSERT_EQ(
        c10::str(N),
        "'N' (index 0 of ['N', 'C', 'H', 'W'])");  // 断言 N 的字符串表示
  }
  {
    auto H = TensorName(names, 2);  // 创建索引为 2 的 "H" 的 TensorName 对象
    ASSERT_EQ(
        c10::str(H),
        "'H' (index 2 of ['N', 'C', 'H', 'W'])");  // 断言 H 的字符串表示
  }
}

TEST(NamedTensorTest, TensorNamesCheckUnique) {
  auto names = nchw();  // 获取 NCHW 维度名向量
  {
    // 烟雾测试以确保不会抛出异常
    TensorNames(names).checkUnique("op_name");
  }
  {
    std::vector<Dimname> nchh = { names[0], names[1], names[2], names[2] };
    auto tensornames = TensorNames(nchh);
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_THROW(tensornames.checkUnique("op_name"), c10::Error);  // 断言重复名称引发异常
  }
}
```