# `.\pytorch\c10\test\util\tempfile_test.cpp`

```py
#include <c10/util/tempfile.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <optional>

#if !defined(_WIN32)
// 检查文件是否存在的函数，非Windows平台使用 POSIX stat 结构
static bool file_exists(const char* path) {
  struct stat st {};  // 定义 stat 结构体
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);  // 调用 stat 函数检查文件状态
}
// 检查目录是否存在的函数，非Windows平台使用 POSIX stat 结构
static bool directory_exists(const char* path) {
  struct stat st {};  // 定义 stat 结构体
  return stat(path, &st) == 0 && S_ISDIR(st.st_mode);  // 调用 stat 函数检查目录状态
}
#else
// 检查文件是否存在的函数，Windows平台使用 _stat 结构
static bool file_exists(const char* path) {
  struct _stat st {};  // 定义 _stat 结构体
  return _stat(path, &st) == 0 && ((st.st_mode & _S_IFMT) == _S_IFREG);  // 调用 _stat 函数检查文件状态
}
// 检查目录是否存在的函数，Windows平台使用 _stat 结构
static bool directory_exists(const char* path) {
  struct _stat st {};  // 定义 _stat 结构体
  return _stat(path, &st) == 0 && ((st.st_mode & _S_IFMT) == _S_IFDIR);  // 调用 _stat 函数检查目录状态
}
#endif // !defined(_WIN32)

// TempFileTest 测试案例，验证临时文件名是否符合预期模式
TEST(TempFileTest, MatchesExpectedPattern) {
  // 创建一个临时文件对象，文件名以 "test-pattern-" 开头
  c10::TempFile file = c10::make_tempfile("test-pattern-");

#if defined(_WIN32)
  ASSERT_TRUE(file.open());  // 在Windows平台确保临时文件打开成功
#endif
  ASSERT_TRUE(file_exists(file.name.c_str()));  // 断言临时文件确实存在

#if !defined(_WIN32)
  // 在非Windows平台，断言临时文件名包含 "test-pattern-" 这个字符串
  ASSERT_NE(file.name.find("test-pattern-"), std::string::npos);
#endif // !defined(_WIN32)
}

// TempDirTest 测试案例，验证临时目录的创建和销毁
TEST(TempDirTest, tryMakeTempdir) {
  // 创建一个可选的临时目录对象，目录名以 "test-dir-" 开头
  std::optional<c10::TempDir> tempdir = c10::make_tempdir("test-dir-");
  std::string tempdir_name = tempdir->name;

  // 断言在临时目录对象存在期间，目录确实存在
  ASSERT_TRUE(directory_exists(tempdir_name.c_str()));

  // 释放临时目录对象后，断言目录已经不存在
  tempdir.reset();
  ASSERT_FALSE(directory_exists(tempdir_name.c_str()));
}
```