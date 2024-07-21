# `.\pytorch\test\cpp\jit\test_jit_logging_levels.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架头文件
#include <test/cpp/jit/test_utils.h>  // 引入测试工具头文件

#include <torch/csrc/jit/jit_log.h>  // 引入 JIT 日志相关头文件
#include <sstream>  // 引入字符串流头文件

namespace torch {
namespace jit {

TEST(JitLoggingTest, CheckSetLoggingLevel) {
  ::torch::jit::set_jit_logging_levels("file_to_test");  // 设置 JIT 日志级别为 "file_to_test"
  ASSERT_TRUE(::torch::jit::is_enabled(
      "file_to_test.cpp", JitLoggingLevels::GRAPH_DUMP));  // 断言是否启用了指定级别的 JIT 日志
}

TEST(JitLoggingTest, CheckSetMultipleLogLevels) {
  ::torch::jit::set_jit_logging_levels("f1:>f2:>>f3");  // 设置多个 JIT 日志级别
  ASSERT_TRUE(::torch::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_DUMP));  // 断言是否启用了 GRAPH_DUMP 级别的 JIT 日志
  ASSERT_TRUE(
      ::torch::jit::is_enabled("f2.cpp", JitLoggingLevels::GRAPH_UPDATE));  // 断言是否启用了 GRAPH_UPDATE 级别的 JIT 日志
  ASSERT_TRUE(
      ::torch::jit::is_enabled("f3.cpp", JitLoggingLevels::GRAPH_DEBUG));  // 断言是否启用了 GRAPH_DEBUG 级别的 JIT 日志
}

TEST(JitLoggingTest, CheckLoggingLevelAfterUnset) {
  ::torch::jit::set_jit_logging_levels("f1");  // 设置 JIT 日志级别为 "f1"
  ASSERT_EQ("f1", ::torch::jit::get_jit_logging_levels());  // 断言获取的 JIT 日志级别是否为 "f1"
  ::torch::jit::set_jit_logging_levels("invalid");  // 设置无效的 JIT 日志级别
  ASSERT_FALSE(
      ::torch::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_DUMP));  // 断言是否未启用 GRAPH_DUMP 级别的 JIT 日志
}

TEST(JitLoggingTest, CheckAfterChangingLevel) {
  ::torch::jit::set_jit_logging_levels("f1");  // 设置 JIT 日志级别为 "f1"
  ::torch::jit::set_jit_logging_levels(">f1");  // 设置更高级别的 JIT 日志
  ASSERT_TRUE(
      ::torch::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_UPDATE));  // 断言是否启用了 GRAPH_UPDATE 级别的 JIT 日志
}

TEST(JitLoggingTest, CheckOutputStreamSetting) {
  ::torch::jit::set_jit_logging_levels("test_jit_logging_levels");  // 设置 JIT 日志级别为 "test_jit_logging_levels"
  std::ostringstream test_stream;  // 创建字符串流对象 test_stream
  ::torch::jit::set_jit_logging_output_stream(test_stream);  // 设置 JIT 日志输出流为 test_stream
  /* Using JIT_LOG checks if this file has logging enabled with
    is_enabled(__FILE__, level) making the test fail. since we are only testing
    the OutputStreamSetting we can forcefully output to it directly.
  */
  ::torch::jit::get_jit_logging_output_stream() << ::torch::jit::jit_log_prefix(
      ::torch::jit::JitLoggingLevels::GRAPH_DUMP,
      __FILE__,
      __LINE__,
      ::c10::str("Message"));  // 输出 JIT 日志前缀和消息到 JIT 日志输出流
  ASSERT_TRUE(test_stream.str().size() > 0);  // 断言输出流中是否有内容
}

} // namespace jit
} // namespace torch
```