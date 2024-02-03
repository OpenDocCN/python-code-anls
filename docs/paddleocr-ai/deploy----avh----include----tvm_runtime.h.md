# `.\PaddleOCR\deploy\avh\include\tvm_runtime.h`

```
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
*/

// 包含标准库头文件
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
// 包含 TVM 运行时 API 头文件
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/stack_allocator.h>

#ifdef __cplusplus
extern "C" {
#endif

// TVM 平台异常终止函数
void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t error_code) {
  // 打印异常信息和错误码
  printf("TVMPlatformAbort: %d\n", error_code);
  printf("EXITTHESIM\n");
  // 退出程序
  exit(-1);
}

// TVM 平台内存分配函数
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  // 返回未实现的错误码
  return kTvmErrorFunctionCallNotImplemented;
}

// TVM 平台内存释放函数
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  // 返回未实现的错误码
  return kTvmErrorFunctionCallNotImplemented;
}

// TVM 日志输出函数
void TVMLogf(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  // 输出格式化字符串到标准输出
  vfprintf(stdout, msg, args);
  va_end(args);
}

// TVM 全局函数注册函数
TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) { return 0; }

#ifdef __cplusplus
}
#endif
```