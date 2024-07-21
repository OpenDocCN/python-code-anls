# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\log.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <inttypes.h>

#include <clog.h>

#ifndef PYTORCH_QNNP_LOG_LEVEL
#define PYTORCH_QNNP_LOG_LEVEL CLOG_WARNING
#endif

// 定义 DEBUG 级别的日志对象，用于输出 QNNPACK 的调试信息
CLOG_DEFINE_LOG_DEBUG(
    pytorch_qnnp_log_debug,
    "QNNPACK",
    PYTORCH_QNNP_LOG_LEVEL);
// 定义 INFO 级别的日志对象，用于输出 QNNPACK 的一般信息
CLOG_DEFINE_LOG_INFO(pytorch_qnnp_log_info, "QNNPACK", PYTORCH_QNNP_LOG_LEVEL);
// 定义 WARNING 级别的日志对象，用于输出 QNNPACK 的警告信息
CLOG_DEFINE_LOG_WARNING(
    pytorch_qnnp_log_warning,
    "QNNPACK",
    PYTORCH_QNNP_LOG_LEVEL);
// 定义 ERROR 级别的日志对象，用于输出 QNNPACK 的错误信息
CLOG_DEFINE_LOG_ERROR(
    pytorch_qnnp_log_error,
    "QNNPACK",
    PYTORCH_QNNP_LOG_LEVEL);
// 定义 FATAL 级别的日志对象，用于输出 QNNPACK 的致命错误信息
CLOG_DEFINE_LOG_FATAL(
    pytorch_qnnp_log_fatal,
    "QNNPACK",
    PYTORCH_QNNP_LOG_LEVEL);
```