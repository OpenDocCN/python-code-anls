# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\operator-delete.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/operator.h>

// 删除 QNNPACK 操作符的函数
enum pytorch_qnnp_status pytorch_qnnp_delete_operator(
    pytorch_qnnp_operator_t op) {
  
  // 检查操作符是否为 NULL
  if (op == NULL) {
    return pytorch_qnnp_status_invalid_parameter;
  }

  // 释放间接缓冲区的内存
  free(op->indirection_buffer);
  // 释放打包权重的内存
  free(op->packed_weights);
  // 释放 a_sum 的内存
  free(op->a_sum);
  // 释放零缓冲区的内存
  free(op->zero_buffer);
  // 释放查找表的内存
  free(op->lookup_table);
  // 释放操作符对象本身的内存
  free(op);
  
  // 返回成功状态
  return pytorch_qnnp_status_success;
}
```