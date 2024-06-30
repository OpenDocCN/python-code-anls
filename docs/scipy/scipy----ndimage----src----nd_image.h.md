# `D:\src\scipysrc\scipy\scipy\ndimage\src\nd_image.h`

```
# 版权声明，此段代码版权归 Peter J. Verveer 所有，遵循以下条件可以自由使用和分发：
# 1. 在源代码中保留版权声明、条件列表和以下免责声明。
# 2. 在二进制形式中，需要在文档或其他提供的材料中重现上述版权声明、条件列表和以下免责声明。
# 3. 未经特定书面许可，不得使用作者的名字来认可或推广基于此软件的产品。
#
# 此软件按“原样”提供，不提供任何明示或暗示的保证，包括但不限于适销性和特定用途的保证。作者不对任何直接、间接、偶然、特殊、示范性或后果性损害承担责任，即使事先被告知可能性。
#

# 如果 ND_IMAGE_H 未定义，则定义 ND_IMAGE_H
#ifndef ND_IMAGE_H
#define ND_IMAGE_H

# 包含 Python.h 头文件
#include "Python.h"

# 定义 PY_ARRAY_UNIQUE_SYMBOL 为 _scipy_ndimage_ARRAY_API
#define PY_ARRAY_UNIQUE_SYMBOL _scipy_ndimage_ARRAY_API

# 包含 numpy/arrayobject.h 头文件
#include <numpy/arrayobject.h>

# 结束 ND_IMAGE_H 的定义
#endif /* ND_IMAGE_H */
```