# `basic-computer-games\90_Tower\csharp\Models\Towers.cs`

```py
# 引入系统和自定义的命名空间
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Tower.Resources;

# 定义名为Towers的内部类，实现IEnumerable接口，元组包含三个整数
namespace Tower.Models
{
    internal class Towers : IEnumerable<(int, int, int)>
    }
}
```