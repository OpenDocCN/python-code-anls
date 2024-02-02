# `basic-computer-games\06_Banner\csharp\banner.cs`

```py
# 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;

# 定义名为Banner的类
namespace banner
{
    class Banner
    }

    # 定义名为Program的类
    class Program
    {
        # 定义名为Main的静态方法，参数为字符串数组
        static void Main(string[] args)
        {
            # 创建Banner类的实例并调用Play方法
            new Banner().Play();
        }
    }
}
```