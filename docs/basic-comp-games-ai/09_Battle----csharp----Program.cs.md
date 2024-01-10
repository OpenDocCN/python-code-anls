# `basic-computer-games\09_Battle\csharp\Program.cs`

```
# 引入 System 命名空间
using System;

# 定义 Battle 命名空间
namespace Battle
{
    # 定义 Program 类
    class Program
    {
        # 定义程序入口 Main 方法，参数为字符串数组
        static void Main(string[] args)
        {
            # 创建 Game 对象并调用其 Run 方法
            new Game().Run();
        }
    }
}
```