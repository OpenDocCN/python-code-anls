# `basic-computer-games\18_Bullseye\csharp\Program.cs`

```py
// 引入 System 命名空间
using System;

// 定义 Bullseye 命名空间
namespace Bullseye
{
    // 定义静态类 Program
    public static class Program
    {
        // 应用程序的入口点；创建游戏类的实例并调用 Run() 方法
        public static void Main(string[] args)
        {
            // 创建 BullseyeGame 类的实例并调用 Run() 方法
            new BullseyeGame().Run();
        }
    }
}
```