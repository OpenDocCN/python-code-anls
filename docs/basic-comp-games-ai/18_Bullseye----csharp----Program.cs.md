# `basic-computer-games\18_Bullseye\csharp\Program.cs`

```

// 命名空间Bullseye
namespace Bullseye
{
    // 创建一个静态类Program
    public static class Program
    {
        // 应用程序的入口点；创建游戏类的实例并调用Run()方法
        public static void Main(string[] args)
        {
            // 创建BullseyeGame类的实例并调用Run()方法
            new BullseyeGame().Run();
        }
    }
}

```