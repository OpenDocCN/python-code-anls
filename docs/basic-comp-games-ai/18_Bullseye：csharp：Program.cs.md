# `d:/src/tocomm/basic-computer-games\18_Bullseye\csharp\Program.cs`

```
# 创建一个名为Bullseye的命名空间
namespace Bullseye
{
    # 创建一个名为Program的静态类
    public static class Program
    {
        # 应用程序的入口点；创建游戏类的实例并调用Run()方法
        public static void Main(string[] args)
        {
            new BullseyeGame().Run();
        }
    }
}
```