# `01_Acey_Ducey\csharp\Program.cs`

```
# 创建一个名为AceyDucey的命名空间
namespace AceyDucey
{
    /// <summary>
    /// 应用程序的入口点
    /// </summary>
    class Program
    {

        /// <summary>
        /// 当应用程序启动时，此函数将自动被调用
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // 创建我们主要的Game类的实例
            Game game = new Game();
            // 调用它的GameLoop函数。这将在循环中无限地播放游戏，直到玩家选择退出。
            game.GameLoop();
        }
    }
}
```