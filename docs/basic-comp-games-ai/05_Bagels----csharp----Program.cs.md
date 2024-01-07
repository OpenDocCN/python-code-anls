# `basic-computer-games\05_Bagels\csharp\Program.cs`

```

// 命名空间声明，表示该类属于 BasicComputerGames.Bagels 命名空间
namespace BasicComputerGames.Bagels
{
    // 定义一个名为 Program 的类
    public class Program
    {
        // 程序的入口方法
        public static void Main(string[] args)
        {
            // 创建 Game 类的实例
            var game = new Game();

            // 调用 GameLoop 方法，这将在循环中无限地玩游戏，直到玩家选择退出
            game.GameLoop();
        }
    }
}

```