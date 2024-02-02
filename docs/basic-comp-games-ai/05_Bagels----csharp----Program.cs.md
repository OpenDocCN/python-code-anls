# `basic-computer-games\05_Bagels\csharp\Program.cs`

```py
// 命名空间声明，定义了代码所在的命名空间
namespace BasicComputerGames.Bagels
{
    // 定义一个公共类 Program
    public class Program
    {
        // 定义程序的入口点
        public static void Main(string[] args)
        {
            // 创建一个 Game 类的实例
            var game = new Game();

            // 调用 GameLoop 函数，这将在循环中无限地玩游戏，直到玩家选择退出
            game.GameLoop();
        }
    }
}
```