# `basic-computer-games\33_Dice\csharp\Program.cs`

```py
// 命名空间声明，表示该类属于BasicComputerGames.Dice命名空间
namespace BasicComputerGames.Dice
{
    // 定义一个公共类Program
    public class Program
    {
        // 定义一个静态的Main方法，程序的入口
        public static void Main(string[] args)
        {
            // 创建一个Game类的实例
            Game game = new Game();

            // 调用GameLoop方法，这将在循环中无限地播放游戏，直到玩家选择退出
            game.GameLoop();
        }
    }
}
```