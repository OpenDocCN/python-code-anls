# `basic-computer-games\74_Rock_Scissors_Paper\csharp\Program.cs`

```
// 命名空间 RockScissorsPaper
namespace RockScissorsPaper
{
    // 静态类 Program
    static class Program
    {
        // 主函数
        static void Main(string[] args)
        {
            // 输出游戏标题
            Console.WriteLine("GAME OF ROCK, SCISSORS, PAPER");
            // 输出游戏信息
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            // 输出空行
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();

            // 获取游戏次数
            var numberOfGames = GetNumberOfGames();

            // 创建游戏对象
            var game = new Game();
            // 循环进行游戏
            for (var gameNumber = 1; gameNumber <= numberOfGames; gameNumber++) {
                // 输出游戏编号
                Console.WriteLine();
                Console.WriteLine("Game number {0}", gameNumber);

                // 进行游戏
                game.PlayGame();
            }

            // 输出最终得分
            game.WriteFinalScore();

            // 输出感谢信息
            Console.WriteLine();
            Console.WriteLine("Thanks for playing!!");
        }

        // 获取游戏次数
        static int GetNumberOfGames()
        {
            // 循环直到输入合法的游戏次数
            while (true) {
                // 提示输入游戏次数
                Console.WriteLine("How many games");
                // 尝试将输入转换为整数
                if (int.TryParse(Console.ReadLine(), out var number))
                {
                    // 判断游戏次数是否在合法范围内
                    if (number < 11 && number > 0)
                        return number;
                    // 输出错误信息
                    Console.WriteLine("Sorry, but we aren't allowed to play that many.");
                }
                else
                {
                    // 输出错误信息
                    Console.WriteLine("Sorry, I didn't understand.");
                }
            }
        }
    }
}
```