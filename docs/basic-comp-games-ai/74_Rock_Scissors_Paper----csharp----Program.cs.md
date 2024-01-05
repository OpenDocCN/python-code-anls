# `74_Rock_Scissors_Paper\csharp\Program.cs`

```
# 导入 System 命名空间
using System;

# 创建静态类 Program
namespace RockScissorsPaper
{
    static class Program
    {
        # 定义程序入口点
        static void Main(string[] args)
        {
            # 输出游戏标题
            Console.WriteLine("GAME OF ROCK, SCISSORS, PAPER");
            # 输出游戏信息
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            # 输出空行
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();

            # 获取游戏次数
            var numberOfGames = GetNumberOfGames();

            # 创建游戏对象
            var game = new Game();
            # 循环进行游戏
            for (var gameNumber = 1; gameNumber <= numberOfGames; gameNumber++) {
                # 输出游戏编号
                Console.WriteLine();
                Console.WriteLine("Game number {0}", gameNumber);
                game.PlayGame();  # 调用游戏对象的PlayGame方法，开始游戏
            }

            game.WriteFinalScore();  # 调用游戏对象的WriteFinalScore方法，记录最终得分

            Console.WriteLine();  # 输出空行
            Console.WriteLine("Thanks for playing!!");  # 输出感谢信息
        }

        static int GetNumberOfGames()  # 定义一个静态方法GetNumberOfGames，返回整数类型
        {
            while (true) {  # 进入无限循环
                Console.WriteLine("How many games");  # 输出提示信息，询问游戏次数
                if (int.TryParse(Console.ReadLine(), out var number))  # 从控制台读取输入，尝试将输入转换为整数类型，将结果存储在number变量中
                {
                    if (number < 11 && number > 0)  # 如果输入的数字小于11且大于0
                        return number;  # 返回输入的数字
                    Console.WriteLine("Sorry, but we aren't allowed to play that many.");  # 输出提示信息，表示不能玩那么多次游戏
                }
                else
                {
                    Console.WriteLine("Sorry, I didn't understand.");  // 如果用户输入的命令不匹配任何已知命令，则打印错误消息
                }
            }
        }
    }
}
```