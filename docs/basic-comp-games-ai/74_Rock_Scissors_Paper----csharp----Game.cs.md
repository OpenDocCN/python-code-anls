# `basic-computer-games\74_Rock_Scissors_Paper\csharp\Game.cs`

```

// 引入 System 和 System.Linq 命名空间
using System;
using System.Linq;

// 创建 RockScissorsPaper 命名空间
namespace RockScissorsPaper
{
    // 创建 Game 类
    public class Game
    {
        // 创建 ComputerWins、HumanWins 和 TieGames 属性，用于记录游戏结果
        public int ComputerWins { get; private set; }
        public int HumanWins { get; private set; }
        public int TieGames { get; private set; }

        // 创建 PlayGame 方法，用于进行游戏
        public void PlayGame()
        {
            // 获取计算机的选择
            var computerChoice = Choices.GetRandom();
            // 获取玩家的选择
            var humanChoice = GetHumanChoice();

            // 输出计算机的选择
            Console.WriteLine("This is my choice...");
            Console.WriteLine("...{0}", computerChoice.Name);

            // 判断游戏结果并更新对应的属性
            if (humanChoice.Beats(computerChoice))
            {
                Console.WriteLine("You win!!!");
                HumanWins++;
            }
            else if (computerChoice.Beats(humanChoice))
            {
                Console.WriteLine("Wow!  I win!!!");
                ComputerWins++;
            }
            else
            {
                Console.WriteLine("Tie game.  No winner.");
                TieGames++;
            }
        }

        // 创建 WriteFinalScore 方法，用于输出最终游戏得分
        public void WriteFinalScore()
        {
            Console.WriteLine();
            Console.WriteLine("Here is the final game score:");
            Console.WriteLine("I have won {0} game(s).", ComputerWins);
            Console.WriteLine("You have one {0} game(s).", HumanWins);
            Console.WriteLine("And {0} game(s) ended in a tie.", TieGames);
        }

        // 创建 GetHumanChoice 方法，用于获取玩家的选择
        public Choice GetHumanChoice()
        {
            while (true)
            {
                Console.WriteLine("3=Rock...2=Scissors...1=Paper");
                Console.WriteLine("1...2...3...What's your choice");
                // 尝试获取玩家输入的选择，并返回对应的 Choice 对象
                if (Choices.TryGetBySelector(Console.ReadLine(), out var choice))
                    return choice;
                Console.WriteLine("Invalid.");
            }
        }
    }
}

```