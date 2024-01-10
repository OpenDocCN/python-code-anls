# `basic-computer-games\74_Rock_Scissors_Paper\csharp\Game.cs`

```
using System;
using System.Linq;

namespace RockScissorsPaper
{
    public class Game
    {
        public int ComputerWins { get; private set; }  // 电脑赢的次数
        public int HumanWins { get; private set; }  // 玩家赢的次数
        public int TieGames { get; private set; }  // 平局次数

        public void PlayGame()
        {
            var computerChoice = Choices.GetRandom();  // 电脑随机选择
            var humanChoice = GetHumanChoice();  // 获取玩家选择

            Console.WriteLine("This is my choice...");  // 输出电脑选择
            Console.WriteLine("...{0}", computerChoice.Name);  // 输出电脑选择的名称

            if (humanChoice.Beats(computerChoice))  // 如果玩家选择胜利
            {
                Console.WriteLine("You win!!!");  // 输出玩家获胜信息
                HumanWins++;  // 玩家获胜次数加一
            }
            else if (computerChoice.Beats(humanChoice))  // 如果电脑选择胜利
            {
                Console.WriteLine("Wow!  I win!!!");  // 输出电脑获胜信息
                ComputerWins++;  // 电脑获胜次数加一
            }
            else  // 如果平局
            {
                Console.WriteLine("Tie game.  No winner.");  // 输出平局信息
                TieGames++;  // 平局次数加一
            }
        }

        public void WriteFinalScore()
        {
            Console.WriteLine();  // 输出空行
            Console.WriteLine("Here is the final game score:");  // 输出最终比分信息
            Console.WriteLine("I have won {0} game(s).", ComputerWins);  // 输出电脑获胜次数
            Console.WriteLine("You have one {0} game(s).", HumanWins);  // 输出玩家获胜次数
            Console.WriteLine("And {0} game(s) ended in a tie.", TieGames);  // 输出平局次数
        }

        public Choice GetHumanChoice()
        {
            while (true)
            {
                Console.WriteLine("3=Rock...2=Scissors...1=Paper");  // 输出玩家选择提示
                Console.WriteLine("1...2...3...What's your choice");  // 输出玩家选择提示
                if (Choices.TryGetBySelector(Console.ReadLine(), out var choice))  // 尝试获取玩家选择
                    return choice;  // 返回玩家选择
                Console.WriteLine("Invalid.");  // 输出无效选择提示
            }
        }
    }
}
```