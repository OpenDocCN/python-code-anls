# `74_Rock_Scissors_Paper\csharp\Game.cs`

```
            # 获取计算机的随机选择
            computerChoice = Choices.GetRandom()
            # 获取玩家的选择
            humanChoice = GetHumanChoice()

            # 打印计算机的选择
            print("This is my choice...")
            print("...{0}".format(computerChoice.Name)

            # 如果玩家的选择击败了计算机的选择
            if humanChoice.Beats(computerChoice):
            {
                Console.WriteLine("You win!!!");  # 如果玩家赢了，打印“你赢了！”
                HumanWins++;  # 玩家赢的次数加一
            }
            else if (computerChoice.Beats(humanChoice))  # 如果电脑赢了
            {
                Console.WriteLine("Wow!  I win!!!");  # 打印“哇！我赢了！”
                ComputerWins++;  # 电脑赢的次数加一
            }
            else  # 如果是平局
            {
                Console.WriteLine("Tie game.  No winner.");  # 打印“平局，没有赢家。”
                TieGames++;  # 平局次数加一
            }
        }

        public void WriteFinalScore()  # 定义一个方法用来打印最终得分
        {
            Console.WriteLine();  # 打印空行
            Console.WriteLine("Here is the final game score:");  # 打印“这是最终得分：”
            Console.WriteLine("I have won {0} game(s).", ComputerWins);  // 打印计算机赢得游戏的次数
            Console.WriteLine("You have one {0} game(s).", HumanWins);  // 打印玩家赢得游戏的次数
            Console.WriteLine("And {0} game(s) ended in a tie.", TieGames);  // 打印游戏以平局结束的次数
        }

        public Choice GetHumanChoice()
        {
            while (true)
            {
                Console.WriteLine("3=Rock...2=Scissors...1=Paper");  // 打印游戏选项
                Console.WriteLine("1...2...3...What's your choice");  // 提示玩家进行选择
                if (Choices.TryGetBySelector(Console.ReadLine(), out var choice))  // 从玩家输入中获取选择
                    return choice;  // 返回玩家的选择
                Console.WriteLine("Invalid.");  // 如果选择无效，则打印错误信息
            }
        }
    }
}
```