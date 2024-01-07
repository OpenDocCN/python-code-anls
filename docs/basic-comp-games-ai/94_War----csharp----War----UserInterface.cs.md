# `basic-computer-games\94_War\csharp\War\UserInterface.cs`

```

// 命名空间 War 包含了游戏中用户界面的相关类
namespace War
{
    // UserInterface 类负责显示游戏中用户看到的所有文本，并处理用户的是/否问题并返回他们的答案
    public class UserInterface
    {
        // 显示游戏介绍
        public void WriteIntro()
        {
            // 输出游戏标题和创意计算的信息
            Console.WriteLine("                                 WAR");
            Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();

            Console.WriteLine("THIS IS THE CARD GAME OF WAR.  EACH CARD IS GIVEN BY SUIT-#");
            Console.Write("AS S-7 FOR SPADE 7.  ");

            // 如果用户想要游戏说明，则显示游戏规则
            if (AskAQuestion("DO YOU WANT DIRECTIONS? "))
            {
                Console.WriteLine("THE COMPUTER GIVES YOU AND IT A 'CARD'.  THE HIGHER CARD");
                Console.WriteLine("(NUMERICALLY) WINS.  THE GAME ENDS WHEN YOU CHOOSE NOT TO");
                Console.WriteLine("CONTINUE OR WHEN YOU HAVE FINISHED THE PACK.");
            }

            Console.WriteLine();
            Console.WriteLine();
        }

        // 显示比赛结果
        public void WriteAResult(Card yourCard, Card computersCard, ref int computersScore, ref int yourScore)
        {
            Console.WriteLine($"YOU: {yourCard}     COMPUTER: {computersCard}");
            // 根据比赛结果更新分数
            if (yourCard < computersCard)
            {
                computersScore++;
                Console.WriteLine($"THE COMPUTER WINS!!! YOU HAVE {yourScore} AND THE COMPUTER HAS {computersScore}");
            }
            else if (yourCard > computersCard)
            {
                yourScore++;
                Console.WriteLine($"YOU WIN. YOU HAVE {yourScore} AND THE COMPUTER HAS {computersScore}");
            }
            else
            {
                Console.WriteLine("TIE.  NO SCORE CHANGE");
            }
        }

        // 提问用户一个问题，并返回他们的答案
        public bool AskAQuestion(string question)
        {
            // 重复提问，直到用户回答"YES"或"NO"
            while (true)
            {
                Console.Write(question);
                string result = Console.ReadLine();

                if (result.ToLower() == "yes")
                {
                    Console.WriteLine();
                    return true;
                }
                else if (result.ToLower() == "no")
                {
                    Console.WriteLine();
                    return false;
                }

                Console.WriteLine("YES OR NO, PLEASE.");
            }
        }

        // 显示游戏结束的相关信息
        public void WriteClosingRemarks(bool usedAllCards, int yourScore, int computersScore)
        {
            if (usedAllCards)
            {
                Console.WriteLine("WE HAVE RUN OUT OF CARDS.");
            }
            Console.WriteLine($"FINAL SCORE:  YOU: {yourScore}  THE COMPUTER: {computersScore}");
            Console.WriteLine("THANKS FOR PLAYING.  IT WAS FUN.");
        }
    }
}

```