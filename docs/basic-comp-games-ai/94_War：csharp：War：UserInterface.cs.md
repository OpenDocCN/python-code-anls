# `d:/src/tocomm/basic-computer-games\94_War\csharp\War\UserInterface.cs`

```
// This class is responsible for displaying text to the user and handling user input.
// WriteIntro method displays the game title and some instructions.
// It also asks the user if they want directions.
// If the user answers yes, it will return true, otherwise it will return false.
            {
                // 打印游戏规则提示信息
                Console.WriteLine("THE COMPUTER GIVES YOU AND IT A 'CARD'.  THE HIGHER CARD");
                Console.WriteLine("(NUMERICALLY) WINS.  THE GAME ENDS WHEN YOU CHOOSE NOT TO");
                Console.WriteLine("CONTINUE OR WHEN YOU HAVE FINISHED THE PACK.");
            }

            // 打印空行
            Console.WriteLine();
            Console.WriteLine();
        }

        // 定义一个方法用于打印比赛结果
        public void WriteAResult(Card yourCard, Card computersCard, ref int computersScore, ref int yourScore)
        {
            // 打印玩家和电脑的卡牌
            Console.WriteLine($"YOU: {yourCard}     COMPUTER: {computersCard}");
            // 如果玩家的卡牌比电脑的小，电脑得分加一，并打印提示信息
            if (yourCard < computersCard)
            {
                computersScore++;
                Console.WriteLine($"THE COMPUTER WINS!!! YOU HAVE {yourScore} AND THE COMPUTER HAS {computersScore}");
            }
            // 如果玩家的卡牌比电脑的大，玩家得分加一，并打印提示信息
            else if (yourCard > computersCard)
            {
                yourScore++;  // 增加玩家得分
                Console.WriteLine($"YOU WIN. YOU HAVE {yourScore} AND THE COMPUTER HAS {computersScore}");  // 打印玩家获胜的消息和当前得分
            }
            else
            {
                Console.WriteLine("TIE.  NO SCORE CHANGE");  // 打印平局的消息
            }
        }

        public bool AskAQuestion(string question)
        {
            // 重复询问问题，直到用户回答"YES"或"NO"
            while (true)
            {
                Console.Write(question);  // 打印问题
                string result = Console.ReadLine();  // 读取用户输入的答案

                if (result.ToLower() == "yes")  // 如果用户回答是"yes"
                {
                    Console.WriteLine();  // 打印空行
                    return true;  # 如果用户输入的结果是 "yes"，则返回 true
                }
                else if (result.ToLower() == "no")  # 如果用户输入的结果是 "no"
                {
                    Console.WriteLine();  # 输出空行
                    return false;  # 返回 false
                }

                Console.WriteLine("YES OR NO, PLEASE.");  # 如果用户输入既不是 "yes" 也不是 "no"，则输出提示信息
            }
        }

        public void WriteClosingRemarks(bool usedAllCards, int yourScore, int computersScore)
        {
            if (usedAllCards)  # 如果所有的牌都已经使用完
            {
                Console.WriteLine("WE HAVE RUN OUT OF CARDS.");  # 输出提示信息
            }
            Console.WriteLine($"FINAL SCORE:  YOU: {yourScore}  THE COMPUTER: {computersScore}");  # 输出最终得分
            Console.WriteLine("THANKS FOR PLAYING.  IT WAS FUN.");  # 输出感谢信息
抱歉，给定的代码片段不完整，无法为其添加注释。
```