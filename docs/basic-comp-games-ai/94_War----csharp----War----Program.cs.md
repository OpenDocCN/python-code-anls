# `basic-computer-games\94_War\csharp\War\Program.cs`

```

// 命名空间 War
namespace War
{
    // 程序入口
    class Program
    {
        static void Main(string[] args)
        {
            // 创建用户界面对象
            var ui = new UserInterface();
            // 输出游戏介绍
            ui.WriteIntro();

            // 创建一副牌并洗牌
            var deck = new Deck();
            deck.Shuffle();

            // 初始化玩家和电脑的分数，以及是否使用完所有的牌
            int yourScore = 0;
            int computersScore = 0;
            bool usedAllCards = true;

            // 循环进行游戏
            for (int i = 0; i < Deck.deckSize; i += 2)
            {
                // 播放下一手牌
                var yourCard = deck.GetCard(i);
                var computersCard = deck.GetCard(i + 1);

                // 输出比赛结果，并更新分数
                ui.WriteAResult(yourCard, computersCard, ref computersScore, ref yourScore);

                // 询问玩家是否继续游戏
                if (!ui.AskAQuestion("DO YOU WANT TO CONTINUE? "))
                {
                    usedAllCards = false;
                    break;
                }
            }

            // 输出游戏结束的话语
            ui.WriteClosingRemarks(usedAllCards, yourScore, computersScore);
        }
    }
}

```