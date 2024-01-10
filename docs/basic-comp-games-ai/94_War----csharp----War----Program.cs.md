# `basic-computer-games\94_War\csharp\War\Program.cs`

```
// 命名空间声明，定义了代码所在的命名空间
namespace War
{
    // 定义了程序的入口类
    class Program
    {
        // 主函数，程序的入口
        static void Main(string[] args)
        {
            // 创建用户界面对象
            var ui = new UserInterface();
            // 输出游戏介绍
            ui.WriteIntro();

            // 创建一副牌并洗牌
            var deck = new Deck();
            deck.Shuffle();

            // 初始化玩家和电脑的分数，以及标记是否用完了所有的牌
            int yourScore = 0;
            int computersScore = 0;
            bool usedAllCards = true;

            // 循环进行游戏
            for (int i = 0; i < Deck.deckSize; i += 2)
            {
                // 播放下一手牌
                var yourCard = deck.GetCard(i);
                var computersCard = deck.GetCard(i + 1);

                // 输出比赛结果，并更新玩家和电脑的分数
                ui.WriteAResult(yourCard, computersCard, ref computersScore, ref yourScore);

                // 询问玩家是否继续游戏
                if (!ui.AskAQuestion("DO YOU WANT TO CONTINUE? "))
                {
                    // 如果玩家选择不继续，标记为未用完所有牌，并跳出循环
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