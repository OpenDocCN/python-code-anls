# `94_War\csharp\War\Program.cs`

```
            // 创建一个 UserInterface 对象
            var ui = new UserInterface();
            // 调用 UserInterface 对象的 WriteIntro 方法，显示游戏介绍
            ui.WriteIntro();

            // 创建一个 Deck 对象
            var deck = new Deck();
            // 调用 Deck 对象的 Shuffle 方法，洗牌
            deck.Shuffle();

            // 初始化玩家和电脑的分数，以及标记是否使用完所有的牌
            int yourScore = 0;
            int computersScore = 0;
            bool usedAllCards = true;

            // 循环进行游戏，每次循环代表一轮游戏
            for (int i = 0; i < Deck.deckSize; i += 2)
            {
                // 获取玩家的牌
                var yourCard = deck.GetCard(i);
                # 从牌堆中获取计算机的牌
                var computersCard = deck.GetCard(i + 1);

                # 调用用户界面的方法，显示你的牌和计算机的牌，并更新计算机和你的分数
                ui.WriteAResult(yourCard, computersCard, ref computersScore, ref yourScore);

                # 如果用户不想继续游戏，则将usedAllCards设置为false，并跳出循环
                if (!ui.AskAQuestion("DO YOU WANT TO CONTINUE? "))
                {
                    usedAllCards = false;
                    break;
                }
            }

            # 调用用户界面的方法，显示游戏结束的相关信息，包括是否用完所有的牌，你的分数和计算机的分数
            ui.WriteClosingRemarks(usedAllCards, yourScore, computersScore);
        }
    }
}
```