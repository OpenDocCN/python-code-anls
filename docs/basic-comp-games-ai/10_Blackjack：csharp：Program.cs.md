# `d:/src/tocomm/basic-computer-games\10_Blackjack\csharp\Program.cs`

```
# 输出游戏标题
Console.WriteLine("{0}BLACK JACK", new string(' ', 31));
# 输出游戏信息
Console.WriteLine("{0}CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY", new string(' ', 15));
# 输出空行
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();

# 提供游戏说明
OfferInstructions();

# 提示用户输入玩家数量，限定在1到6之间
var numberOfPlayers = Prompt.ForInteger("Number of players?", 1, 6);
# 创建游戏对象
var game = new Game(numberOfPlayers);
# 开始游戏
game.PlayGame();
        private static void OfferInstructions()
        {
            // 如果用户不想要说明，则直接返回
            if (!Prompt.ForYesNo("Do you want instructions?"))
                return;

            // 输出游戏规则说明
            Console.WriteLine("This is the game of 21. As many as 7 players may play the");
            Console.WriteLine("game. On each deal, bets will be asked for, and the");
            Console.WriteLine("players' bets should be typed in. The cards will then be");
            Console.WriteLine("dealt, and each player in turn plays his hand. The");
            Console.WriteLine("first response should be either 'D', indicating that the");
            Console.WriteLine("player is doubling down, 'S', indicating that he is");
            Console.WriteLine("standing, 'H', indicating he wants another card, or '/',");
            Console.WriteLine("indicating that he wants to split his cards. After the");
            Console.WriteLine("initial response, all further responses should be 's' or");
            Console.WriteLine("'H', unless the cards were split, in which case doubling");
            Console.WriteLine("down is again permitted. In order to collect for");
            Console.WriteLine("Blackjack, the initial response should be 'S'.");
        }
    }
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```