# `10_Blackjack\csharp\Game.cs`

```
using System;  // 导入 System 命名空间
using System.Collections.Generic;  // 导入 System.Collections.Generic 命名空间
using System.Linq;  // 导入 System.Linq 命名空间

namespace Blackjack  // 命名空间声明
{
    public class Game  // 定义名为 Game 的公共类
    {
        private readonly Deck _deck = new Deck();  // 声明名为 _deck 的只读字段，并初始化为 Deck 类的实例
        private readonly int _numberOfPlayers;  // 声明名为 _numberOfPlayers 的只读字段
        private readonly Player[] _players;  // 声明名为 _players 的只读字段，类型为 Player 类的数组
        private readonly Hand _dealerHand;  // 声明名为 _dealerHand 的只读字段，类型为 Hand 类

        public Game(int numberOfPlayers)  // 定义名为 Game 的公共构造函数，接受一个 int 类型的参数
        {
            _numberOfPlayers = numberOfPlayers;  // 将传入的参数赋值给 _numberOfPlayers 字段
            _players = new Player[_numberOfPlayers];  // 初始化 _players 数组，长度为 _numberOfPlayers
            for (var playerIndex = 0; playerIndex < _numberOfPlayers; playerIndex++)  // 循环 _numberOfPlayers 次
                _players[playerIndex] = new Player(playerIndex);  // 初始化 _players 数组中的每个元素为 Player 类的实例
            _dealerHand = new Hand();  // 初始化 _dealerHand 字段为 Hand 类的实例
        }

        # 定义一个名为 PlayGame 的方法
        public void PlayGame()
        {
            # 创建一个无限循环，直到游戏结束
            while (true)
            {
                # 调用 PlayRound 方法来进行一轮游戏
                PlayRound();
                # 调用 TallyResults 方法来计算结果
                TallyResults();
                # 调用 ResetRoundState 方法来重置游戏状态
                ResetRoundState();
                # 在控制台输出空行
                Console.WriteLine();
            }
        }

        # 定义一个名为 PlayRound 的方法
        public void PlayRound()
        {
            # 调用 GetPlayerBets 方法来获取玩家的赌注
            GetPlayerBets();

            # 调用 DealHands 方法来发牌
            DealHands();

            # 测试是否需要购买保险
            // 检查庄家是否展示了一张 A，如果是则询问是否购买保险
            var dealerIsShowingAce = _dealerHand.Cards[0].IsAce;
            if (dealerIsShowingAce && Prompt.ForYesNo("Any insurance?"))
            {
                // 如果庄家展示了 A 并且玩家选择购买保险，则进行以下操作
                Console.WriteLine("Insurance bets");
                // 创建一个整数数组来存储每个玩家的保险赌注
                var insuranceBets = new int[_numberOfPlayers];
                // 遍历每个玩家，询问他们的保险赌注
                foreach (var player in _players)
                    insuranceBets[player.Index] = Prompt.ForInteger($"# {player.Index + 1} ?", 0, player.RoundBet / 2);

                // 根据庄家是否为黑杰克来确定保险的效果倍数
                var insuranceEffectMultiplier = _dealerHand.IsBlackjack ? 2 : -1;
                // 遍历每个玩家，根据保险赌注和效果倍数来更新他们的回合赢取金额
                foreach (var player in _players)
                    player.RoundWinnings += insuranceBets[player.Index] * insuranceEffectMultiplier;
            }

            // 检查庄家是否为黑杰克
            var concealedCard = _dealerHand.Cards[0];
            if (_dealerHand.IsBlackjack)
            {
                // 如果庄家为黑杰克，则展示庄家暗牌的信息并结束游戏
                Console.WriteLine();
                Console.WriteLine("Dealer has {0} {1} in the hole for blackjack.", concealedCard.IndefiniteArticle, concealedCard.Name);
                return;
            }
            else if (dealerIsShowingAce)
            {
                // 如果庄家亮出了一张A，则输出信息
                Console.WriteLine();
                Console.WriteLine("No dealer blackjack.");
            }

            // 对每个玩家进行游戏
            foreach (var player in _players)
                PlayHand(player);

            // 庄家手牌
            var allPlayersBusted = _players.All(p => p.Hand.IsBusted && (!p.SecondHand.Exists || p.SecondHand.IsBusted));
            // 如果所有玩家都爆牌了
            if (allPlayersBusted)
                Console.WriteLine("Dealer had {0} {1} concealed.", concealedCard.IndefiniteArticle, concealedCard.Name);
            else
            {
                // 如果有玩家没有爆牌
                Console.WriteLine("Dealer has {0} {1} concealed for a total of {2}", concealedCard.IndefiniteArticle, concealedCard.Name, _dealerHand.Total);
                // 如果庄家手牌小于17
                if (_dealerHand.Total < 17)
                {
                    // 输出信息
                    Console.Write("Draws");
                    while (_dealerHand.Total < 17)  # 当庄家手中牌的总点数小于17时
                    {
                        var card = _dealerHand.AddCard(_deck.DrawCard());  # 从牌堆中抽一张牌，加入到庄家手中，并将该牌的名称打印出来
                        Console.Write("  {0}", card.Name);  # 打印出抽到的牌的名称
                    }
                    if (_dealerHand.IsBusted)  # 如果庄家手中的牌点数超过21
                        Console.WriteLine("  ...Busted");  # 打印出庄家爆牌的信息
                    else
                        Console.WriteLine("  ---Total is {0}", _dealerHand.Total);  # 打印出庄家手中牌的总点数
                }
            }
        }

        private void GetPlayerBets()  # 获取玩家的赌注
        {
            Console.WriteLine("Bets:");  # 打印出提示信息
            foreach (var player in _players)  # 遍历玩家列表
                player.RoundBet = Prompt.ForInteger($"# {player.Name} ?", 1, 500);  # 提示玩家输入赌注，并将输入的赌注赋值给玩家的RoundBet属性
        }
        private void DealHands()
        {
            // 输出玩家和庄家的名称
            Console.Write("Player ");
            foreach (var player in _players)
                Console.Write("{0}     ", player.Name);
            Console.WriteLine("Dealer");

            // 发两张牌给每个玩家和庄家
            for (var cardIndex = 0; cardIndex < 2; cardIndex++)
            {
                Console.Write("      ");
                // 给每个玩家发一张牌，并显示牌的名称
                foreach (var player in _players)
                    Console.Write("  {0,-4}", player.Hand.AddCard(_deck.DrawCard()).Name);
                // 给庄家发一张牌，并根据是否是第一张牌显示牌的名称或者隐藏
                var dealerCard = _dealerHand.AddCard(_deck.DrawCard());
                Console.Write("  {0,-4}", (cardIndex == 0) ? "XX" : dealerCard.Name);

                Console.WriteLine();
            }
        }

        private void PlayHand(Player player)
            # 获取玩家的手牌
            var hand = player.Hand;

            # 输出玩家的姓名
            Console.Write("Player {0} ", player.Name);

            # 检查玩家是否可以分牌
            var playerCanSplit = hand.Cards[0].Value == hand.Cards[1].Value;

            # 提示玩家输入命令
            var command = Prompt.ForCommandCharacter("?", playerCanSplit ? "HSD/" : "HSD");

            # 根据玩家输入的命令执行相应的操作
            switch (command)
            {
                # 如果玩家选择双倍下注
                case "D":
                    player.RoundBet *= 2;
                    # 跳转到执行"Hit"操作
                    goto case "H";

                # 如果玩家选择"Hit"
                case "H":
                    # 当玩家选择"Hit"并且需要继续要牌时执行循环
                    while (TakeHit(hand) && PromptForAnotherHit())
                    { }
                    # 如果玩家没有爆牌，则输出手牌总点数
                    if (!hand.IsBusted)
                        Console.WriteLine("Total is {0}", hand.Total);
                    break;
                case "S":  # 如果玩家选择停牌
                    if (hand.IsBlackjack):  # 如果玩家手牌是二十一点
                        Console.WriteLine("Blackjack!")  # 输出“Blackjack!”
                        player.RoundWinnings = (int)(1.5 * player.RoundBet + 0.5)  # 玩家赢得1.5倍下注金额
                        player.RoundBet = 0  # 玩家下注金额清零
                    else:  # 如果玩家手牌不是二十一点
                        Console.WriteLine("Total is {0}", hand.Total)  # 输出玩家手牌的总点数
                    break  # 结束当前情况的判断

                case "/":  # 如果玩家选择分牌
                    hand.SplitHand(player.SecondHand)  # 将手牌分成两手
                    var card = hand.AddCard(_deck.DrawCard())  # 给第一手牌加一张牌
                    Console.WriteLine("First hand receives {0} {1}", card.IndefiniteArticle, card.Name)  # 输出第一手牌获得的牌
                    card = player.SecondHand.AddCard(_deck.DrawCard())  # 给第二手牌加一张牌
                    Console.WriteLine("Second hand receives {0} {1}", card.IndefiniteArticle, card.Name)  # 输出第二手牌获得的牌

                    for (int handNumber = 1; handNumber <= 2; handNumber++):  # 循环两次，分别处理两手牌
                        hand = (handNumber == 1) ? player.Hand : player.SecondHand;  # 根据手牌编号选择当前操作的手牌

                        Console.Write("Hand {0}", handNumber);  # 在控制台输出当前手牌的编号
                        while (PromptForAnotherHit() && TakeHit(hand))  # 当提示需要再要一张牌并且继续要牌时执行循环
                        { }
                        if (!hand.IsBusted)  # 如果手牌没有爆牌
                            Console.WriteLine("Total is {0}", hand.Total);  # 在控制台输出当前手牌的总点数
                    }
                    break;  # 结束当前的 switch 语句
            }
        }

        private bool TakeHit(Hand hand)  # 定义一个方法，表示玩家要牌
        {
            var card = hand.AddCard(_deck.DrawCard());  # 从牌堆中抽一张牌加入到手牌中
            Console.Write("Received {0,-6}", $"{card.IndefiniteArticle} {card.Name}");  # 在控制台输出抽到的牌的信息
            if (hand.IsBusted)  # 如果手牌爆牌
            {
                Console.WriteLine("...Busted");  # 在控制台输出爆牌的信息
                return false;  # 返回 false，表示不再继续要牌
        }

        // 检查是否需要再要牌
        private bool PromptForAnotherHit()
        {
            // 调用 Prompt.ForCommandCharacter 方法，提示用户输入是否要牌，如果输入为 "H" 则返回 true，否则返回 false
            return String.Equals(Prompt.ForCommandCharacter(" Hit?", "HS"), "H");
        }

        // 计算每位玩家的赢钱情况
        private void TallyResults()
        {
            // 在控制台输出空行
            Console.WriteLine();
            // 遍历玩家列表
            foreach (var player in _players)
            {
                // 将玩家本轮的赢钱情况加上当前手牌的赢钱情况
                player.RoundWinnings += CalculateWinnings(player, player.Hand);
                // 如果玩家有第二手牌，则将其赢钱情况也加上
                if (player.SecondHand.Exists)
                    player.RoundWinnings += CalculateWinnings(player, player.SecondHand);
                // 将本轮赢钱情况加到总赢钱情况中
                player.TotalWinnings += player.RoundWinnings;

                // 在控制台输出玩家的编号、本轮赢钱情况、总赢钱情况
                Console.WriteLine("Player {0} {1,-6} {2,3}   Total= {3,5}",
                        player.Name, // 输出玩家的名字
                        (player.RoundWinnings > 0) ? "wins" : (player.RoundWinnings) < 0 ? "loses" : "pushes", // 根据玩家的轮次赢利情况输出相应的结果
                        (player.RoundWinnings != 0) ? Math.Abs(player.RoundWinnings).ToString() : "", // 如果玩家的轮次赢利不为0，则输出其绝对值
                        player.TotalWinnings); // 输出玩家的总赢利
            }
            Console.WriteLine("Dealer's total= {0}", -_players.Sum(p => p.TotalWinnings)); // 输出庄家的总赢利

        }

        private int CalculateWinnings(Player player, Hand hand) // 计算玩家的赢利
        {
            if (hand.IsBusted) // 如果玩家的手牌爆牌
                return -player.RoundBet; // 返回玩家的轮次赢利为负的下注金额
            if (hand.Total == _dealerHand.Total) // 如果玩家的手牌点数等于庄家的手牌点数
                return 0; // 返回玩家的轮次赢利为0
            if (_dealerHand.IsBusted || hand.Total > _dealerHand.Total) // 如果庄家的手牌爆牌或者玩家的手牌点数大于庄家的手牌点数
                return player.RoundBet; // 返回玩家的轮次赢利为下注金额
            return -player.RoundBet; // 其他情况返回玩家的轮次赢利为负的下注金额
        }

        private void ResetRoundState() // 重置轮次状态
        {
            # 遍历玩家列表中的每个玩家
            foreach (var player in _players)
            {
                # 将玩家当前轮次的赢钱数和下注数重置为0
                player.RoundWinnings = 0;
                player.RoundBet = 0;
                # 通过调用玩家对象的Discard方法，将玩家手中的牌放回到牌堆中
                player.Hand.Discard(_deck);
                # 通过调用玩家对象的Discard方法，将玩家的第二手牌放回到牌堆中
                player.SecondHand.Discard(_deck);
            }
            # 通过调用庄家手中的Discard方法，将庄家手中的牌放回到牌堆中
            _dealerHand.Discard(_deck);
        }
    }
}
```