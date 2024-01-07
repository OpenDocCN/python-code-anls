# `basic-computer-games\71_Poker\csharp\Game.cs`

```

// 导入所需的类库
using Poker.Cards;
using Poker.Players;
using Poker.Resources;

// 命名空间 Poker
namespace Poker
{
    // 内部类 Game
    internal class Game
    {
        // 只读字段 _io 和 _random
        private readonly IReadWrite _io;
        private readonly IRandom _random;

        // 构造函数，接受 IReadWrite 和 IRandom 接口实例
        public Game(IReadWrite io, IRandom random)
        {
            _io = io;
            _random = random;
        }

        // 内部方法 Play
        internal void Play()
        {
            // 使用 _io 输出游戏标题
            _io.Write(Resource.Streams.Title);
            // 使用 _io 输出游戏说明
            _io.Write(Resource.Streams.Instructions);

            // 创建一副扑克牌
            var deck = new Deck();
            // 创建一个玩家对象 human
            var human = new Human(200, _io);
            // 创建一个玩家对象 computer
            var computer = new Computer(200, _io, _random);
            // 创建一个桌子对象 table
            var table = new Table(_io, _random, deck, human, computer);

            // 循环进行游戏直到不再需要继续
            do
            {
                table.PlayHand();
            } while (table.ShouldPlayAnotherHand());
        }
    }
}

```