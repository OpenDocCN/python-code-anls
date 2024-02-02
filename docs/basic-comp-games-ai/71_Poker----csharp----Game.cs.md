# `basic-computer-games\71_Poker\csharp\Game.cs`

```py
// 引入所需的命名空间
using Poker.Cards;
using Poker.Players;
using Poker.Resources;

// 定义名为 Game 的类，访问权限为 internal
namespace Poker
{
    // 定义名为 Game 的类
    internal class Game
    {
        // 声明私有只读字段 _io，类型为 IReadWrite 接口
        private readonly IReadWrite _io;
        // 声明私有只读字段 _random，类型为 IRandom 接口
        private readonly IRandom _random;

        // Game 类的构造函数，接受 IReadWrite 和 IRandom 接口类型的参数
        public Game(IReadWrite io, IRandom random)
        {
            // 将传入的 io 参数赋值给 _io 字段
            _io = io;
            // 将传入的 random 参数赋值给 _random 字段
            _random = random;
        }

        // 定义名为 Play 的方法，访问权限为 internal
        internal void Play()
        {
            // 使用 _io 对象输出游戏标题
            _io.Write(Resource.Streams.Title);
            // 使用 _io 对象输出游戏说明
            _io.Write(Resource.Streams.Instructions);

            // 创建一副扑克牌
            var deck = new Deck();
            // 创建一个名为 human 的玩家对象，初始筹码为 200，使用 _io 对象进行交互
            var human = new Human(200, _io);
            // 创建一个名为 computer 的玩家对象，初始筹码为 200，使用 _io 和 _random 对象进行交互
            var computer = new Computer(200, _io, _random);
            // 创建一个名为 table 的桌子对象，使用 _io、_random、deck、human、computer 对象进行交互
            var table = new Table(_io, _random, deck, human, computer);

            // 循环执行以下操作，直到不再需要继续玩下一手
            do
            {
                // 在桌子上进行一手牌的游戏
                table.PlayHand();
            } while (table.ShouldPlayAnotherHand());
        }
    }
}
```