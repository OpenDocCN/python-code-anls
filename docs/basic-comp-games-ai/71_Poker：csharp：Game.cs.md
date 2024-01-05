# `d:/src/tocomm/basic-computer-games\71_Poker\csharp\Game.cs`

```
using Poker.Cards;  # 导入Poker.Cards模块，用于处理扑克牌相关的功能
using Poker.Players;  # 导入Poker.Players模块，用于处理玩家相关的功能
using Poker.Resources;  # 导入Poker.Resources模块，用于处理游戏资源相关的功能

namespace Poker;  # 定义Poker命名空间

internal class Game  # 定义Game类，限定只能在当前程序集内部访问
{
    private readonly IReadWrite _io;  # 声明私有只读字段_io，类型为IReadWrite接口
    private readonly IRandom _random;  # 声明私有只读字段_random，类型为IRandom接口

    public Game(IReadWrite io, IRandom random)  # Game类的构造函数，接受io和random两个参数
    {
        _io = io;  # 将传入的io参数赋值给_io字段
        _random = random;  # 将传入的random参数赋值给_random字段
    }

    internal void Play()  # 定义Play方法，限定只能在当前程序集内部访问
    {
        _io.Write(Resource.Streams.Title)  # 调用_io的Write方法，将Resource.Streams.Title作为参数传入
```
```python
        _io.Write(Resource.Streams.Instructions);  // 将指令写入输出流_io

        var deck = new Deck();  // 创建一副新的扑克牌
        var human = new Human(200, _io);  // 创建一个玩家对象human，初始筹码为200，输出流为_io
        var computer = new Computer(200, _io, _random);  // 创建一个电脑玩家对象computer，初始筹码为200，输出流为_io，随机数生成器为_random
        var table = new Table(_io, _random, deck, human, computer);  // 创建一个牌桌对象table，输出流为_io，随机数生成器为_random，扑克牌为deck，玩家为human和computer

        do
        {
            table.PlayHand();  // 进行一局游戏
        } while (table.ShouldPlayAnotherHand());  // 当需要继续进行下一局游戏时继续循环
    }
}
```