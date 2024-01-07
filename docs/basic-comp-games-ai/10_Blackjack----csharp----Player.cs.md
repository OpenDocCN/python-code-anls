# `basic-computer-games\10_Blackjack\csharp\Player.cs`

```

// 命名空间为Blackjack，定义了一个名为Player的公共类
namespace Blackjack
{
    // 定义了一个名为Player的公共类
    public class Player
    {
        // 构造函数，接受一个整数参数index
        public Player(int index)
        {
            // 设置Index属性为传入的index值
            Index = index;
            // 设置Name属性为index加1后转换为字符串
            Name = (index + 1).ToString();
            // 初始化Hand属性为一个新的Hand对象
            Hand = new Hand();
            // 初始化SecondHand属性为一个新的Hand对象
            SecondHand = new Hand();
        }

        // 只读属性，获取Index值
        public int Index { get; private set; }

        // 只读属性，获取Name值
        public string Name { get; private set; }

        // 只读属性，获取Hand值
        public Hand Hand { get; private set; }

        // 只读属性，获取SecondHand值
        public Hand SecondHand { get; private set;}

        // 可读写属性，获取或设置RoundBet值
        public int RoundBet { get; set; }

        // 可读写属性，获取或设置RoundWinnings值
        public int RoundWinnings { get; set; }

        // 可读写属性，获取或设置TotalWinnings值
        public int TotalWinnings { get; set; }
    }
}

```