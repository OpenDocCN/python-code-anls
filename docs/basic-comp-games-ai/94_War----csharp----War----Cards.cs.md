# `94_War\csharp\War\Cards.cs`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
        three,  // 三
        four,   // 四
        five,   // 五
        six,    // 六
        seven,  // 七
        eight,  // 八
        nine,   // 九
        ten,    // 十
        jack,   // 侍从
        queen,  // 王后
        king,   // 国王
        ace     // 王牌
    }

    // 代表一张扑克牌的类。
    public class Card
    {
        // 一张牌是一个不可变的对象（即它不能被改变），所以它的花色和点数值是只读的；它们只能在构造函数中设置。
        private readonly Suit suit;  // 花色
        private readonly Rank rank; // 声明一个私有的枚举类型变量 rank

        // 这些字典用于将花色或点数值转换为字符串
        private readonly Dictionary<Suit, string> suitNames = new Dictionary<Suit, string>() // 创建一个花色到字符串的字典
        {
            { Suit.clubs, "C"}, // 将花色 clubs 对应的字符串设为 "C"
            { Suit.diamonds, "D"}, // 将花色 diamonds 对应的字符串设为 "D"
            { Suit.hearts, "H"}, // 将花色 hearts 对应的字符串设为 "H"
            { Suit.spades, "S"}, // 将花色 spades 对应的字符串设为 "S"
        };

        private readonly Dictionary<Rank, string> rankNames = new Dictionary<Rank, string>() // 创建一个点数到字符串的字典
        {
            { Rank.two, "2"}, // 将点数 two 对应的字符串设为 "2"
            { Rank.three, "3"}, // 将点数 three 对应的字符串设为 "3"
            { Rank.four, "4"}, // 将点数 four 对应的字符串设为 "4"
            { Rank.five, "5"}, // 将点数 five 对应的字符串设为 "5"
            { Rank.six, "6"}, // 将点数 six 对应的字符串设为 "6"
            { Rank.seven, "7"}, // 将点数 seven 对应的字符串设为 "7"
            { Rank.eight, "8"}, // 将点数 eight 对应的字符串设为 "8"
            { Rank.nine, "9"},  // 将 Rank.nine 映射为字符串 "9"
            { Rank.ten, "10"},  // 将 Rank.ten 映射为字符串 "10"
            { Rank.jack, "J"},  // 将 Rank.jack 映射为字符串 "J"
            { Rank.queen, "Q"},  // 将 Rank.queen 映射为字符串 "Q"
            { Rank.king, "K"},  // 将 Rank.king 映射为字符串 "K"
            { Rank.ace, "A"},  // 将 Rank.ace 映射为字符串 "A"
        };

        public Card(Suit suit, Rank rank)
        {
            this.suit = suit;  // 初始化 Card 对象的花色属性
            this.rank = rank;  // 初始化 Card 对象的点数属性
        }

        // Relational Operator Overloading.
        //
        // You would normally expect the relational operators to consider both the suit and the
        // rank of a card, but in this program suit doesn't matter so we define the operators to just
        // compare rank.
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，封装成字节流
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    zip.close()  # 关闭 ZIP 对象
    return fdict  # 返回结果字典
        }

        public static bool operator >(Card lhs, Card rhs)
        {
            return rhs < lhs;  # 定义了大于操作符的重载，返回 rhs < lhs 的结果
        }

        public static bool operator <=(Card lhs, Card rhs)
        {
            return !(lhs > rhs);  # 定义了小于等于操作符的重载，返回 !(lhs > rhs) 的结果
        }

        public static bool operator >=(Card lhs, Card rhs)
        {
            return !(lhs < rhs);  # 定义了大于等于操作符的重载，返回 !(lhs < rhs) 的结果
        }

        public override string ToString()
        {
            // N.B. We are using string interpolation to create the card name.
            # 重写了 ToString 方法，使用字符串插值创建卡片名称
            return $"{suitNames[suit]}-{rankNames[rank]}";
        }
```
这段代码是一个方法的结束标志，表示方法的返回值是由suitNames和rankNames数组中的元素组成的字符串。

```
    }

    // A class to represent a deck of cards.
    public class Deck
    {
        public const int deckSize = 52;

        private Card[] theDeck = new Card[deckSize];

        public Deck()
        {
            // Populate theDeck with all the cards in order.
            int i = 0;
            for (Suit suit = Suit.clubs; suit <= Suit.spades; suit++)
            {
                for (Rank rank = Rank.two; rank <= Rank.ace; rank++)
                {
                    theDeck[i] = new Card(suit, rank);
```
这段代码是一个类的构造函数，用于初始化Deck类的实例。在构造函数中，使用嵌套的循环来遍历所有的花色和点数，创建对应的Card对象并将其存储在theDeck数组中。
                    i++;  // 增加变量 i 的值
                }
            }
        }

        // 返回牌组中特定位置的牌。
        // 注意：由于这是一个非常简短的方法，我们将其作为表达式体方法。
        public Card GetCard(int i) => theDeck[i];  // 返回牌组中特定位置的牌

        // 洗牌，这里使用了现代版本的 Fisher-Yates 洗牌算法，参考：
        // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
        public void Shuffle()
        {
            var rand = new Random();  // 创建一个随机数生成器

            // 从牌组末尾开始迭代
            for (int i = deckSize - 1; i >= 1; i--)
            {
                // 生成一个介于 0 到 i 之间的随机数
                int j = rand.Next(0, i);

                // 交换第 i 和第 j 位置上的卡片
                Card temp = theDeck[j];
                theDeck[j] = theDeck[i];
                theDeck[i] = temp;
            }
        }
    }
}
```