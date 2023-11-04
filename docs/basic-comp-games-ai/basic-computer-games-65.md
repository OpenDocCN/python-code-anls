# BasicComputerGames源码解析 65

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Poker

You and the computer are opponents in this game of draw poker. At the start of the game, each player is given $200. The game ends when either player runs out of money, although if you go broke the computer will offer to buy back your wristwatch or diamond tie tack.

The computer opens the betting before the draw; you open the betting after the draw. If you don’t have a hand that’s worth anything and you want to fold, bet 0. Prior to the draw, to check the draw, you may bet .5. Of course, if the computer has made a bet, you must match it in order to draw or, if you have a good hand, you may raise the bet at any time.

The author is A. Christopher Hall of Trinity College, Hartford, Connecticut.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=129)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=144)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- If you bet more than the computer has, it will still see you, resulting in a negative balance.  (To handle this properly, the computer would need to go "all in" and reduce your bet to an amount it can match; or else lose the game, which is what happens to the human player in the same situation.)

- If you are low on cash and sell your watch, then make a bet much smaller than the amount you just gained from the watch, it sometimes nonetheless tells you you "blew your wad" and ends the game.

- When the watch is sold (in either direction), the buyer does not actually lose any money.

- The code in the program about selling your tie tack is unreachable due to a logic bug.


#### Porting Notes

(please note any difficulties or challenges in porting here)


# `71_Poker/csharp/Game.cs`

这段代码是一个Poker游戏的实现，它包括以下几个主要部分：

1. 引入了Poker卡牌类、玩家类和资源类。这些类将用于在游戏中处理卡牌、玩家和资源。
2. 构造函数，它接受一个IO读写流和一个随机数生成器。这些流将用于输出游戏标题、游戏说明和随机数。
3. Play()方法，它是游戏的核心部分。它在这里读取输入文件中的游戏标题和游戏说明，创建一个游戏牌、一个人类玩家和一个计算机玩家，然后开始轮流玩牌。
4. 在游戏牌局中，使用PlayHand()方法来处理玩家的请求，如果牌局结束后仍然没有完成游戏，就循环调用ShouldPlayAnotherHand()方法来继续游戏。


```
using Poker.Cards;
using Poker.Players;
using Poker.Resources;

namespace Poker;

internal class Game
{
    private readonly IReadWrite _io;
    private readonly IRandom _random;

    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    internal void Play()
    {
        _io.Write(Resource.Streams.Title);
        _io.Write(Resource.Streams.Instructions);

        var deck = new Deck();
        var human = new Human(200, _io);
        var computer = new Computer(200, _io, _random);
        var table = new Table(_io, _random, deck, human, computer);

        do
        {
            table.PlayHand();
        } while (table.ShouldPlayAnotherHand());
    }
}

```

# `71_Poker/csharp/IReadWriteExtensions.cs`



该代码是一个名为Poker.Strategies的命名类，其中包含多个内部静态方法，用于与用户交互以获取其在Poker游戏中的策略信息。这些方法可以读取用户输入，并在输入与预先定义的答案不匹配时要求重新输入。

下面是每个方法的详细解释：

1. `ReadYesNo(this IReadWrite io, string prompt)`：该方法用于获取用户是否愿意在当前赌注上赌钱。它使用一个 while 循环和一个 `var response = io.ReadString(prompt);` 的方法来读取用户输入并将其存储在 `response` 变量中。如果输入与答案 "是" 完全匹配，该方法返回 `true`，否则返回 `false`。

2. `ReadNumber(this IReadWrite io)`：该方法用于从用户那里读取一个数字，并将其存储在 `response` 变量中。如果用户输入数字，该方法将其读取并将其存储在 `response` 变量中。如果用户输入不是数字，该方法将发出一个询问，并要求用户重新输入。

3. `ReadNumber(this IReadWrite io, string prompt, int max, string maxPrompt)`：该方法用于从用户那里读取一个整数或多个数字，并将其存储在 `response` 变量中。它使用一个 while 循环和一个 `var response = io.ReadNumber("");` 的方法来读取用户输入。如果用户输入是数字，该方法将读取并将其存储在 `response` 变量中。如果用户输入不是数字，该方法将要求用户输入一个最大的数字，并将该数字存储在 `maxPrompt` 变量中。如果用户在 while 循环中输出了 ',' 或 ',' 字符串，该方法将结束并抛出异常。

4. `ReadHumanStrategy(this IReadWrite io, bool noCurrentBets)`：该方法用于从用户那里读取他们的策略，并返回一个具体的Poker策略。它使用一个 while 循环和一个 `var bet = io.ReadNumber("What is your bet");` 的方法来读取用户输入并将其存储在 `bet` 变量中。如果用户输入不是 "是"，该方法将返回Poker.Check策略，否则，如果用户输入是 "是"，该方法将返回Poker.Bet(bet)策略。如果用户在 while 循环中输出了 ',' 或 ',' 字符串，该方法将结束并抛出异常。


```
using Poker.Strategies;
using static System.StringComparison;

namespace Poker;

internal static class IReadWriteExtensions
{
    internal static bool ReadYesNo(this IReadWrite io, string prompt)
    {
        while (true)
        {
            var response = io.ReadString(prompt);
            if (response.Equals("YES", InvariantCultureIgnoreCase)) { return true; }
            if (response.Equals("NO", InvariantCultureIgnoreCase)) { return false; }
            io.WriteLine("Answer Yes or No, please.");
        }
    }

    internal static float ReadNumber(this IReadWrite io) => io.ReadNumber("");

    internal static int ReadNumber(this IReadWrite io, string prompt, int max, string maxPrompt)
    {
        io.Write(prompt);
        while (true)
        {
            var response = io.ReadNumber();
            if (response <= max) { return (int)response; }
            io.WriteLine(maxPrompt);
        }
    }

    internal static Strategy ReadHumanStrategy(this IReadWrite io, bool noCurrentBets)
    {
        while(true)
        {
            io.WriteLine();
            var bet = io.ReadNumber("What is your bet");
            if (bet != (int)bet)
            {
                if (noCurrentBets && bet == .5) { return Strategy.Check; }
                io.WriteLine("No small change, please.");
                continue;
            }
            if (bet == 0) { return Strategy.Fold; }
            return Strategy.Bet(bet);
        }
    }
}
```

# `71_Poker/csharp/Program.cs`

这段代码使用了三个命名空间：Games.Common.IO,Games.Common.Randomness,Poker。

它创建了一个名为Game的类，并继承自Poker类。

新创建的Game实例得到了一个ConsoleIO和一个RandomNumberGenerator对象，分别用于输出和生成随机数。

然后，使用这两个对象，创建了一个新的Game实例，并调用其的Play()方法，这个方法实现了Game类的一个行为，代表游戏开始玩游戏。

最后，这个Game实例被继承并且实现了Poker类，因此在Game开始玩游戏的时候，将使用Poker类中的所有方法和属性。


```
global using Games.Common.IO;
global using Games.Common.Randomness;
global using Poker;

new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `71_Poker/csharp/Table.cs`

This appears to be a text-based card game where players take turns playing cards and the pot is awarded to the player who wins. The game has different types of players, including a Human, a Computer, and a Player, each of which has its own set of abilities and drawbacks.

The Human player has the ability to draw additional cards when they bet, but their hand is limited to a maximum of 10 cards. The Computer player has the ability to take additional cards when they bet, but their hand is limited to a maximum of 20 cards.

The game also has a Pot which is awarded to the player who wins, and a Folded指示物， which indicates whether the current player should draw a card and end their turn, or if they should take additional cards and proceed with their next turn.

There is also a True指示物， which indicates whether the last card dealt was a Folded card or not.


```
using Poker.Cards;
using Poker.Players;
using Poker.Strategies;

namespace Poker;

internal class Table
{
    private readonly IReadWrite _io;
    private readonly IRandom _random;
    public int Pot;

    public Table(IReadWrite io, IRandom random, Deck deck, Human human, Computer computer)
    {
        _io = io;
        _random = random;
        Deck = deck;
        Human = human;
        Computer = computer;

        human.Sit(this);
        computer.Sit(this);
    }

    public int Ante { get; } = 5;
    public Deck Deck { get; }
    public Human Human { get; }
    public Computer Computer { get; }

    internal void PlayHand()
    {
        while (true)
        {
            _io.WriteLine();
            Computer.CheckFunds();
            if (Computer.IsBroke) { return; }

            _io.WriteLine($"The ante is ${Ante}.  I will deal:");
            _io.WriteLine();
            if (Human.Balance <= Ante)
            {
                Human.RaiseFunds();
                if (Human.IsBroke) { return; }
            }

            Deal(_random);

            _io.WriteLine();
            GetWagers("I'll open with ${0}", "I check.", allowRaiseAfterCheck: true);
            if (SomeoneIsBroke() || SomeoneHasFolded()) { return; }

            Draw();

            GetWagers();
            if (SomeoneIsBroke()) { return; }
            if (!Human.HasBet)
            {
                GetWagers("I'll bet ${0}", "I'll check");
            }
            if (SomeoneIsBroke() || SomeoneHasFolded()) { return; }
            if (GetWinner() is { } winner)
            {
                winner.TakeWinnings();
                return;
            }
        }
    }

    private void Deal(IRandom random)
    {
        Deck.Shuffle(random);

        Pot = Human.AnteUp() + Computer.AnteUp();

        Human.NewHand();
        Computer.NewHand();

        _io.WriteLine("Your hand:");
        _io.Write(Human.Hand);
    }

    private void Draw()
    {
        _io.WriteLine();
        _io.Write("Now we draw -- ");
        Human.DrawCards();
        Computer.DrawCards();
        _io.WriteLine();
    }

    private void GetWagers(string betFormat, string checkMessage, bool allowRaiseAfterCheck = false)
    {
        if (Computer.Strategy is Bet)
        {
            Computer.Bet = Computer.GetWager(Computer.Strategy.Value);
            if (Computer.IsBroke) { return; }

            _io.WriteLine(betFormat, Computer.Bet);
        }
        else
        {
            _io.WriteLine(checkMessage);
            if (!allowRaiseAfterCheck) { return; }
        }

        GetWagers();
    }

    private void GetWagers()
    {
        while (true)
        {
            Human.HasBet = false;
            while (true)
            {
                if (Human.SetWager()) { break; }
                if (Human.IsBroke || Human.HasFolded) { return; }
            }
            if (Human.Bet == Computer.Bet)
            {
                CollectBets();
                return;
            }
            if (Computer.Strategy is Fold)
            {
                if (Human.Bet > 5)
                {
                    Computer.Fold();
                    _io.WriteLine("I fold.");
                    return;
                }
            }
            if (Human.Bet > 3 * Computer.Strategy.Value)
            {
                if (Computer.Strategy is not Raise)
                {
                    _io.WriteLine("I'll see you.");
                    Computer.Bet = Human.Bet;
                    CollectBets();
                    return;
                }
            }

            var raise = Computer.GetWager(Human.Bet - Computer.Bet);
            if (Computer.IsBroke) { return; }
            _io.WriteLine($"I'll see you, and raise you {raise}");
            Computer.Bet = Human.Bet + raise;
        }
    }

    internal void CollectBets()
    {
        Human.Balance -= Human.Bet;
        Computer.Balance -= Computer.Bet;
        Pot += Human.Bet + Computer.Bet;
    }

    private bool SomeoneHasFolded()
    {
        if (Human.HasFolded)
        {
            _io.WriteLine();
            Computer.TakeWinnings();
        }
        else if (Computer.HasFolded)
        {
            _io.WriteLine();
            Human.TakeWinnings();
        }
        else
        {
            return false;
        }

        Pot = 0;
        return true;
    }

    private bool SomeoneIsBroke() => Human.IsBroke || Computer.IsBroke;

    private Player? GetWinner()
    {
        _io.WriteLine();
        _io.WriteLine("Now we compare hands:");
        _io.WriteLine("My hand:");
        _io.Write(Computer.Hand);
        _io.WriteLine();
        _io.WriteLine($"You have {Human.Hand.Name}");
        _io.WriteLine($"and I have {Computer.Hand.Name}");
        if (Computer.Hand > Human.Hand) { return Computer; }
        if (Human.Hand > Computer.Hand) { return Human; }
        _io.WriteLine("The hand is drawn.");
        _io.WriteLine($"All $ {Pot} remains in the pot.");
        return null;
    }

    internal bool ShouldPlayAnotherHand()
    {
        if (Computer.IsBroke)
        {
            _io.WriteLine("I'm busted.  Congratulations!");
            return true;
        }

        if (Human.IsBroke)
        {
            _io.WriteLine("Your wad is shot.  So long, sucker!");
            return true;
        }

        _io.WriteLine($"Now I have $ {Computer.Balance} and you have $ {Human.Balance}");
        return _io.ReadYesNo("Do you wish to continue");
    }
}
```

# `71_Poker/csharp/Cards/Card.cs`

这段代码定义了一个名为 "Card" 的内部结构体，它有两个成员变量，一个是 "Rank"，另一个是 "Suit"。

接下来，定义了四个操作符重载为 ">" 和 ">="，它们的作用是用来在排序卡牌时进行排序。

接着，定义了一个名为 "ToString" 的静态方法，它用于将卡牌的字符串表示法输出。

最后，在 "Card" 结构体的 "override" 注明了 "ToString" 方法，这个方法会使用成员变量 "Rank" 和 "Suit" 来创建一个字符串，并将它们组合成一个字符串。


```
namespace Poker.Cards;

internal record struct Card (Rank Rank, Suit Suit)
{
    public override string ToString() => $"{Rank} of {Suit}";

    public static bool operator <(Card x, Card y) => x.Rank < y.Rank;
    public static bool operator >(Card x, Card y) => x.Rank > y.Rank;

    public static int operator -(Card x, Card y) => x.Rank - y.Rank;
}

```

# `71_Poker/csharp/Cards/Deck.cs`

这段代码定义了一个名为 "Deck" 的类，用于代表一个扑克牌一副。这个类包含一个名为 "Decade" 的方法，用于洗牌，以及一个名为 "DealHand" 的方法，用于处理一副扑克牌的牌局。

在 "Deck" 类的 "Decade" 方法中，使用 "SelectMany" 方法从 "Ranks" 命名空间中选取 "Suit" 命名空间中的所有元素，并使用 "ToArray" 方法将结果存储到一个 "Card" 类的一维数组中。这个数组的每个元素都是一个扑克牌的 "Card" 类实例。

在 "DealHand" 方法中，首先使用一个循环来设置 "DealCard" 方法的索引，从数组的第一个元素开始。然后使用另一个循环来从数组的索引随机选择一个元素，并将其复制到 "DealCard" 方法的索引中。这样，每次循环都会从数组的第一个元素开始，数的索引将循环遍历整个数组，从而实现了洗牌的效果。

另外，在 "DealHand" 方法中，还定义了一个 "Hand" 类，用于表示一副扑克牌的牌局。这个类包含一个名为 "DealHand" 的方法，用于返回一副扑克牌的牌局。


```
using static Poker.Cards.Rank;

namespace Poker.Cards;

internal class Deck
{
    private readonly Card[] _cards;
    private int _nextCard;

    public Deck()
    {
        _cards = Ranks.SelectMany(r => Enum.GetValues<Suit>().Select(s => new Card(r, s))).ToArray();
    }

    public void Shuffle(IRandom _random)
    {
        for (int i = 0; i < _cards.Length; i++)
        {
            var j = _random.Next(_cards.Length);
            (_cards[i], _cards[j]) = (_cards[j], _cards[i]);
        }
        _nextCard = 0;
    }

    public Card DealCard() => _cards[_nextCard++];

    public Hand DealHand() => new Hand(Enumerable.Range(0, 5).Select(_ => DealCard()));
}

```

# `71_Poker/csharp/Cards/Hand.cs`

In this implementation, the `Hand` class represents a hand of playing cards. It has a collection of `Card` objects and implements the `IComparable` interface to allow for sorting and comparing hands.

The `Hand` class has a constructor that initializes the hand with a deck of cards and sets the initial rank and high card to null. The hand also has a `sortedCards` field to store the sorted deck of cards and a `keepMask` field to store the mask used for keeping the hand valid.

The `Hand` class has several methods:

* `Initialize`: This method initializes the hand with a deck of cards and sets the initial rank and high card to null.
* `Reduce`: This method determines the best possible hand rank for the cards in the hand and returns it.
* `IsCopied`: This method checks if the hand is copied from another hand.
* `GetRank`: This method returns the rank of the highest card in the hand.
* `GetHighCard`: This method returns the high card of the hand.
* `SetHighCard`: This method sets the high card of the hand to the current high card.
* `SetRank`: This method sets the rank of the hand to the current rank.
* `SetMask`: This method sets the mask used for keeping the hand valid to the current mask.
* `CopyFromHand`: This method creates a copy of the hand by comparing the index of the cards.
* `IsSorted`: This method checks if the hand is sorted in ascending order.

The `Hand` class also implements the `IEnumerable<Hand>` interface, which allows it to be converted to an iterable of `Hand` objects.

Overall, this implementation provides a useful representation of a hand of playing cards in Python.


```
using System.Text;
using static Poker.Cards.HandRank;
namespace Poker.Cards;

internal class Hand
{
    public static readonly Hand Empty = new Hand();

    private readonly Card[] _cards;

    private Hand()
    {
        _cards = Array.Empty<Card>();
        Rank = None;
    }

    public Hand(IEnumerable<Card> cards)
        : this(cards, isAfterDraw: false)
    {
    }

    private Hand(IEnumerable<Card> cards, bool isAfterDraw)
    {
        _cards = cards.ToArray();
        (Rank, HighCard, KeepMask) = Analyze();

        IsWeak = Rank < PartialStraight
            || Rank == PartialStraight && isAfterDraw
            || Rank <= TwoPair && HighCard.Rank <= 6;
    }

    public string Name => Rank.ToString(HighCard);
    public HandRank Rank { get; }
    public Card HighCard { get; }
    public int KeepMask { get; set; }
    public bool IsWeak { get; }

    public Hand Replace(int cardNumber, Card newCard)
    {
        if (cardNumber < 1 || cardNumber > _cards.Length) { return this; }

        _cards[cardNumber - 1] = newCard;
        return new Hand(_cards, isAfterDraw: true);
    }

    private (HandRank, Card, int) Analyze()
    {
        var suitMatchCount = 0;
        for (var i = 0; i < _cards.Length; i++)
        {
            if (i < _cards.Length-1 && _cards[i].Suit == _cards[i+1].Suit)
            {
                suitMatchCount++;
            }
        }
        if (suitMatchCount == 4)
        {
            return (Flush, _cards[0], 0b11111);
        }
        var sortedCards = _cards.OrderBy(c => c.Rank).ToArray();

        var handRank = Schmaltz;
        var keepMask = 0;
        Card highCard = default;
        for (var i = 0; i < sortedCards.Length - 1; i++)
        {
            var matchesNextCard = sortedCards[i].Rank == sortedCards[i+1].Rank;
            var matchesPreviousCard = i > 0 && sortedCards[i].Rank == sortedCards[i - 1].Rank;

            if (matchesNextCard)
            {
                keepMask |= 0b11 << i;
                highCard = sortedCards[i];
                handRank = matchesPreviousCard switch
                {
                    _ when handRank < Pair => Pair,
                    true when handRank == Pair => Three,
                    _ when handRank == Pair => TwoPair,
                    _ when handRank == TwoPair => FullHouse,
                    true => Four,
                    _ => FullHouse
                };
            }
        }
        if (keepMask == 0)
        {
            if (sortedCards[3] - sortedCards[0] == 3)
            {
                keepMask=0b1111;
                handRank=PartialStraight;
            }
            if (sortedCards[4] - sortedCards[1] == 3)
            {
                if (handRank == PartialStraight)
                {
                    return (Straight, sortedCards[4], 0b11111);
                }
                handRank=PartialStraight;
                keepMask=0b11110;
            }
        }
        return handRank < PartialStraight
            ? (Schmaltz, sortedCards[4], 0b11000)
            : (handRank, highCard, keepMask);
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        for (var i = 0; i < _cards.Length; i++)
        {
            var cardDisplay = $" {i+1} --  {_cards[i]}";
            // Emulates the effect of the BASIC PRINT statement using the ',' to align text to 14-char print zones
            sb.Append(cardDisplay.PadRight(cardDisplay.Length + 14 - cardDisplay.Length % 14));
            if (i % 2 == 1)
            {
                sb.AppendLine();
            }
        }
        sb.AppendLine();
        return sb.ToString();
    }

    public static bool operator >(Hand x, Hand y) =>
        x.Rank > y.Rank ||
        x.Rank == y.Rank && x.HighCard > y.HighCard;

    public static bool operator <(Hand x, Hand y) =>
        x.Rank < y.Rank ||
        x.Rank == y.Rank && x.HighCard < y.HighCard;
}

```

# `71_Poker/csharp/Cards/HandRank.cs`

This code defines a `HandRank` class with several constructors that take an integer value and a string display name. The string display name is used to display the rank of the hand. The `_suffixSelector` field is a function that is called with the `Card` object to determine the suffix of the rank string.

The `TwoPair`, `Three`, `Straight`, and `Flush` constructors are calculated based on the rank value. The `Four` constructor is defined but not implemented.

The `>` and `<` operators compare the values of the `_value` field, which is used as the base for the comparison. The `>=` and `<=` operators compare the values to the minimum and maximum values of the `_value` field, respectively.

The `ToString` method is defined to return the display name of the rank string, along with the suffix selector.


```
namespace Poker.Cards;

internal class HandRank
{
    public static HandRank None = new(0, "");
    public static HandRank Schmaltz = new(1, "schmaltz, ", c => $"{c.Rank} high");
    public static HandRank PartialStraight = new(2, ""); // The original code does not assign a display string here
    public static HandRank Pair = new(3, "a pair of ", c => $"{c.Rank}'s");
    public static HandRank TwoPair = new(4, "two pair, ", c => $"{c.Rank}'s");
    public static HandRank Three = new(5, "three ", c => $"{c.Rank}'s");
    public static HandRank Straight = new(6, "straight", c => $"{c.Rank} high");
    public static HandRank Flush = new(7, "a flush in ", c => c.Suit.ToString());
    public static HandRank FullHouse = new(8, "full house, ", c => $"{c.Rank}'s");
    public static HandRank Four = new(9, "four ", c => $"{c.Rank}'s");
    // The original code does not detect a straight flush or royal flush

    private readonly int _value;
    private readonly string _displayName;
    private readonly Func<Card, string> _suffixSelector;

    private HandRank(int value, string displayName, Func<Card, string>? suffixSelector = null)
    {
        _value = value;
        _displayName = displayName;
        _suffixSelector = suffixSelector ?? (_ => "");
    }

    public string ToString(Card highCard) => $"{_displayName}{_suffixSelector.Invoke(highCard)}";

    public static bool operator >(HandRank x, HandRank y) => x._value > y._value;
    public static bool operator <(HandRank x, HandRank y) => x._value < y._value;
    public static bool operator >=(HandRank x, HandRank y) => x._value >= y._value;
    public static bool operator <=(HandRank x, HandRank y) => x._value <= y._value;
}

```

# `71_Poker/csharp/Cards/Rank.cs`

This is a class that defines a `Rank` enum that represents a rank or score. The class defines an instance of the `Rank` enum for each of the values 2 to 14, and also defines a `Two` and `Three` properties that are instances of the `Rank` enum with the value 2 and 3 respectively.

The `Rank` class defines several overloads for the `==` and `!=` operators, the `<` and `>` operators, and the `<>` operator. It also defines the `CompareTo` and `GetHashCode` methods.

The `Rank` class is used to compare `Rank` objects, and is intended to be used in a的反例诊断场景中。例如，为了确认一个用户是否具有指定排名，开发人员可以使用 `Rank.CompareTo(Rank.Two)` 来确保对象具有所需的排名。如果比较结果为 `true`，则表明对象具有所需的排名，否则将返回 `false`。


```
namespace Poker.Cards;

internal struct Rank : IComparable<Rank>
{
    public static IEnumerable<Rank> Ranks => new[]
    {
        Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Jack, Queen, King, Ace
    };

    public static Rank Two = new(2);
    public static Rank Three = new(3);
    public static Rank Four = new(4);
    public static Rank Five = new(5);
    public static Rank Six = new(6);
    public static Rank Seven = new(7);
    public static Rank Eight = new(8);
    public static Rank Nine = new(9);
    public static Rank Ten = new(10);
    public static Rank Jack = new(11, "Jack");
    public static Rank Queen = new(12, "Queen");
    public static Rank King = new(13, "King");
    public static Rank Ace = new(14, "Ace");

    private readonly int _value;
    private readonly string _name;

    private Rank(int value, string? name = null)
    {
        _value = value;
        _name = name ?? $" {value} ";
    }

    public override string ToString() => _name;

    public int CompareTo(Rank other) => this - other;

    public static bool operator <(Rank x, Rank y) => x._value < y._value;
    public static bool operator >(Rank x, Rank y) => x._value > y._value;
    public static bool operator ==(Rank x, Rank y) => x._value == y._value;
    public static bool operator !=(Rank x, Rank y) => x._value != y._value;

    public static int operator -(Rank x, Rank y) => x._value - y._value;

    public static bool operator <=(Rank rank, int value) => rank._value <= value;
    public static bool operator >=(Rank rank, int value) => rank._value >= value;

    public override bool Equals(object? obj) => obj is Rank other && this == other;

    public override int GetHashCode() => _value.GetHashCode();
}

```

# `71_Poker/csharp/Cards/Suit.cs`

这段代码定义了一个名为 "Suit" 的内部枚举类型，包含了四个枚举值：Clubs、Diamonds、Hearts 和 Spades。

在 .NET 框架中，namespace 关键字用于声明命名空间。在这里，namespace 关键字定义了一个名为 "Poker.Cards" 的命名空间，用于包含 "Suit" 类型和其他枚举类型。

.NET 框架的 . class 关键字定义了一个类，而 . namespace 关键字则位于类的外部。因此，这段代码可以被看作是一个位于 "Poker.Cards" 类中的类定义。


```
namespace Poker.Cards;

internal enum Suit
{
    Clubs,
    Diamonds,
    Hearts,
    Spades
}

```

# `71_Poker/csharp/Players/Computer.cs`

This looks like a implementation of the stock market game. The player can place bets on whether they think the stock price will go up or down. Bets are placed on a number between 0 and 100, with a minimum bet of 25. The player can also buy back their watch for a certain amount of money. If the player wins, they are paid out.

There is a catch, the game does not seem to have much logic for how the price of the stock is determined. It is simply set to a random number between 25 and 75, with a default of 0. This could be an area for improvement in a future version of the game.

Additionally, the game does not seem to have any way of telling if the player is broke, which could be an important feature for a stock market game.


```
using Poker.Cards;
using Poker.Strategies;
using static System.StringComparison;

namespace Poker.Players;

internal class Computer : Player
{
    private readonly IReadWrite _io;
    private readonly IRandom _random;

    public Computer(int bank, IReadWrite io, IRandom random)
        : base(bank)
    {
        _io = io;
        _random = random;
        Strategy = Strategy.None;
    }

    public Strategy Strategy { get; set; }

    public override void NewHand()
    {
        base.NewHand();

        Strategy = (Hand.IsWeak, Hand.Rank < HandRank.Three, Hand.Rank < HandRank.FullHouse) switch
        {
            (true, _, _) when _random.Next(10) < 2 => Strategy.Bluff(23, 0b11100),
            (true, _, _) when _random.Next(10) < 2 => Strategy.Bluff(23, 0b11110),
            (true, _, _) when _random.Next(10) < 1 => Strategy.Bluff(23, 0b11111),
            (true, _, _) => Strategy.Fold,
            (false, true, _) => _random.Next(10) < 2 ? Strategy.Bluff(23) : Strategy.Check,
            (false, false, true) => Strategy.Bet(35),
            (false, false, false) => _random.Next(10) < 1 ? Strategy.Bet(35) : Strategy.Raise
        };
    }

    protected override void DrawCards(Deck deck)
    {
        var keepMask = Strategy.KeepMask ?? Hand.KeepMask;
        var count = 0;
        for (var i = 1; i <= 5; i++)
        {
            if ((keepMask & (1 << (i - 1))) == 0)
            {
                Hand = Hand.Replace(i, deck.DealCard());
                count++;
            }
        }

        _io.WriteLine();
        _io.Write($"I am taking {count} card");
        if (count != 1)
        {
            _io.WriteLine("s");
        }

        Strategy = (Hand.IsWeak, Hand.Rank < HandRank.Three, Hand.Rank < HandRank.FullHouse) switch
        {
            _ when Strategy is Bluff => Strategy.Bluff(28),
            (true, _, _) => Strategy.Fold,
            (false, true, _) => _random.Next(10) == 0 ? Strategy.Bet(19) : Strategy.Raise,
            (false, false, true) => _random.Next(10) == 0 ? Strategy.Bet(11) : Strategy.Bet(19),
            (false, false, false) => Strategy.Raise
        };
    }

    public int GetWager(int wager)
    {
        wager += _random.Next(10);
        if (Balance < Table.Human.Bet + wager)
        {
            if (Table.Human.Bet == 0) { return Balance; }

            if (Balance >= Table.Human.Bet)
            {
                _io.WriteLine("I'll see you.");
                Bet = Table.Human.Bet;
                Table.CollectBets();
            }
            else
            {
                RaiseFunds();
            }
        }

        return wager;
    }

    public bool TryBuyWatch()
    {
        if (!Table.Human.HasWatch) { return false; }

        var response = _io.ReadString("Would you like to sell your watch");
        if (response.StartsWith("N", InvariantCultureIgnoreCase)) { return false; }

        var (value, message) = (_random.Next(10) < 7) switch
        {
            true => (75, "I'll give you $75 for it."),
            false => (25, "That's a pretty crummy watch - I'll give you $25.")
        };

        _io.WriteLine(message);
        Table.Human.SellWatch(value);
        // The original code does not have the computer part with any money

        return true;
    }

    public void RaiseFunds()
    {
        if (Table.Human.HasWatch) { return; }

        var response = _io.ReadString("Would you like to buy back your watch for $50");
        if (response.StartsWith("N", InvariantCultureIgnoreCase)) { return; }

        // The original code does not deduct $50 from the player
        Balance += 50;
        Table.Human.ReceiveWatch();
        IsBroke = true;
    }

    public void CheckFunds() { IsBroke = Balance <= Table.Ante; }

    public override void TakeWinnings()
    {
        _io.WriteLine("I win.");
        base.TakeWinnings();
    }
}

```

# `71_Poker/csharp/Players/Human.cs`

This is a sample implementation of a video poker game. It uses a `Cards` class to keep track of the cards in the game, a `Bet` class to keep track of the current bet, a `Table` class to store the game table, and an `IO` class to receive input from the user.

The `Cards` class has methods for dealing a card, getting the number of cards in a hand, and replacing the card from the hand. The `Bet` class has methods for setting the current bet, raising the bet, and folding the hand.

The `Table` class has methods for collecting bets, displaying the game table, and displaying the user's hand. It also has a `Computer` class that implements the logic for the video poker game.

The `IO` class has methods for reading input from the user, displaying the game table, and displaying the user's hand.

The `SetWager` method checks if the player can place a bet and if the player's bet is less than the computer's bet. If the player can't see the computer's bet, it folds the hand. If the player has enough balance to pay the computer's bet, it raises the bet and adds the computer's bet to the player's balance.

The `RaiseFunds` method raises the bet for the player.

The `ReceiveWatch` method displays the game table to the player and waits for them to buy or sell the watch.

The `SellWatch` method sells the watch to the player for a specified amount of tokens.

The `TakeWinnings` method displays the game table to the player and updates the winner.


```
using Poker.Cards;
using Poker.Strategies;

namespace Poker.Players;

internal class Human : Player
{
    private readonly IReadWrite _io;

    public Human(int bank, IReadWrite io)
        : base(bank)
    {
        HasWatch = true;
        _io = io;
    }

    public bool HasWatch { get; set; }

    protected override void DrawCards(Deck deck)
    {
        var count = _io.ReadNumber("How many cards do you want", 3, "You can't draw more than three cards.");
        if (count == 0) { return; }

        _io.WriteLine("What are their numbers:");
        for (var i = 1; i <= count; i++)
        {
            Hand = Hand.Replace((int)_io.ReadNumber(), deck.DealCard());
        }

        _io.WriteLine("Your new hand:");
        _io.Write(Hand);
    }

    internal bool SetWager()
    {
        var strategy = _io.ReadHumanStrategy(Table.Computer.Bet == 0 && Bet == 0);
        if (strategy is Strategies.Bet or Check)
        {
            if (Bet + strategy.Value < Table.Computer.Bet)
            {
                _io.WriteLine("If you can't see my bet, then fold.");
                return false;
            }
            if (Balance - Bet - strategy.Value >= 0)
            {
                HasBet = true;
                Bet += strategy.Value;
                return true;
            }
            RaiseFunds();
        }
        else
        {
            Fold();
            Table.CollectBets();
        }
        return false;
    }

    public void RaiseFunds()
    {
        _io.WriteLine();
        _io.WriteLine("You can't bet with what you haven't got.");

        if (Table.Computer.TryBuyWatch()) { return; }

        // The original program had some code about selling a tie tack, but due to a fault
        // in the logic the code was unreachable. I've omitted it in this port.

        IsBroke = true;
    }

    public void ReceiveWatch()
    {
        // In the original code the player does not pay any money to receive the watch back.
        HasWatch = true;
    }

    public void SellWatch(int amount)
    {
        HasWatch = false;
        Balance += amount;
    }

    public override void TakeWinnings()
    {
        _io.WriteLine("You win.");
        base.TakeWinnings();
    }
}

```

# `71_Poker/csharp/Players/Player.cs`

这段代码定义了一个名为 `Player` 的内部抽象类，用于表示一个玩家。这个类包含了以下属性和方法：

* `Hand`：玩家的手牌，通过 `Table.Deck.DealHand()` 方法从发牌堆中抽出。
* `Balance`：玩家的当前资金，通过 `Balance -= Table.Ante` 计算得到。
* `_hasFolded`：一个布尔值，表示玩家是否已经摊牌。
* `Sit`：玩家坐下，即 `Table = table`。
* `NewHand`：一个新的 `Player` 实例，重置了 `Hand` 和 `Balance`，并更新了 `_hasFolded`。
* `AnteUp`：玩家下注，将 `Balance` 减少了 `Table.Ante`，并将筹码留在了桌面上。
* `DrawCards`：玩家打牌，通过 `DrawCards(Table.Deck)` 方法从发牌堆中抽出了所有牌。
* `TakeWinnings`：玩家获得赌注，将 `Balance` 增加了 `Table.Pot`，并将所有筹码从发牌堆中取回。
* `Fold`：玩家摊牌，将 `_hasFolded` 设置为 `true`。


```
using Poker.Cards;

namespace Poker.Players;

internal abstract class Player
{
    private Table? _table;
    private bool _hasFolded;

    protected Player(int bank)
    {
        Hand = Hand.Empty;
        Balance = bank;
    }

    public Hand Hand { get; set; }
    public int Balance { get; set; }
    public bool HasBet { get; set; }
    public int Bet { get; set; }
    public bool HasFolded => _hasFolded;
    public bool IsBroke { get; protected set; }

    protected Table Table =>
        _table ?? throw new InvalidOperationException("The player must be sitting at the table.");

    public void Sit(Table table) => _table = table;

    public virtual void NewHand()
    {
        Bet = 0;
        Hand = Table.Deck.DealHand();
        _hasFolded = false;
    }

    public int AnteUp()
    {
        Balance -= Table.Ante;
        return Table.Ante;
    }

    public void DrawCards()
    {
        Bet = 0;
        DrawCards(Table.Deck);
    }

    protected abstract void DrawCards(Deck deck);

    public virtual void TakeWinnings()
    {
        Balance += Table.Pot;
        Table.Pot = 0;
    }

    public void Fold()
    {
        _hasFolded = true;
    }
}

```

# `71_Poker/csharp/Resources/Resource.cs`



该代码是一个名为 `Resource` 的类，它从 `System.Reflection` 和 `System.Runtime.CompilerServices` 命名空间中获取类定义。

内部包含一个名为 `Instructions` 的类，它使用 `GetStream` 方法从资源文件中读取指令流。

内部还包含一个名为 `Title` 的类，它使用 `GetStream` 方法从资源文件中读取标题流。

最后，该类提供一个名为 `GetStream` 的静态方法，它接受一个字符串参数 `name`。如果 `name` 为空，它从 `Poker.Resources` 命名空间中获取名为 `.txt` 的资源文件，并返回该文件的字节流。如果 `name` 是一个字符串，它使用 `Assembly.GetExecutingAssembly()` 获取正在运行的应用程序的 `ManifestResourceStream` 对象，并从该对象中获取名为 `{name}.txt` 的资源文件的字节流。

如果调用 `GetStream` 方法时，传入了不存在的资源文件名称，它将抛出一个 `ArgumentException` 异常。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Poker.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Instructions => GetStream();
        public static Stream Title => GetStream();
    }

    private static Stream GetStream([CallerMemberName] string? name = null)
        => Assembly.GetExecutingAssembly().GetManifestResourceStream($"Poker.Resources.{name}.txt")
            ?? throw new ArgumentException($"Resource stream {name} does not exist", nameof(name));
}
```

# `71_Poker/csharp/Strategies/Bet.cs`

这段代码定义了一个名为 "Bet" 的内部类，属于 "Strategies" 命名空间。这个内部类实现了 "Strategy" 接口，意味着它遵循了 "Strategy" 接口中定义的规则。

在 "Bet" 类中，最重要的是 "public Bet(int amount)" 方法，它接收一个整数参数 "amount"，并将其赋值给 "Value" 变量。

在 "public override int Value { get; }" 方法中，返回了 "Value" 变量的引用，以便在外部类中直接使用。

因此，这段代码创建了一个名为 "Bet" 的内部类，它继承自 "Strategy" 接口，具有自己的值类型。这个内部类可以被用来定义一个Bet 实例，并可以设置Bet 实例的金额，从而使用这个Bet 实例来做出策略决策。


```
namespace Poker.Strategies;

internal class Bet : Strategy
{
    public Bet(int amount) => Value = amount;

    public override int Value { get; }
}

```

# `71_Poker/csharp/Strategies/Bluff.cs`

这段代码定义了一个名为Bluff的内部类，继承自名为Bet的类。

这个内部类实现了Bet类中的一个名为Blit的method，这个method接收两个整数参数，第一个参数表示要下注的筹码数量，第二个参数是一个布尔类型的变量，表示在Bet类中KeepMask属性的值。

在Blit方法内部，首先定义了Bluff类的继承类所继承的Bet类中的一个名为Amount的常量，然后将KeepMask类型的变量KeepMask赋值给Amount，最后在Blit方法内部使用了一个try-catch块来确保在方法内部对Amount变量进行了正确的初始化。


```
namespace Poker.Strategies;

internal class Bluff : Bet
{
    public Bluff(int amount, int? keepMask)
        : base(amount)
    {
        KeepMask = keepMask;
    }

    public override int? KeepMask { get; }
}
```

# `71_Poker/csharp/Strategies/Check.cs`

这段代码定义了一个名为 "Check" 的策略类，属于 "Poker.Strategies" 命名空间。这个策略类实现了 "Strategy" 接口，其中包含一个 "Value" 成员变量，其值为 0。

具体来说，这个策略类的实现重写了 "Strategy" 接口中的 "Value" 成员函数，使其返回值为 0。这个策略类可以被用来创建一个 "Check" 类型的策略对象，这个对象将永远返回 0 的值。

这个代码示例展示了如何创建一个 Check 策略对象，它将永远返回 0 的值。这个策略类可以被用于编写任何需要返回固定值的游戏策略。


```
namespace Poker.Strategies;

internal class Check : Strategy
{
    public override int Value => 0;
}

```

# `71_Poker/csharp/Strategies/Fold.cs`



这段代码定义了一个名为 "Fold" 的内部类，它继承自 "Strategy" 类，代表扑克牌策略中的 "fold" 策略。

在 "Fold" 类中，定义了一个名为 "Value" 的内部变量，其值为 -1。

由于 "Fold" 类是 "Strategy" 类的实例化，因此可以类比 "Strategy" 类的实例化，通过 "Fold" 类的实例化来获取扑克牌策略中的 "fold" 策略的值。


```
namespace Poker.Strategies;

internal class Fold : Strategy
{
    public override int Value => -1;
}

```

# `71_Poker/csharp/Strategies/None.cs`

这段代码定义了一个名为"None"的内部类，它继承自名为"Strategy"的内部类，并重写了"Value"属性的值设为-1。

这个"None"类是一个策略类，它的目的是为了覆盖实现"Strategy"类中的"Value"属性，以便在不需要重写"Value"属性的情况下，提供一种特别的策略行为。

具体来说，如果实现了"Strategy"类中的"Value"属性，那么这个策略类的行为将默认为在所有情况下产生-1的输出。而通过将"Value"属性重写为-1，这个策略类将提供一种特定的行为，即在某些情况下产生1的输出。


```
namespace Poker.Strategies;

internal class None : Strategy
{
    public override int Value => -1;
}

```

# `71_Poker/csharp/Strategies/Raise.cs`



这段代码定义了一个名为"Raise"的类，其继承自名为"Bet"的类，代表一个下注策略。

在这个类中，有一个构造函数，用于初始化下注的筹码数，为2。

没有发现其他代码，因此可以推测这个类只是一个简单的下注策略实现，具体实现可能会根据需要进行扩展。


```
namespace Poker.Strategies;

internal class Raise : Bet
{
    public Raise() : base(2) { }
}

```

# `71_Poker/csharp/Strategies/Strategy.cs`

这段代码定义了一个名为`Poker.Strategies`的命名空间，其中包含一个抽象类`Strategy`。

这个`Strategy`类定义了五种不同的策略，分别是`None`、`Fold`、`Check`、`Raise`和`Bet`、`Bet`、`Bluff`。这些策略用于处理在扑克游戏中的不同决策，比如是否跟注、加注、看牌或下注等。

这个类的`Bet`方法可以接受任何形式的金额，并返回一个代表这个金额的整数。这个类的`Bluff`方法接受一个整数和一个可选的`keepMask`参数，用于模拟在游戏中的欺诈行为。

这个类的`Strategy`类是一个抽象类，它的`Value`和`KeepMask`属性是私有的，因此无法从外部类中继承。任何使用这个`Strategy`类的实例都必须自己定义策略的`Value`和`KeepMask`属性。


```
namespace Poker.Strategies;

internal abstract class Strategy
{
    public static Strategy None = new None();
    public static Strategy Fold = new Fold();
    public static Strategy Check = new Check();
    public static Strategy Raise = new Raise();
    public static Strategy Bet(float amount) => new Bet((int)amount);
    public static Strategy Bet(int amount) => new Bet(amount);
    public static Strategy Bluff(int amount, int? keepMask = null) => new Bluff(amount, keepMask);

    public abstract int Value { get; }
    public virtual int? KeepMask { get; }
}

```

# `71_Poker/java/Poker.java`

这段代码是一个Java程序，它实现了CREATIVE COMPUTING Poker游戏的简单版本。它的目的是让用户更好地理解这款游戏的玩法和算法。

程序首先导入了java.util.Random和java.util.Scanner类，用于生成随机数和输入用户输入。

然后，程序定义了一个从0到9的整数变量number，用于存储玩家选择的牌。接下来，程序使用Scanner类从用户那里获取牌的名称，并将其存储在变量hand中。

程序还定义了一个布尔变量isDeuces，用于判断当前是否为双倍筹码。

接下来，程序使用for循环，让用户从0到9选择一张牌，并将其存储在变量选择的牌中。程序还使用if语句检查选择的牌是否为双倍筹码。如果是，程序会跳转到标注/蜂窝函数内进行处理。

程序还定义了一个布尔变量isReserved，用于判断当前是否为保留牌。如果是，程序会跳转到标注/蜂窝函数内进行处理。

程序还定义了一个函数deal，用于处理发牌的过程。在deal函数中，程序会随机从hand中选择一张牌，并将其从hand中删除。然后，程序会检查当前是否为双倍筹码，如果是，程序会从hand中随机选择两张牌，并将其从hand中删除。

最后，程序使用Scanner类从用户那里获取是否继续游戏，如果用户选择继续，程序会继续循环处理，否则程序会结束。

总之，这段代码的目的是提供一个易于理解的CREATIVE COMPUTING Poker游戏的简单实现，其中包括游戏的核心算法和基本的游戏逻辑。


```
import java.util.Random;
import java.util.Scanner;

import static java.lang.System.out;

/**
 * Port of CREATIVE COMPUTING Poker written in Commodore 64 Basic to plain Java
 *
 * Original source scanned from magazine: https://www.atariarchives.org/basicgames/showpage.php?page=129
 *
 * I based my port on the OCR'ed source code here: https://github.com/coding-horror/basic-computer-games/blob/main/71_Poker/poker.bas
 *
 * Why? Because I remember typing this into my C64 when I was a tiny little developer and having great fun playing it!
 *
 * Goal: Keep the algorithms and UX more or less as-is; Improve the control flow a bit (no goto in Java!) and rename some stuff to be easier to follow.
 *
 * Result: There are probably bugs, please let me know.
 */
```

This is a program written in Python that simulates a game of blackjack. The game follows the rules of blackjack, where the player is dealt a hand of cards, and the player must decide whether to "hit" or "stand" until the dealer's hand is completed.

The player has several options to choose from when deciding their next move, such as "playerValuables" (a reference to their current score), "playerSellWatch" (a function that allows the player to sell one of their watches for money), or "playerSellTieTack" (a function that allows the player to sell their tie-tack).

The game also has a "playerBusted" function, which is called if the player decides to "stand" when their "playerSellTieTack" is called.

Overall, this program provides a good simulation of the game of blackjack.


```
public class Poker {

	public static void main(String[] args) {
		new Poker().run();
	}

	float[] cards = new float[50]; // Index 1-5 = Human hand, index 6-10 = Computer hand
	float[] B = new float[15];

	float playerValuables = 1;
	float computerMoney = 200;
	float humanMoney = 200;
	float pot = 0;

	String J$ = "";
	float computerHandValue = 0;

	int K = 0;
	float G = 0;
	float T = 0;
	int M = 0;
	int D = 0;

	int U = 0;
	float N = 1;

	float I = 0;

	float X = 0;

	int Z = 0;

	String handDescription = "";

	float V;

	void run() {
		printWelcome();
		playRound();
		startAgain();
	}

	void printWelcome() {
		tab(33);
		out.println("POKER");
		tab(15);
		out.print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		out.println();
		out.println();
		out.println();
		out.println("WELCOME TO THE CASINO.  WE EACH HAVE $200.");
		out.println("I WILL OPEN THE BETTING BEFORE THE DRAW; YOU OPEN AFTER.");
		out.println("TO FOLD BET 0; TO CHECK BET .5.");
		out.println("ENOUGH TALK -- LET'S GET DOWN TO BUSINESS.");
		out.println();
	}

	void tab(int number) {
		System.out.print("\t".repeat(number));
	}

	int random0to10() {
		return new Random().nextInt(10);
	}

	int removeHundreds(long x) {
		return _int(x - (100F * _int(x / 100F)));
	}

	void startAgain() {
		pot = 0;
		playRound();
	}

	void playRound() {
		if (computerMoney <= 5) {
			computerBroke();
		}

		out.println("THE ANTE IS $5.  I WILL DEAL:");
		out.println();

		if (humanMoney <= 5) {
			playerBroke();
		}

		pot = pot + 10;
		humanMoney = humanMoney - 5;
		computerMoney = computerMoney - 5;
		for (int Z = 1; Z < 10; Z++) {
			generateCards(Z);
		}
		out.println("YOUR HAND:");
		N = 1;
		showHand();
		N = 6;
		I = 2;

		describeHand();

		out.println();

		if (I != 6) {
			if (U >= 13) {
				if (U <= 16) {
					Z = 35;
				} else {
					Z = 2;
					if (random0to10() < 1) {
						Z = 35;
					}
				}
				computerOpens();
				playerMoves();
			} else if (random0to10() >= 2) {
				computerChecks();
			} else {
				I = 7;
				Z = 23;
				computerOpens();
				playerMoves();
			}
		} else if (random0to10() <= 7) {
			if (random0to10() <= 7) {
				if (random0to10() >= 1) {
					Z = 1;
					K = 0;
					out.print("I CHECK. ");
					playerMoves();
				} else {
					X = 11111;
					I = 7;
					Z = 23;
					computerOpens();
					playerMoves();
				}
			} else {
				X = 11110;
				I = 7;
				Z = 23;
				computerOpens();
				playerMoves();
			}
		} else {
			X = 11100;
			I = 7;
			Z = 23;
			computerOpens();
			playerMoves();
		}
	}

	void playerMoves() {
		playersTurn();
		checkWinnerAfterFirstBet();
		promptPlayerDrawCards();
	}

	void computerOpens() {
		V = Z + random0to10();
		computerMoves();
		out.print("I'LL OPEN WITH $" + V);
		K = _int(V);
	}

	@SuppressWarnings("StatementWithEmptyBody")
	void computerMoves() {
		if (computerMoney - G - V >= 0) {
		} else if (G != 0) {
			if (computerMoney - G >= 0) {
				computerSees();
			} else {
				computerBroke();
			}
		} else {
			V = computerMoney;
		}
	}

	void promptPlayerDrawCards() {
		out.println();
		out.println("NOW WE DRAW -- HOW MANY CARDS DO YOU WANT");
		inputPlayerDrawCards();
	}

	void inputPlayerDrawCards() {
		T = Integer.parseInt(readString());
		if (T == 0) {
			computerDrawing();
		} else {
			Z = 10;
			if (T < 4) {
				playerDrawsCards();
			} else {
				out.println("YOU CAN'T DRAW MORE THAN THREE CARDS.");
				inputPlayerDrawCards();
			}
		}
	}

	// line # 980
	void computerDrawing() {
		Z = _int(10 + T);
		for (U = 6; U <= 10; U++) {
			if (_int((float) (X / Math.pow(10F, (U - 6F)))) == (10 * (_int((float) (X / Math.pow(10, (U - 5))))))) {
				drawNextCard();
			}
		}
		out.print("I AM TAKING " + _int(Z - 10 - T) + " CARD");
		if (Z == 11 + T) {
			out.println();
		} else {
			out.println("S");
		}

		N = 6;
		V = I;
		I = 1;
		describeHand();
		startPlayerBettingAndReaction();
	}

	void drawNextCard() {
		Z = Z + 1;
		drawCard();
	}

	@SuppressWarnings("StatementWithEmptyBody")
	void drawCard() {
		cards[Z] = 100 * new Random().nextInt(4) + new Random().nextInt(100);
		if (_int(cards[Z] / 100) > 3) {
			drawCard();
		} else if (cards[Z] - 100 * _int(cards[Z] / 100) > 12) {
			drawCard();
		} else if (Z == 1) {
		} else {
			for (K = 1; K <= Z - 1; K++) {
				if (cards[Z] == cards[K]) {
					drawCard();
				}
			}
			if (Z <= 10) {
			} else {
				N = cards[U];
				cards[U] = cards[Z];
				cards[Z] = N;
			}
		}
	}

	void playerDrawsCards() {
		out.println("WHAT ARE THEIR NUMBERS:");
		for (int Q = 1; Q <= T; Q++) {
			U = Integer.parseInt(readString());
			drawNextCard();
		}

		out.println("YOUR NEW HAND:");
		N = 1;
		showHand();
		computerDrawing();
	}

	void startPlayerBettingAndReaction() {
		computerHandValue = U;
		M = D;

		if (V != 7) {
			if (I != 6) {
				if (U >= 13) {
					if (U >= 16) {
						Z = 2;
						playerBetsAndComputerReacts();
					} else {
						Z = 19;
						if (random0to10() == 8) {
							Z = 11;
						}
						playerBetsAndComputerReacts();
					}
				} else {
					Z = 2;
					if (random0to10() == 6) {
						Z = 19;
					}
					playerBetsAndComputerReacts();
				}
			} else {
				Z = 1;
				playerBetsAndComputerReacts();
			}
		} else {
			Z = 28;
			playerBetsAndComputerReacts();
		}
	}

	void playerBetsAndComputerReacts() {
		K = 0;
		playersTurn();
		if (T != .5) {
			checkWinnerAfterFirstBetAndCompareHands();
		} else if (V == 7 || I != 6) {
			computerOpens();
			promptAndInputPlayerBet();
			checkWinnerAfterFirstBetAndCompareHands();
		} else {
			out.println("I'LL CHECK");
			compareHands();
		}
	}

	void checkWinnerAfterFirstBetAndCompareHands() {
		checkWinnerAfterFirstBet();
		compareHands();
	}

	void compareHands() {
		out.println("NOW WE COMPARE HANDS:");
		J$ = handDescription;
		out.println("MY HAND:");
		N = 6;
		showHand();
		N = 1;
		describeHand();
		out.print("YOU HAVE ");
		K = D;
		printHandDescriptionResult();
		handDescription = J$;
		K = M;
		out.print(" AND I HAVE ");
		printHandDescriptionResult();
		out.print(". ");
		if (computerHandValue > U) {
			computerWins();
		} else if (U > computerHandValue) {
			humanWins();
		} else if (handDescription.contains("A FLUS")) {
			someoneWinsWithFlush();
		} else if (removeHundreds(M) < removeHundreds(D)) {
			humanWins();
		} else if (removeHundreds(M) > removeHundreds(D)) {
			computerWins();
		} else {
			handIsDrawn();
		}
	}

	void printHandDescriptionResult() {
		out.print(handDescription);
		if (!handDescription.contains("A FLUS")) {
			K = removeHundreds(K);
			printCardValue();
			if (handDescription.contains("SCHMAL")) {
				out.print(" HIGH");
			} else if (!handDescription.contains("STRAIG")) {
				out.print("'S");
			} else {
				out.print(" HIGH");
			}
		} else {
			K = K / 100;
			printCardColor();
			out.println();
		}
	}

	void handIsDrawn() {
		out.print("THE HAND IS DRAWN.");
		out.print("ALL $" + pot + " REMAINS IN THE POT.");
		playRound();
	}

	void someoneWinsWithFlush() {
		if (removeHundreds(M) > removeHundreds(D)) {
			computerWins();
		} else if (removeHundreds(D) > removeHundreds(M)) {
			humanWins();
		} else {
			handIsDrawn();
		}
	}

	@SuppressWarnings("StatementWithEmptyBody")
	void checkWinnerAfterFirstBet() {
		if (I != 3) {
			if (I != 4) {
			} else {
				humanWins();
			}
		} else {
			out.println();
			computerWins();
		}
	}

	void computerWins() {
		out.print(". I WIN. ");
		computerMoney = computerMoney + pot;
		potStatusAndNextRoundPrompt();
	}

	void potStatusAndNextRoundPrompt() {
		out.println("NOW I HAVE $" + computerMoney + " AND YOU HAVE $" + humanMoney);
		out.print("DO YOU WISH TO CONTINUE");

		if (yesFromPrompt()) {
			startAgain();
		} else {
			System.exit(0);
		}
	}

	private boolean yesFromPrompt() {
		String h = readString();
		if (h != null) {
			if (h.toLowerCase().matches("y|yes|yep|affirmative|yay")) {
				return true;
			} else if (h.toLowerCase().matches("n|no|nope|fuck off|nay")) {
				return false;
			}
		}
		out.println("ANSWER YES OR NO, PLEASE.");
		return yesFromPrompt();
	}

	void computerChecks() {
		Z = 0;
		K = 0;
		out.print("I CHECK. ");
		playerMoves();
	}

	void humanWins() {
		out.println("YOU WIN.");
		humanMoney = humanMoney + pot;
		potStatusAndNextRoundPrompt();
	}

	// line # 1740
	void generateCards(int Z) {
		cards[Z] = (100 * new Random().nextInt(4)) + new Random().nextInt(100);
		if (_int(cards[Z] / 100) > 3) {
			generateCards(Z);
			return;
		}
		if (cards[Z] - 100 * (_int(cards[Z] / 100)) > 12) {
			generateCards(Z);
			return;
		}
		if (Z == 1) {return;}
		for (int K = 1; K <= Z - 1; K++) {// TO Z-1
			if (cards[Z] == cards[K]) {
				generateCards(Z);
				return;
			}
		}
		if (Z <= 10) {return;}
		float N = cards[U];
		cards[U] = cards[Z];
		cards[Z] = N;
	}

	// line # 1850
	void showHand() {
		for (int cardNumber = _int(N); cardNumber <= N + 4; cardNumber++) {
			out.print(cardNumber + "--  ");
			printCardValueAtIndex(cardNumber);
			out.print(" OF");
			printCardColorAtIndex(cardNumber);
			if (cardNumber / 2 == (cardNumber / 2)) {
				out.println();
			}
		}
	}

	// line # 1950
	void printCardValueAtIndex(int Z) {
		K = removeHundreds(_int(cards[Z]));
		printCardValue();
	}

	void printCardValue() {
		if (K == 9) {
			out.print("JACK");
		} else if (K == 10) {
			out.print("QUEEN");
		} else if (K == 11) {
			out.print("KING");
		} else if (K == 12) {
			out.print("ACE");
		} else if (K < 9) {
			out.print(K + 2);
		}
	}

	// line # 2070
	void printCardColorAtIndex(int Z) {
		K = _int(cards[Z] / 100);
		printCardColor();
	}

	void printCardColor() {
		if (K == 0) {
			out.print(" CLUBS");
		} else if (K == 1) {
			out.print(" DIAMONDS");
		} else if (K == 2) {
			out.print(" HEARTS");
		} else if (K == 3) {
			out.print(" SPADES");
		}
	}

	// line # 2170
	void describeHand() {
		U = 0;
		for (Z = _int(N); Z <= N + 4; Z++) {
			B[Z] = removeHundreds(_int(cards[Z]));
			if (Z == N + 4) {continue;}
			if (_int(cards[Z] / 100) != _int(cards[Z + 1] / 100)) {continue;}
			U = U + 1;
		}
		if (U != 4) {
			for (Z = _int(N); Z <= N + 3; Z++) {
				for (K = Z + 1; K <= N + 4; K++) {
					if (B[Z] <= B[K]) {continue;}
					X = cards[Z];
					cards[Z] = cards[K];
					B[Z] = B[K];
					cards[K] = X;
					B[K] = cards[K] - 100 * _int(cards[K] / 100);
				}
			}
			X = 0;
			for (Z = _int(N); Z <= N + 3; Z++) {
				if (B[Z] != B[Z + 1]) {continue;}
				X = (float) (X + 11 * Math.pow(10, (Z - N)));
				D = _int(cards[Z]);

				if (U >= 11) {
					if (U != 11) {
						if (U > 12) {
							if (B[Z] != B[Z - 1]) {
								fullHouse();
							} else {
								U = 17;
								handDescription = "FOUR ";
							}
						} else {
							fullHouse();
						}
					} else if (B[Z] != B[Z - 1]) {
						handDescription = "TWO PAIR, ";
						U = 12;
					} else {
						handDescription = "THREE ";
						U = 13;
					}
				} else {
					U = 11;
					handDescription = "A PAIR OF ";
				}
			}

			if (X != 0) {
				schmaltzHand();
			} else {
				if (B[_int(N)] + 3 == B[_int(N + 3)]) {
					X = 1111;
					U = 10;
				}
				if (B[_int(N + 1)] + 3 != B[_int(N + 4)]) {
					schmaltzHand();
				} else if (U != 10) {
					U = 10;
					X = 11110;
					schmaltzHand();
				} else {
					U = 14;
					handDescription = "STRAIGHT";
					X = 11111;
					D = _int(cards[_int(N + 4)]);
				}
			}
		} else {
			X = 11111;
			D = _int(cards[_int(N)]);
			handDescription = "A FLUSH IN";
			U = 15;
		}
	}

	void schmaltzHand() {
		if (U >= 10) {
			if (U != 10) {
				if (U > 12) {return;}
				if (removeHundreds(D) <= 6) {
					I = 6;
				}
			} else {
				if (I == 1) {
					I = 6;
				}
			}
		} else {
			D = _int(cards[_int(N + 4)]);
			handDescription = "SCHMALTZ, ";
			U = 9;
			X = 11000;
			I = 6;
		}
	}

	void fullHouse() {
		U = 16;
		handDescription = "FULL HOUSE, ";
	}

	void playersTurn() {
		G = 0;
		promptAndInputPlayerBet();
	}

	String readString() {
		Scanner sc = new Scanner(System.in);
		return sc.nextLine();
	}

	@SuppressWarnings("StatementWithEmptyBody")
	void promptAndInputPlayerBet() {
		out.println("WHAT IS YOUR BET");
		T = readFloat();
		if (T - _int(T) == 0) {
			processPlayerBet();
		} else if (K != 0) {
			playerBetInvalidAmount();
		} else if (G != 0) {
			playerBetInvalidAmount();
		} else if (T == .5) {
		} else {
			playerBetInvalidAmount();
		}
	}

	private float readFloat() {
		try {
			return Float.parseFloat(readString());
		} catch (Exception ex) {
			System.out.println("INVALID INPUT, PLEASE TYPE A FLOAT. ");
			return readFloat();
		}
	}

	void playerBetInvalidAmount() {
		out.println("NO SMALL CHANGE, PLEASE.");
		promptAndInputPlayerBet();
	}

	void processPlayerBet() {
		if (humanMoney - G - T >= 0) {
			humanCanAffordBet();
		} else {
			playerBroke();
			promptAndInputPlayerBet();
		}
	}

	void humanCanAffordBet() {
		if (T != 0) {
			if (G + T >= K) {
				processComputerMove();
			} else {
				out.println("IF YOU CAN'T SEE MY BET, THEN FOLD.");
				promptAndInputPlayerBet();
			}
		} else {
			I = 3;
			moveMoneyToPot();
		}
	}

	void processComputerMove() {
		G = G + T;
		if (G == K) {
			moveMoneyToPot();
		} else if (Z != 1) {
			if (G > 3 * Z) {
				computerRaisesOrSees();
			} else {
				computerRaises();
			}
		} else if (G > 5) {
			if (T <= 25) {
				computerRaisesOrSees();
			} else {
				computerFolds();
			}
		} else {
			V = 5;
			if (G > 3 * Z) {
				computerRaisesOrSees();
			} else {
				computerRaises();
			}
		}
	}

	void computerRaises() {
		V = G - K + random0to10();
		computerMoves();
		out.println("I'LL SEE YOU, AND RAISE YOU" + V);
		K = _int(G + V);
		promptAndInputPlayerBet();
	}

	void computerFolds() {
		I = 4;
		out.println("I FOLD.");
	}

	void computerRaisesOrSees() {
		if (Z == 2) {
			computerRaises();
		} else {
			computerSees();
		}
	}

	void computerSees() {
		out.println("I'LL SEE YOU.");
		K = _int(G);
		moveMoneyToPot();
	}

	void moveMoneyToPot() {
		humanMoney = humanMoney - G;
		computerMoney = computerMoney - K;
		pot = pot + G + K;
	}

	void computerBusted() {
		out.println("I'M BUSTED.  CONGRATULATIONS!");
		System.exit(0);
	}

	@SuppressWarnings("StatementWithEmptyBody")
	private void computerBroke() {
		if ((playerValuables / 2) == _int(playerValuables / 2) && playerBuyBackWatch()) {
		} else if (playerValuables / 3 == _int(playerValuables / 3) && playerBuyBackTieRack()) {
		} else {
			computerBusted();
		}
	}

	private int _int(float v) {
		return (int) Math.floor(v);
	}

	private boolean playerBuyBackWatch() {
		out.println("WOULD YOU LIKE TO BUY BACK YOUR WATCH FOR $50");
		if (yesFromPrompt()) {
			computerMoney = computerMoney + 50;
			playerValuables = playerValuables / 2;
			return true;
		} else {
			return false;
		}
	}

	private boolean playerBuyBackTieRack() {
		out.println("WOULD YOU LIKE TO BUY BACK YOUR TIE TACK FOR $50");
		if (yesFromPrompt()) {
			computerMoney = computerMoney + 50;
			playerValuables = playerValuables / 3;
			return true;
		} else {
			return false;
		}
	}

	// line # 3830
	@SuppressWarnings("StatementWithEmptyBody")
	void playerBroke() {
		out.println("YOU CAN'T BET WITH WHAT YOU HAVEN'T GOT.");
		if (playerValuables / 2 != _int(playerValuables / 2) && playerSellWatch()) {
		} else if (playerValuables / 3 != _int(playerValuables / 3) && playerSellTieTack()) {
		} else {
			playerBusted();
		}
	}

	private void playerBusted() {
		out.println("YOUR WAD IS SHOT. SO LONG, SUCKER!");
		System.exit(0);
	}

	private boolean playerSellWatch() {
		out.println("WOULD YOU LIKE TO SELL YOUR WATCH");
		if (yesFromPrompt()) {
			if (random0to10() < 7) {
				out.println("I'LL GIVE YOU $75 FOR IT.");
				humanMoney = humanMoney + 75;
			} else {
				out.println("THAT'S A PRETTY CRUMMY WATCH - I'LL GIVE YOU $25.");
				humanMoney = humanMoney + 25;
			}
			playerValuables = playerValuables * 2;
			return true;
		} else {
			return false;
		}
	}

	private boolean playerSellTieTack() {
		out.println("WILL YOU PART WITH THAT DIAMOND TIE TACK");

		if (yesFromPrompt()) {
			if (random0to10() < 6) {
				out.println("YOU ARE NOW $100 RICHER.");
				humanMoney = humanMoney + 100;
			} else {
				out.println("IT'S PASTE.  $25.");
				humanMoney = humanMoney + 25;
			}
			playerValuables = playerValuables * 3;
			return true;
		} else {
			return false;
		}
	}

}

```