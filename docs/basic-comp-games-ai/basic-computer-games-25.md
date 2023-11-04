# BasicComputerGames源码解析 25

# `16_Bug/csharp/Parts/ParentPart.cs`

这段代码定义了一个内部类 `ParentPart`，继承自 `Part` 类，用于在游戏中处理玩家操作和游戏资源。

`ParentPart` 类有两个构造函数，分别用于指定在添加部件时需要发送的上下文信息。第一个构造函数接受两个参数 `message` 和 `isPresent`，第二个构造函数 那么就必须慎重考虑 returnsType` 参数的实例。

`ParentPart` 类还有一个名为 `TryAdd` 的方法，用于添加部件。该方法接收两个参数 `part` 和 `message`，返回一个 `Message` 类型的变量 `message`。

`TryAdd` 方法的实现基于三个条件：

1. `part.GetType()` 是否与 `ParentPart` 类中的 `GetType()` 方法返回类型相同？如果是，说明此部件已经存在于游戏中，调用 `TryAdd` 方法时将返回已存在的 `message` 并返回。

2. 如果 `GetType()` 不同，尝试添加部件并将返回的 `message` 传递给调用者。

3. 如果 `part` 为空，调用 `ReportDoNotHave` 方法，并返回一个 `Message` 类型的变量 `message`。

`TryAddCore` 方法是 `TryAdd` 的派生类，用于在实际创建部件时执行实际的 `Add` 操作。该方法的实现与 `TryAdd` 方法类似，但使用 `IPart` 类型的 `part` 参数，以避免对 `ParentPart` 类的修改。

`ReportDoNotHave` 方法用于在玩家不存在的部件上返回一个特定的 `Message`。如果 `ReportDoNotHave` 方法返回 `false`，则说明部件已经存在于游戏中，调用 `TryAdd` 方法时将返回已存在的 `message`。


```
using BugGame.Resources;

namespace BugGame.Parts;

internal abstract class ParentPart : Part
{
    public ParentPart(Message addedMessage, Message duplicateMessage)
        : base(addedMessage, duplicateMessage)
    {
    }

    public bool TryAdd(IPart part, out Message message)
        => (part.GetType() == GetType(), IsPresent) switch
        {
            (true, _) => TryAdd(out message),
            (false, false) => ReportDoNotHave(out message),
            _ => TryAddCore(part, out message)
        };

    protected abstract bool TryAddCore(IPart part, out Message message);

    private bool ReportDoNotHave(out Message message)
    {
        message = Message.DoNotHaveA(this);
        return false;
    }
}

```

# `16_Bug/csharp/Parts/Part.cs`

这段代码定义了一个名为Part的内部类，继承自名为IPart的接口。这个Part类包含两个私有变量，分别是_addedMessage和_duplicateMessage，它们分别用于记录已经被添加过的消息和已经被复制过的消息。

在Part的构造函数中，参数被赋值为_addedMessage和_duplicateMessage，分别用于记录新增和复制时的消息。

Part类还包含一个IsComplete虚拟方法，用于判断该Part是否已经完成，如果所有消息都被添加并且添加顺序正确，则返回true。

另外，Part类还有一个IsPresent方法，用于获取该Part是否已经完成，如果IsComplete方法的返回值为true，则IsPresent方法直接返回true。

最后，Part类还包括一个TryAdd方法，用于尝试将一个消息添加到Part中，如果Part已经完成，则返回false，否则返回true。


```
using BugGame.Resources;

namespace BugGame.Parts;

internal class Part : IPart
{
    private readonly Message _addedMessage;
    private readonly Message _duplicateMessage;

    public Part(Message addedMessage, Message duplicateMessage)
    {
        _addedMessage = addedMessage;
        _duplicateMessage = duplicateMessage;
    }

    public virtual bool IsComplete => IsPresent;

    protected bool IsPresent { get; private set; }

    public string Name => GetType().Name;

    public bool TryAdd(out Message message)
    {
        if (IsPresent)
        {
            message = _duplicateMessage;
            return false;
        }

        message = _addedMessage;
        IsPresent = true;
        return true;
    }
}

```

# `16_Bug/csharp/Parts/PartCollection.cs`



这段代码是一个名为 `PartCollection` 的类，用于在 `BugGame` 游戏引擎中管理零件。

该类有两个私有字段 `_maxCount` 和 `_count`，分别表示零件的最大数量和当前已添加的零件数量。

该类还有一个私有字段 `_addedMessage`，一个用于添加新零件的消息，以及一个私有字段 `_fullMessage`，用于在所有零件添加完成时显示的最终消息。

该类的构造函数接受三个参数，分别表示最大零件数量、添加新零件时需要显示的消息，以及添加新零件成功后需要显示的消息。

该类有一个名为 `IsComplete` 的字段，用于判断是否已经添加了所有零件。

该类有一个名为 `TryAddOne` 的方法，用于尝试添加一个新的零件并返回它。该方法接受一个 `Message` 类型的参数 `message`，用于在添加新零件成功后显示的消息。

该类还有一个名为 `AppendTo` 的方法，用于将零件添加到消息中。该方法接受一个 `StringBuilder` 类型的参数 `builder`，用于存储消息，以及一个整数 `offset`，用于指定从消息开始的位置和长度。

最后，该类定义了一个名为 `PartCollection` 的类，该类继承自 `Collections<Part>` 类，用于管理一组零件。


```
using System.Text;
using BugGame.Resources;

namespace BugGame.Parts;

internal class PartCollection
{
    private readonly int _maxCount;
    private readonly Message _addedMessage;
    private readonly Message _fullMessage;
    private int _count;

    public PartCollection(int maxCount, Message addedMessage, Message fullMessage)
    {
        _maxCount = maxCount;
        _addedMessage = addedMessage;
        _fullMessage = fullMessage;
    }

    public bool IsComplete => _count == _maxCount;

    public bool TryAddOne(out Message message)
    {
        if (_count < _maxCount)
        {
            _count++;
            message = _addedMessage.ForValue(_count);
            return true;
        }

        message = _fullMessage;
        return false;
    }

    protected void AppendTo(StringBuilder builder, int offset, int length, char character)
    {
        if (_count == 0) { return; }

        for (var i = 0; i < length; i++)
        {
            builder.Append(' ', offset);

            for (var j = 0; j < _count; j++)
            {
                builder.Append(character).Append(' ');
            }
            builder.AppendLine();
        }
    }
}

```

# `16_Bug/csharp/Parts/Tail.cs`



这段代码是一个名为“Tail”的类，它是“BugGame.Parts”命名空间中的一部分。该类继承自名为“Part”的类，可能还有其他类继承自“Part”。

该类的一个构造函数是“构造函数Tail()：base(Message.TailAdded, Message.TailNotNeeded)”，可能是用来设置该类的父类“Part”的一些默认设置，但具体设置的内容并不清楚。

该类还有一个名为“AppendTo(StringBuilder builder)”的方法，可能是用来将类的实例复制到“StringBuilder”对象中，但具体内容也不清楚。

最后，该类定义了一个名为“Tail”的类，但并没有定义任何方法。


```
using System.Text;
using BugGame.Resources;

namespace BugGame.Parts;

internal class Tail : Part
{
    public Tail()
        : base(Message.TailAdded, Message.TailNotNeeded)
    {
    }

    public void AppendTo(StringBuilder builder)
    {
        if (IsPresent)
        {
            builder.AppendLine("TTTTTB          B");
        }
    }
}
```

# `16_Bug/csharp/Resources/Message.cs`

这段代码定义了一个名为“Message”的类，该类用于在游戏中生成随机消息，用于提醒玩家在一些重要时刻，例如“获得了新的装备”或“完成了一个任务”。

该类包含多个消息类，每个消息类都有一个独特的参数列表和相应的解析字符串。这些消息类用于在玩家获得重要物品或完成任务时，生成不同的消息通知。

例如，如果你在游戏中获得了一个新的金币，游戏可能会生成一个包含“并获得了一枚金币”的消息。又如，如果你完成了一个任务，游戏可能会生成一个包含“你已经完成了任务”的消息。

该类还包含一个名为“BodyAdded”的消息类，用于告知玩家他们现在拥有了身体，可以进行更加复杂的操作。还包括一些其他的主题消息，例如“现在你有一个头”和“现在你有一个尾巴”，用于告知玩家他们现在拥有了新的部位。

总结起来，该代码是为了在游戏中生成各种不同类型的消息，以提醒玩家一些重要的信息。


```
using BugGame.Parts;

namespace BugGame.Resources;

internal class Message
{
    public static Message Rolled = new("rolled a {0}");

    public static Message BodyAdded = new("now have a body.");
    public static Message BodyNotNeeded = new("do not need a body.");

    public static Message NeckAdded = new("now have a neck.");
    public static Message NeckNotNeeded = new("do not need a neck.");

    public static Message HeadAdded = new("needed a head.");
    public static Message HeadNotNeeded = new("I do not need a head.", "You have a head.");

    public static Message TailAdded = new("I now have a tail.", "I now give you a tail.");
    public static Message TailNotNeeded = new("I do not need a tail.", "You already have a tail.");

    public static Message FeelerAdded = new("I get a feeler.", "I now give you a feeler");
    public static Message FeelersFull = new("I have 2 feelers already.", "You have two feelers already");

    public static Message LegAdded = new("now have {0} legs");
    public static Message LegsFull = new("I have 6 feet.", "You have 6 feet already");

    public static Message Complete = new("bug is finished.");

    private Message(string common)
        : this("I " + common, "You " + common)
    {
    }

    private Message(string i, string you)
    {
        I = i;
        You = you;
    }

    public string I { get; }
    public string You { get; }

    public static Message DoNotHaveA(Part part) => new($"do not have a {part.Name}");

    public Message ForValue(int quantity) => new(string.Format(I, quantity), string.Format(You, quantity));
}
```

# `16_Bug/csharp/Resources/Resource.cs`



这段代码是一个自定义的类，名为`Resource`，旨在在程序运行时加载资源文件。它从两个构造函数开始，一个是`GetStream`，另一个是`Stream.GetStream`，这两个方法都使用了`Assembly.GetExecutingAssembly()`方法来获取当前程序的执行集，然后使用这些方法获取资源文件的字节流。

具体来说，代码的作用是加载两个文本文件，分别是`Bug.Resources. introduction.txt`和`Bug.Resources.instructions.txt`，并将它们的内容存储在`Streams`类中的三个方法中，分别是`Introduction`,`Instructions`，和`PlayAgain`。

由于代码中使用了`[CallerMemberName]`标记，因此可以知道这些方法是作为`GetStream`方法的参数传递的，也就是说，当程序运行时，它将调用这些方法中的一个或多个，并按照定义的顺序依次执行。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace BugGame.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Introduction => GetStream();
        public static Stream Instructions => GetStream();
        public static Stream PlayAgain => GetStream();
    }

    private static Stream GetStream([CallerMemberName] string? name = null) =>
        Assembly.GetExecutingAssembly()
            .GetManifestResourceStream($"Bug.Resources.{name}.txt")
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `16_Bug/java/src/Bug.java`

This is a Java program that simulates a game of "Babylon the Beast" using the Swing library. The program uses the Keyboard to receive input from the player.

The program starts by printing instructions to the screen, then enters a loop where the player can interact with the game. The loop receives input from the Keyboard, converts it to lowercase, and then uses a loop to check if the entered text is for "N" or "NO" to a question. If the entered text is "N" or "NO", the program prints a message and then accepts the next input.

If the entered text is not "N" or "NO", the program displays a full screen message asking the player if they entered "N" or "NO" to a question. The program also generates a random number to add to the random input.

The program then exits the loop and the game starts a new round.


```
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Game of Bug
 * <p>
 * Based on the Basic game of Bug here
 * https://github.com/coding-horror/basic-computer-games/blob/main/16%20Bug/bug.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Bug {

    // Dice roll
    public static final int SIX = 6;

    private enum GAME_STATE {
        START,
        PLAYER_TURN,
        COMPUTER_TURN,
        CHECK_FOR_WINNER,
        GAME_OVER
    }

    // Used for keyboard input
    private final Scanner kbScanner;

    // Current game state
    private GAME_STATE gameState;


    private final Insect playersBug;

    private final Insect computersBug;

    // Used to show the result of dice roll.
    private final String[] ROLLS = new String[]{"BODY", "NECK", "HEAD", "FEELERS", "TAIL", "LEGS"};

    public Bug() {

        playersBug = new PlayerBug();
        computersBug = new ComputerBug();

        gameState = GAME_STATE.START;

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                // Show an introduction the first time the game is played.
                // And optionally instructions.
                case START:
                    intro();
                    if (!noEntered(displayTextAndGetInput("DO YOU WANT INSTRUCTIONS? "))) {
                        instructions();
                    }

                    gameState = GAME_STATE.PLAYER_TURN;
                    break;

                case PLAYER_TURN:
                    int playersRoll = randomNumber();
                    System.out.println("YOU ROLLED A " + playersRoll + "=" + ROLLS[playersRoll - 1]);
                    switch (playersRoll) {
                        case 1:
                            System.out.println(playersBug.addBody());
                            break;
                        case 2:
                            System.out.println(playersBug.addNeck());
                            break;
                        case 3:
                            System.out.println(playersBug.addHead());
                            break;
                        case 4:
                            System.out.println(playersBug.addFeelers());
                            break;
                        case 5:
                            System.out.println(playersBug.addTail());
                            break;
                        case 6:
                            System.out.println(playersBug.addLeg());
                            break;
                    }

                    gameState = GAME_STATE.COMPUTER_TURN;
                    break;

                case COMPUTER_TURN:
                    int computersRoll = randomNumber();
                    System.out.println("I ROLLED A " + computersRoll + "=" + ROLLS[computersRoll - 1]);
                    switch (computersRoll) {
                        case 1:
                            System.out.println(computersBug.addBody());
                            break;
                        case 2:
                            System.out.println(computersBug.addNeck());
                            break;
                        case 3:
                            System.out.println(computersBug.addHead());
                            break;
                        case 4:
                            System.out.println(computersBug.addFeelers());
                            break;
                        case 5:
                            System.out.println(computersBug.addTail());
                            break;
                        case 6:
                            System.out.println(computersBug.addLeg());
                            break;
                    }

                    gameState = GAME_STATE.CHECK_FOR_WINNER;
                    break;

                case CHECK_FOR_WINNER:
                    boolean gameOver = false;

                    if (playersBug.complete()) {
                        System.out.println("YOUR BUG IS FINISHED.");
                        gameOver = true;
                    } else if (computersBug.complete()) {
                        System.out.println("MY BUG IS FINISHED.");
                        gameOver = true;
                    }

                    if (noEntered(displayTextAndGetInput("DO YOU WANT THE PICTURES? "))) {
                        gameState = GAME_STATE.PLAYER_TURN;
                    } else {
                        System.out.println("*****YOUR BUG*****");
                        System.out.println();
                        draw(playersBug);

                        System.out.println();
                        System.out.println("*****MY BUG*****");
                        System.out.println();
                        draw(computersBug);
                        gameState = GAME_STATE.PLAYER_TURN;
                    }
                    if (gameOver) {
                        System.out.println("I HOPE YOU ENJOYED THE GAME, PLAY IT AGAIN SOON!!");
                        gameState = GAME_STATE.GAME_OVER;
                    }
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    /**
     * Draw the bug (player or computer) based on what has
     * already been added to it.
     *
     * @param bug The bug to be drawn.
     */
    private void draw(Insect bug) {
        ArrayList<String> insectOutput = bug.draw();
        for (String s : insectOutput) {
            System.out.println(s);
        }
    }

    /**
     * Display an intro
     */
    private void intro() {
        System.out.println("BUG");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THE GAME BUG");
        System.out.println("I HOPE YOU ENJOY THIS GAME.");
    }

    private void instructions() {
        System.out.println("THE OBJECT OF BUG IS TO FINISH YOUR BUG BEFORE I FINISH");
        System.out.println("MINE. EACH NUMBER STANDS FOR A PART OF THE BUG BODY.");
        System.out.println("I WILL ROLL THE DIE FOR YOU, TELL YOU WHAT I ROLLED FOR YOU");
        System.out.println("WHAT THE NUMBER STANDS FOR, AND IF YOU CAN GET THE PART.");
        System.out.println("IF YOU CAN GET THE PART I WILL GIVE IT TO YOU.");
        System.out.println("THE SAME WILL HAPPEN ON MY TURN.");
        System.out.println("IF THERE IS A CHANGE IN EITHER BUG I WILL GIVE YOU THE");
        System.out.println("OPTION OF SEEING THE PICTURES OF THE BUGS.");
        System.out.println("THE NUMBERS STAND FOR PARTS AS FOLLOWS:");
        System.out.println("NUMBER\tPART\tNUMBER OF PART NEEDED");
        System.out.println("1\tBODY\t1");
        System.out.println("2\tNECK\t1");
        System.out.println("3\tHEAD\t1");
        System.out.println("4\tFEELERS\t2");
        System.out.println("5\tTAIL\t1");
        System.out.println("6\tLEGS\t6");
        System.out.println();

    }

    /**
     * Checks whether player entered N or NO to a question.
     *
     * @param text player string from kb
     * @return true if N or NO was entered, otherwise false
     */
    private boolean noEntered(String text) {
        return stringIsAnyValue(text, "N", "NO");
    }

    /**
     * Check whether a string equals one of a variable number of values
     * Useful to check for Y or YES for example
     * Comparison is case insensitive.
     *
     * @param text   source string
     * @param values a range of values to compare against the source string
     * @return true if a comparison was found in one of the variable number of strings passed
     */
    private boolean stringIsAnyValue(String text, String... values) {

        // Cycle through the variable number of values and test each
        for (String val : values) {
            if (text.equalsIgnoreCase(val)) {
                return true;
            }
        }

        // no matches
        return false;
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * Generate random number
     *
     * @return random number
     */
    private int randomNumber() {
        return (int) (Math.random()
                * (SIX) + 1);
    }
}

```

# `16_Bug/java/src/BugGame.java`

这段代码定义了一个名为`BugGame`的类，其`main`方法是程序的入口点。在`main`方法中，创建了一个名为`Bug`的实例对象，并调用其`play`方法。

由于没有提供具体的`Bug`类，因此无法了解`Bug`类的行为。但通常情况下，`Bug`类可能是一个游戏中的“虫子”，或者是一个模拟程序中的“漏洞”。`play`方法则可能是用于让`Bug`类做出某种行为的尝试。

显然，以上解释只是一种玩笑式的假设，实际情况可能会与此大相径庭。


```


public class BugGame {

    public static void main(String[] args) {

        Bug bug = new Bug();
        bug.play();
    }
}

```

# `16_Bug/java/src/ComputerBug.java`

这段代码定义了一个名为`ComputerBug`的类，继承自`Insect`类。在这个类中，我们创建了一些与电脑玩家特有的消息，包括计算机玩家可以使用的信息，例如“我得到了一个法师”、“我已经有 < 最大感觉数 > 个感觉”、“我需要一个头部”、“我有一个头部”、“我现在有了一个身体”、“我有一个尾巴”、“我现在有了 < 最大身体数 > 个身体”。

在这个类的构造函数中，我们调用了父类的构造函数来进行初始化。在构造函数中，我们使用`super()`来调用父类的构造函数。接着，我们使用`addMessages()`方法来添加这些消息。`addMessages()`方法接收一个字符串数组和一段参数，第一个字符串数组包含与当前消息相关的信息，第二个参数包含用于格式化消息信息的字符数组。在`ComputerBug`类中，我们将这些参数`{String[]{"I GET A FEELER.", "I HAVE " + MAX_FEELERS + " FEELERS ALREADY.", "I DO NOT HAVE A HEAD."}, PARTS.FEELERS}`和`{String[]{"I NEEDED A HEAD.", "I DO NOT NEED A HEAD.", "I DO NOT HAVE A NECK."}, PARTS.HEAD}`和`{String[]{"I NOW HAVE A NECK.", "I DO NOT NEED A NECK.", "I DO NOT HAVE A BODY."}, PARTS.BODY}`和`{String[]{"I NOW HAVE A TAIL.", "I DO NOT NEED A TAIL.", "I DO NOT HAVE A BODY."}, PARTS.TAIL}`和`{String[]{"I NOW HAVE ^^^" + " LEG", "I HAVE " + MAX_LEGS + " FEET.", "I DO NOT HAVE A BODY."}, PARTS.LEGS}`分别将这些信息添加到了消息中。

最后，我们在`ComputerBug`类的体中进行了注释，以使我们不知道该做什么以及警告隐藏了潜在的编译错误。


```
public class ComputerBug extends Insect {

    // Create messages specific to the computer player.

    public ComputerBug() {
        // Call superclass constructor for initialization.
        super();
        addMessages(new String[]{"I GET A FEELER.", "I HAVE " + MAX_FEELERS + " FEELERS ALREADY.", "I DO NOT HAVE A HEAD."}, PARTS.FEELERS);
        addMessages(new String[]{"I NEEDED A HEAD.", "I DO NOT NEED A HEAD.", "I DO NOT HAVE A NECK."}, PARTS.HEAD);
        addMessages(new String[]{"I NOW HAVE A NECK.", "I DO NOT NEED A NECK.", "I DO NOT HAVE A BODY."}, PARTS.NECK);
        addMessages(new String[]{"I NOW HAVE A BODY.", "I DO NOT NEED A BODY."}, PARTS.BODY);
        addMessages(new String[]{"I NOW HAVE A TAIL.", "I DO NOT NEED A TAIL.", "I DO NOT HAVE A BODY."}, PARTS.TAIL);
        addMessages(new String[]{"I NOW HAVE ^^^" + " LEG", "I HAVE " + MAX_LEGS + " FEET.", "I DO NOT HAVE A BODY."}, PARTS.LEGS);
    }
}

```

# `16_Bug/java/src/Insect.java`

这段代码定义了一个 `Bug` 类，用于表示电脑中出现的兼容性问题。该类实现了三个方法：`createBug`,`createLineBug`，和 `createBugWithBody`.

- `createBug` 方法接受一个字符串参数，表示电脑中出现的兼容性问题。该方法将调用 `createLineBug` 和 `createBugWithBody` 方法来检查该问题是否已经被创建过。如果已经创建过，该方法将返回之前创建的问题。如果还没有创建过，该方法将创建一个新的问题，并将它返回。

- `createLineBug` 方法接受一个字符串参数，表示电脑中出现的兼容性问题。与 `createBug` 方法不同的是，该方法将返回一个已经创建过的问题。如果已经创建过，该方法将直接返回该问题。否则，它将创建一个新的问题，并将它返回。

- `createBugWithBody` 方法与 `createBug` 类似，只是它将返回一个已经创建过的问题。如果已经创建过，该方法将直接返回该问题。否则，它将创建一个新的问题，并将它返回。

该类还实现了两个附加方法：`getBug` 和 `getLegs`，用于获取问题的信息和提问题目的机器人腿的数量。

- `getBug` 方法从 `Bug` 对象中获取问题的信息，并返回问题的字符串表示形式。

- `getLegs` 方法从 `Bug` 对象中获取问题的信息和机器人腿的数量，并返回问题的字符串表示形式。


```
import java.util.ArrayList;
import java.util.Arrays;

/**
 * This tracks the insect (bug) and has methods to
 * add body parts, create an array of output so it
 * can be drawn and to determine if a bug is complete.
 * N.B. This is a super class for ComputerBug and PlayerBug
 */
public class Insect {

    public static final int MAX_FEELERS = 2;
    public static final int MAX_LEGS = 6;

    public static final int ADDED = 0;
    public static final int NOT_ADDED = 1;
    public static final int MISSING = 2;

    // Various parts of the bug
    public enum PARTS {
        FEELERS,
        HEAD,
        NECK,
        BODY,
        TAIL,
        LEGS
    }

    // Tracks what parts of the bug have been added
    private boolean body;
    private boolean neck;
    private boolean head;
    private int feelers;
    private boolean tail;
    private int legs;

    // Messages about for various body parts
    // These are set in the subclass ComputerBug or PlayerBug
    private String[] bodyMessages;
    private String[] neckMessages;
    private String[] headMessages;
    private String[] feelerMessages;
    private String[] tailMessages;
    private String[] legMessages;

    public Insect() {
        init();
    }

    /**
     * Add a body to the bug if there is not one already added.
     *
     * @return return an appropriate message about the status of the operation.
     */
    public String addBody() {

        boolean currentState = false;

        if (!body) {
            body = true;
            currentState = true;
        }

        return addBodyMessage(currentState);
    }

    /**
     * Create output based on adding the body or it being already added previously
     *
     * @return contains the output message
     */

    private String addBodyMessage(boolean wasAdded) {

        // Return the appropriate message depending on whether the
        // body was added or not.
        if (wasAdded) {
            return bodyMessages[ADDED];
        } else {
            return bodyMessages[NOT_ADDED];
        }
    }

    /**
     * Add a neck if a) a body has previously been added and
     * b) a neck has not previously been added.
     *
     * @return text containing the status of the operation
     */
    public String addNeck() {

        int status = NOT_ADDED;  // Default is not added

        if (!body) {
            // No body, cannot add a neck
            status = MISSING;
        } else if (!neck) {
            neck = true;
            status = ADDED;
        }

        return neckMessages[status];
    }

    /**
     * Add a head to the bug if a) there already exists a neck and
     * b) a head has not previously been added
     *
     * @return text outlining the success of the operation
     */
    public String addHead() {

        int status = NOT_ADDED;  // Default is not added

        if (!neck) {
            // No neck, cannot add a head
            status = MISSING;
        } else if (!head) {
            head = true;
            status = ADDED;
        }

        return headMessages[status];
    }

    /**
     * Add a feeler to the head if a) there has been a head added to
     * the bug previously, and b) there are not already 2 (MAX_FEELERS)
     * feelers previously added to the bug.
     *
     * @return text outlining the status of the operation
     */
    public String addFeelers() {

        int status = NOT_ADDED;  // Default is not added

        if (!head) {
            // No head, cannot add a feeler
            status = MISSING;
        } else if (feelers < MAX_FEELERS) {
            feelers++;
            status = ADDED;
        }

        return feelerMessages[status];
    }

    /**
     * Add a tail to the bug if a) there is already a body previously added
     * to the bug and b) there is not already a tail added.
     *
     * @return text outlining the status of the operation.
     */
    public String addTail() {

        int status = NOT_ADDED;  // Default is not added

        if (!body) {
            // No body, cannot add a tail
            status = MISSING;
        } else if (!tail) {
            tail = true;
            status = ADDED;
        }

        return tailMessages[status];
    }

    /**
     * Add a leg to the bug if a) there is already a body previously added
     * b) there are less than 6 (MAX_LEGS) previously added.
     *
     * @return text outlining status of the operation.
     */
    public String addLeg() {

        int status = NOT_ADDED;  // Default is not added

        if (!body) {
            // No body, cannot add a leg
            status = MISSING;
        } else if (legs < MAX_LEGS) {
            legs++;
            status = ADDED;
        }

        String message = "";

        // Create a string showing the result of the operation

        switch(status) {
            case ADDED:
                // Replace # with number of legs
                message = legMessages[status].replace("^^^", String.valueOf(legs));
                // Add text S. if >1 leg, or just . if one leg.
                if (legs > 1) {
                    message += "S.";
                } else {
                    message += ".";
                }
                break;

            case NOT_ADDED:

                // Deliberate fall through to next case as its the
                // same code to be executed
            case MISSING:
                message = legMessages[status];
                break;
        }

        return message;
    }

    /**
     * Initialise
     */
    public void init() {
        body = false;
        neck = false;
        head = false;
        feelers = 0;
        tail = false;
        legs = 0;
    }

    /**
     * Add unique messages depending on type of player
     * A subclass of this class calls this method
     * e.g. See ComputerBug or PlayerBug classes
     *
     * @param messages an array of messages
     * @param bodyPart the bodypart the messages relate to.
     */
    public void addMessages(String[] messages, PARTS bodyPart) {

        switch (bodyPart) {
            case FEELERS:
                feelerMessages = messages;
                break;

            case HEAD:
                headMessages = messages;
                break;

            case NECK:
                neckMessages = messages;
                break;

            case BODY:
                bodyMessages = messages;
                break;

            case TAIL:
                tailMessages = messages;
                break;

            case LEGS:
                legMessages = messages;
                break;
        }
    }

    /**
     * Returns a string array containing
     * the "bug" that can be output to console
     *
     * @return the bug ready to draw
     */
    public ArrayList<String> draw() {
        ArrayList<String> bug = new ArrayList<>();
        StringBuilder lineOutput;

        // Feelers
        if (feelers > 0) {
            for (int i = 0; i < 4; i++) {
                lineOutput = new StringBuilder(addSpaces(10));
                for (int j = 0; j < feelers; j++) {
                    lineOutput.append("A ");
                }
                bug.add(lineOutput.toString());
            }
        }

        if (head) {
            lineOutput = new StringBuilder(addSpaces(8) + "HHHHHHH");
            bug.add(lineOutput.toString());
            lineOutput = new StringBuilder(addSpaces(8) + "H" + addSpaces(5) + "H");
            bug.add(lineOutput.toString());
            lineOutput = new StringBuilder(addSpaces(8) + "H O O H");
            bug.add(lineOutput.toString());
            lineOutput = new StringBuilder(addSpaces(8) + "H" + addSpaces(5) + "H");
            bug.add(lineOutput.toString());
            lineOutput = new StringBuilder(addSpaces(8) + "H" + addSpaces(2) + "V" + addSpaces(2) + "H");
            bug.add(lineOutput.toString());
            lineOutput = new StringBuilder(addSpaces(8) + "HHHHHHH");
            bug.add(lineOutput.toString());
        }

        if (neck) {
            for (int i = 0; i < 2; i++) {
                lineOutput = new StringBuilder(addSpaces(10) + "N N");
                bug.add(lineOutput.toString());
            }
        }

        if (body) {
            lineOutput = new StringBuilder(addSpaces(5) + "BBBBBBBBBBBB");
            bug.add(lineOutput.toString());
            for (int i = 0; i < 2; i++) {
                lineOutput = new StringBuilder(addSpaces(5) + "B" + addSpaces(10) + "B");
                bug.add(lineOutput.toString());
            }
            if (tail) {
                lineOutput = new StringBuilder("TTTTTB" + addSpaces(10) + "B");
                bug.add(lineOutput.toString());
            }
            lineOutput = new StringBuilder(addSpaces(5) + "BBBBBBBBBBBB");
            bug.add(lineOutput.toString());
        }

        if (legs > 0) {
            for (int i = 0; i < 2; i++) {
                lineOutput = new StringBuilder(addSpaces(5));
                for (int j = 0; j < legs; j++) {
                    lineOutput.append(" L");
                }
                bug.add(lineOutput.toString());
            }
        }

        return bug;
    }

    /**
     * Check if the bug is complete i.e. it has
     * 2 (MAX_FEELERS) feelers, a head, a neck, a body
     * a tail and 6 (MAX_FEET) feet.
     *
     * @return true if complete.
     */
    public boolean complete() {
        return (feelers == MAX_FEELERS)
                && head
                && neck
                && body
                && tail
                && (legs == MAX_LEGS);
    }

    /**
     * Simulate tabs be creating a string of X spaces.
     *
     * @param number contains number of spaces needed.
     * @return a String containing the spaces
     */
    private String addSpaces(int number) {
        char[] spaces = new char[number];
        Arrays.fill(spaces, ' ');
        return new String(spaces);

    }
}

```

# `16_Bug/java/src/PlayerBug.java`

这段代码定义了一个名为PlayerBug的类，继承自Insect类。在PlayerBug类中，创建了一些消息，针对玩家，例如，“我现在给你一个感觉”,“你已经拥有 ”+MAX_FEELERS+“感觉”,“你不需要一个头”。接着，分别添加了身体，颈部，腿部消息。这些消息将被打印在控制台。 

PlayerBug类中的构造函数会先调用父类的构造函数进行初始化，然后在初始化过程中添加这些消息。 在此之后，这些消息将打印到控制台，以便开发人员能够看到玩家的消息，开发人员可以通过注入PlayerBug对象来更改这些消息。 

例如，可以在玩家对象中注入以下内容： 

```
PlayerBug player = new PlayerBug();
player.send("I NOW GIVE YOU A FEELER");
player.send("YOU HAVE " + MAX_FEELERS + " FEELERS ALREADY.");
player.send("YOU DO NOT HAVE A HEAD.");
``` 

这段代码将打印三个消息到控制台： 

```
I NOW GIVE YOU A FEELER
YOU HAVE 12 FEELERS ALREADY.
YOU DO NOT HAVE A HEAD.
``` 

其中，第一个消息 "I NOW GIVE YOU A FEELER" 是从MAX_FEELERS中随机选择的，用于表示玩家拥有多少个感觉。第二个消息 "YOU HAVE " + MAX_FEELERS + " FEELERS ALREADY." 是告诉玩家他们已经拥有多少个感觉，MAX_FEELERS是消息的参数。第三个消息 "YOU DO NOT HAVE A HEAD." 是告诉玩家他们是否需要头部，此处的头部指的是MAX_HEADS中的最大头部数量。


```
public class PlayerBug extends Insect {

    // Create messages specific to the player.

    public PlayerBug() {
        // Call superclass constructor for initialization.
        super();
        addMessages(new String[]{"I NOW GIVE YOU A FEELER.", "YOU HAVE " + MAX_FEELERS + " FEELERS ALREADY.", "YOU DO NOT HAVE A HEAD."}, PARTS.FEELERS);
        addMessages(new String[]{"YOU NEEDED A HEAD.", "YOU HAVE A HEAD.", "YOU DO NOT HAVE A NECK."}, PARTS.HEAD);
        addMessages(new String[]{"YOU NOW HAVE A NECK.", "YOU DO NOT NEED A NECK.", "YOU DO NOT HAVE A BODY."}, PARTS.NECK);
        addMessages(new String[]{"YOU NOW HAVE A BODY.", "YOU DO NOT NEED A BODY."}, PARTS.BODY);
        addMessages(new String[]{"I NOW GIVE YOU A TAIL.", "YOU ALREADY HAVE A TAIL.", "YOU DO NOT HAVE A BODY."}, PARTS.TAIL);
        addMessages(new String[]{"YOU NOW HAVE ^^^ LEG", "YOU HAVE " + MAX_LEGS + " FEET ALREADY.", "YOU DO NOT HAVE A BODY."}, PARTS.LEGS);
    }


}

```

# `16_Bug/javascript/bug.js`

这段代码存在两个主要的功能：

1. `print()` 函数用于将文本内容输出到网页上的一个元素。它接受一个字符串参数，并将其作为文本节点添加到文档的 `output` 元素中。
2. `input()` 函数接受一个Promise对象，并在其内部创建一个输入元素。它使用户可以输入字符，并在输入框中 Focus（获得焦点）并添加一个事件监听器，以便在用户按下键盘上的Enter键时接收输入。当用户输入字符时，监听器会获取其类型并记录在文化自信中。然后，使用 Promise 对象解决存储用户输入并将其附加到网页上的元素。


```
// BUG
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    return new Promise(function (resolve) {
                       const input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_element.addEventListener("keydown",
                           function (event) {
                                      if (event.keyCode === 13) {
                                          const input_str = input_element.value;
                                          document.getElementById("output").removeChild(input_element);
                                          print(input_str);
                                          print("\n");
                                          resolve(input_str);
                                      }
                                  });
                       });
}

```

这段代码定义了三个函数，各自具有不同的作用：

1. `tab` 函数：该函数的作用是接收一个字符串变量 `space`，然后将其中的所有空格去掉并返回处理后的字符串。

2. `waitNSeconds` 函数：该函数的作用是接收一个参数 `n`，然后 returns一个 Promise 对象，其中 `n` 毫秒等于 `n` 微秒。函数内部使用 `setTimeout` 函数来等待 `n` 毫秒，并在超时时调用 `resolve` 函数来返回处理后的结果。

3. `scrollToBottom` 函数：该函数的作用是获取当前页面的滚动高度，并将其设置为 `document.body.scrollHeight` 的值，将页面内容滚动到底部。


```
function tab(space)
{
    let str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

function waitNSeconds(n) {
    return new Promise(resolve => setTimeout(resolve, n*1000));
}

function scrollToBottom() {
    window.scrollTo(0, document.body.scrollHeight);
}

```

这两段代码定义了两个函数，分别是 `draw_head()` 和 `drawFeelers()`。

1. `draw_head()` 函数的作用是在屏幕上打印出 HHHHHHH 这个字符串，然后打印出 H 和 OOOH 这两个字符，接着打印出 H 和 OO 这两个字符，最后打印出 H 和 VVH 这两个字符，并输出屏幕。

2. `drawFeelers(feelerCount, character)` 函数的作用是在屏幕上打印出指定的字符 `character`，并且根据传入的 `feelerCount` 参数打印出指定的字符 `character` 重复指定的次数。该函数还会打印出一些空格来使字符更加明显。


```
function draw_head()
{
    print("        HHHHHHH\n");
    print("        H     H\n");
    print("        H O O H\n");
    print("        H     H\n");
    print("        H  V  H\n");
    print("        HHHHHHH\n");
}

function drawFeelers(feelerCount, character) {
    for (let z = 1; z <= 4; z++) {
        print(tab(10));
        for (let x = 1; x <= feelerCount; x++) {
            print(character + " ");
        }
        print("\n");
    }
}

```



这三段代码都定义了不同的函数，分别用于打印不同部位的身体结构。

1. drawNeck()函数用于打印出头部的字符，从1到2，共打印了4个字符。

2. drawBody()函数打印出一个八分音符，包含7个字符。然后，打印出一个句号，包含3个字符。接下来，打印出两个大写字母B，共打印了6个字符。最后，根据计算机会的尾部数量，打印一个句号，包含3个字符。总共打印了21个字符。

3. drawFeet()函数打印出两个大写字母B，共打印了6个字符。接着，打印出两个下划线，共打印了4个字符。然后，打印出两个句号，共打印了4个字符。接下来，打印出两个反斜杠，共打印了4个字符。最后，打印出两个空格，共打印了4个字符。总共打印了20个字符。


```
function drawNeck() {
    for (let z = 1; z <= 2; z++)
        print("          N N\n");
}

function drawBody(computerTailCount) {
    print("     BBBBBBBBBBBB\n");
    for (let z = 1; z <= 2; z++)
        print("     B          B\n");
    if (computerTailCount === 1)
        print("TTTTTB          B\n");
    print("     BBBBBBBBBBBB\n");
}

function drawFeet(computerFeetCount) {
    for (let z = 1; z <= 2; z++) {
        print(tab(5));
        for (let x = 1; x <= computerFeetCount; x++)
            print(" L");
        print("\n");
    }
}

```



这是一个JavaScript函数，作用是绘制游戏中玩家的身体部位，并添加Feelies(感觉我的东西)特效。

具体来说，该函数会根据玩家身体部位的数量来决定要绘制多少个Feelies。然后，分别绘制感觉杆、头部、颈部、身体和 feet。最后，为了让游戏的画面看起来更生动，每经过一个身体部位时，函数会在控制台输出一个换行符。

感觉我的东西(Feelies)是一种特效，用于模拟玩家在游戏中的动作。当玩家移动时，会看到一个绿色的箭头在移动。


```
function drawBug(playerFeelerCount, playerHeadCount, playerNeckCount, playerBodyCount, playerTailCount, playerFeetCount, feelerCharacter) {
    if (playerFeelerCount !== 0) {
        drawFeelers(playerFeelerCount, feelerCharacter);
    }
    if (playerHeadCount !== 0)
        draw_head();
    if (playerNeckCount !== 0) {
        drawNeck();
    }
    if (playerBodyCount !== 0) {
        drawBody(playerTailCount)
    }
    if (playerFeetCount !== 0) {
        drawFeet(playerFeetCount);
    }
    for (let z = 1; z <= 4; z++)
        print("\n");
}

```

This appears to be a game where the player and the computer are trying to find a way to complete a task. The player has to navigate through a series of levels, while the computer has to navigate through a series of levels to find a way to complete a task.

The levels seem to be structured in a grid-like fashion, with the player having to navigate through four different environments - grassy field, woodland, desert, and snowy field. Each environment has a series of obstacles that the player must navigate through in order to reach the other side.

The player has to find a way to collect a series of items, while the computer has to find a way to collect a series of items and place them in the correct location. The player has to navigate through a series of puzzles, while the computer has to navigate through a series of puzzles in order to complete their task.

Overall, it appears to be a simple game with a few different environments and a series of puzzles that the player must navigate through in order to complete their task.


```
// Main program
async function main()
{
    print(tab(34) + "BUG\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    let playerFeelerCount = 0;
    let playerHeadCount = 0;
    let playerNeckCount = 0;
    let playerBodyCount = 0;
    let playerFeetCount = 0;
    let playerTailCount = 0;

    let computerFeelerCount = 0;
    let computerHeadCount = 0;
    let computerNeckCount = 0;
    let computerBodyCount = 0;
    let computerTailCount = 0;
    let computerFeetCount = 0;

    print("THE GAME BUG\n");
    print("I HOPE YOU ENJOY THIS GAME.\n");
    print("\n");
    print("DO YOU WANT INSTRUCTIONS");
    const instructionsRequired = await input();
    if (instructionsRequired.toUpperCase() !== "NO") {
        print("THE OBJECT OF BUG IS TO FINISH YOUR BUG BEFORE I FINISH\n");
        print("MINE. EACH NUMBER STANDS FOR A PART OF THE BUG BODY.\n");
        print("I WILL ROLL THE DIE FOR YOU, TELL YOU WHAT I ROLLED FOR YOU\n");
        print("WHAT THE NUMBER STANDS FOR, AND IF YOU CAN GET THE PART.\n");
        print("IF YOU CAN GET THE PART I WILL GIVE IT TO YOU.\n");
        print("THE SAME WILL HAPPEN ON MY TURN.\n");
        print("IF THERE IS A CHANGE IN EITHER BUG I WILL GIVE YOU THE\n");
        print("OPTION OF SEEING THE PICTURES OF THE BUGS.\n");
        print("THE NUMBERS STAND FOR PARTS AS FOLLOWS:\n");
        print("NUMBER\tPART\tNUMBER OF PART NEEDED\n");
        print("1\tBODY\t1\n");
        print("2\tNECK\t1\n");
        print("3\tHEAD\t1\n");
        print("4\tFEELERS\t2\n");
        print("5\tTAIL\t1\n");
        print("6\tLEGS\t6\n");
        print("\n");
        print("\n");
    }

    let gameInProgress = true;
    while (gameInProgress) {
        let dieRoll = Math.floor(6 * Math.random() + 1);
        let partFound = false;
        print("YOU ROLLED A " + dieRoll + "\n");
        switch (dieRoll) {
            case 1:
                print("1=BODY\n");
                if (playerBodyCount === 0) {
                    print("YOU NOW HAVE A BODY.\n");
                    playerBodyCount = 1;
                    partFound = true;
                } else {
                    print("YOU DO NOT NEED A BODY.\n");
                }
                break;
            case 2:
                print("2=NECK\n");
                if (playerNeckCount === 0) {
                    if (playerBodyCount === 0) {
                        print("YOU DO NOT HAVE A BODY.\n");
                    } else {
                        print("YOU NOW HAVE A NECK.\n");
                        playerNeckCount = 1;
                        partFound = true;
                    }
                } else {
                    print("YOU DO NOT NEED A NECK.\n");
                }
                break;
            case 3:
                print("3=HEAD\n");
                if (playerNeckCount === 0) {
                    print("YOU DO NOT HAVE A NECK.\n");
                } else if (playerHeadCount === 0) {
                    print("YOU NEEDED A HEAD.\n");
                    playerHeadCount = 1;
                    partFound = true;
                } else {
                    print("YOU HAVE A HEAD.\n");
                }
                break;
            case 4:
                print("4=FEELERS\n");
                if (playerHeadCount === 0) {
                    print("YOU DO NOT HAVE A HEAD.\n");
                } else if (playerFeelerCount === 2) {
                    print("YOU HAVE TWO FEELERS ALREADY.\n");
                } else {
                    print("I NOW GIVE YOU A FEELER.\n");
                    playerFeelerCount ++;
                    partFound = true;
                }
                break;
            case 5:
                print("5=TAIL\n");
                if (playerBodyCount === 0) {
                    print("YOU DO NOT HAVE A BODY.\n");
                } else if (playerTailCount === 1) {
                    print("YOU ALREADY HAVE A TAIL.\n");
                } else {
                    print("I NOW GIVE YOU A TAIL.\n");
                    playerTailCount++;
                    partFound = true;
                }
                break;
            case 6:
                print("6=LEG\n");
                if (playerFeetCount === 6) {
                    print("YOU HAVE 6 FEET ALREADY.\n");
                } else if (playerBodyCount === 0) {
                    print("YOU DO NOT HAVE A BODY.\n");
                } else {
                    playerFeetCount++;
                    partFound = true;
                    print("YOU NOW HAVE " + playerFeetCount + " LEGS.\n");
                }
                break;
        }
        dieRoll = Math.floor(6 * Math.random() + 1) ;
        print("\n");
        scrollToBottom();
        await waitNSeconds(1);

        print("I ROLLED A " + dieRoll + "\n");
        switch (dieRoll) {
            case 1:
                print("1=BODY\n");
                if (computerBodyCount === 1) {
                    print("I DO NOT NEED A BODY.\n");
                } else {
                    print("I NOW HAVE A BODY.\n");
                    partFound = true;
                    computerBodyCount = 1;
                }
                break;
            case 2:
                print("2=NECK\n");
                if (computerNeckCount === 1) {
                    print("I DO NOT NEED A NECK.\n");
                } else if (computerBodyCount === 0) {
                    print("I DO NOT HAVE A BODY.\n");
                } else {
                    print("I NOW HAVE A NECK.\n");
                    computerNeckCount = 1;
                    partFound = true;
                }
                break;
            case 3:
                print("3=HEAD\n");
                if (computerNeckCount === 0) {
                    print("I DO NOT HAVE A NECK.\n");
                } else if (computerHeadCount === 1) {
                    print("I DO NOT NEED A HEAD.\n");
                } else {
                    print("I NEEDED A HEAD.\n");
                    computerHeadCount = 1;
                    partFound = true;
                }
                break;
            case 4:
                print("4=FEELERS\n");
                if (computerHeadCount === 0) {
                    print("I DO NOT HAVE A HEAD.\n");
                } else if (computerFeelerCount === 2) {
                    print("I HAVE 2 FEELERS ALREADY.\n");
                } else {
                    print("I GET A FEELER.\n");
                    computerFeelerCount++;
                    partFound = true;
                }
                break;
            case 5:
                print("5=TAIL\n");
                if (computerBodyCount === 0) {
                    print("I DO NOT HAVE A BODY.\n");
                } else if (computerTailCount === 1) {
                    print("I DO NOT NEED A TAIL.\n");
                } else {
                    print("I NOW HAVE A TAIL.\n");
                    computerTailCount = 1;
                    partFound = true;
                }
                break;
            case 6:
                print("6=LEGS\n");
                if (computerFeetCount === 6) {
                    print("I HAVE 6 FEET.\n");
                } else if (computerBodyCount === 0) {
                    print("I DO NOT HAVE A BODY.\n");
                } else {
                    computerFeetCount++;
                    partFound = true;
                    print("I NOW HAVE " + computerFeetCount + " LEGS.\n");
                }
                break;
        }
        if (playerFeelerCount === 2 && playerTailCount === 1 && playerFeetCount === 6) {
            print("YOUR BUG IS FINISHED.\n");
            gameInProgress = false;
        }
        if (computerFeelerCount === 2 && computerBodyCount === 1 && computerFeetCount === 6) {
            print("MY BUG IS FINISHED.\n");
            gameInProgress = false;
        }
        if (!partFound)
            continue;
        print("DO YOU WANT THE PICTURES");
        const showPictures = await input();
        if (showPictures.toUpperCase() === "NO")
            continue;
        print("*****YOUR BUG*****\n");
        print("\n");
        print("\n");
        drawBug(playerFeelerCount, playerHeadCount, playerNeckCount, playerBodyCount, playerTailCount, playerFeetCount, "A");
        print("*****MY BUG*****\n");
        print("\n");
        print("\n");
        drawBug(computerFeelerCount, computerHeadCount, computerNeckCount, computerBodyCount, computerTailCount, computerFeetCount, "F");
        for (let z = 1; z <= 4; z++)
            print("\n");
    }
    print("I HOPE YOU ENJOYED THE GAME, PLAY IT AGAIN SOON!!\n");
    scrollToBottom();
}

```

这道题的代码是一个 C 语言的主函数，也就是程序的入口。在 C 语言中，每个程序都必须包含一个 main 函数，程序的控制权在 main 函数内。

main 函数内包含程序的主要操作，通常包括定义变量、设置计数器、判断条件、调用函数等。main 函数还可以包含程序的输入输出操作，用于与用户交互。

在这道题中，main 函数没有做任何具体的操作，它只是一个程序的入口。当你运行这个程序时，程序的控制权会从 main 函数开始，然后依次执行 main 函数内的代码。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)
