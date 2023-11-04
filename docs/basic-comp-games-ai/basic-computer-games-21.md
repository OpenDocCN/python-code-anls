# BasicComputerGames源码解析 21

# `12_Bombs_Away/csharp/BombsAwayGame/GermanySide.cs`

这段代码定义了一个名为 "GermanySide" 的类，继承自 "MissionSide" 类，用于表示德国的游戏角色。该角色可以飞往俄罗斯、英国和法国。

该类包含一个构造函数，用于初始化游戏中的用户界面，并覆盖 "ChooseMissionMessage" 和 "AllMissions" 方法，用于选择任务和显示可用的任务列表。

"ChooseMissionMessage" 方法返回一个德语格式的字符串，其中包含 "A NAZI, EH?" 和 "OH WELL. ARE YOU GOING FOR" 两部分，表示在选择任务时需要输入德国语。

"AllMissions" 方法返回一个列表，其中包含德国可以飞往的所有任务的列表。这些任务通常来自于 "MissionSide" 类。


```
﻿namespace BombsAwayGame;

/// <summary>
/// Germany protagonist. Can fly missions to Russia, England, and France.
/// </summary>
internal class GermanySide : MissionSide
{
    public GermanySide(IUserInterface ui)
        : base(ui)
    {
    }

    protected override string ChooseMissionMessage => "A NAZI, EH?  OH WELL.  ARE YOU GOING FOR";

    protected override IList<Mission> AllMissions => new Mission[]
    {
        new("RUSSIA", "YOU'RE NEARING STALINGRAD."),
        new("ENGLAND", "NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR."),
        new("FRANCE", "NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS.")
    };
}

```

# `12_Bombs_Away/csharp/BombsAwayGame/ItalySide.cs`

这段代码定义了一个名为"ItalySide"的类，属于"MissionSide"类。这个类有一个内部成员"ui"，表示用户界面，继承自"BaseMissionSide"类。

在"ui"的构造函数中，传递了一个用户界面实例，并将其设置为当前对象的父对象。

"ChooseMissionMessage"方法是一个摘要，它返回一个字符串，用于从可供选择的任务中选择一个任务。

"AllMissions"方法返回一个包含意大利可以执行的所有任务的列表。

最后，定义了"Albania"、"Greece"和"North Africa"三个任务的名称。


```
﻿namespace BombsAwayGame;

/// <summary>
/// Italy protagonist. Can fly missions to Albania, Greece, and North Africa.
/// </summary>
internal class ItalySide : MissionSide
{
    public ItalySide(IUserInterface ui)
        : base(ui)
    {
    }

    protected override string ChooseMissionMessage => "YOUR TARGET";

    protected override IList<Mission> AllMissions => new Mission[]
    {
        new("ALBANIA", "SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE."),
        new("GREECE", "BE CAREFUL!!!"),
        new("NORTH AFRICA", "YOU'RE GOING FOR THE OIL, EH?")
    };
}

```

# `12_Bombs_Away/csharp/BombsAwayGame/IUserInterface.cs`

这是一个namespace中的接口定义，它描述了一个为游戏提供数据接口的接口。

该接口定义了三个方法：

1. `Output(string message)`：用于在游戏中输出给玩家一个消息。

2. `Choose(string message, IList<string> choices, out int index)`：从给定的选择中选择一个选项，并返回用户选择的选项编号。

3. `ChooseYesOrNo(string message)`：询问用户是“是”还是“否”，返回对应的True或False。

4. `InputInteger()`：从用户那里获取一个整数，并返回给游戏。


```
﻿namespace BombsAwayGame;

/// <summary>
/// Represents an interface for supplying data to the game.
/// </summary>
/// <remarks>
/// Abstracting the UI allows us to concentrate its concerns in one part of our code and to change UI behavior
/// without creating any risk of changing the game logic. It also allows us to supply an automated UI for tests.
/// </remarks>
public interface IUserInterface
{
    /// <summary>
    /// Display the given message.
    /// </summary>
    /// <param name="message">Message to display.</param>
    void Output(string message);

    /// <summary>
    /// Choose an item from the given choices.
    /// </summary>
    /// <param name="message">Message to display.</param>
    /// <param name="choices">Choices to choose from.</param>
    /// <returns>Index of choice in <paramref name="choices"/> that user chose.</returns>
    int Choose(string message, IList<string> choices);

    /// <summary>
    /// Allow user to choose Yes or No.
    /// </summary>
    /// <param name="message">Message to display.</param>
    /// <returns>True if user chose Yes, false if user chose No.</returns>
    bool ChooseYesOrNo(string message);

    /// <summary>
    /// Get integer from user.
    /// </summary>
    /// <returns>Integer supplied by user.</returns>
    int InputInteger();
}

```

# `12_Bombs_Away/csharp/BombsAwayGame/JapanSide.cs`

这段代码是一个名为"BombsAwayGame"的命名空间中的类，代表了日本玩家在游戏中的角色。这个类定义了一个名为"JapanSide"的内部类，这个类继承自"Side"类，可能代表着游戏中的一个角色或者游戏中的一个玩家。

在这个类的"Play"方法中，执行了日本玩家飞行一个神风突袭任务。这个任务的逻辑与"MissionSide"类不同，可能是因为"BombsAwayGame"游戏引擎与"MissionSide"游戏的引擎不同。

首先，这个方法会询问玩家是否是第一次飞行神风突袭任务，如果是，那么游戏会成功65%的概率。如果不是第一次，那么这个方法会执行一个敌人反攻，试图摧毁美国的航母。

然后，在这个方法的实现中，我们调用了"UI.ChooseYesOrNo"方法来获取玩家对飞行神风突袭任务的第一人称视图的答案。然后，我们使用"EnemyCounterattack"方法来处理敌人反攻。如果随机分数大于0.65，我们调用"MissionSucceeded"方法，如果失败，我们调用"MissionFailed"方法。


```
﻿namespace BombsAwayGame;

/// <summary>
/// Japan protagonist. Flies a kamikaze mission, which has a different logic from <see cref="MissionSide"/>s.
/// </summary>
internal class JapanSide : Side
{
    public JapanSide(IUserInterface ui)
        : base(ui)
    {
    }

    /// <summary>
    /// Perform a kamikaze mission. If first kamikaze mission, it will succeed 65% of the time. If it's not
    /// first kamikaze mission, perform an enemy counterattack.
    /// </summary>
    public override void Play()
    {
        UI.Output("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.");

        bool isFirstMission = UI.ChooseYesOrNo("YOUR FIRST KAMIKAZE MISSION(Y OR N)?");
        if (!isFirstMission)
        {
            // LINE 207 of original BASIC: hitRatePercent is initialized to 0,
            // but R, the type of artillery, is not initialized at all. Setting
            // R = 1, which is to say EnemyArtillery = Guns, gives the same result.
            EnemyCounterattack(Guns, hitRatePercent: 0);
        }
        else if (RandomFrac() > 0.65)
        {
            MissionSucceeded();
        }
        else
        {
            MissionFailed();
        }
    }
}

```

# `12_Bombs_Away/csharp/BombsAwayGame/Mission.cs`

这段代码定义了一个名为"Mission"的类，用于表示一个可以在游戏中进行的任务。这个类包含两个字符串类型的成员变量，分别名为"Name"和"Description"，用于存储任务名称和描述。

这个类的定义是在namespace中进行的，说明这个类属于"BombsAwayGame"命名空间。


```
﻿namespace BombsAwayGame;

/// <summary>
/// Represents a mission that can be flown by a <see cref="MissionSide"/>.
/// </summary>
/// <param name="Name">Name of mission.</param>
/// <param name="Description">Description of mission.</param>
internal record class Mission(string Name, string Description);

```

# `12_Bombs_Away/csharp/BombsAwayGame/MissionSide.cs`

This is a class that appears to be part of a game. It has a method called `ChooseEnemyArtillery()` that allows the player to choose between different types of enemy artilleries (e.g. guns, missiles) and returns the chosen one.

It also has two constants called `MinEnemyHitRatePercent` and `MaxEnemyHitRatePercent` which limit the allowed hit rate percent for the player's own AI to prevent it from being easily bypassed.

The class also has a method called `EnemyHitRatePercentFromUI()` which gets the player's hit rate percent from the UI. It displays a message to the player asking them to enter a hit rate between the specified minimum and maximum (10% to 50%). If the player enters a number that is outside of the specified range, the mission is considered to have failed and the player is punished.


```
﻿namespace BombsAwayGame;

/// <summary>
/// Represents a protagonist that chooses a standard (non-kamikaze) mission.
/// </summary>
internal abstract class MissionSide : Side
{
    /// <summary>
    /// Create instance using the given UI.
    /// </summary>
    /// <param name="ui">UI to use.</param>
    public MissionSide(IUserInterface ui)
        : base(ui)
    {
    }

    /// <summary>
    /// Reasonable upper bound for missions flown previously.
    /// </summary>
    private const int MaxMissionCount = 160;

    /// <summary>
    /// Choose a mission and attempt it. If attempt fails, perform an enemy counterattack.
    /// </summary>
    public override void Play()
    {
        Mission mission = ChooseMission();
        UI.Output(mission.Description);

        int missionCount = MissionCountFromUI();
        CommentOnMissionCount(missionCount);

        AttemptMission(missionCount);
    }

    /// <summary>
    /// Choose a mission.
    /// </summary>
    /// <returns>Mission chosen.</returns>
    private Mission ChooseMission()
    {
        IList<Mission> missions = AllMissions;
        string[] missionNames = missions.Select(a => a.Name).ToArray();
        int index = UI.Choose(ChooseMissionMessage, missionNames);
        return missions[index];
    }

    /// <summary>
    /// Message to display when choosing a mission.
    /// </summary>
    protected abstract string ChooseMissionMessage { get; }

    /// <summary>
    /// All aviailable missions to choose from.
    /// </summary>
    protected abstract IList<Mission> AllMissions { get; }

    /// <summary>
    /// Get mission count from UI. If mission count exceeds a reasonable maximum, ask UI again.
    /// </summary>
    /// <returns>Mission count from UI.</returns>
    private int MissionCountFromUI()
    {
        const string HowManyMissions = "HOW MANY MISSIONS HAVE YOU FLOWN?";
        string inputMessage = HowManyMissions;

        bool resultIsValid;
        int result;
        do
        {
            UI.Output(inputMessage);
            result = UI.InputInteger();
            if (result < 0)
            {
                UI.Output($"NUMBER OF MISSIONS CAN'T BE NEGATIVE.");
                resultIsValid = false;
            }
            else if (result > MaxMissionCount)
            {
                resultIsValid = false;
                UI.Output($"MISSIONS, NOT MILES...{MaxMissionCount} MISSIONS IS HIGH EVEN FOR OLD-TIMERS.");
                inputMessage = "NOW THEN, " + HowManyMissions;
            }
            else
            {
                resultIsValid = true;
            }
        }
        while (!resultIsValid);

        return result;
    }

    /// <summary>
    /// Display a message about the given mission count, if it is unusually high or low.
    /// </summary>
    /// <param name="missionCount">Mission count to comment on.</param>
    private void CommentOnMissionCount(int missionCount)
    {
        if (missionCount >= 100)
        {
            UI.Output("THAT'S PUSHING THE ODDS!");
        }
        else if (missionCount < 25)
        {
            UI.Output("FRESH OUT OF TRAINING, EH?");
        }
    }

    /// <summary>
    /// Attempt mission.
    /// </summary>
    /// <param name="missionCount">Number of missions previously flown. Higher mission counts will yield a higher probability of success.</param>
    private void AttemptMission(int missionCount)
    {
        if (missionCount < RandomInteger(0, MaxMissionCount))
        {
            MissedTarget();
        }
        else
        {
            MissionSucceeded();
        }
    }

    /// <summary>
    /// Display message indicating that target was missed. Choose enemy artillery and perform a counterattack.
    /// </summary>
    private void MissedTarget()
    {
        UI.Output("MISSED TARGET BY " + (2 + RandomInteger(0, 30)) + " MILES!");
        UI.Output("NOW YOU'RE REALLY IN FOR IT !!");

        // Choose enemy and counterattack.
        EnemyArtillery enemyArtillery = ChooseEnemyArtillery();

        if (enemyArtillery == Missiles)
        {
            EnemyCounterattack(enemyArtillery, hitRatePercent: 0);
        }
        else
        {
            int hitRatePercent = EnemyHitRatePercentFromUI();
            if (hitRatePercent < MinEnemyHitRatePercent)
            {
                UI.Output("YOU LIE, BUT YOU'LL PAY...");
                MissionFailed();
            }
            else
            {
                EnemyCounterattack(enemyArtillery, hitRatePercent);
            }
        }
    }

    /// <summary>
    /// Choose enemy artillery from UI.
    /// </summary>
    /// <returns>Artillery chosen.</returns>
    private EnemyArtillery ChooseEnemyArtillery()
    {
        EnemyArtillery[] artilleries = new EnemyArtillery[] { Guns, Missiles, Both };
        string[] artilleryNames = artilleries.Select(a => a.Name).ToArray();
        int index = UI.Choose("DOES THE ENEMY HAVE", artilleryNames);
        return artilleries[index];
    }

    /// <summary>
    /// Minimum allowed hit rate percent.
    /// </summary>
    private const int MinEnemyHitRatePercent = 10;

    /// <summary>
    /// Maximum allowed hit rate percent.
    /// </summary>
    private const int MaxEnemyHitRatePercent = 50;

    /// <summary>
    /// Get the enemy hit rate percent from UI. Value must be between zero and <see cref="MaxEnemyHitRatePercent"/>.
    /// If value is less than <see cref="MinEnemyHitRatePercent"/>, mission fails automatically because the user is
    /// assumed to be untruthful.
    /// </summary>
    /// <returns>Enemy hit rate percent from UI.</returns>
    private int EnemyHitRatePercentFromUI()
    {
        UI.Output($"WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS ({MinEnemyHitRatePercent} TO {MaxEnemyHitRatePercent})");

        bool resultIsValid;
        int result;
        do
        {
            result = UI.InputInteger();
            // Let them enter a number below the stated minimum, as they will be caught and punished.
            if (0 <= result && result <= MaxEnemyHitRatePercent)
            {
                resultIsValid = true;
            }
            else
            {
                resultIsValid = false;
                UI.Output($"NUMBER MUST BE FROM {MinEnemyHitRatePercent} TO {MaxEnemyHitRatePercent}");
            }
        }
        while (!resultIsValid);

        return result;
    }
}

```

# `12_Bombs_Away/csharp/BombsAwayGame/Side.cs`



This is a class written in the标志性 programming language that implements a simple text-based shooter game. The game has three different types of enemy artillery: Guns, Missiles, and Both Guns and Missiles. Each type of enemy artillery has a unique hit rate and can be used to attack the player up to 35 times.

The `MissionSucceeded()` method is called when the player defeats the enemy in a mission. It displays a message indicating that the mission is successful and then displays a random number of enemies that were killed.

The `MissionFailed()` method is called when the player fails a mission. It displays a message indicating that the mission is over and then displays a string of messages indicating that the player has been shot down.

The `EnemyCounterattack()` method is called when the player uses an enemy's artillery attack to counterattack. It takes two parameters: the enemy artillery to use and the hit rate percent. It performs an attack against the enemy and displays a message indicating whether the attack was successful or not.

The game also has a `Both` type of enemy artillery, which can be used to attack the enemy up to 35 times.


```
﻿namespace BombsAwayGame;

/// <summary>
/// Represents a protagonist in the game.
/// </summary>
internal abstract class Side
{
    /// <summary>
    /// Create instance using the given UI.
    /// </summary>
    /// <param name="ui">UI to use.</param>
    public Side(IUserInterface ui)
    {
        UI = ui;
    }

    /// <summary>
    /// Play this side.
    /// </summary>
    public abstract void Play();

    /// <summary>
    /// User interface supplied to ctor.
    /// </summary>
    protected IUserInterface UI { get; }

    /// <summary>
    /// Random-number generator for this play-through.
    /// </summary>
    private readonly Random _random = new();

    /// <summary>
    /// Gets a random floating-point number greater than or equal to zero, and less than one.
    /// </summary>
    /// <returns>Random floating-point number greater than or equal to zero, and less than one.</returns>
    protected double RandomFrac() => _random.NextDouble();

    /// <summary>
    /// Gets a random integer in a range.
    /// </summary>
    /// <param name="minValue">The inclusive lower bound of the number returned.</param>
    /// <param name="maxValue">The exclusive upper bound of the number returned.</param>
    /// <returns>Random integer in a range.</returns>
    protected int RandomInteger(int minValue, int maxValue) => _random.Next(minValue: minValue, maxValue: maxValue);

    /// <summary>
    /// Display messages indicating the mission succeeded.
    /// </summary>
    protected void MissionSucceeded()
    {
        UI.Output("DIRECT HIT!!!! " + RandomInteger(0, 100) + " KILLED.");
        UI.Output("MISSION SUCCESSFUL.");
    }

    /// <summary>
    /// Gets the Guns type of enemy artillery.
    /// </summary>
    protected EnemyArtillery Guns { get; } = new("GUNS", 0);

    /// <summary>
    /// Gets the Missiles type of enemy artillery.
    /// </summary>
    protected EnemyArtillery Missiles { get; } = new("MISSILES", 35);

    /// <summary>
    /// Gets the Both Guns and Missiles type of enemy artillery.
    /// </summary>
    protected EnemyArtillery Both { get; } = new("BOTH", 35);

    /// <summary>
    /// Perform enemy counterattack using the given artillery and hit rate percent.
    /// </summary>
    /// <param name="artillery">Enemy artillery to use.</param>
    /// <param name="hitRatePercent">Hit rate percent for enemy.</param>
    protected void EnemyCounterattack(EnemyArtillery artillery, int hitRatePercent)
    {
        if (hitRatePercent + artillery.Accuracy > RandomInteger(0, 100))
        {
            MissionFailed();
        }
        else
        {
            UI.Output("YOU MADE IT THROUGH TREMENDOUS FLAK!!");
        }
    }

    /// <summary>
    /// Display messages indicating the mission failed.
    /// </summary>
    protected void MissionFailed()
    {
        UI.Output("* * * * BOOM * * * *");
        UI.Output("YOU HAVE BEEN SHOT DOWN.....");
        UI.Output("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR");
        UI.Output("LAST TRIBUTE...");
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `12_Bombs_Away/java/src/BombsAway.java`

这段代码是一个Java类，名为"BombsAway"，它从命令行输入用户输入，通过控制台输出游戏玩法。这个游戏是基于1970年代一款名为"Bombs Away"的原始游戏的玩法而创建的，它没有添加新的功能，只是简单地重新实现了这个游戏。

具体来说，这段代码实现了以下功能：

1. 读取用户输入：使用Java类Scanner的读取文本方法，从控制台读取用户的输入。
2. 打印游戏信息：使用Java类System的println方法，将游戏信息打印到控制台。
3. 判断用户选择：通过控制台输入告诉用户他们可以选择继续游戏还是退出游戏。
4. 分发炸弹：使用for循环，根据用户输入的数字，从1到100（包括100）生成不同数量的炸弹，然后将这些炸弹按照数学规律分发给用户。
5. 游戏循环：使用无限循环，让用户一直玩游戏，直到他们选择退出。

由于没有添加新的功能，所以游戏跟1970年的Bombs Away游戏玩法的体验基本一致。


```
import java.util.Scanner;

/**
 * Game of Bombs Away
 *
 * Based on the Basic game of Bombs Away here
 * https://github.com/coding-horror/basic-computer-games/blob/main/12_Bombs_Away/bombsaway.bas
 *
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without adding new features.
 * Obvious bugs where found have been fixed, but the playability and overlook and feel
 * of the game have been faithfully reproduced.
 *
 * Modern Java coding conventions have been employed and JDK 11 used for maximum compatibility.
 *
 * Java port by https://github.com/journich
 *
 */
```

This is a Java class that defines a player interface. It includes methods for getting a number from the keyboard, displaying text and getting input, checking whether the player entered "Y" or "YES" to a question, and generating a random number.

The class implements the `IInteractivePlayer` interface, which defines the methods for playing the game. The class has a `displayTextAndGetInput` method that displays text on the screen and accepts input from the keyboard. The `getNumberFromKeyboard` method accepts a string from the keyboard and converts it to an integer. The `yesEntered` method checks whether the player entered "Y" or "YES" to a question. The `stringIsAnyValue` method checks whether a string is equal to one of a variable number of values.

The class also has a `randomNumber` method that generates a random number.

Overall, this class is designed to provide a simple interface for a player in a game.


```
public class BombsAway {

    public static final int MAX_PILOT_MISSIONS = 160;
    public static final int MAX_CASUALTIES = 100;
    public static final int MISSED_TARGET_CONST_1 = 2;
    public static final int MISSED_TARGET_CONST_2 = 30;
    public static final int CHANCE_OF_BEING_SHOT_DOWN_BASE = 100;
    public static final double SIXTY_FIVE_PERCENT = .65;

    private enum GAME_STATE {
        START,
        CHOOSE_SIDE,
        CHOOSE_PLANE,
        CHOOSE_TARGET,
        CHOOSE_MISSIONS,
        CHOOSE_ENEMY_DEFENCES,
        FLY_MISSION,
        DIRECT_HIT,
        MISSED_TARGET,
        PROCESS_FLAK,
        SHOT_DOWN,
        MADE_IT_THROUGH_FLAK,
        PLAY_AGAIN,
        GAME_OVER
    }

    public enum SIDE {
        ITALY(1),
        ALLIES(2),
        JAPAN(3),
        GERMANY(4);

        private final int value;

        SIDE(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }


    }

    public enum TARGET {
        ALBANIA(1),
        GREECE(2),
        NORTH_AFRICA(3),
        RUSSIA(4),
        ENGLAND(5),
        FRANCE(6);

        private final int value;

        TARGET(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    public enum ENEMY_DEFENCES {
        GUNS(1),
        MISSILES(2),
        BOTH(3);

        private final int value;

        ENEMY_DEFENCES(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    public enum AIRCRAFT {
        LIBERATOR(1),
        B29(2),
        B17(3),
        LANCASTER(4);

        private final int value;

        AIRCRAFT(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    // Used for keyboard input
    private final Scanner kbScanner;

    // Current game state
    private GAME_STATE gameState;

    private SIDE side;

    private int missions;

    private int chanceToBeHit;
    private int percentageHitRateOfGunners;
    private boolean liar;

    public BombsAway() {

        gameState = GAME_STATE.START;

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);
    }

    /**
     * Main game loop
     *
     */
    public void play() {

        do {
            switch (gameState) {

                // Show an introduction the first time the game is played.
                case START:
                    intro();
                    chanceToBeHit = 0;
                    percentageHitRateOfGunners = 0;
                    liar = false;

                    gameState = GAME_STATE.CHOOSE_SIDE;
                    break;

                case CHOOSE_SIDE:
                    side = getSide("WHAT SIDE -- ITALY(1), ALLIES(2), JAPAN(3), GERMANY(4) ? ");
                    if (side == null) {
                        System.out.println("TRY AGAIN...");
                    } else {
                        // Different game paths depending on which side was chosen
                        switch (side) {
                            case ITALY:
                            case GERMANY:
                                gameState = GAME_STATE.CHOOSE_TARGET;
                                break;
                            case ALLIES:
                            case JAPAN:
                                gameState = GAME_STATE.CHOOSE_PLANE;
                                break;
                        }
                    }
                    break;

                case CHOOSE_TARGET:
                    String prompt;
                    if (side == SIDE.ITALY) {
                        prompt = "YOUR TARGET -- ALBANIA(1), GREECE(2), NORTH AFRICA(3) ? ";
                    } else {
                        // Germany
                        System.out.println("A NAZI, EH?  OH WELL.  ARE YOU GOING FOR RUSSIA(1),");
                        prompt = "ENGLAND(2), OR FRANCE(3) ? ";
                    }
                    TARGET target = getTarget(prompt);
                    if (target == null) {
                        System.out.println("TRY AGAIN...");
                    } else {
                        displayTargetMessage(target);
                        gameState = GAME_STATE.CHOOSE_MISSIONS;
                    }

                case CHOOSE_MISSIONS:
                    missions = getNumberFromKeyboard("HOW MANY MISSIONS HAVE YOU FLOWN? ");

                    if(missions <25) {
                        System.out.println("FRESH OUT OF TRAINING, EH?");
                        gameState = GAME_STATE.FLY_MISSION;
                    } else if(missions < 100) {
                        System.out.println("THAT'S PUSHING THE ODDS!");
                        gameState = GAME_STATE.FLY_MISSION;
                    } else if(missions >=160) {
                        System.out.println("MISSIONS, NOT MILES...");
                        System.out.println("150 MISSIONS IS HIGH EVEN FOR OLD-TIMERS.");
                        System.out.println("NOW THEN, ");
                    } else {
                        // No specific message if missions is 100-159, but still valid
                        gameState = GAME_STATE.FLY_MISSION;
                    }
                    break;

                case CHOOSE_PLANE:
                    switch(side) {
                        case ALLIES:
                            AIRCRAFT plane = getPlane("AIRCRAFT -- LIBERATOR(1), B-29(2), B-17(3), LANCASTER(4)? ");
                            if(plane == null) {
                                System.out.println("TRY AGAIN...");
                            } else {
                                switch(plane) {

                                    case LIBERATOR:
                                        System.out.println("YOU'VE GOT 2 TONS OF BOMBS FLYING FOR PLOESTI.");
                                        break;
                                    case B29:
                                        System.out.println("YOU'RE DUMPING THE A-BOMB ON HIROSHIMA.");
                                        break;
                                    case B17:
                                        System.out.println("YOU'RE CHASING THE BISMARK IN THE NORTH SEA.");
                                        break;
                                    case LANCASTER:
                                        System.out.println("YOU'RE BUSTING A GERMAN HEAVY WATER PLANT IN THE RUHR.");
                                        break;
                                }

                                gameState = GAME_STATE.CHOOSE_MISSIONS;
                            }
                            break;

                        case JAPAN:
                            System.out.println("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.");
                            if(yesEntered(displayTextAndGetInput("YOUR FIRST KAMIKAZE MISSION(Y OR N) ? "))) {
                                if(randomNumber(1) > SIXTY_FIVE_PERCENT) {
                                    gameState = GAME_STATE.DIRECT_HIT;
                                } else {
                                    // It's a miss
                                    gameState = GAME_STATE.MISSED_TARGET;
                                }
                            } else {
                                gameState = GAME_STATE.PROCESS_FLAK;
                            }
                            break;
                    }
                    break;

                case FLY_MISSION:
                    double missionResult = (MAX_PILOT_MISSIONS * randomNumber(1));
                    if(missions > missionResult) {
                        gameState = GAME_STATE.DIRECT_HIT;
                    } else {
                        gameState = GAME_STATE.MISSED_TARGET;
                    }

                    break;

                case DIRECT_HIT:
                    System.out.println("DIRECT HIT!!!! " + (int) Math.round(randomNumber(MAX_CASUALTIES)) + " KILLED.");
                    System.out.println("MISSION SUCCESSFUL.");
                    gameState = GAME_STATE.PLAY_AGAIN;
                    break;

                case MISSED_TARGET:
                    System.out.println("MISSED TARGET BY " + (int) Math.round(MISSED_TARGET_CONST_1 + MISSED_TARGET_CONST_2 * (randomNumber(1))) + " MILES!");
                    System.out.println("NOW YOU'RE REALLY IN FOR IT !!");
                    System.out.println();
                    gameState = GAME_STATE.CHOOSE_ENEMY_DEFENCES;
                    break;

                case CHOOSE_ENEMY_DEFENCES:
                    percentageHitRateOfGunners = 0;

                    ENEMY_DEFENCES enemyDefences = getEnemyDefences("DOES THE ENEMY HAVE GUNS(1), MISSILES(2), OR BOTH(3) ? ");
                    if(enemyDefences == null) {
                        System.out.println("TRY AGAIN...");
                    } else {
                        chanceToBeHit = 35;
                        switch(enemyDefences) {
                            case MISSILES:
                                // MISSILES... An extra 35 but cannot specify percentage hit rate for gunners
                                break;

                            case GUNS:
                                    // GUNS...  No extra 35 but can specify percentage hit rate for gunners
                                chanceToBeHit = 0;
                                // fall through (no break) on purpose because remaining code is applicable
                                // for both GUNS and BOTH options.

                            case BOTH:
                                // BOTH... An extra 35 and percentage hit rate for gunners can be specified.
                                percentageHitRateOfGunners = getNumberFromKeyboard("WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS (10 TO 50)? ");
                                if(percentageHitRateOfGunners < 10) {
                                    System.out.println("YOU LIE, BUT YOU'LL PAY...");
                                    liar = true;
                                }
                                break;
                        }
                    }
                    // If player didn't lie when entering percentage hit rate of gunners continue with game
                    // Otherwise shoot down the player.
                    if(!liar) {
                        gameState = GAME_STATE.PROCESS_FLAK;
                    } else {
                        gameState = GAME_STATE.SHOT_DOWN;
                    }
                    break;

                // Determine if the player's airplane makes it through the Flak.
                case PROCESS_FLAK:
                    double calc = (CHANCE_OF_BEING_SHOT_DOWN_BASE * randomNumber(1));

                    if ((chanceToBeHit + percentageHitRateOfGunners) > calc) {
                        gameState = GAME_STATE.SHOT_DOWN;
                    } else {
                        gameState = GAME_STATE.MADE_IT_THROUGH_FLAK;
                    }
                    break;

                case SHOT_DOWN:
                    System.out.println("* * * * BOOM * * * *");
                    System.out.println("YOU HAVE BEEN SHOT DOWN.....");
                    System.out.println("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR");
                    System.out.println("LAST TRIBUTE...");
                    gameState = GAME_STATE.PLAY_AGAIN;
                    break;

                case MADE_IT_THROUGH_FLAK:
                    System.out.println("YOU MADE IT THROUGH TREMENDOUS FLAK!!");
                    gameState = GAME_STATE.PLAY_AGAIN;
                    break;

                case PLAY_AGAIN:
                    if(yesEntered(displayTextAndGetInput("ANOTHER MISSION (Y OR N) ? "))) {
                        gameState = GAME_STATE.START;
                    } else {
                        System.out.println("CHICKEN !!!");
                        gameState = GAME_STATE.GAME_OVER;
                    }
                    break;
            }
        } while (gameState != GAME_STATE.GAME_OVER) ;
    }

    /**
     * Display a (brief) intro
     */
    public void intro() {
        System.out.println("YOU ARE A PILOT IN A WORLD WAR II BOMBER.");
    }

    /**
     * Determine the side the player is going to play on.
     * @param message displayed before the kb input
     * @return the SIDE enum selected by the player
     */
    private SIDE getSide(String message) {
        int valueEntered = getNumberFromKeyboard(message);
        for(SIDE side : SIDE.values()) {
            if(side.getValue() == valueEntered) {
                return side;
            }
        }

        // Input out of range
        return null;
    }

    /**
     * Determine the target the player is going for.
     * @param message displayed before the kb input
     * @return the TARGET enum selected by the player
     */
    private TARGET getTarget(String message) {
        int valueEntered = getNumberFromKeyboard(message);

        for(TARGET target : TARGET.values()) {
            if(target.getValue() == valueEntered) {
                return target;
            }
        }

        // Input out of range
        return null;
    }

    /**
     * Determine the airplane the player is going to fly.
     * @param message displayed before the kb input
     * @return the AIRCRAFT enum selected by the player
     */
    private AIRCRAFT getPlane(String message) {
        int valueEntered = getNumberFromKeyboard(message);

        for(AIRCRAFT plane : AIRCRAFT.values()) {
            if(plane.getValue() == valueEntered) {
                return plane;
            }
        }

        // Input out of range
        return null;

    }

    /**
     * Select the type of enemy defences.
     *
     * @param message displayed before kb input
     * @return the ENEMY_DEFENCES enum as selected by player
     */
    private ENEMY_DEFENCES getEnemyDefences(String message) {
        int valueEntered = getNumberFromKeyboard(message);
        for (ENEMY_DEFENCES enemyDefences : ENEMY_DEFENCES.values()) {
            if(enemyDefences.getValue() == valueEntered) {
                return enemyDefences;
            }
        }

        // Input out of range
        return null;
    }

    // output a specific message based on the target selected
    private void displayTargetMessage(TARGET target) {

        switch (target) {

            case ALBANIA:
                System.out.println("SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE.");
                break;
            case GREECE:
                System.out.println("BE CAREFUL!!!");
                break;
            case NORTH_AFRICA:
                System.out.println("YOU'RE GOING FOR THE OIL, EH?");
                break;
            case RUSSIA:
                System.out.println("YOU'RE NEARING STALINGRAD.");
                break;
            case ENGLAND:
                System.out.println("NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR.");
                break;
            case FRANCE:
                System.out.println("NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS.");
                break;
        }
    }

    /**
     * Accepts a string from the keyboard, and converts to an int
     *
     * @param message displayed text on screen before keyboard input
     *
     * @return the number entered by the player
     */
    private int getNumberFromKeyboard(String message) {

        String answer = displayTextAndGetInput(message);
        return Integer.parseInt(answer);
    }

    /**
     * Checks whether player entered Y or YES to a question.
     *
     * @param text  player string from kb
     * @return true of Y or YES was entered, otherwise false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * Check whether a string equals one of a variable number of values
     * Useful to check for Y or YES for example
     * Comparison is case-insensitive.
     *
     * @param text source string
     * @param values a range of values to compare against the source string
     * @return true if a comparison was found in one of the variable number of strings passed
     */
    private boolean stringIsAnyValue(String text, String... values) {

        // Cycle through the variable number of values and test each
        for(String val:values) {
            if(text.equalsIgnoreCase(val)) {
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
     * Used as a single digit of the computer player
     *
     * @return random number
     */
    private double randomNumber(int range) {
        return (Math.random()
                * (range));
    }
}

```

# `12_Bombs_Away/java/src/BombsAwayGame.java`

这段代码定义了一个名为BombsAwayGame的类，其中包含一个名为main的方法。

在main方法中，使用new关键字创建了一个BombsAway对象，并调用其的play()方法。

BombsAway是一个类，我无法确定它的具体功能，因为我不知道它的内部代码。但是，根据它的名称和常见的BombsAway游戏的实现，我假设它是一个用来对抗BombsAway游戏中的炸弹的类。在这个游戏中，玩家需要点击或按键来发球，并使用点击或按键来接球。

当玩家成功接球后，游戏将判定胜利。


```
public class BombsAwayGame {

    public static void main(String[] args) {

        BombsAway bombsAway = new BombsAway();
        bombsAway.play();
    }
}

```

# `12_Bombs_Away/javascript/bombsaway.js`

这段代码的作用是创建一个简单的 Web 应用程序，用于在浏览器中输入两个字符串，然后允许用户输入一个字符并输出其对应的字符。

具体来说，该应用程序包括以下功能：

1. 将创建 input 元素和文本输入框的代码放在一个函数中，以便在需要时调用。
2. 通过调用 print 函数将在文档中创建一个输出框，用于显示输入的字符串。
3. 通过调用 input 函数，用户将被要求输入一个字符，并将其存储在变量 input_str 中。
4. 事件监听器将监听输入元素的关键Down事件，当事件发生时，将捕获到字符并将其存储在 input_str 中。
5. 调用 print 函数将输入的字符串输出到文档中。
6. 调用 print 函数将处理输入的字符串和输出框进行交互，允许用户输入并输出字符。


```
// BOMBS AWAY
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

```

This is a mission game where the player is given a randomly generated target number and is trying to hit it with a laser. The player has a limited amount of time and must hit the target within that time or be labeled as having failed the mission. The player can also be hit by enemy gunners, but this is optional and increases the difficulty of the mission. The player can give their answer to the enemy's question, but they must give a number between 1 and 50 to do so. If the player answers correctly, the enemy will give them a percentage hit rate. If the player does not answer correctly, the enemy will give them a fail rate. The player will then have to try again to complete the mission.



```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// Main program
async function main()
{
    s = 0;
    t = 0;
    while (1) {
        print("YOU ARE A PILOT IN A WORLD WAR II BOMBER.\n");
        while (1) {
            print("WHAT SIDE -- ITALY(1), ALLIES(2), JAPAN(3), GERMANY(4)");
            a = parseInt(await input());
            if (a < 1 || a > 4)
                print("TRY AGAIN...\n");
            else
                break;
        }
        if (a == 1) {
            while (1) {
                print("YOUR TARGET -- ALBANIA(1), GREECE(2), NORTH AFRICA(3)");
                b = parseInt(await input());
                if (b < 1 || b > 3)
                    print("TRY AGAIN...\n");
                else
                    break;
            }
            print("\n");
            if (b == 1) {
                print("SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE.\n");
            } else if (b == 2) {
                print("BE CAREFUL!!!\n");
            } else {
                print("YOU'RE GOING FOR THE OIL, EH?\n");
            }
        } else if (a == 2) {
            while (1) {
                print("AIRCRAFT -- LIBERATOR(1), B-29(2), B-17(3), LANCASTER(4)");
                g = parseInt(await input());
                if (g < 1 || g > 4)
                    print("TRY AGAIN...\n");
                else
                    break;
            }
            print("\n");
            if (g == 1) {
                print("YOU'VE GOT 2 TONS OF BOMBS FLYING FOR PLOESTI.\n");
            } else if (g == 2) {
                print("YOU'RE DUMPING THE A-BOMB ON HIROSHIMA.\n");
            } else if (g == 3) {
                print("YOU'RE CHASING THE BISMARK IN THE NORTH SEA.\n");
            } else {
                print("YOU'RE BUSTING A GERMAN HEAVY WATER PLANT IN THE RUHR.\n");
            }
        } else if (a == 3) {
            print("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.\n");
            print("YOUR FIRST KAMIKAZE MISSION(Y OR N)");
            str = await input();
            if (str == "N") {
                s = 0;
            } else {
                s = 1;
                print("\n");
            }
        } else {
            while (1) {
                print("A NAZI, EH?  OH WELL.  ARE YOU GOING FOR RUSSIA(1),\n");
                print("ENGLAND(2), OR FRANCE(3)");
                m = parseInt(await input());
                if (m < 1 || m > 3)
                    print("TRY AGAIN...\n");
                else
                    break;
            }
            print("\n");
            if (m == 1) {
                print("YOU'RE NEARING STALINGRAD.\n");
            } else if (m == 2) {
                print("NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR.\n");
            } else if (m == 3) {
                print("NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS.\n");
            }
        }
        if (a != 3) {
            print("\n");
            while (1) {
                print("HOW MANY MISSIONS HAVE YOU FLOWN");
                d = parseInt(await input());
                if (d < 160)
                    break;
                print("MISSIONS, NOT MILES...\n");
                print("150 MISSIONS IS HIGH EVEN FOR OLD-TIMERS.\n");
                print("NOW THEN, ");
            }
            print("\n");
            if (d >= 100) {
                print("THAT'S PUSHING THE ODDS!\n");
            } else if (d < 25) {
                print("FRESH OUT OF TRAINING, EH?\n");
            }
            print("\n");
            if (d >= 160 * Math.random())
                hit = true;
            else
                hit = false;
        } else {
            if (s == 0) {
                hit = false;
            } else if (Math.random() > 0.65) {
                hit = true;
            } else {
                hit = false;
                s = 100;
            }
        }
        if (hit) {
            print("DIRECT HIT!!!! " + Math.floor(100 * Math.random()) + " KILLED.\n");
            print("MISSION SUCCESSFUL.\n");
        } else {
            t = 0;
            if (a != 3) {
                print("MISSED TARGET BY " + Math.floor(2 + 30 * Math.random()) + " MILES!\n");
                print("NOW YOU'RE REALLY IN FOR IT !!\n");
                print("\n");
                while (1) {
                    print("DOES THE ENEMY HAVE GUNS(1), MISSILE(2), OR BOTH(3)");
                    r = parseInt(await input());
                    if (r < 1 || r > 3)
                        print("TRY AGAIN...\n");
                    else
                        break;
                }
                print("\n");
                if (r != 2) {
                    print("WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS (10 TO 50)");
                    s = parseInt(await input());
                    if (s < 10)
                        print("YOU LIE, BUT YOU'LL PAY...\n");
                    print("\n");
                }
                print("\n");
                if (r > 1)
                    t = 35;
            }
            if (s + t <= 100 * Math.random()) {
                print("YOU MADE IT THROUGH TREMENDOUS FLAK!!\n");
            } else {
                print("* * * * BOOM * * * *\n");
                print("YOU HAVE BEEN SHOT DOWN.....\n");
                print("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR\n");
                print("LAST TRIBUTE...\n");
            }
        }
        print("\n");
        print("\n");
        print("\n");
        print("ANOTHER MISSION (Y OR N)");
        str = await input();
        if (str != "Y")
            break;
    }
    print("CHICKEN !!!\n");
    print("\n");
}

```

这是C++中的一个标准函数，名为“main()”。这个函数是程序的入口点，当程序运行时，它首先会执行这个函数。 main()函数的作用是启动程序，通常会打开一个控制台窗口，接受用户输入的命令。

对于这段代码，虽然它们没有实际的输出，但这是一个标准输入/输出声明。通常，在main()函数之前，会包含一些来自用户输入的代码，然后将这些输入保存到程序变量中。然后，程序会根据用户输入的值执行不同的操作。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `12_Bombs_Away/python/bombs_away.py`

这段代码是一个用于在BASIC和Python 3中玩文字接龙游戏的函数。它实现了从BASIC到Python 3的迁移，通过Bernard Cooke (bernardcooke53)。

函数接收一个字符串参数 prompt，以及一个或多个字符串参数 choices。这个函数的作用是在提示用户输入一个选择后，从多个可供选择的字符串中选择一个并返回。如果用户输入的不是从 choices 中选择的字符串，函数将提示用户重新输入。

函数的核心部分是使用 Python 标准输入（通常是键盘输入）读取字符串 prompt，并将其存储在一个变量 ret 中。然后，函数使用 while 循环来读取用户输入，并在每次读取后检查当前输入是否属于 choices 中的一个字符串。如果是，函数将返回输入的字符串，否则将提示用户再次输入。

通过调用这个函数，用户可以在BASIC和Python 3之间进行文字接龙游戏。


```
"""
Bombs away

Ported from BASIC to Python3 by Bernard Cooke (bernardcooke53)
Tested with Python 3.8.10, formatted with Black and type checked with mypy.
"""
import random
from typing import Iterable


def _stdin_choice(prompt: str, *, choices: Iterable[str]) -> str:
    ret = input(prompt)
    while ret not in choices:
        print("TRY AGAIN...")
        ret = input(prompt)
    return ret


```

这段代码定义了三个函数，`player_survived()`、`player_death()` 和 `mission_success()`。

`player_survived()` 的作用是打印 "YOU MADE IT THROUGH TREMENDOUS FLAKE!!"。

`player_death()` 的作用是打印 "* * * * BOOM * * * *"。然后打印 "YOU HAVE BEEN SHOT DOWN....."。接着打印 "DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR"。最后打印 "LAST TRIBUTE..."。

`mission_success()` 的作用是打印 "DIRECT HIT!!!! {int(100 * random.random())} KILLED。" 和 "MISSION SUCCESSFUL。"。


```
def player_survived() -> None:
    print("YOU MADE IT THROUGH TREMENDOUS FLAK!!")


def player_death() -> None:
    print("* * * * BOOM * * * *")
    print("YOU HAVE BEEN SHOT DOWN.....")
    print("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR")
    print("LAST TRIBUTE...")


def mission_success() -> None:
    print(f"DIRECT HIT!!!! {int(100 * random.random())} KILLED.")
    print("MISSION SUCCESSFUL.")


```



This code defines a function called `death_with_chance`, which takes a float parameter `p_death` and returns a boolean value. The function uses the `random.random()` function to generate a random number between 0 and 1, and returns `True` if the player has not died (based on the random chance), and `False` if the player has died.

The second function `commence_non_kamikadi_attack()` takes an integer parameter `nmissions` and repeatedly asks the player how many missions they have flown. If the player has not flown 160 missions or more, the function prints a message and prompts the player to try again. If the player has flown 160 or more missions, the function returns the result of a random number between 0 and 1, based on the implementation of the function.

It is not clear from the code what the random number is being generated for, but it is possible that it is being generated using the `random.random()` function. This function generates a random number between the minimum and maximum values specified, so in this case it is likely generating a random number between 0 and 1.


```
def death_with_chance(p_death: float) -> bool:
    """
    Takes a float between 0 and 1 and returns a boolean
    if the player has survived (based on random chance)

    Returns True if death, False if survived
    """
    return p_death > random.random()


def commence_non_kamikazi_attack() -> None:
    while True:
        try:
            nmissions = int(input("HOW MANY MISSIONS HAVE YOU FLOWN? "))

            while nmissions >= 160:
                print("MISSIONS, NOT MILES...")
                print("150 MISSIONS IS HIGH EVEN FOR OLD-TIMERS")
                nmissions = int(input("NOW THEN, HOW MANY MISSIONS HAVE YOU FLOWN? "))
            break
        except ValueError:
            # In the BASIC implementation this
            # wasn't accounted for
            print("TRY AGAIN...")
            continue

    if nmissions >= 100:
        print("THAT'S PUSHING THE ODDS!")

    if nmissions < 25:
        print("FRESH OUT OF TRAINING, EH?")

    print()
    return (
        mission_success() if nmissions >= 160 * random.random() else mission_failure()
    )


```

这段代码定义了一个名为“mission_failure”的函数，它会导致程序陷入死循环，直到用户手动中断程序。函数内部包含了以下操作：

1. 从一个名为“weapons_choices”的字典中选择一种武器，可以使用数字1、2或3，代码中使用的是平方加上随机数再选择一个武器。
2. 打印一条消息，说明玩家的目标被导弹击中，并提示玩家面临真正的危险。
3. 使用一个名为“_stdin_choice”的函数，提示玩家选择敌人是否拥有枪或导弹，代码中使用的是一个格式化字符串，将选项打印出来。
4. 如果玩家选择的不是2，则说明枪手没有武器，代码中处理这种情况的方式是使用一个变量“enemy_gunner_accuracy”来记录每个枪手的准确率，如果准确率低于10，就说明玩家死亡。
5. 如果玩家选择了2，则根据枪手的准确率计算死亡概率，并将结果存储在变量“p_death”中。
6. 如果没有死亡，则使用“player_survived”函数来存储，否则继续循环执行“mission_failure”函数。

整个函数的目的是让玩家在不同的武器威胁下尝试生存，直到他们犯了一个错误，这个错误可以是选择错误的武器或者没有死亡。


```
def mission_failure() -> None:
    weapons_choices = {
        "1": "GUNS",
        "2": "MISSILES",
        "3": "BOTH",
    }
    print(f"MISSED TARGET BY {int(2 + 30 * random.random())} MILES!")
    print("NOW YOU'RE REALLY IN FOR IT !!")
    print()
    enemy_weapons = _stdin_choice(
        prompt="DOES THE ENEMY HAVE GUNS(1), MISSILES(2), OR BOTH(3)? ",
        choices=weapons_choices,
    )

    # If there are no gunners (i.e. weapon choice 2) then
    # we say that the gunners have 0 accuracy for the purposes
    # of calculating probability of player death

    enemy_gunner_accuracy = 0.0
    if enemy_weapons != "2":
        # If the enemy has guns, how accurate are the gunners?
        while True:
            try:
                enemy_gunner_accuracy = float(
                    input("WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS (10 TO 50)? ")
                )
                break
            except ValueError:
                # In the BASIC implementation this
                # wasn't accounted for
                print("TRY AGAIN...")
                continue

        if enemy_gunner_accuracy < 10:
            print("YOU LIE, BUT YOU'LL PAY...")
            return player_death()

    missile_threat_weighting = 0 if enemy_weapons == "1" else 35

    death = death_with_chance(
        p_death=(enemy_gunner_accuracy + missile_threat_weighting) / 100
    )

    return player_survived() if not death else player_death()


```

这段代码定义了一个函数 `play_italy()`，它接受一个参数 `None`，这意味着函数不会返回任何值。函数内部定义了一个字典 `targets_to_messages`，其中包含三个键值对，分别对应三个目标国家，以及相应的提示信息。

函数内部的 `_stdin_choice` 函数是一个从三个选项中选择一个目标国家的函数，它接受一个参数 `prompt`，用于在终端输入一个选择目标国家的提示信息。这个函数将 `targets_to_messages` 中的键值对映射到提示信息，然后从 `targets_to_messages` 中选择一个键，并输出相应的提示信息。

最后，函数内部调用了一个名为 `commence_non_kamikazi_attack` 的函数，但并没有定义任何参数。


```
def play_italy() -> None:
    targets_to_messages = {
        # 1 - ALBANIA, 2 - GREECE, 3 - NORTH AFRICA
        "1": "SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE.",
        "2": "BE CAREFUL!!!",
        "3": "YOU'RE GOING FOR THE OIL, EH?",
    }
    target = _stdin_choice(
        prompt="YOUR TARGET -- ALBANIA(1), GREECE(2), NORTH AFRICA(3)",
        choices=targets_to_messages,
    )

    print(targets_to_messages[target])
    return commence_non_kamikazi_attack()


```

这段代码定义了一个名为 `play_allies()` 的函数，它接受一个空括号作为参数，并返回一个空括号。函数内部定义了一个名为 `aircraft_to_message` 的字典，它包含了一些与不同航空兵种相关的信息，以空气八字为输入并输出相应的信息。接下来，函数使用 `_stdin_choice()` 函数从标准输入读取一行字符，然后使用 `choices` 参数从 `aircraft_to_message` 字典中选择相应的信息并输出。最后，函数调用 `commit_non_kamikazi_attack()` 函数，但不会执行任何操作，因此返回一个空括号。


```
def play_allies() -> None:
    aircraft_to_message = {
        "1": "YOU'VE GOT 2 TONS OF BOMBS FLYING FOR PLOESTI.",
        "2": "YOU'RE DUMPING THE A-BOMB ON HIROSHIMA.",
        "3": "YOU'RE CHASING THE BISMARK IN THE NORTH SEA.",
        "4": "YOU'RE BUSTING A GERMAN HEAVY WATER PLANT IN THE RUHR.",
    }
    aircraft = _stdin_choice(
        prompt="AIRCRAFT -- LIBERATOR(1), B-29(2), B-17(3), LANCASTER(4): ",
        choices=aircraft_to_message,
    )

    print(aircraft_to_message[aircraft])
    return commence_non_kamikazi_attack()


```

这道题目是一个简单的 Python 代码，主要目的是指导用户完成两个不同的游戏。这两个游戏都是关于如何在不同的游戏中选择不同的目标并成功完成任务。

游戏1是“您正在执行一项Kamikaze（神风）任务，您打算袭击USS LEXINGTON吗？”在回答是“是（Y）还是不是（N）”的情况下，它将返回玩家死亡（player_death）或任务成功（mission_success）。

游戏2是“您正在执行德国的游戏，您打算袭击哪些目标？”它将返回与游戏1中选择的目标相关的提示信息。然后，用户需要在这个游戏中选择一个目标，游戏将告诉用户他们选择的目标的防御能力。最后，它将返回一个提示，告诉用户是否开始攻击指定的目标。


```
def play_japan() -> None:
    print("YOU'RE FLYING A KAMIKAZE MISSION OVER THE USS LEXINGTON.")
    first_mission = input("YOUR FIRST KAMIKAZE MISSION? (Y OR N): ")
    if first_mission.lower() == "n":
        return player_death()
    return mission_success() if random.random() > 0.65 else player_death()


def play_germany() -> None:
    targets_to_messages = {
        # 1 - RUSSIA, 2 - ENGLAND, 3 - FRANCE
        "1": "YOU'RE NEARING STALINGRAD.",
        "2": "NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR.",
        "3": "NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS.",
    }
    target = _stdin_choice(
        prompt="A NAZI, EH?  OH WELL.  ARE YOU GOING FOR RUSSIA(1),\nENGLAND(2), OR FRANCE(3)? ",
        choices=targets_to_messages,
    )

    print(targets_to_messages[target])

    return commence_non_kamikazi_attack()


```

这段代码是一个函数，名为 `play_game()`，它返回一个 `None` 类型的值。

函数的作用是在程序中执行一系列动作，并返回一个指定动作的结果。具体来说，它通过 `print()` 函数告诉程序员自己在二战中的一名飞行员，然后定义了一个包含四个选择项的 `sides` 字典，其中每个键都有一个对应的选择项。

接下来，程序会要求用户选择一个选择项，它会首先在 `sides` 字典中查找指定的选择项，然后返回该选择项的结果。如果用户选择了指定以外的任何选择项，函数将返回 `None`。

在程序的主循环中，调用 `play_game()` 函数，然后程序将进入无限循环，要求用户再次选择任务。


```
def play_game() -> None:
    print("YOU ARE A PILOT IN A WORLD WAR II BOMBER.")
    sides = {"1": play_italy, "2": play_allies, "3": play_japan, "4": play_germany}
    side = _stdin_choice(
        prompt="WHAT SIDE -- ITALY(1), ALLIES(2), JAPAN(3), GERMANY(4): ", choices=sides
    )
    return sides[side]()


if __name__ == "__main__":
    again = True
    while again:
        play_game()
        again = input("ANOTHER MISSION? (Y OR N): ").upper() == "Y"

```