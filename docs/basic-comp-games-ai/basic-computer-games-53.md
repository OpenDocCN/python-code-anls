# BasicComputerGames源码解析 53

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


## King

This is one of the most comprehensive, difficult, and interesting games. (If you've never played one of these games, start with HAMMURABI.)

In this game, you are Premier of Setats Detinu, a small communist island 30 by 70 miles long. Your job is to decide upon the budget of your country and distribute money to your country from the communal treasury.

The money system is Rollods; each person needs 100 Rallods per year to survive. Your country's income comes from farm produce and tourists visiting your magnificent forests, hunting, fishing, etc. Part of your land is farm land but it also has an excellent mineral content and may be sold to foreign industry for strip mining. Industry import and support their own workers. Crops cost between 10 and 15 Rallods per square mile to plant, cultivate, and harvest. Your goal is to complete an eight-year term of office without major mishap. A word of warning: it isn't easy!

The author of this program is James A. Storer who wrote it while a student at Lexington High School.

⚠️ This game includes references to suicide or self-harm.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=96)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=111)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

Implementers should be aware that this game contains bugs.

### Bug 1

On basic line 1450

    1450 V3=INT(A+V3)
    1451 A=INT(A+V3)

...where A is the current treasury, and V3 is initially zero.
This would mean that the treasury doubles at the end of the first year, and all calculations for an increase in the treasury due to tourism are discarded.
Possibly, this made the game more playable, although impossible for the player to understand why the treasury was increasing?

A quick fix for this bug in the original code would be

    1450 V3=ABS(INT(V1-V2))
    1451 A=INT(A+V3)

...judging from the description of tourist income on basic line 1410

    1410 PRINT " YOU MADE";ABS(INT(V1-V2));"RALLODS FROM TOURIST TRADE."

### Bug 2

On basic line 1330 following was the variable T1 never assigned:

    1330 PRINT " YOU HARVESTED ";INT(J-U2);"SQ. MILES OF CROPS."
    1340 IF U2=0 THEN 1370
    1344 IF T1>=2 THEN 1370
    1350 PRINT "   (DUE TO ";
    1355 IF T1=0 THEN 1365
    1360 PRINT "INCREASED ";

Likely it should be the difference of the current years crop loss compared to the
last years crop loss.

### Bug 3

On basic line 1997 it is:

    1997 PRINT "   AND 10,000 SQ. MILES OF FOREST LAND."

but it should be:

    1997 PRINT "   AND 1,000 SQ. MILES OF FOREST LAND."

### Bug 4

On basic line 1310 we see this:

    1310 IF C=0 THEN 1324
    1320 PRINT "OF ";INT(J);"SQ. MILES PLANTED,";
    1324 ...

but it should probably be:

    1310 IF J=0 THEN 1324

### Bug 5

On basic line 1390 the income from tourism is calculated:

```
1390 A=INT(A+Q)
1400 V1=INT(((B-P1)*22)+(RND(1)*500))
1405 V2=INT((2000-D)*15)
1410 PRINT " YOU MADE";ABS(INT(V1-V2));"RALLODS FROM TOURIST TRADE."
```

It is very easily possible that `V2` is larger than `V1` e.g. if all of the land has been sold. In the original game this does not make a difference because of Bug 1 (see above).

However, judging by how `V1` and `V2` are handled in the code, it looks like `V1` is the basic income from tourism and `V2` is a deduction for pollution. When `ABS(INT(V1-V2))` is used as earnings from tourism, the player actually _gets_ money for a large enough pollution. So a better solution would be to let `V1 - V2` cap out at 0, so once the pollution is large enough, there is no income from tourists anymore.


# `53_King/csharp/Country.cs`

This is a class that simulates a farming simulation game. It allows the player to control various aspects of their farm, such as planting and harvesting crops, polluting the land, and spending or earning money.

The `PlantationProductivity` class represents the land's productivity. It sets the maximum productivity that can be achieved by farming the land. It does not actually perform the planting or harvesting, but instead sets the maximum amount of money the farmer can earn by doing so.

The `Farmland` class represents the land where the farmer can farm crops. It can be used to farm crops, but is also expensive and requires planting and harvesting to be done manually.

The `Worker` class represents the worker who can be hired to work on the farm. It can be used to perform various tasks, such as planting and harvesting crops, polluting the land, or selling crops.

The `Migration` class represents the migration of the worker. It allows the worker to move to a new location, which can be useful for expanding the farm to new areas.

The `TouristIncome` class represents the income that can be earned by entertaining tourists. This is earned by selling crops and worker services to tourists.

The `PollutionPrompt` class represents the pollution level of the land. It allows the player to spend money to pollute the land, which can increase the player's money but also gives the player pollution penalties.

The `RallodsSpent` class represents the amount of money spent by the player. It is the result of the player spending money on crops, worker services, or other purchases.

The `SpendMoney` class represents the player spending money on crops or other purchases.

The `AddWorkers` class allows the player to hire more workers to help with the farm's work.

The `Migration` class allows the player to move the worker to a new location.

The `TouristIncome` class allows the player to earn income from entertaining tourists.

The `PollutionPrompt` class allows the player to pollute the land, which can increase the player's money but also gives the player pollution penalties.

The `Rallods` class represents the amount of money spent by the player. It is the result of the player spending money on crops, worker services, or other purchases.

The `SpendMoney` class represents the player spending money on crops or other purchases.

The `RallodsSpent` class represents the amount of money spent by the player. It is the result of the player spending money on crops, worker services, or other purchases.


```
namespace King;

internal class Country
{
    private const int InitialLand = 1000;

    private readonly IReadWrite _io;
    private readonly IRandom _random;
    private float _rallods;
    private float _countrymen;
    private float _foreigners;
    private float _arableLand;
    private float _industryLand;

    public Country(IReadWrite io, IRandom random)
        : this(
            io,
            random,
            (int)(60000 + random.NextFloat(1000) - random.NextFloat(1000)),
            (int)(500 + random.NextFloat(10) - random.NextFloat(10)),
            0,
            InitialLand)
    {
    }

    public Country(IReadWrite io, IRandom random, float rallods, float countrymen, float foreigners, float land)
    {
        _io = io;
        _random = random;
        _rallods = rallods;
        _countrymen = countrymen;
        _foreigners = foreigners;
        _arableLand = land;
    }

    public string GetStatus(int landValue, int plantingCost) 
        => Resource.Status(_rallods, _countrymen, _foreigners, _arableLand, landValue, plantingCost);
    
    public float Countrymen => _countrymen;
    public float Workers => _foreigners;
    public bool HasWorkers => _foreigners > 0;
    private float FarmLand => _arableLand;
    public bool HasRallods => _rallods > 0;
    public float Rallods => _rallods;
    public float IndustryLand => InitialLand - _arableLand;
    public int PreviousTourismIncome { get; private set; }

    public bool SellLand(int landValue, out float landSold)
    {
        if (_io.TryReadValue(
                SellLandPrompt, 
                out landSold, 
                new ValidityTest(v => v <= FarmLand, () => SellLandError(FarmLand))))
        {
            _arableLand = (int)(_arableLand - landSold);
            _rallods = (int)(_rallods + landSold * landValue);
            return true;
        }

        return false;
    }

    public bool DistributeRallods(out float rallodsGiven)
    {
        if (_io.TryReadValue(
                GiveRallodsPrompt,
                out rallodsGiven, 
                new ValidityTest(v => v <= _rallods, () => GiveRallodsError(_rallods))))
        {
            _rallods = (int)(_rallods - rallodsGiven);
            return true;
        }

        return false;
    }

    public bool PlantLand(int plantingCost, out float landPlanted)
    {
        if (_io.TryReadValue(
                PlantLandPrompt, 
                out landPlanted, 
                new ValidityTest(v => v <= _countrymen * 2, PlantLandError1),
                new ValidityTest(v => v <= FarmLand, PlantLandError2(FarmLand)),
                new ValidityTest(v => v * plantingCost <= _rallods, PlantLandError3(_rallods))))
        {
            _rallods -= (int)(landPlanted * plantingCost);
            return true;
        }

        return false;
    }

    public bool ControlPollution(out float rallodsSpent)
    {
        if (_io.TryReadValue(
                PollutionPrompt,
                out rallodsSpent, 
                new ValidityTest(v => v <= _rallods, () => PollutionError(_rallods))))
        {
            _rallods = (int)(_rallods - rallodsSpent);
            return true;
        }

        return false;
    }

    public bool TrySpend(float amount, float landValue)
    {
        if (_rallods >= amount)
        {
            _rallods -= amount;
            return true;
        }
        
        _arableLand = (int)(_arableLand - (int)(amount - _rallods) / landValue);
        _rallods = 0;
        return false;
    }

    public void RemoveTheDead(int deaths) => _countrymen = (int)(_countrymen - deaths);

    public void Migration(int migration) => _countrymen = (int)(_countrymen + migration);

    public void AddWorkers(int newWorkers) => _foreigners = (int)(_foreigners + newWorkers);

    public void SellCrops(int income) => _rallods = (int)(_rallods + income);

    public void EntertainTourists(int income)
    {
        PreviousTourismIncome = income;
        _rallods = (int)(_rallods + income);
    }
}

```

# `53_King/csharp/Game.cs`



这段代码是一个名为 King 的namespace内部类，定义了一个名为 Game 的类。

Game类包含一个私有变量IReadWrite和一个私有变量IRandom，以及构造函数和Play方法。

构造函数初始化IReadWrite和IRandom，用于写入和生成随机数。

Play方法包含从IO中读取标题，并调用SetUpReign方法，如果设置成功，则从随机数生成器中获取年份，并在while循环中递增。

SetUpReign方法从用户输入中读取游戏提示，如果用户输入为“再次”，则尝试从随机数生成器中获取游戏数据，并返回它。如果从随机数生成器中获取失败或用户输入不正确，则返回null。

在Play方法中，从随机数生成器中获取一个年份，并将其写入文件。

此外，还包含一个InstructionsPrompt成员函数，用于从用户输入中读取游戏提示，如果无法从文件中读取，则从控制台获取提示信息。


```
namespace King;

internal class Game
{
    const int TermOfOffice = 8;

    private readonly IReadWrite _io;
    private readonly IRandom _random;

    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    public void Play()
    {
        _io.Write(Title);

        var reign = SetUpReign();
        if (reign != null)
        {
            while (reign.PlayYear());
        }

        _io.WriteLine();
        _io.WriteLine();
    }

    private Reign? SetUpReign()
    {
        var response = _io.ReadString(InstructionsPrompt).ToUpper();

        if (response.Equals("Again", StringComparison.InvariantCultureIgnoreCase))
        {
            return _io.TryReadGameData(_random, out var reign) ? reign : null;
        }
        
        if (!response.StartsWith("N", StringComparison.InvariantCultureIgnoreCase))
        {
            _io.Write(InstructionsText(TermOfOffice));
        }

        _io.WriteLine();
        return new Reign(_io, _random);
    }
}

```

# `53_King/csharp/IOExtensions.cs`

This code appears to be a context-sensitive library for reading and writing values to a PostgreSQL database. It contains several functions for reading and writing data to the database, as well as functions for handling errors.

The `TryReadValue` function attempts to read a value from the database and return it. If the read is successful, it returns `true`. If not, it returns `false`.

The `TryReadValue` function can also be used to read a value from the database and handle errors. It takes a `string` prompt and a predicate for determining if the value is valid. It returns the value if the value is valid, or an error message if the value is not.

The `ReadValue` function reads the value from the database and returns it. It takes a `string` prompt and a function for handling errors. It returns the value if the read is successful, or the specified error if the read is not.

The `WriteValue` function writes a value to the database. It takes a `string` prompt and a function for handling errors. It returns the success message if the write was successful, or the specified error if the write was not.

The `SavedTreasuryPrompt` and `SavedCountrymenPrompt` and `SavedWorkersPrompt` functions are not used in this context. It is not clear what they are intended to do.

The `SavedLandPrompt` function is used to handle errors. It takes a `string` prompt and a predicate for determining if the value is valid. It returns the value if the value is valid, or an error message if the value is not. It also has an optional `Func<string>` parameter for getting the error message.

The `TryReadValue` function is used to read values from the database. It takes a `ReadWrite` object, a `string` prompt, and a predicate for determining if the value is valid. It returns the value if the read is successful, or the specified error if the read is not.


```
using System.Diagnostics.CodeAnalysis;
using static King.Resources.Resource;

namespace King;

internal static class IOExtensions
{
    internal static bool TryReadGameData(this IReadWrite io, IRandom random, [NotNullWhen(true)] out Reign? reign)
    {
        if (io.TryReadValue(SavedYearsPrompt, v => v < Reign.MaxTerm, SavedYearsError(Reign.MaxTerm), out var years) &&
            io.TryReadValue(SavedTreasuryPrompt, out var rallods) &&
            io.TryReadValue(SavedCountrymenPrompt, out var countrymen) &&
            io.TryReadValue(SavedWorkersPrompt, out var workers) &&
            io.TryReadValue(SavedLandPrompt, v => v is > 1000 and <= 2000, SavedLandError, out var land))
        {
            reign = new Reign(io, random, new Country(io, random, rallods, countrymen, workers, land), years + 1);
            return true;
        }

        reign = default;
        return false;
    }

    internal static bool TryReadValue(this IReadWrite io, string prompt, out float value, params ValidityTest[] tests)
    {
        while (true)
        {
            var response = value = io.ReadNumber(prompt);
            if (response == 0) { return false; }
            if (tests.All(test => test.IsValid(response, io))) { return true; }
        } 
    }

    internal static bool TryReadValue(this IReadWrite io, string prompt, out float value)
        => io.TryReadValue(prompt, _ => true, "", out value);
    
    internal static bool TryReadValue(
        this IReadWrite io,
        string prompt,
        Predicate<float> isValid,
        string error,
        out float value)
        => io.TryReadValue(prompt, isValid, () => error, out value);

    internal static bool TryReadValue(
        this IReadWrite io,
        string prompt,
        Predicate<float> isValid,
        Func<string> getError,
        out float value)
    {
        while (true)
        {
            value = io.ReadNumber(prompt);
            if (value < 0) { return false; }
            if (isValid(value)) { return true; }
            
            io.Write(getError());
        }
    }
}

```

# `53_King/csharp/Program.cs`

这段代码的作用是创建一个名为 "Game" 的类，其中包含几个全局变量：

1. using Games.Common.IO;
2. using Games.Common.Randomness;
3. using King.Resources;
4. using static King.Resources.Resource;

第一个新创建的类 called "Game" 中包含一个名为 "new Game" 的方法，该方法接收两个参数：

 1. new ConsoleIO();
 2. new RandomNumberGenerator();

这些参数创建一个新的 instance of the "ConsoleIO" 和 "RandomNumberGenerator" 类，并将其存储在名为 "IO" 和 "RandomNumberGenerator" 的全局变量中。

该代码接下来导入了 King.Resources 命名空间，该命名空间中包含一个名为 "Resource" 的类。然后，通过使用 "using" 语句，在 "Game" 类中声明一个名为 "King.Resources.Resource" 的类型别名。

最后，该代码调用了名为 "Game.Play" 的方法，该方法接收一个空字符串作为参数，并在控制台输出 "Hello, World!"。


```
global using Games.Common.IO;
global using Games.Common.Randomness;
global using King.Resources;
global using static King.Resources.Resource;
using King;

new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `53_King/csharp/Reign.cs`



这段代码定义了一个名为Reign的类，旨在模拟在一个虚拟王国的游戏中，玩家能否在一年内成功结束游戏。

Reign类包含以下成员变量：

- IReadWrite：用于读写游戏数据的接口。
- IRandom：用于生成随机数的接口。
- Country：用于获取游戏世界的国家的接口。
- float：用于存储游戏世界当前年份的浮点数。

Reign类包含以下构造函数：

- Reign(IReadWrite io, IRandom random, Country country, float year)：初始化一个Reign对象，使用指定的IO、随机数生成器和游戏世界的国家来初始化对象。
- Reign(IReadWrite io, IRandom random, Country country)：初始化一个Reign对象，使用指定的IO、随机数生成器和游戏世界的国家来初始化对象，并指定一个年份。

Reign类包含一个名为PlayYear的方法，用于尝试玩一年游戏，如果游戏结束，则返回false，否则继续游戏。

- Reign.PlayYear()方法使用Reign类构造函数中的参数，分别初始化IO、随机数生成器和游戏世界的国家，并尝试玩一年游戏。使用Reign类的方法IsAtEndOfTerm()来检查游戏是否结束，如果游戏结束，则返回false，否则继续游戏。


```
namespace King;

internal class Reign
{
    public const int MaxTerm = 8;

    private readonly IReadWrite _io;
    private readonly IRandom _random;
    private readonly Country _country;
    private float _yearNumber;

    public Reign(IReadWrite io, IRandom random)
        : this(io, random, new Country(io, random), 1)
    {
    }

    public Reign(IReadWrite io, IRandom random, Country country, float year)
    {
        _io = io;
        _random = random;
        _country = country;
        _yearNumber = year;
    }

    public bool PlayYear()
    {
        var year = new Year(_country, _random, _io);

        _io.Write(year.Status);

        var result = year.GetPlayerActions() ?? year.EvaluateResults() ?? IsAtEndOfTerm();
        if (result.IsGameOver)
        {
            _io.WriteLine(result.Message);
            return false;
        }

        return true;
    }

    private Result IsAtEndOfTerm() 
        => _yearNumber == MaxTerm 
            ? Result.GameOver(EndCongratulations(MaxTerm)) 
            : Result.Continue;
}

```

# `53_King/csharp/Result.cs`

这段代码定义了一个名为 `Result` 的内部结构体，它包含两个属性：`IsGameOver` 和 `Message`。

同时，还定义了一个名为 `GameOver` 的内部函数，它接受一个字符串参数并返回一个名为 `Result` 的内部结构体，其中 `IsGameOver` 属性为 `true`，`Message` 属性为传递给该函数的字符串。

另外，还定义了一个名为 `Continue` 的内部函数，它接受一个字符串参数并返回一个名为 `Result` 的内部结构体，其中 `IsGameOver` 属性为 `false`，`Message` 属性为传递给该函数的字符串，该字符串常量为 `""`。

最后，在 `namespace King` 的命名空间中声明了这些内部函数和结构体。


```
namespace King;

internal record struct Result (bool IsGameOver, string Message)
{
    internal static Result GameOver(string message) => new(true, message);
    internal static Result Continue => new(false, "");
}

```

# `53_King/csharp/ValidityTest.cs`



这段代码定义了一个名为 "ValidityTest" 的内部类，其作用是验证传入的参数 "value" 是否符合 "float" 类型的要求，如果不符合，则输出一个错误消息，否则返回 true。

具体来说，代码中定义了一个 "isValid" 私有成员变量，其类型为 "Predicate<float>"，即一个预测值为 "float" 的条件判断类型。接着，定义了一个 "getError" 私有成员函数，其类型为 "Func<string>"，即一个可以将 "错误消息" 打印出来的函数类型。

在 "ValidityTest" 的构造函数中，第一个参数是一个 "Predicate<float>" 和一个 "string" 类型的参数，分别作为 "isValid" 和 "getError" 函数的参数。第二个参数则是传递给构造函数的第二个参数，即 "error" 的参数。

在 "ValidityTest" 的 "IsValid" 方法中，首先检查传入的参数 "value" 是否符合 "float" 类型的要求，如果是，则返回 true，否则调用 "getError" 函数将一个错误消息打印出来，并将返回结果设置为 false。最后，在 "ValidityTest" 的 "isValid" 方法中，使用 Predicate 类型来验证参数是否符合 "float" 类型，如果不符合，则返回 false。


```
namespace King;

internal class ValidityTest
{
    private readonly Predicate<float> _isValid;
    private readonly Func<string> _getError;

    public ValidityTest(Predicate<float> isValid, string error)
        : this(isValid, () => error)
    {
    }

    public ValidityTest(Predicate<float> isValid, Func<string> getError)
    {
        _isValid = isValid;
        _getError = getError;
    }

    public bool IsValid(float value, IReadWrite io)
    {
        if (_isValid(value)) { return true; }
        
        io.Write(_getError());
        return false;
    }
}
```

# `53_King/csharp/Year.cs`

This is a programming language game written in Csharp. It appears to be a simulation game where the player must manage a virtual country with various resources and industries. The player must also manage tourism and starvation. The game has different outcomes depending on the decisions the player makes.

The game has several functions such as `_country.IndustryLand`, `_migration`, `_random`, `EndManyDead`, `EndOneThirdDead`, `EndMoneyLeftOver`, `_starvationDeaths`, `_workers`, `_country.Countrymen`, `EndOneFifthDead`, `EndAllDead`, `GameOver`, `EndThroughDead`, `EndTerrorism`.

It appears that the game has multiple possible outcomes for each function that the player can choose. It also appears that the game has different levels of difficulty and the player must also manage the game's economy.


```
using System.Text;

namespace King;

internal class Year
{
    private readonly Country _country;
    private readonly IRandom _random;
    private readonly IReadWrite _io;
    private readonly int _plantingCost;
    private readonly int _landValue;

    private float _landSold;
    private float _rallodsDistributed;
    private float _landPlanted;
    private float _pollutionControlCost;

    private float _citizenSupport;
    private int _deaths;
    private float _starvationDeaths;
    private int _pollutionDeaths;
    private int _migration;

    public Year(Country country, IRandom random, IReadWrite io)
    {
        _country = country;
        _random = random;
        _io = io;
        
        _plantingCost = random.Next(10, 15);
        _landValue = random.Next(95, 105);
    }

    public string Status => _country.GetStatus(_landValue, _plantingCost);

    public Result? GetPlayerActions()
    {
        var playerSoldLand = _country.SellLand(_landValue, out _landSold);
        var playerDistributedRallods = _country.DistributeRallods(out _rallodsDistributed);
        var playerPlantedLand = _country.HasRallods && _country.PlantLand(_plantingCost, out _landPlanted);
        var playerControlledPollution = _country.HasRallods && _country.ControlPollution(out _pollutionControlCost);

        return playerSoldLand || playerDistributedRallods || playerPlantedLand || playerControlledPollution
            ? null
            : Result.GameOver(Goodbye);
    }

    public Result? EvaluateResults()
    {
        var rallodsUnspent = _country.Rallods;

        _io.WriteLine();
        _io.WriteLine();

        return EvaluateDeaths() 
            ?? EvaluateMigration() 
            ?? EvaluateAgriculture()
            ?? EvaluateTourism()
            ?? DetermineResult(rallodsUnspent);
    }

    public Result? EvaluateDeaths()
    {
        var supportedCountrymen = _rallodsDistributed / 100;
        _citizenSupport = supportedCountrymen - _country.Countrymen;
        _starvationDeaths = -_citizenSupport;
        if (_starvationDeaths > 0)
        {
            if (supportedCountrymen < 50) { return Result.GameOver(EndOneThirdDead(_random)); }
            _io.WriteLine(DeathsStarvation(_starvationDeaths));
        }

        var pollutionControl = _pollutionControlCost >= 25 ? _pollutionControlCost / 25 : 1;
        _pollutionDeaths = (int)(_random.Next((int)_country.IndustryLand) / pollutionControl);
        if (_pollutionDeaths > 0)
        {
            _io.WriteLine(DeathsPollution(_pollutionDeaths));
        }

        _deaths = (int)(_starvationDeaths + _pollutionDeaths);
        if (_deaths > 0)
        {
            var funeralCosts = _deaths * 9;
            _io.WriteLine(FuneralExpenses(funeralCosts));

            if (!_country.TrySpend(funeralCosts, _landValue))
            {
                _io.WriteLine(InsufficientReserves);
            }

            _country.RemoveTheDead(_deaths);
        }

        return null;
    }

    private Result? EvaluateMigration()
    {
        if (_landSold > 0)
        {
            var newWorkers = (int)(_landSold + _random.NextFloat(10) - _random.NextFloat(20));
            if (!_country.HasWorkers) { newWorkers += 20; }
            _io.Write(WorkerMigration(newWorkers));
            _country.AddWorkers(newWorkers);
        }

        _migration = 
            (int)(_citizenSupport / 10 + _pollutionControlCost / 25 - _country.IndustryLand / 50 - _pollutionDeaths / 2);
        _io.WriteLine(Migration(_migration));
        _country.Migration(_migration);

        return null;
    }

    private Result? EvaluateAgriculture()
    {
        var ruinedCrops = (int)Math.Min(_country.IndustryLand * (_random.NextFloat() + 1.5f) / 2, _landPlanted);
        var yield = (int)(_landPlanted - ruinedCrops);
        var income = (int)(yield * _landValue / 2f);

        _io.Write(LandPlanted(_landPlanted));
        _io.Write(Harvest(yield, income, _country.IndustryLand > 0));

        _country.SellCrops(income);

        return null;
    }

    private Result? EvaluateTourism()
    {
        var reputationValue = (int)((_country.Countrymen - _migration) * 22 + _random.NextFloat(500));
        var industryAdjustment = (int)(_country.IndustryLand * 15);
        var tourismIncome = Math.Abs(reputationValue - industryAdjustment);

        _io.WriteLine(TourismEarnings(tourismIncome));
        if (industryAdjustment > 0 && tourismIncome < _country.PreviousTourismIncome)
        {
            _io.Write(TourismDecrease(_random));
        }

        _country.EntertainTourists(tourismIncome);

        return null;
    }

    private Result? DetermineResult(float rallodsUnspent)
    {
        if (_deaths > 200) { return Result.GameOver(EndManyDead(_deaths, _random)); }
        if (_country.Countrymen < 343) { return Result.GameOver(EndOneThirdDead(_random)); }
        if (rallodsUnspent / 100 > 5 && _starvationDeaths >= 2) { return Result.GameOver(EndMoneyLeftOver()); }
        if (_country.Workers > _country.Countrymen) { return Result.GameOver(EndForeignWorkers(_random)); }
        return null;
    }
}

```

# `53_King/csharp/Resources/Resource.cs`

This is a TypeScript class that defines a `Switch` that takes a `string` as input and returns a string based on the input value.

The class defines several methods that use the `Switch` to apply the appropriate action to each input value. For example, the `EndCongratulations` method takes the input value and returns the string "Congratulations" if the input value is less than or equal to 3, otherwise it returns the string "Goodbye".

The `EndForeignWorkers` method takes the input value and returns the string "Goodbye" if the input value is less than or equal to 3, otherwise it returns the string "EndForeignWorkers".

The `EndManyDead` method takes the input value and returns the string "Goodbye" if the input value is less than or equal to 2, otherwise it returns the string "EndManyDead".

The `EndMoneyLeftOver` method does not take any input value and always returns the string "Goodbye".

The `EndOneThirdDead` method takes the input value and returns the string "Goodbye" if the input value is less than or equal to 1, otherwise it returns the string "EndOneThirdDead".

The `SavedYearsPrompt`, `SavedYearsError`, `SavedTreasuryPrompt`, `SavedCountrymenPrompt`, `SavedWorkersPrompt`, `SavedLandPrompt`, `SavedLandError` and `GetString` methods do not take any input value and always return a string based on the name of the method.


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace King.Resources;

internal static class Resource
{
    private static bool _sellLandErrorShown;

    public static Stream Title => GetStream();
    
    public static string InstructionsPrompt => GetString();
    public static string InstructionsText(int years) => string.Format(GetString(), years);

    public static string Status(
        float rallods,
        float countrymen,
        float workers,
        float land,
        float landValue,
        float plantingCost)
        => string.Format(
            workers == 0 ? StatusWithWorkers : StatusSansWorkers,
            rallods,
            (int)countrymen,
            (int)workers,
            (int)land,
            landValue,
            plantingCost);

    private static string StatusWithWorkers => GetString();
    private static string StatusSansWorkers => GetString();

    public static string SellLandPrompt => GetString();
    public static string SellLandError(float farmLand)
    {
        var error = string.Format(GetString(), farmLand, _sellLandErrorShown ? "" : SellLandErrorReason);
        _sellLandErrorShown = true;
        return error;
    }
    private static string SellLandErrorReason => GetString();

    public static string GiveRallodsPrompt => GetString();
    public static string GiveRallodsError(float rallods) => string.Format(GetString(), rallods);

    public static string PlantLandPrompt => GetString();
    public static string PlantLandError1 => GetString();
    public static string PlantLandError2(float farmLand) => string.Format(GetString(), farmLand);
    public static string PlantLandError3(float rallods) => string.Format(GetString(), rallods);

    public static string PollutionPrompt => GetString();
    public static string PollutionError(float rallods) => string.Format(GetString(), rallods);

    public static string DeathsStarvation(float deaths) => string.Format(GetString(), (int)deaths);
    public static string DeathsPollution(int deaths) => string.Format(GetString(), deaths);
    public static string FuneralExpenses(int expenses) => string.Format(GetString(), expenses);
    public static string InsufficientReserves => GetString();

    public static string WorkerMigration(int newWorkers) => string.Format(GetString(), newWorkers);
    public static string Migration(int migration) 
        => string.Format(migration < 0 ? Emigration : Immigration, Math.Abs(migration));
    public static string Emigration => GetString();
    public static string Immigration => GetString();

    public static string LandPlanted(float landPlanted) 
        => landPlanted > 0 ? string.Format(GetString(), (int)landPlanted) : "";
    public static string Harvest(int yield, int income, bool hasIndustry) 
        => string.Format(GetString(), yield, HarvestReason(hasIndustry), income);
    private static string HarvestReason(bool hasIndustry) => hasIndustry ? GetString() : "";

    public static string TourismEarnings(int income) => string.Format(GetString(), income);
    public static string TourismDecrease(IRandom random) => string.Format(GetString(), TourismReason(random));
    private static string TourismReason(IRandom random) => GetStrings()[random.Next(5)];

    private static string EndAlso(IRandom random)
        => random.Next(10) switch
        {
            <= 3 => GetStrings()[0],
            <= 6 => GetStrings()[1],
            _ => GetStrings()[2]
        };

    public static string EndCongratulations(int termLength) => string.Format(GetString(), termLength);
    private static string EndConsequences(IRandom random) => GetStrings()[random.Next(2)];
    public static string EndForeignWorkers(IRandom random) => string.Format(GetString(), EndConsequences(random));
    public static string EndManyDead(int deaths, IRandom random) => string.Format(GetString(), deaths, EndAlso(random));
    public static string EndMoneyLeftOver() => GetString();
    public static string EndOneThirdDead(IRandom random) => string.Format(GetString(), EndConsequences(random));
    
    public static string SavedYearsPrompt => GetString();
    public static string SavedYearsError(int years) => string.Format(GetString(), years);
    public static string SavedTreasuryPrompt => GetString();
    public static string SavedCountrymenPrompt => GetString();
    public static string SavedWorkersPrompt => GetString();
    public static string SavedLandPrompt => GetString();
    public static string SavedLandError => GetString();

    public static string Goodbye => GetString();

    private static string[] GetStrings([CallerMemberName] string? name = null) => GetString(name).Split(';');

    private static string GetString([CallerMemberName] string? name = null)
    {
        using var stream = GetStream(name);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    private static Stream GetStream([CallerMemberName] string? name = null) =>
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `53_King/javascript/king.js`

这段代码定义了两个函数，分别是`print`函数和`input`函数。

`print`函数的作用是在页面上打印一段字符串，接收一个字符串参数，并将该参数附加给一个由`document.getElementById("output")`创建的`text`元素的`appendChild`方法。

`input`函数的作用是从用户接收一个字符串，接收用户输入后返回一个Promise对象，解决后进入该函数内部执行一些操作，然后返回用户输入的字符串。

具体来说，`input`函数创建了一个`<input>`元素，设置了一些基本的属性，如`type="text"`，`length="50"`，并将其添加到了页面上由`document.getElementById("output")`创建的元素中，该元素在函数内部具有`appendChild`方法。然后，函数使用`input`元素的`addEventListener`方法监听键盘事件，当事件处理程序捕获到事件时，函数内部的`<keydown>`事件处理程序会获取当前事件，如果事件处理程序捕获到的按键是`keyCode=13`，则函数内部的`input`字符串将被赋值给`input_str`，并将结果打印到页面上，并删除`<input>`元素。之后，函数内部使用`Promise`对象解决，并在解决后返回用户的输入字符串。


```
// KING
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

这两段代码分别是JavaScript中的一个函数和一个全局函数，其作用各不相同。

1. `function tab(space)` 是一个内部函数，被称为 `hate_your_guts` 函数。其作用是在一个字符串变量 `str` 中添加 `space` 个空格，并在末尾添加一个空格。这通常用于在代码中输出一些额外的信息，例如行距或者错误消息。

2. `function hate_your_guts()` 是全局函数，其作用是在控制台输出一行字符。具体来说，它会输出两行字符，然后输出 "OVER ONE THIRD OF THE POPULATION HAS DIED SINCE YOU" 和 "WERE ELECTED TO OFFICE. THE PEOPLE (REMAINING)" 这两行。最后两行字符串中，每行都有一个感叹号，其效果类似于 "炸雷"。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

function hate_your_guts()
{
    print("\n");
    print("\n");
    print("OVER ONE THIRD OF THE POPULATION HAS DIED SINCE YOU\n");
    print("WERE ELECTED TO OFFICE. THE PEOPLE (REMAINING)\n");
    print("HATE YOUR GUTS.\n");
}

```

It looks like you have provided a sample code for a game that simulates the experience of being an office worker. The game allows the player to choose between committing suicide or resigning. If the player chooses to resign, the game prints a message asking them to turn off their computer before proceeding. If the player chooses to commit suicide, the game prints a message explaining that the number of foreign workers has exceeded the number of countrymen and advising them to turn off their computer before proceeding. If the player completes their office term, the game prints a message and恭喜 them. Otherwise, the game prints a message indicating that they have been fired and are now living in prison. The game also includes a random event that can randomly throw the player out of office.

Overall, the game seems to be a serious and thought-provoking representation of the reality of being an office worker. It is important to remember that suicide is a serious issue that should not be taken lightly and should always be an option for those who are struggling with mental health problems. It is also important to recognize that some people may be experiencing discrimination and prejudice, and it is important to stand up against such actions.


```
// Main program
async function main()
{
    print(tab(34) + "KING\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("DO YOU WANT INSTRUCTIONS");
    str = await input();
    n5 = 8;
    if (str == "AGAIN") {
        while (1) {
            print("HOW MANY YEARS HAD YOU BEEN IN OFFICE WHEN INTERRUPTED");
            x5 = parseInt(await input());
            if (x5 == 0)
                return;
            if (x5 < 8)
                break;
            print("   COME ON, YOUR TERM IN OFFICE IS ONLY " + n5 + " YEARS.\n");
        }
        print("HOW MUCH DID YOU HAVE IN THE TREASURY");
        a = parseInt(await input());
        if (a < 0)
            return;
        print("HOW MANY COUNTRYMEN");
        b = parseInt(await input());
        if (b < 0)
            return;
        print("HOW MANY WORKERS");
        c = parseInt(await input());
        if (c < 0)
            return;
        while (1) {
            print("HOW MANY SQUARE MILES OF LAND");
            d = parseInt(await input());
            if (d < 0)
                return;
            if (d > 1000 && d <= 2000)
                break;
            print("   COME ON, YOU STARTED WITH 1000 SQ. MILES OF FARM LAND\n");
            print("   AND 10,000 SQ. MILES OF FOREST LAND.\n");
        }
    } else {
        if (str.substr(0, 1) != "N") {
            print("\n");
            print("\n");
            print("\n");
            print("CONGRATULATIONS! YOU'VE JUST BEEN ELECTED PREMIER OF SETATS\n");
            print("DETINU, A SMALL COMMUNIST ISLAND 30 BY 70 MILES LONG. YOUR\n");
            print("JOB IS TO DECIDE UPON THE CONTRY'S BUDGET AND DISTRIBUTE\n");
            print("MONEY TO YOUR COUNTRYMEN FROM THE COMMUNAL TREASURY.\n");
            print("THE MONEY SYSTEM IS RALLODS, AND EACH PERSON NEEDS 100\n");
            print("RALLODS PER YEAR TO SURVIVE. YOUR COUNTRY'S INCOME COMES\n");
            print("FROM FARM PRODUCE AND TOURISTS VISITING YOUR MAGNIFICENT\n");
            print("FORESTS, HUNTING, FISHING, ETC. HALF YOUR LAND IS FARM LAND\n");
            print("WHICH ALSO HAS AN EXCELLENT MINERAL CONTENT AND MAY BE SOLD\n");
            print("TO FOREIGN INDUSTRY (STRIP MINING) WHO IMPORT AND SUPPORT\n");
            print("THEIR OWN WORKERS. CROPS COST BETWEEN 10 AND 15 RALLODS PER\n");
            print("SQUARE MILE TO PLANT.\n");
            print("YOUR GOAL IS TO COMPLETE YOUR " + n5 + " YEAR TERM OF OFFICE.\n");
            print("GOOD LUCK!\n");
        }
        print("\n");
        a = Math.floor(60000 + (1000 * Math.random()) - (1000 * Math.random()));
        b = Math.floor(500 + (10 * Math.random()) - (10 * Math.random()));
        c = 0;
        d = 2000;
        x5 = 0;
    }
    v3 = 0;
    b5 = 0;
    x = false;
    while (1) {
        w = Math.floor(10 * Math.random() + 95);
        print("\n");
        print("YOU NOW HAVE " + a + " RALLODS IN THE TREASURY.\n");
        print(b + " COUNTRYMEN, ");
        v9 = Math.floor(((Math.random() / 2) * 10 + 10));
        if (c != 0)
            print(c + " FOREIGN WORKERS, ");
        print("AND " + Math.floor(d) + " SQ. MILES OF LAND.\n");
        print("THIS YEAR INDUSTRY WILL BUY LAND FOR " + w + " ");
        print("RALLODS PER SQUARE MILE.\n");
        print("LAND CURRENTLY COSTS " + v9 + " RALLODS PER SQUARE MILE TO PLANT.\n");
        print("\n");
        while (1) {
            print("HOW MANY SQUARE MILES DO YOU WISH TO SELL TO INDUSTRY");
            h = parseInt(await input());
            if (h < 0)
                continue;
            if (h <= d - 1000)
                break;
            print("***  THINK AGAIN. YOU ONLY HAVE " + (d - 1000) + " SQUARE MILES OF FARM LAND.\n");
            if (x == false) {
                print("\n");
                print("(FOREIGN INDUSTRY WILL ONLY BUY FARM LAND BECAUSE\n");
                print("FOREST LAND IS UNECONOMICAL TO STRIP MINE DUE TO TREES,\n");
                print("THICKER TOP SOIL, ETC.)\n");
                x = true;
            }
        }
        d = Math.floor(d - h);
        a = Math.floor(a + (h * w));
        while (1) {
            print("HOW MANY RALLODS WILL YOU DISTRIBUTE AMONG YOUR COUNTRYMEN");
            i = parseInt(await input());
            if (i < 0)
                continue;
            if (i < a)
                break;
            if (i == a) {
                j = 0;
                k = 0;
                a = 0;
                break;
            }
            print("   THINK AGAIN. YOU'VE ONLY " + a + " RALLODS IN THE TREASURY\n");
        }
        if (a) {
            a = Math.floor(a - i);
            while (1) {
                print("HOW MANY SQUARE MILES DO YOU WISH TO PLANT");
                j = parseInt(await input());
                if (j < 0)
                    continue;
                if (j <= b * 2) {
                    if (j <= d - 1000) {
                        u1 = Math.floor(j * v9);
                        if (u1 > a) {
                            print("   THINK AGAIN. YOU'VE ONLY " + a + " RALLODS LEFT IN THE TREASURY.\n");
                            continue;
                        } else if (u1 == a) {
                            k = 0;
                            a = 0;
                        }
                        break;
                    }
                    print("   SORRY, BUT YOU'VE ONLY " + (d - 1000) + " SQ. MILES OF FARM LAND.\n");
                    continue;
                }
                print("   SORRY, BUT EACH COUNTRYMAN CAN ONLY PLANT 2 SQ. MILES.\n");
            }
        }
        if (a) {
            a -= u1;
            while (1) {
                print("HOW MANY RALLODS DO YOU WISH TO SPEND ON POLLUTION CONTROL");
                k = parseInt(await input());
                if (k < 0)
                    continue;
                if (k <= a)
                    break;
                print("   THINK AGAIN. YOU ONLY HAVE " + a + " RALLODS REMAINING.\n");
            }
        }
        if (h == 0 && i == 0 && j == 0 && k == 0) {
            print("GOODBYE.\n");
            print("(IF YOU WISH TO CONTINUE THIS GAME AT A LATER DATE, ANSWER\n");
            print("'AGAIN' WHEN ASKED IF YOU WANT INSTRUCTIONS AT THE START\n");
            print("OF THE GAME).\n");
            return;
        }
        print("\n");
        print("\n");
        a = Math.floor(a - k);
        a4 = a;
        if (Math.floor(i / 100 - b) < 0) {
            if (i / 100 < 50) {
                hate_your_guts();
                break;
            }
            print(Math.floor(b - (i / 100)) + " COUNTRYMEN DIED OF STARVATION\n");
        }
        f1 = Math.floor(Math.random() * (2000 - d));
        if (k >= 25)
            f1 = Math.floor(f1 / (k / 25));
        if (f1 > 0)
            print(f1 + " COUNTRYMEN DIED OF CARBON-MONOXIDE AND DUST INHALATION\n");
        funeral = false;
        if (Math.floor((i / 100) - b) >= 0) {
            if (f1 > 0) {
                print("   YOU WERE FORCED TO SPEND " + Math.floor(f1 * 9) + " RALLODS ON ");
                print("FUNERAL EXPENSES.\n");
                b5 = f1;
                a = Math.floor(a - (f1 * 9));
                funeral = true;
            }
        } else {
            print("   YOU WERE FORCED TO SPEND " + Math.floor((f1 + (b - (i / 100))) * 9));
            print(" RALLODS ON FUNERAL EXPENSES.\n");
            b5 = Math.floor(f1 + (b - (i / 100)));
            a = Math.floor(a - ((f1 + (b - (i / 100))) * 9));
            funeral = true;
        }
        if (funeral) {
            if (a < 0) {
                print("   INSUFFICIENT RESERVES TO COVER COST - LAND WAS SOLD\n");
                d = Math.floor(d + (a / w));
                a = 0;
            }
            b = Math.floor(b - b5);
        }
        c1 = 0;
        if (h != 0) {
            c1 = Math.floor(h + (Math.random() * 10) - (Math.random() * 20));
            if (c <= 0)
                c1 += 20;
            print(c1 + " WORKERS CAME TO THE COUNTRY AND ");
        }
        p1 = Math.floor(((i / 100 - b) / 10) + (k / 25) - ((2000 - d) / 50) - (f1 / 2));
        print(Math.abs(p1) + " COUNTRYMEN ");
        if (p1 >= 0)
            print("CAME TO");
        else
            print("LEFT");
        print(" THE ISLAND.\n");
        b = Math.floor(b + p1);
        c = Math.floor(c + c1);
        u2 = Math.floor(((2000 - d) * ((Math.random() + 1.5) / 2)));
        if (c != 0) {
            print("OF " + Math.floor(j) + " SQ. MILES PLANTED,");
        }
        if (j <= u2)
            u2 = j;
        print(" YOU HARVESTED " + Math.floor(j - u2) + " SQ. MILES OF CROPS.\n");
        if (u2 != 0 && t1 < 2) {
            print("   (DUE TO ");
            if (t1 != 0)
                print("INCREASED ");
            print("AIR AND WATER POLLUTION FROM FOREIGN INDUSTRY.)\n");
        }
        q = Math.floor((j - u2) * (w / 2));
        print("MAKING " + q + " RALLODS.\n");
        a = Math.floor(a + q);
        v1 = Math.floor(((b - p1) * 22) + (Math.random() * 500));
        v2 = Math.floor((2000 - d) * 15);
        print(" YOU MADE " + Math.abs(Math.floor(v1 - v2)) + " RALLODS FROM TOURIST TRADE.\n");
        if (v2 != 0 && v1 - v2 < v3) {
            print("   DECREASE BECAUSE ");
            g1 = 10 * Math.random();
            if (g1 <= 2)
                print("FISH POPULATION HAS DWINDLED DUE TO WATER POLLUTION.\n");
            else if (g1 <= 4)
                print("AIR POLLUTION IS KILLING GAME BIRD POPULATION.\n");
            else if (g1 <= 6)
                print("MINERAL BATHS ARE BEING RUINED BY WATER POLLUTION.\n");
            else if (g1 <= 8)
                print("UNPLEASANT SMOG IS DISCOURAGING SUN BATHERS.\n");
            else if (g1 <= 10)
                print("HOTELS ARE LOOKING SHABBY DUE TO SMOG GRIT.\n");
        }
        v3 = Math.floor(a + v3);    // Probable bug from original game
        a = Math.floor(a + v3);
        if (b5 > 200) {
            print("\n");
            print("\n");
            print(b5 + " COUNTRYMEN DIED IN ONE YEAR!!!!!\n");
            print("DUE TO THIS EXTREME MISMANAGEMENT, YOU HAVE NOT ONLY\n");
            print("BEEN IMPEACHED AND THROWN OUT OF OFFICE, BUT YOU\n");
            m6 = Math.floor(Math.random() * 10);
            if (m6 <= 3)
                print("ALSO HAD YOUR LEFT EYE GOUGED OUT!\n");
            else if (m6 <= 6)
                print("HAVE ALSO GAINED A VERY BAD REPUTATION.\n");
            else
                print("HAVE ALSO BEEN DECLARED NATIONAL FINK.\n");
            print("\n");
            print("\n");
            return;
        }
        if (b < 343) {
            hate_your_guts();
            break;
        }
        if (a4 / 100 > 5 && b5 - f1 >= 2) {
            print("\n");
            print("MONEY WAS LEFT OVER IN THE TREASURY WHICH YOU DID\n");
            print("NOT SPEND. AS A RESULT, SOME OF YOUR COUNTRYMEN DIED\n");
            print("OF STARVATION. THE PUBLIC IS ENRAGED AND YOU HAVE\n");
            print("BEEN FORCED TO EITHER RESIGN OR COMMIT SUICIDE.\n");
            print("THE CHOICE IS YOURS.\n");
            print("IF YOU CHOOSE THE LATTER, PLEASE TURN OFF YOUR COMPUTER\n");
            print("BEFORE PROCEEDING.\n");
            print("\n");
            print("\n");
            return;
        }
        if (c > b) {
            print("\n");
            print("\n");
            print("THE NUMBER OF FOREIGN WORKERS HAS EXCEEDED THE NUMBER\n");
            print("OF COUNTRYMEN. AS A MINORITY, THEY HAVE REVOLTED AND\n");
            print("TAKEN OVER THE COUNTRY.\n");
            break;
        }
        if (n5 - 1 == x5) {
            print("\n");
            print("\n");
            print("CONGRATULATIONS!!!!!!!!!!!!!!!!!!\n");
            print("YOU HAVE SUCCESFULLY COMPLETED YOUR " + n5 + " YEAR TERM\n");
            print("OF OFFICE. YOU WERE, OF COURSE, EXTREMELY LUCKY, BUT\n");
            print("NEVERTHELESS, IT'S QUITE AN ACHIEVEMENT. GOODBYE AND GOOD\n");
            print("LUCK - YOU'LL PROBABLY NEED IT IF YOU'RE THE TYPE THAT\n");
            print("PLAYS THIS GAME.\n");
            print("\n");
            print("\n");
            return;
        }
        x5++;
        b5 = 0;
    }
    if (Math.random() <= 0.5) {
        print("YOU HAVE BEEN ASSASSINATED.\n");
    } else {
        print("YOU HAVE BEEN THROWN OUT OF OFFICE AND ARE NOW\n");
        print("RESIDING IN PRISON.\n");
    }
    print("\n");
    print("\n");
}

```

这道题是一个简单的C语言程序，包含了两个主要部分：`main()`函数和`printf()`函数。我们需要深入了解这两部分的功能，才能完整地解释这段代码的作用。

1. `main()`函数：

`main()`函数是程序的入口点，程序从这里开始执行。在这个函数中，首先定义了一个没有参数的函数，即`void`类型的`main()`函数。然后，定义了一个整数变量`i`，并将其赋值为256。接下来，定义了一个`printf()`函数，用于输出字符串`Hello World`。

2. `printf()`函数：

`printf()`函数是一个标准库函数，用于在屏幕上打印字符串。`printf()`函数的第一个参数是一个字符串，后继的字符用`%s`替换。例如，`printf()`函数的第一个参数是`'%s'`，第二个参数是`'%s'`，那么它会在屏幕上打印字符串`"Hello World"`。

综合以上分析，这段代码的作用是：输出字符串`"Hello World"`。当你或你的程序在运行这段代码时，`main()`函数会先执行`printf()`函数，然后`printf()`函数会在屏幕上打印`"Hello World"`这个字符串。


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


# `53_King/python/king.py`

这段代码定义了一个名为"Forest_Land"的类，表示一片森林。这个类有以下属性和方法：

- **FOREST_LAND**: 这是一个整数，表示一片森林的初始面积。
- **INITIAL_LANDS**: 这是一个整数，表示除了一片森林之外的土地的初始面积。
- **COST_OF_LIVING**: 这是一个整数，表示每个人每天的食品消耗成本。

此外，代码中还有一段注释，指出这是一份用于计算年金的公式。这个公式计算出每年从一片森林中获得的收益，用于 simulate 游戏中玩家每年的生存和扩张。


```
"""
KING

A strategy game where the player is the king.

Ported to Python by Martin Thoma in 2022
"""

import sys
from dataclasses import dataclass
from random import randint, random

FOREST_LAND = 1000
INITIAL_LAND = FOREST_LAND + 1000
COST_OF_LIVING = 100
```

This is a Python game where the player is given the option to either spend or save money. The game has different outcomes if the player spends or saves the money. The game also has a mechanism for handling too many foreign workers and ends when the player is either assassinated or thrown out of office.


```
COST_OF_FUNERAL = 9
YEARS_IN_TERM = 8
POLLUTION_CONTROL_FACTOR = 25


def ask_int(prompt) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            continue


@dataclass
class GameState:
    rallods: int = -1
    countrymen: int = -1
    land: int = INITIAL_LAND
    foreign_workers: int = 0
    years_in_office: int = 0

    # previous year stats
    crop_loss_last_year: int = 0

    # current year stats
    died_contrymen: int = 0
    pollution_deaths: int = 0
    population_change: int = 0

    # current year - market situation (in rallods per square mile)
    planting_cost: int = -1
    land_buy_price: int = -1

    tourism_earnings: int = 0

    def set_market_conditions(self) -> None:
        self.land_buy_price = randint(95, 105)
        self.planting_cost = randint(10, 15)

    @property
    def farmland(self) -> int:
        return self.land - FOREST_LAND

    @property
    def settled_people(self) -> int:
        return self.countrymen - self.population_change

    def sell_land(self, amount: int) -> None:
        assert amount <= self.farmland
        self.land -= amount
        self.rallods += self.land_buy_price * amount

    def distribute_rallods(self, distribute: int) -> None:
        self.rallods -= distribute

    def spend_pollution_control(self, spend: int) -> None:
        self.rallods -= spend

    def plant(self, sq_to_plant: int) -> None:
        self.rallods -= sq_to_plant * self.planting_cost

    def print_status(self) -> None:
        print(f"\n\nYOU NOW HAVE {self.rallods} RALLODS IN THE TREASURY.")
        print(f"{int(self.countrymen)} COUNTRYMEN, ", end="")
        if self.foreign_workers > 0:
            print(f"{int(self.foreign_workers)} FOREIGN WORKERS, ", end="")
        print(f"AND {self.land} SQ. MILES OF LAND.")
        print(
            f"THIS YEAR INDUSTRY WILL BUY LAND FOR {self.land_buy_price} "
            "RALLODS PER SQUARE MILE."
        )
        print(
            f"LAND CURRENTLY COSTS {self.planting_cost} RALLODS "
            "PER SQUARE MILE TO PLANT.\n"
        )

    def handle_deaths(
        self, distributed_rallods: int, pollution_control_spendings: int
    ) -> None:
        starved_countrymen = max(
            0, int(self.countrymen - distributed_rallods / COST_OF_LIVING)
        )

        if starved_countrymen > 0:
            print(f"{starved_countrymen} COUNTRYMEN DIED OF STARVATION")

        self.pollution_deaths = int(random() * (INITIAL_LAND - self.land))
        if pollution_control_spendings >= POLLUTION_CONTROL_FACTOR:
            self.pollution_deaths = int(
                self.pollution_deaths
                / (pollution_control_spendings / POLLUTION_CONTROL_FACTOR)
            )
        if self.pollution_deaths > 0:
            print(
                f"{self.pollution_deaths} COUNTRYMEN DIED OF CARBON-MONOXIDE "
                f"AND DUST INHALATION"
            )

        self.died_contrymen = starved_countrymen + self.pollution_deaths
        if self.died_contrymen > 0:
            funeral_cost = self.died_contrymen * COST_OF_FUNERAL
            print(f"   YOU WERE FORCED TO SPEND {funeral_cost} RALLODS ON ")
            print("FUNERAL EXPENSES.")
            self.rallods -= funeral_cost
            if self.rallods < 0:
                print("   INSUFFICIENT RESERVES TO COVER COST - LAND WAS SOLD")
                self.land += int(self.rallods / self.land_buy_price)
                self.rallods = 0
            self.countrymen -= self.died_contrymen

    def handle_tourist_trade(self) -> None:
        V1 = int(self.settled_people * 22 + random() * 500)
        V2 = int((INITIAL_LAND - self.land) * 15)
        tourist_trade_earnings = 0
        if V1 > V2:
            tourist_trade_earnings = V1 - V2
        print(f" YOU MADE {tourist_trade_earnings} RALLODS FROM TOURIST TRADE.")
        if V2 != 0 and not (V1 - V2 >= self.tourism_earnings):
            print("   DECREASE BECAUSE ", end="")
            reason = randint(0, 10)
            if reason <= 2:
                print("FISH POPULATION HAS DWINDLED DUE TO WATER POLLUTION.")
            elif reason <= 4:
                print("AIR POLLUTION IS KILLING GAME BIRD POPULATION.")
            elif reason <= 6:
                print("MINERAL BATHS ARE BEING RUINED BY WATER POLLUTION.")
            elif reason <= 8:
                print("UNPLEASANT SMOG IS DISCOURAGING SUN BATHERS.")
            else:
                print("HOTELS ARE LOOKING SHABBY DUE TO SMOG GRIT.")

        # NOTE: The following two lines had a bug in the original game:
        self.tourism_earnings = abs(int(V1 - V2))
        self.rallods += self.tourism_earnings

    def handle_harvest(self, planted_sq: int) -> None:
        crop_loss = int((INITIAL_LAND - self.land) * ((random() + 1.5) / 2))
        if self.foreign_workers != 0:
            print(f"OF {planted_sq} SQ. MILES PLANTED,")
        if planted_sq <= crop_loss:
            crop_loss = planted_sq
        harvested = int(planted_sq - crop_loss)
        print(f" YOU HARVESTED {harvested} SQ. MILES OF CROPS.")
        unlucky_harvesting_worse = crop_loss - self.crop_loss_last_year
        if crop_loss != 0:
            print("   (DUE TO ", end="")
            if unlucky_harvesting_worse > 2:
                print("INCREASED ", end="")
            print("AIR AND WATER POLLUTION FROM FOREIGN INDUSTRY.)")
        revenue = int((planted_sq - crop_loss) * (self.land_buy_price / 2))
        print(f"MAKING {revenue} RALLODS.")
        self.crop_loss_last_year = crop_loss
        self.rallods += revenue

    def handle_foreign_workers(
        self,
        sm_sell_to_industry: int,
        distributed_rallods: int,
        polltion_control_spendings: int,
    ) -> None:
        foreign_workers_influx = 0
        if sm_sell_to_industry != 0:
            foreign_workers_influx = int(
                sm_sell_to_industry + (random() * 10) - (random() * 20)
            )
            if self.foreign_workers <= 0:
                foreign_workers_influx = foreign_workers_influx + 20
            print(f"{foreign_workers_influx} WORKERS CAME TO THE COUNTRY AND")

        surplus_distributed = distributed_rallods / COST_OF_LIVING - self.countrymen
        population_change = int(
            (surplus_distributed / 10)
            + (polltion_control_spendings / POLLUTION_CONTROL_FACTOR)
            - ((INITIAL_LAND - self.land) / 50)
            - (self.died_contrymen / 2)
        )
        print(f"{abs(population_change)} COUNTRYMEN ", end="")
        if population_change < 0:
            print("LEFT ", end="")
        else:
            print("CAME TO ", end="")
        print("THE ISLAND")
        self.countrymen += population_change
        self.foreign_workers += int(foreign_workers_influx)

    def handle_too_many_deaths(self) -> None:
        print(f"\n\n\n{self.died_contrymen} COUNTRYMEN DIED IN ONE YEAR!!!!!")
        print("\n\n\nDUE TO THIS EXTREME MISMANAGEMENT, YOU HAVE NOT ONLY")
        print("BEEN IMPEACHED AND THROWN OUT OF OFFICE, BUT YOU")
        message = randint(0, 10)
        if message <= 3:
            print("ALSO HAD YOUR LEFT EYE GOUGED OUT!")
        if message <= 6:
            print("HAVE ALSO GAINED A VERY BAD REPUTATION.")
        if message <= 10:
            print("HAVE ALSO BEEN DECLARED NATIONAL FINK.")
        sys.exit()

    def handle_third_died(self) -> None:
        print()
        print()
        print("OVER ONE THIRD OF THE POPULTATION HAS DIED SINCE YOU")
        print("WERE ELECTED TO OFFICE. THE PEOPLE (REMAINING)")
        print("HATE YOUR GUTS.")
        self.end_game()

    def handle_money_mismanagement(self) -> None:
        print()
        print("MONEY WAS LEFT OVER IN THE TREASURY WHICH YOU DID")
        print("NOT SPEND. AS A RESULT, SOME OF YOUR COUNTRYMEN DIED")
        print("OF STARVATION. THE PUBLIC IS ENRAGED AND YOU HAVE")
        print("BEEN FORCED TO EITHER RESIGN OR COMMIT SUICIDE.")
        print("THE CHOICE IS YOURS.")
        print("IF YOU CHOOSE THE LATTER, PLEASE TURN OFF YOUR COMPUTER")
        print("BEFORE PROCEEDING.")
        sys.exit()

    def handle_too_many_foreigners(self) -> None:
        print("\n\nTHE NUMBER OF FOREIGN WORKERS HAS EXCEEDED THE NUMBER")
        print("OF COUNTRYMEN. AS A MINORITY, THEY HAVE REVOLTED AND")
        print("TAKEN OVER THE COUNTRY.")
        self.end_game()

    def end_game(self) -> None:
        if random() <= 0.5:
            print("YOU HAVE BEEN ASSASSINATED.")
        else:
            print("YOU HAVE BEEN THROWN OUT OF OFFICE AND ARE NOW")
            print("RESIDING IN PRISON.")
        sys.exit()

    def handle_congratulations(self) -> None:
        print("\n\nCONGRATULATIONS!!!!!!!!!!!!!!!!!!")
        print(f"YOU HAVE SUCCESFULLY COMPLETED YOUR {YEARS_IN_TERM} YEAR TERM")
        print("OF OFFICE. YOU WERE, OF COURSE, EXTREMELY LUCKY, BUT")
        print("NEVERTHELESS, IT'S QUITE AN ACHIEVEMENT. GOODBYE AND GOOD")
        print("LUCK - YOU'LL PROBABLY NEED IT IF YOU'RE THE TYPE THAT")
        print("PLAYS THIS GAME.")
        sys.exit()


```

这段代码定义了两个函数，分别是 `print_header` 和 `print_instructions`。它们的作用是打印预先格式化的文本，然后输出一些文本。具体来说，这些文本包括一个政治领袖的讲话和一个关于一座较小的共产主义岛屿（这个岛屿有30英里长，15英里宽）的介绍。

`print_header` 函数首先输出一个34行、宽度为15英寸、然后是"KING" 和 "CREATIVE COMPUTING" 的文本。然后是15行，宽度为1英寸的 "JERSEY"。接下来，是11行，宽度为15英寸的 "SPECIAL襟带"。最后，是34行，宽度为15英寸的 "BREAKDOWN BANKS"。

`print_instructions` 函数的输出比较长，但基本上和 `print_header` 函数一样，只是输出了一些描述性的文本。具体来说，这个函数输出的文本包括：

"恭喜您当选为Setat斯的领袖！您负责决定国家的预算和把钱分给您的Countrymen。这个货币系统是RALLODS，每个人每年需要COST_OF_LIVING RALLODS来维持生活。您的国家的收入来自农场生产、游客旅游等。您有两块土地属于FARMLAND。HUNTING, FISHING, ETC。"

然后是两行，宽度为1英寸的 "HALF我们会" 和 "YOU NOW FORECOMES Setat斯领导人"。


```
def print_header() -> None:
    print(" " * 34 + "KING")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


def print_instructions() -> None:
    print(
        f"""\n\n\nCONGRATULATIONS! YOU'VE JUST BEEN ELECTED PREMIER OF SETATS
DETINU, A SMALL COMMUNIST ISLAND 30 BY 70 MILES LONG. YOUR
JOB IS TO DECIDE UPON THE CONTRY'S BUDGET AND DISTRIBUTE
MONEY TO YOUR COUNTRYMEN FROM THE COMMUNAL TREASURY.
THE MONEY SYSTEM IS RALLODS, AND EACH PERSON NEEDS {COST_OF_LIVING}
RALLODS PER YEAR TO SURVIVE. YOUR COUNTRY'S INCOME COMES
FROM FARM PRODUCE AND TOURISTS VISITING YOUR MAGNIFICENT
FORESTS, HUNTING, FISHING, ETC. HALF YOUR LAND IS FARM LAND
```

这段代码是一个AI，它的目的是帮助农民决定他们想要种植多少土地。它使用了玩家输入的信息来计算他们的作物成本，并输出一个提示，告诉他们他们的目标作物等级的年限。这个AI似乎还考虑了农民的土地，他们的货币和他们的工人。


```
WHICH ALSO HAS AN EXCELLENT MINERAL CONTENT AND MAY BE SOLD
TO FOREIGN INDUSTRY (STRIP MINING) WHO IMPORT AND SUPPORT
THEIR OWN WORKERS. CROPS COST BETWEEN 10 AND 15 RALLODS PER
SQUARE MILE TO PLANT.
YOUR GOAL IS TO COMPLETE YOUR {YEARS_IN_TERM} YEAR TERM OF OFFICE.
GOOD LUCK!"""
    )


def ask_how_many_sq_to_plant(state: GameState) -> int:
    while True:
        sq = ask_int("HOW MANY SQUARE MILES DO YOU WISH TO PLANT? ")
        if sq < 0:
            continue
        elif sq > 2 * state.countrymen:
            print("   SORRY, BUT EACH COUNTRYMAN CAN ONLY PLANT 2 SQ. MILES.")
        elif sq > state.farmland:
            print(
                f"   SORRY, BUT YOU ONLY HAVE {state.farmland} "
                "SQ. MILES OF FARM LAND."
            )
        elif sq * state.planting_cost > state.rallods:
            print(
                f"   THINK AGAIN. YOU'VE ONLY {state.rallods} RALLODS "
                "LEFT IN THE TREASURY."
            )
        else:
            return sq


```

这段代码是一个人工智能助手，无法访问互联网，无法了解最新的游戏版本。根据所提供的信息，这段代码的作用是询问玩家在污染控制和出售给工业之间如何选择，并返回选择数字。

在函数内部，首先创建一个名为rallods的变量，用于存储玩家希望用于污染控制和出售给工业的铸币数量。然后，使用ask_int函数向玩家询问他们希望投入多少铸币用于污染控制。

如果玩家的铸币数量大于剩余的铸币数量，函数会打印一条消息并返回0。如果玩家的铸币数量小于或等于0，函数会继续等待，不会做出任何决定。

如果玩家的铸币数量正确，函数将返回铸币数量，并继续等待下一次询问。


```
def ask_pollution_control(state: GameState) -> int:
    while True:
        rallods = ask_int(
            "HOW MANY RALLODS DO YOU WISH TO SPEND ON POLLUTION CONTROL? "
        )
        if rallods > state.rallods:
            print(f"   THINK AGAIN. YOU ONLY HAVE {state.rallods} RALLODS REMAINING.")
        elif rallods < 0:
            continue
        else:
            return rallods


def ask_sell_to_industry(state: GameState) -> int:
    had_first_err = False
    first = """(FOREIGN INDUSTRY WILL ONLY BUY FARM LAND BECAUSE
```

这段代码是一个用于计算农业用地的面积并将其转换为平方英亩的 Python 程序。程序中首先定义了一个名为 FOREST LAND IS UNECONOMICAL TO STRIP MINE 的变量，它表示农业用地的总面积是不经济开发的原因，比如树林、较厚度的土壤等等。

程序接着定义了一个名为是一次输出的函数 sm，该函数用于输入想要出售的农业用地的多少平方公里。程序接着从用户那里获取这个数字，然后用这个数字去除掉农业用地的总面积，如果这个数字大于农业用地的总面积，程序将会输出错误信息，如果这个数字小于0，程序将会继续运行。

程序最后将用户输入的数字作为农业用地的面积，返回这个数字。这段代码对于想要出售农业用地，或者需要知道农业用地能够支持多少平方公里的工业活动非常有用。


```
FOREST LAND IS UNECONOMICAL TO STRIP MINE DUE TO TREES,
THICKER TOP SOIL, ETC.)"""
    err = f"""***  THINK AGAIN. YOU ONLY HAVE {state.farmland} SQUARE MILES OF FARM LAND."""
    while True:
        sm = input("HOW MANY SQUARE MILES DO YOU WISH TO SELL TO INDUSTRY? ")
        try:
            sm_sell = int(sm)
        except ValueError:
            if not had_first_err:
                print(first)
                had_first_err = True
            print(err)
            continue
        if sm_sell > state.farmland:
            print(err)
        elif sm_sell < 0:
            continue
        else:
            return sm_sell


```

这段代码是一个名为 `ask_distribute_rallods` 的函数，用于处理游戏中的财政部。它的功能是询问玩家要分布式多少财富，并将玩家的回答转换为整数类型。如果玩家的回答小于0，程序会停止并提示玩家重新考虑。否则，程序返回玩家的回答作为财富数量，并将其更新到游戏中的 `state.rallods` 变量中。这个函数在游戏进程中一直循环执行，直到玩家结束游戏或程序停止。


```
def ask_distribute_rallods(state: GameState) -> int:
    while True:
        rallods = ask_int(
            "HOW MANY RALLODS WILL YOU DISTRIBUTE AMONG YOUR COUNTRYMEN? "
        )
        if rallods < 0:
            continue
        elif rallods > state.rallods:
            print(
                f"   THINK AGAIN. YOU'VE ONLY {state.rallods} RALLODS IN THE TREASURY"
            )
        else:
            return rallods


```

这段代码是一个Python函数，名为“resume”，它返回一个名为“GameState”的类游戏状态对象。在这个函数中，开发人员可以询问用户一些信息，然后根据用户提供的信息来评估游戏状态。

具体来说，这段代码功能如下：

1. 询问用户有多少年进入办公室，如果用户提供的数字小于0，函数会退出。
2. 询问用户他们有多少资金，如果用户提供的数字小于0，函数会退出。
3. 询问用户有多少国家公民，如果用户提供的数字小于0，函数会退出。
4. 询问用户有多少工人，如果用户提供的数字小于0，函数会退出。
5. 询问用户有多少平方公里的土地，如果用户提供的数字小于0，函数会退出。
6. 如果用户提供的数字大于初始的土地面积，函数会计算出农场土地和森林土地，然后打印农场土地和森林土地。
7. 如果用户提供的数字大于初始的游戏中的公民数量，函数会打印游戏中的公民数量。
8. 如果用户提供的数字大于初始的游戏中的工人数量，函数会打印游戏中的工人数量。
9. 无限循环地询问用户各种信息，直到用户结束对话。
10. 返回一个名为“GameState”的类游戏状态对象，包含rallods属性、countrymen属性、foreign_workers属性和years_in_office属性。


```
def resume() -> GameState:
    while True:
        years = ask_int("HOW MANY YEARS HAD YOU BEEN IN OFFICE WHEN INTERRUPTED? ")
        if years < 0:
            sys.exit()
        if years >= YEARS_IN_TERM:
            print(f"   COME ON, YOUR TERM IN OFFICE IS ONLY {YEARS_IN_TERM} YEARS.")
        else:
            break
    treasury = ask_int("HOW MUCH DID YOU HAVE IN THE TREASURY? ")
    if treasury < 0:
        sys.exit()
    countrymen = ask_int("HOW MANY COUNTRYMEN? ")
    if countrymen < 0:
        sys.exit()
    workers = ask_int("HOW MANY WORKERS? ")
    if workers < 0:
        sys.exit()
    while True:
        land = ask_int("HOW MANY SQUARE MILES OF LAND? ")
        if land < 0:
            sys.exit()
        if land > INITIAL_LAND:
            farm_land = INITIAL_LAND - FOREST_LAND
            print(f"   COME ON, YOU STARTED WITH {farm_land:,} SQ. MILES OF FARM LAND")
            print(f"   AND {FOREST_LAND:,} SQ. MILES OF FOREST LAND.")
        if land > FOREST_LAND:
            break
    return GameState(
        rallods=treasury,
        countrymen=countrymen,
        foreign_workers=workers,
        years_in_office=years,
    )


```

It looks like you have provided a Python script called `run_the_year.py`. I am not able to understand the functionality of this script as it is written in Python and it appears to be using a `pygame` module to display some图形 on the screen.

If you have any specific questions or if there is anything else I can help you with, please let me know.


```
def main() -> None:
    print_header()
    want_instructions = input("DO YOU WANT INSTRUCTIONS? ").upper()
    if want_instructions == "AGAIN":
        state = resume()
    else:
        state = GameState(
            rallods=randint(59000, 61000),
            countrymen=randint(490, 510),
            planting_cost=randint(10, 15),
        )
    if want_instructions != "NO":
        print_instructions()

    while True:
        state.set_market_conditions()
        state.print_status()

        # Users actions
        sm_sell_to_industry = ask_sell_to_industry(state)
        state.sell_land(sm_sell_to_industry)

        distributed_rallods = ask_distribute_rallods(state)
        state.distribute_rallods(distributed_rallods)

        planted_sq = ask_how_many_sq_to_plant(state)
        state.plant(planted_sq)
        polltion_control_spendings = ask_pollution_control(state)
        state.spend_pollution_control(polltion_control_spendings)

        # Run the year
        state.handle_deaths(distributed_rallods, polltion_control_spendings)
        state.handle_foreign_workers(
            sm_sell_to_industry, distributed_rallods, polltion_control_spendings
        )
        state.handle_harvest(planted_sq)
        state.handle_tourist_trade()

        if state.died_contrymen > 200:
            state.handle_too_many_deaths()
        if state.countrymen < 343:
            state.handle_third_died()
        elif (
            state.rallods / 100
        ) > 5 and state.died_contrymen - state.pollution_deaths >= 2:
            state.handle_money_mismanagement()
        if state.foreign_workers > state.countrymen:
            state.handle_too_many_foreigners()
        elif YEARS_IN_TERM - 1 == state.years_in_office:
            state.handle_congratulations()
        else:
            state.years_in_office += 1
            state.died_contrymen = 0


```

这段代码是一个条件判断语句，它会判断当前脚本是否作为主程序运行。如果是主程序运行，那么程序会直接进入__main__函数，否则跳过__main__函数。

具体来说，当脚本作为主程序运行时，执行的指令是`if __name__ == "__main__": main()`，这会先执行`__main__`函数，所以`main()`函数中的代码会被执行。如果脚本不是主程序运行，那么条件判断为假，程序直接跳过`__main__`函数，不会执行`main()`函数中的代码。


```
if __name__ == "__main__":
    main()

```