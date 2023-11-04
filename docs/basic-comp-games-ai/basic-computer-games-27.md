# BasicComputerGames源码解析 27

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `17_Bullfight/python/bullfight.py`

The `determine_player_kills` function takes in several arguments:

-   `bull_quality`: The quality of the bull, ranging from 1 to 10.
-   `player_type`: The type of the player, either "TOREDOAD" or "PICADO".
-   `plural_form`: The plural form of the player's job quality, either "ORES" or "RES".
-   `job_qualities`: A list of job qualities that the player has.

It returns a float that represents the job quality factor.

The function first calculates the job quality factor based on the bull performance and the player's job quality. If the bull performance is less than 0.37, the job quality factor is 0.5. If the bull performance is between 0.37 and 0.5, the job quality factor is 0.4. If the bull performance is between 0.5 and 0.63, the job quality factor is 0.3. If the bull performance is greater than or equal to 0.63, the job quality factor is 0.2.

Then, based on the job quality factor, the function prints a message indicating whether the player and/or the bull were killed, and/or how many killers were found.

If the player was killed, the function uses the `random.choice` function to choose one of the horses of the player and/or one of the killers. If the player was not killed, the function uses the `random.randint` function to randomly choose one of the horses or the killers.

If the player was not killed and the job quality factor is greater than or equal to 4, the function will print a message indicating that one of the horses of the player was killed, and/or how many killers were found.


```
import math
import random
from typing import Dict, List, Literal, Tuple, Union


def print_n_newlines(n: int) -> None:
    for _ in range(n):
        print()


def determine_player_kills(
    bull_quality: int,
    player_type: Literal["TOREAD", "PICADO"],
    plural_form: Literal["ORES", "RES"],
    job_qualities: List[str],
) -> float:
    bull_performance = 3 / bull_quality * random.random()
    if bull_performance < 0.37:
        job_quality_factor = 0.5
    elif bull_performance < 0.5:
        job_quality_factor = 0.4
    elif bull_performance < 0.63:
        job_quality_factor = 0.3
    elif bull_performance < 0.87:
        job_quality_factor = 0.2
    else:
        job_quality_factor = 0.1
    job_quality = math.floor(10 * job_quality_factor + 0.2)  # higher is better
    print(f"THE {player_type}{plural_form} DID A {job_qualities[job_quality]} JOB.")
    if job_quality >= 4:
        if job_quality == 5:
            player_was_killed = random.choice([True, False])
            if player_was_killed:
                print(f"ONE OF THE {player_type}{plural_form} WAS KILLED.")
            elif player_was_killed:
                print(f"NO {player_type}{plural_form} WERE KILLED.")
        else:
            if player_type != "TOREAD":
                killed_horses = random.randint(1, 2)
                print(
                    f"{killed_horses} OF THE HORSES OF THE {player_type}{plural_form} KILLED."
                )
            killed_players = random.randint(1, 2)
            print(f"{killed_players} OF THE {player_type}{plural_form} KILLED.")
    print()
    return job_quality_factor


```



这段代码定义了一个名为 `calculate_final_score` 的函数，用于计算最终得分。该函数的输入参数包括：

- `move_risk_sum`：一个浮点数，表示当前工作中所承担的风险。
- `job_quality_by_round`：一个字典，其中包含每个工作包的 `job_quality` 值。`job_quality` 是一个关键绩效指标(KPI)，用于评估工作表现。
- `bull_quality`：一个整数，表示当前市场中投资经理的素质。

函数的最终输出是一个浮点数，表示根据所输入的参数计算出来的得分。

函数的实现中，首先对输入参数进行了多个加权平均，以反映不同 KPI 对最终得分的影响。具体地，首先计算了 `move_risk_sum` 对 `job_quality_by_round` 的加权平均，然后计算了 `job_quality_by_round` 中所有键的加权平均值，接着计算了 `job_quality_by_round[1]` 和 `job_quality_by_round[2]` 的乘积，然后计算了 `job_quality_by_round[4]` 和 `job_quality_by_round[5]` 的乘积，接着计算了 `job_quality_by_round[3]` 的平方除以 120，以及一个随机数。最后，将这些加权平均值和 `bull_quality` 值与 `quality` 计算得出最终得分。

如果 `quality` 的值小于 2.4，则返回 0；如果 `quality` 的值小于 4.9，则返回 1；如果 `quality` 的值小于 7.4，则返回 2；否则，返回 3。


```
def calculate_final_score(
    move_risk_sum: float, job_quality_by_round: Dict[int, float], bull_quality: int
) -> float:
    quality = (
        4.5
        + move_risk_sum / 6
        - (job_quality_by_round[1] + job_quality_by_round[2]) * 2.5
        + 4 * job_quality_by_round[4]
        + 2 * job_quality_by_round[5]
        - (job_quality_by_round[3] ** 2) / 120
        - bull_quality
    ) * random.random()
    if quality < 2.4:
        return 0
    elif quality < 4.9:
        return 1
    elif quality < 7.4:
        return 2
    else:
        return 3


```

这段代码定义了两个函数，分别是 print_header 和 print_instructions。

print_header 函数的作用是在控制台输出一个包含 34 个空格的字符串，以及一个以 " " 为开头，以 "BULL" 为结尾的字符串，然后调用 print_n_newlines 函数以在字符串中插入指定的新行。

print_instructions 函数的作用是在控制台输出一个字符串，其中包含一系列有关如何杀死公牛的指令，包括在哪个手中杀死公牛、如何移动以及使用哪个得分等。函数的输出会在每次杀死公牛后改变。

通过调用 print_header 和 print_instructions，我们可以让用户在游戏中根据提示杀死公牛以获得更高的得分。


```
def print_header() -> None:
    print(" " * 34 + "BULL")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print_n_newlines(2)


def print_instructions() -> None:
    print("HELLO, ALL YOU BLOODLOVERS AND AFICIONADOS.")
    print("HERE IS YOUR BIG CHANCE TO KILL A BULL.")
    print()
    print("ON EACH PASS OF THE BULL, YOU MAY TRY")
    print("0 - VERONICA (DANGEROUS INSIDE MOVE OF THE CAPE)")
    print("1 - LESS DANGEROUS OUTSIDE MOVE OF THE CAPE")
    print("2 - ORDINARY SWIRL OF THE CAPE.")
    print()
    print("INSTEAD OF THE ABOVE, YOU MAY TRY TO KILL THE BULL")
    print("ON ANY TURN: 4 (OVER THE HORNS), 5 (IN THE CHEST).")
    print("BUT IF I WERE YOU,")
    print("I WOULDN'T TRY IT BEFORE THE SEVENTH PASS.")
    print()
    print("THE CROWD WILL DETERMINE WHAT AWARD YOU DESERVE")
    print("(POSTHUMOUSLY IF NECESSARY).")
    print("THE BRAVER YOU ARE, THE BETTER THE AWARD YOU RECEIVE.")
    print()
    print("THE BETTER THE JOB THE PICADORES AND TOREADORES DO,")
    print("THE BETTER YOUR CHANCES ARE.")


```

这段代码定义了一个名为 "print_intro" 的函数，用于输出一个包含问候语和一些说明的文本。接着，定义了一个名为 "ask_bool" 的函数，用于询问用户是否需要进一步指导和输出一条信息。

具体来说，代码中的 "print_intro" 函数会先调用一个名为 "print_header" 的函数，输出一个包含头部的文本。然后，它会等待用户输入 "DO YOU WANT INSTRUCTIONS?" 这条消息，如果用户选择 "是的"，那么 "print_instructions" 函数会被调用，输出一些说明。最后，它会输出两个新行。

而 "ask_bool" 函数会先输入一条消息，提示用户需要输入一个布尔值(真或假)。然后，它会等待用户输入一个字符串，直到用户输入 "YES" 或 "NO"，才会返回一个布尔值。如果用户输入 "NO"，函数会输出一个消息，指出他们的回答是错误的。如果输入 "YES"，函数会返回 True，否则会再次输入消息。


```
def print_intro() -> None:
    print_header()
    want_instructions = input("DO YOU WANT INSTRUCTIONS? ")
    if want_instructions != "NO":
        print_instructions()
    print_n_newlines(2)


def ask_bool(prompt: str) -> bool:
    while True:
        answer = input(prompt).lower()
        if answer == "yes":
            return True
        elif answer == "no":
            return False
        else:
            print("INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO'.")


```

这两段代码是在定义两个函数，一个叫做ask_int()和一个叫做did_bull_hit()。

ask_int()函数的作用是询问用户输入一个浮点数，并返回用户输入的浮点数类型的整数。函数内部使用了一个while循环，会一直询问用户输入 until 用户输入不再有误。如果用户输入的不是一个数字，函数会输出一个错误消息并退出。如果用户输入的是一个数字，函数会判断该数字是否为3，如果是，则退出while循环。

did_bull_hit()函数的作用是判断一头公牛是否成功越过了障碍物。函数内部定义了一些变量，用于存储公牛的质量和 Move Risk 分数。如果公牛跳到了障碍物上方，则Move Risk分数为2，否则Move Risk分数为0.5。公牛的强度被定义为6减去公牛的得分，得分范围是1到10。得分乘以0.5加上公牛移动风险得分/10，然后加上5，得到一个范围在0.51到1之间的浮点数，表示公牛是否成功越过了障碍物。如果公牛成功越过了障碍物，则返回True和移动风险得分，否则返回False和移动风险得分。


```
def ask_int() -> int:
    while True:
        foo = float(input())
        if foo != float(int(abs(foo))):  # we actually want an integer
            print("DON'T PANIC, YOU IDIOT!  PUT DOWN A CORRECT NUMBER")
        elif foo < 3:
            break
    return int(foo)


def did_bull_hit(
    bull_quality: int,
    cape_move: int,
    job_quality_by_round: Dict[int, float],
    move_risk_sum: float,
) -> Tuple[bool, float]:
    # The bull quality is a grade: The lower the grade, the better the bull
    if cape_move == 0:
        move_risk: Union[int, float] = 3
    elif cape_move == 1:
        move_risk = 2
    else:
        move_risk = 0.5
    move_risk_sum += move_risk
    bull_strength = 6 - bull_quality
    bull_hit_factor = (  # the higher the factor, the more "likely" it hits
        (bull_strength + move_risk / 10)
        * random.random()
        / (
            (
                job_quality_by_round[1]
                + job_quality_by_round[2]
                + job_quality_by_round[3] / 10
            )
            * 5
        )
    )
    bull_hit = bull_hit_factor >= 0.51
    return bull_hit, move_risk_sum


```

这段代码是一个名为 `handle_bullkill_attempt` 的函数，它接受四个参数：`kill_method`、`job_quality_by_round`、`bull_quality` 和 `gore`。函数的作用是判断用户是否成功击杀一头牛，并返回相应的值。

具体来说，函数首先检查 `kill_method` 是否在 [4, 5] 范围内，如果不在这个范围内，函数会输出一个笑话，并将 `gore` 设为 2。否则，函数会计算出牛的攻击概率，公式为：攻击概率 = (攻击力 * 10 * 随机数 / (目标质量 + 目标质量))，其中 `攻击力` 为牛的攻击力，即 6，`目标质量` 为牛的质量，即 `job_quality_by_round` 中的值，随机数在 [0, 1] 之间。

如果 `kill_method` 是 4，攻击概率大于 0.8，则 `gore` 被设为 1；否则，如果攻击概率大于 0.2，则 `gore` 被设为 1。最后，如果 `gore` 为 0，函数会输出“你杀死了那头牛！”，并将 `job_quality_by_round` 中的 `job_quality_by_round[5]` 设为 2，并返回 `gore`。


```
def handle_bullkill_attempt(
    kill_method: int,
    job_quality_by_round: Dict[int, float],
    bull_quality: int,
    gore: int,
) -> int:
    if kill_method not in [4, 5]:
        print("YOU PANICKED.  THE BULL GORED YOU.")
        gore = 2
    else:
        bull_strength = 6 - bull_quality
        kill_probability = (
            bull_strength
            * 10
            * random.random()
            / (
                (job_quality_by_round[1] + job_quality_by_round[2])
                * 5
                * job_quality_by_round[3]
            )
        )
        if kill_method == 4:
            if kill_probability > 0.8:
                gore = 1
        else:
            if kill_probability > 0.2:
                gore = 1
        if gore == 0:
            print("YOU KILLED THE BULL!")
            job_quality_by_round[5] = 2
            return gore
    return gore


```

这段代码定义了一个名为 `final_message` 的函数，它接受四个参数：`job_quality_by_round`、`bull_quality` 和 `move_risk_sum`。函数内部先调用一个名为 `print_n_newlines` 的函数，这个函数会在屏幕上输出指定的行数。然后根据 `job_quality_by_round` 中第四个元素的值来决定是否输出 "ALERT成交" 或 "THE CROWD BOOS FOR TEN MINUTES. IF YOU EVER DARE TO SHOW YOUR FACE IN A RING AGAIN, THEY SWEAR THEY WILL KILL YOU--"。如果是这种情况，函数将输出 "THE CROWD BOOS FOR TEN MINUTES. IF YOU ANYWAY DARE TO SHOW YOUR FACE IN A RING AGAIN, THEY SWEAR THEY WILL KILL YOU--"。否则，函数将根据 `job_quality_by_round` 中的第四个元素来输出 "THE CROWD CHEERS WILDERLY!" 或 "THE CROWD CHEERS!"，然后输出 "THE CROWD AWARDS YOU"。接着函数调用一个名为 `calculate_final_score` 的函数，这个函数接受三个参数：`move_risk_sum`、`job_quality_by_round` 和 `bull_quality`。函数内部根据传入的参数计算出 `final_score`，然后根据 `final_score` 的值来输出 "OLE！YOU ARE 'MUY HOMBRE'！！OLE！OLE！" 或 "OLE！YOU ARE 'MUY HOMBRE'！！OLE！OLE！"。最后，函数再次调用 `print_n_newlines` 函数来输出三行内容。


```
def final_message(
    job_quality_by_round: Dict[int, float], bull_quality: int, move_risk_sum: float
) -> None:
    print_n_newlines(3)
    if job_quality_by_round[4] == 0:
        print("THE CROWD BOOS FOR TEN MINUTES.  IF YOU EVER DARE TO SHOW")
        print("YOUR FACE IN A RING AGAIN, THEY SWEAR THEY WILL KILL YOU--")
        print("UNLESS THE BULL DOES FIRST.")
    else:
        if job_quality_by_round[4] == 2:
            print("THE CROWD CHEERS WILDLY!")
        elif job_quality_by_round[5] == 2:
            print("THE CROWD CHEERS!")
            print()
        print("THE CROWD AWARDS YOU")
        score = calculate_final_score(move_risk_sum, job_quality_by_round, bull_quality)
        if score == 0:
            print("NOTHING AT ALL.")
        elif score == 1:
            print("ONE EAR OF THE BULL.")
        elif score == 2:
            print("BOTH EARS OF THE BULL!")
            print("OLE!")
        else:
            print("OLE!  YOU ARE 'MUY HOMBRE'!! OLE!  OLE!")
        print()
        print("ADIOS")
        print_n_newlines(3)


```

I'm sorry, but I'm not sure what you are asking. It looks like you are trying to write a story, but you are having trouble with formatting and character清空。你能向我提供更具体的问题或需要我帮助的代码吗？


```
def main() -> None:
    print_intro()
    move_risk_sum: float = 1
    job_quality_by_round: Dict[int, float] = {4: 1, 5: 1}
    job_quality = ["", "SUPERB", "GOOD", "FAIR", "POOR", "AWFUL"]
    bull_quality = random.randint(
        1, 5
    )  # the lower the number, the more powerful the bull
    print(f"YOU HAVE DRAWN A {job_quality[bull_quality]} BULL.")
    if bull_quality > 4:
        print("YOU'RE LUCKY.")
    elif bull_quality < 2:
        print("GOOD LUCK.  YOU'LL NEED IT.")
        print()
    print()

    # Round 1: Run Picadores
    player_type: Literal["TOREAD", "PICADO"] = "PICADO"
    plural_form: Literal["ORES", "RES"] = "RES"
    job_quality_factor = determine_player_kills(
        bull_quality, player_type, plural_form, job_quality
    )
    job_quality_by_round[1] = job_quality_factor

    # Round 2: Run Toreadores
    player_type = "TOREAD"
    plural_form = "ORES"
    determine_player_kills(bull_quality, player_type, plural_form, job_quality)
    job_quality_by_round[2] = job_quality_factor
    print_n_newlines(2)

    # Round 3
    job_quality_by_round[3] = 0
    while True:
        job_quality_by_round[3] = job_quality_by_round[3] + 1
        print(f"PASS NUMBER {job_quality_by_round[3]}")
        if job_quality_by_round[3] >= 3:
            run_from_ring = ask_bool("HERE COMES THE BULL.  TRY FOR A KILL? ")
            if not run_from_ring:
                print("CAPE MOVE? ", end="")
        else:
            print("THE BULL IS CHARGING AT YOU!  YOU ARE THE MATADOR--")
            run_from_ring = ask_bool("DO YOU WANT TO KILL THE BULL? ")
            if not run_from_ring:
                print("WHAT MOVE DO YOU MAKE WITH THE CAPE? ", end="")
        gore = 0
        if not run_from_ring:
            cape_move = ask_int()
            bull_hit, move_risk_sum = did_bull_hit(
                bull_quality, cape_move, job_quality_by_round, move_risk_sum
            )
            if bull_hit:
                gore = 1
            else:
                continue
        else:
            print()
            print("IT IS THE MOMENT OF TRUTH.")
            print()
            kill_method = int(input("HOW DO YOU TRY TO KILL THE BULL? "))
            gore = handle_bullkill_attempt(
                kill_method, job_quality_by_round, bull_quality, gore
            )
            if gore == 0:
                break
        if gore > 0:
            if gore == 1:
                print("THE BULL HAS GORED YOU!")
            death = False
            while True:
                if random.randint(1, 2) == 1:
                    print("YOU ARE DEAD.")
                    job_quality_by_round[4] = 1.5
                    death = True
                    break
                else:
                    print("YOU ARE STILL ALIVE.")
                    print()
                    print("DO YOU RUN FROM THE RING? ", end="")
                    run_from_ring = ask_bool("DO YOU RUN FROM THE RING? ")
                    if not run_from_ring:
                        print("YOU ARE BRAVE.  STUPID, BUT BRAVE.")
                        if random.randint(1, 2) == 1:
                            job_quality_by_round[4] = 2
                            death = True
                            break
                        else:
                            print("YOU ARE GORED AGAIN!")
                    else:
                        print("COWARD")
                        job_quality_by_round[4] = 0
                        death = True
                        break

            if death:
                break

    final_message(job_quality_by_round, bull_quality, move_risk_sum)


```

这段代码是一个条件判断语句，它判断当前程序是否作为主程序运行。如果当前程序是作为主程序运行，那么程序会执行if语句块内的内容。

if语句块内包含了一个函数main，它可能是程序的入口函数。如果程序作为主程序运行，那么程序会先执行main函数，然后执行if语句块内的内容。如果程序不是作为主程序运行，那么程序不会执行if语句块内的内容，直接执行程序的下一条语句。

总之，这段代码的作用是判断当前程序是否作为主程序运行，如果是，就执行main函数，否则不执行。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Bullseye

In this game, up to 20 players throw darts at a target with 10-, 20-, 30-, and 40-point zones. The objective is to get 200 points.

You have a choice of three methods of throwing:

| Throw | Description        | Probable Score            |
|-------|--------------------|---------------------------|
| 1     | Fast overarm       | Bullseye or complete miss |
| 2     | Controlled overarm | 10, 20, or 30 points      |
| 3     | Underarm           | Anything                  |

You will find after playing a while that different players will swear by different strategies. However, considering the expected score per throw by always using throw 3:

| Score (S) | Probability (P) | S x P |
|-----------|-----------------|-------|
|     40    |  1.00-.95 = .05 |   2   |
|     30    |  .95-.75 = .20  |   6   |
|     30    |  .75-.45 = .30  |   6   |
|     10    |  .45-.05 = .40  |   4   |
|     0     |  .05-.00 = .05  |   0   |

Expected score per throw = 18

Calculate the expected score for the other throws and you may be surprised!

The program was written by David Ahl of Creative Computing.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=34)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=49)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)



# `18_Bullseye/csharp/BullseyeGame.cs`

Welcome to Bullseye, a game of cryptic猜数字。

You are a player, and you play against other players. Each player has a unique name, and you will play against one player at a time.

The game is played in rounds. Each round is a different number, and the player with the highest score at the end of the round wins.

There are 6 rounds in total.

祝你好运，并取得伟大的成绩！";

private const string Operations = @"

for (var i = 1; i <= 6; i++)
{
println($"ROUND ${i}");
}";
```

Here is the code for the `Bullseye` game program.

This program does the following:
- 6 players play against 1 player
- Each player has a unique name, and is displayed on the console
- The game is played in 6 rounds
- The rounds are 1 to 6, and the player with the highest score at the end of the round wins
- The code also includes a `PrintIntroduction` and `PrintResults` method to display information about the game

Please note that this code is a snippet of implementation, it is not complete and ready for production use.
```


```
namespace Bullseye
{
    /// <summary>
    /// Class encompassing the game
    /// </summary>
    public class BullseyeGame
    {
        private readonly List<Player> _players;

        // define a constant for the winning score so that it is
        // easy to change again in the future
        private const int WinningScore = 200;

        public BullseyeGame()
        {
            // create the initial list of players; list is empty, but
            // the setup of the game will add items to this list
            _players = new List<Player>();
        }

        public void Run()
        {
            PrintIntroduction();

            SetupGame();

            PlayGame();

            PrintResults();
        }

        private void SetupGame()
        {
            // First, allow the user to enter how many players are going
            // to play. This could be weird if the user enters negative
            // numbers, words, or too many players, so there are some
            // extra checks on the input to make sure the user didn't do
            // anything too crazy. Loop until the user enters valid input.
            bool validPlayerCount;
            int playerCount;
            do
            {
                Console.WriteLine();
                Console.Write("HOW MANY PLAYERS? ");
                string? input = Console.ReadLine();

                // assume the user has entered something incorrect - the
                // next steps will validate the input
                validPlayerCount = false;

                if (Int32.TryParse(input, out playerCount))
                {
                    if (playerCount > 0 && playerCount <= 20)
                    {
                        validPlayerCount = true;
                    }
                    else
                    {
                        Console.WriteLine("YOU MUST ENTER A NUMBER BETWEEN 1 AND 20!");
                    }
                }
                else
                {
                    Console.WriteLine("YOU MUST ENTER A NUMBER");
                }

            }
            while (!validPlayerCount);

            // Next, allow the user to enter names for the players; as each
            // name is entered, create a Player object to track the name
            // and their score, and save the object to the list in this class
            // so the rest of the game has access to the set of players
            for (int i = 0; i < playerCount; i++)
            {
                string? playerName = String.Empty;
                do
                {
                    Console.Write($"NAME OF PLAYER #{i+1}? ");
                    playerName = Console.ReadLine();

                    // names can be any sort of text, so allow whatever the user
                    // enters as long as it isn't a blank space
                }
                while (String.IsNullOrWhiteSpace(playerName));

                _players.Add(new Player(playerName));
            }
        }

        private void PlayGame()
        {
            Random random = new Random(DateTime.Now.Millisecond);

            int round = 0;
            bool isOver = false;
            do
            {
                // starting a new round, increment the counter
                round++;
                Console.WriteLine($"ROUND {round}");
                Console.WriteLine("--------------");

                foreach (Player player in _players)
                {
                    // ask the user how they want to throw
                    Console.Write($"{player.Name.ToUpper()}'S THROW: ");
                    string? input = Console.ReadLine();

                    // based on the input, figure out the probabilities
                    int[] probabilities;
                    switch (input)
                    {
                        case "1":
                        {
                            probabilities = new int[] { 65, 55, 50, 50 };
                            break;
                        }
                        case "2":
                        {
                            probabilities = new int[] { 99, 77, 43, 1 };
                            break;
                        }
                        case "3":
                        {
                            probabilities = new int[] { 95, 75, 45, 5 };
                            break;
                        }
                        default:
                        {
                            // in case the user types something bad, pretend it's
                            // as if they tripped over themselves while throwing
                            // the dart - they'll either hit a bullseye or completely
                            // miss
                            probabilities = new int[] { 95, 95, 95, 95 };
                            Console.Write("TRIP! ");
                            break;
                        }
                    }


                    // Next() returns a number in the range: min <= num < max, so specify 101
                    // as the maximum so that 100 is a number that could be returned
                    int chance = random.Next(0, 101);

                    if (chance > probabilities[0])
                    {
                        player.Score += 40;
                        Console.WriteLine("BULLSEYE!!  40 POINTS!");
                    }
                    else if (chance > probabilities[1])
                    {
                        player.Score += 30;
                        Console.WriteLine("30-POINT ZONE!");
                    }
                    else if (chance > probabilities[2])
                    {
                        player.Score += 20;
                        Console.WriteLine("20-POINT ZONE");
                    }
                    else if (chance > probabilities[3])
                    {
                        player.Score += 10;
                        Console.WriteLine("WHEW!  10 POINTS.");
                    }
                    else
                    {
                        // missed it
                        Console.WriteLine("MISSED THE TARGET!  TOO BAD.");
                    }

                    // check to see if the player has won - if they have, then
                    // break out of the loops
                    if (player.Score > WinningScore)
                    {
                        Console.WriteLine();
                        Console.WriteLine("WE HAVE A WINNER!!");
                        Console.WriteLine($"{player.Name.ToUpper()} SCORED {player.Score} POINTS.");
                        Console.WriteLine();

                        isOver = true; // out of the do/while round loop
                        break; // out of the foreach (player) loop
                    }

                    Console.WriteLine();
                }
            }
            while (!isOver);
        }

        private void PrintResults()
        {
            // For bragging rights, print out all the scores, but sort them
            // by who had the highest score
            var sorted = _players.OrderByDescending(p => p.Score);

            // padding is used to get things to line up nicely - the results
            // should look something like:
            //      PLAYER       SCORE
            //      Bravo          210
            //      Charlie         15
            //      Alpha            1
            Console.WriteLine("PLAYER       SCORE");
            foreach (var player in sorted)
            {
                Console.WriteLine($"{player.Name.PadRight(12)} {player.Score.ToString().PadLeft(5)}");
            }

            Console.WriteLine();
            Console.WriteLine("THANKS FOR THE GAME.");
        }

        private void PrintIntroduction()
        {
            Console.WriteLine(Title);
            Console.WriteLine();
            Console.WriteLine(Introduction);
            Console.WriteLine();
            Console.WriteLine(Operations);
        }

        private const string Title = @"
                    BULLSEYE
    CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY";

        private const string Introduction = @"
```

这段代码定义了一个名为 "Game" 的类，其中包含一个名为 "UpTo20Players" 的属性，它的类型为 "max"，表示可以有最高20个玩家。接着，在 "Game" 类中，定义了一个名为 "Operations" 的属性，其值为 "THROW DARTS AT A TARGET" 和 "WITH 10, 20, 30, AND 40 POINT ZONES。THE OBJECTIVE IS TO GET 200 POINTS。"，表示游戏的目标是让最高20个玩家投掷飞镖，并取得最高分数200分。此外，还定义了一个名为 "private const string Operations = @"THROW   DESCRIPTION         PROBABLE SCORE" 和 "private const string Operations = @"THROW DARTS AT A TARGET" 和 "private const string Operations = @"THROW DARTS AT A TARGET" 和 "private const string Operations = @"THROW DARTS AT A TARGET" 的字符串变量 Operations，用于存储不同飞镖的得分描述。


```
IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET
WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS
TO GET 200 POINTS.";

        private const string Operations = @"
THROW   DESCRIPTION         PROBABLE SCORE
  1     FAST OVERARM        BULLSEYE OR COMPLETE MISS
  2     CONTROLLED OVERARM  10, 20, OR 30 POINTS
  3     UNDERARM            ANYTHING";
    }
}

```

# `18_Bullseye/csharp/Player.cs`

这段代码定义了一个名为 `Player` 的类，用于表示一个在游戏中的玩家。该类包含两个私有属性和一个构造函数。

构造函数用于初始化玩家的姓名和得分。玩家对象的姓名存储在 `Name` 属性中，得分存储在 `Score` 属性中。

`public Player(string name)` 构造函数接受一个字符串参数，用于指定玩家的姓名，并将其存储在 `Name` 属性中。

`public void SetScore(int score)` 方法用于设置玩家的得分，将得分存储在 `Score` 属性中。

该类没有提供任何方法或属性来操作玩家本身，但是可以被用来创建玩家的实例并设置其属性。


```
namespace Bullseye
{
    /// <summary>
    /// Object to track the name and score of a player
    /// </summary>
    public class Player
    {
        /// <summary>
        /// Creates a play with the given name
        /// </summary>
        /// <param name="name">Name of the player</param>
        public Player(string name)
        {
            Name = name;
            Score = 0;
        }

        /// <summary>
        /// Name of the player
        /// </summary>
        public string Name { get; private set; }

        /// <summary>
        /// Current score of the player
        /// </summary>
        public int Score { get; set; }
    }
}

```

# `18_Bullseye/csharp/Program.cs`

这段代码是一个C#程序，定义了一个名为"Bullseye"的游戏类，并创建了一个名为"Program"的类，该类具有一个名为"Main"的静态方法。

运行程序时，首先会创建一个"BullseyeGame"的实例，然后调用该实例的"Run"方法。

具体来说，"Run"方法会在游戏窗口中心显示一个小目标，然后让玩家通过点击目标来控制游戏中小鸟的移动。


```
﻿using System;

namespace Bullseye
{
    public static class Program
    {
        // Entry point to the application; create an instance of the
        // game class and call Run()
        public static void Main(string[] args)
        {
            new BullseyeGame().Run();
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `18_Bullseye/java/src/Bullseye.java`



This code appears to be a simple command-line interface (CLI) for a baseball game. It has different functions for getting input from the players, displaying text and numbers on the screen, and getting the number of players to guess.

The `Player` class appears to have some basic information about each player, like their name and the number of runs they scored in different games.

The `getPlayersThrow` method appears to be getting the number of runs scored by each player when they throw the baseball. It takes a `Player` object and returns an integer. It uses the `displayTextAndGetInput` method to get the input from the players. If the input is "1", "2", or "3", it returns 1, otherwise it returns the integer. It is not clear what is being returned by the `displayTextAndGetInput` method.

The `chooseNumberOfPlayers` method appears to be getting the number of players from the players. It takes a displayTextAndGetInput and returns an integer. It uses the `displayTextAndGetInput` method to get the input from the players. It is not clear what is being returned by the `displayTextAndGetInput` method.

The `displayTextAndGetInput` method appears to be getting input from the players. It takes a text message and returns the input as a `String`. It is not clear what is being returned by this method.

The `paddedString` method appears to be getting input from the `displayTextAndGetInput` method and formats it with spaces. It takes three strings and a format string, and returns the formatted string. It is not clear what is being passed to the `paddedString` method.

Overall, it appears that this CLI is used to get input from players in a baseball game and displays it on the screen. It is not clear what all the functions in this code do and how they are being used.


```
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Game of Bullseye
 * <p>
 * Based on the Basic game of Bullseye here
 * https://github.com/coding-horror/basic-computer-games/blob/main/18%20Bullseye/bullseye.bas
 * <p>
 * Note:  The idea was to create a version of 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Bullseye {

    // Used for formatting output
    public static final int FIRST_IDENT = 10;
    public static final int SECOND_IDENT = 30;
    public static final int THIRD_INDENT = 30;

    // Used to decide throw result
    public static final double[] SHOT_ONE = new double[]{.65, .55, .5, .5};
    public static final double[] SHOT_TWO = new double[]{.99, .77, .43, .01};
    public static final double[] SHOT_THREE = new double[]{.95, .75, .45, .05};

    private enum GAME_STATE {
        STARTING,
        START_GAME,
        PLAYING,
        GAME_OVER
    }

    private GAME_STATE gameState;

    private final ArrayList<Player> players;

    private final Shot[] shots;

    // Used for keyboard input
    private final Scanner kbScanner;

    private int round;

    public Bullseye() {

        gameState = GAME_STATE.STARTING;
        players = new ArrayList<>();

        // Save the random chances of points based on shot type

        shots = new Shot[3];
        shots[0] = new Shot(SHOT_ONE);
        shots[1] = new Shot(SHOT_TWO);
        shots[2] = new Shot(SHOT_THREE);

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
                case STARTING:
                    intro();
                    gameState = GAME_STATE.START_GAME;
                    break;

                // Start the game, set the number of players, names and round
                case START_GAME:

                    int numberOfPlayers = chooseNumberOfPlayers();

                    for (int i = 0; i < numberOfPlayers; i++) {
                        String name = displayTextAndGetInput("NAME OF PLAYER #" + (i + 1) + "? ");
                        Player player = new Player(name);
                        this.players.add(player);
                    }

                    this.round = 1;

                    gameState = GAME_STATE.PLAYING;
                    break;

                // Playing round by round until we have a winner
                case PLAYING:
                    System.out.println();
                    System.out.println("ROUND " + this.round);
                    System.out.println("=======");

                    // Each player takes their turn
                    for (Player player : players) {
                        int playerThrow = getPlayersThrow(player);
                        int points = calculatePlayerPoints(playerThrow);
                        player.addScore(points);
                        System.out.println("TOTAL SCORE = " + player.getScore());
                    }

                    boolean foundWinner = false;

                    // Check if any player won
                    for (Player thePlayer : players) {
                        int score = thePlayer.getScore();
                        if (score >= 200) {
                            if (!foundWinner) {
                                System.out.println("WE HAVE A WINNER!!");
                                System.out.println();
                                foundWinner = true;
                            }
                            System.out.println(thePlayer.getName() + " SCORED "
                                    + thePlayer.getScore() + " POINTS");
                        }
                    }

                    if (foundWinner) {
                        System.out.println("THANKS FOR THE GAME.");
                        gameState = GAME_STATE.GAME_OVER;
                    } else {
                        // No winner found, continue on with the next round
                        this.round++;
                    }

                    break;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    /**
     * Display info about the game
     */
    private void intro() {
        System.out.println("BULLSEYE");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET");
        System.out.println("WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS");
        System.out.println("TO GET 200 POINTS.");
        System.out.println();
        System.out.println(paddedString("THROW", "DESCRIPTION", "PROBABLE SCORE"));
        System.out.println(paddedString("1", "FAST OVERARM", "BULLSEYE OR COMPLETE MISS"));
        System.out.println(paddedString("2", "CONTROLLED OVERARM", "10, 20 OR 30 POINTS"));
        System.out.println(paddedString("3", "UNDERARM", "ANYTHING"));
    }

    /**
     * Calculate the players score
     * Score is based on the type of shot plus a random factor
     *
     * @param playerThrow 1,2, or 3 indicating the type of shot
     * @return player score
     */
    private int calculatePlayerPoints(int playerThrow) {

        // -1 is because of 0 base Java array
        double p1 = this.shots[playerThrow - 1].getShot(0);
        double p2 = this.shots[playerThrow - 1].getShot(1);
        double p3 = this.shots[playerThrow - 1].getShot(2);
        double p4 = this.shots[playerThrow - 1].getShot(3);

        double random = Math.random();

        int points;

        if (random >= p1) {
            System.out.println("BULLSEYE!!  40 POINTS!");
            points = 40;
            // If the throw was 1 (bullseye or missed, then make it missed
            // N.B. This is a fix for the basic code which for shot type 1
            // allowed a bullseye but did not make the score zero if a bullseye
            // was not made (which it should have done).
        } else if (playerThrow == 1) {
            System.out.println("MISSED THE TARGET!  TOO BAD.");
            points = 0;
        } else if (random >= p2) {
            System.out.println("30-POINT ZONE!");
            points = 30;
        } else if (random >= p3) {
            System.out.println("20-POINT ZONE");
            points = 20;
        } else if (random >= p4) {
            System.out.println("WHEW!  10 POINTS.");
            points = 10;
        } else {
            System.out.println("MISSED THE TARGET!  TOO BAD.");
            points = 0;
        }

        return points;
    }

    /**
     * Get players shot 1,2, or 3 - ask again if invalid input
     *
     * @param player the player we are calculating the throw on
     * @return 1, 2, or 3 indicating the players shot
     */
    private int getPlayersThrow(Player player) {
        boolean inputCorrect = false;
        String theThrow;
        do {
            theThrow = displayTextAndGetInput(player.getName() + "'S THROW ");
            if (theThrow.equals("1") || theThrow.equals("2") || theThrow.equals("3")) {
                inputCorrect = true;
            } else {
                System.out.println("INPUT 1, 2, OR 3!");
            }

        } while (!inputCorrect);

        return Integer.parseInt(theThrow);
    }


    /**
     * Get players guess from kb
     *
     * @return players guess as an int
     */
    private int chooseNumberOfPlayers() {

        return Integer.parseInt((displayTextAndGetInput("HOW MANY PLAYERS? ")));
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
     * Format three strings to a given number of spaces
     * Replacing the original basic code which used tabs
     *
     * @param first  String to print in pos 1
     * @param second String to print in pos 2
     * @param third  String to print in pos 3
     * @return formatted string
     */
    private String paddedString(String first, String second, String third) {
        String output = String.format("%1$" + FIRST_IDENT + "s", first);
        output += String.format("%1$" + SECOND_IDENT + "s", second);
        output += String.format("%1$" + THIRD_INDENT + "s", third);
        return output;
    }
}

```

# `18_Bullseye/java/src/BullseyeGame.java`

这段代码定义了一个名为BullseyeGame的公共类，其中包括一个名为main的静态方法，该方法接受一个字符串数组args作为参数，表示该类的其他部分需要传递给主方法的外部参数。

在main方法中，创建了一个名为Bullseye的类的实例变量，并将其赋值为一个新的Bullseye对象。然后，调用Bullseye对象的play()方法来让该对象表演一次" Bullseye "的游戏。

由于Bullseye是一个游戏，它的具体实现可能会因游戏而异。因此，该代码无法确定其具体的作用，但是可以看出它是一个命令，让用户可以运行一段游戏代码。


```
public class BullseyeGame {

    public static void main(String[] args) {

        Bullseye bullseye = new Bullseye();
        bullseye.play();
    }
}

```

# `18_Bullseye/java/src/Player.java`

这段代码定义了一个名为 `Player` 的类，表示游戏中的一个玩家。该类包含两个私有成员变量 `name` 和 `score`，分别表示玩家的姓名和得分。

构造函数 `Player()` 用于初始化这两个成员变量，其中 `name` 参数是一个字符串类型的字符串，表示玩家的姓名，而 `score` 参数是一个整数类型的整数，表示玩家的得分。

该类还定义了一个名为 `addScore()` 的方法，用于添加得分到 `score` 成员变量中。

另外，该类还定义了一个名为 `getName()` 的方法，用于返回玩家的姓名，以及一个名为 `getScore()` 的方法，用于返回玩家的得分。

总结起来，该代码定义了一个简单的玩家类，用于在游戏中记录玩家的信息，包括姓名和得分。玩家类中定义了一些方法来实现玩家信息的获取和修改操作。


```
/**
 * A Player in the game - consists of name and score
 *
 */
public class Player {

    private final String name;

    private int score;

    Player(String name) {
        this.name = name;
        this.score = 0;
    }

    public void addScore(int score) {
        this.score += score;
    }

    public String getName() {
        return name;
    }

    public int getScore() {
        return score;
    }
}

```

# `18_Bullseye/java/src/Shot.java`



这段代码定义了一个名为 Shot 的类，用于记录特定类型射击得分的可能性。它实现了 ArrayToBullet 类中的 pointsCalcation 方法，该方法接受一个整数数组和一个整数，表示每个样本的得分。

Shot 类有两个方法：

- `chances`：这是一个 double 数组，用于存储每个样本的得分机会。该数组的索引与输入的样本数相同，因此，如果 shots 数组长度为 N,`chances` 数组长度也为 N。
- `getShot(int index)`：它接受一个整数参数，表示要返回的样本。这个方法首先通过 `chances` 数组的中间索引(即索引为 N/2 对称的位置)获取该样本的得分，然后返回该得分。

通过 `Shot` 类，可以存储一系列不同类型的射击得分机会，并可以轻松地获取每个样本的得分。这非常有用，特别是在需要计算出射击得分的相对重要性时。


```
/**
 * This class records the percentage chance of a given type of shot
 * scoring specific points
 * see Bullseye class points calculation method where its used
 */
public class Shot {

    double[] chances;

    // Array of doubles are passed for a specific type of shot
    Shot(double[] shots) {
        chances = new double[shots.length];
        System.arraycopy(shots, 0, chances, 0, shots.length);
    }

    public double getShot(int index) {
        return chances[index];
    }
}

```

# `18_Bullseye/javascript/bullseye.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是接收一个字符串参数（`str`），将其输出到网页上的一个元素中。这个元素被称为`output`，在网页上是一个文本框。函数创建了一个`document.createTextNode()`对象来获取输入的字符串，将其添加到指定的元素中，并将其元素的`appendChild()`方法返回给调用者。

`input()`函数的作用是接收一个字符串参数（`str`），并返回一个Promise对象。函数会创建一个`document.createElement("INPUT")`对象来获取输入框，其中元素的`type`属性设置为"text"，`length`属性设置为"50"。它会将元素的`appendChild()`方法返回给调用者，然后设置元素的`focus()`方法，使得输入框可以被点击并获取输入。函数还定义了一个事件监听器，当输入框的`keydown`事件被触发时，监听器会获取当前的输入值（`event.key`），并将其存储在变量`input_str`中。最后，函数会将存储的输入值输出并输出一个换行符，并将其返回给调用者。


```
// BULLSEYE
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

This appears to be a Java program that simulates a game of target shooting. It randomly generates a score based on the position of a player (with the same number of players as the number of teams in the game), and then displays the score and the team that scored the most points.

The program has four conditions that determine the score based on the position of the player. If the player is in the center position, the program will calculate the score as 40 points for a goal, 30 points for a zone, 20 points for a zone, and 10 points for a goal. If the player is in the right position, the program will calculate the score as 3 points for a goal, 2 points for a long pass, 2 points for a short pass, and 1 point for a goal. If the player is in the left position, the program will calculate the score as 2 points for a goal, 1 point for a long pass, 1 point for a short pass, and 0 points for a goal.

The program also has a loop that continues until one of the teams reaches 200 points. At the end, it prints out the team that scored the most points, and then loops through the scores and prints out the team and the score.

It is important to note that this program is very basic and lacks some essentials that would make it more interesting like randomizing the score, introducing the goalpost, or limiting the number of shots a player can have etc.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var as = [];
var s = [];
var w = [];

// Main program
async function main()
{
    print(tab(32) + "BULLSEYE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET\n");
    print("WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS\n");
    print("TO GET 200 POINTS.\n");
    print("\n");
    print("THROW\t\tDESCRIPTION\t\tPROBABLE SCORE\n");
    print("1\t\tFAST OVERARM\t\tBULLSEYE OR COMPLETE MISS\n");
    print("2\t\tCONTROLLED OVERARM\t10, 20 OR 30 POINTS\n");
    print("3\t\tUNDERARM\t\tANYTHING\n");
    print("\n");
    m = 0;
    r = 0;
    for (i = 1; i <= 20; i++)
        s[i] = 0;
    print("HOW MANY PLAYERS");
    n = parseInt(await input());
    print("\n");
    for (i = 1; i <= n; i++) {
        print("NAME OF PLAYER #" + i);
        as[i] = await input();
    }
    do {
        r++;
        print("\n");
        print("ROUND " + r + "\n");
        print("---------\n");
        for (i = 1; i <= n; i++) {
            do {
                print("\n");
                print(as[i] + "'S THROW");
                t = parseInt(await input());
                if (t < 1 || t > 3)
                    print("INPUT 1, 2, OR 3!\n");
            } while (t < 1 || t > 3) ;
            if (t == 1) {
                p1 = 0.65;
                p2 = 0.55;
                p3 = 0.5;
                p4 = 0.5;
            } else if (t == 2) {
                p1 = 0.99;
                p2 = 0.77;
                p3 = 0.43;
                p4 = 0.01;
            } else {
                p1 = 0.95;
                p2 = 0.75;
                p3 = 0.45;
                p4 = 0.05;
            }
            u = Math.random();
            if (u >= p1) {
                print("BULLSEYE!!  40 POINTS!\n");
                b = 40;
            } else if (u >= p2) {
                print("30-POINT ZONE!\n");
                b = 30;
            } else if (u >= p3) {
                print("20-POINT ZONE\n");
                b = 20;
            } else if (u >= p4) {
                print("WHEW!  10 POINT.\n");
                b = 10;
            } else {
                print("MISSED THE TARGET!  TOO BAD.\n");
                b = 0;
            }
            s[i] += b;
            print("TOTAL SCORE = " + s[i] + "\n");
        }
        for (i = 1; i <= n; i++) {
            if (s[i] >= 200) {
                m++;
                w[m] = i;
            }
        }
    } while (m == 0) ;
    print("\n");
    print("WE HAVE A WINNER!!\n");
    print("\n");
    for (i = 1; i <= m; i++)
        print(as[w[i]] + " SCORED " + s[w[i]] + " POINTS.\n");
    print("\n");
    print("THANKS FOR THE GAME.\n");
}

```

这道题的代码是 `main()`，这是一个程序的入口函数。在程序中，`main()` 函数是程序执行的第一个函数，它定义了程序的总体行为。

`main()` 函数的作用是启动程序并负责程序的初始化、加载和配置。`main()` 函数的代码通常包括一些全局变量、函数指针、加载脚本文件等。

对于本题，由于没有提供具体的程序，我们无法进一步了解 `main()` 函数的具体作用。在实际程序中，`main()` 函数可能会有更加复杂的逻辑和功能。


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


# `18_Bullseye/python/bullseye.py`

这段代码是一个名为“Bullseye”的游戏，玩家需要向一个射箭靶扔出它们手中的飞镖，以赚取分数并尝试达到最高得分200分。

具体来说，每个玩家都有一个名字(通过“name”变量)，一个得分(通过“score”变量)，这两个变量都初始化为0。此外，定义了一个名为“Player”的类，继承自“dataclass”类，为玩家类添加了更多的属性和方法。

然后，定义了一个“print_intro”函数，这个函数用于输出游戏开始的介绍信息。通过“print”函数，在屏幕上输出了一段话，其中包括游戏相关的信息，例如玩家数量、游戏难度、射箭靶的得分范围等等。

最后，在程序的最后，定义了“main”函数，它会在程序开始时被调用。在“main”函数中，游戏会随机生成20个玩家，每个玩家会向射箭靶扔出3个飞镖，然后根据所得分数计算它们在游戏中的表现。如果某个玩家获得了最高得分200分，游戏会打印出一条消息并停止程序。


```
import random
from dataclasses import dataclass
from typing import List


@dataclass
class Player:
    name: str
    score: int = 0


def print_intro() -> None:
    print(" " * 32 + "BULLSEYE")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n" * 3, end="")
    print("IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET")
    print("WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS")
    print("TO GET 200 POINTS.")
    print()
    print("THROW", end="")
    print(" " * 20 + "DESCRIPTION", end="")
    print(" " * 45 + "PROBABLE SCORE")
    print(" 1", end="")
    print(" " * 20 + "FAST OVERARM", end="")
    print(" " * 45 + "BULLSEYE OR COMPLETE MISS")
    print(" 2", end="")
    print(" " * 20 + "CONTROLLED OVERARM", end="")
    print(" " * 45 + "10, 20 OR 30 POINTS")
    print(" 3", end="")
    print(" " * 20 + "UNDERARM", end="")
    print(" " * 45 + "ANYTHING")
    print()


```

This appears to be a Python game program that simulates a game of产出 (shooting pool) where players take turns throwing a ball into a pool and earning points for hitting the ball onto the pool using their名字 (optional).

The game has a total of 4 rounds, with each round having a different number of players participating. After each round, the players' score is added to a running total.

If a player manages to hit the ball onto the pool, they earn points, and their score is added to the running total. If the player misses the ball or hits the ball onto an obstacle, they lose a point, and their score is not updated.

The game also has a random element, with the ball always rolling a 1-sided die (1 or 2). The player's throwing luck is determined by the roll, with a probability of 1/2.

In the last round, the top 2 players (based on their score) earn points and the game ends.


```
def print_outro(players: List[Player], winners: List[int]) -> None:
    print()
    print("WE HAVE A WINNER!!")
    print()
    for winner in winners:
        print(f"{players[winner].name} SCORED {players[winner].score} POINTS.")
    print()
    print("THANKS FOR THE GAME.")


def main() -> None:
    print_intro()
    players: List[Player] = []

    winners: List[int] = []  # will point to indices of player_names

    nb_players = int(input("HOW MANY PLAYERS? "))
    for _ in range(nb_players):
        player_name = input("NAME OF PLAYER #")
        players.append(Player(player_name))

    round_number = 0
    while len(winners) == 0:
        round_number += 1
        print()
        print(f"ROUND {round_number}---------")
        for player in players:
            print()
            while True:
                throw = int(input(f"{player.name}'S THROW? "))
                if throw not in [1, 2, 3]:
                    print("INPUT 1, 2, OR 3!")
                else:
                    break
            if throw == 1:
                probability_1 = 0.65
                probability_2 = 0.55
                probability_3 = 0.5
                probability_4 = 0.5
            elif throw == 2:
                probability_1 = 0.99
                probability_2 = 0.77
                probability_3 = 0.43
                probability_4 = 0.01
            elif throw == 3:
                probability_1 = 0.95
                probability_2 = 0.75
                probability_3 = 0.45
                probability_4 = 0.05
            throwing_luck = random.random()
            if throwing_luck >= probability_1:
                print("BULLSEYE!!  40 POINTS!")
                points = 40
            elif throwing_luck >= probability_2:
                print("30-POINT ZONE!")
                points = 30
            elif throwing_luck >= probability_3:
                print("20-POINT ZONE")
                points = 20
            elif throwing_luck >= probability_4:
                print("WHEW!  10 POINTS.")
                points = 10
            else:
                print("MISSED THE TARGET!  TOO BAD.")
                points = 0
            player.score += points
            print(f"TOTAL SCORE = {player.score}")
        for player_index, player in enumerate(players):
            if player.score > 200:
                winners.append(player_index)

    print_outro(players, winners)


```

这段代码是一个条件判断语句，它的作用是当程序运行在命令行时（即不是脚本文件而是直接运行程序），执行if语句块内的代码。这里的作用是在程序运行时判断是否支持命令行参数，如果支持，则执行if语句块内的代码，否则跳过if语句块并继续执行。

具体来说，当程序直接运行时，会首先判断是否使用了命令行参数。如果是，则程序会检查命令行参数是否全部指定，如果全部指定，则会执行if语句块内的代码，否则会跳过if语句块并继续执行。如果命令行参数不指定，则程序不会执行if语句块内的代码，而是执行else语句块内的代码，即执行程序的主函数（如果没有定义主函数，则会自动创建一个主函数）。


```
if __name__ == "__main__":
    main()

```