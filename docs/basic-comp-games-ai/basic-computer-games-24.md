# BasicComputerGames源码解析 24

# `15_Boxing/java/Basic.java`

这段代码是一个名为 "Basic" 的类，它提供了一些基本的编程语言行为的模拟。

其中，*randomOf() 方法从 1 到给定整数 base 的随机整数。

*console.readLine() 方法从标准输入（通常是键盘）读取一行字符串，并将其返回。

*console.readInt() 方法从标准输入读取一个整数，并将其存储在变量中。如果读取整数时发生了错误（例如输入不是整数），此方法将返回 -1，并设置 failedInput 变量为 true。

*console.print() 方法从标准输出（通常是屏幕）读取字符串，将其格式化并打印到控制台。


```
import java.util.Scanner;

/**
 * It provide some kind of BASIC language behaviour simulations.
 */
final class Basic {

    public static int randomOf(int base) {
        return (int)Math.round(Math.floor(base* Math.random() + 1));
    }

    /**
     * The Console "simulate" the message error when input does not match with the expected type.
     * Specifically for this game if you enter an String when and int was expected.
     */
    public static class Console {
        private final Scanner input = new Scanner(System.in);

        public String readLine() {
            return input.nextLine();
        }

        public int readInt() {
            int ret = -1;
            boolean failedInput = true;
            do {
                boolean b = input.hasNextInt();
                if (b) {
                    ret = input.nextInt();
                    failedInput = false;
                } else {
                    input.next(); // discard read
                    System.out.print("!NUMBER EXPECTED - RETRY INPUT LINE\n? ");
                }

            } while (failedInput);

            return ret;
        }

        public void print(String message, Object... args) {
            System.out.printf(message, args);
        }
    }
}

```

# `15_Boxing/java/Boxing.java`

This appears to be a Java program that simulates a boxing match between two players. It starts by introducing the players and their abilities, then it explains the rules of the game. If one player is winning (based on their abilities), it prints a message and the game continues. Otherwise, it prints a message and the game ends. It also shows the names of the players and their countries, and it asks the player to input their name and the name of their opponent. It then explains the different punching options and the order in which they can be used. Finally, it shows the introduction message and the start button for the game.


```
/**
 * Boxing
 *
 * <p>
 * Based on the Basic game of BatNum here
 * https://github.com/coding-horror/basic-computer-games/tree/main/15%20Boxing
 * <p>
 */
public class Boxing {

    private static final Basic.Console console = new Basic.Console();

    private GameSession session;

    public void play() {
        showIntro();

        loadPlayers();

        console.print("%s'S ADVANTAGE IS %d AND VULNERABILITY IS SECRET.\n", session.getOpponent().getName(), session.getOpponent().getBestPunch().getCode());


        for (int roundNro = 1; roundNro <= 3; roundNro++) {
            if (session.isOver())
                break;

            session.resetPoints();
            console.print("\nROUND %d BEGINS...%n", roundNro);

            for (int majorPunches = 1; majorPunches <= 7; majorPunches++) {
                long i = Basic.randomOf(10);

                if (i > 5) {
                    boolean stopPunches = opponentPunch();
                    if (stopPunches ) break;
                } else {
                    playerPunch();
                }
            }
            showRoundWinner(roundNro);
        }
        showWinner();
    }

    private boolean opponentPunch() {
        final Punch punch = Punch.random();

        if (punch == session.getOpponent().getBestPunch()) session.addOpponentPoints(2);

        if (punch == Punch.FULL_SWING) {
            console.print("%s TAKES A FULL SWING AND", session.getOpponent().getName());
            long r6 = Basic.randomOf(60);

            if (session.getPlayer().hitVulnerability(Punch.FULL_SWING) || r6 < 30) {
                console.print(" POW!!!!! HE HITS HIM RIGHT IN THE FACE!\n");
                if (session.getPoints(session.getOpponent()) > 35) {

                    session.setKnocked();
                    return true;
                }
                session.addOpponentPoints(15);
            } else {
                console.print(" IT'S BLOCKED!\n");
            }
        }

        if (punch == Punch.HOOK  || punch == Punch.UPPERCUT) {
            if (punch == Punch.HOOK) {
                console.print("%s GETS %s IN THE JAW (OUCH!)\n", session.getOpponent().getName(), session.getPlayer().getName());

                session.addOpponentPoints(7);
                console.print("....AND AGAIN!\n");

                session.addOpponentPoints(5);
                if (session.getPoints(session.getOpponent()) > 35) {
                    session.setKnocked();
                    return true;
                }
                console.print("\n");

            }
            console.print("%s IS ATTACKED BY AN UPPERCUT (OH,OH)...\n", session.getPlayer().getName());
            long q4 = Basic.randomOf(200);
            if (session.getPlayer().hitVulnerability(Punch.UPPERCUT) || q4 <= 75) {
                console.print("AND %s CONNECTS...\n", session.getOpponent().getName());

                session.addOpponentPoints(8);
            } else {
                console.print(" BLOCKS AND HITS %s WITH A HOOK.\n", session.getOpponent().getName());

                session.addPlayerPoints(5);
            }
        }
        else {
            console.print("%s JABS AND ", session.getOpponent().getName());
            long z4 = Basic.randomOf(7);
            if (session.getPlayer().hitVulnerability(Punch.JAB))

                session.addOpponentPoints(5);
            else if (z4 > 4) {
                console.print(" BLOOD SPILLS !!!\n");

                session.addOpponentPoints(5);
            } else {
                console.print("IT'S BLOCKED!\n");
            }
        }
        return true;
    }

    private void playerPunch() {
        console.print("%s'S PUNCH? ", session.getPlayer().getName());
        final Punch punch = Punch.fromCode(console.readInt());

        if (punch == session.getPlayer().getBestPunch()) session.addPlayerPoints(2);

        switch (punch) {
            case FULL_SWING -> {
                console.print("%s SWINGS AND ", session.getPlayer().getName());
                if (session.getOpponent().getBestPunch() == Punch.JAB) {
                    console.print("HE CONNECTS!\n");
                    if (session.getPoints(session.getPlayer()) <= 35) session.addPlayerPoints(15);
                } else {
                    long x3 = Basic.randomOf(30);
                    if (x3 < 10) {
                        console.print("HE CONNECTS!\n");
                        if (session.getPoints(session.getPlayer()) <= 35) session.addPlayerPoints(15);
                    } else {
                        console.print("HE MISSES \n");
                        if (session.getPoints(session.getPlayer()) != 1) console.print("\n\n");
                    }
                }
            }
            case HOOK -> {
                console.print("\n%s GIVES THE HOOK... ", session.getPlayer().getName());
                long h1 = Basic.randomOf(2);
                if (session.getOpponent().getBestPunch() == Punch.HOOK) {

                    session.addPlayerPoints(7);
                } else if (h1 == 1) {
                    console.print("BUT IT'S BLOCKED!!!!!!!!!!!!!\n");
                } else {
                    console.print("CONNECTS...\n");

                    session.addPlayerPoints(7);
                }
            }
            case UPPERCUT -> {
                console.print("\n%s  TRIES AN UPPERCUT ", session.getPlayer().getName());
                long d5 = Basic.randomOf(100);
                if (session.getOpponent().getBestPunch() == Punch.UPPERCUT || d5 < 51) {
                    console.print("AND HE CONNECTS!\n");

                    session.addPlayerPoints(4);
                } else {
                    console.print("AND IT'S BLOCKED (LUCKY BLOCK!)\n");
                }
            }
            default -> {
                console.print("%s JABS AT %s'S HEAD \n", session.getPlayer().getName(), session.getOpponent().getName());
                if (session.getOpponent().getBestPunch() == Punch.JAB) {

                    session.addPlayerPoints(3);
                } else {
                    long c = Basic.randomOf(8);
                    if (c < 4) {
                        console.print("IT'S BLOCKED.\n");
                    } else {

                        session.addPlayerPoints(3);
                    }
                }
            }
        }
    }

    private void showRoundWinner(int roundNro) {
        if (session.isRoundWinner(session.getPlayer())) {
            console.print("\n %s WINS ROUND %d\n", session.getPlayer().getName(), roundNro);
            session.addRoundWind(session.getPlayer());
        } else {
            console.print("\n %s WINS ROUND %d\n", session.getOpponent().getName(), roundNro);
            session.addRoundWind(session.getOpponent());
        }
    }

    private void showWinner() {
        if (session.isGameWinner(session.getOpponent())) {
            console.print("%s WINS (NICE GOING, " + session.getOpponent().getName() + ").", session.getOpponent().getName());
        } else if (session.isGameWinner(session.getPlayer())) {
            console.print("%s AMAZINGLY WINS!!", session.getPlayer().getName());
        } else if (session.isPlayerKnocked()) {
            console.print("%s IS KNOCKED COLD AND %s IS THE WINNER AND CHAMP!", session.getPlayer().getName(), session.getOpponent().getName());
        } else {
            console.print("%s IS KNOCKED COLD AND %s IS THE WINNER AND CHAMP!", session.getOpponent().getName(), session.getPlayer().getName());
        }

        console.print("\n\nAND NOW GOODBYE FROM THE OLYMPIC ARENA.\n");
    }

    private void loadPlayers() {
        console.print("WHAT IS YOUR OPPONENT'S NAME? ");
        final String opponentName = console.readLine();

        console.print("INPUT YOUR MAN'S NAME? ");
        final String playerName = console.readLine();

        console.print("DIFFERENT PUNCHES ARE: (1) FULL SWING; (2) HOOK; (3) UPPERCUT; (4) JAB.\n");
        console.print("WHAT IS YOUR MANS BEST? ");

        final int b = console.readInt();

        console.print("WHAT IS HIS VULNERABILITY? ");
        final int d = console.readInt();

        final Player player = new Player(playerName, Punch.fromCode(b), Punch.fromCode(d));
        final Player opponent = new Player(opponentName);

        session = new GameSession(player, opponent);
    }

    private void showIntro () {
        console.print("                                 BOXING\n");
        console.print("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n");
        console.print("BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS)\n\n");
    }
}

```

# `15_Boxing/java/BoxingGame.java`

这段代码定义了一个名为 "BoxingGame" 的类，其中包含一个名为 "main" 的方法。

在 "main" 方法中，使用 "new" 关键字创建了一个名为 "Boxing" 的对象，并调用该对象的 "play" 方法。

具体来说，"play" 方法可能是一个方法，用于模拟拳击比赛的过程。它包括以下步骤：

1. 初始化拳击手和对手。
2. 处理拳击手和对手的行动。
3. 检查拳击是否发生。
4. 如果没有发生拳击，则重置计数器。

这里 "play" 方法的实现可能因拳击游戏的版本和设计而异。


```
public class BoxingGame {

    public static void main(String[] args) {
        new Boxing().play();
    }
}

```

# `15_Boxing/java/GameSession.java`

这段代码定义了一个名为 `GameSession` 的类，用于存储游戏会话的状态信息。这个类有两个实例变量 `player` 和 `opponent`，分别代表玩家的实例变量，另外还有两个计数器 `opponentRoundWins` 和 `playerRoundWins`，用于记录获胜的玩家 round 获胜次数。

在 `GameSession` 构造函数中，将 `player` 和 `opponent` 实例化并传入，以初始化两个计数器。

`GameSession` 还包含一些方法，比如 `resetPoints` 用于重置分数，`addPlayerPoints` 和 `addOpponentPoints` 分别用于添加玩家和对手的分数，`getPoints` 用于获取当前分数，`addRoundWind` 用于增加 round 获胜次数，`isOver` 用于判断游戏是否结束，`isRoundWinner` 和 `isGameWinner` 分别用于判断玩家是否获胜，以及判断游戏是否结束。

最后，`GameSession` 的实例化通常在游戏开始时调用，而不是在每次需要时都创建一个实例。


```
/**
 * Game Session
 * The session store the state of the game
 */
public class GameSession {
    private final Player player;
    private final Player opponent;
    private int opponentRoundWins = 0;
    private int playerRoundWins = 0;

    int playerPoints = 0;
    int opponentPoints = 0;
    boolean knocked = false;

    GameSession(Player player, Player opponent) {
        this.player = player;
        this.opponent = opponent;
    }

    public Player getPlayer() { return player;}
    public Player getOpponent() { return opponent;}

    public void setKnocked() {
        knocked = true;
    }

    public void resetPoints() {
        playerPoints = 0;
        opponentPoints = 0;
    }

    public void addPlayerPoints(int ptos) { playerPoints+=ptos;}
    public void addOpponentPoints(int ptos) { opponentPoints+=ptos;}

    public int getPoints(Player player) {
        if(player.isPlayer())
            return playerPoints;
        else
            return opponentPoints;
    }

    public void addRoundWind(Player player) {
        if(player.isPlayer()) playerRoundWins++; else opponentRoundWins++;
    }

    public boolean isOver() {
        return (opponentRoundWins >= 2 || playerRoundWins >= 2);
    }

    public boolean isRoundWinner(Player player) {
        if (player.isPlayer())
            return playerPoints > opponentPoints;
        else
            return opponentPoints > playerPoints;
    }

    public boolean isGameWinner(Player player) {
        if (player.isPlayer())
            return playerRoundWins > 2;
        else
            return opponentRoundWins > 2;
    }

    public boolean isPlayerKnocked() {
        return knocked;
    }
}

```

# `15_Boxing/java/Player.java`

这段代码定义了一个名为 `Player` 的类，用于代表玩家。该类包含玩家姓名、最优秀的拳击手和最脆弱的拳击手。

在 `Player` 类中，构造函数接受三个参数 `name`、`bestPunch` 和 `vulnerability`。这些参数用于初始化玩家的属性。

`isPlayer` 变量指示玩家身份，初始值为 `false`。

` hittingVulnerability` 方法用于检查玩家是否受到拳击士的最大防御力所对应的拳击士是否与玩家当前防御力相等。

` Basic.randomOf` 函数用于生成指定范围内的随机整数。在 `Player` 类中，`do` 循环用于生成四个随机整数，然后将这些随机整数作为 `b1` 和 `d1` 初始化 `bestPunch` 和 `vulnerability`。

`Punch.fromCode` 函数用于将字符串表示的拳击代码转换为实际拳击代码。


```
/**
 * The Player class model the user and compuer player
 */
public class Player {
    private final String name;
    private final Punch bestPunch;
    private final Punch vulnerability;
    private boolean isPlayer = false;

    public Player(String name, Punch bestPunch, Punch vulnerability) {
        this.name = name;
        this.bestPunch = bestPunch;
        this.vulnerability = vulnerability;
        this.isPlayer = true;
    }

    /**
     * Player with random Best Punch and Vulnerability
     */
    public Player(String name) {
        this.name = name;

        int b1;
        int d1;

        do {
            b1 = Basic.randomOf(4);
            d1 = Basic.randomOf(4);
        } while (b1 == d1);

        this.bestPunch = Punch.fromCode(b1);
        this.vulnerability = Punch.fromCode(d1);
    }

    public boolean isPlayer() { return isPlayer; }
    public String getName() { return  name; }
    public Punch getBestPunch() { return bestPunch; }

    public boolean hitVulnerability(Punch punch) {
        return vulnerability == punch;
    }
}

```

# `15_Boxing/java/Punch.java`

这段代码定义了一个枚举类型 `Punch`，表示四种不同类型的拳击动作，分别为 `FULL_SWING(1)`, `HOOK(2)`, `UPPERCUT(3)`, 和 `JAB(4)`。

`Punch` 类包括一个私有成员变量 `code`，表示拳击动作的编码，以及一个公有成员函数 `fromCode`，用于从给定的编码中返回 `Punch` 对象。

`fromCode` 函数使用 `Arrays.stream()` 方法将所有 `Punch` 对象转换为编码为整数的数组，并使用 `filter()` 方法筛选出编码等于给定编码的 `Punch` 对象，如果找不到匹配的对象，则返回 `null`。然后使用 `findAny()` 方法返回编码为整数的数组中的第一个对象，即 `Punch` 对象。

`random` 函数使用 `Basic.randomOf(4)` 方法生成一个随机的 `Punch` 对象，它的 `code` 属性被设置为 4。


```
import java.util.Arrays;

/**
 * Types of Punches
 */
public enum Punch {
    FULL_SWING(1),
    HOOK(2),
    UPPERCUT(3),
    JAB(4);

    private final int code;

    Punch(int code) {
        this.code = code;
    }

    int getCode() { return  code;}

    public static Punch fromCode(int code) {
        return Arrays.stream(Punch.values()).filter(p->p.code == code).findAny().orElse(null);
    }

    public static Punch random() {
        return Punch.fromCode(Basic.randomOf(4));
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `15_Boxing/javascript/boxing.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是将一个字符串打印到页面上，并在字符串前添加了"output"元素的样式。具体实现是通过在文档中创建一个`<textarea>`元素，然后将字符串插入到该元素中，最后将元素添加到`document.getElementById("output")`上。

`input()`函数的作用是从用户那里获取输入值，并在获取到用户输入后将其存储在变量`input_str`中。具体实现是通过创建一个`<input>`元素，设置其`type`属性为"text"，`length`属性为"50"，并将`input_str`存储到该元素的`value`属性中。然后，该元素的焦点被设置，当用户按下键盘上的`13`键时，将存储在`input_str`中的字符串打印到页面上，并将其从`document.getElementById("output")`中删除，最后将字符串打印出来。


```
// BOWLING
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

It looks like you are trying to write a program that generates text描述符， which is a common task in natural language processing and is related to the OPLAN project. The text describes符主要包括用于描述游戏内战斗的结果， such as "AND"，"BLOCK"，"HIT"，"CONNECT"。

The program generates text by reading a HTML file that contains information about the game matches, including the number of users in a certain game mode, the number of matches in a certain game mode, and the results of each match. The program then generates text based on the information in the HTML file.

The text describes符的生成过程 includes several steps, including reading the HTML file, parsing the data, and generating the text. The program reads the HTML file using the `HTMLParser` class, which is not included in the code you provided. If you have any interest in learning more about natural language processing and the OPLAN project, I would recommend checking out the OPLAN project website, which provides more information about the project and its goals.


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
    print(tab(33) + "BOXING\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS)\n");
    j = 0;
    l = 0;
    print("\n");
    print("WHAT IS YOUR OPPONENT'S NAME");
    js = await input();
    print("INPUT YOUR MAN'S NAME");
    ls = await input();
    print("DIFFERENT PUNCHES ARE: (1) FULL SWING; (2) HOOK; (3) UPPERCUT; (4) JAB.\n");
    print("WHAT IS YOUR MANS BEST");
    b = parseInt(await input());
    print("WHAT IS HIS VULNERABILITY");
    d = parseInt(await input());
    do {
        b1 = Math.floor(4 * Math.random() + 1);
        d1 = Math.floor(4 * Math.random() + 1);
    } while (b1 == d1) ;
    print(js + "'S ADVANTAGE IS " + b1 + " AND VULNERABILITY IS SECRET.\n");
    print("\n");
    knocked = 0;
    for (r = 1; r <= 3; r++) {
        if (j >= 2)
            break;
        if (l >= 2)
            break;
        x = 0;
        y = 0;
        print("ROUND " + r + " BEGIN...\n");
        for (r1 = 1; r1 <= 7; r1++) {
            i = Math.floor(10 * Math.random() + 1);
            if (i <= 5) {
                print(ls + "'S PUNCH");
                p = parseInt(await input());
                if (p == b)
                    x += 2;
                if (p == 1) {
                    print(ls + " SWINGS AND ");
                    x3 = Math.floor(30 * Math.random() + 1);
                    if (d1 == 4 || x3 < 10) {
                        print("HE CONNECTS!\n");
                        if (x > 35) {
                            r = 3;
                            break;
                        }
                        x += 15;
                    } else {
                        print("HE MISSES \n");
                        if (x != 1)
                            print("\n\n");
                    }
                } else if (p == 2) {
                    print(ls + " GIVES THE HOOK... ");
                    h1 = Math.floor(2 * Math.random() + 1);
                    if (d1 == 2) {
                        x += 7;
                    } else if (h1 != 1) {
                        print("CONNECTS...\n");
                        x += 7;
                    } else {
                        print("BUT IT'S BLOCKED!!!!!!!!!!!!!\n");
                    }
                } else if (p == 3) {
                    print(ls + " TRIES AN UPPERCUT ");
                    d5 = Math.floor(100 * Math.random() + 1);
                    if (d1 == 3 || d5 < 51) {
                        print("AND HE CONNECTS!\n");
                        x += 4;
                    } else {
                        print("AND IT'S BLOCKED (LUCKY BLOCK!)\n");
                    }
                } else {
                    print(ls + " JABS AT " + js + "'S HEAD ");
                    c = Math.floor(8 * Math.random() + 1);
                    if (d1 == 4 || c >= 4) {
                        x += 3;
                    } else {
                        print("IT'S BLOCKED.\n");
                    }
                }
            } else {
                j7 = Math.random(4 * Math.random() + 1);
                if (j7 == b1)
                    y += 2;
                if (j7 == 1) {
                    print(js + " TAKES A FULL SWING AND");
                    r6 = Math.floor(60 * Math.random() + 1);
                    if (d == 1 || r6 < 30) {
                        print(" POW!!!!! HE HITS HIM RIGHT IN THE FACE!\n");
                        if (y > 35) {
                            knocked = 1;
                            r = 3;
                            break;
                        }
                        y += 15;
                    } else {
                        print(" IT'S BLOCKED!\n");
                    }
                } else if (j7 == 2 || j7 == 3) {
                    if (j7 == 2) {
                        print(js + " GETS " + ls + " IN THE JAW (OUCH!)\n");
                        y += 7;
                        print("....AND AGAIN!\n");
                        y += 5;
                        if (y > 35) {
                            knocked = 1;
                            r = 3;
                            break;
                        }
                        print("\n");
                        // From original, it goes over from handling 2 to handling 3
                    }
                    print(ls + " IS ATTACKED BY AN UPPERCUT (OH,OH)...\n");
                    q4 = Math.floor(200 * Math.random() + 1);
                    if (d == 3 || q4 <= 75) {
                        print("AND " + js + " CONNECTS...\n");
                        y += 8;
                    } else {
                        print(" BLOCKS AND HITS " + js + " WITH A HOOK.\n");
                        x += 5;
                    }
                } else {
                    print(js + " JABS AND ");
                    z4 = Math.floor(7 * Math.random() + 1);
                    if (d == 4)
                        y += 5;
                    else if (z4 > 4) {
                        print(" BLOOD SPILLS !!!\n");
                        y += 5;
                    } else {
                        print("IT'S BLOCKED!\n");
                    }
                }
            }
        }
        if (x > y) {
            print("\n");
            print(ls + " WINS ROUND " + r + "\n");
            l++;
        } else {
            print("\n");
            print(js + " WINS ROUND " + r + "\n");
            j++;
        }
    }
    if (j >= 2) {
        print(js + " WINS (NICE GOING, " + js + ").\n");
    } else if (l >= 2) {
        print(ls + " AMAZINGLY WINS!!\n");
    } else if (knocked) {
        print(ls + " IS KNOCKED COLD AND " + js + " IS THE WINNER AND CHAMP!\n");
    } else {
        print(js + " IS KNOCKED COLD AND " + ls + " IS THE WINNER AND CHAMP!\n");
    }
    print("\n");
    print("\n");
    print("AND NOW GOODBYE FROM THE OLYMPIC ARENA.\n");
    print("\n");
}

```

这道题目是一个简单的C语言程序，包含一个名为“main”的函数。程序的主要目的是在命令行界面中输出“Hello, World!”。

“main”函数的具体作用如下：

1. 首先，将程序计数器（PC）的值设置为1，以便跟踪程序的执行情况。

2. 进入一个无限循环，每当遇到一个“break”语句时，循环将会终止。

3. 在循环的每一次中，使用printf函数输出字符“%c”。

4. 输出完“%c”后，程序计数器（PC）会自增1，以便下一次循环时能正确找到输出的起点。

5. 循环会一直执行下去，直到遇到一个没有“break”语句的标签，或者输出完所有的字符后，计数器（PC）变成0，此时程序将终止。

整个程序简单明了，但需要注意的是，在实际应用中，熟练掌握“break”语句可以帮助我们优化程序的性能，减少不必要的运行时间。


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


# `15_Boxing/python/boxing.py`

这段代码定义了一个名为 `PunchProfile` 的类，它是一个 `dataclass` 类型的数据定义。这个类包含了一些可读性很强的属性，包括 `choices`、`threshold`、`hit_damage`、`block_damage` 等等，这些属性都是用来描述 punch 游戏中玩家行动的。

此外，这个类还包含了一些用于 determining whether a player has been hit or not 的方法，包括 `is_hit` 方法。这个方法使用了 `random` 模块中的 `randint` 函数来生成一个 0 到 `choices` 之间的随机整数，如果这个随机整数小于或等于 `threshold`，则说明这个玩家没有被打中。

最后，这个类还定义了一个 `PunchProfile` 类的 `__init__` 方法，用于初始化这个类的各个属性，以及一个 `__str__` 方法，用于将这个类的对象转换成字符串形式，以便在输出时进行显示。


```
#!/usr/bin/env python3
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, NamedTuple, Tuple


class PunchProfile(NamedTuple):
    choices: int
    threshold: int
    hit_damage: int
    block_damage: int

    pre_msg: str
    hit_msg: str
    blocked_msg: str

    knockout_possible: bool = False

    def is_hit(self) -> bool:
        return random.randint(1, self.choices) <= self.threshold


```

这是一个定义了一个 `Player` 类的作用于表示玩家和他们的攻击类型、伤害以及击中的概率。其中，攻击类型有 4 种，每种攻击类型都有一个 `PunchProfile` 类来描述该攻击的详细信息，例如攻击伤害、概率、是否是电脑发送的等等。

`Player` 类包含了一些基本属性，例如 `name`、`best`、`weakness`、`is_computer` 和 `damage`、`score` 和 `knockedout`，它们用于表示玩家的基本信息、攻击力、防御力、击中概率、伤害输出、是否击中以及是否击倒等。

`get_punch_choice` 方法用于让玩家选择一次攻击，根据 `is_computer` 属性，如果是电脑发送的攻击，就随机选择攻击类型，否则每次攻击选项减少一个，直到选择成功或全部分析完畢。


```
@dataclass
class Player:
    name: str
    best: int  # this hit guarantees 2 damage on opponent
    weakness: int  # you're always hit when your opponent uses this punch
    is_computer: bool

    # for each of the 4 punch types, we have a probability of hitting
    punch_profiles: Dict[Literal[1, 2, 3, 4], PunchProfile]

    damage: int = 0
    score: int = 0
    knockedout: bool = False

    def get_punch_choice(self) -> Literal[1, 2, 3, 4]:
        if self.is_computer:
            return random.randint(1, 4)  # type: ignore
        else:
            punch = -1
            while punch not in [1, 2, 3, 4]:
                print(f"{self.name}'S PUNCH", end="? ")
                punch = int(input())
            return punch  # type: ignore


```

这段代码是一个用于解决KNOCKOUT游戏的的人工智能程序。它包括了以下两个主要函数：

1. get_vulnerability()函数，该函数的作用是获取玩家的脆弱性（即玩家在游戏中的表现水平，比如得分、胜利率等）。它首先向玩家询问关于他们自己的脆弱性，然后返回玩家的脆弱性值，这个值在1到4之间。

2. get_opponent_stats()函数，该函数的作用是获取玩家和对手的统计数据。它包括两个变量，一个是玩家在游戏中的得分，另一个是玩家和对手中的一个胜者和一个失败者。它首先从1到4中随机选择一个数字作为玩家最好的得分，然后从1到4中随机选择一个数字作为玩家和对手中的一个胜者和一个失败者。函数返回玩家最好的得分和玩家和对手中的一个胜者和一个失败者。


```
KNOCKOUT_THRESHOLD = 35

QUESTION_PROMPT = "? "
KNOCKED_COLD = "{loser} IS KNOCKED COLD AND {winner} IS THE WINNER AND CHAMP"


def get_vulnerability() -> int:
    print("WHAT IS HIS VULNERABILITY", end=QUESTION_PROMPT)
    vulnerability = int(input())
    return vulnerability


def get_opponent_stats() -> Tuple[int, int]:
    opponent_best = 0
    opponent_weakness = 0
    while opponent_best == opponent_weakness:
        opponent_best = random.randint(1, 4)
        opponent_weakness = random.randint(1, 4)
    return opponent_best, opponent_weakness


```

It looks like this is a Python program that simulates a boxing match between two players. The program uses the `Player` class to represent each player, which has attributes such as their name, best fighting style, and weakness. It also uses the `get_vulnerability` function to determine an opponent's weakness and the `get_opponent_stats` function to determine an opponent's best fighting style and weakness.

The program has three rounds of simulated boxing and in the end, the program outputs the winner and the knockout status of each player. It also outputs a goodbye message from the Olympic Arena.


```
def read_punch_profiles(filepath: Path) -> Dict[Literal[1, 2, 3, 4], PunchProfile]:
    with open(filepath) as f:
        punch_profile_dict = json.load(f)
    result = {}
    for key, value in punch_profile_dict.items():
        result[int(key)] = PunchProfile(**value)
    return result  # type: ignore


def main() -> None:
    print("BOXING")
    print("CREATIVE COMPUTING   MORRISTOWN, NEW JERSEY")
    print("\n\n")
    print("BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS)")

    print("WHAT IS YOUR OPPONENT'S NAME", end=QUESTION_PROMPT)
    opponent_name = input()
    print("WHAT IS YOUR MAN'S NAME", end=QUESTION_PROMPT)
    player_name = input()

    print("DIFFERENT PUNCHES ARE 1 FULL SWING 2 HOOK 3 UPPERCUT 4 JAB")
    print("WHAT IS YOUR MAN'S BEST", end=QUESTION_PROMPT)
    player_best = int(input())  # noqa: TODO - this likely is a bug!
    player_weakness = get_vulnerability()
    player = Player(
        name=player_name,
        best=player_best,
        weakness=player_weakness,
        is_computer=False,
        punch_profiles=read_punch_profiles(
            Path(__file__).parent / "player-profile.json"
        ),
    )

    opponent_best, opponent_weakness = get_opponent_stats()
    opponent = Player(
        name=opponent_name,
        best=opponent_best,
        weakness=opponent_weakness,
        is_computer=True,
        punch_profiles=read_punch_profiles(
            Path(__file__).parent / "opponent-profile.json"
        ),
    )

    print(
        f"{opponent.name}'S ADVANTAGE is {opponent.weakness} AND VULNERABILITY IS SECRET."
    )

    for round_number in (1, 2, 3):
        play_round(round_number, player, opponent)

    if player.knockedout:
        print(KNOCKED_COLD.format(loser=player.name, winner=opponent.name))
    elif opponent.knockedout:
        print(KNOCKED_COLD.format(loser=opponent.name, winner=player.name))
    elif opponent.score > player.score:
        print(f"{opponent.name} WINS (NICE GOING), {player.name}")
    else:
        print(f"{player.name} AMAZINGLY WINS")

    print("\n\nAND NOW GOODBYE FROM THE OLYMPIC ARENA.")


```

这段代码定义了一个函数 `is_opponents_turn()` 和一个函数 `play_round` 用于在玩家和对手之间进行回合制游戏。

函数 `is_opponents_turn()` 是一个布尔函数，它使用随机数生成器生成一个介于 1 和 10 之间的随机整数，如果生成的数字大于 5，那么它的返回值就是 `True`，否则就是 `False`。

函数 `play_round()` 是 `play_round` 函数的入口函数，它接收一个游戏轮数 `round_number`、玩家 `player` 和对手 `opponent` 对象作为参数。函数首先打印出当前游戏轮数，然后进行游戏循环。

在游戏循环中，函数 `for_action` 会循环 7 次，每次循环都会执行以下操作：

1. 如果当前是 opponents 的 turn，那么就执行 opponents 的行动操作，即获取 opponents 选择的攻击动作，然后判断是否是最佳动作并计算攻击效果。
2. 如果当前是 player 的 turn，那么就执行 player 的行动操作，即获取 player 选择的攻击动作，然后判断是否是最佳动作并计算攻击效果。
3. 如果对手的得分大于或等于 2 或 player 的得分大于或等于 2，那么就直接返回，不执行任何行动操作。
4. 否则就执行 round_number 次游戏循环内的行动操作。

函数 `play_round()` 最终的结果是赢得游戏的一方所得到的分数，如果平局，则返回。


```
def is_opponents_turn() -> bool:
    return random.randint(1, 10) > 5


def play_round(round_number: int, player: Player, opponent: Player) -> None:
    print(f"ROUND {round_number} BEGINS...\n")
    if opponent.score >= 2 or player.score >= 2:
        return

    for _action in range(7):
        if is_opponents_turn():
            punch = opponent.get_punch_choice()
            active = opponent
            passive = player
        else:
            punch = player.get_punch_choice()
            active = player
            passive = opponent

        # Load the hit characteristics of the current player's punch
        punch_profile = active.punch_profiles[punch]

        if punch == active.best:
            passive.damage += 2

        print(punch_profile.pre_msg.format(active=active, passive=passive), end=" ")
        if passive.weakness == punch or punch_profile.is_hit():
            print(punch_profile.hit_msg.format(active=active, passive=passive))
            if punch_profile.knockout_possible and passive.damage > KNOCKOUT_THRESHOLD:
                passive.knockedout = True
                break
            passive.damage += punch_profile.hit_damage
        else:
            print(punch_profile.blocked_msg.format(active=active, passive=passive))
            active.damage += punch_profile.block_damage

    if player.knockedout or opponent.knockedout:
        return
    elif player.damage > opponent.damage:
        print(f"{opponent.name} WINS ROUND {round_number}")
        opponent.score += 1
    else:
        print(f"{player.name} WINS ROUND {round_number}")
        player.score += 1


```

这段代码是一个条件判断语句，它用于检查当前程序是否作为主程序运行。如果程序作为主程序运行，那么程序会执行if语句中的代码。

if语句是一个用于检查条件的语句，它的语法是“条件 == 表达式”。如果条件成立，那么表达式的值会被替换为True，否则表达式的值被保留为原来的值。

在这段代码中，if语句的检查对象是__name__，这是一个特殊的环境变量，用于检查当前程序是否作为主程序运行。如果__name__的值为"__main__"，那么程序将执行if语句中的代码，否则程序不会执行if语句中的代码。

这段代码的作用是用于确保程序在作为主程序运行时才执行if语句中的代码，否则程序将不会执行if语句中的代码。


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


### Bug

The object of this game is to finish your drawing of a bug before the computer finishes.

You and the computer roll a die alternately with each number standing for a part of the bug. You must add the parts in the right order; in other words, you cannot have a neck until you have a body, you cannot have a head until you have a neck, and so on. After each new part has been added, you have the option of seeing pictures of the two bugs.

If you elect to see all the pictures, this program has the ability of consuming well over six feet of terminal paper per run. We can only suggest recycling the paper by using the other side.

Brian Leibowitz wrote this program while in the 7th grade at Harrison Jr-Se High School in Harrison, New York.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=30)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=45)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `16_Bug/csharp/Bug.cs`



这段代码定义了一个名为Bug的类，具有以下几个方法：

1. `using System.Text` - 引入了System.Text命名空间中的类和接口，以便能够使用其中定义的文本类型。

2. `using BugGame.Parts` - 引入了BugGame.Parts命名空间中的类和接口，以便能够使用其中定义的游戏部分类和接口。

3. `using BugGame.Resources` - 引入了BugGame.Resources命名空间中的类和接口，以便能够使用其中定义的游戏资源类和接口。

4. ` internal class Bug` - 定义了Bug类的层次结构，继承自BugGame.Parts.Internal，表示它是一个内部类，用于实现游戏中的Bug。

5. `private readonly Body _body = new()` - 定义了一个名为_body的私有Body属性，用于保存Bug类的实例身体，它是一个BugGame.Parts.Body类型的引用。

6. `public bool IsComplete => _body.IsComplete` - 定义了一个名为IsComplete的公共布尔类型变量，用于表示Bug是否已完成，它使用_body.IsComplete属性来获取身体是否已完成。

7. `public bool TryAdd(IPart part, out Message message) => _body.TryAdd(part, out message)` - 定义了一个名为TryAdd的公共方法，用于尝试将给定的部分添加到身体中，返回一个Message类型的变量。它使用_body.TryAdd(part, out message)方法来实现。

8. `public string ToString(string pronoun, char feelerCharacter)` - 定义了一个名为ToString的公共方法，用于将Bug转换为指定的格式的字符串。它使用Builder类来构建字符串，并使用feelerCharacter参数来指定格子中应该显示的字符。

9. `private void UpdateBody(IPart part, string[] newBody)` - 私有方法，用于在每次迭代中更新Bug类的实例身体。它接受两个参数，第一个是part，表示要更新的部分，第二个是新的Body，表示新的身体部分。它将新的Body替换为新的身体部分，并返回更新后的身体。


```
using System.Text;
using BugGame.Parts;
using BugGame.Resources;

namespace BugGame;

internal class Bug
{
    private readonly Body _body = new();

    public bool IsComplete => _body.IsComplete;

    public bool TryAdd(IPart part, out Message message) => _body.TryAdd(part, out message);

    public string ToString(string pronoun, char feelerCharacter)
    {
        var builder = new StringBuilder($"*****{pronoun} Bug*****").AppendLine().AppendLine().AppendLine();
        _body.AppendTo(builder, feelerCharacter);
        return builder.ToString();
    }
}
```

# `16_Bug/csharp/Game.cs`

This appears to be a class that simulates the build process of a software program, including the installation of various bug fixes. The class includes a method called `TryBuild()` that takes a `Bug` object and a message transform function as input, and returns a `True` or `False` depending on whether the build was successful.

The `TryBuild()` method first generates a random roll value between 1 and 6, and uses that to select the type of bug to install. It then creates an instance of the bug and passes it to the `TryAdd()` method of the `Bug` class, which installs the bug and returns a `True` or `False` depending on the result.

Finally, the method uses the `Invoke()` method of the `messageTransform` function to get the message as written by the `TryBuild()` method, and returns the result of that method.


```
using BugGame.Parts;
using BugGame.Resources;
using Games.Common.IO;
using Games.Common.Randomness;
using static System.StringComparison;
namespace BugGame;

internal class Game
{
    private readonly IReadWrite _io;
    private readonly IRandom _random;

    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    public void Play()
    {
        _io.Write(Resource.Streams.Introduction);
        if (!_io.ReadString("Do you want instructions").Equals("no", InvariantCultureIgnoreCase))
        {
            _io.Write(Resource.Streams.Instructions);
        }

        BuildBugs();

        _io.Write(Resource.Streams.PlayAgain);
    }

    private void BuildBugs()
    {
        var yourBug = new Bug();
        var myBug = new Bug();

        while (true)
        {
            var partAdded = TryBuild(yourBug, m => m.You);
            Thread.Sleep(500);
            _io.WriteLine();
            partAdded |= TryBuild(myBug, m => m.I);

            if (partAdded)
            {
                if (yourBug.IsComplete) { _io.WriteLine("Your bug is finished."); }
                if (myBug.IsComplete) { _io.WriteLine("My bug is finished."); }

                if (!_io.ReadString("Do you want the picture").Equals("no", InvariantCultureIgnoreCase))
                {
                    _io.Write(yourBug.ToString("Your", 'A'));
                    _io.WriteLine();
                    _io.WriteLine();
                    _io.WriteLine();
                    _io.WriteLine();
                    _io.Write(myBug.ToString("My", 'F'));
                }
            }

            if (yourBug.IsComplete || myBug.IsComplete) { break; }
        }
    }

    private bool TryBuild(Bug bug, Func<Message, string> messageTransform)
    {
        var roll = _random.Next(6) + 1;
        _io.WriteLine(messageTransform(Message.Rolled.ForValue(roll)));

        IPart part = roll switch
        {
            1 => new Body(),
            2 => new Neck(),
            3 => new Head(),
            4 => new Feeler(),
            5 => new Tail(),
            6 => new Leg(),
            _ => throw new Exception("Unexpected roll value")
        };
        _io.WriteLine($"{roll}={part.GetType().Name}");

        var partAdded = bug.TryAdd(part, out var message);
        _io.WriteLine(messageTransform.Invoke(message));

        return partAdded;
    }
}
```

# `16_Bug/csharp/Program.cs`

这段代码是一个JavaScript游戏引擎中的主要函数，用于启动并运行一个名为“BugGame”的游戏实例。下面是每行代码的作用：

1. `using BugGame;` 导入“BugGame”游戏引擎的命名空间。
2. `using Games.Common.IO;` 导入“Games.Common.IO”命名空间，可能是用于从标准输入/输出（通常是终端）读取/写入数据。
3. `using Games.Common.Randomness;` 导入“Games.Common.Randomness”命名空间，可能是用于生成随机数。
4. `new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();` 创建一个名为“Game”的类实例，该类实例可能继承自另一个类“ConsoleGame”或“GameplayGame”。这个类应该包含游戏的一些核心功能，例如玩 start 方法。
5. `.` 表示这是一个 start 方法，即将游戏引擎启动并运行。
6. `using System;` 导入“System”命名空间，可能是用于支持通用编程概念。
7. `class Program` 定义了一个名为“Program”的类，继承自“System.Program”。
8. `{` 和 `}` 分别表示一个类和其成员的文档字符串。
9. `int main(string[] args)` 是该程序的主函数，程序从这里开始执行。
10. `{` 和 `}` 分别表示一个类和其成员的文档字符串。
11. `static void Main(string[] args)` 是该程序的主函数，程序从这里开始执行。
12. `int x = 10;` 在程序的内存中定义了一个整数类型的变量“x”，并将其值设为 10。
13. `int y = 5;` 在程序的内存中定义了一个整数类型的变量“y”，并将其值设为 5。
14. `void ShowBug()` 是一个静态方法，用于显示一个额外的缺陷或错误。
15. `static void ShowBug()` 是该方法的作用域，即该方法只能在定义它的程序中访问。
16. `}` 表示方法文档字符串的结束。
17. `static void ShowBug()` 是该方法的作用域，即该方法只能在定义它的程序中访问。
18. `}` 表示类文档字符串的结束。
19. `}` 表示程序文档字符串的结束。


```
using BugGame;
using Games.Common.IO;
using Games.Common.Randomness;

new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `16_Bug/csharp/Parts/Body.cs`

这段代码是一个名为 Body 的类，继承自 ParentPart 类。它用于表示一个生物体的各个部分，如头、颈、身体和腿等。

首先，它定义了一个颈和一个尾巴，分别继承自 Neck 和 Tail 类。然后，定义了一个本类的一个构造函数，用于初始化身体的各个部分。

接着，重写了 IsComplete 方法，用于判断身体是否已经准备就绪。在 TryAddCore 方法中，尝试将每个部分添加到身体中。如果 neck、head 或 feeler 类中任何一部分成功添加，则会生成一个消息并输出到控制台。

最后，定义了一个 AppendTo 方法，用于将部分添加到生成的字符串中。在实例中，如果身体已经存在，则会先添加 neck，然后添加 tail，接着添加 legs，最后添加其他部分。这样，生成的字符串就是该生物体的描述。


```
using System.Text;
using BugGame.Resources;

namespace BugGame.Parts;

internal class Body : ParentPart
{
    private readonly Neck _neck = new();
    private readonly Tail _tail = new();
    private readonly Legs _legs = new();

    public Body()
        : base(Message.BodyAdded, Message.BodyNotNeeded)
    {
    }

    public override bool IsComplete => _neck.IsComplete && _tail.IsComplete && _legs.IsComplete;

    protected override bool TryAddCore(IPart part, out Message message)
        => part switch
        {
            Neck => _neck.TryAdd(out message),
            Head or Feeler => _neck.TryAdd(part, out message),
            Tail => _tail.TryAdd(out message),
            Leg => _legs.TryAddOne(out message),
            _ => throw new NotSupportedException($"Can't add a {part.Name} to a {Name}.")
        };

    public void AppendTo(StringBuilder builder, char feelerCharacter)
    {
        if (IsPresent)
        {
            _neck.AppendTo(builder, feelerCharacter);
            builder
                .AppendLine("     BBBBBBBBBBBB")
                .AppendLine("     B          B")
                .AppendLine("     B          B");
            _tail.AppendTo(builder);
            builder
                .AppendLine("     BBBBBBBBBBBB");
            _legs.AppendTo(builder);
        }
    }
}

```

# `16_Bug/csharp/Parts/Feeler.cs`

这段代码定义了一个名为"Feeler"的内部类，它继承自名为"IPart"的接口。

这个内部类实现了"IPart"接口中的"Name"属性，这个属性返回的是对象自己的名称(即类名)。因此，这个内部类的实例可以被命名为"Feeler"，这样它的名称就是"Feeler"。

这个代码片段可能是在一个更大的程序中定义的类或接口的实例。由于缺少上下文，无法确定这个"BugGame.Parts"命名空间中包含这个类的定义。


```
namespace BugGame.Parts;

internal class Feeler : IPart
{
    public string Name => nameof(Feeler);
}

```

# `16_Bug/csharp/Parts/Feelers.cs`

这段代码定义了一个名为 `Feelers` 的类，属于 `BugGame.Parts` 命名空间。

该类继承自 `PartCollection` 类，其中包含一个构造函数、一个方法 `AppendTo`、一个 `Feeler` 成员变量以及一个 `Message.FeelerAdded` 和 `Message.FeelersFull` 的事件。

构造函数初始化 `Feeler` 对象时，传递一个包含两个参数的字符串和四个字符，表示该对象的四个属性。这些属性似乎与代码中定义的方法和事件没有直接关系，因此具体情况可能需要进一步了解。

`AppendTo` 方法用于将一个字符添加到字符串中。方法接受一个字符串参数 `builder`、一个字符参数 `character` 和一个整数参数 `count`。这个字符被添加到字符串中的位置从 `count` 开始，以从左到右的顺序依次添加。

该类的作用似乎是为了解决某个问题，但具体情况还需要进一步了解。


```
using System.Text;
using BugGame.Resources;

namespace BugGame.Parts;

internal class Feelers : PartCollection
{
    public Feelers()
        : base(2, Message.FeelerAdded, Message.FeelersFull)
    {
    }

    public void AppendTo(StringBuilder builder, char character) => AppendTo(builder, 10, 4, character);
}

```

# `16_Bug/csharp/Parts/Head.cs`



这段代码是一个类名为 `Head` 的继承自 `ParentPart` 的物品类，用于表示游戏中的头部。

在这个类中，`Feelers` 类是一个私有的 `Feelers` 类型，用于跟踪当前头部的状态。

`Head` 类继承自 `ParentPart` 类，并重写了 `IsComplete` 和 `TryAddCore` 方法，用于检测是否已经添加完整和尝试将部分添加到头部。

`AppendTo` 方法用于将当前头部上的所有感觉器(Feeler)添加到游戏引擎的字符串中。

最后，在构造函数中，`Head` 类将 `Message.HeadAdded` 和 `Message.HeadNotNeeded` 作为超参数传递给父类 `ParentPart` 的构造函数。


```
using System.Text;
using BugGame.Resources;

namespace BugGame.Parts;

internal class Head : ParentPart
{
    private Feelers _feelers = new();

    public Head()
        : base(Message.HeadAdded, Message.HeadNotNeeded)
    {
    }

    public override bool IsComplete => _feelers.IsComplete;

    protected override bool TryAddCore(IPart part, out Message message)
        => part switch
        {
            Feeler => _feelers.TryAddOne(out message),
            _ => throw new NotSupportedException($"Can't add a {part.Name} to a {Name}.")
        };

    public void AppendTo(StringBuilder builder, char feelerCharacter)
    {
        if (IsPresent)
        {
            _feelers.AppendTo(builder, feelerCharacter);
            builder
                .AppendLine("        HHHHHHH")
                .AppendLine("        H     H")
                .AppendLine("        H O O H")
                .AppendLine("        H     H")
                .AppendLine("        H  V  H")
                .AppendLine("        HHHHHHH");
        }
    }
}

```

# `16_Bug/csharp/Parts/IPart.cs`

这段代码定义了一个名为“BugGame.Parts”的命名空间，其中定义了一个名为“IPart”的接口，以及一个内部类型“Part”和一个名为“Name”的内部字段。

IPart接口定义了一个“Part”类的实例，具有一个名为“Name”的内部字段，通过该字段获取该“Part”类的实例。

“Part”类是一个定义了“Part”接口的类，可以被看作是一个实现了IPart接口的类。

“BugGame.Parts”命名空间中可能包含了多个类，这些类可以实现IPart接口，也可以包含和使用这些实现的类。


```
namespace BugGame.Parts;

internal interface IPart
{
    string Name { get; }
}

```

# `16_Bug/csharp/Parts/Leg.cs`

这段代码定义了一个名为"Leg"的内部类，它继承自名为"IPart"的接口。这个内部类实现了两个方法：一个是"nameof"，用来获取该内部类的名称(即"Leg")；另一个是"Leg"，这个方法返回一个字符串类型的变量"name"。

这段代码的作用是定义了一个内部类"Leg"，它继承自"IPart"接口。这个内部类中定义了两个方法，"nameof"和"Leg"。其中，"nameof"方法返回该内部类的名称，"Leg"方法返回一个字符串类型的变量'name'。

虽然这个代码没有输出任何信息，但是这段代码对于理解这段代码的作用非常重要。


```
namespace BugGame.Parts;

internal class Leg : IPart
{
    public string Name => nameof(Leg);
}

```

# `16_Bug/csharp/Parts/Legs.cs`

这段代码定义了一个名为"Legs"的类，它是PartCollection派生的类。这个类的构造函数包括一个默认构造函数，它接受一个整数参数，用于指定这个类的父类。这个构造函数的行为是在创建一个空集合之后，将6个元素添加到集合中，并将消息记录为"LegsAdded"和"LegsFull"。

在Legs的AppendTo方法中，使用了两个StringBuilder类，分别用于将元素添加到集合中。第一个StringBuilder用于将元素添加到指定的字符串中，第二个StringBuilder用于输出拼接后的字符串。

具体来说，这段代码的作用是创建一个名为"Legs"的类，该类继承自一个名为"PartCollection"的类。这个类的构造函数接受了两个参数，第一个参数用于指定这个类的父类，第二个参数用于指定这个类的消息记录。在构造函数中，创建了一个空集合，并添加了6个元素到其中。然后，将这些元素添加到了指定的字符串中，并将消息记录为"LegsAdded"和"LegsFull"。AppendTo方法则接受一个StringBuilder参数，用于将元素添加到指定的字符串中，并输出了拼接后的字符串。


```
using System.Text;
using BugGame.Resources;

namespace BugGame.Parts;

internal class Legs : PartCollection
{
    public Legs()
        : base(6, Message.LegAdded, Message.LegsFull)
    {
    }

    public void AppendTo(StringBuilder builder) => AppendTo(builder, 6, 2, 'L');
}

```

# `16_Bug/csharp/Parts/Neck.cs`

这段代码定义了一个名为“Neck”的类，继承自“ParentPart”类。该类用于表示 neck 部件，负责与其他部件进行连接和安装。

在“Neck”类中，首先创建一个名为“_head”的头部部件，然后使用基类的构造函数初始化该头部部件。

接着定义了一个名为“IsComplete”的函数，用于判断当前头部部件是否已经完成，如果已完成则设置为真，否则设置为假。

然后定义了一个名为“TryAddCore”的函数，用于尝试将“IPart”类型的部件添加到头部部件中，并返回相应的“Message”类型的数据。具体来说，如果部件是头部部件，则尝试使用当前头部部件添加，如果是其他部件，则会抛出“NotSupportedException”异常。

最后定义了一个名为“AppendTo”的函数，用于将当前头部部件添加到字符串 builder 中，并在字符串的结尾添加两个“N”字符。

总的来说，该代码定义了一个用于表示 neck 部件的类，用于在游戏开发中进行 neck 部件的添加、安装等操作。


```
using System.Text;
using BugGame.Resources;

namespace BugGame.Parts;

internal class Neck : ParentPart
{
    private Head _head = new();

    public Neck()
        : base(Message.NeckAdded, Message.NeckNotNeeded)
    {
    }

    public override bool IsComplete => _head.IsComplete;

    protected override bool TryAddCore(IPart part, out Message message)
        => part switch
        {
            Head => _head.TryAdd(out message),
            Feeler => _head.TryAdd(part, out message),
            _ => throw new NotSupportedException($"Can't add a {part.Name} to a {Name}.")
        };

    public void AppendTo(StringBuilder builder, char feelerCharacter)
    {
        if (IsPresent)
        {
            _head.AppendTo(builder, feelerCharacter);
            builder.AppendLine("          N N").AppendLine("          N N");
        }
    }
}

```