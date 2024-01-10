# `basic-computer-games\71_Poker\java\Poker.java`

```
import java.util.Random;  // 导入 Random 类
import java.util.Scanner;  // 导入 Scanner 类
import static java.lang.System.out;  // 静态导入 System 类的 out 对象

/**
 * 将 CREATIVE COMPUTING Poker 游戏从 Commodore 64 Basic 移植到普通的 Java
 *
 * 原始源码扫描自杂志: https://www.atariarchives.org/basicgames/showpage.php?page=129
 *
 * 我基于这里的 OCR'ed 源码进行移植: https://github.com/coding-horror/basic-computer-games/blob/main/71_Poker/poker.bas
 *
 * 为什么？因为我记得小时候在 C64 上输入这个程序，玩得很开心！
 *
 * 目标：保持算法和用户体验基本不变；改进控制流程（在 Java 中不使用 goto！）并重命名一些东西以便更容易理解。
 *
 * 结果：可能会有 bug，请告诉我。
 */
public class Poker {

    public static void main(String[] args) {
        new Poker().run();  // 创建 Poker 对象并调用 run 方法
    }

    float[] cards = new float[50];  // 创建长度为 50 的浮点数数组 cards
    float[] B = new float[15];  // 创建长度为 15 的浮点数数组 B

    float playerValuables = 1;  // 玩家的财产值为 1
    float computerMoney = 200;  // 计算机的钱数为 200
    float humanMoney = 200;  // 玩家的钱数为 200
    float pot = 0;  // 底池为 0

    String J$ = "";  // 创建空字符串 J$
    float computerHandValue = 0;  // 计算机手牌的值为 0

    int K = 0;  // 创建整型变量 K，初始值为 0
    float G = 0;  // 创建浮点数变量 G，初始值为 0
    float T = 0;  // 创建浮点数变量 T，初始值为 0
    int M = 0;  // 创建整型变量 M，初始值为 0
    int D = 0;  // 创建整型变量 D，初始值为 0

    int U = 0;  // 创建整型变量 U，初始值为 0
    float N = 1;  // 创建浮点数变量 N，初始值为 1

    float I = 0;  // 创建浮点数变量 I，初始值为 0

    float X = 0;  // 创建浮点数变量 X，初始值为 0

    int Z = 0;  // 创建整型变量 Z，初始值为 0

    String handDescription = "";  // 创建空字符串 handDescription

    float V;  // 创建浮点数变量 V

    void run() {  // 定义 run 方法
        printWelcome();  // 调用 printWelcome 方法
        playRound();  // 调用 playRound 方法
        startAgain();  // 调用 startAgain 方法
    }

    void printWelcome() {  // 定义 printWelcome 方法
        tab(33);  // 调用 tab 方法，参数为 33
        out.println("POKER");  // 输出 "POKER"
        tab(15);  // 调用 tab 方法，参数为 15
        out.print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 输出 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
        out.println();  // 输出空行
        out.println();  // 输出空行
        out.println();  // 输出空行
        out.println("WELCOME TO THE CASINO.  WE EACH HAVE $200.");  // 输出欢迎信息
        out.println("I WILL OPEN THE BETTING BEFORE THE DRAW; YOU OPEN AFTER.");  // 输出下注规则
        out.println("TO FOLD BET 0; TO CHECK BET .5.");  // 输出弃牌和跟注规则
        out.println("ENOUGH TALK -- LET'S GET DOWN TO BUSINESS.");  // 输出提示信息
        out.println();  // 输出空行
    }
    # 定义一个函数，用于打印指定数量的制表符
    void tab(int number) {
        System.out.print("\t".repeat(number));
    }
    
    # 定义一个函数，返回一个0到10之间的随机整数
    int random0to10() {
        return new Random().nextInt(10);
    }
    
    # 定义一个函数，用于去除一个长整型数值的百位部分
    int removeHundreds(long x) {
        return _int(x - (100F * _int(x / 100F)));
    }
    
    # 定义一个函数，用于重新开始游戏
    void startAgain() {
        pot = 0;
        playRound();
    }
    // 进行一轮游戏
    void playRound() {
        // 如果电脑的钱少于等于5，调用电脑破产函数
        if (computerMoney <= 5) {
            computerBroke();
        }

        // 输出底注为$5，开始发牌
        out.println("THE ANTE IS $5.  I WILL DEAL:");
        out.println();

        // 如果玩家的钱少于等于5，调用玩家破产函数
        if (humanMoney <= 5) {
            playerBroke();
        }

        // 奖池增加10，玩家和电脑的钱减少5
        pot = pot + 10;
        humanMoney = humanMoney - 5;
        computerMoney = computerMoney - 5;
        // 生成10张牌
        for (int Z = 1; Z < 10; Z++) {
            generateCards(Z);
        }
        // 输出玩家的手牌
        out.println("YOUR HAND:");
        N = 1;
        showHand();
        N = 6;
        I = 2;

        // 描述玩家的手牌
        describeHand();

        out.println();

        // 根据条件进行不同的操作
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
                // 电脑开牌
                computerOpens();
                // 玩家行动
                playerMoves();
            } else if (random0to10() >= 2) {
                // 电脑跟注
                computerChecks();
            } else {
                I = 7;
                Z = 23;
                // 电脑开牌
                computerOpens();
                // 玩家行动
                playerMoves();
            }
        } else if (random0to10() <= 7) {
            if (random0to10() <= 7) {
                if (random0to10() >= 1) {
                    Z = 1;
                    K = 0;
                    out.print("I CHECK. ");
                    // 玩家行动
                    playerMoves();
                } else {
                    X = 11111;
                    I = 7;
                    Z = 23;
                    // 电脑开牌
                    computerOpens();
                    // 玩家行动
                    playerMoves();
                }
            } else {
                X = 11110;
                I = 7;
                Z = 23;
                // 电脑开牌
                computerOpens();
                // 玩家行动
                playerMoves();
            }
        } else {
            X = 11100;
            I = 7;
            Z = 23;
            // 电脑开牌
            computerOpens();
            // 玩家行动
            playerMoves();
        }
    }
    // 玩家进行移动的方法
    void playerMoves() {
        // 玩家轮次
        playersTurn();
        // 检查第一次下注后的赢家
        checkWinnerAfterFirstBet();
        // 提示玩家抽牌
        promptPlayerDrawCards();
    }
    
    // 电脑开局的方法
    void computerOpens() {
        // V 等于 Z 加上 0 到 10 的随机数
        V = Z + random0to10();
        // 电脑移动
        computerMoves();
        // 输出电脑开局的赌注
        out.print("I'LL OPEN WITH $" + V);
        // K 等于 V 的整数部分
        K = _int(V);
    }
    
    // 电脑移动的方法
    @SuppressWarnings("StatementWithEmptyBody")
    void computerMoves() {
        // 如果电脑的钱减去 G 减去 V 大于等于 0
        if (computerMoney - G - V >= 0) {
            // 空语句
        } else if (G != 0) {
            // 如果 G 不等于 0
            if (computerMoney - G >= 0) {
                // 电脑看牌
                computerSees();
            } else {
                // 电脑破产
                computerBroke();
            }
        } else {
            // V 等于电脑的钱
            V = computerMoney;
        }
    }
    
    // 提示玩家抽牌的方法
    void promptPlayerDrawCards() {
        out.println();
        out.println("NOW WE DRAW -- HOW MANY CARDS DO YOU WANT");
        // 输入玩家抽牌的数量
        inputPlayerDrawCards();
    }
    
    // 输入玩家抽牌的数量的方法
    void inputPlayerDrawCards() {
        // T 等于解析输入的字符串为整数
        T = Integer.parseInt(readString());
        // 如果 T 等于 0
        if (T == 0) {
            // 电脑抽牌
            computerDrawing();
        } else {
            // Z 等于 10
            Z = 10;
            // 如果 T 小于 4
            if (T < 4) {
                // 玩家抽牌
                playerDrawsCards();
            } else {
                // 输出提示信息
                out.println("YOU CAN'T DRAW MORE THAN THREE CARDS.");
                // 重新输入玩家抽牌的数量
                inputPlayerDrawCards();
            }
        }
    }
    
    // 电脑抽牌的方法
    void computerDrawing() {
        // Z 等于 10 加上 T 的整数部分
        Z = _int(10 + T);
        // 循环从 6 到 10
        for (U = 6; U <= 10; U++) {
            // 如果 X 除以 10 的 (U-6) 次方的整数部分等于 10 乘以 X 除以 10 的 (U-5) 次方的整数部分
            if (_int((float) (X / Math.pow(10F, (U - 6F)))) == (10 * (_int((float) (X / Math.pow(10, (U - 5))))))) {
                // 抽下一张牌
                drawNextCard();
            }
        }
        // 输出电脑抽牌的数量
        out.print("I AM TAKING " + _int(Z - 10 - T) + " CARD");
        // 如果 Z 等于 11 加上 T
        if (Z == 11 + T) {
            out.println();
        } else {
            out.println("S");
        }
    
        // N 等于 6
        N = 6;
        // V 等于 I
        V = I;
        // I 等于 1
        I = 1;
        // 描述手牌
        describeHand();
        // 开始玩家下注和反应
        startPlayerBettingAndReaction();
    }
    
    // 抽下一张牌的方法
    void drawNextCard() {
        // Z 加上 1
        Z = Z + 1;
        // 抽牌
        drawCard();
    }
    
    // 空语句
    @SuppressWarnings("StatementWithEmptyBody")
    // 抽取一张卡牌
    void drawCard() {
        // 生成一个随机数，表示卡牌的数值
        cards[Z] = 100 * new Random().nextInt(4) + new Random().nextInt(100);
        // 如果卡牌的花色大于3，则重新抽取
        if (_int(cards[Z] / 100) > 3) {
            drawCard();
        } else if (cards[Z] - 100 * _int(cards[Z] / 100) > 12) {
            // 如果卡牌的点数大于12，则重新抽取
            drawCard();
        } else if (Z == 1) {
            // 如果 Z 等于 1，则不做任何操作
        } else {
            // 检查新抽取的卡牌是否与之前抽取的卡牌重复，如果重复则重新抽取
            for (K = 1; K <= Z - 1; K++) {
                if (cards[Z] == cards[K]) {
                    drawCard();
                }
            }
            // 如果 Z 小于等于 10，则不做任何操作
            if (Z <= 10) {
            } else {
                // 交换卡牌的位置
                N = cards[U];
                cards[U] = cards[Z];
                cards[Z] = N;
            }
        }
    }

    // 玩家抽取卡牌
    void playerDrawsCards() {
        // 输出提示信息
        out.println("WHAT ARE THEIR NUMBERS:");
        // 循环抽取卡牌
        for (int Q = 1; Q <= T; Q++) {
            U = Integer.parseInt(readString());
            drawNextCard();
        }

        // 输出新的手牌
        out.println("YOUR NEW HAND:");
        N = 1;
        showHand();
        // 让电脑抽取卡牌
        computerDrawing();
    }

    // 开始玩家下注和反应
    void startPlayerBettingAndReaction() {
        // 记录电脑的手牌值
        computerHandValue = U;
        M = D;

        // 根据不同情况进行下注和反应
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
    // 玩家下注并且计算机做出反应
    void playerBetsAndComputerReacts() {
        // 初始化变量 K
        K = 0;
        // 玩家轮次
        playersTurn();
        // 如果 T 不等于 0.5
        if (T != .5) {
            // 检查第一次下注后的赢家并比较手牌
            checkWinnerAfterFirstBetAndCompareHands();
        } else if (V == 7 || I != 6) {
            // 计算机开牌
            computerOpens();
            // 提示并输入玩家下注
            promptAndInputPlayerBet();
            // 检查第一次下注后的赢家并比较手牌
            checkWinnerAfterFirstBetAndCompareHands();
        } else {
            // 输出信息
            out.println("I'LL CHECK");
            // 比较手牌
            compareHands();
        }
    }
    
    // 检查第一次下注后的赢家并比较手牌
    void checkWinnerAfterFirstBetAndCompareHands() {
        // 检查第一次下注后的赢家
        checkWinnerAfterFirstBet();
        // 比较手牌
        compareHands();
    }
    
    // 比较手牌
    void compareHands() {
        // 输出信息
        out.println("NOW WE COMPARE HANDS:");
        // 保存手牌描述
        J$ = handDescription;
        // 输出信息
        out.println("MY HAND:");
        // 设置变量 N
        N = 6;
        // 展示手牌
        showHand();
        // 设置变量 N
        N = 1;
        // 描述手牌
        describeHand();
        // 输出信息
        out.print("YOU HAVE ");
        // 保存变量 K 的值
        K = D;
        // 输出手牌描述结果
        printHandDescriptionResult();
        // 重置手牌描述
        handDescription = J$;
        // 保存变量 K 的值
        K = M;
        // 输出信息
        out.print(" AND I HAVE ");
        // 输出手牌描述结果
        printHandDescriptionResult();
        // 输出信息
        out.print(". ");
        // 如果计算机手牌值大于 U
        if (computerHandValue > U) {
            // 计算机获胜
            computerWins();
        } else if (U > computerHandValue) {
            // 玩家获胜
            humanWins();
        } else if (handDescription.contains("A FLUS")) {
            // 有人以同花获胜
            someoneWinsWithFlush();
        } else if (removeHundreds(M) < removeHundreds(D)) {
            // 玩家获胜
            humanWins();
        } else if (removeHundreds(M) > removeHundreds(D)) {
            // 计算机获胜
            computerWins();
        } else {
            // 手牌平局
            handIsDrawn();
        }
    }
    
    // 输出手牌描述结果
    void printHandDescriptionResult() {
        // 输出手牌描述
        out.print(handDescription);
        // 如果手牌描述不包含 "A FLUS"
        if (!handDescription.contains("A FLUS")) {
            // 移除变量 K 的百位数
            K = removeHundreds(K);
            // 输出卡牌值
            printCardValue();
            // 如果手牌描述包含 "SCHMAL"
            if (handDescription.contains("SCHMAL")) {
                // 输出信息
                out.print(" HIGH");
            } else if (!handDescription.contains("STRAIG")) {
                // 输出信息
                out.print("'S");
            } else {
                // 输出信息
                out.print(" HIGH");
            }
        } else {
            // 变量 K 除以 100
            K = K / 100;
            // 输出卡牌颜色
            printCardColor();
            // 输出换行
            out.println();
        }
    }
    // 当手牌已经画好时，输出提示信息并进行下一轮游戏
    void handIsDrawn() {
        out.print("THE HAND IS DRAWN.");
        out.print("ALL $" + pot + " REMAINS IN THE POT.");
        playRound();
    }

    // 当有玩家以同花牌型获胜时的处理逻辑
    void someoneWinsWithFlush() {
        // 如果电脑的同花牌型大于玩家的同花牌型，电脑获胜
        if (removeHundreds(M) > removeHundreds(D)) {
            computerWins();
        } 
        // 如果玩家的同花牌型大于电脑的同花牌型，玩家获胜
        else if (removeHundreds(D) > removeHundreds(M)) {
            humanWins();
        } 
        // 如果两者的同花牌型相同，则手牌平局
        else {
            handIsDrawn();
        }
    }

    // 在第一轮下注后检查获胜者
    @SuppressWarnings("StatementWithEmptyBody")
    void checkWinnerAfterFirstBet() {
        // 如果 I 不等于 3
        if (I != 3) {
            // 如果 I 不等于 4，则不执行任何操作
            if (I != 4) {
            } 
            // 如果 I 等于 4，则玩家获胜
            else {
                humanWins();
            }
        } 
        // 如果 I 等于 3，则电脑获胜
        else {
            out.println();
            computerWins();
        }
    }

    // 电脑获胜时的处理逻辑
    void computerWins() {
        out.print(". I WIN. ");
        computerMoney = computerMoney + pot;
        potStatusAndNextRoundPrompt();
    }

    // 输出奖池状态并提示下一轮游戏是否继续
    void potStatusAndNextRoundPrompt() {
        out.println("NOW I HAVE $" + computerMoney + " AND YOU HAVE $" + humanMoney);
        out.print("DO YOU WISH TO CONTINUE");

        // 如果从提示中得到肯定回答，则重新开始游戏
        if (yesFromPrompt()) {
            startAgain();
        } 
        // 如果得到否定回答，则退出游戏
        else {
            System.exit(0);
        }
    }

    // 从提示中得到肯定回答的私有方法
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

    // 电脑进行检查的处理逻辑
    void computerChecks() {
        Z = 0;
        K = 0;
        out.print("I CHECK. ");
        playerMoves();
    }

    // 玩家获胜时的处理逻辑
    void humanWins() {
        out.println("YOU WIN.");
        humanMoney = humanMoney + pot;
        potStatusAndNextRoundPrompt();
    }

    // line # 1740
    // 生成一副扑克牌，总共 Z 张
    void generateCards(int Z) {
        // 生成一张随机的扑克牌，存入 cards 数组的第 Z 个位置
        cards[Z] = (100 * new Random().nextInt(4)) + new Random().nextInt(100);
        // 如果扑克牌的值大于 300，重新生成
        if (_int(cards[Z] / 100) > 3) {
            generateCards(Z);
            return;
        }
        // 如果扑克牌的值大于 12，重新生成
        if (cards[Z] - 100 * (_int(cards[Z] / 100)) > 12) {
            generateCards(Z);
            return;
        }
        // 如果 Z 等于 1，直接返回
        if (Z == 1) {return;}
        // 检查新生成的扑克牌是否与之前的重复，如果重复则重新生成
        for (int K = 1; K <= Z - 1; K++) {// TO Z-1
            if (cards[Z] == cards[K]) {
                generateCards(Z);
                return;
            }
        }
        // 如果 Z 小于等于 10，直接返回
        if (Z <= 10) {return;}
        // 交换 cards 数组中的 U 和 Z 位置的扑克牌
        float N = cards[U];
        cards[U] = cards[Z];
        cards[Z] = N;
    }

    // line # 1850
    // 展示手中的扑克牌
    void showHand() {
        // 循环展示 N 到 N+4 的扑克牌
        for (int cardNumber = _int(N); cardNumber <= N + 4; cardNumber++) {
            out.print(cardNumber + "--  ");
            // 打印扑克牌的值
            printCardValueAtIndex(cardNumber);
            out.print(" OF");
            // 打印扑克牌的花色
            printCardColorAtIndex(cardNumber);
            // 如果是偶数张牌，换行
            if (cardNumber / 2 == (cardNumber / 2)) {
                out.println();
            }
        }
    }

    // line # 1950
    // 打印指定位置的扑克牌的值
    void printCardValueAtIndex(int Z) {
        // 获取扑克牌的值并打印
        K = removeHundreds(_int(cards[Z]));
        printCardValue();
    }

    // 打印扑克牌的值
    void printCardValue() {
        // 根据扑克牌的值打印相应的文字
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
    // 打印指定位置的扑克牌的花色
    void printCardColorAtIndex(int Z) {
        // 获取扑克牌的花色并打印
        K = _int(cards[Z] / 100);
        printCardColor();
    }

    // 打印扑克牌的花色
    void printCardColor() {
        // 根据扑克牌的花色打印相应的文字
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
    }
    // 定义函数，处理特定情况下的手牌
    void schmaltzHand() {
        // 如果 U 大于等于 10
        if (U >= 10) {
            // 如果 U 不等于 10
            if (U != 10) {
                // 如果 U 大于 12，直接返回
                if (U > 12) {return;}
                // 如果去除 D 的百位后的值小于等于 6
                if (removeHundreds(D) <= 6) {
                    // 将 I 设为 6
                    I = 6;
                }
            } else {
                // 如果 U 等于 10 且 I 等于 1，将 I 设为 6
                if (I == 1) {
                    I = 6;
                }
            }
        } else {
            // 如果 U 小于 10，根据 N 的值读取 D 的值
            D = _int(cards[_int(N + 4)]);
            // 设置手牌描述为 "SCHMALTZ, "
            handDescription = "SCHMALTZ, ";
            // 设置 U 为 9
            U = 9;
            // 设置 X 为 11000
            X = 11000;
            // 设置 I 为 6
            I = 6;
        }
    }

    // 定义函数，处理满堂彩的情况
    void fullHouse() {
        // 设置 U 为 16
        U = 16;
        // 设置手牌描述为 "FULL HOUSE, "
        handDescription = "FULL HOUSE, ";
    }

    // 定义函数，处理玩家的回合
    void playersTurn() {
        // 设置 G 为 0
        G = 0;
        // 提示并输入玩家的赌注
        promptAndInputPlayerBet();
    }

    // 定义函数，读取字符串输入
    String readString() {
        // 创建 Scanner 对象
        Scanner sc = new Scanner(System.in);
        // 返回输入的字符串
        return sc.nextLine();
    }

    // 定义函数，提示并输入玩家的赌注
    @SuppressWarnings("StatementWithEmptyBody")
    void promptAndInputPlayerBet() {
        // 输出提示信息
        out.println("WHAT IS YOUR BET");
        // 读取浮点数输入
        T = readFloat();
        // 如果 T 减去 T 的整数部分等于 0，处理玩家的赌注
        if (T - _int(T) == 0) {
            processPlayerBet();
        } else if (K != 0) {
            // 如果 K 不等于 0，提示玩家赌注无效
            playerBetInvalidAmount();
        } else if (G != 0) {
            // 如果 G 不等于 0，提示玩家赌注无效
            playerBetInvalidAmount();
        } else if (T == .5) {
            // 如果 T 等于 0.5，不做任何处理
        } else {
            // 其他情况下，提示玩家赌注无效
            playerBetInvalidAmount();
        }
    }

    // 定义函数，读取浮点数输入
    private float readFloat() {
        // 尝试将输入的字符串转换为浮点数
        try {
            return Float.parseFloat(readString());
        } catch (Exception ex) {
            // 如果转换失败，输出提示信息并重新读取浮点数输入
            System.out.println("INVALID INPUT, PLEASE TYPE A FLOAT. ");
            return readFloat();
        }
    }

    // 定义函数，处理玩家赌注无效的情况
    void playerBetInvalidAmount() {
        // 输出提示信息
        out.println("NO SMALL CHANGE, PLEASE.");
        // 提示并输入玩家的赌注
        promptAndInputPlayerBet();
    }

    // 定义函数，处理玩家赌注的情况
    void processPlayerBet() {
        // 如果玩家的余额减去 G 和 T 大于等于 0，处理玩家可以承担的赌注
        if (humanMoney - G - T >= 0) {
            humanCanAffordBet();
        } else {
            // 否则，玩家破产，提示并输入玩家的赌注
            playerBroke();
            promptAndInputPlayerBet();
        }
    }
    // 判断玩家是否能够支付赌注
    void humanCanAffordBet() {
        // 如果 T 不等于 0
        if (T != 0) {
            // 如果玩家的赌注加上当前赌注大于等于总赌注
            if (G + T >= K) {
                // 处理计算机的移动
                processComputerMove();
            } else {
                // 输出信息并提示玩家输入赌注
                out.println("IF YOU CAN'T SEE MY BET, THEN FOLD.");
                promptAndInputPlayerBet();
            }
        } else {
            // 将 I 设置为 3
            I = 3;
            // 将赌注移动到奖池
            moveMoneyToPot();
        }
    }

    // 处理计算机的移动
    void processComputerMove() {
        // 玩家的赌注加上当前赌注
        G = G + T;
        // 如果玩家的赌注等于总赌注
        if (G == K) {
            // 将赌注移动到奖池
            moveMoneyToPot();
        } else if (Z != 1) {
            // 如果计算机的赌注大于 3 倍的 Z
            if (G > 3 * Z) {
                // 计算机加注或跟注
                computerRaisesOrSees();
            } else {
                // 计算机加注
                computerRaises();
            }
        } else if (G > 5) {
            // 如果计算机的赌注大于 5
            if (T <= 25) {
                // 计算机加注或跟注
                computerRaisesOrSees();
            } else {
                // 计算机弃牌
                computerFolds();
            }
        } else {
            // 将 V 设置为 5
            V = 5;
            // 如果计算机的赌注大于 3 倍的 Z
            if (G > 3 * Z) {
                // 计算机加注或跟注
                computerRaisesOrSees();
            } else {
                // 计算机加注
                computerRaises();
            }
        }
    }

    // 计算机加注
    void computerRaises() {
        // 计算 V 的值
        V = G - K + random0to10();
        // 计算机移动
        computerMoves();
        // 输出信息
        out.println("I'LL SEE YOU, AND RAISE YOU" + V);
        // 更新总赌注
        K = _int(G + V);
        // 提示玩家输入赌注
        promptAndInputPlayerBet();
    }

    // 计算机弃牌
    void computerFolds() {
        // 将 I 设置为 4
        I = 4;
        // 输出信息
        out.println("I FOLD.");
    }

    // 计算机加注或跟注
    void computerRaisesOrSees() {
        // 如果 Z 等于 2
        if (Z == 2) {
            // 计算机加注
            computerRaises();
        } else {
            // 计算机跟注
            computerSees();
        }
    }

    // 计算机跟注
    void computerSees() {
        // 输出信息
        out.println("I'LL SEE YOU.");
        // 更新总赌注
        K = _int(G);
        // 将赌注移动到奖池
        moveMoneyToPot();
    }

    // 将赌注移动到奖池
    void moveMoneyToPot() {
        // 玩家的钱减去赌注
        humanMoney = humanMoney - G;
        // 计算机的钱减去总赌注
        computerMoney = computerMoney - K;
        // 更新奖池的值
        pot = pot + G + K;
    }

    // 计算机破产
    void computerBusted() {
        // 输出信息
        out.println("I'M BUSTED.  CONGRATULATIONS!");
        // 退出程序
        System.exit(0);
    }

    // 忽略带有空体的语句的警告
    @SuppressWarnings("StatementWithEmptyBody")
    // 当玩家破产时执行的方法
    private void computerBroke() {
        // 如果玩家财物除以2的余数为0，并且玩家选择买回手表
        if ((playerValuables / 2) == _int(playerValuables / 2) && playerBuyBackWatch()) {
        } 
        // 如果玩家财物除以3的余数为0，并且玩家选择买回领带夹
        else if (playerValuables / 3 == _int(playerValuables / 3) && playerBuyBackTieRack()) {
        } 
        // 否则，电脑宣布玩家破产
        else {
            computerBusted();
        }
    }

    // 将浮点数转换为整数
    private int _int(float v) {
        return (int) Math.floor(v);
    }

    // 玩家选择是否买回手表
    private boolean playerBuyBackWatch() {
        out.println("WOULD YOU LIKE TO BUY BACK YOUR WATCH FOR $50");
        // 如果玩家选择是
        if (yesFromPrompt()) {
            computerMoney = computerMoney + 50;
            playerValuables = playerValuables / 2;
            return true;
        } else {
            return false;
        }
    }

    // 玩家选择是否买回领带夹
    private boolean playerBuyBackTieRack() {
        out.println("WOULD YOU LIKE TO BUY BACK YOUR TIE TACK FOR $50");
        // 如果玩家选择是
        if (yesFromPrompt()) {
            computerMoney = computerMoney + 50;
            playerValuables = playerValuables / 3;
            return true;
        } else {
            return false;
        }
    }

    // 玩家破产时执行的方法
    @SuppressWarnings("StatementWithEmptyBody")
    void playerBroke() {
        out.println("YOU CAN'T BET WITH WHAT YOU HAVEN'T GOT.");
        // 如果玩家财物除以2的余数不为0，并且玩家选择卖掉手表
        if (playerValuables / 2 != _int(playerValuables / 2) && playerSellWatch()) {
        } 
        // 如果玩家财物除以3的余数不为0，并且玩家选择卖掉领带夹
        else if (playerValuables / 3 != _int(playerValuables / 3) && playerSellTieTack()) {
        } 
        // 否则，玩家宣布破产
        else {
            playerBusted();
        }
    }

    // 玩家宣布破产时执行的方法
    private void playerBusted() {
        out.println("YOUR WAD IS SHOT. SO LONG, SUCKER!");
        System.exit(0);
    }
    # 检查玩家是否愿意出售手表
    private boolean playerSellWatch() {
        # 打印提示信息，询问玩家是否愿意出售手表
        out.println("WOULD YOU LIKE TO SELL YOUR WATCH");
        # 如果玩家同意出售
        if (yesFromPrompt()) {
            # 如果随机数小于7
            if (random0to10() < 7) {
                # 打印出价信息，增加玩家金钱
                out.println("I'LL GIVE YOU $75 FOR IT.");
                humanMoney = humanMoney + 75;
            } else {
                # 打印出价信息，增加玩家金钱
                out.println("THAT'S A PRETTY CRUMMY WATCH - I'LL GIVE YOU $25.");
                humanMoney = humanMoney + 25;
            }
            # 增加玩家财产价值
            playerValuables = playerValuables * 2;
            return true;
        } else {
            return false;
        }
    }

    # 检查玩家是否愿意出售领带别针
    private boolean playerSellTieTack() {
        # 打印提示信息，询问玩家是否愿意出售领带别针
        out.println("WILL YOU PART WITH THAT DIAMOND TIE TACK");
        # 如果玩家同意出售
        if (yesFromPrompt()) {
            # 如果随机数小于6
            if (random0to10() < 6) {
                # 打印出价信息，增加玩家金钱
                out.println("YOU ARE NOW $100 RICHER.");
                humanMoney = humanMoney + 100;
            } else {
                # 打印出价信息，增加玩家金钱
                out.println("IT'S PASTE.  $25.");
                humanMoney = humanMoney + 25;
            }
            # 增加玩家财产价值
            playerValuables = playerValuables * 3;
            return true;
        } else {
            return false;
        }
    }
# 闭合前面的函数定义
```