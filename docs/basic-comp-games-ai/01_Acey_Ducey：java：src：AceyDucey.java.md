# `01_Acey_Ducey\java\src\AceyDucey.java`

```
import java.util.Scanner;

/**
 * Game of AceyDucey
 * <p>
 * Based on the Basic game of AceyDucey here
 * https://github.com/coding-horror/basic-computer-games/blob/main/01%20Acey%20Ducey/aceyducey.bas
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class AceyDucey {

    // Current amount of players cash
    private int playerAmount; // 玩家的现金金额

    // First drawn dealer card
    private Card firstCard; // 庄家第一张牌

    // Second drawn dealer card
    private Card secondCard; // 庄家第二张牌
    // 玩家抽取的卡片
    private Card playersCard;

    // 用于显示游戏介绍/说明的用户
    private boolean firstTimePlaying = true;

    // 游戏状态，用于确定游戏是否结束
    private boolean gameOver = false;

    // 用于键盘输入
    private final Scanner kbScanner;

    // 从一副牌中的常量值 - 2最低，14（Ace）最高
    public static final int LOW_CARD_RANGE = 2;
    public static final int HIGH_CARD_RANGE = 14;

    public AceyDucey() {
        // 初始化玩家的现金
        playerAmount = 100;
        // 初始化键盘扫描器
        kbScanner = new Scanner(System.in);
    }

    // 再玩一次的方法 - 从调用游戏的类中调用的公共方法
    // 如果玩家输入YES，则可以再次玩游戏（返回true）
    // 否则不行（返回false）
    public boolean playAgain() {
        System.out.println();
        System.out.println("SORRY, FRIEND, BUT YOU BLEW YOUR WAD.");
        System.out.println();
        System.out.println();
        System.out.print("TRY AGAIN (YES OR NO) ");
        String playAgain = kbScanner.next().toUpperCase();
        System.out.println();
        System.out.println();
        if (playAgain.equals("YES")) {
            return true;
        } else {
            System.out.println("O.K., HOPE YOU HAD FUN!");  // 打印消息到控制台，表示游戏结束
            return false;  // 返回 false，表示游戏结束
        }
    }

    // game loop method

    public void play() {

        // Keep playing hands until player runs out of cash
        do {
            if (firstTimePlaying) {  // 如果是第一次玩游戏
                intro();  // 调用 intro() 方法，显示游戏介绍
                firstTimePlaying = false;  // 将 firstTimePlaying 设置为 false，表示不是第一次玩游戏
            }
            displayBalance();  // 调用 displayBalance() 方法，显示玩家余额
            drawCards();  // 调用 drawCards() 方法，发牌
            displayCards();  // 调用 displayCards() 方法，显示玩家手牌
            int betAmount = getBet();  // 调用 getBet() 方法，获取玩家下注金额
            playersCard = randomCard();  // 将 playersCard 设置为随机的一张牌
            displayPlayerCard(); // 显示玩家的卡片
            if (playerWon()) { // 如果玩家赢了
                System.out.println("YOU WIN!!"); // 打印“你赢了！”
                playerAmount += betAmount; // 玩家金额增加下注金额
            } else { // 如果玩家输了
                System.out.println("SORRY, YOU LOSE"); // 打印“对不起，你输了”
                playerAmount -= betAmount; // 玩家金额减去下注金额
                // Player run out of money? 玩家是否用完了钱？
                if (playerAmount <= 0) { // 如果玩家金额小于等于0
                    gameOver = true; // 游戏结束标志设为true
                }
            }

        } while (!gameOver); // 当游戏结束标志为false时继续游戏
    }

    // Method to determine if player won (true returned) or lost (false returned)
    // to win a players card has to be in the range of the first and second dealer
    // drawn cards inclusive of first and second cards.
    private boolean playerWon() { // 确定玩家是否赢了的方法
        // 检查玩家的牌是否比第一张牌大且小于等于第二张牌，返回结果
        return (playersCard.getValue() >= firstCard.getValue())
                && playersCard.getValue() <= secondCard.getValue();

    }

    // 显示玩家的牌
    private void displayPlayerCard() {
        System.out.println(playersCard.getName());
    }

    // 获取玩家的赌注，并返回赌注金额
    // 0 被视为有效赌注，但超过玩家可用金额的赌注不被接受
    // 方法将循环直到输入有效的赌注
    private int getBet() {
        boolean validBet = false;
        int amount;
        do {
            System.out.print("WHAT IS YOUR BET ");
            amount = kbScanner.nextInt();
            if (amount == 0) {
                System.out.println("CHICKEN!!");
                // 打印出字符串"CHICKEN!!"
                validBet = true;
                // 将validBet设置为true
            } else if (amount > playerAmount) {
                System.out.println("SORRY, MY FRIEND, BUT YOU BET TOO MUCH.");
                // 打印出字符串"SORRY, MY FRIEND, BUT YOU BET TOO MUCH."
                System.out.println("YOU HAVE ONLY " + playerAmount + " DOLLARS TO BET.");
                // 打印出字符串"YOU HAVE ONLY "，加上playerAmount的值，再加上字符串" DOLLARS TO BET."
            } else {
                validBet = true;
                // 将validBet设置为true
            }
        } while (!validBet);
        // 当validBet为false时，继续循环

        return amount;
        // 返回变量amount的值
    }

    private void displayBalance() {
        System.out.println("YOU NOW HAVE " + playerAmount + " DOLLARS.");
        // 打印出字符串"YOU NOW HAVE "，加上playerAmount的值，再加上字符串" DOLLARS."
    }

    private void displayCards() {
        System.out.println("HERE ARE YOUR NEXT TWO CARDS: ");
        // 打印出字符串"HERE ARE YOUR NEXT TWO CARDS: "
        System.out.println(firstCard.getName());
        // 打印出firstCard的名称
        System.out.println(secondCard.getName());  // 打印出第二张牌的名称

    }

    // Draw two dealer cards, and save them for later use.
    // ensure that the first card is a smaller value card than the second one
    private void drawCards() {
        // 从牌堆中随机抽取两张牌，并保存它们以备后用
        // 确保第一张牌的值比第二张牌小
        do {
            firstCard = randomCard();  // 抽取第一张牌
            secondCard = randomCard();  // 抽取第二张牌
        } while (firstCard.getValue() >= secondCard.getValue());  // 如果第一张牌的值大于等于第二张牌的值，则重新抽牌
    }

    // Creates a random card
    private Card randomCard() {
        return new Card((int) (Math.random()
                * (HIGH_CARD_RANGE - LOW_CARD_RANGE + 1) + LOW_CARD_RANGE));  // 创建一张随机的牌
    }

    public void intro() {
# 打印游戏标题
System.out.println("ACEY DUCEY CARD GAME")
# 打印游戏开发者信息
System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
# 打印空行
System.out.println()
# 打印空行
System.out.println()
# 打印游戏规则说明
System.out.println("ACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER")
System.out.println("THE DEALER (COMPUTER) DEALS TWO CARDS FACE UP")
System.out.println("YOU HAVE AN OPTION TO BET OR NOT BET DEPENDING")
System.out.println("ON WHETHER OR NOT YOU FEEL THE CARD WILL HAVE")
System.out.println("A VALUE BETWEEN THE FIRST TWO.")
System.out.println("IF YOU DO NOT WANT TO BET, INPUT: 0")
```