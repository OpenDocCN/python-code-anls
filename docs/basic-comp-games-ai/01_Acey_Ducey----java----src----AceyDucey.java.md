# `basic-computer-games\01_Acey_Ducey\java\src\AceyDucey.java`

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
    private int playerAmount;

    // First drawn dealer card
    private Card firstCard;

    // Second drawn dealer card
    private Card secondCard;

    // Players drawn card
    private Card playersCard;

    // User to display game intro/instructions
    private boolean firstTimePlaying = true;

    // game state to determine if game over
    private boolean gameOver = false;

    // Used for keyboard input
    private final Scanner kbScanner;

    // Constant value for cards from a deck - 2 lowest, 14 (Ace) highest
    public static final int LOW_CARD_RANGE = 2;
    public static final int HIGH_CARD_RANGE = 14;

    // Constructor for initializing player's cash and keyboard scanner
    public AceyDucey() {
        // Initialise players cash
        playerAmount = 100;

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);
    }

    // Play again method - public method called from class invoking game
    // If player enters YES then the game can be played again (true returned)
    // otherwise not (false)
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
            System.out.println("O.K., HOPE YOU HAD FUN!");
            return false;
        }
    }

    // game loop method
    public void play() {

        // 持续进行游戏直到玩家用完所有现金
        do {
            if (firstTimePlaying) {
                intro(); // 调用intro()方法，介绍游戏规则
                firstTimePlaying = false; // 设置firstTimePlaying为false，表示不是第一次玩游戏
            }
            displayBalance(); // 显示玩家的余额
            drawCards(); // 抽取牌
            displayCards(); // 显示抽取的牌
            int betAmount = getBet(); // 获取玩家的赌注
            playersCard = randomCard(); // 随机抽取一张牌作为玩家的牌
            displayPlayerCard(); // 显示玩家的牌
            if (playerWon()) { // 判断玩家是否赢得了游戏
                System.out.println("YOU WIN!!"); // 打印玩家赢得游戏的消息
                playerAmount += betAmount; // 玩家赢得的赌注加到玩家的余额中
            } else {
                System.out.println("SORRY, YOU LOSE"); // 打印玩家输掉游戏的消息
                playerAmount -= betAmount; // 玩家输掉的赌注从玩家的余额中扣除
                // 玩家是否已经用完所有现金？
                if (playerAmount <= 0) {
                    gameOver = true; // 如果玩家的余额小于等于0，游戏结束
                }
            }

        } while (!gameOver); // 持续进行游戏直到玩家用完所有现金
    }

    // 判断玩家是否赢得了游戏（返回true）或输掉了游戏（返回false）
    // 要赢得游戏，玩家的牌必须在庄家抽取的第一张和第二张牌的范围内，包括第一张和第二张牌。
    private boolean playerWon() {
        // 赢家
        return (playersCard.getValue() >= firstCard.getValue())
                && playersCard.getValue() <= secondCard.getValue();

    }

    private void displayPlayerCard() {
        System.out.println(playersCard.getName()); // 显示玩家的牌的名称
    }

    // 获取玩家的赌注，并返回赌注金额
    // 0被视为有效的赌注，但超过玩家可用金额的赌注不被接受
    // 方法将循环直到输入有效的赌注
    # 获取玩家下注金额
    private int getBet() {
        # 初始化下注有效性标志
        boolean validBet = false;
        # 初始化下注金额
        int amount;
        # 循环直到玩家输入有效的下注金额
        do {
            # 提示玩家输入下注金额
            System.out.print("WHAT IS YOUR BET ");
            # 从键盘输入下注金额
            amount = kbScanner.nextInt();
            # 如果下注金额为0，则输出“CHICKEN!!”，并将下注有效性标志设为true
            if (amount == 0) {
                System.out.println("CHICKEN!!");
                validBet = true;
            } 
            # 如果下注金额大于玩家拥有的金额，则输出提示信息
            else if (amount > playerAmount) {
                System.out.println("SORRY, MY FRIEND, BUT YOU BET TOO MUCH.");
                System.out.println("YOU HAVE ONLY " + playerAmount + " DOLLARS TO BET.");
            } 
            # 否则将下注有效性标志设为true
            else {
                validBet = true;
            }
        } while (!validBet);  # 循环直到下注有效性标志为true

        return amount;  # 返回下注金额
    }

    # 显示玩家当前余额
    private void displayBalance() {
        System.out.println("YOU NOW HAVE " + playerAmount + " DOLLARS.");
    }

    # 显示玩家的两张牌
    private void displayCards() {
        System.out.println("HERE ARE YOUR NEXT TWO CARDS: ");
        System.out.println(firstCard.getName());
        System.out.println(secondCard.getName());
    }

    # 抽取两张庄家的牌，并保存它们以备后用
    # 确保第一张牌的值小于第二张牌的值
    private void drawCards() {

        do {
            firstCard = randomCard();  # 抽取第一张牌
            secondCard = randomCard();  # 抽取第二张牌
        } while (firstCard.getValue() >= secondCard.getValue());  # 循环直到第一张牌的值小于第二张牌的值
    }

    # 创建一张随机的牌
    private Card randomCard() {
        return new Card((int) (Math.random()
                * (HIGH_CARD_RANGE - LOW_CARD_RANGE + 1) + LOW_CARD_RANGE));  # 返回一个随机值在指定范围内的牌
    }
    // 输出游戏标题
    System.out.println("ACEY DUCEY CARD GAME");
    // 输出游戏开发者信息
    System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    // 输出空行
    System.out.println();
    System.out.println();
    // 输出游戏规则说明
    System.out.println("ACEY-DUCEY IS PLAYED IN THE FOLLOWING MANNER");
    System.out.println("THE DEALER (COMPUTER) DEALS TWO CARDS FACE UP");
    System.out.println("YOU HAVE AN OPTION TO BET OR NOT BET DEPENDING");
    System.out.println("ON WHETHER OR NOT YOU FEEL THE CARD WILL HAVE");
    System.out.println("A VALUE BETWEEN THE FIRST TWO.");
    System.out.println("IF YOU DO NOT WANT TO BET, INPUT: 0");
# 闭合前面的函数定义
```