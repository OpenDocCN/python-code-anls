# `d:/src/tocomm/basic-computer-games\94_War\java\War.java`

```
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.Arrays;  // 导入 Arrays 类
import java.util.Collections;  // 导入 Collections 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 * Converted FROM BASIC to Java by Nahid Mondol.  // 作者和转换信息的注释
 *
 * Based on Trevor Hobsons approach.  // 基于Trevor Hobsons的方法的注释
 */
public class War {  // 定义 War 类

    private static final int CARD_DECK_SIZE = 52;  // 定义常量 CARD_DECK_SIZE 为 52，表示卡牌的数量
    private static int playerTotalScore = 0;  // 定义玩家总分变量并初始化为 0
    private static int computerTotalScore = 0;  // 定义计算机总分变量并初始化为 0
    private static boolean invalidInput;  // 定义布尔类型变量 invalidInput
    private static Scanner userInput = new Scanner(System.in);  // 创建 Scanner 对象 userInput

    // Simple approach for storing a deck of cards.  // 存储卡牌的简单方法的注释
    // Suit-Value, ex: 2 of Spades = S-2, King of Diamonds = D-K, etc...  // 卡牌的表示方法的注释
    // 创建一个包含标准扑克牌的 ArrayList 对象
    private static ArrayList<String> deckOfCards = new ArrayList<String>(
            Arrays.asList("S-2", "H-2", "C-2", "D-2", "S-3", "H-3", "C-3", "D-3", "S-4", "H-4", "C-4", "D-4", "S-5",
                    "H-5", "C-5", "D-5", "S-6", "H-6", "C-6", "D-6", "S-7", "H-7", "C-7", "D-7", "S-8", "H-8", "C-8",
                    "D-8", "S-9", "H-9", "C-9", "D-9", "S-10", "H-10", "C-10", "D-10", "S-J", "H-J", "C-J", "D-J",
                    "S-Q", "H-Q", "C-Q", "D-Q", "S-K", "H-K", "C-K", "D-K", "S-A", "H-A", "C-A", "D-A"));

    public static void main(String[] args) {
        // 显示游戏介绍信息
        introMessage();
        // 根据用户输入显示游戏指导
        showDirectionsBasedOnInput();
        // 开始游戏
        playGame();
    }

    // 显示游戏介绍信息
    private static void introMessage() {
        System.out.println("\t         WAR");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("THIS IS THE CARD GAME OF WAR. EACH CARD IS GIVEN BY SUIT-#");
        System.out.print("AS S-7 FOR SPADE 7. DO YOU WANT DIRECTIONS? ");
    }

    // 根据用户输入显示游戏指导
    private static void showDirectionsBasedOnInput() {
// 在玩家选择选项之前保持循环。
boolean invalidInput = true;
while (invalidInput) {
    // 将用户输入转换为小写，并根据不同的选项执行相应的操作
    switch (userInput.nextLine().toLowerCase()) {
        case "yes":
            System.out.println("计算机给你和它一张'卡牌'。数值较高的卡牌获胜。当你选择不继续或者你用完整副牌时游戏结束。\n");
            invalidInput = false;
            break;
        case "no":
            System.out.println();
            invalidInput = false;
            break;
        default:
            System.out.print("请回答'是'或'否'。   ");
    }
}
        private static void playGame() {

        // Checks to see if the player ends the game early.
        // Ending early will cause a different output to appear.
        boolean gameEndedEarly = false; // 声明一个布尔变量，用于标记玩家是否提前结束游戏

        // Shuffle the deck of cards.
        Collections.shuffle(deckOfCards); // 对卡牌进行洗牌

        // Since the deck is already suffled, pull each card until the deck is empty or
        // until the user quits.
        outerloop: // 定义一个外部循环标签
        for (int i = 1; i <= CARD_DECK_SIZE; i += 2) { // 遍历卡牌
            System.out.println("YOU: " + deckOfCards.get(i - 1) + "\t " + "COMPUTER: " + deckOfCards.get(i)); // 打印玩家和计算机的卡牌
            getWinner(deckOfCards.get(i - 1), deckOfCards.get(i)); // 调用函数判断获胜者

            invalidInput = true; // 初始化一个布尔变量
            while (invalidInput) { // 进入循环，直到输入有效
                if (endedEarly()) { // 如果玩家提前结束游戏
                    // Player ended game early.
                    // 设置游戏结束标志为真，跳出外层循环
                    gameEndedEarly = true;
                    break outerloop;
                }
            }
        }

        // 调用结束游戏输出函数
        endGameOutput(gameEndedEarly);
    }

    /**
     * 输出当前回合的赢家。
     *
     * @param playerCard   玩家的卡牌。
     * @param computerCard 电脑的卡牌。
     */
    private static void getWinner(String playerCard, String computerCard) {

        // 返回抽取的卡牌的数值
        String playerCardScore = (playerCard.length() == 3) ? Character.toString(playerCard.charAt(2))
        // 从玩家卡片中提取分数，如果长度为3，则取第三个字符，否则取第二到第四个字符
        String playerCardScore = (playerCard.length() == 3) ? Character.toString(playerCard.charAt(2))
                : playerCard.substring(2, 4);
        // 从电脑卡片中提取分数，如果长度为3，则取第三个字符，否则取第二到第四个字符
        String computerCardScore = (computerCard.length() == 3) ? Character.toString(computerCard.charAt(2))
                : computerCard.substring(2, 4);

        // 比较玩家卡片和电脑卡片的分数，输出对应的结果
        if (checkCourtCards(playerCardScore) > checkCourtCards(computerCardScore)) {
            System.out.println("YOU WIN.   YOU HAVE " + playerWonRound() + "   COMPUTER HAS " + getComputerScore());
        } else if (checkCourtCards(playerCardScore) < checkCourtCards(computerCardScore)) {
            System.out.println(
                    "COMPUTER WINS!!!   YOU HAVE " + getPlayerScore() + "   COMPUTER HAS " + computerWonRound());
        } else {
            System.out.println("TIE.  NO SCORE CHANGE");
        }

        // 提示玩家是否继续游戏
        System.out.print("DO YOU WANT TO CONTINUE? ");
    }

    /**
     * @param cardScore Score of the card being pulled.
     * @return an integer value of the current card's score.
     */
    private static int checkCourtCards(String cardScore) {
        // 检查卡片分数，根据不同的情况返回相应的整数值
        switch (cardScore) {
            case "J":
                return Integer.parseInt("11");
            case "Q":
                return Integer.parseInt("12");
            case "K":
                return Integer.parseInt("13");
            case "A":
                return Integer.parseInt("14");
            default:
                return Integer.parseInt(cardScore);
        }
    }

    /**
     * @return true if the player ended the game early. false otherwise.
     */
    private static boolean endedEarly() {
        // 检查玩家是否提前结束游戏
        switch (userInput.nextLine().toLowerCase()) {
            case "yes":  # 如果输入是 "yes"
                invalidInput = false;  # 将 invalidInput 设为 false
                return false;  # 返回 false
            case "no":  # 如果输入是 "no"
                invalidInput = false;  # 将 invalidInput 设为 false
                return true;  # 返回 true
            default:  # 如果输入不是 "yes" 也不是 "no"
                invalidInput = true;  # 将 invalidInput 设为 true
                System.out.print("YES OR NO, PLEASE.   ");  # 打印提示信息
                return false;  # 返回 false
        }
    }

    /**
     * 根据游戏是否提前结束来显示输出。
     *
     * @param endedEarly 游戏是否提前结束，true 表示游戏提前结束，false 表示游戏未提前结束。
     */
    private static void endGameOutput(boolean endedEarly) {
        if (endedEarly) {  # 如果游戏提前结束
    // 打印游戏结束时的最终得分，包括玩家得分和计算机得分
    System.out.println("YOU HAVE ENDED THE GAME. FINAL SCORE:  YOU: " + getPlayerScore() + " COMPUTER: "
                    + getComputerScore());
    // 打印感谢信息
    System.out.println("THANKS FOR PLAYING.  IT WAS FUN.");
    // 如果游戏结束时玩家获胜，则打印玩家得分并返回
    } else {
    // 如果游戏结束时卡牌用完，打印最终得分和感谢信息
    System.out.println("WE HAVE RUN OUT OF CARDS. FINAL SCORE:  YOU: " + getPlayerScore() + " COMPUTER: "
                    + getComputerScore());
    // 打印感谢信息
    System.out.println("THANKS FOR PLAYING.  IT WAS FUN.");
    // 如果玩家赢得这一轮，增加玩家总得分并返回
    private static int playerWonRound() {
    return playerTotalScore += 1;
    // 获取玩家的总得分
    private static int getPlayerScore() {
    return playerTotalScore;
    private static int getPlayerScore() {
        return playerTotalScore;  # 返回玩家的总分
    }

    /**
     * Increment the computer's total score if they have won the round.
     */
    private static int computerWonRound() {
        return computerTotalScore += 1;  # 如果计算机赢得了这一轮，增加计算机的总分
    }

    /**
     * Get the computer's total score.
     */
    private static int getComputerScore() {
        return computerTotalScore;  # 获取计算机的总分
    }
}
```