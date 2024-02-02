# `basic-computer-games\94_War\java\War.java`

```py
# 导入所需的类
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

/**
 * 由Nahid Mondol将BASIC转换为Java。
 *
 * 基于Trevor Hobson的方法。
 */
public class War {

    # 定义常量，表示卡牌的数量
    private static final int CARD_DECK_SIZE = 52;
    # 玩家总分数
    private static int playerTotalScore = 0;
    # 计算机总分数
    private static int computerTotalScore = 0;
    # 用于标记输入是否无效
    private static boolean invalidInput;
    # 创建Scanner对象，用于接收用户输入
    private static Scanner userInput = new Scanner(System.in);

    // 用于存储一副牌的简单方法。
    // 花色-数值，例如：梅花2 = S-2，方块K = D-K，等等...
    private static ArrayList<String> deckOfCards = new ArrayList<String>(
            Arrays.asList("S-2", "H-2", "C-2", "D-2", "S-3", "H-3", "C-3", "D-3", "S-4", "H-4", "C-4", "D-4", "S-5",
                    "H-5", "C-5", "D-5", "S-6", "H-6", "C-6", "D-6", "S-7", "H-7", "C-7", "D-7", "S-8", "H-8", "C-8",
                    "D-8", "S-9", "H-9", "C-9", "D-9", "S-10", "H-10", "C-10", "D-10", "S-J", "H-J", "C-J", "D-J",
                    "S-Q", "H-Q", "C-Q", "D-Q", "S-K", "H-K", "C-K", "D-K", "S-A", "H-A", "C-A", "D-A"));

    public static void main(String[] args) {
        # 显示游戏介绍信息
        introMessage();
        # 根据用户输入显示游戏规则
        showDirectionsBasedOnInput();
        # 开始游戏
        playGame();
    }

    # 显示游戏介绍信息
    private static void introMessage() {
        System.out.println("\t         WAR");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("THIS IS THE CARD GAME OF WAR. EACH CARD IS GIVEN BY SUIT-#");
        System.out.print("AS S-7 FOR SPADE 7. DO YOU WANT DIRECTIONS? ");
    }
    // 根据用户输入展示方向
    private static void showDirectionsBasedOnInput() {
        // 在玩家选择选项之前保持循环
        invalidInput = true;
        while (invalidInput) {
            switch (userInput.nextLine().toLowerCase()) {
                case "yes":
                    System.out.println("THE COMPUTER GIVES YOU AND IT A 'CARD'. THE HIGHER CARD");
                    System.out.println("(NUMERICALLY) WINS. THE GAME ENDS WHEN YOU CHOOSE NOT TO ");
                    System.out.println("CONTINUE OR WHEN YOU HAVE FINISHED THE PACK.\n");
                    invalidInput = false;
                    break;
                case "no":
                    System.out.println();
                    invalidInput = false;
                    break;
                default:
                    System.out.print("YES OR NO, PLEASE.   ");
            }
        }
    }

    // 开始游戏
    private static void playGame() {

        // 检查玩家是否提前结束游戏
        // 提前结束会导致不同的输出
        boolean gameEndedEarly = false;

        // 洗牌
        Collections.shuffle(deckOfCards);

        // 由于牌已经洗好，每次从牌堆中抽取一张牌，直到牌堆为空或玩家退出
        outerloop:
        for (int i = 1; i <= CARD_DECK_SIZE; i += 2) {
            System.out.println("YOU: " + deckOfCards.get(i - 1) + "\t " + "COMPUTER: " + deckOfCards.get(i));
            getWinner(deckOfCards.get(i - 1), deckOfCards.get(i));

            invalidInput = true;
            while (invalidInput) {
                if (endedEarly()) {
                    // 玩家提前结束游戏
                    // 跳出游戏循环并显示游戏结束输出
                    gameEndedEarly = true;
                    break outerloop;
                }
            }
        }

        endGameOutput(gameEndedEarly);
    }
    /**
     * 输出当前回合的赢家。
     *
     * @param playerCard   玩家的卡牌。
     * @param computerCard 电脑的卡牌。
     */
    private static void getWinner(String playerCard, String computerCard) {

        // 返回抽取的卡牌的数字值。
        String playerCardScore = (playerCard.length() == 3) ? Character.toString(playerCard.charAt(2))
                : playerCard.substring(2, 4);
        String computerCardScore = (computerCard.length() == 3) ? Character.toString(computerCard.charAt(2))
                : computerCard.substring(2, 4);

        if (checkCourtCards(playerCardScore) > checkCourtCards(computerCardScore)) {
            System.out.println("YOU WIN.   YOU HAVE " + playerWonRound() + "   COMPUTER HAS " + getComputerScore());
        } else if (checkCourtCards(playerCardScore) < checkCourtCards(computerCardScore)) {
            System.out.println(
                    "COMPUTER WINS!!!   YOU HAVE " + getPlayerScore() + "   COMPUTER HAS " + computerWonRound());
        } else {
            System.out.println("TIE.  NO SCORE CHANGE");
        }

        System.out.print("DO YOU WANT TO CONTINUE? ");
    }

    /**
     * @param cardScore 抽取卡牌的分数。
     * @return 当前卡牌分数的整数值。
     */
    private static int checkCourtCards(String cardScore) {
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
     * @return 如果玩家提前结束游戏则返回true，否则返回false。
     */
    // 根据用户输入判断是否提前结束游戏
    private static boolean endedEarly() {
        switch (userInput.nextLine().toLowerCase()) {
            case "yes":
                invalidInput = false;
                return false;
            case "no":
                invalidInput = false;
                return true;
            default:
                invalidInput = true;
                System.out.print("YES OR NO, PLEASE.   ");
                return false;
        }
    }

    /**
     * 根据游戏是否提前结束显示输出
     *
     * @param endedEarly 游戏是否提前结束，true表示提前结束，false表示未提前结束
     */
    private static void endGameOutput(boolean endedEarly) {
        if (endedEarly) {
            System.out.println("YOU HAVE ENDED THE GAME. FINAL SCORE:  YOU: " + getPlayerScore() + " COMPUTER: "
                    + getComputerScore());
            System.out.println("THANKS FOR PLAYING.  IT WAS FUN.");
        } else {
            System.out.println("WE HAVE RUN OUT OF CARDS. FINAL SCORE:  YOU: " + getPlayerScore() + " COMPUTER: "
                    + getComputerScore());
            System.out.println("THANKS FOR PLAYING.  IT WAS FUN.");
        }
    }

    /**
     * 如果玩家赢得了这一轮，增加玩家的总分数
     */
    private static int playerWonRound() {
        return playerTotalScore += 1;
    }

    /**
     * 获取玩家的总分数
     */
    private static int getPlayerScore() {
        return playerTotalScore;
    }

    /**
     * 如果计算机赢得了这一轮，增加计算机的总分数
     */
    private static int computerWonRound() {
        return computerTotalScore += 1;
    }

    /**
     * 获取计算机的总分数
     */
    private static int getComputerScore() {
        return computerTotalScore;
    }
# 闭合前面的函数定义
```