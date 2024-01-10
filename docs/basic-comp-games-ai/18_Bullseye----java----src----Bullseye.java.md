# `basic-computer-games\18_Bullseye\java\src\Bullseye.java`

```
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.Scanner;  // 导入 Scanner 类

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
    public static final int FIRST_IDENT = 10;  // 第一个缩进的长度
    public static final int SECOND_IDENT = 30;  // 第二个缩进的长度
    public static final int THIRD_INDENT = 30;  // 第三个缩进的长度

    // Used to decide throw result
    public static final double[] SHOT_ONE = new double[]{.65, .55, .5, .5};  // 第一种投掷的概率
    public static final double[] SHOT_TWO = new double[]{.99, .77, .43, .01};  // 第二种投掷的概率
    public static final double[] SHOT_THREE = new double[]{.95, .75, .45, .05};  // 第三种投掷的概率

    private enum GAME_STATE {  // 定义游戏状态的枚举类型
        STARTING,
        START_GAME,
        PLAYING,
        GAME_OVER
    }

    private GAME_STATE gameState;  // 游戏状态变量

    private final ArrayList<Player> players;  // 玩家列表

    private final Shot[] shots;  // 投掷数组

    // Used for keyboard input
    private final Scanner kbScanner;  // 键盘输入扫描器

    private int round;  // 回合数

    public Bullseye() {  // 构造函数

        gameState = GAME_STATE.STARTING;  // 初始游戏状态为 STARTING
        players = new ArrayList<>();  // 初始化玩家列表

        // Save the random chances of points based on shot type
        // 根据投掷类型保存随机得分的概率
        shots = new Shot[3];  // 初始化投掷数组
        shots[0] = new Shot(SHOT_ONE);  // 第一种投掷
        shots[1] = new Shot(SHOT_TWO);  // 第二种投掷
        shots[2] = new Shot(SHOT_THREE);  // 第三种投掷

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);  // 初始化键盘输入扫描器
    }

    /**
     * Main game loop
     */
    }

    /**
     * Display info about the game
     */
    // 打印游戏介绍信息
    private void intro() {
        System.out.println("BULLSEYE");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET");
        System.out.println("WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS");
        System.out.println("TO GET 200 POINTS.");
        System.out.println();
        // 打印表头
        System.out.println(paddedString("THROW", "DESCRIPTION", "PROBABLE SCORE"));
        // 打印不同投掷方式的描述和可能得分
        System.out.println(paddedString("1", "FAST OVERARM", "BULLSEYE OR COMPLETE MISS"));
        System.out.println(paddedString("2", "CONTROLLED OVERARM", "10, 20 OR 30 POINTS"));
        System.out.println(paddedString("3", "UNDERARM", "ANYTHING"));
    }

    /**
     * 计算玩家得分
     * 得分基于投掷类型加上一个随机因素
     *
     * @param playerThrow 1,2, or 3 表示投掷类型
     * @return 玩家得分
     */
    // 计算玩家得分，根据玩家投掷的结果
    private int calculatePlayerPoints(int playerThrow) {

        // -1是因为 Java 数组是从0开始的
        double p1 = this.shots[playerThrow - 1].getShot(0);
        double p2 = this.shots[playerThrow - 1].getShot(1);
        double p3 = this.shots[playerThrow - 1].getShot(2);
        double p4 = this.shots[playerThrow - 1].getShot(3);

        double random = Math.random();

        int points;

        // 根据随机数和投掷结果计算得分
        if (random >= p1) {
            System.out.println("BULLSEYE!!  40 POINTS!");
            points = 40;
            // 如果投掷是1（靶心或未命中），则将其视为未命中
            // 注意：这是对基本代码的修复，对于投掷类型1，它允许靶心，但如果未命中靶心，则不会将分数设为零（实际上应该这样做）。
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
     * 获取玩家的第1、2或3次投掷 - 如果输入无效，则重新询问
     *
     * @param player 我们正在计算投掷的玩家
     * @return 1、2或3，表示玩家的投掷
     */
    # 获取玩家的投掷结果
    private int getPlayersThrow(Player player) {
        # 初始化输入正确标志为假
        boolean inputCorrect = false;
        # 初始化投掷结果字符串
        String theThrow;
        # 循环直到输入正确
        do {
            # 显示提示信息并获取玩家输入的投掷结果
            theThrow = displayTextAndGetInput(player.getName() + "'S THROW ");
            # 如果输入为1、2或3，则标志输入正确
            if (theThrow.equals("1") || theThrow.equals("2") || theThrow.equals("3")) {
                inputCorrect = true;
            } else {
                # 如果输入不为1、2或3，则提示输入1、2或3
                System.out.println("INPUT 1, 2, OR 3!");
            }

        } while (!inputCorrect);

        # 将投掷结果字符串转换为整数并返回
        return Integer.parseInt(theThrow);
    }

    /**
     * 从键盘获取玩家的猜测
     *
     * @return 玩家的猜测作为整数
     */
    private int chooseNumberOfPlayers() {
        # 获取玩家输入的玩家数量并转换为整数返回
        return Integer.parseInt((displayTextAndGetInput("HOW MANY PLAYERS? ")));
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        # 打印消息并从键盘获取输入返回
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * 将三个字符串格式化为给定数量的空格
     * 替换原始基本代码使用制表符
     *
     * @param first  要打印在位置1的字符串
     * @param second 要打印在位置2的字符串
     * @param third  要打印在位置3的字符串
     * @return 格式化后的字符串
     */
    private String paddedString(String first, String second, String third) {
        # 格式化三个字符串并返回
        String output = String.format("%1$" + FIRST_IDENT + "s", first);
        output += String.format("%1$" + SECOND_IDENT + "s", second);
        output += String.format("%1$" + THIRD_INDENT + "s", third);
        return output;
    }
# 闭合前面的函数定义
```