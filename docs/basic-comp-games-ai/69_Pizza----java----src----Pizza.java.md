# `basic-computer-games\69_Pizza\java\src\Pizza.java`

```
import java.util.Scanner;

/**
 * Game of Pizza
 * <p>
 * Based on the Basic game of Hurkle here
 * https://github.com/coding-horror/basic-computer-games/blob/main/69%20Pizza/pizza.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Pizza {

    private final int MAX_DELIVERIES = 5; // 最大配送次数

    private enum GAME_STATE { // 游戏状态枚举
        STARTING, // 开始
        ENTER_NAME, // 输入名字
        DRAW_MAP, // 绘制地图
        MORE_DIRECTIONS, // 更多方向
        START_DELIVER, // 开始配送
        DELIVER_PIZZA, // 送达披萨
        TOO_DIFFICULT, // 太困难
        END_GAME, // 结束游戏
        GAME_OVER // 游戏结束
    }

    // houses that can order pizza
    private final char[] houses = new char[]{'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P'}; // 可以订购披萨的房屋

    // size of grid
    private final int[] gridPos = new int[]{1, 2, 3, 4}; // 网格大小

    private GAME_STATE gameState; // 游戏状态

    private String playerName; // 玩家名字

    // How many pizzas have been successfully delivered
    private int pizzaDeliveryCount; // 成功配送的披萨数量

    // current house that ordered a pizza
    private int currentHouseDelivery; // 当前订购披萨的房屋

    // Used for keyboard input
    private final Scanner kbScanner; // 用于键盘输入的扫描器

    public Pizza() {

        gameState = GAME_STATE.STARTING; // 初始化游戏状态为开始

        // Initialise kb scanner
        kbScanner = new Scanner(System.in); // 初始化键盘扫描器
    }

    /**
     * Main game loop
     */
}
    // 绘制城市海茨维尔的地图
    private void drawMap() {

        // 打印地图标题
        System.out.println("MAP OF THE CITY OF HYATTSVILLE");
        System.out.println();
        System.out.println(" -----1-----2-----3-----4-----");
        // 初始化变量 k
        int k = 3;
        // 循环绘制地图
        for (int i = 1; i < 5; i++) {
            System.out.println("-");
            System.out.println("-");
            System.out.println("-");
            System.out.println("-");

            // 打印地图上的位置和房屋信息
            System.out.print(gridPos[k]);
            int pos = 16 - 4 * i;
            System.out.print("     " + houses[pos]);
            System.out.print("     " + houses[pos + 1]);
            System.out.print("     " + houses[pos + 2]);
            System.out.print("     " + houses[pos + 3]);
            System.out.println("     " + gridPos[k]);
            k = k - 1;
        }
        System.out.println("-");
        System.out.println("-");
        System.out.println("-");
        System.out.println("-");
        System.out.println(" -----1-----2-----3-----4-----");
    }

    /**
     * 游戏的基本信息
     */
    private void intro() {
        // 打印游戏标题和地点
        System.out.println("PIZZA");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println();
        System.out.println("PIZZA DELIVERY GAME");
        System.out.println();
    }

    // 扩展游戏介绍
    private void extendedIntro() {
        System.out.println("THE OUTPUT IS A MAP OF THE HOMES WHERE");
        System.out.println("YOU ARE TO SEND PIZZAS.");
        System.out.println();
        System.out.println("YOUR JOB IS TO GIVE A TRUCK DRIVER");
        System.out.println("THE LOCATION OR COORDINATES OF THE");
        System.out.println("HOME ORDERING THE PIZZA.");
        System.out.println();
    }
    // 显示更多的指示信息
    private void displayMoreDirections() {
        System.out.println();
        System.out.println("SOMEBODY WILL ASK FOR A PIZZA TO BE");
        System.out.println("DELIVERED.  THEN A DELIVERY BOY WILL");
        System.out.println("ASK YOU FOR THE LOCATION.");
        System.out.println("     EXAMPLE:");
        System.out.println("THIS IS J.  PLEASE SEND A PIZZA.");
        System.out.println("DRIVER TO " + playerName + ".  WHERE DOES J LIVE?");
        System.out.println("YOUR ANSWER WOULD BE 2,3");
        System.out.println();
    }

    // 初始化 pizzaDeliveryCount 为 1
    private void init() {
        pizzaDeliveryCount = 1;
    }

    /**
     * 接受一个由逗号分隔的字符串，并返回第 n 个被分隔的值（从计数 0 开始）
     *
     * @param text - 由逗号分隔的值组成的字符串
     * @param pos  - 要返回值的位置
     * @return 值的整数表示
     */
    private int getDelimitedValue(String text, int pos) {
        String[] tokens = text.split(",");
        return Integer.parseInt(tokens[pos]);
    }

    /**
     * 如果给定的字符串等于调用 stringIsAnyValue 方法中指定的至少一个值，则返回 true
     *
     * @param text 要搜索的字符串
     * @return 如果字符串等于 varargs 中的一个值，则返回 true
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * 如果在文本中找到了 Y、YES、N 或 NO 中的一个值，则返回 true（不区分大小写）
     *
     * @param text 搜索的字符串
     * @return 如果在文本中找到了 varargs 中的一个值，则返回 true
     */
    private boolean yesOrNoEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES", "N", "NO");
    }
    /**
     * Returns true if a given string contains at least one of the varargs (2nd parameter).
     * Note: Case insensitive comparison.
     *
     * @param text   string to search
     * @param values varargs of type string containing values to compare
     * @return true if one of the varargs arguments was found in text
     */
    private boolean stringIsAnyValue(String text, String... values) {

        // Cycle through the variable number of values and test each
        for (String val : values) {
            // Check if the text contains the current value (case insensitive comparison)
            if (text.equalsIgnoreCase(val)) {
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
        // Display the message on the screen
        System.out.print(text);
        // Accept input from the keyboard and return it
        return kbScanner.next();
    }
# 闭合前面的函数定义
```