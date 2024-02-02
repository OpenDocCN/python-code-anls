# `basic-computer-games\22_Change\java\src\Change.java`

```py
import java.util.Arrays;  // 导入 Arrays 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 * Game of Change
 * <p>
 * Based on the Basic game of Change here
 * https://github.com/coding-horror/basic-computer-games/blob/main/22%20Change/change.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Change {

    // Used for keyboard input
    private final Scanner kbScanner;  // 创建 Scanner 对象用于键盘输入

    private enum GAME_STATE {  // 创建枚举类型 GAME_STATE
        START_GAME,  // 开始游戏
        INPUT,  // 输入
        CALCULATE,  // 计算
        END_GAME,  // 结束游戏
        GAME_OVER  // 游戏结束
    }

    // Current game state
    private GAME_STATE gameState;  // 当前游戏状态

    // Amount of change needed to be given
    private double change;  // 需要找零的金额

    public Change() {
        kbScanner = new Scanner(System.in);  // 初始化 Scanner 对象

        gameState = GAME_STATE.START_GAME;  // 设置游戏状态为开始游戏
    }

    /**
     * Main game loop
     */
    // 游戏进行的方法
    public void play() {

        do {
            // 根据游戏状态进行不同的操作
            switch (gameState) {
                // 游戏开始状态
                case START_GAME:
                    // 进行游戏介绍
                    intro();
                    // 切换游戏状态为输入状态
                    gameState = GAME_STATE.INPUT;
                    break;

                // 输入状态
                case INPUT:

                    // 获取物品成本
                    double costOfItem = displayTextAndGetNumber("COST OF ITEM ");
                    // 获取支付金额
                    double amountPaid = displayTextAndGetNumber("AMOUNT OF PAYMENT ");
                    // 计算找零
                    change = amountPaid - costOfItem;
                    if (change == 0) {
                        // 无需找零
                        System.out.println("CORRECT AMOUNT, THANK YOU.");
                        // 切换游戏状态为结束状态
                        gameState = GAME_STATE.END_GAME;
                    } else if (change < 0) {
                        System.out.println("YOU HAVE SHORT-CHANGES ME $" + (costOfItem - amountPaid));
                        // 不改变游戏状态，使其循环回去再试一次
                    } else {
                        // 需要找零
                        gameState = GAME_STATE.CALCULATE;
                    }
                    break;

                // 计算状态
                case CALCULATE:
                    System.out.println("YOUR CHANGE, $" + change);
                    // 计算找零
                    calculateChange();
                    // 切换游戏状态为结束状态
                    gameState = GAME_STATE.END_GAME;
                    break;

                // 结束状态
                case END_GAME:
                    System.out.println("THANK YOU, COME AGAIN");
                    System.out.println();
                    // 切换游戏状态为输入状态
                    gameState = GAME_STATE.INPUT;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    /**
     * 根据支付的金额计算并输出所需的找零
     */
    // 计算找零
    private void calculateChange() {

        // 保存原始找零金额
        double originalChange = change;

        // 计算十美元纸币的数量
        int tenDollarBills = (int) change / 10;
        // 如果有十美元纸币，则打印数量
        if (tenDollarBills > 0) {
            System.out.println(tenDollarBills + " TEN DOLLAR BILL(S)");
        }
        // 更新找零金额
        change = originalChange - (tenDollarBills * 10);

        // 计算五美元纸币的数量
        int fiveDollarBills = (int) change / 5;
        // 如果有五美元纸币，则打印数量
        if (fiveDollarBills > 0) {
            System.out.println(fiveDollarBills + " FIVE DOLLAR BILL(S)");
        }
        // 更新找零金额
        change = originalChange - (tenDollarBills * 10 + fiveDollarBills * 5);

        // 计算一美元纸币的数量
        int oneDollarBills = (int) change;
        // 如果有一美元纸币，则打印数量
        if (oneDollarBills > 0) {
            System.out.println(oneDollarBills + " ONE DOLLAR BILL(S)");
        }
        // 更新找零金额
        change = originalChange - (tenDollarBills * 10 + fiveDollarBills * 5 + oneDollarBills);

        // 将找零金额转换为分
        change = change * 100;
        double cents = change;

        // 计算五角硬币的数量
        int halfDollars = (int) change / 50;
        // 如果有五角硬币，则打印数量
        if (halfDollars > 0) {
            System.out.println(halfDollars + " ONE HALF DOLLAR(S)");
        }
        // 更新找零金额
        change = cents - (halfDollars * 50);

        // 计算25美分硬币的数量
        int quarters = (int) change / 25;
        // 如果有25美分硬币，则打印数量
        if (quarters > 0) {
            System.out.println(quarters + " QUARTER(S)");
        }
        // 更新找零金额
        change = cents - (halfDollars * 50 + quarters * 25);

        // 计算10美分硬币的数量
        int dimes = (int) change / 10;
        // 如果有10美分硬币，则打印数量
        if (dimes > 0) {
            System.out.println(dimes + " DIME(S)");
        }
        // 更新找零金额
        change = cents - (halfDollars * 50 + quarters * 25 + dimes * 10);

        // 计算5美分硬币的数量
        int nickels = (int) change / 5;
        // 如果有5美分硬币，则打印数量
        if (nickels > 0) {
            System.out.println(nickels + " NICKEL(S)");
        }
        // 更新找零金额
        change = cents - (halfDollars * 50 + quarters * 25 + dimes * 10 + nickels * 5);

        // 计算1美分硬币的数量
        int pennies = (int) (change + .5);
        // 如果有1美分硬币，则打印数量
        if (pennies > 0) {
            System.out.println(pennies + " PENNY(S)");
        }

    }
    // 打印介绍信息
    private void intro() {
        System.out.println(simulateTabs(33) + "CHANGE");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE");
        System.out.println("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.");
        System.out.println();
    }

    /*
     * 打印屏幕上的消息，然后从键盘接受输入。
     * 将输入转换为双精度数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private double displayTextAndGetNumber(String text) {
        return Double.parseDouble(displayTextAndGetInput(text));
    }

    /*
     * 打印屏幕上的消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * 模拟旧的基本tab(xx)命令，该命令通过xx个空格缩进文本。
     *
     * @param spaces 需要的空格数
     * @return 具有指定空格数的字符串
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }
# 闭合前面的函数定义
```