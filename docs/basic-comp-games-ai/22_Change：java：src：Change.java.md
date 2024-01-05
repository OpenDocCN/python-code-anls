# `22_Change\java\src\Change.java`

```
import java.util.Arrays;
import java.util.Scanner;

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
    private final Scanner kbScanner;  // 创建一个Scanner对象用于键盘输入

    private enum GAME_STATE {  // 创建一个枚举类型GAME_STATE
        START_GAME,  // 游戏开始状态
        INPUT,  // 输入状态
        CALCULATE,  // 定义枚举类型，表示计算状态
        END_GAME,   // 定义枚举类型，表示游戏结束状态
        GAME_OVER   // 定义枚举类型，表示游戏失败状态
    }

    // 当前游戏状态
    private GAME_STATE gameState;  // 声明私有变量，表示游戏状态

    // 需要找零的金额
    private double change;  // 声明私有变量，表示需要找零的金额

    public Change() {
        kbScanner = new Scanner(System.in);  // 创建一个用于接收用户输入的 Scanner 对象

        gameState = GAME_STATE.START_GAME;  // 初始化游戏状态为开始游戏
    }

    /**
     * 主游戏循环
     */
    public void play() {  # 定义一个名为play的公共方法

        do {  # 开始一个do-while循环
            switch (gameState) {  # 根据gameState的值进行不同的操作
                case START_GAME:  # 如果gameState为START_GAME
                    intro();  # 调用intro方法
                    gameState = GAME_STATE.INPUT;  # 将gameState设置为GAME_STATE.INPUT
                    break;  # 跳出switch语句

                case INPUT:  # 如果gameState为INPUT

                    double costOfItem = displayTextAndGetNumber("COST OF ITEM ");  # 调用displayTextAndGetNumber方法获取商品成本
                    double amountPaid = displayTextAndGetNumber("AMOUNT OF PAYMENT ");  # 调用displayTextAndGetNumber方法获取支付金额
                    change = amountPaid - costOfItem;  # 计算找零金额
                    if (change == 0) {  # 如果找零金额为0
                        // No change needed  # 输出提示信息
                        System.out.println("CORRECT AMOUNT, THANK YOU.");  # 输出提示信息
                        gameState = GAME_STATE.END_GAME;  # 将gameState设置为GAME_STATE.END_GAME
                    } else if (change < 0) {  # 如果找零金额小于0
                        System.out.println("YOU HAVE SHORT-CHANGES ME $" + (costOfItem - amountPaid));  # 输出提示信息
                    } else {
                        // 如果需要改变，将游戏状态改为CALCULATE
                        gameState = GAME_STATE.CALCULATE;
                    }
                    break;

                case CALCULATE:
                    // 打印出找零金额
                    System.out.println("YOUR CHANGE, $" + change);
                    // 调用calculateChange方法计算找零
                    calculateChange();
                    // 将游戏状态改为END_GAME
                    gameState = GAME_STATE.END_GAME;
                    break;

                case END_GAME:
                    // 打印结束游戏的提示语
                    System.out.println("THANK YOU, COME AGAIN");
                    System.out.println();
                    // 将游戏状态改为INPUT
                    gameState = GAME_STATE.INPUT;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }
    /**
     * 计算并输出根据支付的金额所需的找零。
     */
    private void calculateChange() {

        double originalChange = change;

        // 计算需要的十元纸币数量
        int tenDollarBills = (int) change / 10;
        if (tenDollarBills > 0) {
            System.out.println(tenDollarBills + " TEN DOLLAR BILL(S)");
        }
        change = originalChange - (tenDollarBills * 10);

        // 计算需要的五元纸币数量
        int fiveDollarBills = (int) change / 5;
        if (fiveDollarBills > 0) {
            System.out.println(fiveDollarBills + " FIVE DOLLAR BILL(S)");
        }
        change = originalChange - (tenDollarBills * 10 + fiveDollarBills * 5);
        // 将找零金额转换为整数部分，表示美元的数量
        int oneDollarBills = (int) change;
        // 如果有一美元的零钱，则打印出数量
        if (oneDollarBills > 0) {
            System.out.println(oneDollarBills + " ONE DOLLAR BILL(S)");
        }
        // 更新找零金额，减去已经计算过的一美元的数量
        change = originalChange - (tenDollarBills * 10 + fiveDollarBills * 5 + oneDollarBills);

        // 将找零金额转换为以分为单位的整数
        change = change * 100;
        double cents = change;

        // 计算半美元的数量
        int halfDollars = (int) change / 50;
        // 如果有半美元的零钱，则打印出数量
        if (halfDollars > 0) {
            System.out.println(halfDollars + " ONE HALF DOLLAR(S)");
        }
        // 更新找零金额，减去已经计算过的半美元的数量
        change = cents - (halfDollars * 50);

        // 计算25美分硬币的数量
        int quarters = (int) change / 25;
        // 如果有25美分的零钱，则打印出数量
        if (quarters > 0) {
            System.out.println(quarters + " QUARTER(S)");
        }
        # 计算剩余的零钱
        change = cents - (halfDollars * 50 + quarters * 25);

        # 计算需要的 10 分硬币数量
        int dimes = (int) change / 10;
        # 如果需要的 10 分硬币数量大于 0，则打印出数量和类型
        if (dimes > 0) {
            System.out.println(dimes + " DIME(S)");
        }

        # 计算剩余的零钱
        change = cents - (halfDollars * 50 + quarters * 25 + dimes * 10);

        # 计算需要的 5 分硬币数量
        int nickels = (int) change / 5;
        # 如果需要的 5 分硬币数量大于 0，则打印出数量和类型
        if (nickels > 0) {
            System.out.println(nickels + " NICKEL(S)");
        }

        # 计算剩余的零钱
        change = cents - (halfDollars * 50 + quarters * 25 + dimes * 10 + nickels * 5);

        # 计算需要的 1 分硬币数量
        int pennies = (int) (change + .5);
        # 如果需要的 1 分硬币数量大于 0，则打印出数量和类型
        if (pennies > 0) {
            System.out.println(pennies + " PENNY(S)");
    }

    private void intro() {
        // 打印欢迎信息和提示信息
        System.out.println(simulateTabs(33) + "CHANGE");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE");
        System.out.println("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.");
        System.out.println();
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     * 将输入转换为Double类型。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private double displayTextAndGetNumber(String text) {
        // 显示文本消息并从键盘获取输入，然后将输入转换为 double 类型并返回
        return Double.parseDouble(displayTextAndGetInput(text));
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        // 在屏幕上打印文本消息
        System.out.print(text);
        // 从键盘获取输入并返回
        return kbScanner.next();
    }

    /**
     * 模拟旧的基本 tab(xx) 命令，该命令将文本缩进 xx 个空格。
     *
     * @param spaces 需要的空格数
     * @return 包含指定数量空格的字符串
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```