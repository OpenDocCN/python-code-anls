# `69_Pizza\java\src\Pizza.java`

```
import java.util.Scanner;  // 导入 Scanner 类，用于从控制台读取输入

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

    private final int MAX_DELIVERIES = 5;  // 定义常量 MAX_DELIVERIES，表示最大配送次数

    private enum GAME_STATE {  // 定义枚举类型 GAME_STATE，表示游戏状态
        STARTING,  // 游戏开始状态
        ENTER_NAME,  // 输入玩家姓名状态
        DRAW_MAP,  // 绘制地图状态
        MORE_DIRECTIONS,  // 提示更多方向状态
        START_DELIVER, // 定义游戏状态为开始派送
        DELIVER_PIZZA, // 定义游戏状态为派送披萨
        TOO_DIFFICULT, // 定义游戏状态为太困难
        END_GAME, // 定义游戏状态为结束游戏
        GAME_OVER // 定义游戏状态为游戏结束
    }

    // 可以订购披萨的房屋
    private final char[] houses = new char[]{'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P'};

    // 网格的大小
    private final int[] gridPos = new int[]{1, 2, 3, 4};

    private GAME_STATE gameState; // 游戏状态

    private String playerName; // 玩家姓名

    // 成功派送的披萨数量
    private int pizzaDeliveryCount;
    // 当前订购披萨的房屋
    private int currentHouseDelivery;

    // 用于键盘输入
    private final Scanner kbScanner;

    public Pizza() {
        // 设置游戏状态为开始状态
        gameState = GAME_STATE.STARTING;

        // 初始化键盘扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * 主游戏循环
     */
    public void play() {
// 执行一个 do-while 循环，根据游戏状态执行不同的操作
do {
    // 当游戏状态为 STARTING 时，显示游戏介绍，并将游戏状态设置为 ENTER_NAME
    case STARTING:
        init(); // 初始化游戏
        intro(); // 显示游戏介绍
        gameState = GAME_STATE.ENTER_NAME; // 设置游戏状态为 ENTER_NAME
        break;

    // 当游戏状态为 ENTER_NAME 时，要求玩家输入名字，并显示相应信息，然后将游戏状态设置为 DRAW_MAP
    case ENTER_NAME:
        playerName = displayTextAndGetInput("WHAT IS YOUR FIRST NAME? "); // 要求玩家输入名字
        System.out.println("HI " + playerName + ". IN GAME YOU ARE TO TAKE ORDERS"); // 显示欢迎信息
        System.out.println("FOR PIZZAS.  THEN YOU ARE TO TELL A DELIVERY BOY"); // 显示游戏相关信息
        System.out.println("WHERE TO DELIVER THE ORDERED PIZZAS."); // 显示游戏相关信息
        System.out.println();
        gameState = GAME_STATE.DRAW_MAP; // 设置游戏状态为 DRAW_MAP
        break;
                // 绘制地图
                case DRAW_MAP:
                    drawMap();  // 调用绘制地图的函数
                    gameState = GAME_STATE.MORE_DIRECTIONS;  // 设置游戏状态为需要更多指示
                    break;

                // 需要更多指示（如何玩）？
                case MORE_DIRECTIONS:
                    extendedIntro();  // 调用扩展介绍的函数
                    String moreInfo = displayTextAndGetInput("DO YOU NEED MORE DIRECTIONS? ");  // 显示文本并获取输入
                    if (!yesOrNoEntered(moreInfo)) {  // 如果输入不是yes或no
                        System.out.println("'YES' OR 'NO' PLEASE, NOW THEN,");  // 输出提示信息
                    } else {
                        // 选择了更多指示
                        if (yesEntered(moreInfo)) {  // 如果输入是yes
                            displayMoreDirections();  // 显示更多指示
                            // 玩家现在明白了吗？
                            if (yesEntered(displayTextAndGetInput("UNDERSTAND? "))) {  // 如果输入是yes
                                System.out.println("GOOD.  YOU ARE NOW READY TO START TAKING ORDERS.");  // 输出准备好开始接受订单的信息
                                System.out.println();  // 输出空行
# 打印"GOOD LUCK!!"到控制台
System.out.println("GOOD LUCK!!");
# 打印空行到控制台
System.out.println();
# 将游戏状态设置为开始派送
gameState = GAME_STATE.START_DELIVER;
# 如果玩家的输入不被理解，游戏状态设置为太难，相当于游戏结束
} else {
    gameState = GAME_STATE.TOO_DIFFICULT;
# 如果不再需要更多的指示，开始派送披萨
} else {
    gameState = GAME_STATE.START_DELIVER;
# 如果任务太难理解，游戏结束
case TOO_DIFFICULT:
    System.out.println("JOB IS DEFINITELY TOO DIFFICULT FOR YOU. THANKS ANYWAY");
    gameState = GAME_STATE.GAME_OVER;
                // 开始配送披萨
                case START_DELIVER:
                    // 选择一个随机的房子，并为他们“订购”一份披萨。
                    currentHouseDelivery = (int) (Math.random()
                            * (houses.length) + 1) - 1; // 为基于0的数组减去1

                    System.out.println("HELLO " + playerName + "'S PIZZA.  THIS IS "
                            + houses[currentHouseDelivery] + ".");
                    System.out.println("  PLEASE SEND A PIZZA.");
                    gameState = GAME_STATE.DELIVER_PIZZA;
                    break;

                // 尝试送出披萨
                case DELIVER_PIZZA:

                    String question = "  DRIVER TO " + playerName + ":  WHERE DOES "
                            + houses[currentHouseDelivery] + " LIVE ? ";
                    String answer = displayTextAndGetInput(question);
                    // 将玩家输入的x，y转换为房屋的网格位置
                    int x = getDelimitedValue(answer, 0);  // 获取玩家输入中的x值
                    int y = getDelimitedValue(answer, 1);  // 获取玩家输入中的y值
                    int calculatedPos = (x + (y - 1) * 4) - 1;  // 计算房屋在网格中的位置

                    // 玩家选择了正确的房屋进行投递吗？
                    if (calculatedPos == currentHouseDelivery) {  // 如果计算出的位置与当前需要投递的房屋位置相同
                        System.out.println("HELLO " + playerName + ".  THIS IS " + houses[currentHouseDelivery]
                                + ", THANKS FOR THE PIZZA.");  // 打印投递成功的消息
                        pizzaDeliveryCount++;  // 增加投递次数

                        // 已经投递了足够的披萨吗？
                        if (pizzaDeliveryCount > MAX_DELIVERIES) {  // 如果投递次数超过了最大投递次数
                            gameState = GAME_STATE.END_GAME;  // 切换游戏状态为结束游戏
                        } else {
                            gameState = GAME_STATE.START_DELIVER;  // 否则切换游戏状态为开始投递
                        }
                    } else {
                        System.out.println("THIS IS " + houses[calculatedPos] + ".  I DID NOT ORDER A PIZZA.");  // 打印投递失败的消息
// 打印玩家所在位置的信息
System.out.println("I LIVE AT " + x + "," + y);
// 不改变游戏状态，以便再次执行状态

// 游戏结束时的签退信息，用于当警长不生气的情况
case END_GAME:
    // 如果玩家输入是肯定的，则初始化游戏并设置游戏状态为开始派送
    if (yesEntered(displayTextAndGetInput("DO YOU WANT TO DELIVER MORE PIZZAS? "))) {
        init();
        gameState = GAME_STATE.START_DELIVER;
    } else {
        System.out.println();
        System.out.println("O.K. " + playerName + ", SEE YOU LATER!");
        System.out.println();
        // 设置游戏状态为游戏结束
        gameState = GAME_STATE.GAME_OVER;
    }
    break;

// 游戏结束状态没有具体的情况
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }
```
这段代码是一个 do-while 循环的结束标志，当游戏状态不是 GAME_OVER 时继续执行循环。

```java
    private void drawMap() {
```
这是一个名为 drawMap 的私有方法，用于绘制地图。

```java
        System.out.println("MAP OF THE CITY OF HYATTSVILLE");
        System.out.println();
```
打印地图的标题和空行。

```java
        System.out.println(" -----1-----2-----3-----4-----");
```
打印地图的横向坐标。

```java
        int k = 3;
        for (int i = 1; i < 5; i++) {
```
初始化变量 k，并开始一个 for 循环，循环条件是 i 小于 5。

```java
            System.out.println("-");
            System.out.println("-");
            System.out.println("-");
            System.out.println("-");
```
打印分隔线。

```java
            System.out.print(gridPos[k]);
            int pos = 16 - 4 * i;
            System.out.print("     " + houses[pos]);
            System.out.print("     " + houses[pos + 1]);
```
打印地图上的位置和房屋信息。
            // 打印数组中位置为 pos+2 和 pos+3 的元素
            System.out.print("     " + houses[pos + 2]);
            System.out.print("     " + houses[pos + 3]);
            // 打印 gridPos 数组中位置为 k 的元素
            System.out.println("     " + gridPos[k]);
            // k 减一
            k = k - 1;
        }
        // 打印分隔线
        System.out.println("-");
        System.out.println("-");
        System.out.println("-");
        System.out.println("-");
        // 打印游戏提示信息
        System.out.println(" -----1-----2-----3-----4-----");
    }

    /**
     * 游戏的基本信息
     */
    private void intro() {
        // 打印游戏名称
        System.out.println("PIZZA");
        // 打印游戏开发者信息
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        // 打印空行
        System.out.println();
        // 打印空行
        System.out.println();
        # 打印"Pizza Delivery Game"的提示信息
        System.out.println("PIZZA DELIVERY GAME")
        # 打印空行
        System.out.println()
    }

    # 打印扩展介绍信息
    private void extendedIntro() {
        System.out.println("THE OUTPUT IS A MAP OF THE HOMES WHERE")
        System.out.println("YOU ARE TO SEND PIZZAS.")
        System.out.println()
        System.out.println("YOUR JOB IS TO GIVE A TRUCK DRIVER")
        System.out.println("THE LOCATION OR COORDINATES OF THE")
        System.out.println("HOME ORDERING THE PIZZA.")
        System.out.println()
    }

    # 显示更多指示信息
    private void displayMoreDirections() {
        System.out.println()
        System.out.println("SOMEBODY WILL ASK FOR A PIZZA TO BE")
        System.out.println("DELIVERED.  THEN A DELIVERY BOY WILL")
        System.out.println("ASK YOU FOR THE LOCATION.")
        System.out.println("     EXAMPLE:")
        System.out.println("THIS IS J.  PLEASE SEND A PIZZA.");  // 打印出提示信息
        System.out.println("DRIVER TO " + playerName + ".  WHERE DOES J LIVE?");  // 打印出提示信息，包含玩家的名字
        System.out.println("YOUR ANSWER WOULD BE 2,3");  // 打印出提示信息
        System.out.println();  // 打印空行
    }

    private void init() {
        pizzaDeliveryCount = 1;  // 初始化 pizzaDeliveryCount 为 1
    }

    /**
     * Accepts a string delimited by comma's and returns the nth delimited
     * value (starting at count 0).
     *
     * @param text - text with values separated by comma's  // 参数说明，接受由逗号分隔的字符串
     * @param pos  - which position to return a value for  // 参数说明，要返回值的位置
     * @return the int representation of the value  // 返回值的说明
     */
    private int getDelimitedValue(String text, int pos) {
        String[] tokens = text.split(",");  // 使用逗号分隔字符串，存储到 tokens 数组中
        return Integer.parseInt(tokens[pos]);  // 将字符串转换为整数并返回

    /**
     * Returns true if a given string is equal to at least one of the values specified in the call
     * to the stringIsAnyValue method
     *
     * @param text string to search  // 要搜索的字符串
     * @return true if string is equal to one of the varargs  // 如果字符串等于其中一个变量，则返回true
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");  // 调用stringIsAnyValue方法，判断text是否等于"Y"或"YES"
    }

    /**
     * returns true if Y, YES, N, or NO was the compared value in text
     * case-insensitive
     *
     * @param text search string  // 要搜索的字符串
     * @return true if one of the varargs was found in text  // 如果在text中找到其中一个变量，则返回true
    */
    // 检查用户输入的文本是否为指定的值之一，不区分大小写
    private boolean yesOrNoEntered(String text) {
        // 调用 stringIsAnyValue 方法，判断用户输入的文本是否为 "Y", "YES", "N", "NO" 中的一个
        return stringIsAnyValue(text, "Y", "YES", "N", "NO");
    }

    /**
     * 返回 true 如果给定的字符串包含至少一个可变参数（第二个参数）。
     * 注意：不区分大小写比较。
     *
     * @param text   要搜索的字符串
     * @param values 包含要比较的值的可变参数类型的字符串
     * @return 如果在文本中找到了可变参数中的一个，则返回 true
     */
    // 检查文本是否包含可变参数中的任意一个值，不区分大小写
    private boolean stringIsAnyValue(String text, String... values) {

        // 循环遍历可变数量的值，并测试每个值
        for (String val : values) {
            // 如果文本与值相等（不区分大小写），则返回 true
            if (text.equalsIgnoreCase(val)) {
                return true;
            }
        }

        // no matches
        return false;
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text); // 在屏幕上打印消息
        return kbScanner.next(); // 从键盘接受输入并返回
    }

}
```