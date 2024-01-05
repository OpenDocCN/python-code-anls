# `43_Hammurabi\java\src\Hamurabi.java`

```
import java.util.Arrays;  // 导入 Arrays 类，用于操作数组
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

/**
 * Game of Hamurabi
 * <p>
 * Based on the Basic game of Hamurabi here
 * https://github.com/coding-horror/basic-computer-games/blob/main/43%20Hammurabi/hammurabi.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Hamurabi {

    public static final int INITIAL_POPULATION = 95;  // 初始人口数量
    public static final int INITIAL_BUSHELS = 2800;  // 初始粮食数量
    public static final int INITIAL_HARVEST = 3000;  // 初始收成数量
    public static final int INITIAL_LAND_TRADING_AT = 3;  // 初始土地交易价格
    public static final int INITIAL_CAME_TO_CITY = 5;  // 初始来到城市的人数
    public static final int MAX_GAME_YEARS = 10;  // 游戏最大年限
    # 定义一个常量，表示一年中最大的饥饿程度
    public static final double MAX_STARVATION_IN_A_YEAR = .45d;

    # 定义私有变量，表示年份、人口数量、土地面积、小麦数量、收获量、土地交易价格、进城人数、一年中饿死的人数、总共饿死的人数、是否发生瘟疫、被老鼠吃掉的小麦数量、养活的人口数量、饿死人口的百分比、用于喂养人口的小麦数量
    private int year;
    private int population;
    private int acres;
    private int bushels;
    private int harvest;
    private int landTradingAt;
    private int cameToCity;
    private int starvedInAYear;
    private int starvedOverall;
    private boolean chanceOfPlague;
    private int ratsAte;
    private double peopleFed;
    private double percentageStarved;
    private int bushelsToFeedPeople;

    # 用于键盘输入的 Scanner 对象
    private final Scanner kbScanner;
    // 定义游戏状态的枚举类型
    private enum GAME_STATE {
        STARTUP, // 游戏开始状态
        INIT, // 初始化状态
        YEAR_CYCLE, // 年度循环状态
        BUY_ACRES, // 购买土地状态
        SELL_ACRES, // 出售土地状态
        FEED_PEOPLE, // 喂养人口状态
        PLANT_SEED, // 种植种子状态
        CALCULATE_HARVEST, // 计算收成状态
        CALCULATE_BABIES, // 计算新生儿状态
        RESULTS, // 游戏结果状态
        FINISH_GAME, // 游戏结束状态
        GAME_OVER // 游戏失败状态
    }

    // 当前游戏状态
    private GAME_STATE gameState;

    // 构造函数，初始化键盘输入扫描器
    public Hamurabi() {
        kbScanner = new Scanner(System.in);
        gameState = GAME_STATE.STARTUP;  # 初始化游戏状态为启动状态

    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                case STARTUP:
                    intro();  # 调用intro()函数，显示游戏介绍
                    gameState = GAME_STATE.INIT;  # 将游戏状态设置为初始化状态
                    break;

                case INIT:

                    // These are hard coded startup figures from the basic program
                    year = 0;  # 将年份初始化为0
                    # 设置初始人口
                    population = INITIAL_POPULATION;
                    # 设置初始谷物数量
                    bushels = INITIAL_BUSHELS;
                    # 设置初始收成
                    harvest = INITIAL_HARVEST;
                    # 设置初始土地交易价格
                    landTradingAt = INITIAL_LAND_TRADING_AT;
                    # 计算初始土地数量
                    acres = INITIAL_HARVEST / INITIAL_LAND_TRADING_AT;
                    # 设置初始进城人数
                    cameToCity = INITIAL_CAME_TO_CITY;
                    # 初始年内饥饿人数
                    starvedInAYear = 0;
                    # 总共饥饿人数
                    starvedOverall = 0;
                    # 是否发生瘟疫
                    chanceOfPlague = false;
                    # 老鼠吃掉的谷物数量
                    ratsAte = INITIAL_HARVEST - INITIAL_BUSHELS;
                    # 被喂养的人数
                    peopleFed = 0;
                    # 饥饿人口占总人口的百分比
                    percentageStarved = 0;
                    # 用于喂养人口的谷物数量
                    bushelsToFeedPeople = 0;

                    # 设置游戏状态为年周期
                    gameState = GAME_STATE.YEAR_CYCLE;
                    # 跳出switch语句
                    break;

                case YEAR_CYCLE:
                    # 打印空行
                    System.out.println();
                    # 年数加一
                    year += 1;
                    // 游戏结束了吗？
                    if (year > MAX_GAME_YEARS) {
                        gameState = GAME_STATE.RESULTS;
                        break;
                    }
                    System.out.println("HAMURABI:  I BEG TO REPORT TO YOU,");
                    System.out.println("IN YEAR " + year + "," + starvedInAYear + " PEOPLE STARVED," + cameToCity + " CAME TO THE CITY,");
                    population += cameToCity;
                    if (chanceOfPlague) {
                        population /= 2;
                        System.out.println("A HORRIBLE PLAGUE STRUCK!  HALF THE PEOPLE DIED.");
                    }
                    System.out.println("POPULATION IS NOW " + population);
                    System.out.println("THE CITY NOW OWNS " + acres + " ACRES.");
                    System.out.println("YOU HARVESTED " + landTradingAt + " BUSHELS PER ACRE.");
                    System.out.println("THE RATS ATE " + ratsAte + " BUSHELS.");
                    System.out.println("YOU NOW HAVE " + bushels + " BUSHELS IN STORE.");
                    System.out.println();
                    landTradingAt = (int) (Math.random() * 10) + 17;  // 生成一个介于17到26之间的随机整数，表示土地的交易价格
                    System.out.println("LAND IS TRADING AT " + landTradingAt + " BUSHELS PER ACRE.");  // 打印土地的交易价格

                    gameState = GAME_STATE.BUY_ACRES;  // 设置游戏状态为购买土地
                    break;  // 跳出switch语句

                case BUY_ACRES:  // 当游戏状态为购买土地时执行以下代码
                    int acresToBuy = displayTextAndGetNumber("HOW MANY ACRES DO YOU WISH TO BUY? ");  // 获取玩家希望购买的土地数量
                    if (acresToBuy < 0) {  // 如果土地数量小于0
                        gameState = GAME_STATE.FINISH_GAME;  // 设置游戏状态为结束游戏
                    }

                    if (acresToBuy > 0) {  // 如果土地数量大于0
                        if ((landTradingAt * acresToBuy) > bushels) {  // 如果购买土地所需的总价格大于玉米数量
                            notEnoughBushelsMessage();  // 提示玉米数量不足
                        } else {
                            acres += acresToBuy;  // 增加玩家拥有的土地数量
                            bushels -= (landTradingAt * acresToBuy);  // 减去购买土地所需的玉米数量
                            peopleFed = 0;  // 重置已喂养的人数
                            gameState = GAME_STATE.FEED_PEOPLE;  // 设置游戏状态为喂养人口
                } else {
                    // 如果输入的购买数量为0，则尝试出售
                    gameState = GAME_STATE.SELL_ACRES;
                }
                break;

            case SELL_ACRES:
                // 获取玩家希望出售的土地数量
                int acresToSell = displayTextAndGetNumber("HOW MANY ACRES DO YOU WISH TO SELL? ");
                // 如果输入的土地数量小于0，则结束游戏
                if (acresToSell < 0) {
                    gameState = GAME_STATE.FINISH_GAME;
                }
                // 如果输入的土地数量小于当前拥有的土地数量
                if (acresToSell < acres) {
                    // 减去出售的土地数量
                    acres -= acresToSell;
                    // 增加相应数量的粮食
                    bushels += (landTradingAt * acresToSell);
                    // 进入下一个阶段：喂养人口
                    gameState = GAME_STATE.FEED_PEOPLE;
                } else {
                    // 如果土地数量不足，则显示提示信息
                    notEnoughLandMessage();
                }
                break;
# 如果选择了FEED_PEOPLE状态
case FEED_PEOPLE:
    # 获取玩家输入的想要用多少粮食来喂养人口
    bushelsToFeedPeople = displayTextAndGetNumber("HOW MANY BUSHELS DO YOU WISH TO FEED YOUR PEOPLE ? ");
    # 如果输入的粮食数量小于0，则游戏状态变为结束游戏
    if (bushelsToFeedPeople < 0) {
        gameState = GAME_STATE.FINISH_GAME;
    }
    # 如果输入的粮食数量小于等于当前拥有的粮食数量
    if (bushelsToFeedPeople <= bushels) {
        # 减去相应数量的粮食
        bushels -= bushelsToFeedPeople;
        # 喂养人口数量设为1
        peopleFed = 1;
        # 游戏状态变为种植种子
        gameState = GAME_STATE.PLANT_SEED;
    } else {
        # 如果输入的粮食数量大于当前拥有的粮食数量，则显示粮食不足的消息
        notEnoughBushelsMessage();
    }
    break;

# 如果选择了PLANT_SEED状态
case PLANT_SEED:
    # 获取玩家输入的想要种植的土地面积
    int acresToPlant = displayTextAndGetNumber("HOW MANY ACRES DO YOU WISH TO PLANT WITH SEED ? ");
                    # 如果要种植的土地面积小于0，则游戏状态变为结束游戏
                    if (acresToPlant < 0) {
                        gameState = GAME_STATE.FINISH_GAME;
                    }

                    # 如果要种植的土地面积小于等于当前拥有的土地面积
                    if (acresToPlant <= acres) {
                        # 如果要种植的土地面积的一半小于等于当前拥有的小麦数量
                        if (acresToPlant / 2 <= bushels) {
                            # 如果要种植的土地面积小于当前人口的10倍
                            if (acresToPlant < 10 * population) {
                                # 减去用于种植的小麦数量
                                bushels -= acresToPlant / 2;
                                # 计算喂养的人口数量
                                peopleFed = (int) (Math.random() * 5) + 1;
                                # 土地交易价格等于喂养的人口数量
                                landTradingAt = (int) peopleFed;
                                # 计算收获的小麦数量
                                harvest = acresToPlant * landTradingAt;
                                # 老鼠吃掉的小麦数量为0
                                ratsAte = 0;
                                # 游戏状态变为计算收获
                                gameState = GAME_STATE.CALCULATE_HARVEST;
                            } else {
                                # 如果要种植的土地面积大于等于当前人口的10倍，则显示人口不足的消息
                                notEnoughPeopleMessage();
                            }
                        } else {
                            # 如果要种植的土地面积的一半大于当前拥有的小麦数量，则显示小麦不足的消息
                            notEnoughBushelsMessage();
                        }
                    } else {
                    notEnoughLandMessage();
                    // 如果土地不够，显示土地不足的消息
                    break;

                case CALCULATE_HARVEST:
                    if ((int) (peopleFed / 2) == peopleFed / 2) {
                        // 如果人口的一半是偶数，表示老鼠肆虐
                        ratsAte = (int) (bushels / peopleFed);
                    }
                    // 减去老鼠吃掉的粮食
                    bushels = bushels - ratsAte;
                    // 增加收获的粮食
                    bushels += harvest;
                    // 设置游戏状态为计算人口增长
                    gameState = GAME_STATE.CALCULATE_BABIES;
                    break;

                case CALCULATE_BABIES:
                    // 计算前来城市的人口
                    cameToCity = (int) (peopleFed * (20 * acres + bushels) / population / 100 + 1);
                    // 计算喂养人口所需的粮食
                    peopleFed = (bushelsToFeedPeople / 20.0d);
                    // 将瘟疫的几率简化为真/假
                    // 计算是否发生瘟疫的概率
                    chanceOfPlague = (int) ((10 * (Math.random() * 2) - .3)) == 0;
                    // 如果人口数量小于被喂养的人口数量，则游戏状态变为年度循环
                    if (population < peopleFed) {
                        gameState = GAME_STATE.YEAR_CYCLE;
                    }

                    // 计算饿死的人数
                    double starved = population - peopleFed;
                    if (starved < 0.0d) {
                        // 如果饿死人数小于0，则将一年内饿死人数设为0，游戏状态变为年度循环
                        starvedInAYear = 0;
                        gameState = GAME_STATE.YEAR_CYCLE;
                    } else {
                        // 否则，记录一年内饿死的人数，并累加到总饿死人数中
                        starvedInAYear = (int) starved;
                        starvedOverall += starvedInAYear;
                        // 如果饿死人数超过了一年内最大饿死人数的限制，则显示消息，游戏状态变为结束游戏
                        if (starved > MAX_STARVATION_IN_A_YEAR * population) {
                            starvedTooManyPeopleMessage((int) starved);
                            gameState = GAME_STATE.FINISH_GAME;
                        } else {
                            // 否则，计算饿死人数的百分比，并更新人口数量，游戏状态变为年度循环
                            percentageStarved = ((year - 1) * percentageStarved + starved * 100 / population) / year;
                            population = (int) peopleFed;
                            gameState = GAME_STATE.YEAR_CYCLE;
                        }
                    }

                    break;


                case RESULTS:

                    // 计算每个人的土地面积
                    int acresPerPerson = acres / population;

                    // 打印输出结果
                    System.out.println("IN YOUR 10-YEAR TERM OF OFFICE," + String.format("%.2f", percentageStarved) + "% PERCENT OF THE");
                    System.out.println("POPULATION STARVED PER YEAR ON THE AVERAGE, I.E. A TOTAL OF");
                    System.out.println(starvedOverall + " PEOPLE DIED!!");
                    System.out.println("YOU STARTED WITH 10 ACRES PER PERSON AND ENDED WITH");
                    System.out.println(acresPerPerson + " ACRES PER PERSON.");
                    System.out.println();

                    // 根据条件输出不同的消息
                    if (percentageStarved > 33.0d || acresPerPerson < 7) {
                        starvedTooManyPeopleMessage(starvedOverall);
                    } else if (percentageStarved > 10.0d || acresPerPerson < 9) {
                    heavyHandedMessage();  # 如果饥饿人数超过30%或者每人耕种面积小于10英亩，则输出重手段的消息
                } else if (percentageStarved > 3.0d || acresPerPerson < 10) {
                    couldHaveBeenBetterMessage();  # 如果饥饿人数超过30%或者每人耕种面积小于10英亩，则输出本可以更好的消息
                } else {
                    fantasticPerformanceMessage();  # 否则输出表现出色的消息
                }


                gameState = GAME_STATE.FINISH_GAME;  # 设置游戏状态为结束游戏

            case FINISH_GAME:
                System.out.println("SO LONG FOR NOW.");  # 输出“现在就说再见。”
                gameState = GAME_STATE.GAME_OVER;  # 设置游戏状态为游戏结束

        }

    } while (gameState != GAME_STATE.GAME_OVER);  # 当游戏状态不是游戏结束时继续循环
}

private void starvedTooManyPeopleMessage(int starved) {  # 定义了一个私有方法，用于输出饥饿人数过多的消息
        System.out.println();
        // 打印空行
        System.out.println("YOU STARVED " + starved + " PEOPLE IN ONE YEAR!!!");
        // 打印一年内饿死的人数
        System.out.println("DUE TO THIS EXTREME MISMANAGEMENT YOU HAVE NOT ONLY");
        // 打印由于极端管理不善导致的结果
        System.out.println("BEEN IMPEACHED AND THROWN OUT OF OFFICE BUT YOU HAVE");
        // 打印被弹劾并被赶出办公室的结果
        System.out.println("ALSO BEEN DECLARED NATIONAL FINK!!!!");
        // 打印被宣布为国家的失败者的结果

    }

    private void heavyHandedMessage() {
        System.out.println("DUE TO THIS EXTREME MISMANAGEMENT YOU HAVE NOT ONLY");
        // 打印由于极端管理不善导致的结果
        System.out.println("BEEN IMPEACHED AND THROWN OUT OF OFFICE BUT YOU HAVE");
        // 打印被弹劾并被赶出办公室的结果
        System.out.println("ALSO BEEN DECLARED NATIONAL FINK!!!!");
        // 打印被宣布为国家的失败者的结果
    }

    private void couldHaveBeenBetterMessage() {
        System.out.println("YOUR PERFORMANCE COULD HAVE BEEN SOMEWHAT BETTER, BUT");
        // 打印你的表现本可以更好，但是
        System.out.println("REALLY WASN'T TOO BAD AT ALL. " + (int) (Math.random() * (population * .8)) + " PEOPLE");
        // 打印实际上并不是太糟糕，但是有一些人希望你被暗杀
        System.out.println("WOULD DEARLY LIKE TO SEE YOU ASSASSINATED BUT WE ALL HAVE OUR");
        // 打印渴望看到你被暗杀，但是我们都有自己的琐事
        System.out.println("TRIVIAL PROBLEMS.");
        // 打印琐碎的问题
    }
    // 打印出杰出表现的消息
    private void fantasticPerformanceMessage() {
        System.out.println("A FANTASTIC PERFORMANCE!!!  CHARLEMANGE, DISRAELI, AND");
        System.out.println("JEFFERSON COMBINED COULD NOT HAVE DONE BETTER!");
    }

    // 打印出人口不足的消息
    private void notEnoughPeopleMessage() {
        System.out.println("BUT YOU HAVE ONLY " + population + " PEOPLE TO TEND THE FIELDS!  NOW THEN,");
    }

    // 打印出粮食不足的消息
    private void notEnoughBushelsMessage() {
        System.out.println("HAMURABI:  THINK AGAIN.  YOU HAVE ONLY");
        System.out.println(bushels + " BUSHELS OF GRAIN.  NOW THEN,");
    }

    // 打印出土地不足的消息
    private void notEnoughLandMessage() {
        System.out.println("HAMURABI:  THINK AGAIN.  YOU OWN ONLY " + acres + " ACRES.  NOW THEN,");
    }
    // 打印介绍信息
    private void intro() {
        System.out.println(simulateTabs(32) + "HAMURABI");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("TRY YOUR HAND AT GOVERNING ANCIENT SUMERIA");
        System.out.println("FOR A TEN-YEAR TERM OF OFFICE.");
        System.out.println();
    }

    /*
     * 在屏幕上打印消息，然后接受键盘输入。
     * 将输入转换为整数
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }
    }

    # 在屏幕上打印一条消息，然后接受键盘输入。
    # @param text 要显示在屏幕上的消息。
    # @return 玩家输入的内容。
    def displayTextAndGetInput(text):
        print(text)
        return kbScanner.next()

    # 模拟旧的基本tab(xx)命令，该命令通过xx个空格缩进文本。
    # @param spaces 需要的空格数
    # @return 具有指定空格数的字符串
    def simulateTabs(spaces):
        // 创建一个长度为spaces的字符数组
        char[] spacesTemp = new char[spaces];
        // 用空格填充字符数组
        Arrays.fill(spacesTemp, ' ');
        // 将字符数组转换为字符串并返回
        return new String(spacesTemp);
    }
}
```