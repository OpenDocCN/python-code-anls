# `12_Bombs_Away\java\src\BombsAway.java`

```
import java.util.Scanner;
```
这行代码是导入Java中的Scanner类，用于从控制台读取用户的输入。

```python
/**
 * Game of Bombs Away
 *
 * Based on the Basic game of Bombs Away here
 * https://github.com/coding-horror/basic-computer-games/blob/main/12_Bombs_Away/bombsaway.bas
 *
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without adding new features.
 * Obvious bugs where found have been fixed, but the playability and overlook and feel
 * of the game have been faithfully reproduced.
 *
 * Modern Java coding conventions have been employed and JDK 11 used for maximum compatibility.
 *
 * Java port by https://github.com/journich
 *
 */
```
这段代码是一个多行注释，用于描述游戏的基本信息，包括游戏的名称、基于哪个版本的游戏、作者的想法和修改，以及Java版本的兼容性信息。

```python
public class BombsAway {
```
这行代码定义了一个名为BombsAway的公共类。

```python
    public static final int MAX_PILOT_MISSIONS = 160;
```
这行代码定义了一个名为MAX_PILOT_MISSIONS的常量，其值为160。
    # 定义最大伤亡人数常量
    public static final int MAX_CASUALTIES = 100;
    # 定义未命中目标常量1
    public static final int MISSED_TARGET_CONST_1 = 2;
    # 定义未命中目标常量2
    public static final int MISSED_TARGET_CONST_2 = 30;
    # 定义被击落的机会基数
    public static final int CHANCE_OF_BEING_SHOT_DOWN_BASE = 100;
    # 定义65%的概率
    public static final double SIXTY_FIVE_PERCENT = .65;

    # 定义游戏状态枚举
    private enum GAME_STATE {
        START,  # 游戏开始
        CHOOSE_SIDE,  # 选择阵营
        CHOOSE_PLANE,  # 选择飞机
        CHOOSE_TARGET,  # 选择目标
        CHOOSE_MISSIONS,  # 选择任务
        CHOOSE_ENEMY_DEFENCES,  # 选择敌方防御
        FLY_MISSION,  # 进行任务飞行
        DIRECT_HIT,  # 直接命中
        MISSED_TARGET,  # 未命中目标
        PROCESS_FLAK,  # 处理高射炮
        SHOT_DOWN,  # 被击落
        MADE_IT_THROUGH_FLAK,  # 成功通过高射炮
        PLAY_AGAIN,  # 再玩一次
        GAME_OVER
    }
```
这部分代码是一个枚举类型的定义，其中包含了四个枚举常量：ITALY、ALLIES、JAPAN和GERMANY。每个枚举常量都有一个与之对应的整数值。

```
    public enum SIDE {
        ITALY(1),
        ALLIES(2),
        JAPAN(3),
        GERMANY(4);
```
这部分代码定义了一个枚举类型SIDE，其中包含了四个枚举常量ITALY、ALLIES、JAPAN和GERMANY，并且为每个枚举常量指定了一个整数值。

```
        private final int value;

        SIDE(int value) {
            this.value = value;
        }
```
这部分代码定义了枚举类型SIDE的私有成员变量value，并且为枚举类型SIDE添加了一个构造函数，用于初始化枚举常量的整数值。

```
        public int getValue() {
            return value;
        }
```
这部分代码定义了一个公共方法getValue，用于获取枚举常量的整数值。
    }  # 结束枚举定义

    public enum TARGET {  # 定义枚举类型TARGET
        ALBANIA(1),  # 枚举值ALBANIA，对应值1
        GREECE(2),  # 枚举值GREECE，对应值2
        NORTH_AFRICA(3),  # 枚举值NORTH_AFRICA，对应值3
        RUSSIA(4),  # 枚举值RUSSIA，对应值4
        ENGLAND(5),  # 枚举值ENGLAND，对应值5
        FRANCE(6);  # 枚举值FRANCE，对应值6

        private final int value;  # 声明私有变量value，用于存储枚举值对应的值

        TARGET(int value) {  # 枚举类型TARGET的构造函数，用于初始化枚举值对应的值
            this.value = value;  # 将传入的值赋给私有变量value
        }

        public int getValue() {  # 定义公共方法getValue，用于获取枚举值对应的值
            return value;  # 返回枚举值对应的值
        }
    }
        // 定义敌人的防御方式枚举类型
        public enum ENEMY_DEFENCES {
            // 枚举值 GUNS 表示敌人使用枪支防御
            GUNS(1),
            // 枚举值 MISSILES 表示敌人使用导弹防御
            MISSILES(2),
            // 枚举值 BOTH 表示敌人同时使用枪支和导弹防御
            BOTH(3);

            // 私有变量，存储枚举值对应的整数值
            private final int value;

            // 枚举类型的构造函数，用于初始化枚举值对应的整数值
            ENEMY_DEFENCES(int value) {
                this.value = value;
            }

            // 获取枚举值对应的整数值
            public int getValue() {
                return value;
            }
        }

        // 定义飞机类型枚举
        public enum AIRCRAFT {
            // 枚举值 LIBERATOR 表示飞机类型为解放者
            LIBERATOR(1),
            // 枚举值 B29 表示飞机类型为B29
        B17(3),  // 创建一个枚举类型的实例B17，其值为3
        LANCASTER(4);  // 创建一个枚举类型的实例LANCASTER，其值为4

        private final int value;  // 创建一个私有的整型变量value

        AIRCRAFT(int value) {  // 定义一个枚举类型AIRCRAFT的构造函数，接受一个整型参数
            this.value = value;  // 将传入的参数赋值给实例的value变量
        }

        public int getValue() {  // 定义一个公有的方法getValue，用于获取实例的value值
            return value;  // 返回实例的value值
        }
    }

    // Used for keyboard input
    private final Scanner kbScanner;  // 创建一个私有的Scanner类型变量kbScanner，用于处理键盘输入

    // Current game state
    private GAME_STATE gameState;  // 创建一个私有的GAME_STATE类型变量gameState，用于表示当前游戏状态
    private SIDE side; // 声明一个私有的枚举类型变量side

    private int missions; // 声明一个私有的整型变量missions

    private int chanceToBeHit; // 声明一个私有的整型变量chanceToBeHit
    private int percentageHitRateOfGunners; // 声明一个私有的整型变量percentageHitRateOfGunners
    private boolean liar; // 声明一个私有的布尔类型变量liar

    public BombsAway() { // BombsAway类的构造函数

        gameState = GAME_STATE.START; // 初始化游戏状态为开始状态

        // Initialise kb scanner
        kbScanner = new Scanner(System.in); // 初始化kbScanner为一个从标准输入流System.in中读取数据的Scanner对象
    }

    /**
     * Main game loop
     *
     */
    public void play() {

        do {
            switch (gameState) {

                // Show an introduction the first time the game is played.
                case START:
                    intro();  # 调用intro()函数，显示游戏介绍
                    chanceToBeHit = 0;  # 初始化被击中的机会为0
                    percentageHitRateOfGunners = 0;  # 初始化炮手的命中率为0
                    liar = false;  # 初始化liar变量为false

                    gameState = GAME_STATE.CHOOSE_SIDE;  # 将游戏状态设置为选择阵营
                    break;

                case CHOOSE_SIDE:
                    side = getSide("WHAT SIDE -- ITALY(1), ALLIES(2), JAPAN(3), GERMANY(4) ? ");  # 调用getSide()函数，获取玩家选择的阵营
                    if (side == null) {  # 如果玩家选择的阵营为空
                        System.out.println("TRY AGAIN...");  # 输出提示信息
                    } else {
// 根据所选择的方阵不同，设置游戏状态为选择目标或选择飞机
switch (side) {
    case ITALY:
    case GERMANY:
        gameState = GAME_STATE.CHOOSE_TARGET;
        break;
    case ALLIES:
    case JAPAN:
        gameState = GAME_STATE.CHOOSE_PLANE;
        break;
}

// 选择目标状态下的代码
case CHOOSE_TARGET:
    String prompt;
    // 根据选择的方阵不同，设置不同的提示信息
    if (side == SIDE.ITALY) {
        prompt = "YOUR TARGET -- ALBANIA(1), GREECE(2), NORTH AFRICA(3) ? ";
    } else {
        // Germany
# 打印一条消息到控制台
System.out.println("A NAZI, EH?  OH WELL.  ARE YOU GOING FOR RUSSIA(1),");
# 设置一个提示消息
prompt = "ENGLAND(2), OR FRANCE(3) ? ";
# 获取目标对象
TARGET target = getTarget(prompt);
# 如果目标对象为空，则打印一条消息到控制台
if (target == null) {
    System.out.println("TRY AGAIN...");
# 否则，显示目标消息并设置游戏状态为选择任务
} else {
    displayTargetMessage(target);
    gameState = GAME_STATE.CHOOSE_MISSIONS;
}

# 选择任务状态
case CHOOSE_MISSIONS:
    # 从键盘获取飞行任务数量
    missions = getNumberFromKeyboard("HOW MANY MISSIONS HAVE YOU FLOWN? ");
    # 如果飞行任务数量小于25，则打印一条消息到控制台，并设置游戏状态为飞行任务
    if(missions <25) {
        System.out.println("FRESH OUT OF TRAINING, EH?");
        gameState = GAME_STATE.FLY_MISSION;
    # 如果飞行任务数量小于100，则打印一条消息到控制台，并设置游戏状态为飞行任务
    } else if(missions < 100) {
        System.out.println("THAT'S PUSHING THE ODDS!");
        gameState = GAME_STATE.FLY_MISSION;
                    } else if(missions >=160) {  # 如果任务数量大于等于160
                        System.out.println("MISSIONS, NOT MILES...");  # 打印“任务，不是英里…”
                        System.out.println("150 MISSIONS IS HIGH EVEN FOR OLD-TIMERS.");  # 打印“即使对老手来说，150个任务也很多。”
                        System.out.println("NOW THEN, ");  # 打印“现在，…”
                    } else {
                        // 如果任务数量在100-159之间，没有特定的消息，但仍然有效
                        gameState = GAME_STATE.FLY_MISSION;  # 将游戏状态设置为“飞行任务”
                    }
                    break;

                case CHOOSE_PLANE:  # 选择飞机
                    switch(side) {  # 根据阵营选择
                        case ALLIES:  # 如果是盟军
                            AIRCRAFT plane = getPlane("AIRCRAFT -- LIBERATOR(1), B-29(2), B-17(3), LANCASTER(4)? ");  # 获取飞机类型
                            if(plane == null) {  # 如果飞机为空
                                System.out.println("TRY AGAIN...");  # 打印“再试一次…”
                            } else {
                                switch(plane) {  # 根据飞机类型选择
                                    case LIBERATOR:  # 如果是解放者飞机
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 以二进制只读模式打开文件，读取文件内容并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名列表，读取文件数据并组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
// 如果随机数大于65%，则游戏状态为直接命中，否则为未命中目标
if(randomNumber(1) > SIXTY_FIVE_PERCENT) {
    gameState = GAME_STATE.DIRECT_HIT;
} else {
    // 这是一次未命中
    gameState = GAME_STATE.MISSED_TARGET;
}
// 如果未命中目标，则游戏状态为处理高射炮
} else {
    gameState = GAME_STATE.PROCESS_FLAK;
}
break;
```

```
case FLY_MISSION:
    // 计算任务结果，如果飞行任务数大于任务结果，则游戏状态为直接命中，否则为未命中目标
    double missionResult = (MAX_PILOT_MISSIONS * randomNumber(1));
    if(missions > missionResult) {
        gameState = GAME_STATE.DIRECT_HIT;
    } else {
        gameState = GAME_STATE.MISSED_TARGET;
    }
                    break;  // 结束当前的 case，跳出 switch 语句

                case DIRECT_HIT:  // 如果击中目标
                    System.out.println("DIRECT HIT!!!! " + (int) Math.round(randomNumber(MAX_CASUALTIES)) + " KILLED.");  // 打印直接命中目标并造成的伤亡人数
                    System.out.println("MISSION SUCCESSFUL.");  // 打印任务成功
                    gameState = GAME_STATE.PLAY_AGAIN;  // 将游戏状态设置为再玩一次
                    break;  // 结束当前的 case，跳出 switch 语句

                case MISSED_TARGET:  // 如果未命中目标
                    System.out.println("MISSED TARGET BY " + (int) Math.round(MISSED_TARGET_CONST_1 + MISSED_TARGET_CONST_2 * (randomNumber(1))) + " MILES!");  // 打印未命中目标的距离
                    System.out.println("NOW YOU'RE REALLY IN FOR IT !!");  // 打印现在你真的要麻烦了
                    System.out.println();  // 打印空行
                    gameState = GAME_STATE.CHOOSE_ENEMY_DEFENCES;  // 将游戏状态设置为选择敌方防御
                    break;  // 结束当前的 case，跳出 switch 语句

                case CHOOSE_ENEMY_DEFENCES:  // 选择敌方防御
                    percentageHitRateOfGunners = 0;  // 将炮手的命中率设置为0

                    ENEMY_DEFENCES enemyDefences = getEnemyDefences("DOES THE ENEMY HAVE GUNS(1), MISSILES(2), OR BOTH(3) ? ");  // 获取敌方防御的类型
                    if(enemyDefences == null) {  # 如果敌人的防御为空
                        System.out.println("TRY AGAIN...")  # 打印“再试一次…”
                    } else {
                        chanceToBeHit = 35  # 设置被击中的机会为35
                        switch(enemyDefences) {  # 根据敌人的防御类型进行不同的操作
                            case MISSILES:  # 如果是导弹
                                // MISSILES... An extra 35 but cannot specify percentage hit rate for gunners
                                break  # 结束switch语句
                            case GUNS:  # 如果是枪支
                                    // GUNS...  No extra 35 but can specify percentage hit rate for gunners
                                chanceToBeHit = 0  # 将被击中的机会设置为0
                                // fall through (no break) on purpose because remaining code is applicable
                                // for both GUNS and BOTH options.
                                // 故意不加break，因为后续代码适用于GUNS和BOTH两种选项
                            case BOTH:  # 如果是两者都有
                                // BOTH... An extra 35 and percentage hit rate for gunners can be specified.
                                percentageHitRateOfGunners = getNumberFromKeyboard("WHAT'S THE PERCENT HIT RATE OF ENEMY GUNNERS (10 TO 50)? ");  # 从键盘获取敌方枪手的命中率
                                if(percentageHitRateOfGunners < 10) {  # 如果枪手的命中率小于10
                                    System.out.println("YOU LIE, BUT YOU'LL PAY...")  # 打印“你撒谎了，但你会付出代价…”
                                    liar = true;  # 设置 liar 变量为 true
                                }
                                break;
                        }
                    }
                    // 如果玩家在输入炮手命中率时没有撒谎，继续游戏
                    // 否则击落玩家
                    if(!liar) {  # 如果 liar 变量为 false
                        gameState = GAME_STATE.PROCESS_FLAK;  # 设置游戏状态为 PROCESS_FLAK
                    } else {
                        gameState = GAME_STATE.SHOT_DOWN;  # 设置游戏状态为 SHOT_DOWN
                    }
                    break;

                // 确定玩家的飞机是否通过了防空火力网
                case PROCESS_FLAK:
                    double calc = (CHANCE_OF_BEING_SHOT_DOWN_BASE * randomNumber(1));  # 计算被击落的基础几率

                    if ((chanceToBeHit + percentageHitRateOfGunners) > calc) {  # 如果被击中的几率加上炮手命中率大于计算值
                        gameState = GAME_STATE.SHOT_DOWN;  # 设置游戏状态为 SHOT_DOWN
                } else {
                    gameState = GAME_STATE.GAME_OVER;
                }
                break;

            case SHOT_DOWN:
                // 打印飞机被击落的信息
                System.out.println("* * * * BOOM * * * *");
                System.out.println("YOU HAVE BEEN SHOT DOWN.....");
                System.out.println("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR");
                System.out.println("LAST TRIBUTE...");
                // 将游戏状态设置为再玩一次
                gameState = GAME_STATE.PLAY_AGAIN;
                break;

            case MADE_IT_THROUGH_FLAK:
                // 打印成功通过防空火力的信息
                System.out.println("YOU MADE IT THROUGH TREMENDOUS FLAK!!");
                // 将游戏状态设置为再玩一次
                gameState = GAME_STATE.PLAY_AGAIN;
                break;

            case PLAY_AGAIN:
                // 如果玩家输入了"Y"，则再次进行游戏，否则游戏结束
                if(yesEntered(displayTextAndGetInput("ANOTHER MISSION (Y OR N) ? "))) {
                        gameState = GAME_STATE.START;  # 设置游戏状态为开始
                    } else {
                        System.out.println("CHICKEN !!!");  # 打印"CHICKEN !!!"
                        gameState = GAME_STATE.GAME_OVER;  # 设置游戏状态为游戏结束
                    }
                    break;  # 跳出循环
            }
        } while (gameState != GAME_STATE.GAME_OVER) ;  # 当游戏状态不是游戏结束时继续循环

    /**
     * Display a (brief) intro
     */
    public void intro() {  # 定义一个方法用于显示简要介绍
        System.out.println("YOU ARE A PILOT IN A WORLD WAR II BOMBER.");  # 打印介绍内容
    }

    /**
     * Determine the side the player is going to play on.
     * @param message displayed before the kb input  # 确定玩家要玩的一方，并显示键盘输入前的消息
    /**
     * 根据玩家输入的消息确定玩家选择的 SIDE 枚举
     * @param message 在键盘输入之前显示的消息
     * @return 玩家选择的 SIDE 枚举
     */
    private SIDE getSide(String message) {
        // 从键盘输入获取值
        int valueEntered = getNumberFromKeyboard(message);
        // 遍历 SIDE 枚举，查找与输入值相匹配的枚举
        for(SIDE side : SIDE.values()) {
            if(side.getValue() == valueEntered) {
                return side;
            }
        }

        // 输入超出范围
        return null;
    }

    /**
     * 确定玩家选择的目标
     * @param message 在键盘输入之前显示的消息
     * @return 玩家选择的 TARGET 枚举
     */
    private TARGET getTarget(String message) {
        # 从键盘获取输入的值
        int valueEntered = getNumberFromKeyboard(message);

        # 遍历 AIRCRAFT 枚举类型的所有值
        for(TARGET target : TARGET.values()) {
            # 如果枚举值的值等于输入的值，则返回该枚举值
            if(target.getValue() == valueEntered) {
                return target;
            }
        }

        # 输入超出范围，返回空值
        // Input out of range
        return null;
    }

    /**
     * 确定玩家将要驾驶的飞机。
     * @param message 在键盘输入之前显示的消息
     * @return 玩家选择的 AIRCRAFT 枚举类型
     */
    private AIRCRAFT getPlane(String message) {
        # 从键盘获取输入的值
        int valueEntered = getNumberFromKeyboard(message);
        for(AIRCRAFT plane : AIRCRAFT.values()) {  # 遍历 AIRCRAFT 枚举类型的所有值
            if(plane.getValue() == valueEntered) {  # 如果枚举类型的值等于输入的值
                return plane;  # 返回匹配的枚举类型的值
            }
        }

        // Input out of range  # 输入超出范围
        return null;  # 返回空值

    }

    /**
     * Select the type of enemy defences.
     *
     * @param message displayed before kb input  # 在键盘输入之前显示的消息
     * @return the ENEMY_DEFENCES enum as selected by player  # 返回玩家选择的 ENEMY_DEFENCES 枚举类型
     */
    private ENEMY_DEFENCES getEnemyDefences(String message) {  # 选择敌人防御类型
        int valueEntered = getNumberFromKeyboard(message);  # 从键盘输入获取数字
        for (ENEMY_DEFENCES enemyDefences : ENEMY_DEFENCES.values()) {  # 遍历 ENEMY_DEFENCES 枚举类型的所有值
            if(enemyDefences.getValue() == valueEntered) {  # 如果敌人的防御值等于输入的值
                return enemyDefences;  # 返回敌人的防御值
            }
        }

        // Input out of range  # 输入超出范围
        return null;  # 返回空值
    }

    // output a specific message based on the target selected  # 根据选择的目标输出特定的消息
    private void displayTargetMessage(TARGET target) {  # 根据目标显示消息

        switch (target) {  # 根据目标进行切换

            case ALBANIA:  # 如果是阿尔巴尼亚
                System.out.println("SHOULD BE EASY -- YOU'RE FLYING A NAZI-MADE PLANE.");  # 应该很容易--你在驾驶纳粹制造的飞机
                break;  # 结束
            case GREECE:  # 如果是希腊
                System.out.println("BE CAREFUL!!!");  # 小心！！！
                break;  # 结束
            case NORTH_AFRICA:  // 如果目的地是北非
                System.out.println("YOU'RE GOING FOR THE OIL, EH?");  // 打印出“你要去搞石油，嗯？”
                break;  // 结束switch语句
            case RUSSIA:  // 如果目的地是俄罗斯
                System.out.println("YOU'RE NEARING STALINGRAD.");  // 打印出“你快到斯大林格勒了。”
                break;  // 结束switch语句
            case ENGLAND:  // 如果目的地是英格兰
                System.out.println("NEARING LONDON.  BE CAREFUL, THEY'VE GOT RADAR.");  // 打印出“快到伦敦了。小心，他们有雷达。”
                break;  // 结束switch语句
            case FRANCE:  // 如果目的地是法国
                System.out.println("NEARING VERSAILLES.  DUCK SOUP.  THEY'RE NEARLY DEFENSELESS.");  // 打印出“快到凡尔赛了。小菜一碟。他们几乎没有防御。”
                break;  // 结束switch语句
        }
    }

    /**
     * Accepts a string from the keyboard, and converts to an int
     *
     * @param message displayed text on screen before keyboard input
     *
    /**
     * 从键盘获取玩家输入的数字
     * @param message 提示信息
     * @return 玩家输入的数字
     */
    private int getNumberFromKeyboard(String message) {
        // 显示提示信息并获取玩家输入
        String answer = displayTextAndGetInput(message);
        // 将输入的字符串转换为整数并返回
        return Integer.parseInt(answer);
    }

    /**
     * 检查玩家是否输入了Y或YES
     * @param text 从键盘获取的玩家输入
     * @return 如果输入了Y或YES则返回true，否则返回false
     */
    private boolean yesEntered(String text) {
        // 检查输入的字符串是否等于Y或YES
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * 检查一个字符串是否等于多个可能的值中的一个
     */
# Useful to check for Y or YES for example
# Comparison is case-insensitive.
# @param text source string
# @param values a range of values to compare against the source string
# @return true if a comparison was found in one of the variable number of strings passed
def stringIsAnyValue(text, *values):
    # Cycle through the variable number of values and test each
    for val in values:
        if text.lower() == val.lower():
            return True
    # no matches
    return False
    /*
     * 在屏幕上打印一条消息，然后接受键盘输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * 生成随机数
     * 用作计算机玩家的单个数字
     *
     * @return 随机数
     */
    private double randomNumber(int range) {
        return (Math.random()
                * (range));
    }
```

这部分代码是一个缩进错误，应该删除。
```