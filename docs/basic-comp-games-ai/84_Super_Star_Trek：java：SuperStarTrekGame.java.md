# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\java\SuperStarTrekGame.java`

```
import java.util.stream.IntStream;  // 导入 Java 中的 IntStream 类

/**
 * SUPER STARTREK - MAY 16,1978
 * ****        **** STAR TREK ****        ****
 * **** SIMULATION OF A MISSION OF THE STARSHIP ENTERPRISE,
 * **** AS SEEN ON THE STAR TREK TV SHOW.
 * **** ORIGINAL PROGRAM BY MIKE MAYFIELD, MODIFIED VERSION
 * **** PUBLISHED IN DEC'S "101 BASIC GAMES", BY DAVE AHL.
 * **** MODIFICATIONS TO THE LATTER (PLUS DEBUGGING) BY BOB
 * *** LEEDOM - APRIL & DECEMBER 1974,
 * *** WITH A LITTLE HELP FROM HIS FRIENDS . . .
 *
 * Ported to Java in Jan-Mar 2022 by
 * Taciano Dreckmann Perez (taciano.perez@gmail.com)
 */
public class SuperStarTrekGame implements GameCallback {  // 定义一个名为 SuperStarTrekGame 的类，实现 GameCallback 接口

    // commands
    static final int COMMAND_NAV = 1;  // 定义一个名为 COMMAND_NAV 的静态常量，值为 1
    static final int COMMAND_SRS = 2; // 定义常量，表示舰船向右转
    static final int COMMAND_LRS = 3; // 定义常量，表示舰船向左转
    static final int COMMAND_PHA = 4; // 定义常量，表示舰船发射光子鱼雷
    static final int COMMAND_TOR = 5; // 定义常量，表示舰船进行跃迁
    static final int COMMAND_SHE = 6; // 定义常量，表示舰船进行护盾充能
    static final int COMMAND_DAM = 7; // 定义常量，表示舰船进行损伤修复
    static final int COMMAND_COM = 8; // 定义常量，表示舰船进行通讯
    static final int COMMAND_XXX = 9; // 定义常量，表示舰船进行未知操作

    // computer commands
    static final int COMPUTER_COMMAND_CUMULATIVE_GALACTIC_RECORD = 1; // 定义常量，表示计算机执行累积银河记录命令
    static final int COMPUTER_COMMAND_STATUS_REPORT = 2; // 定义常量，表示计算机执行状态报告命令
    static final int COMPUTER_COMMAND_PHOTON_TORPEDO_DATA = 3; // 定义常量，表示计算机执行光子鱼雷数据命令
    static final int COMPUTER_COMMAND_STARBASE_NAV_DATA = 4; // 定义常量，表示计算机执行星舰基地导航数据命令
    static final int COMPUTER_COMMAND_DIR_DIST_CALC = 5; // 定义常量，表示计算机执行方向和距离计算命令
    static final int COMPUTER_COMMAND_GALAXY_MAP = 6; // 定义常量，表示计算机执行银河地图命令

    // other constants
    static final String COMMANDS = "NAVSRSLRSPHATORSHEDAMCOMXXX"; // 定义常量，表示舰船可执行的命令字符串
    // 创建一个新的GalaxyMap对象来表示游戏的星系地图
    final GalaxyMap galaxyMap = new GalaxyMap();
    // 生成一个随机的星期时间，范围在20到40之间
    double stardate = Util.toInt(Util.random() * 20 + 20);
    // 生成一个随机的任务持续时间，范围在25到35之间，并且至少为星系中克林贡人的数量加1
    int missionDuration = Math.max((25 + Util.toInt(Util.random() * 10)), galaxyMap.getKlingonsInGalaxy()+1);    // T9 (mission duration in stardates)
    // 用于标记游戏是否需要重新开始
    boolean restart = false;

    // 保存初始的星期时间
    final double initialStardate = stardate;

    public static void main(String[] args) {
        // 创建一个新的SuperStarTrekGame对象
        final SuperStarTrekGame game = new SuperStarTrekGame();
        // 打印游戏横幅
        printBanner();
        // 游戏主循环
        while (true) {
            // 接收玩家指令
            game.orders();
            // 进入新的象限
            game.enterNewQuadrant();
            // 重置restart标记
            game.restart = false;
            // 进入命令循环
            game.commandLoop();
        }
    }
    // 定义静态方法，用于打印横幅
    static void printBanner() {
        // 使用 IntStream 创建范围为 1 到 10 的整数流，对每个整数执行指定操作
        IntStream.range(1, 10).forEach(i -> {
            // 调用 Util 类的 println 方法打印空行
            Util.println("");
        });
        // 调用 Util 类的 println 方法打印多行字符串，显示 USS Enterprise 的横幅
        Util.println(
                """
                                                            ,------*------,
                                            ,-------------   '---  ------'
                                             '-------- --'      / /
                                                 ,---' '-------/ /--,
                                                  '----------------'

                                            THE USS ENTERPRISE --- NCC-1701"

                        """
        );
    }

    // 定义方法，用于打印订单信息
    void orders() {
        // 调用 Util 类的 println 方法打印订单信息
        Util.println("YOUR ORDERS ARE AS FOLLOWS:\n" +
                "     DESTROY THE " + galaxyMap.getKlingonsInGalaxy() + " KLINGON WARSHIP" + ((galaxyMap.getKlingonsInGalaxy() == 1) ? "" : "S") + " WHICH HAVE INVADED\n" + 
                "   THE GALAXY BEFORE THEY CAN ATTACK FEDERATION HEADQUARTERS\n" + 
                "   ON STARDATE " + initialStardate + missionDuration + "  THIS GIVES YOU " + missionDuration + " DAYS.  THERE " + ((galaxyMap.getBasesInGalaxy() == 1) ? "IS" : "ARE") + "\n" + 
                "  " + galaxyMap.getBasesInGalaxy() + " STARBASE" + ((galaxyMap.getBasesInGalaxy() == 1) ? "" : "S") + " IN THE GALAXY FOR RESUPPLYING YOUR SHIP");
    }
```
这段代码是一个字符串拼接的过程，根据galaxyMap中的信息和变量的值来构建一段文本描述。

```
    public void enterNewQuadrant() {
        galaxyMap.newQuadrant(stardate, initialStardate);
        shortRangeSensorScan();
    }
```
这段代码定义了一个公共方法enterNewQuadrant()，在该方法中调用了galaxyMap对象的newQuadrant()方法和shortRangeSensorScan()方法。

```
    void commandLoop() {
        while (!this.restart) {
            checkShipEnergy();
            String cmdStr = "";
            while ("".equals(cmdStr)) cmdStr = Util.inputStr("COMMAND");
            boolean foundCommand = false;
            for (int i = 1; i <= 9; i++) {
                if (Util.leftStr(cmdStr, 3).equals(Util.midStr(COMMANDS, 3 * i - 2, 3))) {
                    switch (i) {
```
这段代码定义了一个名为commandLoop()的方法，其中包含了一个while循环，不断检查this.restart的值。在循环中调用了checkShipEnergy()方法，并且通过Util.inputStr("COMMAND")获取用户输入的命令。然后通过for循环和switch语句来处理用户输入的命令。
# 如果接收到的命令是导航命令，则执行导航函数
case COMMAND_NAV:
    navigation();
    foundCommand = true;  # 标记找到了命令
    break;  # 结束当前的 case 分支
# 如果接收到的命令是短程传感器扫描命令，则执行短程传感器扫描函数
case COMMAND_SRS:
    shortRangeSensorScan();
    foundCommand = true;  # 标记找到了命令
    break;  # 结束当前的 case 分支
# 如果接收到的命令是长程传感器扫描命令，则执行长程传感器扫描函数
case COMMAND_LRS:
    longRangeSensorScan();
    foundCommand = true;  # 标记找到了命令
    break;  # 结束当前的 case 分支
# 如果接收到的命令是火炮命令，则执行火炮函数
case COMMAND_PHA:
    firePhasers();
    foundCommand = true;  # 标记找到了命令
    break;  # 结束当前的 case 分支
# 如果接收到的命令是光子鱼雷命令，则执行光子鱼雷函数
case COMMAND_TOR:
    firePhotonTorpedo();
    foundCommand = true;  # 标记找到了命令
    break;  # 结束当前的 case 分支
# 如果接收到的命令是COMMAND_SHE，则执行shieldControl()函数
case COMMAND_SHE:
    shieldControl();
    foundCommand = true;
    break;
# 如果接收到的命令是COMMAND_DAM，则执行galaxyMap.getEnterprise().damageControl(this)函数
case COMMAND_DAM:
    galaxyMap.getEnterprise().damageControl(this);
    foundCommand = true;
    break;
# 如果接收到的命令是COMMAND_COM，则执行libraryComputer()函数
case COMMAND_COM:
    libraryComputer();
    foundCommand = true;
    break;
# 如果接收到的命令是COMMAND_XXX，则执行endGameFail(false)函数
case COMMAND_XXX:
    endGameFail(false);
    foundCommand = true;
    break;
# 如果接收到的命令不是以上任何一种，则执行printCommandOptions()函数
default:
    printCommandOptions();
    foundCommand = true;
    }
            }  # 结束内部循环
        }  # 结束外部循环
        if (!foundCommand) printCommandOptions();  # 如果没有找到命令，则打印命令选项
    }  # 结束方法

    void checkShipEnergy() {  # 检查飞船能量的方法
        final Enterprise enterprise = galaxyMap.getEnterprise();  # 获取星际地图上的企业飞船
        if (enterprise.getTotalEnergy() < 10 && (enterprise.getEnergy() <= 10 || enterprise.getDeviceStatus()[Enterprise.DEVICE_SHIELD_CONTROL] != 0)) {  # 如果总能量小于10并且（能量小于等于10或者护盾控制设备状态不为0）
            Util.println("\n** FATAL ERROR **   YOU'VE JUST STRANDED YOUR SHIP IN ");  # 打印致命错误信息
            Util.println("SPACE");  # 打印空间信息
            Util.println("YOU HAVE INSUFFICIENT MANEUVERING ENERGY,");  # 打印能量不足信息
            Util.println(" AND SHIELD CONTROL");  # 打印护盾控制信息
            Util.println("IS PRESENTLY INCAPABLE OF CROSS");  # 打印目前无法穿越信息
            Util.println("-CIRCUITING TO ENGINE ROOM!!");  # 打印到引擎室的信息
            endGameFail(false);  # 游戏失败
        }
    }  # 结束方法

    void printCommandOptions() {  # 打印命令选项的方法
        // 打印用户可输入的命令选项
        Util.println("ENTER ONE OF THE FOLLOWING:");
        Util.println("  NAV  (TO SET COURSE)");
        Util.println("  SRS  (FOR SHORT RANGE SENSOR SCAN)");
        Util.println("  LRS  (FOR LONG RANGE SENSOR SCAN)");
        Util.println("  PHA  (TO FIRE PHASERS)");
        Util.println("  TOR  (TO FIRE PHOTON TORPEDOES)");
        Util.println("  SHE  (TO RAISE OR LOWER SHIELDS)");
        Util.println("  DAM  (FOR DAMAGE CONTROL REPORTS)");
        Util.println("  COM  (TO CALL ON LIBRARY-COMPUTER)");
        Util.println("  XXX  (TO RESIGN YOUR COMMAND)\n");
    }

    void navigation() {
        // 获取用户输入的航向，并转换为浮点数
        float course = Util.toInt(Util.inputFloat("COURSE (0-9)"));
        // 如果用户输入9，则将其转换为1
        if (course == 9) course = 1;
        // 如果航向小于1或者大于等于9，则打印错误信息并返回
        if (course < 1 || course >= 9) {
            Util.println("   LT. SULU REPORTS, 'INCORRECT COURSE DATA, SIR!'");
            return;
        }
        // 获取星际地图上的企业号飞船对象
        final Enterprise enterprise = galaxyMap.getEnterprise();
        // 从企业对象中获取设备状态数组
        final double[] deviceStatus = enterprise.getDeviceStatus();
        // 打印"WARP FACTOR (0-8)"的提示信息
        Util.println("WARP FACTOR (0-" + ((deviceStatus[Enterprise.DEVICE_WARP_ENGINES] < 0) ? "0.2" : "8") + ")");
        // 获取用户输入的WARP值
        float warp = Util.inputFloat("");
        // 如果WARP引擎受损且用户输入的WARP值大于0.2，则打印提示信息并返回
        if (deviceStatus[Enterprise.DEVICE_WARP_ENGINES] < 0 && warp > .2) {
            Util.println("WARP ENGINES ARE DAMAGED.  MAXIMUM SPEED = WARP 0.2");
            return;
        }
        // 如果用户输入的WARP值为0，则直接返回
        if (warp == 0) return;
        // 如果用户输入的WARP值在0到8之间
        if (warp > 0 && warp <= 8) {
            // 计算需要消耗的能量
            int n = Util.toInt(warp * 8);
            // 如果企业的能量减去消耗的能量大于等于0
            if (enterprise.getEnergy() - n >= 0) {
                // 星际地图中克林贡人移动并开火
                galaxyMap.klingonsMoveAndFire(this);
                // 修复受损的设备
                repairDamagedDevices(course, warp, n);
                // 星际地图中移动企业
                galaxyMap.moveEnterprise(course, warp, n, stardate, initialStardate, missionDuration, this);
            } else {
                // 打印能量不足的提示信息
                Util.println("ENGINEERING REPORTS   'INSUFFICIENT ENERGY AVAILABLE");
                Util.println("                       FOR MANEUVERING AT WARP " + warp + "!'");
                // 如果企业的护盾值小于需要消耗的能量与企业能量之差，或者护盾控制设备受损，则直接返回
                if (enterprise.getShields() < n - enterprise.getEnergy() || deviceStatus[Enterprise.DEVICE_SHIELD_CONTROL] < 0) return;
                // 打印护盾控制室的提示信息
                Util.println("DEFLECTOR CONTROL ROOM ACKNOWLEDGES " + enterprise.getShields() + " UNITS OF ENERGY");
                Util.println("                         PRESENTLY DEPLOYED TO SHIELDS.");
void repairDamagedDevices(final float course, final float warp, final int N) {
        final Enterprise enterprise = galaxyMap.getEnterprise();  // 获取星际地图上的企业号实例
        // 修复受损设备并打印损坏报告
        enterprise.repairDamagedDevices(warp);  // 调用企业号实例的修复受损设备方法
        if (Util.random() > .2) return;  // 80% 的概率没有损坏也没有修复，直接返回
        int randomDevice = Util.fnr();    // 随机选择一个设备
        final double[] deviceStatus = enterprise.getDeviceStatus();  // 获取企业号实例的设备状态数组
        if (Util.random() >= .6) {   // 40% 的概率随机设备被修复
            enterprise.setDeviceStatus(randomDevice, deviceStatus[randomDevice] + Util.random() * 3 + 1);  // 更新随机设备的状态为修复状态
            Util.println("DAMAGE CONTROL REPORT:  " + Enterprise.printDeviceName(randomDevice) + " STATE OF REPAIR IMPROVED\n");  // 打印修复报告
        } else {            // 60% 的概率随机设备被损坏
            enterprise.setDeviceStatus(randomDevice, deviceStatus[randomDevice] - (Util.random() * 5 + 1));  // 更新随机设备的状态为损坏状态
            Util.println("DAMAGE CONTROL REPORT:  " + Enterprise.printDeviceName(randomDevice) + " DAMAGED");  // 打印损坏报告
    }
```
这是一个函数结束的标志，表示上一个函数的结束。

```
    void longRangeSensorScan() {
        // LONG RANGE SENSOR SCAN CODE
        galaxyMap.longRangeSensorScan();
    }
```
这是一个名为longRangeSensorScan的函数，用于执行长程传感器扫描。函数内部调用了galaxyMap对象的longRangeSensorScan方法。

```
    void firePhasers() {
        galaxyMap.firePhasers(this);
    }
```
这是一个名为firePhasers的函数，用于发射相位炮。函数内部调用了galaxyMap对象的firePhasers方法，并传入当前对象的引用。

```
    void firePhotonTorpedo() {
        galaxyMap.firePhotonTorpedo(stardate, initialStardate, missionDuration, this);
    }
```
这是一个名为firePhotonTorpedo的函数，用于发射光子鱼雷。函数内部调用了galaxyMap对象的firePhotonTorpedo方法，并传入了stardate、initialStardate、missionDuration和当前对象的引用。

```
    void shieldControl() {
        galaxyMap.getEnterprise().shieldControl();
    }
```
这是一个名为shieldControl的函数，用于控制护盾。函数内部调用了galaxyMap对象的getEnterprise方法，然后再调用返回的对象的shieldControl方法。
    // 定义 shortRangeSensorScan 方法，用于执行短程传感器扫描
    void shortRangeSensorScan() {
        // 调用 galaxyMap 对象的 shortRangeSensorScan 方法，传入 stardate 参数
        galaxyMap.shortRangeSensorScan(stardate);
    }

    // 定义 libraryComputer 方法，用于操作图书馆电脑
    void libraryComputer() {
        // 检查企业舰的图书馆电脑设备状态，如果小于 0 则输出“COMPUTER DISABLED”并返回
        if (galaxyMap.getEnterprise().getDeviceStatus()[Enterprise.DEVICE_LIBRARY_COMPUTER] < 0) {
            Util.println("COMPUTER DISABLED");
            return;
        }
        // 进入循环，等待用户输入指令
        while (true) {
            // 获取用户输入的指令
            final float commandInput = Util.inputFloat("COMPUTER ACTIVE AND AWAITING COMMAND");
            // 如果用户输入小于 0，则返回
            if (commandInput < 0) return;
            Util.println("");
            // 将用户输入的指令转换为整数并加 1
            int command = Util.toInt(commandInput) + 1;
            // 如果指令在指定范围内
            if (command >= COMPUTER_COMMAND_CUMULATIVE_GALACTIC_RECORD && command <= COMPUTER_COMMAND_GALAXY_MAP) {
                // 根据指令执行相应的操作
                switch (command) {
                    case COMPUTER_COMMAND_CUMULATIVE_GALACTIC_RECORD:
                        // 执行 galaxyMap 对象的 cumulativeGalacticRecord 方法，并传入 true 参数
                        galaxyMap.cumulativeGalacticRecord(true);
                    return; // 返回空值
                    case COMPUTER_COMMAND_STATUS_REPORT: // 如果命令是状态报告
                        statusReport(); // 调用状态报告函数
                        return; // 返回空值
                    case COMPUTER_COMMAND_PHOTON_TORPEDO_DATA: // 如果命令是光子鱼雷数据
                        galaxyMap.photonTorpedoData(); // 调用星系地图的光子鱼雷数据函数
                        return; // 返回空值
                    case COMPUTER_COMMAND_STARBASE_NAV_DATA: // 如果命令是星舰基地导航数据
                        galaxyMap.starbaseNavData(); // 调用星系地图的星舰基地导航数据函数
                        return; // 返回空值
                    case COMPUTER_COMMAND_DIR_DIST_CALC: // 如果命令是方向距离计算
                        galaxyMap.directionDistanceCalculator(); // 调用星系地图的方向距离计算函数
                        return; // 返回空值
                    case COMPUTER_COMMAND_GALAXY_MAP: // 如果命令是星系地图
                        galaxyMap.cumulativeGalacticRecord(false); // 调用星系地图的累积银河记录函数，传入false参数
                        return; // 返回空值
                }
            } else {
                // invalid command
                Util.println("FUNCTIONS AVAILABLE FROM LIBRARY-COMPUTER:"); // 打印输出信息，表示无效命令
                Util.println("   0 = CUMULATIVE GALACTIC RECORD"); // 打印信息
                Util.println("   1 = STATUS REPORT"); // 打印信息
                Util.println("   2 = PHOTON TORPEDO DATA"); // 打印信息
                Util.println("   3 = STARBASE NAV DATA"); // 打印信息
                Util.println("   4 = DIRECTION/DISTANCE CALCULATOR"); // 打印信息
                Util.println("   5 = GALAXY 'REGION NAME' MAP"); // 打印信息
                Util.println(""); // 打印空行
            }
        }
    }

    void statusReport() {
        Util.println("   STATUS REPORT:"); // 打印信息
        Util.println("KLINGON" + ((galaxyMap.getKlingonsInGalaxy() > 1)? "S" : "")  + " LEFT: " + galaxyMap.getKlingonsInGalaxy()); // 打印信息
        Util.println("MISSION MUST BE COMPLETED IN " + .1 * Util.toInt((initialStardate + missionDuration - stardate) * 10) + " STARDATES"); // 打印信息
        if (galaxyMap.getBasesInGalaxy() >= 1) { // 如果星际地图中的星舰基地数量大于等于1
            Util.println("THE FEDERATION IS MAINTAINING " + galaxyMap.getBasesInGalaxy() + " STARBASE" + ((galaxyMap.getBasesInGalaxy() > 1)? "S" : "") + " IN THE GALAXY"); // 打印信息
        } else { // 否则
            Util.println("YOUR STUPIDITY HAS LEFT YOU ON YOUR OWN IN"); // 打印信息
            Util.println("  THE GALAXY -- YOU HAVE NO STARBASES LEFT!"); // 打印信息
        }
        galaxyMap.getEnterprise().damageControl(this);  // 调用galaxyMap对象的getEnterprise方法获取星际飞船对象，然后调用其damageControl方法进行损坏控制
    }

    public void incrementStardate(double increment) {
        this.stardate += increment;  // 增加星际日期的值
    }

    public void endGameFail(final boolean enterpriseDestroyed) {    // 6220
        if (enterpriseDestroyed) {  // 如果星际飞船被摧毁
            Util.println("\nTHE ENTERPRISE HAS BEEN DESTROYED.  THEN FEDERATION ");  // 打印星际飞船被摧毁的消息
            Util.println("WILL BE CONQUERED");
        }
        Util.println("\nIT IS STARDATE " + stardate);  // 打印当前星际日期
        Util.println("THERE WERE " + galaxyMap.getKlingonsInGalaxy() + " KLINGON BATTLE CRUISERS LEFT AT");  // 打印剩余克林贡战舰的数量
        repeatGame();  // 重新开始游戏
    }

    public void endGameSuccess() {
        // 打印恭喜消息
        Util.println("CONGRATULATION, CAPTAIN!  THE LAST KLINGON BATTLE CRUISER");
        // 打印销毁克林贡战舰的消息
        Util.println("MENACING THE FEDERATION HAS BEEN DESTROYED.\n");
        // 打印效率评级
        Util.println("YOUR EFFICIENCY RATING IS " + (Math.sqrt(1000 * (galaxyMap.getRemainingKlingons() / (stardate - initialStardate))));
        // 重复游戏
        repeatGame();
    }

    void repeatGame() {
        // 打印换行
        Util.println("\n");
        // 如果星际地图中还有联邦基地
        if (galaxyMap.getBasesInGalaxy() != 0) {
            // 打印联邦需要新的星舰指挥官的消息
            Util.println("THE FEDERATION IS IN NEED OF A NEW STARSHIP COMMANDER");
            // 打印寻找志愿者的消息
            Util.println("FOR A SIMILAR MISSION -- IF THERE IS A VOLUNTEER,");
            // 获取用户输入
            final String reply = Util.inputStr("LET HIM STEP FORWARD AND ENTER 'AYE'");
            // 如果用户输入为"AYE"
            if ("AYE".equals(reply)) {
                // 设置重新开始游戏的标志为true
                this.restart = true;
            } else {
                // 退出程序
                System.exit(0);
            }
        }
    }
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```