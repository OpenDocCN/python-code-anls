# `basic-computer-games\84_Super_Star_Trek\java\SuperStarTrekGame.java`

```py
import java.util.stream.IntStream;

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
public class SuperStarTrekGame implements GameCallback {

    // commands
    static final int COMMAND_NAV = 1;  // 定义导航命令的常量
    static final int COMMAND_SRS = 2;  // 定义短程传感器扫描命令的常量
    static final int COMMAND_LRS = 3;  // 定义长程传感器扫描命令的常量
    static final int COMMAND_PHA = 4;  // 定义光子鱼雷命令的常量
    static final int COMMAND_TOR = 5;  // 定义跃迁命令的常量
    static final int COMMAND_SHE = 6;  // 定义护盾命令的常量
    static final int COMMAND_DAM = 7;  // 定义损坏报告命令的常量
    static final int COMMAND_COM = 8;  // 定义通讯命令的常量
    static final int COMMAND_XXX = 9;  // 定义未知命令的常量

    // computer commands
    static final int COMPUTER_COMMAND_CUMULATIVE_GALACTIC_RECORD = 1;  // 定义计算机命令的常量
    static final int COMPUTER_COMMAND_STATUS_REPORT = 2;  // 定义计算机命令的常量
    static final int COMPUTER_COMMAND_PHOTON_TORPEDO_DATA = 3;  // 定义计算机命令的常量
    static final int COMPUTER_COMMAND_STARBASE_NAV_DATA = 4;  // 定义计算机命令的常量
    static final int COMPUTER_COMMAND_DIR_DIST_CALC = 5;  // 定义计算机命令的常量
    static final int COMPUTER_COMMAND_GALAXY_MAP = 6;  // 定义计算机命令的常量

    // other constants
    static final String COMMANDS = "NAVSRSLRSPHATORSHEDAMCOMXXX";  // 定义包含所有命令的字符串常量

    // game state
    final GalaxyMap galaxyMap = new GalaxyMap();  // 创建星系地图对象
    double stardate = Util.toInt(Util.random() * 20 + 20);  // 初始化星日期
    int missionDuration = Math.max((25 + Util.toInt(Util.random() * 10)), galaxyMap.getKlingonsInGalaxy()+1);    // T9 (mission duration in stardates)  // 初始化任务持续时间
    boolean restart = false;  // 初始化重启标志

    // initial values
    final double initialStardate = stardate;  // 初始化星日期的初始值
    // 主程序入口
    public static void main(String[] args) {
        // 创建超级星际迷航游戏对象
        final SuperStarTrekGame game = new SuperStarTrekGame();
        // 打印游戏横幅
        printBanner();
        // 游戏循环
        while (true) {
            // 接收玩家指令
            game.orders();
            // 进入新的象限
            game.enterNewQuadrant();
            // 重置游戏状态
            game.restart = false;
            // 进入指令循环
            game.commandLoop();
        }
    }

    // 打印游戏横幅
    static void printBanner() {
        // 打印空行
        IntStream.range(1, 10).forEach(i -> {
            Util.println("");
        });
        // 打印游戏横幅
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

    // 打印玩家指令
    void orders() {
        // 打印玩家指令
        Util.println("YOUR ORDERS ARE AS FOLLOWS:\n" +
                "     DESTROY THE " + galaxyMap.getKlingonsInGalaxy() + " KLINGON WARSHIP" + ((galaxyMap.getKlingonsInGalaxy() == 1) ? "" : "S") + " WHICH HAVE INVADED\n" +
                "   THE GALAXY BEFORE THEY CAN ATTACK FEDERATION HEADQUARTERS\n" +
                "   ON STARDATE " + initialStardate + missionDuration + "  THIS GIVES YOU " + missionDuration + " DAYS.  THERE " + ((galaxyMap.getBasesInGalaxy() == 1) ? "IS" : "ARE") + "\n" +
                "  " + galaxyMap.getBasesInGalaxy() + " STARBASE" + ((galaxyMap.getBasesInGalaxy() == 1) ? "" : "S") + " IN THE GALAXY FOR RESUPPLYING YOUR SHIP");
    }

    // 进入新的象限
    public void enterNewQuadrant() {
        // 创建新的象限
        galaxyMap.newQuadrant(stardate, initialStardate);
        // 进行短程传感器扫描
        shortRangeSensorScan();
    }
    // 检查飞船能量，如果能量不足并且护盾控制设备状态不为0，则输出错误信息并结束游戏
    void checkShipEnergy() {
        // 获取星图上的企业飞船对象
        final Enterprise enterprise = galaxyMap.getEnterprise();
        // 如果总能量小于10并且（能量小于等于10或者护盾控制设备状态不为0）
        if (enterprise.getTotalEnergy() < 10 && (enterprise.getEnergy() <= 10 || enterprise.getDeviceStatus()[Enterprise.DEVICE_SHIELD_CONTROL] != 0)) {
            // 输出错误信息
            Util.println("\n** FATAL ERROR **   YOU'VE JUST STRANDED YOUR SHIP IN ");
            Util.println("SPACE");
            Util.println("YOU HAVE INSUFFICIENT MANEUVERING ENERGY,");
            Util.println(" AND SHIELD CONTROL");
            Util.println("IS PRESENTLY INCAPABLE OF CROSS");
            Util.println("-CIRCUITING TO ENGINE ROOM!!");
            // 结束游戏并失败
            endGameFail(false);
        }
    }

    // 输出可用的命令选项
    void printCommandOptions() {
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
    # 定义一个名为 navigation 的函数
    void navigation() {
        # 获取用户输入的航向角度，并转换为整数
        float course = Util.toInt(Util.inputFloat("COURSE (0-9)"));
        # 如果航向角度为9，则将其设为1
        if (course == 9) course = 1;
        # 如果航向角度小于1或大于等于9，则输出错误信息并返回
        if (course < 1 || course >= 9) {
            Util.println("   LT. SULU REPORTS, 'INCORRECT COURSE DATA, SIR!'");
            return;
        }
        # 获取星际企业对象
        final Enterprise enterprise = galaxyMap.getEnterprise();
        # 获取星际企业设备状态
        final double[] deviceStatus = enterprise.getDeviceStatus();
        # 输出最大的飞行速度范围
        Util.println("WARP FACTOR (0-" + ((deviceStatus[Enterprise.DEVICE_WARP_ENGINES] < 0) ? "0.2" : "8") + ")");
        # 获取用户输入的飞行速度
        float warp = Util.inputFloat("");
        # 如果飞行速度大于0.2且飞行引擎受损，则输出错误信息并返回
        if (deviceStatus[Enterprise.DEVICE_WARP_ENGINES] < 0 && warp > .2) {
            Util.println("WARP ENGINES ARE DAMAGED.  MAXIMUM SPEED = WARP 0.2");
            return;
        }
        # 如果飞行速度为0，则返回
        if (warp == 0) return;
        # 如果飞行速度在0到8之间
        if (warp > 0 && warp <= 8) {
            # 计算需要消耗的能量
            int n = Util.toInt(warp * 8);
            # 如果企业能量足够，则克林贡人移动并开火，修复受损设备，星际地图移动企业
            if (enterprise.getEnergy() - n >= 0) {
                galaxyMap.klingonsMoveAndFire(this);
                repairDamagedDevices(course, warp, n);
                galaxyMap.moveEnterprise(course, warp, n, stardate, initialStardate, missionDuration, this);
            } else {
                # 如果能量不足，则输出错误信息
                Util.println("ENGINEERING REPORTS   'INSUFFICIENT ENERGY AVAILABLE");
                Util.println("                       FOR MANEUVERING AT WARP " + warp + "!'");
                # 如果护盾能量不足或者护盾控制设备受损，则返回
                if (enterprise.getShields() < n - enterprise.getEnergy() || deviceStatus[Enterprise.DEVICE_SHIELD_CONTROL] < 0) return;
                # 输出护盾能量信息
                Util.println("DEFLECTOR CONTROL ROOM ACKNOWLEDGES " + enterprise.getShields() + " UNITS OF ENERGY");
                Util.println("                         PRESENTLY DEPLOYED TO SHIELDS.");
            }
        } else {
            # 如果飞行速度超出范围，则输出错误信息
            Util.println("   CHIEF ENGINEER SCOTT REPORTS 'THE ENGINES WON'T TAKE");
            Util.println(" WARP " + warp + "!'");
        }
    }
    // 修复受损设备并打印损坏报告
    void repairDamagedDevices(final float course, final float warp, final int N) {
        // 获取星系地图上的企业对象
        final Enterprise enterprise = galaxyMap.getEnterprise();
        // 修复受损设备
        enterprise.repairDamagedDevices(warp);
        // 如果随机数大于0.2，则有80%的概率没有损坏也没有修复
        if (Util.random() > .2) return;
        // 生成随机设备编号
        int randomDevice = Util.fnr();
        // 获取设备状态数组
        final double[] deviceStatus = enterprise.getDeviceStatus();
        // 如果随机数大于等于0.6，则有40%的概率修复随机设备
        if (Util.random() >= .6) {
            // 修改随机设备的状态为当前状态加上随机数乘以3再加1
            enterprise.setDeviceStatus(randomDevice, deviceStatus[randomDevice] + Util.random() * 3 + 1);
            // 打印修复报告
            Util.println("DAMAGE CONTROL REPORT:  " + Enterprise.printDeviceName(randomDevice) + " STATE OF REPAIR IMPROVED\n");
        } else {
            // 60%的概率损坏随机设备
            enterprise.setDeviceStatus(randomDevice, deviceStatus[randomDevice] - (Util.random() * 5 + 1));
            // 打印损坏报告
            Util.println("DAMAGE CONTROL REPORT:  " + Enterprise.printDeviceName(randomDevice) + " DAMAGED");
        }
    }

    // 进行长程传感器扫描
    void longRangeSensorScan() {
        // 调用星系地图的长程传感器扫描方法
        galaxyMap.longRangeSensorScan();
    }

    // 发射相位炮
    void firePhasers() {
        // 调用星系地图的发射相位炮方法
        galaxyMap.firePhasers(this);
    }

    // 发射光子鱼雷
    void firePhotonTorpedo() {
        // 调用星系地图的发射光子鱼雷方法
        galaxyMap.firePhotonTorpedo(stardate, initialStardate, missionDuration, this);
    }

    // 控制护盾
    void shieldControl() {
        // 调用星系地图上企业对象的护盾控制方法
        galaxyMap.getEnterprise().shieldControl();
    }

    // 进行短程传感器扫描
    void shortRangeSensorScan() {
        // 短程传感器扫描和启动子程序
        galaxyMap.shortRangeSensorScan(stardate);
    }
}
    // 输出状态报告
    void statusReport() {
        Util.println("   STATUS REPORT:");
        Util.println("KLINGON" + ((galaxyMap.getKlingonsInGalaxy() > 1)? "S" : "")  + " LEFT: " + galaxyMap.getKlingonsInGalaxy());
        Util.println("MISSION MUST BE COMPLETED IN " + .1 * Util.toInt((initialStardate + missionDuration - stardate) * 10) + " STARDATES");
        // 如果星际基地数量大于等于1，输出星际基地数量信息；否则输出无星际基地信息
        if (galaxyMap.getBasesInGalaxy() >= 1) {
            Util.println("THE FEDERATION IS MAINTAINING " + galaxyMap.getBasesInGalaxy() + " STARBASE" + ((galaxyMap.getBasesInGalaxy() > 1)? "S" : "") + " IN THE GALAXY");
        } else {
            Util.println("YOUR STUPIDITY HAS LEFT YOU ON YOUR OWN IN");
            Util.println("  THE GALAXY -- YOU HAVE NO STARBASES LEFT!");
        }
        // 进行企业损伤控制
        galaxyMap.getEnterprise().damageControl(this);
    }

    // 增加星际日期
    public void incrementStardate(double increment) {
        this.stardate += increment;
    }

    // 游戏失败结束
    public void endGameFail(final boolean enterpriseDestroyed) {    // 6220
        // 如果企业被摧毁，输出相应信息
        if (enterpriseDestroyed) {
            Util.println("\nTHE ENTERPRISE HAS BEEN DESTROYED.  THEN FEDERATION ");
            Util.println("WILL BE CONQUERED");
        }
        Util.println("\nIT IS STARDATE " + stardate);
        Util.println("THERE WERE " + galaxyMap.getKlingonsInGalaxy() + " KLINGON BATTLE CRUISERS LEFT AT");
        Util.println("THE END OF YOUR MISSION.");
        // 重复游戏
        repeatGame();
    }

    // 游戏成功结束
    public void endGameSuccess() {
        Util.println("CONGRATULATION, CAPTAIN!  THE LAST KLINGON BATTLE CRUISER");
        Util.println("MENACING THE FEDERATION HAS BEEN DESTROYED.\n");
        // 输出效率评级
        Util.println("YOUR EFFICIENCY RATING IS " + (Math.sqrt(1000 * (galaxyMap.getRemainingKlingons() / (stardate - initialStardate))));
        // 重复游戏
        repeatGame();
    }
    // 重复游戏的方法
    void repeatGame() {
        // 打印空行
        Util.println("\n");
        // 如果星系地图中存在基地
        if (galaxyMap.getBasesInGalaxy() != 0) {
            // 打印提示信息
            Util.println("THE FEDERATION IS IN NEED OF A NEW STARSHIP COMMANDER");
            Util.println("FOR A SIMILAR MISSION -- IF THERE IS A VOLUNTEER,");
            // 获取用户输入的回复
            final String reply = Util.inputStr("LET HIM STEP FORWARD AND ENTER 'AYE'");
            // 如果回复为'AYE'
            if ("AYE".equals(reply)) {
                // 设置重启标志为true
                this.restart = true;
            } else {
                // 退出程序
                System.exit(0);
            }
        }
    }
# 闭合前面的函数定义
```