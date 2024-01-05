# `84_Super_Star_Trek\java\GalaxyMap.java`

```
import java.util.stream.IntStream;  // 导入 Java 的 IntStream 类

/**
 * Map of the galaxy divided in Quadrants and Sectors,
 * populated with stars, starbases, klingons, and the Enterprise.
 */
public class GalaxyMap {

    // markers
    static final String MARKER_EMPTY = "   ";  // 定义空位置的标记
    static final String MARKER_ENTERPRISE = "<*>";  // 定义企业号的标记
    static final String MARKER_KLINGON = "+K+";  // 定义克林贡的标记
    static final String MARKER_STARBASE = ">!<";  // 定义星舰基地的标记
    static final String MARKER_STAR = " * ";  // 定义星星的标记

    static final int AVG_KLINGON_SHIELD_ENERGY = 200;  // 定义克林贡护盾平均能量值

    // galaxy map
    public static final String QUADRANT_ROW = "                         ";  // 定义一个空的行
    String quadrantMap = QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + Util.leftStr(QUADRANT_ROW, 17);       // 当前象限地图
```
```java
    final int[][] galaxy = new int[9][9];    // 创建一个 9x9 的整数数组，表示 8x8 的星系地图 G
    final int[][] klingonQuadrants = new int[4][4];    // 创建一个 4x4 的整数数组，表示 3x3 克林贡位置 K
    final int[][] chartedGalaxy = new int[9][9];    // 创建一个 9x9 的整数数组，表示 8x8 的已绘制星系地图 Z

    // 星系状态
    int basesInGalaxy = 0;    // 星系中的基地数量
    int remainingKlingons;    // 剩余的克林贡数量
    int klingonsInGalaxy = 0;    // 星系中的克林贡数量
    final Enterprise enterprise = new Enterprise();    // 创建一个企业飞船对象

    // 象限状态
    int klingons = 0;    // 克林贡数量
    int starbases = 0;    // 星球基地数量
    int stars = 0;    // 星球数量
    int starbaseX = 0; // 星球基地的 X 坐标
    int starbaseY = 0; // 星球基地的 Y 坐标

    public Enterprise getEnterprise() {
        return enterprise;    // 返回企业飞船对象
    }
    // 返回星系中的基地数量
    public int getBasesInGalaxy() {
        return basesInGalaxy;
    }

    // 返回剩余的克林贡人数量
    public int getRemainingKlingons() {
        return remainingKlingons;
    }

    // 返回星系中的克林贡人数量
    public int getKlingonsInGalaxy() {
        return klingonsInGalaxy;
    }

    // 计算到第i个象限的距离
    double fnd(int i) {
        return Math.sqrt((klingonQuadrants[i][1] - enterprise.getSector()[Enterprise.COORD_X]) ^ 2 + (klingonQuadrants[i][2] - enterprise.getSector()[Enterprise.COORD_Y]) ^ 2);
    }

    // 构造函数，初始化星系地图
    public GalaxyMap() {
        // 获取企业号所在的象限坐标
        int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        // populate Klingons, Starbases, Stars
        // 填充克林贡人、星舰基地、星星
        IntStream.range(1, 8).forEach(x -> {
            IntStream.range(1, 8).forEach(y -> {
                klingons = 0; // 初始化克林贡人数量为0
                chartedGalaxy[x][y] = 0; // 初始化星图中的位置为0
                float random = Util.random(); // 生成一个随机数
                if (random > .98) { // 如果随机数大于0.98
                    klingons = 3; // 克林贡人数量为3
                    klingonsInGalaxy += 3; // 星系中的克林贡人数量增加3
                } else if (random > .95) { // 如果随机数大于0.95
                    klingons = 2; // 克林贡人数量为2
                    klingonsInGalaxy += 2; // 星系中的克林贡人数量增加2
                } else if (random > .80) { // 如果随机数大于0.80
                    klingons = 1; // 克林贡人数量为1
                    klingonsInGalaxy += 1; // 星系中的克林贡人数量增加1
                }
                starbases = 0; // 初始化星舰基地数量为0
                if (Util.random() > .96) { // 如果生成的随机数大于0.96
                    starbases = 1; // 星舰基地数量为1
                    basesInGalaxy = +1; // 星系中的星舰基地数量增加1
                }
                galaxy[x][y] = klingons * 100 + starbases * 10 + Util.fnr();  # 将星系中的克林贡数量乘以100，星舰基地数量乘以10，加上随机数，存储在星系数组中的特定位置
            });
        });
        if (basesInGalaxy == 0) {  # 如果星系中没有星舰基地
            if (galaxy[quadrantX][quadrantY] < 200) {  # 如果星系中特定位置的值小于200
                galaxy[quadrantX][quadrantY] = galaxy[quadrantX][quadrantY] + 120;  # 将特定位置的值加上120
                klingonsInGalaxy = +1;  # 星系中的克林贡数量加1
            }
            basesInGalaxy = 1;  # 星系中的星舰基地数量设为1
            galaxy[quadrantX][quadrantY] = +10;  # 特定位置的值设为10
            enterprise.setQuadrant(new int[]{ Util.fnr(), Util.fnr() });  # 设置企业号所在的星系坐标为随机生成的坐标
        }
        remainingKlingons = klingonsInGalaxy;  # 剩余的克林贡数量设为星系中的克林贡数量
    }

    void newQuadrant(final double stardate, final double initialStardate) {   // 1320
        final int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];  # 获取企业号所在的星系坐标的X值
        final int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];  # 获取企业号所在的星系坐标的Y值
        klingons = 0;  # 初始化克林贡数量为0
        starbases = 0;  // 初始化 starbases 变量为 0
        stars = 0;  // 初始化 stars 变量为 0
        enterprise.randomRepairCost();  // 调用 enterprise 对象的 randomRepairCost 方法
        chartedGalaxy[quadrantX][quadrantY] = galaxy[quadrantX][quadrantY];  // 将 galaxy 数组中指定位置的值赋给 chartedGalaxy 数组中相同位置的值
        if (!(quadrantX < 1 || quadrantX > 8 || quadrantY < 1 || quadrantY > 8)) {  // 如果 quadrantX 或 quadrantY 不在 1-8 的范围内
            final String quadrantName = getQuadrantName(false, quadrantX, quadrantY);  // 调用 getQuadrantName 方法获取象限名称并赋给 quadrantName 变量
            if (initialStardate == stardate) {  // 如果 initialStardate 等于 stardate
                Util.println("YOUR MISSION BEGINS WITH YOUR STARSHIP LOCATED\n" +
                        "IN THE GALACTIC QUADRANT, '" + quadrantName + "'.");  // 打印提示信息
            } else {
                Util.println("NOW ENTERING " + quadrantName + " QUADRANT . . .");  // 打印提示信息
            }
            Util.println("");  // 打印空行
            klingons = (int) Math.round(galaxy[quadrantX][quadrantY] * .01);  // 计算 klingons 的数量
            starbases = (int) Math.round(galaxy[quadrantX][quadrantY] * .1) - 10 * klingons;  // 计算 starbases 的数量
            stars = galaxy[quadrantX][quadrantY] - 100 * klingons - 10 * starbases;  // 计算 stars 的数量
            if (klingons != 0) {  // 如果 klingons 不等于 0
                Util.println("COMBAT AREA      CONDITION RED");  // 打印提示信息
                if (enterprise.getShields() <= 200) {  // 如果 enterprise 的护盾值小于等于 200
                    Util.println("   SHIELDS DANGEROUSLY LOW");  // 打印提示信息
        }
        // 使用字节流里面内容创建 ZIP 对象
        zip = zipfile.ZipFile(bio, 'r')
        // 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
        fdict = {n:zip.read(n) for n in zip.namelist()}
        // 关闭 ZIP 对象
        zip.close()
        // 返回结果字典
        return fdict
        }
        }
        // position bases
        if (starbases >= 1) {
            // 如果星舰基地数量大于等于1，找到象限地图中的空位置
            final int[] emptyCoordinate = findEmptyPlaceInQuadrant(quadrantMap);
            // 将星舰基地放置在找到的空位置上
            starbaseX = emptyCoordinate[0];
            starbaseY = emptyCoordinate[1];
            insertMarker(MARKER_STARBASE, emptyCoordinate[0], emptyCoordinate[1]);
        }
        // position stars
        // 放置星星
        for (int i = 1; i <= stars; i++) {
            // 找到象限地图中的空位置
            final int[] emptyCoordinate = findEmptyPlaceInQuadrant(quadrantMap);
            // 在找到的空位置上放置星星
            insertMarker(MARKER_STAR, emptyCoordinate[0], emptyCoordinate[1]);
        }
    }

    public void klingonsMoveAndFire(GameCallback callback) {
        // 克林贡人移动并开火
        for (int i = 1; i <= klingons; i++) {
            // 如果克林贡人所在象限的第三个位置为0，跳过本次循环
            if (klingonQuadrants[i][3] == 0) continue;
            // 在克林贡人所在位置放置空标记
            insertMarker(MARKER_EMPTY, klingonQuadrants[i][1], klingonQuadrants[i][2]);
            // 在象限地图中找到空位置的坐标
            final int[] newCoords = findEmptyPlaceInQuadrant(quadrantMap);
            // 将新的坐标存储到克林贡象限数组中
            klingonQuadrants[i][1] = newCoords[0];
            klingonQuadrants[i][2] = newCoords[1];
            // 在新的坐标位置插入克林贡标记
            insertMarker(MARKER_KLINGON, klingonQuadrants[i][1], klingonQuadrants[i][2]);
        }
        // 克林贡开火
        klingonsShoot(callback);
    }

    void klingonsShoot(GameCallback callback) {
        // 如果没有克林贡，则返回
        if (klingons <= 0) return; // no klingons
        // 如果企业停靠，则输出信息并返回
        if (enterprise.isDocked()) {
            Util.println("STARBASE SHIELDS PROTECT THE ENTERPRISE");
            return;
        }
        // 循环处理克林贡的射击
        for (int i = 1; i <= 3; i++) {
            // 如果克林贡象限数组中的第三个元素小于等于0，则继续下一次循环
            if (klingonQuadrants[i][3] <= 0) continue;
            // 计算克林贡的命中单位数
            int hits = Util.toInt((klingonQuadrants[i][3] / fnd(1)) * (2 + Util.random()));
            // 企业受到伤害
            enterprise.sufferHitPoints(hits);
            // 更新克林贡象限数组中的第三个元素
            klingonQuadrants[i][3] = Util.toInt(klingonQuadrants[i][3] / (3 + Util.random()));
            // 输出受到的伤害信息
            Util.println(hits + " UNIT HIT ON ENTERPRISE FROM SECTOR " + klingonQuadrants[i][1] + "," + klingonQuadrants[i][2]);
            # 如果企业的护盾值小于等于0，调用回调函数结束游戏并失败
            if (enterprise.getShields() <= 0) callback.endGameFail(true);
            # 打印输出企业护盾值下降到指定单位的信息
            Util.println("      <SHIELDS DOWN TO " + enterprise.getShields() + " UNITS>");
            # 如果被击中次数小于20，继续循环
            if (hits < 20) continue;
            # 如果随机数大于0.6，或者被击中次数除以企业护盾值小于等于0.02，继续循环
            if ((Util.random() > .6) || (hits / enterprise.getShields() <= .02)) continue;
            # 生成随机设备编号
            int randomDevice = Util.fnr();
            # 更新企业指定设备的状态
            enterprise.setDeviceStatus(randomDevice, enterprise.getDeviceStatus()[randomDevice]- hits / enterprise.getShields() - .5 * Util.random());
            # 打印输出设备受到损坏的信息
            Util.println("DAMAGE CONTROL REPORTS " + Enterprise.printDeviceName(randomDevice) + " DAMAGED BY THE HIT'");
        }
    }

    public void moveEnterprise(final float course, final float warp, final int n, final double stardate, final double initialStardate, final int missionDuration, final GameCallback callback) {
        # 在当前位置插入空标记
        insertMarker(MARKER_EMPTY, Util.toInt(enterprise.getSector()[Enterprise.COORD_X]), Util.toInt(enterprise.getSector()[Enterprise.COORD_Y]));
        # 移动企业飞船，并返回新的位置坐标
        final int[] sector = enterprise.moveShip(course, n, quadrantMap, stardate, initialStardate, missionDuration, callback);
        int sectorX = sector[Enterprise.COORD_X];
        int sectorY = sector[Enterprise.COORD_Y];
        # 在新位置插入企业飞船标记
        insertMarker(MARKER_ENTERPRISE, Util.toInt(sectorX), Util.toInt(sectorY));
        # 调整企业飞船的能量
        enterprise.maneuverEnergySR(n);
        # 计算星际日期的增量
        double stardateDelta = 1;
        if (warp < 1) stardateDelta = .1 * Util.toInt(10 * warp);
        # 调用回调函数增加星际日期
        callback.incrementStardate(stardateDelta);
        // 如果当前星日期大于初始星日期加上任务持续时间，调用回调函数结束游戏并失败
        if (stardate > initialStardate + missionDuration) callback.endGameFail(false);
    }

    // 进行短程传感器扫描
    void shortRangeSensorScan(final double stardate) {
        // 获取企业号所在的扇区坐标
        final int sectorX = enterprise.getSector()[Enterprise.COORD_X];
        final int sectorY = enterprise.getSector()[Enterprise.COORD_Y];
        boolean docked = false; // 是否停靠在星舰基地
        String shipCondition; // 飞船状态（停靠、红色、黄色、绿色）
        // 遍历周围的扇区
        for (int i = sectorX - 1; i <= sectorX + 1; i++) {
            for (int j = sectorY - 1; j <= sectorY + 1; j++) {
                // 判断扇区坐标是否在合法范围内
                if ((Util.toInt(i) >= 1) && (Util.toInt(i) <= 8) && (Util.toInt(j) >= 1) && (Util.toInt(j) <= 8)) {
                    // 如果当前扇区有星舰基地标记，则设置为停靠状态
                    if (compareMarker(quadrantMap, MARKER_STARBASE, i, j)) {
                        docked = true;
                    }
                }
            }
        }
        // 如果未停靠在星舰基地
        if (!docked) {
            enterprise.setDocked(false); // 设置企业号未停靠
            // 如果克林贡人数大于0
            if (klingons > 0) {
                shipCondition = "*RED*";  # 如果星际飞船的能量低于10%，则将飞船状态设置为红色
            } else {
                shipCondition = "GREEN";  # 如果星际飞船的能量高于10%，则将飞船状态设置为绿色
                if (enterprise.getEnergy() < enterprise.getInitialEnergy() * .1) {  # 如果星际飞船的能量低于10%
                    shipCondition = "YELLOW";  # 则将飞船状态设置为黄色
                }
            }
        } else {
            enterprise.setDocked(true);  # 如果星际飞船停靠，则设置飞船状态为停靠
            shipCondition = "DOCKED";  # 设置飞船状态为停靠
            enterprise.replenishSupplies();  # 补充飞船的补给
            Util.println("SHIELDS DROPPED FOR DOCKING PURPOSES");  # 打印信息，表示为对接目的而关闭护盾
            enterprise.dropShields();  # 关闭护盾
        }
        if (enterprise.getDeviceStatus()[Enterprise.DEVICE_SHORT_RANGE_SENSORS] < 0) {  # 如果短程传感器故障
            Util.println("\n*** SHORT RANGE SENSORS ARE OUT ***\n");  # 打印信息，表示短程传感器故障
            return;  # 结束函数
        }
        final String row = "---------------------------------";  # 创建一个横线字符串
        Util.println(row);  # 打印横线字符串
        for (int i = 1; i <= 8; i++) {  // 循环8次，i从1到8
            String sectorMapRow = "";  // 初始化一个空字符串
            for (int j = (i - 1) * 24 + 1; j <= (i - 1) * 24 + 22; j += 3) {  // 循环计算j的值，j的范围是从(i-1)*24+1到(i-1)*24+22，每次增加3
                sectorMapRow += " " + Util.midStr(quadrantMap, j, 3);  // 将Util.midStr(quadrantMap, j, 3)的结果添加到sectorMapRow字符串后面
            }
            switch (i) {  // 根据i的值进行不同的操作
                case 1:
                    Util.println(sectorMapRow + "        STARDATE           " + Util.toInt(stardate * 10) * .1);  // 打印包含STARDATE和stardate值的字符串
                    break;
                case 2:
                    Util.println(sectorMapRow + "        CONDITION          " + shipCondition);  // 打印包含CONDITION和shipCondition值的字符串
                    break;
                case 3:
                    Util.println(sectorMapRow + "        QUADRANT           " + enterprise.getQuadrant()[Enterprise.COORD_X] + "," + enterprise.getQuadrant()[Enterprise.COORD_Y]);  // 打印包含QUADRANT和enterprise.getQuadrant()[Enterprise.COORD_X]、enterprise.getQuadrant()[Enterprise.COORD_Y]的字符串
                    break;
                case 4:
                    Util.println(sectorMapRow + "        SECTOR             " + sectorX + "," + sectorY);  // 打印包含SECTOR和sectorX、sectorY的字符串
                    break;
                case 5:
                    Util.println(sectorMapRow + "        PHOTON TORPEDOES   " + Util.toInt(enterprise.getTorpedoes()));  // 打印包含PHOTON TORPEDOES和enterprise.getTorpedoes()的字符串
                    break;  // 结束当前的 case 分支
                case 6:  // 如果 switch 表达式的值为 6，则执行以下代码
                    Util.println(sectorMapRow + "        TOTAL ENERGY       " + Util.toInt(enterprise.getTotalEnergy()));  // 打印企业的总能量
                    break;  // 结束当前的 case 分支
                case 7:  // 如果 switch 表达式的值为 7，则执行以下代码
                    Util.println(sectorMapRow + "        SHIELDS            " + Util.toInt(enterprise.getShields()));  // 打印企业的护盾能量
                    break;  // 结束当前的 case 分支
                case 8:  // 如果 switch 表达式的值为 8，则执行以下代码
                    Util.println(sectorMapRow + "        KLINGONS REMAINING " + Util.toInt(klingonsInGalaxy));  // 打印银河中剩余的克林贡数量
            }
        }
        Util.println(row);  // 打印行
    }

    void longRangeSensorScan() {  // 定义长程传感器扫描方法
        final int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];  // 获取企业所在的象限的 X 坐标
        final int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];  // 获取企业所在的象限的 Y 坐标
        if (enterprise.getDeviceStatus()[Enterprise.DEVICE_LONG_RANGE_SENSORS] < 0) {  // 如果企业的长程传感器状态小于 0
            Util.println("LONG RANGE SENSORS ARE INOPERABLE");  // 打印“长程传感器不可用”
            return;  // 结束方法
        }
        # 打印长距离扫描的象限信息
        Util.println("LONG RANGE SCAN FOR QUADRANT " + quadrantX + "," + quadrantY);
        # 创建一个包含20个'-'的字符串
        final String rowStr = "-------------------";
        # 打印rowStr字符串
        Util.println(rowStr);
        # 创建一个包含4个整数的数组
        final int[] n = new int[4];
        # 循环遍历象限X-1到X+1
        for (int i = quadrantX - 1; i <= quadrantX + 1; i++) {
            n[1] = -1;
            n[2] = -2;
            n[3] = -3;
            # 循环遍历象限Y-1到Y+1
            for (int j = quadrantY - 1; j <= quadrantY + 1; j++) {
                # 如果i和j在指定范围内
                if (i > 0 && i < 9 && j > 0 && j < 9) {
                    # 将galaxy[i][j]的值赋给n[j - quadrantY + 2]
                    n[j - quadrantY + 2] = galaxy[i][j];
                    # 将galaxy[i][j]的值赋给chartedGalaxy[i][j]
                    chartedGalaxy[i][j] = galaxy[i][j];
                }
            }
            # 循环遍历1到3
            for (int l = 1; l <= 3; l++) {
                # 打印": "
                Util.print(": ");
                # 如果n[l]小于0，打印"*** "并继续下一次循环
                if (n[l] < 0) {
                    Util.print("*** ");
                    continue;
                }
                Util.print(Util.rightStr(Integer.toString(n[l] + 1000), 3) + " ");
            }
            Util.println(": \n" + rowStr);
        }
    }
```
这部分代码缺少注释，需要添加解释说明每个语句的作用。
        // 输出信息，表示准备锁定目标
        Util.println("PHASERS LOCKED ON TARGET;  ");
        // 声明整型变量 nrUnitsToFire
        int nrUnitsToFire;
        // 进入循环，输出当前能量值，等待用户输入要发射的能量单位
        while (true) {
            Util.println("ENERGY AVAILABLE = " + enterprise.getEnergy() + " UNITS");
            nrUnitsToFire = Util.toInt(Util.inputFloat("NUMBER OF UNITS TO FIRE"));
            // 如果用户输入的能量单位小于等于0，则返回
            if (nrUnitsToFire <= 0) return;
            // 如果企业的能量减去要发射的能量单位大于等于0，则跳出循环
            if (enterprise.getEnergy() - nrUnitsToFire >= 0) break;
        }
        // 企业的能量减去要发射的能量单位
        enterprise.decreaseEnergy(nrUnitsToFire);
        // 如果企业的护盾控制设备状态小于0，则将要发射的能量单位随机化
        if (deviceStatus[Enterprise.DEVICE_SHIELD_CONTROL] < 0) nrUnitsToFire = Util.toInt(nrUnitsToFire * Util.random());
        // 计算每个克林贡象限需要受到的能量单位
        int h1 = Util.toInt(nrUnitsToFire / klingons);
        // 遍历克林贡象限
        for (int i = 1; i <= 3; i++) {
            // 如果当前象限的克林贡舰船数量小于等于0，则跳出循环
            if (klingonQuadrants[i][3] <= 0) break;
            // 计算克林贡舰船受到的伤害值
            int hitPoints = Util.toInt((h1 / fnd(0)) * (Util.random() + 2));
            // 如果受到的伤害值小于等于克林贡象限舰船当前的15%，则输出无伤害信息，继续下一次循环
            if (hitPoints <= .15 * klingonQuadrants[i][3]) {
                Util.println("SENSORS SHOW NO DAMAGE TO ENEMY AT " + klingonQuadrants[i][1] + "," + klingonQuadrants[i][2]);
                continue;
            }
            // 更新克林贡象限舰船的剩余能量值，输出受到的伤害值和受到伤害的克林贡象限舰船的位置
            klingonQuadrants[i][3] = klingonQuadrants[i][3] - hitPoints;
            Util.println(hitPoints + " UNIT HIT ON KLINGON AT SECTOR " + klingonQuadrants[i][1] + "," + klingonQuadrants[i][2]);
            // 如果克林贡星区的敌舰数量小于等于0，表示克林贡星舰被摧毁
            if (klingonQuadrants[i][3] <= 0) {
                // 打印信息表示克林贡星舰被摧毁
                Util.println("*** KLINGON DESTROYED ***");
                // 敌舰数量减一
                klingons -= 1;
                // 星系中的克林贡星舰数量减一
                klingonsInGalaxy -= 1;
                // 在克林贡星区插入空标记
                insertMarker(MARKER_EMPTY, klingonQuadrants[i][1], klingonQuadrants[i][2]);
                // 将克林贡星区的敌舰数量设为0
                klingonQuadrants[i][3] = 0;
                // 星系中当前象限的值减去100
                galaxy[quadrantX][quadrantY] -= 100;
                // 记录星系中当前象限的值
                chartedGalaxy[quadrantX][quadrantY] = galaxy[quadrantX][quadrantY];
                // 如果星系中的克林贡星舰数量小于等于0，调用回调函数结束游戏
                if (klingonsInGalaxy <= 0) callback.endGameSuccess();
            } else {
                // 打印传感器显示的剩余敌舰数量
                Util.println("   (SENSORS SHOW " + klingonQuadrants[i][3] + " UNITS REMAINING)");
            }
        }
        // 克林贡星舰开火
        klingonsShoot(callback);
    }

    // 发射光子鱼雷
    void firePhotonTorpedo(final double stardate, final double initialStardate, final double missionDuration, GameCallback callback) {
        // 如果企业号的光子鱼雷数量小于等于0，打印信息并返回
        if (enterprise.getTorpedoes() <= 0) {
            Util.println("ALL PHOTON TORPEDOES EXPENDED");
            return;
        }
        # 如果光子管状态小于0，打印警告信息
        if (enterprise.getDeviceStatus()[Enterprise.DEVICE_PHOTON_TUBES] < 0) {
            Util.println("PHOTON TUBES ARE NOT OPERATIONAL");
        }
        # 输入光子鱼雷航向
        float c1 = Util.inputFloat("PHOTON TORPEDO COURSE (1-9)");
        # 如果航向为9，则改为1
        if (c1 == 9) c1 = 1;
        # 如果航向不在1到9之间，打印错误信息并返回
        if (c1 < 1 && c1 >= 9) {
            Util.println("ENSIGN CHEKOV REPORTS,  'INCORRECT COURSE DATA, SIR!'");
            return;
        }
        # 将航向转换为整数
        int ic1 = Util.toInt(c1);
        # 获取星舰的基本方向数组
        final int[][] cardinalDirections = enterprise.getCardinalDirections();
        # 计算光子鱼雷的x坐标
        float x1 = cardinalDirections[ic1][1] + (cardinalDirections[ic1 + 1][1] - cardinalDirections[ic1][1]) * (c1 - ic1);
        # 减少能量和鱼雷数量
        enterprise.decreaseEnergy(2);
        enterprise.decreaseTorpedoes(1);
        # 计算光子鱼雷的y坐标
        float x2 = cardinalDirections[ic1][2] + (cardinalDirections[ic1 + 1][2] - cardinalDirections[ic1][2]) * (c1 - ic1);
        # 获取星舰的当前x和y坐标
        float x = enterprise.getSector()[Enterprise.COORD_X];
        float y = enterprise.getSector()[Enterprise.COORD_Y];
        # 打印光子鱼雷轨迹信息
        Util.println("TORPEDO TRACK:");
        # 进入循环，等待后续操作
        while (true) {
            x = x + x1;  // 将变量 x 增加 x1 的值
            y = y + x2;  // 将变量 y 增加 x2 的值
            int x3 = Math.round(x);  // 将 x 取整并赋值给 x3
            int y3 = Math.round(y);  // 将 y 取整并赋值给 y3
            if (x3 < 1 || x3 > 8 || y3 < 1 || y3 > 8) {  // 如果 x3 或 y3 超出范围
                Util.println("TORPEDO MISSED"); // 打印信息 "TORPEDO MISSED"
                klingonsShoot(callback);  // 调用 klingonsShoot 函数并传入 callback 参数
                return;  // 结束当前函数
            }
            Util.println("               " + x3 + "," + y3);  // 打印坐标信息
            if (compareMarker(quadrantMap, MARKER_EMPTY, Util.toInt(x), Util.toInt(y)))  {  // 如果指定坐标为空
                continue;  // 继续下一次循环
            } else if (compareMarker(quadrantMap, MARKER_KLINGON, Util.toInt(x), Util.toInt(y))) {  // 如果指定坐标为克林贡战舰
                Util.println("*** KLINGON DESTROYED ***");  // 打印信息 "KLINGON DESTROYED"
                klingons = klingons - 1;  // 减少 klingons 变量的值
                klingonsInGalaxy = klingonsInGalaxy - 1;  // 减少 klingonsInGalaxy 变量的值
                if (klingonsInGalaxy <= 0) callback.endGameSuccess();  // 如果 klingonsInGalaxy 小于等于 0，调用 callback 的 endGameSuccess 函数
                for (int i = 1; i <= 3; i++) {  // 循环 3 次
                    if (x3 == klingonQuadrants[i][1] && y3 == klingonQuadrants[i][2]) break;  // 如果坐标与克林贡战舰的坐标匹配，跳出循环
                }
                int i = 3;  // 声明一个整型变量 i，并赋值为 3
                klingonQuadrants[i][3] = 0;  // 将 klingonQuadrants 数组中索引为 i 的子数组的第 3 个元素赋值为 0
            } else if (compareMarker(quadrantMap, MARKER_STAR, Util.toInt(x), Util.toInt(y))) {  // 如果 quadrantMap 中指定坐标位置的标记为星星
                Util.println("STAR AT " + x3 + "," + y3 + " ABSORBED TORPEDO ENERGY.");  // 打印星星吸收了鱼雷能量的信息
                klingonsShoot(callback);  // 克林贡人射击
                return;  // 返回
            } else if (compareMarker(quadrantMap, MARKER_STARBASE, Util.toInt(x), Util.toInt(y))) {  // 如果 quadrantMap 中指定坐标位置的标记为星舰基地
                Util.println("*** STARBASE DESTROYED ***");  // 打印星舰基地被摧毁的信息
                starbases = starbases - 1;  // 星舰基地数量减 1
                basesInGalaxy = basesInGalaxy - 1;  // 星系中的星舰基地数量减 1
                if (basesInGalaxy == 0 && klingonsInGalaxy <= stardate - initialStardate - missionDuration) {  // 如果星系中没有星舰基地，并且克林贡人数量小于等于当前日期减去初始日期减去任务持续时间
                    Util.println("THAT DOES IT, CAPTAIN!!  YOU ARE HEREBY RELIEVED OF COMMAND");  // 打印“够了，舰长！你被免除指挥权”的信息
                    Util.println("AND SENTENCED TO 99 STARDATES AT HARD LABOR ON CYGNUS 12!!");  // 打印“并被判处在赛格努斯 12 号上进行 99 个星期的苦役”的信息
                    callback.endGameFail(false);  // 回调游戏结束失败的函数
                } else {
                    Util.println("STARFLEET COMMAND REVIEWING YOUR RECORD TO CONSIDER");  // 打印“星际舰队司令部正在审查你的记录以考虑”的信息
                    Util.println("COURT MARTIAL!");  // 打印“军事法庭！”的信息
                    enterprise.setDocked(false);  // 设置企业号未对接
                }
            }
            // 在给定坐标位置插入标记
            insertMarker(MARKER_EMPTY, Util.toInt(x), Util.toInt(y));
            // 获取企业飞船所在的象限坐标
            final int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
            final int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
            // 在星系图中记录克林贡星舰、星舰基地和星球的数量
            galaxy[quadrantX][quadrantY] = klingons * 100 + starbases * 10 + stars;
            // 在已记录的星系图中更新当前象限的信息
            chartedGalaxy[quadrantX][quadrantY] = galaxy[quadrantX][quadrantY];
            // 克林贡星舰开火
            klingonsShoot(callback);
        }
    }

    public void cumulativeGalacticRecord(final boolean cumulativeReport) {
        // 获取企业飞船所在的象限坐标
        final int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        final int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        // 如果需要累积报告
        if (cumulativeReport) {
            // 打印象限坐标和标题
            Util.println("");
            Util.println("        ");
            Util.println("COMPUTER RECORD OF GALAXY FOR QUADRANT " + quadrantX + "," + quadrantY);
            Util.println("");
        } else {
            // 打印标题
            Util.println("                        THE GALAXY");
        }
        # 打印表头
        Util.println("       1     2     3     4     5     6     7     8")
        # 打印行分隔符
        final String rowDivider = "     ----- ----- ----- ----- ----- ----- ----- -----"
        Util.println(rowDivider)
        # 遍历每一行
        for (int i = 1; i <= 8; i++) {
            Util.print(i + "  ")
            # 如果是累积报告
            if (cumulativeReport) {
                int y = 1
                # 获取象限名称并打印
                String quadrantName = getQuadrantName(false, i, y)
                int tabLen = Util.toInt(15 - .5 * Util.strlen(quadrantName))
                Util.println(Util.tab(tabLen) + quadrantName)
                # 更新 y 值并获取象限名称并打印
                y = 5
                quadrantName = getQuadrantName(false, i, y)
                tabLen = Util.toInt(39 - .5 * Util.strlen(quadrantName))
                Util.println(Util.tab(tabLen) + quadrantName)
            } else {
                # 如果不是累积报告，遍历每一列
                for (int j = 1; j <= 8; j++) {
                    Util.print("   ")
                    # 如果星系已探索，打印星系编号
                    if (chartedGalaxy[i][j] == 0) {
                        Util.print("***")
                    } else {
                        Util.print(Util.rightStr(Integer.toString(chartedGalaxy[i][j] + 1000), 3));  # 使用Util类的print方法打印格式化后的星图数据
                    }
                }
            }
            Util.println("");  # 使用Util类的println方法打印空行
            Util.println(rowDivider);  # 使用Util类的println方法打印行分隔符
        }
        Util.println("");  # 使用Util类的println方法打印空行
    }

    public void photonTorpedoData() {  # 定义名为photonTorpedoData的公共方法
        int sectorX = enterprise.getSector()[Enterprise.COORD_X];  # 获取enterprise对象的X坐标
        int sectorY = enterprise.getSector()[Enterprise.COORD_Y];  # 获取enterprise对象的Y坐标
        if (klingons <= 0) {  # 如果klingons小于等于0
            printNoEnemyShipsMessage();  # 调用printNoEnemyShipsMessage方法
            return;  # 返回
        }
        Util.println("FROM ENTERPRISE TO KLINGON BATTLE CRUISER" + ((klingons > 1)? "S" : ""));  # 使用Util类的println方法打印信息
        for (int i = 1; i <= 3; i++) {  # 循环3次
            if (klingonQuadrants[i][3] > 0) {  # 如果klingonQuadrants[i][3]大于0
    // 打印方向和距离计算结果
    void directionDistanceCalculator() {
        // 获取企业的象限和扇区坐标
        int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        int sectorX = enterprise.getSector()[Enterprise.COORD_X];
        int sectorY = enterprise.getSector()[Enterprise.COORD_Y];
        // 打印当前位置信息
        Util.println("DIRECTION/DISTANCE CALCULATOR:");
        Util.println("YOU ARE AT QUADRANT " + quadrantX + "," + quadrantY + " SECTOR " + sectorX + "," + sectorY);
        Util.print("PLEASE ENTER ");
        // 获取用户输入的初始坐标和最终坐标
        int[] initialCoords = Util.inputCoords("  INITIAL COORDINATES (X,Y)");
        int[] finalCoords = Util.inputCoords("  FINAL COORDINATES (X,Y)");
        // 打印方向和距离
        printDirection(initialCoords[0], initialCoords[1], finalCoords[0], finalCoords[1]);
    }

    // 打印方向
    void printDirection(int from_x, int from_y, int to_x, int to_y) {
        // 计算 Y 轴的变化
        to_y = to_y - from_y;  // delta 2
        from_y = from_x - to_x;    // 计算 Y 轴的位移
        if (to_y > 0) {  // 如果目标位置在 Y 轴上方
            if (from_y < 0) {  // 如果起始位置在 Y 轴下方
                from_x = 7;  // 将起始位置移动到 X 轴的最右侧
            } else {  // 如果起始位置在 Y 轴上方
                from_x = 1;  // 将起始位置移动到 X 轴的最左侧
                int tempA = from_y;  // 临时变量存储起始位置的 Y 轴位移
                from_y = to_y;  // 将起始位置的 Y 轴位移设置为目标位置的 Y 轴位移
                to_y = tempA;  // 将目标位置的 Y 轴位移设置为临时变量中存储的起始位置的 Y 轴位移
            }
        } else {  // 如果目标位置在 Y 轴下方
            if (from_y > 0) {  // 如果起始位置在 Y 轴上方
                from_x = 3;  // 将起始位置移动到 X 轴的中间位置
            } else {  // 如果起始位置在 Y 轴下方
                from_x = 5;  // 将起始位置移动到 X 轴的中间位置
                int tempA = from_y;  // 临时变量存储起始位置的 Y 轴位移
                from_y = to_y;  // 将起始位置的 Y 轴位移设置为目标位置的 Y 轴位移
                to_y = tempA;  // 将目标位置的 Y 轴位移设置为临时变量中存储的起始位置的 Y 轴位移
            }
        }
        from_y = Math.abs(from_y);  # 将 from_y 取绝对值
        to_y = Math.abs(to_y);  # 将 to_y 取绝对值

        if (from_y > 0 || to_y > 0) {  # 如果 from_y 或 to_y 大于 0
            if (from_y >= to_y) {  # 如果 from_y 大于等于 to_y
                Util.println("DIRECTION = " + (from_x + to_y / from_y));  # 打印方向
            } else {
                Util.println("DIRECTION = " + (from_x + 2 - to_y / from_y));  # 打印方向
            }
        }
        Util.println("DISTANCE = " + Util.round(Math.sqrt(to_y ^ 2 + from_y ^ 2), 6));  # 打印距离
    }

    void starbaseNavData() {
        int sectorX = enterprise.getSector()[Enterprise.COORD_X];  # 获取企业的 X 坐标
        int sectorY = enterprise.getSector()[Enterprise.COORD_Y];  # 获取企业的 Y 坐标
        if (starbases != 0) {  # 如果星舰基地不等于 0
            Util.println("FROM ENTERPRISE TO STARBASE:");  # 打印从企业到星舰基地的信息
            printDirection(sectorX, sectorY, starbaseX, starbaseY);  # 打印方向
    } else {
        # 如果没有星际基地，则打印相应信息
        Util.println("MR. SPOCK REPORTS,  'SENSORS SHOW NO STARBASES IN THIS");
        Util.println(" QUADRANT.'");
    }
}

# 打印没有敌舰的信息
void printNoEnemyShipsMessage() {
    Util.println("SCIENCE OFFICER SPOCK REPORTS  'SENSORS SHOW NO ENEMY SHIPS");
    Util.println("                                IN THIS QUADRANT'");
}

# 获取区域名称
String getRegionName(final boolean regionNameOnly, final int y) {
    if (!regionNameOnly) {
        # 根据 y 的值返回相应的区域名称
        switch (y % 4) {
            case 0:
                return " I";
            case 1:
                return " II";
            case 2:
                return " III";
                case 3:  # 如果x的值为3
                    return " IV";  # 返回字符串" IV"
            }
        }
        return "";  # 如果不满足以上条件，返回空字符串
    }

    String getQuadrantName(final boolean regionNameOnly, final int x, final int y) {  # 定义一个名为getQuadrantName的函数，接受三个参数：regionNameOnly（布尔类型）、x（整数类型）、y（整数类型）
        if (y <= 4) {  # 如果y的值小于等于4
            switch (x) {  # 根据x的值进行判断
                case 1:  # 如果x的值为1
                    return "ANTARES" + getRegionName(regionNameOnly, y);  # 返回字符串"ANTARES"和调用getRegionName函数的结果
                case 2:  # 如果x的值为2
                    return "RIGEL" + getRegionName(regionNameOnly, y);  # 返回字符串"RIGEL"和调用getRegionName函数的结果
                case 3:  # 如果x的值为3
                    return "PROCYON" + getRegionName(regionNameOnly, y);  # 返回字符串"PROCYON"和调用getRegionName函数的结果
                case 4:  # 如果x的值为4
                    return "VEGA" + getRegionName(regionNameOnly, y);  # 返回字符串"VEGA"和调用getRegionName函数的结果
                case 5:  # 如果x的值为5
                    return "CANOPUS" + getRegionName(regionNameOnly, y);  # 返回字符串"CANOPUS"和调用getRegionName函数的结果
                case 6:  # 如果 x 的值为 6
                    return "ALTAIR" + getRegionName(regionNameOnly, y);  # 返回 "ALTAIR" 和 getRegionName(regionNameOnly, y) 的组合
                case 7:  # 如果 x 的值为 7
                    return "SAGITTARIUS" + getRegionName(regionNameOnly, y);  # 返回 "SAGITTARIUS" 和 getRegionName(regionNameOnly, y) 的组合
                case 8:  # 如果 x 的值为 8
                    return "POLLUX" + getRegionName(regionNameOnly, y);  # 返回 "POLLUX" 和 getRegionName(regionNameOnly, y) 的组合
            }
        } else {  # 如果 x 的值不是 6, 7, 8
            switch (x) {  # 根据 x 的值进行判断
                case 1:  # 如果 x 的值为 1
                    return "SIRIUS" + getRegionName(regionNameOnly, y);  # 返回 "SIRIUS" 和 getRegionName(regionNameOnly, y) 的组合
                case 2:  # 如果 x 的值为 2
                    return "DENEB" + getRegionName(regionNameOnly, y);  # 返回 "DENEB" 和 getRegionName(regionNameOnly, y) 的组合
                case 3:  # 如果 x 的值为 3
                    return "CAPELLA" + getRegionName(regionNameOnly, y);  # 返回 "CAPELLA" 和 getRegionName(regionNameOnly, y) 的组合
                case 4:  # 如果 x 的值为 4
                    return "BETELGEUSE" + getRegionName(regionNameOnly, y);  # 返回 "BETELGEUSE" 和 getRegionName(regionNameOnly, y) 的组合
                case 5:  # 如果 x 的值为 5
                    return "ALDEBARAN" + getRegionName(regionNameOnly, y);  # 返回 "ALDEBARAN" 和 getRegionName(regionNameOnly, y) 的组合
                case 6:  # 如果 x 的值为 6
                    return "REGULUS" + getRegionName(regionNameOnly, y);  # 返回以"REGULUS"开头的地区名称和地区名的后缀
                case 7:  # 如果条件为7
                    return "ARCTURUS" + getRegionName(regionNameOnly, y);  # 返回以"ARCTURUS"开头的地区名称和地区名的后缀
                case 8:  # 如果条件为8
                    return "SPICA" + getRegionName(regionNameOnly, y);  # 返回以"SPICA"开头的地区名称和地区名的后缀
            }
        }
        return "UNKNOWN - ERROR";  # 如果条件不满足，则返回"UNKNOWN - ERROR"
    }

    void insertMarker(final String marker, final int x, final int y) {  # 插入标记的方法，参数为标记、x坐标和y坐标
        final int pos = Util.toInt(y) * 3 + Util.toInt(x) * 24 + 1;  # 计算位置
        if (marker.length() != 3) {  # 如果标记长度不为3
            System.err.println("ERROR");  # 输出错误信息
            System.exit(-1);  # 退出程序
        }
        if (pos == 1) {  # 如果位置为1
            quadrantMap = marker + Util.rightStr(quadrantMap, 189);  # 在quadrantMap的开头插入标记
        }
        if (pos == 190) {  # 如果位置为190
        quadrantMap = Util.leftStr(quadrantMap, 189) + marker;  # 从字符串 quadrantMap 中取出前189个字符，然后拼接上 marker 字符串

        quadrantMap = Util.leftStr(quadrantMap, (pos - 1)) + marker + Util.rightStr(quadrantMap, (190 - pos));  # 从字符串 quadrantMap 中取出前 (pos - 1) 个字符，然后拼接上 marker 字符串，再拼接上从字符串 quadrantMap 中取出的后 (190 - pos) 个字符

    }

    /**
     * 在一个象限中找到随机的空坐标。
     *
     * @param quadrantString
     * @return 一个包含一对坐标 x, y 的数组
     */
    int[] findEmptyPlaceInQuadrant(final String quadrantString) {
        final int x = Util.fnr();  # 调用 Util 类的 fnr 方法生成一个随机数并赋值给 x
        final int y = Util.fnr();  # 调用 Util 类的 fnr 方法生成一个随机数并赋值给 y
        if (!compareMarker(quadrantString, MARKER_EMPTY, x, y)) {  # 调用 compareMarker 方法比较 quadrantString 中坐标 (x, y) 处的标记是否为 MARKER_EMPTY
            return findEmptyPlaceInQuadrant(quadrantString);  # 如果不是空坐标，则递归调用 findEmptyPlaceInQuadrant 方法直到找到空坐标
        }
        return new int[]{x, y};  # 返回包含 x 和 y 坐标的数组
    }
    # 定义一个方法，用于比较给定位置的标记是否与指定的标记相同
    def compareMarker(final String quadrantString, final String marker, final int x, final int y) {
        # 计算标记所在区域的起始位置
        final int markerRegion = (y - 1) * 3 + (x - 1) * 24 + 1;
        # 如果给定位置的标记与指定的标记相同，则返回 true
        if (Util.midStr(quadrantString, markerRegion, 3).equals(marker)) {
            return true;
        }
        # 否则返回 false
        return false;
    }
```
在这段代码中，定义了一个方法`compareMarker`，用于比较给定位置的标记是否与指定的标记相同。首先计算了标记所在区域的起始位置`markerRegion`，然后通过`Util.midStr`方法获取给定位置的标记，并与指定的标记进行比较，如果相同则返回`true`，否则返回`false`。
```