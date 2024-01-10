# `basic-computer-games\84_Super_Star_Trek\java\GalaxyMap.java`

```
import java.util.stream.IntStream;

/**
 * Map of the galaxy divided in Quadrants and Sectors,
 * populated with stars, starbases, klingons, and the Enterprise.
 */
public class GalaxyMap {

    // markers
    static final String MARKER_EMPTY = "   ";  // 空白标记
    static final String MARKER_ENTERPRISE = "<*>";  // 企业标记
    static final String MARKER_KLINGON = "+K+";  // 克林贡标记
    static final String MARKER_STARBASE = ">!<";  // 星舰基地标记
    static final String MARKER_STAR = " * ";  // 星星标记

    static final int AVG_KLINGON_SHIELD_ENERGY = 200;  // 平均克林贡护盾能量

    // galaxy map
    public static final String QUADRANT_ROW = "                         ";  // 四象限行
    String quadrantMap = QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + QUADRANT_ROW + Util.leftStr(QUADRANT_ROW, 17);       // 当前象限地图
    final int[][] galaxy = new int[9][9];    // 8x8 galaxy map G，星系地图
    final int[][] klingonQuadrants = new int[4][4];    // 3x3 position of klingons K，克林贡所在象限
    final int[][] chartedGalaxy = new int[9][9];    // 8x8 charted galaxy map Z，已绘制的星系地图

    // galaxy state
    int basesInGalaxy = 0;  // 星系中的星舰基地数量
    int remainingKlingons;  // 剩余的克林贡数量
    int klingonsInGalaxy = 0;  // 星系中的克林贡数量
    final Enterprise enterprise = new Enterprise();  // 企业

    // quadrant state
    int klingons = 0;  // 克林贡数量
    int starbases = 0;  // 星舰基地数量
    int stars = 0;  // 星星数量
    int starbaseX = 0; // X coordinate of starbase，星舰基地的X坐标
    int starbaseY = 0; // Y coord of starbase，星舰基地的Y坐标

    public Enterprise getEnterprise() {
        return enterprise;
    }

    public int getBasesInGalaxy() {
        return basesInGalaxy;
    }

    public int getRemainingKlingons() {
        return remainingKlingons;
    }

    public int getKlingonsInGalaxy() {
        return klingonsInGalaxy;
    }

    double fnd(int i) {
        return Math.sqrt((klingonQuadrants[i][1] - enterprise.getSector()[Enterprise.COORD_X]) ^ 2 + (klingonQuadrants[i][2] - enterprise.getSector()[Enterprise.COORD_Y]) ^ 2);
    }
}
    // 构造函数，初始化星系地图
    public GalaxyMap() {
        // 获取企业飞船所在的象限坐标
        int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        // 填充克林贡人、星舰基地、星星
        IntStream.range(1, 8).forEach(x -> {
            IntStream.range(1, 8).forEach(y -> {
                // 初始化克林贡人数量和星系地图探索状态
                klingons = 0;
                chartedGalaxy[x][y] = 0;
                // 生成随机数
                float random = Util.random();
                // 根据随机数确定克林贡人的数量
                if (random > .98) {
                    klingons = 3;
                    klingonsInGalaxy += 3;
                } else if (random > .95) {
                    klingons = 2;
                    klingonsInGalaxy += 2;
                } else if (random > .80) {
                    klingons = 1;
                    klingonsInGalaxy += 1;
                }
                // 初始化星舰基地数量
                starbases = 0;
                // 根据随机数确定星舰基地的数量
                if (Util.random() > .96) {
                    starbases = 1;
                    basesInGalaxy = +1; // 应为 basesInGalaxy += 1;
                }
                // 在星系地图中记录克林贡人数量、星舰基地数量和星星的信息
                galaxy[x][y] = klingons * 100 + starbases * 10 + Util.fnr();
            });
        });
        // 如果星系中没有星舰基地
        if (basesInGalaxy == 0) {
            // 如果企业飞船所在象限的信息小于200
            if (galaxy[quadrantX][quadrantY] < 200) {
                // 在该象限增加星舰基地数量
                galaxy[quadrantX][quadrantY] = galaxy[quadrantX][quadrantY] + 120;
                klingonsInGalaxy = +1; // 应为 klingonsInGalaxy += 1;
            }
            // 设置星舰基地数量为1
            basesInGalaxy = 1;
            // 在该象限增加星舰基地数量
            galaxy[quadrantX][quadrantY] = +10; // 应为 galaxy[quadrantX][quadrantY] += 10;
            // 设置企业飞船所在象限的坐标
            enterprise.setQuadrant(new int[]{ Util.fnr(), Util.fnr() });
        }
        // 设置剩余克林贡人数量
        remainingKlingons = klingonsInGalaxy;
    }
}
    // Klingons移动并开火的方法
    public void klingonsMoveAndFire(GameCallback callback) {
        // 遍历所有克林贡战舰
        for (int i = 1; i <= klingons; i++) {
            // 如果克林贡战舰已被摧毁，则跳过
            if (klingonQuadrants[i][3] == 0) continue;
            // 在克林贡战舰所在的象限插入空标记
            insertMarker(MARKER_EMPTY, klingonQuadrants[i][1], klingonQuadrants[i][2]);
            // 在象限地图中找到一个空位置
            final int[] newCoords = findEmptyPlaceInQuadrant(quadrantMap);
            // 更新克林贡战舰的坐标
            klingonQuadrants[i][1] = newCoords[0];
            klingonQuadrants[i][2] = newCoords[1];
            // 在新位置插入克林贡战舰标记
            insertMarker(MARKER_KLINGON, klingonQuadrants[i][1], klingonQuadrants[i][2]);
        }
        // 克林贡战舰开火
        klingonsShoot(callback);
    }

    // 克林贡战舰开火的方法
    void klingonsShoot(GameCallback callback) {
        // 如果没有克林贡战舰，则返回
        if (klingons <= 0) return; // no klingons
        // 如果企业号停靠在星舰基地，则输出信息并返回
        if (enterprise.isDocked()) {
            Util.println("STARBASE SHIELDS PROTECT THE ENTERPRISE");
            return;
        }
        // 遍历前三个象限
        for (int i = 1; i <= 3; i++) {
            // 如果克林贡战舰已被摧毁，则跳过
            if (klingonQuadrants[i][3] <= 0) continue;
            // 计算克林贡战舰的命中数
            int hits = Util.toInt((klingonQuadrants[i][3] / fnd(1)) * (2 + Util.random()));
            // 减少企业号的生命值
            enterprise.sufferHitPoints(hits);
            // 更新克林贡战舰的生命值
            klingonQuadrants[i][3] = Util.toInt(klingonQuadrants[i][3] / (3 + Util.random()));
            // 输出受到攻击的信息
            Util.println(hits + " UNIT HIT ON ENTERPRISE FROM SECTOR " + klingonQuadrants[i][1] + "," + klingonQuadrants[i][2]);
            // 如果企业号的护盾值小于等于0，则游戏失败
            if (enterprise.getShields() <= 0) callback.endGameFail(true);
            Util.println("      <SHIELDS DOWN TO " + enterprise.getShields() + " UNITS>");
            // 如果命中数小于20，则继续下一次循环
            if (hits < 20) continue;
            // 如果随机数大于0.6，或者命中数除以企业号护盾值小于等于0.02，则继续下一次循环
            if ((Util.random() > .6) || (hits / enterprise.getShields() <= .02)) continue;
            // 随机选择一个设备，更新其状态
            int randomDevice = Util.fnr();
            enterprise.setDeviceStatus(randomDevice, enterprise.getDeviceStatus()[randomDevice]- hits / enterprise.getShields() - .5 * Util.random());
            // 输出设备受损的信息
            Util.println("DAMAGE CONTROL REPORTS " + Enterprise.printDeviceName(randomDevice) + " DAMAGED BY THE HIT'");
        }
    }
    // 移动企业号飞船到指定的坐标
    public void moveEnterprise(final float course, final float warp, final int n, final double stardate, final double initialStardate, final int missionDuration, final GameCallback callback) {
        // 在当前位置插入一个空标记
        insertMarker(MARKER_EMPTY, Util.toInt(enterprise.getSector()[Enterprise.COORD_X]), Util.toInt(enterprise.getSector()[Enterprise.COORD_Y]));
        // 移动企业号飞船到指定的坐标，并返回新的坐标
        final int[] sector = enterprise.moveShip(course, n, quadrantMap, stardate, initialStardate, missionDuration, callback);
        int sectorX = sector[Enterprise.COORD_X];
        int sectorY = sector[Enterprise.COORD_Y];
        // 在新的位置插入企业号飞船的标记
        insertMarker(MARKER_ENTERPRISE, Util.toInt(sectorX), Util.toInt(sectorY));
        // 进行飞船的机动能量消耗
        enterprise.maneuverEnergySR(n);
        double stardateDelta = 1;
        // 根据飞行速度调整星际日期增量
        if (warp < 1) stardateDelta = .1 * Util.toInt(10 * warp);
        // 增加星际日期
        callback.incrementStardate(stardateDelta);
        // 如果当前星际日期超过任务结束的星际日期，调用游戏结束回调函数
        if (stardate > initialStardate + missionDuration) callback.endGameFail(false);
    }
    // 执行长距离传感器扫描
    void longRangeSensorScan() {
        // 获取企业的象限坐标
        final int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        final int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        // 如果长距离传感器状态小于0，输出错误信息并返回
        if (enterprise.getDeviceStatus()[Enterprise.DEVICE_LONG_RANGE_SENSORS] < 0) {
            Util.println("LONG RANGE SENSORS ARE INOPERABLE");
            return;
        }
        // 输出正在扫描的象限坐标
        Util.println("LONG RANGE SCAN FOR QUADRANT " + quadrantX + "," + quadrantY);
        // 创建分隔行字符串
        final String rowStr = "-------------------";
        Util.println(rowStr);
        // 创建一个长度为4的整型数组
        final int[] n = new int[4];
        // 遍历周围的象限
        for (int i = quadrantX - 1; i <= quadrantX + 1; i++) {
            n[1] = -1;
            n[2] = -2;
            n[3] = -3;
            for (int j = quadrantY - 1; j <= quadrantY + 1; j++) {
                // 如果象限坐标在合法范围内
                if (i > 0 && i < 9 && j > 0 && j < 9) {
                    // 将周围象限的值存入数组，并更新已探索的星系象限
                    n[j - quadrantY + 2] = galaxy[i][j];
                    chartedGalaxy[i][j] = galaxy[i][j];
                }
            }
            for (int l = 1; l <= 3; l++) {
                // 输出分隔符
                Util.print(": ");
                // 如果象限值小于0，输出星号并继续下一次循环
                if (n[l] < 0) {
                    Util.print("*** ");
                    continue;
                }
                // 输出格式化后的象限值
                Util.print(Util.rightStr(Integer.toString(n[l] + 1000), 3) + " ");
            }
            // 输出分隔符和换行
            Util.println(": \n" + rowStr);
        }
    }
}
    // 计算并输出星际记录，根据参数决定是否输出累积报告
    public void cumulativeGalacticRecord(final boolean cumulativeReport) {
        // 获取企业的象限坐标
        final int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        final int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        // 如果需要输出累积报告
        if (cumulativeReport) {
            // 输出象限的星际记录标题
            Util.println("");
            Util.println("        ");
            Util.println("COMPUTER RECORD OF GALAXY FOR QUADRANT " + quadrantX + "," + quadrantY);
            Util.println("");
        } else {
            // 输出整个星际的标题
            Util.println("                        THE GALAXY");
        }
        // 输出星际记录的列标
        Util.println("       1     2     3     4     5     6     7     8");
        // 输出星际记录的行分隔线
        final String rowDivider = "     ----- ----- ----- ----- ----- ----- ----- -----";
        Util.println(rowDivider);
        // 遍历每一行
        for (int i = 1; i <= 8; i++) {
            // 输出行号
            Util.print(i + "  ");
            // 如果需要输出累积报告
            if (cumulativeReport) {
                // 输出当前象限的星际名称和位置
                int y = 1;
                String quadrantName = getQuadrantName(false, i, y);
                int tabLen = Util.toInt(15 - .5 * Util.strlen(quadrantName));
                Util.println(Util.tab(tabLen) + quadrantName);
                y = 5;
                quadrantName = getQuadrantName(false, i, y);
                tabLen = Util.toInt(39 - .5 * Util.strlen(quadrantName));
                Util.println(Util.tab(tabLen) + quadrantName);
            } else {
                // 如果不需要输出累积报告，则输出当前象限的星际情况
                for (int j = 1; j <= 8; j++) {
                    Util.print("   ");
                    // 如果星际未探索，则输出***
                    if (chartedGalaxy[i][j] == 0) {
                        Util.print("***");
                    } else {
                        // 否则输出星际的编号
                        Util.print(Util.rightStr(Integer.toString(chartedGalaxy[i][j] + 1000), 3));
                    }
                }
            }
            // 输出换行
            Util.println("");
            // 输出行分隔线
            Util.println(rowDivider);
        }
        // 输出空行
        Util.println("");
    }
    // 计算光子鱼雷的数据
    public void photonTorpedoData() {
        // 获取企业号所在的扇区坐标
        int sectorX = enterprise.getSector()[Enterprise.COORD_X];
        int sectorY = enterprise.getSector()[Enterprise.COORD_Y];
        // 如果克林贡舰船数量小于等于0，则打印无敌舰船消息并返回
        if (klingons <= 0) {
            printNoEnemyShipsMessage();
            return;
        }
        // 打印从企业号到克林贡战舰的消息
        Util.println("FROM ENTERPRISE TO KLINGON BATTLE CRUISER" + ((klingons > 1)? "S" : ""));
        // 遍历克林贡舰船所在的四个象限
        for (int i = 1; i <= 3; i++) {
            // 如果该象限内有克林贡舰船，则打印方向信息
            if (klingonQuadrants[i][3] > 0) {
                printDirection(sectorX, sectorY, klingonQuadrants[i][1], klingonQuadrants[i][2]);
            }
        }
    }

    // 方向距离计算器
    void directionDistanceCalculator() {
        // 获取企业号所在的象限坐标和扇区坐标
        int quadrantX = enterprise.getQuadrant()[Enterprise.COORD_X];
        int quadrantY = enterprise.getQuadrant()[Enterprise.COORD_Y];
        int sectorX = enterprise.getSector()[Enterprise.COORD_X];
        int sectorY = enterprise.getSector()[Enterprise.COORD_Y];
        // 打印方向/距离计算器的消息
        Util.println("DIRECTION/DISTANCE CALCULATOR:");
        Util.println("YOU ARE AT QUADRANT " + quadrantX + "," + quadrantY + " SECTOR " + sectorX + "," + sectorY);
        Util.print("PLEASE ENTER ");
        // 获取初始坐标和最终坐标
        int[] initialCoords = Util.inputCoords("  INITIAL COORDINATES (X,Y)");
        int[] finalCoords = Util.inputCoords("  FINAL COORDINATES (X,Y)");
        // 打印方向信息
        printDirection(initialCoords[0], initialCoords[1], finalCoords[0], finalCoords[1]);
    }
    // 打印从一个坐标到另一个坐标的方向和距离
    void printDirection(int from_x, int from_y, int to_x, int to_y) {
        // 计算 y 轴的变化量
        to_y = to_y - from_y;  // delta 2
        // 计算 x 轴的变化量
        from_y = from_x - to_x;    // delta 1
        // 根据 y 轴的变化量和 x 轴的变化量确定方向
        if (to_y > 0) {
            if (from_y < 0) {
                from_x = 7;
            } else {
                from_x = 1;
                int tempA = from_y;
                from_y = to_y;
                to_y = tempA;
            }
        } else {
            if (from_y > 0) {
                from_x = 3;
            } else {
                from_x = 5;
                int tempA = from_y;
                from_y = to_y;
                to_y = tempA;
            }
        }

        // 将 y 轴的变化量转换为正数
        from_y = Math.abs(from_y);
        to_y = Math.abs(to_y);

        // 根据 x 轴和 y 轴的变化量计算方向
        if (from_y > 0 || to_y > 0) {
            if (from_y >= to_y) {
                Util.println("DIRECTION = " + (from_x + to_y / from_y));
            } else {
                Util.println("DIRECTION = " + (from_x + 2 - to_y / from_y));
            }
        }
        // 计算从一个坐标到另一个坐标的距离
        Util.println("DISTANCE = " + Util.round(Math.sqrt(to_y ^ 2 + from_y ^ 2), 6));
    }

    // 打印从企业到星舰基地的导航数据
    void starbaseNavData() {
        int sectorX = enterprise.getSector()[Enterprise.COORD_X];
        int sectorY = enterprise.getSector()[Enterprise.COORD_Y];
        // 如果星舰基地存在，则打印导航数据
        if (starbases != 0) {
            Util.println("FROM ENTERPRISE TO STARBASE:");
            printDirection(sectorX, sectorY, starbaseX, starbaseY);
        } else {
            // 如果星舰基地不存在，则打印相应信息
            Util.println("MR. SPOCK REPORTS,  'SENSORS SHOW NO STARBASES IN THIS");
            Util.println(" QUADRANT.'");
        }
    }

    // 打印没有敌方飞船的信息
    void printNoEnemyShipsMessage() {
        Util.println("SCIENCE OFFICER SPOCK REPORTS  'SENSORS SHOW NO ENEMY SHIPS");
        Util.println("                                IN THIS QUADRANT'");
    }
    # 定义一个方法，根据参数 regionNameOnly 和 y 返回地区名称
    String getRegionName(final boolean regionNameOnly, final int y) {
        # 如果 regionNameOnly 为 false，则执行以下代码块
        if (!regionNameOnly) {
            # 根据 y 取模 4 的结果进行判断
            switch (y % 4) {
                # 如果余数为 0，则返回 " I"
                case 0:
                    return " I";
                # 如果余数为 1，则返回 " II"
                case 1:
                    return " II";
                # 如果余数为 2，则返回 " III"
                case 2:
                    return " III";
                # 如果余数为 3，则返回 " IV"
                case 3:
                    return " IV";
            }
        }
        # 如果 regionNameOnly 为 true 或者执行了 switch 语句后未返回结果，则返回空字符串
        return "";
    }
    # 根据给定的参数确定是否只返回区域名称，以及坐标位置，返回对应的象限名称
    String getQuadrantName(final boolean regionNameOnly, final int x, final int y) {
        # 如果 y 坐标小于等于 4
        if (y <= 4) {
            # 根据 x 坐标进行判断
            switch (x) {
                case 1:
                    return "ANTARES" + getRegionName(regionNameOnly, y);
                case 2:
                    return "RIGEL" + getRegionName(regionNameOnly, y);
                case 3:
                    return "PROCYON" + getRegionName(regionNameOnly, y);
                case 4:
                    return "VEGA" + getRegionName(regionNameOnly, y);
                case 5:
                    return "CANOPUS" + getRegionName(regionNameOnly, y);
                case 6:
                    return "ALTAIR" + getRegionName(regionNameOnly, y);
                case 7:
                    return "SAGITTARIUS" + getRegionName(regionNameOnly, y);
                case 8:
                    return "POLLUX" + getRegionName(regionNameOnly, y);
            }
        } else {
            # 如果 y 坐标大于 4
            switch (x) {
                case 1:
                    return "SIRIUS" + getRegionName(regionNameOnly, y);
                case 2:
                    return "DENEB" + getRegionName(regionNameOnly, y);
                case 3:
                    return "CAPELLA" + getRegionName(regionNameOnly, y);
                case 4:
                    return "BETELGEUSE" + getRegionName(regionNameOnly, y);
                case 5:
                    return "ALDEBARAN" + getRegionName(regionNameOnly, y);
                case 6:
                    return "REGULUS" + getRegionName(regionNameOnly, y);
                case 7:
                    return "ARCTURUS" + getRegionName(regionNameOnly, y);
                case 8:
                    return "SPICA" + getRegionName(regionNameOnly, y);
            }
        }
        # 如果以上条件都不满足，则返回未知错误
        return "UNKNOWN - ERROR";
    }
    // 在指定坐标位置插入标记
    void insertMarker(final String marker, final int x, final int y) {
        // 计算在一维数组中的位置
        final int pos = Util.toInt(y) * 3 + Util.toInt(x) * 24 + 1;
        // 如果标记长度不为3，输出错误信息并退出程序
        if (marker.length() != 3) {
            System.err.println("ERROR");
            System.exit(-1);
        }
        // 如果位置为1，将标记插入到指定位置
        if (pos == 1) {
            quadrantMap = marker + Util.rightStr(quadrantMap, 189);
        }
        // 如果位置为190，将标记插入到指定位置
        if (pos == 190) {
            quadrantMap = Util.leftStr(quadrantMap, 189) + marker;
        }
        // 在指定位置插入标记
        quadrantMap = Util.leftStr(quadrantMap, (pos - 1)) + marker + Util.rightStr(quadrantMap, (190 - pos));
    }

    /**
     * 在象限中查找随机的空坐标
     *
     * @param quadrantString
     * @return 一个包含一对坐标 x, y 的数组
     */
    int[] findEmptyPlaceInQuadrant(final String quadrantString) {
        // 生成随机的 x 和 y 坐标
        final int x = Util.fnr();
        final int y = Util.fnr();
        // 如果指定坐标不为空，递归调用查找空坐标的方法
        if (!compareMarker(quadrantString, MARKER_EMPTY, x, y)) {
            return findEmptyPlaceInQuadrant(quadrantString);
        }
        return new int[]{x, y};
    }

    // 比较指定坐标处的标记是否与给定标记相同
    boolean compareMarker(final String quadrantString, final String marker, final int x, final int y) {
        // 计算标记在一维数组中的位置
        final int markerRegion = (y - 1) * 3 + (x - 1) * 24 + 1;
        // 如果指定位置的标记与给定标记相同，返回 true，否则返回 false
        if (Util.midStr(quadrantString, markerRegion, 3).equals(marker)) {
            return true;
        }
        return false;
    }
# 闭合前面的函数定义
```