# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\java\Enterprise.java`

```
import java.util.stream.IntStream;  // 导入 Java 的 IntStream 类

/**
 * The starship Enterprise.  // 星际飞船企业
 */
public class Enterprise {

    public static final int COORD_X = 0;  // 定义 X 坐标的常量
    public static final int COORD_Y = 1;  // 定义 Y 坐标的常量

    // devices  // 设备
    static final int DEVICE_WARP_ENGINES = 1;  // 定义超空间引擎的常量
    static final int DEVICE_SHORT_RANGE_SENSORS = 2;  // 定义短程传感器的常量
    static final int DEVICE_LONG_RANGE_SENSORS = 3;  // 定义长程传感器的常量
    static final int DEVICE_PHASER_CONTROL = 4;  // 定义相位控制的常量
    static final int DEVICE_PHOTON_TUBES = 5;  // 定义光子管的常量
    static final int DEVICE_DAMAGE_CONTROL = 6;  // 定义损伤控制的常量
    static final int DEVICE_SHIELD_CONTROL = 7;  // 定义护盾控制的常量
    static final int DEVICE_LIBRARY_COMPUTER = 8;  // 定义图书馆计算机的常量
    final double[] deviceStatus = new double[9];   // 8  device damage stats  // 定义包含 9 个元素的 double 类型数组，用于存储设备损伤状态
}
    // position
    final int[][] cardinalDirections = new int[10][3];   // 9x2 vectors in cardinal directions
    int quadrantX;  // X coordinate of the quadrant
    int quadrantY;  // Y coordinate of the quadrant
    int sectorX;    // X coordinate of the sector
    int sectorY;    // Y coordinate of the sector

    // ship status
    boolean docked = false;  // indicates whether the ship is docked
    int energy = 3000;       // current energy level of the ship
    int torpedoes = 10;      // number of torpedoes available
    int shields = 0;         // current shield strength
    double repairCost;       // cost of repairing the ship

    final int initialEnergy = energy;    // initial energy level of the ship
    final int initialTorpedoes = torpedoes;  // initial number of torpedoes available

    public Enterprise() {
        // random initial position
        // 设置象限
        this.setQuadrant(new int[]{ Util.fnr(), Util.fnr() });
        // 设置扇区
        this.setSector(new int[]{ Util.fnr(), Util.fnr() });
        // 初始化基本方向
        IntStream.range(1, 9).forEach(i -> {
            cardinalDirections[i][1] = 0;
            cardinalDirections[i][2] = 0;
        });
        // 设置基本方向的值
        cardinalDirections[3][1] = -1;
        cardinalDirections[2][1] = -1;
        cardinalDirections[4][1] = -1;
        cardinalDirections[4][2] = -1;
        cardinalDirections[5][2] = -1;
        cardinalDirections[6][2] = -1;
        cardinalDirections[1][2] = 1;
        cardinalDirections[2][2] = 1;
        cardinalDirections[6][1] = 1;
        cardinalDirections[7][1] = 1;
        cardinalDirections[8][1] = 1;
        cardinalDirections[8][2] = 1;
        cardinalDirections[9][2] = 1;
        // 初始化设备状态数组，将所有设备状态初始化为0
        IntStream.range(1, 8).forEach(i -> deviceStatus[i] = 0);
    }

    // 获取护盾值
    public int getShields() {
        return shields;
    }

    /**
     * Enterprise 被敌人击中。
     * @param hits 被击中的点数
     */
    // 减少护盾值
    public void sufferHitPoints(int hits) {
        this.shields = shields - hits;
    }

    // 获取能量值
    public int getEnergy() {
        return energy;
    }
    public void replenishSupplies() {
        // 重置能量和鱼雷数量为初始值
        this.energy = this.initialEnergy;
        this.torpedoes = this.initialTorpedoes;
    }

    public void decreaseEnergy(final double amount) {
        // 减少能量
        this.energy -= amount;
    }

    public void decreaseTorpedoes(final int amount) {
        // 减少鱼雷数量
        torpedoes -= amount;
    }

    public void dropShields() {
        // 关闭护盾
        this.shields = 0;
    }

    public int getTotalEnergy() {
        // 返回总能量（包括护盾和能量）
        return (shields + energy);
    }
    # 返回初始能量值
    public int getInitialEnergy() {
        return initialEnergy;
    }

    # 返回鱼雷数量
    public int getTorpedoes() {
        return torpedoes;
    }

    # 返回设备状态数组
    public double[] getDeviceStatus() {
        return deviceStatus;
    }

    # 返回基本方向数组
    public int[][] getCardinalDirections() {
        return cardinalDirections;
    }

    # 设置特定设备的状态
    public void setDeviceStatus(final int device, final double status) {
        this.deviceStatus[device] = status;
    }
    # 返回当前飞船是否停靠在太空站
    public boolean isDocked() {
        return docked;
    }

    # 设置飞船是否停靠在太空站
    public void setDocked(boolean docked) {
        this.docked = docked;
    }

    # 返回飞船所在的象限坐标
    public int[] getQuadrant() {
        return new int[] {quadrantX, quadrantY};
    }

    # 设置飞船所在的象限坐标
    public void setQuadrant(final int[] quadrant) {
        this.quadrantX = quadrant[COORD_X];
        this.quadrantY = quadrant[COORD_Y];
    }

    # 返回飞船所在的扇区坐标
    public int[] getSector() {
        return new int[] {sectorX, sectorY};
    }
    }

    public void setSector(final int[] sector) {
        this.sectorX = sector[COORD_X];  # 设置当前飞船所在的 X 轴坐标
        this.sectorY = sector[COORD_Y];  # 设置当前飞船所在的 Y 轴坐标
    }

    public int[] moveShip(final float course, final int n, final String quadrantMap, final double stardate, final double initialStardate, final int missionDuration, final GameCallback callback) {
        int ic1 = Util.toInt(course);  # 将航向转换为整数
        float x1 = cardinalDirections[ic1][1] + (cardinalDirections[ic1 + 1][1] - cardinalDirections[ic1][1]) * (course - ic1);  # 计算 X 轴上的移动距离
        float x = sectorX;  # 保存当前 X 轴坐标
        float y = sectorY;  # 保存当前 Y 轴坐标
        float x2 = cardinalDirections[ic1][2] + (cardinalDirections[ic1 + 1][2] - cardinalDirections[ic1][2]) * (course - ic1);  # 计算 Y 轴上的移动距离
        final int initialQuadrantX = quadrantX;  # 保存初始象限 X 轴坐标
        final int initialQuadrantY = quadrantY;  # 保存初始象限 Y 轴坐标
        for (int i = 1; i <= n; i++) {  # 循环移动飞船
            sectorX += x1;  # 更新 X 轴坐标
            sectorY += x2;  # 更新 Y 轴坐标
            if (sectorX < 1 || sectorX >= 9 || sectorY < 1 || sectorY >= 9) {  # 判断是否超出象限限制
                // exceeded quadrant limits  # 超出象限限制
                x = 8 * quadrantX + x + n * x1;  # 计算 x 坐标的值
                y = 8 * quadrantY + y + n * x2;  # 计算 y 坐标的值
                quadrantX = Util.toInt(x / 8);  # 计算 x 坐标所在的象限
                quadrantY = Util.toInt(y / 8);  # 计算 y 坐标所在的象限
                sectorX = Util.toInt(x - quadrantX * 8);  # 计算 x 坐标所在的扇区
                sectorY = Util.toInt(y - quadrantY * 8);  # 计算 y 坐标所在的扇区
                if (sectorX == 0) {  # 如果 x 坐标所在的扇区为 0
                    quadrantX = quadrantX - 1;  # 更新 x 坐标所在的象限
                    sectorX = 8;  # 更新 x 坐标所在的扇区
                }
                if (sectorY == 0) {  # 如果 y 坐标所在的扇区为 0
                    quadrantY = quadrantY - 1;  # 更新 y 坐标所在的象限
                    sectorY = 8;  # 更新 y 坐标所在的扇区
                }
                boolean hitEdge = false;  # 初始化是否触碰边缘的标志为 false
                if (quadrantX < 1) {  # 如果 x 坐标所在的象限小于 1
                    hitEdge = true;  # 更新触碰边缘的标志为 true
                    quadrantX = 1;  # 更新 x 坐标所在的象限
                    sectorX = 1;  # 更新 x 坐标所在的扇区
                }
                # 如果飞船超出了X轴边界
                if (quadrantX > 8) {
                    # 标记飞船撞到了边界
                    hitEdge = true;
                    # 将飞船所在象限设置为8
                    quadrantX = 8;
                    # 将飞船所在扇区设置为8
                    sectorX = 8;
                }
                # 如果飞船超出了Y轴边界
                if (quadrantY < 1) {
                    # 标记飞船撞到了边界
                    hitEdge = true;
                    # 将飞船所在象限设置为8
                    quadrantY = 8;
                    # 将飞船所在扇区设置为8
                    sectorY = 8;
                }
                # 如果飞船超出了Y轴边界
                if (quadrantY > 8) {
                    # 标记飞船撞到了边界
                    hitEdge = true;
                    # 将飞船所在象限设置为8
                    quadrantY = 8;
                    # 将飞船所在扇区设置为8
                    sectorY = 8;
                }
                # 如果飞船撞到了边界
                if (hitEdge) {
                    # 打印星际舰队的消息
                    Util.println("LT. UHURA REPORTS MESSAGE FROM STARFLEET COMMAND:");
                    Util.println("  'PERMISSION TO ATTEMPT CROSSING OF GALACTIC PERIMETER");
                    Util.println("  IS HEREBY *DENIED*.  SHUT DOWN YOUR ENGINES.'");
                    Util.println("CHIEF ENGINEER SCOTT REPORTS  'WARP ENGINES SHUT DOWN");
                    Util.println("  AT SECTOR " + sectorX + "," + sectorY + " OF QUADRANT " + quadrantX + "," + quadrantY + ".'");
                    # 打印当前所在的象限和扇区的位置信息
                    if (stardate > initialStardate + missionDuration) callback.endGameFail(false);
                    # 如果当前星期大于初始星期加上任务持续时间，调用回调函数结束游戏并失败
                }
                if (8 * quadrantX + quadrantY == 8 * initialQuadrantX + initialQuadrantY) {
                    break;
                    # 如果当前象限的坐标等于初始象限的坐标，跳出循环
                }
                callback.incrementStardate(1);
                # 调用回调函数增加星期
                maneuverEnergySR(n);
                # 调用机动能量函数
                callback.enterNewQuadrant();
                # 调用回调函数进入新的象限
                return this.getSector();
                # 返回当前扇区
            } else {
                int S8 = Util.toInt(sectorX) * 24 + Util.toInt(sectorY) * 3 - 26; // S8 = pos
                # 计算当前扇区的位置
                if (!("  ".equals(Util.midStr(quadrantMap, S8, 2)))) {
                    sectorX = Util.toInt(sectorX - x1);
                    sectorY = Util.toInt(sectorY - x2);
                    # 如果当前扇区不为空白，则更新扇区坐标
                    Util.println("WARP ENGINES SHUT DOWN AT ");
                    Util.println("SECTOR " + sectorX + "," + sectorY + " DUE TO BAD NAVIGATION");
                    # 打印超空间引擎关闭的信息和由于错误导航而关闭的扇区坐标
                    break;
                    # 跳出循环
                }
            }
        }
        sectorX = Util.toInt(sectorX);  # 将sectorX转换为整数
        sectorY = Util.toInt(sectorY);  # 将sectorY转换为整数
        return this.getSector();  # 返回当前扇区
    }

    void randomRepairCost() {
        repairCost = .5 * Util.random();  # 生成随机的修复成本
    }

    public void repairDamagedDevices(final float warp) {
        // 修复受损设备并打印损坏报告
        for (int i = 1; i <= 8; i++) {  # 遍历设备列表
            if (deviceStatus[i] < 0) {  # 如果设备受损
                deviceStatus[i] += Math.min(warp, 1);  # 修复设备状态
                if ((deviceStatus[i] > -.1) && (deviceStatus[i] < 0)) {  # 如果设备状态介于-0.1和0之间
                    deviceStatus[i] = -.1;  # 将设备状态设置为-0.1
                    break;  # 退出循环
                } else if (deviceStatus[i] >= 0) {  # 如果设备状态大于等于0
                    Util.println("DAMAGE CONTROL REPORT:  ");  # 打印损坏控制报告
    public void maneuverEnergySR(final int N) {
        // 减去能量消耗和额外的10单位能量
        energy = energy - N - 10;
        // 如果能量仍然大于等于0，则返回
        if (energy >= 0) return;
        // 如果能量不足，则输出信息并将剩余能量转移到护盾上
        Util.println("SHIELD CONTROL SUPPLIES ENERGY TO COMPLETE THE MANEUVER.");
        shields = shields + energy;
        energy = 0;
        // 如果护盾能量小于等于0，则将护盾能量设为0
        if (shields <= 0) shields = 0;
    }

    void shieldControl() {
        // 如果护盾控制设备状态小于0，则输出信息并返回
        if (deviceStatus[DEVICE_SHIELD_CONTROL] < 0) {
            Util.println("SHIELD CONTROL INOPERABLE");
            return;
        }
        // 打印能量加护盾的总量
        Util.println("ENERGY AVAILABLE = " + (energy + shields));
        // 从用户输入获取要转移到护盾的能量单位数
        int energyToShields = Util.toInt(Util.inputFloat("NUMBER OF UNITS TO SHIELDS"));
        // 如果输入的能量单位数小于0或者等于当前护盾能量单位数，则打印<SHIELDS UNCHANGED>并返回
        if (energyToShields < 0 || shields == energyToShields) {
            Util.println("<SHIELDS UNCHANGED>");
            return;
        }
        // 如果输入的能量单位数大于总能量单位数或者大于当前能量加护盾的总量，则打印警告信息并返回
        if (energyToShields > energy + energyToShields) {
            Util.println("SHIELD CONTROL REPORTS  'THIS IS NOT THE FEDERATION TREASURY.'");
            Util.println("<SHIELDS UNCHANGED>");
            return;
        }
        // 更新能量和护盾的值
        energy = energy + shields - energyToShields;
        shields = energyToShields;
        // 打印护盾更新后的信息
        Util.println("DEFLECTOR CONTROL ROOM REPORT:");
        Util.println("  'SHIELDS NOW AT " + Util.toInt(shields) + " UNITS PER YOUR COMMAND.'");
    }

    // 损伤控制方法
    void damageControl(GameCallback callback) {
        // 如果损伤控制设备状态小于0，则打印报告不可用
        if (deviceStatus[DEVICE_DAMAGE_CONTROL] < 0) {
            Util.println("DAMAGE CONTROL REPORT NOT AVAILABLE");
        } else {
            # 打印设备状态表头
            Util.println("\nDEVICE             STATE OF REPAIR");
            # 遍历设备状态数组，打印设备名称和状态
            for (int deviceNr = 1; deviceNr <= 8; deviceNr++) {
                Util.print(printDeviceName(deviceNr) + Util.leftStr(GalaxyMap.QUADRANT_ROW, 25 - Util.strlen(printDeviceName(deviceNr))) + " " + Util.toInt(deviceStatus[deviceNr] * 100) * .01 + "\n");
            }
        }
        # 如果未停靠，则返回
        if (!docked) return;

        # 计算需要修理的设备数量
        double deltaToRepair = 0;
        for (int i = 1; i <= 8; i++) {
            if (deviceStatus[i] < 0) deltaToRepair += .1;
        }
        # 如果有需要修理的设备
        if (deltaToRepair > 0) {
            # 计算修理成本
            deltaToRepair += repairCost;
            # 如果修理成本大于等于1，则修理成本为0.9
            if (deltaToRepair >= 1) deltaToRepair = .9;
            # 打印修理信息
            Util.println("TECHNICIANS STANDING BY TO EFFECT REPAIRS TO YOUR SHIP;");
            Util.println("ESTIMATED TIME TO REPAIR:'" + .01 * Util.toInt(100 * deltaToRepair) + " STARDATES");
            # 接受用户输入，是否授权修理
            final String reply = Util.inputStr("WILL YOU AUTHORIZE THE REPAIR ORDER (Y/N)");
            # 如果用户授权修理
            if ("Y".equals(reply)) {
                # 遍历设备，进行修理
                for (int deviceNr = 1; deviceNr <= 8; deviceNr++) {
                    if (deviceStatus[deviceNr] < 0) deviceStatus[deviceNr] = 0;  // 如果设备状态小于0，则将设备状态设为0
                }
                callback.incrementStardate(deltaToRepair + .1);  // 调用回调函数，增加星际日期
            }
        }
    }

    public static String printDeviceName(final int deviceNumber) {  // 定义一个静态方法，用于打印设备名称，参数为设备编号
        // PRINTS DEVICE NAME  // 打印设备名称
        switch (deviceNumber) {  // 根据设备编号进行切换
            case DEVICE_WARP_ENGINES:  // 如果是WARP ENGINES设备
                return "WARP ENGINES";  // 返回"WARP ENGINES"
            case DEVICE_SHORT_RANGE_SENSORS:  // 如果是SHORT RANGE SENSORS设备
                return "SHORT RANGE SENSORS";  // 返回"SHORT RANGE SENSORS"
            case DEVICE_LONG_RANGE_SENSORS:  // 如果是LONG RANGE SENSORS设备
                return "LONG RANGE SENSORS";  // 返回"LONG RANGE SENSORS"
            case DEVICE_PHASER_CONTROL:  // 如果是PHASER CONTROL设备
                return "PHASER CONTROL";  // 返回"PHASER CONTROL"
            case DEVICE_PHOTON_TUBES:  // 如果是PHOTON TUBES设备
                return "PHOTON TUBES";  // 返回"PHOTON TUBES"
# 根据设备类型返回相应的控制名称
public String getControlName(int deviceType) {
    # 根据设备类型返回相应的控制名称
    switch (deviceType) {
        # 如果设备类型为损坏控制，返回“DAMAGE CONTROL”
        case DEVICE_DAMAGE_CONTROL:
            return "DAMAGE CONTROL";
        # 如果设备类型为护盾控制，返回“SHIELD CONTROL”
        case DEVICE_SHIELD_CONTROL:
            return "SHIELD CONTROL";
        # 如果设备类型为图书馆计算机，返回“LIBRARY-COMPUTER”
        case DEVICE_LIBRARY_COMPUTER:
            return "LIBRARY-COMPUTER";
    }
    # 如果设备类型不匹配以上任何一种情况，返回空字符串
    return "";
}
```