# `basic-computer-games\55_Life\java\src\java\Life.java`

```

import java.util.ArrayList; // 导入 ArrayList 类
import java.util.List; // 导入 List 接口
import java.util.Scanner; // 导入 Scanner 类

/**
 * The Game of Life class.<br>
 * <br>
 * Mimics the behaviour of the BASIC version, however the Java code does not have much in common with the original.
 * <br>
 * Differences in behaviour:
 * <ul>
 *     <li>Input supports the "." character, but it's optional.</li>
 *     <li>Input regarding the "DONE" string is case insensitive.</li>
 * </ul>
 */
}

/**
 * Represents a state change for a single cell within the matrix.
 *
 * @param y the y coordinate (row) of the cell
 * @param x the x coordinate (column) of the cell
 * @param newState the new state of the cell (either DEAD or ALIVE)
 */
record Transition(int y, int x, byte newState) { } // 定义了一个记录类型 Transition，表示单个细胞的状态变化，包括 y 坐标、x 坐标和新状态

```