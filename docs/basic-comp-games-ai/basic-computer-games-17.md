# BasicComputerGames源码解析 17

# `09_Battle/java/Input.java`



This is a simple Java class that reads input from the user. The input can be either a single coordinate or multiple coordinates separated by commas. The class has a boolean flag `isQuit` to indicate whether the input has ended early and a 2D array `coords` to store the last coordinates read.

The `Input` class has a constructor that takes an integer `scale` and a `BufferedReader` and `NumberFormat` instance as arguments. The `readCoordinates` method reads the input coordinates from the user and returns a boolean indicating whether the input has ended.

The `readCoordinates` method first writes a prompt to the console and reads the input from the user. If the input stream is ended, the method prints "Game quit" and sets the `isQuit` flag to `true`. If the input is a single coordinate, the method checks if it is valid and sets the `isQuit` flag to `true` if it is not. If the input is multiple coordinates separated by commas, the method reads each coordinate from the input and stores it in the `coords` array.

The `main` method uses an instance of the `Input` class and prints out the example usage.


```
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.text.NumberFormat;

// This class handles reading input from the player
// Each input is an x and y coordinate
// e.g. 5,3
public class Input {
    private BufferedReader reader;
    private NumberFormat parser;
    private int scale;             // size of the sea, needed to validate input
    private boolean isQuit;        // whether the input has ended
    private int[] coords;          // the last coordinates read

    public Input(int seaSize) {
        scale = seaSize;
        reader = new BufferedReader(new InputStreamReader(System.in));
        parser = NumberFormat.getIntegerInstance();
    }

    public boolean readCoordinates() throws IOException {
        while (true) {
            // Write a prompt
            System.out.print("\nTarget x,y\n> ");
            String inputLine = reader.readLine();
            if (inputLine == null) {
                // If the input stream is ended, there is no way to continue the game
                System.out.println("\nGame quit\n");
                isQuit = true;
                return false;
            }

            // split the input into two fields
            String[] fields = inputLine.split(",");
            if (fields.length != 2) {
                // has to be exactly two
                System.out.println("Need two coordinates separated by ','");
                continue;
            }

            coords = new int[2];
            boolean error = false;
            // each field should contain an integer from 1 to the size of the sea
            try {
                for (int c = 0 ; c < 2; ++c ) {
                    int val = Integer.parseInt(fields[c].strip());
                    if ((val < 1) || (val > scale)) {
                        System.out.println("Coordinates must be from 1 to " + scale);
                        error = true;
                    } else {
                        coords[c] = val;
                    }
                }
            }
            catch (NumberFormatException ne) {
                // this happens if the field is not a valid number
                System.out.println("Coordinates must be numbers");
                error = true;
            }
            if (!error) return true;
        }
    }

    public int x() { return coords[0]; }
    public int y() { return coords[1]; }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `09_Battle/java/Sea.java`



This is a Java class that represents a sea that can hold a certain number of ships. It has a size property that specifies the number of tiles in the sea, and a tiles property that is an array of integers representing the position of each tile in the sea. The class also has a get method that retrieves the value of a tile at a given coordinate, and a set method that allows the value of a tile at a given coordinate to be changed.

The class has a constructor that takes an integer value for the sea size, and initializes the size property with that value. The constructor also creates an empty sea with the specified size, and assigns the default value 0 to each tile.

The class has an encodedDump method that returns a string representation of the sea in the form of afunny order. The string representation includes the row and column indices of each tile, which are determined by the constructor.

The class has an isEmpty method that returns false if a coordinate is out of the range of 0-size, and true otherwise.

The class has a get method that takes a row and column coordinate and returns the value of the tile at that coordinate.

The class has a set method that takes a row and column coordinate and the new value to replace it with.

The class has a method to validate the input coordinate
```
if ((x < 0) || (x >= size)||(y < 0)||(y >= size)) return false;
```
The class also has a map method that maps the coordinates to the array index
```
private int index(int x, int y) {
   if ((x < 0) || (x >= size))
       throw new ArrayIndexOutOfBoundsException("Program error: x cannot be " + x);
   if ((y < 0) || (y >= size))
       throw new ArrayIndexOutOfBoundsException("Program error: y cannot be " + y);

   return y*size + x;
}
```
It also has an exception handling to check if the coordinate is valid before using it.


```
// Track the content of the sea
class Sea {
    // the sea is a square grid of tiles. It is a one-dimensional array, and this
    // class maps x and y coordinates to an array index
    // Each tile is either empty (value of tiles at index is 0)
    // or contains a ship (value of tiles at index is the ship number)
    private int tiles[];

    private int size;

    public Sea(int make_size) {
        size = make_size;
        tiles = new int[size*size];
    }

    public int size() { return size; }

    // This writes out a representation of the sea, but in a funny order
    // The idea is to give the player the job of working it out
    public String encodedDump() {
        StringBuilder out = new StringBuilder();
        for (int x = 0; x < size; ++x) {
            for (int y = 0; y < size; ++y)
                out.append(Integer.toString(get(x, y)));
            out.append('\n');
        }
        return out.toString();
    }

    /* return true if x,y is in the sea and empty
     * return false if x,y is occupied or is out of range
     * Doing this in one method makes placing ships much easier
     */
    public boolean isEmpty(int x, int y) {
        if ((x<0)||(x>=size)||(y<0)||(y>=size)) return false;
        return (get(x,y) == 0);
    }

    /* return the ship number, or zero if no ship.
     * Unlike isEmpty(x,y), these other methods require that the
     * coordinates passed be valid
     */
    public int get(int x, int y) {
        return tiles[index(x,y)];
    }

    public void set(int x, int y, int value) {
        tiles[index(x, y)] = value;
    }

    // map the coordinates to the array index
    private int index(int x, int y) {
        if ((x < 0) || (x >= size))
            throw new ArrayIndexOutOfBoundsException("Program error: x cannot be " + x);
        if ((y < 0) || (y >= size))
            throw new ArrayIndexOutOfBoundsException("Program error: y cannot be " + y);

        return y*size + x;
    }
}

```

# `09_Battle/java/Ship.java`

It looks like you are implementing a simple game where the player can place ships on a sea floor. The game has two ships that can be placed on the sea floor. Each ship has an x and y position on the sea floor, as well as an "id" that is unique for that tile.

The game has a main loop that checks for the placement of ships on the sea floor. If a ship can be placed, the game increments a counter that indicates the number of tiles placed. If the ship cannot be placed because it would collide with another ship or the edge of the sea floor, the game returns false.

If a ship has been placed, the game marks it on the sea floor using the `set` method. It also marks the tile as placed using the `set` method.

Finally, the game checks whether the ship can extend its length by placing tiles on the sea floor that are on the opposite side of the ship from the "to" coordinate. If the ship can extend its length, the game marks it as extended. If the ship cannot extend its length, the game returns false.


```
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;
import java.util.function.Predicate;

/** A single ship, with its position and where it has been hit */
class Ship {
    // These are the four directions that ships can be in
    public static final int ORIENT_E=0;   // goes East from starting position
    public static final int ORIENT_SE=1;  // goes SouthEast from starting position
    public static final int ORIENT_S=2;   // goes South from starting position
    public static final int ORIENT_SW=3;  // goes SouthWest from starting position

    private int id;                   // ship number
    private int size;                 // how many tiles it occupies
    private boolean placed;           // whether this ship is in the sea yet
    private boolean sunk;             // whether this ship has been sunk
    private ArrayList<Boolean> hits;  // which tiles of the ship have been hit

    private int startX;               // starting position coordinates
    private int startY;
    private int orientX;              // x and y deltas from each tile occupied to the next
    private int orientY;

    public Ship(int i, int sz) {
        id = i; size = sz;
        sunk = false; placed = false;
        hits = new ArrayList<>(Collections.nCopies(size, false));
    }

    /** @returns the ship number */
    public int id() { return id; }
    /** @returns the ship size */
    public int size() { return size; }

    /* record the ship as having been hit at the given coordinates */
    public void hit(int x, int y) {
        // need to work out how many tiles from the ship's starting position the hit is at
        // that can be worked out from the difference between the starting X coord and this one
        // unless the ship runs N-S, in which case use the Y coord instead
        int offset;
        if (orientX != 0) {
            offset = (x - startX) / orientX;
        } else {
            offset = (y - startY) / orientY;
        }
        hits.set(offset, true);

        // if every tile of the ship has been hit, the ship is sunk
        sunk = hits.stream().allMatch(Predicate.isEqual(true));
    }

    public boolean isSunk() { return sunk; }

    // whether the ship has already been hit at the given coordinates
    public boolean wasHit(int x, int y) {
        int offset;
        if (orientX != 0) {
            offset = (x - startX) / orientX;
        } else {
            offset = (y - startY) / orientY;
        }
        return hits.get(offset);
    };

    // Place the ship in the sea.
    // choose a random starting position, and a random direction
    // if that doesn't fit, keep picking different positions and directions
    public void placeRandom(Sea s) {
        Random random = new Random();
        for (int tries = 0 ; tries < 1000 ; ++tries) {
            int x = random.nextInt(s.size());
            int y = random.nextInt(s.size());
            int orient = random.nextInt(4);

            if (place(s, x, y, orient)) return;
        }

        throw new RuntimeException("Could not place any more ships");
    }

    // Attempt to fit the ship into the sea, starting from a given position and
    // in a given direction
    // This is by far the most complicated part of the program.
    // It will start at the position provided, and attempt to occupy tiles in the
    // requested direction. If it does not fit, either because of the edge of the
    // sea, or because of ships already in place, it will try to extend the ship
    // in the opposite direction instead. If that is not possible, it fails.
    public boolean place(Sea s, int x, int y, int orient) {
        if (placed) {
            throw new RuntimeException("Program error - placed ship " + id + " twice");
        }
        switch(orient) {
        case ORIENT_E:                 // east is increasing X coordinate
            orientX = 1; orientY = 0;
            break;
        case ORIENT_SE:                // southeast is increasing X and Y
            orientX = 1; orientY = 1;
            break;
        case ORIENT_S:                 // south is increasing Y
            orientX = 0; orientY = 1;
            break;
        case ORIENT_SW:                // southwest is increasing Y but decreasing X
            orientX = -1; orientY = 1;
            break;
        default:
            throw new RuntimeException("Invalid orientation " + orient);
        }

        if (!s.isEmpty(x, y)) return false; // starting position is occupied - placing fails

        startX = x; startY = y;
        int tilesPlaced = 1;
        int nextX = startX;
        int nextY = startY;
        while (tilesPlaced < size) {
            if (extendShip(s, nextX, nextY, nextX + orientX, nextY + orientY)) {
                // It is clear to extend the ship forwards
                tilesPlaced += 1;
                nextX = nextX + orientX;
                nextY = nextY + orientY;
            } else {
                int backX = startX - orientX;
                int backY = startY - orientY;

                if (extendShip(s, startX, startY, backX, backY)) {
                    // We can move the ship backwards, so it can be one tile longer
                    tilesPlaced +=1;
                    startX = backX;
                    startY = backY;
                } else {
                    // Could not make it longer or move it backwards
                    return false;
                }
            }
        }

        // Mark in the sea which tiles this ship occupies
        for (int i = 0; i < size; ++i) {
            int sx = startX + i * orientX;
            int sy = startY + i * orientY;
            s.set(sx, sy, id);
        }

        placed = true;
        return true;
    }

    // Check whether a ship which already occupies the "from" coordinates,
    // can also occupy the "to" coordinates.
    // They must be within the sea area, empty, and not cause the ship to cross
    // over another ship
    private boolean extendShip(Sea s, int fromX, int fromY, int toX, int toY) {
        if (!s.isEmpty(toX, toY)) return false;                  // no space
        if ((fromX == toX)||(fromY == toY)) return true;         // horizontal or vertical

        // we can extend the ship without colliding, but we are going diagonally
        // and it should not be possible for two ships to cross each other on
        // opposite diagonals.

        // check the two tiles that would cross us here - if either is empty, we are OK
        // if they both contain different ships, we are OK
        // but if they both contain the same ship, we are crossing!
        int corner1 = s.get(fromX, toY);
        int corner2 = s.get(toX, fromY);
        if ((corner1 == 0) || (corner1 != corner2)) return true;
        return false;
    }
}

```

# `09_Battle/javascript/battle.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是打印一段字符串，将字符串添加到页面上。该函数通过访问一个叫做`output`的元素的`appendChild()`方法来添加文本节点，将字符串作为参数传递给该方法，最终在页面上显示出来。

`input()`函数的作用是接收用户输入的字符串，返回一个Promise对象。该函数会创建一个带有type属性值为"text"，长度为50的输入元素，并将输入元素添加到页面上。该函数还添加了一个键盘事件监听器，当按下数字13时，将用户输入的字符串作为参数打印出来，并从页面上移除输入元素。最终，该函数将返回用户输入的字符串。


```
// BATTLE
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

```

这段代码定义了一个名为 `tab` 的函数，它会接受一个参数 `space`，它是一个整数，表示要在空格中输出字符。

函数体中，首先定义了一个字符串变量 `str`，并将其初始化为一个空字符串。然后，使用了一个 while 循环，条件是 `space` 大于 0，即 `space` 还有值。在循环中，每次将一个空格添加到 `str` 的开头。

循环结束后，将 `str` 返回，它现在包含了一个由空格组成的字符串。

接下来，定义了几个变量，它们的作用不在这里展开，可以看出来是用来存储一些数据的。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var fa = [];
var ha = [];
var aa = [];
var ba = [];
var ca = [];
var la = [];

```

这是一段简单的游戏脚本，名为“The Third德川家康”。这个脚本似乎在描写一个游戏场景，其中玩家需要操作各种军事舰艇，在地图上摧毁敌方的舰艇。

这个脚本使用了德川家康的名字来命名一些功能，例如创建舰队、攻击敌方舰艇和升级舰队。看起来玩家需要通过操作各种军事舰艇，在地图上摧毁敌方的舰艇来获得胜利。

这个脚本还使用了“我们党完全摧毁敌方的舰队”来描述玩家的目标。这可能是为了告诉玩家这个游戏的主要目标是什么。

总的来说，这个脚本似乎在向玩家介绍这个游戏，并告诉他们如何操作各种军事舰艇来获得胜利。



```
// Main program
async function main()
{
    print(tab(33) + "BATTLE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // -- BATTLE WRITTEN BY RAY WESTERGARD  10/70
    // COPYRIGHT 1971 BY THE REGENTS OF THE UNIV. OF CALIF.
    // PRODUCED AT THE LAWRENCE HALL OF SCIENCE, BERKELEY
    while (1) {
        for (x = 1; x <= 6; x++) {
            fa[x] = [];
            ha[x] = [];
            for (y = 1; y <= 6; y++) {
                fa[x][y] = 0;
                ha[x][y] = 0;
            }
        }
        for (i = 1; i <= 3; i++) {
            n = 4 - i;
            for (j = 1; j <= 2; j++) {
                while (1) {
                    a = Math.floor(6 * Math.random() + 1);
                    b = Math.floor(6 * Math.random() + 1);
                    d = Math.floor(4 * Math.random() + 1);
                    if (fa[a][b] > 0)
                        continue;
                    m = 0;
                    switch (d) {
                        case 1:
                            ba[1] = b;
                            ba[2] = 7;
                            ba[3] = 7;
                            for (k = 1; k <= n; k++) {
                                if (m <= 1 && ba[k] != 6 && fa[a][ba[k] + 1] <= 0) {
                                    ba[k + 1] = ba[k] + 1;
                                } else {
                                    m = 2;
                                    if (ba[1] < ba[2] && ba[1] < ba[3])
                                        z = ba[1];
                                    if (ba[2] < ba[1] && ba[2] < ba[3])
                                        z = ba[2];
                                    if (ba[3] < ba[1] && ba[3] < ba[2])
                                        z = ba[3];
                                    if (z == 1)
                                        break;
                                    if (fa[a][z - 1] > 0)
                                        break;
                                    ba[k + 1] = z - 1;
                                }
                            }
                            if (k <= n)
                                continue;
                            fa[a][b] = 9 - 2 * i - j;
                            for (k = 1; k <= n; k++)
                                fa[a][ba[k + 1]] = fa[a][b];
                            break;
                        case 2:
                            aa[1] = a;
                            ba[1] = b;
                            aa[2] = 0;
                            aa[3] = 0;
                            ba[2] = 0;
                            ba[3] = 0;
                            for (k = 1; k <= n; k++) {
                                if (m <= 1 && aa[k] != 1 && ba[k] != 1 && fa[aa[k] - 1][ba[k] - 1] <= 0 && (fa[aa[k] - 1][ba[k]] <= 0 || fa[aa[k] - 1][ba[k]] != fa[aa[k]][ba[k] - 1])) {
                                    aa[k + 1] = aa[k] - 1;
                                    ba[k + 1] = ba[k] - 1;
                                } else {
                                    m = 2;
                                    if (aa[1] > aa[2] && aa[1] > aa[3])
                                        z1 = aa[1];
                                    if (aa[2] > aa[1] && aa[2] > aa[3])
                                        z1 = aa[2];
                                    if (aa[3] > aa[1] && aa[3] > aa[2])
                                        z1 = aa[3];
                                    if (ba[1] > ba[2] && ba[1] > ba[3])
                                        z2 = ba[1];
                                    if (ba[2] > ba[1] && ba[2] > ba[3])
                                        z2 = ba[2];
                                    if (ba[3] > ba[1] && ba[3] > ba[2])
                                        z2 = ba[3];
                                    if (z1 == 6 || z2 == 6)
                                        break;
                                    if (fa[z1 + 1][z2 + 1] > 0)
                                        break;
                                    if (fa[z1][z2 + 1] > 0 && fa[z1][z2 + 1] == fa[z1 + 1][z2])
                                        break;
                                    aa[k + 1] = z1 + 1;
                                    ba[k + 1] = z2 + 1;
                                }
                            }
                            if (k <= n)
                                continue;
                            fa[a][b] = 9 - 2 * i - j;
                            for (k = 1; k <= n; k++)
                                fa[aa[k + 1]][ba[k + 1]] = fa[a][b];
                            break;
                        case 3:
                            aa[1] = a;
                            aa[2] = 7;
                            aa[3] = 7;
                            for (k = 1; k <= n; k++) {
                                if (m <= 1 && aa[k] != 6 && fa[aa[k] + 1][b] <= 0) {
                                    aa[k + 1] = aa[k] + 1;
                                } else {
                                    m = 2;
                                    if (aa[1] < aa[2] && aa[1] < aa[3])
                                        z = aa[1];
                                    if (aa[2] < aa[1] && aa[2] < aa[3])
                                        z = aa[2];
                                    if (aa[3] < aa[1] && aa[3] < aa[2])
                                        z = aa[3];
                                    if (z == 1)
                                        break;
                                    if (fa[z - 1][b] > 0)
                                        break;
                                    aa[k + 1] = z - 1;
                                }
                            }
                            if (k <= n)
                                continue;
                            fa[a][b] = 9 - 2 * i - j;
                            for (k = 1; k <= n; k++)
                                fa[aa[k + 1]][b] = fa[a][b];
                            break;
                        case 4:
                            aa[1] = a;
                            ba[1] = b;
                            aa[2] = 7;
                            aa[3] = 7;
                            ba[2] = 0;
                            ba[3] = 0;
                            for (k = 1; k <= n; k++) {
                                if (m <= 1 && aa[k] != 6 && ba[k] != 1 && fa[aa[k] + 1][ba[k] - 1] <= 0 && (fa[aa[k] + 1][ba[k]] <= 0 || fa[aa[k] + 1][ba[k]] != fa[aa[k]][ba[k] - 1])) {
                                    aa[k + 1] = aa[k] + 1;
                                    ba[k + 1] = ba[k] - 1;
                                } else {
                                    m = 2;
                                    if (aa[1] < aa[2] && aa[1] < aa[3])
                                        z1 = aa[1];
                                    if (aa[2] < aa[1] && aa[2] < aa[3])
                                        z1 = aa[2];
                                    if (aa[3] < aa[1] && aa[3] < aa[2])
                                        z1 = aa[3];
                                    if (ba[1] > ba[2] && ba[1] > ba[3])
                                        z2 = ba[1];
                                    if (ba[2] > ba[1] && ba[2] > ba[3])
                                        z2 = ba[2];
                                    if (ba[3] > ba[1] && ba[3] > ba[2])
                                        z2 = ba[3];
                                    if (z1 == 1 || z2 == 6)
                                        break;
                                    if (fa[z1 - 1][z2 + 1] > 0)
                                        break;
                                    if (fa[z1][z2 + 1] > 0 && fa[z1][z2 + 1] == fa[z1 - 1][z2])
                                        break;
                                    aa[k + 1] = z1 - 1;
                                    ba[k + 1] = z2 + 1;
                                }
                            }
                            if (k <= n)
                                continue;
                            fa[a][b] = 9 - 2 * i - j;
                            for (k = 1; k <= n; k++)
                                fa[aa[k + 1]][ba[k + 1]] = fa[a][b];
                            break;
                    }
                    break;
                }
            }
        }
        print("\n");
        print("THE FOLLOWING CODE OF THE BAD GUYS' FLEET DISPOSITION\n");
        print("HAS BEEN CAPTURED BUT NOT DECODED:\n");
        print("\n");
        for (i = 1; i <= 6; i++) {
            for (j = 1; j <= 6; j++) {
                ha[i][j] = fa[j][i];
            }
        }
        for (i = 1; i <= 6; i++) {
            str = "";
            for (j = 1; j <= 6; j++) {
                str += " " + ha[i][j] + " ";
            }
            print(str + "\n");
        }
        print("\n");
        print("DE-CODE IT AND USE IT IF YOU CAN\n");
        print("BUT KEEP THE DE-CODING METHOD A SECRET.\n");
        print("\n");
        for (i = 1; i <= 6; i++) {
            for (j = 1; j <= 6; j++) {
                ha[i][j] = 0;
            }
        }
        for (i = 1; i <= 3; i++)
            la[i] = 0;
        ca[1] = 2;
        ca[2] = 2;
        ca[3] = 1;
        ca[4] = 1;
        ca[5] = 0;
        ca[6] = 0;
        s = 0;
        h = 0;
        print("START GAME\n");
        while (1) {
            str = await input();
            // Check if user types anything other than a number
            if (isNaN(str)) {
                print("INVALID INPUT. TRY ENTERING A NUMBER INSTEAD.\n");
                continue;
            }
            x = parseInt(str);
            y = parseInt(str.substr(str.indexOf(",") + 1));
            if (x < 1 || x > 6 || y < 1 || y > 6) {
                print("INVALID INPUT.  TRY AGAIN.\n");
                continue;
            }
            r = 7 - y;
            c = x;
            if (fa[r][c] <= 0) {
                s++;
                print("SPLASH!  TRY AGAIN.\n");
                continue;
            }
            if (ca[fa[r][c]] >= 4) {
                print("THERE USED TO BE A SHIP AT THAT POINT, BUT YOU SUNK IT.\n");
                print("SPLASH!  TRY AGAIN.\n");
                s++;
                continue;
            }
            if (ha[r][c] > 0) {
                print("YOU ALREADY PUT A HOLE IN SHIP NUMBER " + fa[r][c] + " AT THAT POINT.\n");
                print("SPLASH!  TRY AGAIN.\n");
                s++;
                continue;
            }
            h++;
            ha[r][c] = fa[r][c];
            print("A DIRECT HIT ON SHIP NUMBER " + fa[r][c] + "\n");
            ca[fa[r][c]]++;
            if (ca[fa[r][c]] < 4) {
                print("TRY AGAIN.\n");
                continue;
            }
            la[Math.floor((fa[r][c] - 1) / 2) + 1]++;
            print("AND YOU SUNK IT.  HURRAH FOR THE GOOD GUYS.\n");
            print("SO FAR, THE BAD GUYS HAVE LOST\n");
            print(" " + la[1] + " DESTROYER(S), " + la[2] + " CRUISER(S), AND");
            print(" " + la[3] + " AIRCRAFT CARRIER(S).\n");
            print("YOUR CURRENT SPLASH/HIT RATIO IS " + s / h + "\n");
            if (la[1] + la[2] + la[3] < 6)
                continue;
            print("\n");
            print("YOU HAVE TOTALLY WIPED OUT THE BAD GUYS' FLEET\n");
            print("WITH A FINAL SPLASH/HIT RATIO OF " + s / h + "\n");
            if (s / h <= 0) {
                print("CONGRATULATIONS -- A DIRECT HIT EVERY TIME.\n");
            }
            print("\n");
            print("****************************\n");
            print("\n");
            break;
        }
    }
}

```

这是经典的 "Hello, World!" 程序，用于在 Unix 和类 Unix 系统的程序中打印出 "Hello, World!" 消息。

具体来说，这个程序的作用是调用 bios 命令，在命令行界面上输出 "Hello, World!" 消息。在 Linux 和 macOS 中，`main()` 函数通常是指程序的入口点，这个函数可以进行任何需要的初始化操作，然后执行程序的主要操作。在这个例子中，`main()` 函数调用 `system()` 函数来运行 `bioc` 命令，然后输出 "Hello, World!" 消息。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language))


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `09_Battle/python/battle.py`

这段代码定义了一个名为SEA_WIDTH的常量，一个名为DESTROYER_LENGTH的常量，一个名为CRUISER_LENGTH的常量，以及一个名为AIRCRAFT_CARRIER_LENGTH的常量。

接着，它从random库中引入了randrange函数，用于生成随机整数。

然后，定义了一个PointType类型的变量，表示一个两个整数的矢量。

接着，定义了一个VectorType类型的变量，表示一个两个整数的矢量。

接着，定义了一个SeaType类型的变量，表示由多个整数的列表组成。

在函数random_vector()中，使用了两个while循环。第一个while循环在生成随机矢量后，检查生成的矢量是否为零，如果不是，则说明产生了有效的矢量，然后返回该矢量。

在main函数中，使用了该函数生成一个长度为6，宽度为2的海上航行船，一个长度为2的破坏者，一个长度为3的巡航船，和一个长度为4的客机，并打印出了相关信息。


```
#!/usr/bin/env python3
from random import randrange
from typing import List, Tuple

PointType = Tuple[int, int]
VectorType = PointType
SeaType = Tuple[List[int], ...]

SEA_WIDTH = 6
DESTROYER_LENGTH = 2
CRUISER_LENGTH = 3
AIRCRAFT_CARRIER_LENGTH = 4


def random_vector() -> Tuple[int, int]:
    while True:
        vector = (randrange(-1, 2), randrange(-1, 2))

        if vector == (0, 0):
            # We can't have a zero vector, so try again
            continue

        return vector


```

这段代码定义了两个函数，`add_vector()` 和 `place_ship()`。这两个函数的功能是：

1. `add_vector()` 函数接收两个 `PointType` 类型的参数 `point` 和 `vector`，并返回一个新的 `PointType` 类型的结果。`add_vector()` 函数通过 `point` 和 `vector` 的 `x` 和 `y` 坐标之和来生成一个新的向量。
2. `place_ship()` 函数接收一个 `SeaType` 类型的参数 `sea`、一个 `int` 类型的参数 `size`，并返回 `None`，即没有返回任何值。`place_ship()` 函数会在 `while` 循环中随机生成一个起点，然后生成 `size` 个点，其中每个点都按照 `add_vector()` 函数生成。接下来，函数会遍历所有点，检查它们是否在 `sea` 中，如果是，则尝试将该点设置为 `sea` 中的值。最后，如果所有点都在 `sea` 中，则生成位置的所有值都设置为 `sea` 中的值，这样就会将该点放置在正确的位置上。


```
def add_vector(point: PointType, vector: VectorType) -> PointType:
    return (point[0] + vector[0], point[1] + vector[1])


def place_ship(sea: SeaType, size: int, code: int) -> None:
    while True:
        start = (randrange(1, SEA_WIDTH + 1), randrange(1, SEA_WIDTH + 1))
        vector = random_vector()

        # Get potential ship points
        point = start
        points = []

        for _ in range(size):
            point = add_vector(point, vector)
            points.append(point)

        if not all([is_within_sea(point, sea) for point in points]) or any(
            [value_at(point, sea) for point in points]
        ):
            # ship out of bounds or crosses other ship, trying again
            continue

        # We found a valid spot, so actually place it now
        for point in points:
            set_value_at(code, point, sea)

        break


```

该代码是一个 Python 语言编写的函数和数据类型。具体解释如下：
```python
def print_encoded_sea(sea: SeaType) -> None:
   for x in range(len(sea)):
       print(" ".join([str(sea[y][x]) for y in range(len(sea) - 1, -1, -1)]))
```
该函数接收一个 SeaType 类型的参数 sea，并打印出 sea 中每个位置的船的位置信息。通过遍历 sea 中的每个位置，将 sea 中每个位置的字符串连接成一个空格字符串，然后将连接好的字符串打印出来。
```kotlin
def is_within_sea(point: PointType, sea: SeaType) -> bool:
   return (1 <= point[0] <= len(sea)) and (1 <= point[1] <= len(sea))
```
该函数接收一个 PointType 类型的参数 point 和一个 SeaType 类型的参数 sea。函数返回 point 的[0]和[1]坐标值是否在 sea 中。
```less
def has_ship(sea: SeaType, code: int) -> bool:
   return any(code in row for row in sea)
```
该函数接收一个 SeaType 类型的参数 sea 和一个 int 类型的参数 code。函数返回将这些代码是否在 sea 的任何一行中。
```less
def count_sunk(sea: SeaType, *codes: int) -> int:
   return sum(not has_ship(sea, codes[i]) for i in range(*codes))
```
该函数接收一个 SeaType 类型的参数 sea 和一个或多个 int 类型的参数 codes。函数返回将 sea 中所有被 codes[i] 所代替的船的数量相加。


```
def print_encoded_sea(sea: SeaType) -> None:
    for x in range(len(sea)):
        print(" ".join([str(sea[y][x]) for y in range(len(sea) - 1, -1, -1)]))


def is_within_sea(point: PointType, sea: SeaType) -> bool:
    return (1 <= point[0] <= len(sea)) and (1 <= point[1] <= len(sea))


def has_ship(sea: SeaType, code: int) -> bool:
    return any(code in row for row in sea)


def count_sunk(sea: SeaType, *codes: int) -> int:
    return sum(not has_ship(sea, code) for code in codes)


```

该代码定义了两个函数，分别用于计算一个点在一个海底面上的值以及在海底面上的一个点的下一个目标点。

首先，我们来看 `value_at` 函数。它接收一个 `PointType` 类型的点坐标和一个 `SeaType` 类型的海底面类型。通过调用 `sea` 中的四个元素（即该点所在海底面上的四个元素），它返回了该点对应的海底面上的一个整数。

接下来，我们来看 `set_value_at` 函数。它接收一个 `int` 类型的目标值，一个 `PointType` 类型的点坐标和一个 `SeaType` 类型的海底面类型。它将目标值设置为海底面上对应点的整数部分。

最后，我们来看 `get_next_target` 函数。它接收一个 `SeaType` 类型的海底面类型，并在海底面上循环查找一个点，直到找到一个目标点或者循环完所有海底面。它返回这个目标点的坐标，如果不存在，则返回原点。


```
def value_at(point: PointType, sea: SeaType) -> int:
    return sea[point[1] - 1][point[0] - 1]


def set_value_at(value: int, point: PointType, sea: SeaType) -> None:
    sea[point[1] - 1][point[0] - 1] = value


def get_next_target(sea: SeaType) -> PointType:
    while True:
        try:
            guess = input("? ")
            point_str_list = guess.split(",")

            if len(point_str_list) != 2:
                raise ValueError()

            point = (int(point_str_list[0]), int(point_str_list[1]))

            if not is_within_sea(point, sea):
                raise ValueError()

            return point
        except ValueError:
            print(
                f"INVALID. SPECIFY TWO NUMBERS FROM 1 TO {len(sea)}, SEPARATED BY A COMMA."
            )


```

This code appears to be a script for a game of "warcraft", specifically the "Battlegrounds" game mode. It defines a function called "setup\_ships" that takes a single argument "sea", which is a tuple of ship types (e.g. "Mechanic", "Destroyer", "Cruiser", "Aircraft Carrier").

The function "setup\_ships" prints a series of ship placing statements, which place指定 number of units of the specified ship type in the given "sea" tuple at the center of the map. The numbers 1-4 and 6 place the corresponding units of the Destroyer and Cruiser, while the number 5 and 6 place the corresponding units of the Aircraft Carrier.

The main function calls the "setup\_ships" function with a tuple of all zeroes as an argument, which effectively clear the map of all units.


```
def setup_ships(sea: SeaType) -> None:
    place_ship(sea, DESTROYER_LENGTH, 1)
    place_ship(sea, DESTROYER_LENGTH, 2)
    place_ship(sea, CRUISER_LENGTH, 3)
    place_ship(sea, CRUISER_LENGTH, 4)
    place_ship(sea, AIRCRAFT_CARRIER_LENGTH, 5)
    place_ship(sea, AIRCRAFT_CARRIER_LENGTH, 6)


def main() -> None:
    sea = tuple([0 for _ in range(SEA_WIDTH)] for _ in range(SEA_WIDTH))
    setup_ships(sea)
    print(
        """
                BATTLE
```

TOTALLY WIPED OUT THE BAD GUYS' FLEET WITH A
Splashes += 1

This statement is a part of the code that keeps the de-coding method a secret. It does not seem to have any significant impact on the game, but it is included here for clarity.


```
CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY

THE FOLLOWING CODE OF THE BAD GUYS' FLEET DISPOSITION
HAS BEEN CAPTURED BUT NOT DECODED:

"""
    )
    print_encoded_sea(sea)
    print(
        """

DE-CODE IT AND USE IT IF YOU CAN
BUT KEEP THE DE-CODING METHOD A SECRET.

START GAME"""
    )
    splashes = 0
    hits = 0

    while True:
        target = get_next_target(sea)
        target_value = value_at(target, sea)

        if target_value < 0:
            print(
                f"YOU ALREADY PUT A HOLE IN SHIP NUMBER {abs(target_value)} AT THAT POINT."
            )

        if target_value <= 0:
            print("SPLASH! TRY AGAIN.")
            splashes += 1
            continue

        print(f"A DIRECT HIT ON SHIP NUMBER {target_value}")
        hits += 1
        set_value_at(-target_value, target, sea)

        if not has_ship(sea, target_value):
            print("AND YOU SUNK IT. HURRAH FOR THE GOOD GUYS.")
            print("SO FAR, THE BAD GUYS HAVE LOST")
            print(
                f"{count_sunk(sea, 1, 2)} DESTROYER(S),",
                f"{count_sunk(sea, 3, 4)} CRUISER(S),",
                f"AND {count_sunk(sea, 5, 6)} AIRCRAFT CARRIER(S).",
            )

        if any(has_ship(sea, code) for code in range(1, 7)):
            print(f"YOUR CURRENT SPLASH/HIT RATIO IS {splashes}/{hits}")
            continue

        print(
            "YOU HAVE TOTALLY WIPED OUT THE BAD GUYS' FLEET "
            f"WITH A FINAL SPLASH/HIT RATIO OF {splashes}/{hits}"
        )

        if not splashes:
            print("CONGRATULATIONS -- A DIRECT HIT EVERY TIME.")

        print("\n****************************")
        break


```

这段代码是一个条件判断语句，它的作用是在程序运行时判断是否是作为主程序运行。如果程序被作为主程序运行，那么程序会执行if语句块内的内容。

if __name__ == "__main__":
   main()

这段代码中包含两个部分。第一部分是一个条件判断语句，如果程序被作为主程序运行（即 "__main__" 等于 "__main__"），那么程序会执行 if 语句块内的内容。第二部分是一个函数 main()，它是程序的主函数，程序通常会在这里调用它来执行具体的任务。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Blackjack

This is a simulation of the card game of Blackjack or 21, Las Vegas style. This rather comprehensive version allows for up to seven players. On each hand a player may get another card (a hit), stand, split a hand in the event two identical cards were received or double down. Also, the dealer will ask for an insurance bet if he has an exposed ace.

Cards are automatically reshuffled as the 51st card is reached. For greater realism, you may wish to change this to the 41st card. Actually, fanatical purists will want to modify the program so it uses three decks of cards instead of just one.

This program originally surfaced at Digital Equipment Corp.; the author is unknown.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=18)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=33)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html


#### Porting Notes

The program makes extensive use of the assumption that a boolean expression evaluates to **-1** for true.  This was the case in some classic BASIC environments but not others; and it is not the case in [JS Basic](https://troypress.com/wp-content/uploads/user/js-basic/index.html), leading to nonsensical results.  In an environment that uses **1** instead of **-1** for truth, you would need to negate the boolean expression in the following lines:
	- 10
	- 570
	- 590
	- 2220
	- 2850
	- 3100
	- 3400
	- 3410
	- 3420


# `10_Blackjack/csharp/Card.cs`

这段代码定义了一个名为 "Card" 的类，用于表示扑克牌中的一个单牌。

在 "Card" 类中，定义了一个名为 "Index" 的静态字段，用于存储单牌的索引，从0开始。

在 "Card" 类中，还定义了一个名为 "Name" 的静态字段，用于存储单牌的花色名称，由于每个花色在扑克牌中对应一个编号，因此单牌的名称就是扑克牌中的编号对应的花色名称。

在 "Card" 类中，定义了一个名为 "IndefiniteArticle" 的静态字段，用于存储单牌的不定冠词，如果单牌的索引为0或者为7，则使用不定冠词"an"或"a"。

在 "Card" 类中，定义了一个名为 "IsAce" 的静态字段，用于存储单牌是否为大王，如果单牌的索引为0，则默认判断为大王，否则根据索引判断。

在 "Card" 类中，定义了一个名为 "Value" 的静态字段，用于存储单牌的点数，如果单牌为大王，则默认点数为11，否则根据索引计算得到单牌的点数。


```
namespace Blackjack
{
    public class Card
    {
        private static readonly string[] _names = new[] {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"};

        public Card(int index)
        {
            Index = index;
        }

        public int Index { get; private set; }

        public string Name => _names[Index];

        public string IndefiniteArticle => (Index == 0 || Index == 7) ? "an" : "a";

        public bool IsAce => Index == 0;

        public int Value
        {
            get
            {
                if (IsAce)
                    return 11;
                if (Index > 8)
                    return 10;
                return Index + 1;
            }
        }
    }
}

```

# `10_Blackjack/csharp/Deck.cs`



这段代码是一个基于随机化的扑克牌程序。它包含一个扑克牌面，一个牌堆，以及一个洗牌函数。

牌面的初始化在代码中进行。使用了一个循环来遍历牌面的每个区域，并添加了4种花色(即红桃，方块，梅花，黑桃)的牌。然后使用Reshuffle函数来重新洗牌，将牌重新混匀。

牌堆的作用是保留最新的几张牌，以便在需要的时候进行出牌。Discard函数允许用户将几张牌丢弃，以从牌堆中抽出牌来。


```
using System;
using System.Collections.Generic;

namespace Blackjack
{
    public class Deck
    {
        private static readonly Random _random = new Random();

        private readonly List<Card> _cards = new List<Card>(52);
        private readonly List<Card> _discards = new List<Card>(52);

        public Deck()
        {
            for (var index = 0; index < 12; index++)
            {
                for (var suit = 0; suit < 4; suit++)
                {
                    _discards.Add(new Card(index));
                }
            }
            Reshuffle();
        }

        private void Reshuffle()
        {
            Console.WriteLine("Reshuffling");

            _cards.AddRange(_discards);
            _discards.Clear();

            for (var index1 = _cards.Count - 1; index1 > 0; index1--)
            {
                var index2 = _random.Next(0, index1);
                var swapCard = _cards[index1];
                _cards[index1] = _cards[index2];
                _cards[index2] = swapCard;
            }
        }

        public Card DrawCard()
        {
            if (_cards.Count < 2)
                Reshuffle();

            var card = _cards[_cards.Count - 1];
            _cards.RemoveAt(_cards.Count - 1);
            return card;
        }

        public void Discard(IEnumerable<Card> cards)
        {
            _discards.AddRange(cards);
        }
    }
}

```

# `10_Blackjack/csharp/Game.cs`

This appears to be a implementation of a command-line program that takes in a number of arguments for a game of cards. It appears to be written in C# and uses the naming convention "AppendixG".

The program appears to have several main functions including:

* `TallyResults()` which outputs the results of the game to the console, including the round wins for each player and the total win for the dealer.
* `CalculateWinnings()` which calculates the win for each player based on the cards they are playing with and updates the player's win total.
* `ResetRoundState()` which resets the state of the game for each player and the dealer.
* `string[]` which appears to be a generic type, but without more context it is unclear what it is used for.
* `System.Linq` which appears to be a linear query library for LINQ.

It appears that the program is intended to be used as a command-line tool, with the `-C` flag passed to it to indicate that it should be treated as a command.
It should be noted that the program is not complete and has a lot of missing things.



```
using System;
using System.Collections.Generic;
using System.Linq;

namespace Blackjack
{
    public class Game
    {
        private readonly Deck _deck = new Deck();
        private readonly int _numberOfPlayers;
        private readonly Player[] _players;
        private readonly Hand _dealerHand;

        public Game(int numberOfPlayers)
        {
            _numberOfPlayers = numberOfPlayers;
            _players = new Player[_numberOfPlayers];
            for (var playerIndex = 0; playerIndex < _numberOfPlayers; playerIndex++)
                _players[playerIndex] = new Player(playerIndex);
            _dealerHand = new Hand();
        }

        public void PlayGame()
        {
            while (true)
            {
                PlayRound();
                TallyResults();
                ResetRoundState();
                Console.WriteLine();
            }
        }

        public void PlayRound()
        {
            GetPlayerBets();

            DealHands();

            // Test for insurance
            var dealerIsShowingAce = _dealerHand.Cards[0].IsAce;
            if (dealerIsShowingAce && Prompt.ForYesNo("Any insurance?"))
            {
                Console.WriteLine("Insurance bets");
                var insuranceBets = new int[_numberOfPlayers];
                foreach (var player in _players)
                    insuranceBets[player.Index] = Prompt.ForInteger($"# {player.Index + 1} ?", 0, player.RoundBet / 2);

                var insuranceEffectMultiplier = _dealerHand.IsBlackjack ? 2 : -1;
                foreach (var player in _players)
                    player.RoundWinnings += insuranceBets[player.Index] * insuranceEffectMultiplier;
            }

            // Test for dealer blackjack
            var concealedCard = _dealerHand.Cards[0];
            if (_dealerHand.IsBlackjack)
            {
                Console.WriteLine();
                Console.WriteLine("Dealer has {0} {1} in the hole for blackjack.", concealedCard.IndefiniteArticle, concealedCard.Name);
                return;
            }
            else if (dealerIsShowingAce)
            {
                Console.WriteLine();
                Console.WriteLine("No dealer blackjack.");
            }

            foreach (var player in _players)
                PlayHand(player);

            // Dealer hand
            var allPlayersBusted = _players.All(p => p.Hand.IsBusted && (!p.SecondHand.Exists || p.SecondHand.IsBusted));
            if (allPlayersBusted)
                Console.WriteLine("Dealer had {0} {1} concealed.", concealedCard.IndefiniteArticle, concealedCard.Name);
            else
            {
                Console.WriteLine("Dealer has {0} {1} concealed for a total of {2}", concealedCard.IndefiniteArticle, concealedCard.Name, _dealerHand.Total);
                if (_dealerHand.Total < 17)
                {
                    Console.Write("Draws");
                    while (_dealerHand.Total < 17)
                    {
                        var card = _dealerHand.AddCard(_deck.DrawCard());
                        Console.Write("  {0}", card.Name);
                    }
                    if (_dealerHand.IsBusted)
                        Console.WriteLine("  ...Busted");
                    else
                        Console.WriteLine("  ---Total is {0}", _dealerHand.Total);
                }
            }
        }

        private void GetPlayerBets()
        {
            Console.WriteLine("Bets:");
            foreach (var player in _players)
                player.RoundBet = Prompt.ForInteger($"# {player.Name} ?", 1, 500);
        }

        private void DealHands()
        {
            Console.Write("Player ");
            foreach (var player in _players)
                Console.Write("{0}     ", player.Name);
            Console.WriteLine("Dealer");

            for (var cardIndex = 0; cardIndex < 2; cardIndex++)
            {
                Console.Write("      ");
                foreach (var player in _players)
                    Console.Write("  {0,-4}", player.Hand.AddCard(_deck.DrawCard()).Name);
                var dealerCard = _dealerHand.AddCard(_deck.DrawCard());
                Console.Write("  {0,-4}", (cardIndex == 0) ? "XX" : dealerCard.Name);

                Console.WriteLine();
            }
        }

        private void PlayHand(Player player)
        {
            var hand = player.Hand;

            Console.Write("Player {0} ", player.Name);

            var playerCanSplit = hand.Cards[0].Value == hand.Cards[1].Value;
            var command = Prompt.ForCommandCharacter("?", playerCanSplit ? "HSD/" : "HSD");
            switch (command)
            {
                case "D":
                    player.RoundBet *= 2;
                    goto case "H";

                case "H":
                    while (TakeHit(hand) && PromptForAnotherHit())
                    { }
                    if (!hand.IsBusted)
                        Console.WriteLine("Total is {0}", hand.Total);
                    break;

                case "S":
                    if (hand.IsBlackjack)
                    {
                        Console.WriteLine("Blackjack!");
                        player.RoundWinnings = (int)(1.5 * player.RoundBet + 0.5);
                        player.RoundBet = 0;
                    }
                    else
                        Console.WriteLine("Total is {0}", hand.Total);
                    break;

                case "/":
                    hand.SplitHand(player.SecondHand);
                    var card = hand.AddCard(_deck.DrawCard());
                    Console.WriteLine("First hand receives {0} {1}", card.IndefiniteArticle, card.Name);
                    card = player.SecondHand.AddCard(_deck.DrawCard());
                    Console.WriteLine("Second hand receives {0} {1}", card.IndefiniteArticle, card.Name);

                    for (int handNumber = 1; handNumber <= 2; handNumber++)
                    {
                        hand = (handNumber == 1) ? player.Hand : player.SecondHand;

                        Console.Write("Hand {0}", handNumber);
                        while (PromptForAnotherHit() && TakeHit(hand))
                        { }
                        if (!hand.IsBusted)
                            Console.WriteLine("Total is {0}", hand.Total);
                    }
                    break;
            }
        }

        private bool TakeHit(Hand hand)
        {
            var card = hand.AddCard(_deck.DrawCard());
            Console.Write("Received {0,-6}", $"{card.IndefiniteArticle} {card.Name}");
            if (hand.IsBusted)
            {
                Console.WriteLine("...Busted");
                return false;
            }
            return true;
        }

        private bool PromptForAnotherHit()
        {
            return String.Equals(Prompt.ForCommandCharacter(" Hit?", "HS"), "H");
        }

        private void TallyResults()
        {
            Console.WriteLine();
            foreach (var player in _players)
            {
                player.RoundWinnings += CalculateWinnings(player, player.Hand);
                if (player.SecondHand.Exists)
                    player.RoundWinnings += CalculateWinnings(player, player.SecondHand);
                player.TotalWinnings += player.RoundWinnings;

                Console.WriteLine("Player {0} {1,-6} {2,3}   Total= {3,5}",
                        player.Name,
                        (player.RoundWinnings > 0) ? "wins" : (player.RoundWinnings) < 0 ? "loses" : "pushes",
                        (player.RoundWinnings != 0) ? Math.Abs(player.RoundWinnings).ToString() : "",
                        player.TotalWinnings);
            }
            Console.WriteLine("Dealer's total= {0}", -_players.Sum(p => p.TotalWinnings));
        }

        private int CalculateWinnings(Player player, Hand hand)
        {
            if (hand.IsBusted)
                return -player.RoundBet;
            if (hand.Total == _dealerHand.Total)
                return 0;
            if (_dealerHand.IsBusted || hand.Total > _dealerHand.Total)
                return player.RoundBet;
            return -player.RoundBet;
        }

        private void ResetRoundState()
        {
            foreach (var player in _players)
            {
                player.RoundWinnings = 0;
                player.RoundBet = 0;
                player.Hand.Discard(_deck);
                player.SecondHand.Discard(_deck);
            }
            _dealerHand.Discard(_deck);
        }
    }
}

```

# `10_Blackjack/csharp/Hand.cs`



该代码定义了一个名为 Hand 的类，代表着一副扑克牌。这个 Hand 类包含了一些方法，用于添加、发牌和处理牌的流出，以及检查是否为黑杰克。下面是 Hand 类的详细解释：

- AddCard(添加一张牌到 Hand，并更新总点数)

```
public Card AddCard(Card card)
{
   _cards.Add(card);
   _cachedTotal = 0;
   return card;
}
```

这个方法允许你在 Hand 中添加一张新的扑克牌，并将加入的点数更新为 0。

- SplitHand(将一张牌从 Hand 分发到第二个 Hand，第二个 Hand 必须要有至少两个牌)

```
public void SplitHand(Hand secondHand)
{
   if (Count != 2 || secondHand.Count != 0)
       throw new InvalidOperationException();
   secondHand.AddCard(_cards[1]);
   _cards.RemoveAt(1);
   _cachedTotal = 0;
}
```

这个方法允许你在 Hand 中选择一张牌，然后从 Hand 中删除这张牌，并且将 Hand 中的点数更新为 0。然后，将选择这张牌的 Hand 中的第一个牌作为新的牌加入 Hand。

- Exists(检查 Hand 中是否有牌)

```
public bool Exists(int count)
{
   return _cards.Count > 0;
}
```

这个方法检查 Hand 中是否有牌，返回一个布尔值。

- Total(返回 Hand 中所有牌的总点数)

```
public int Total(int count)
{
   if (_cachedTotal == 0)
   {
       var aceCount = 0;
       foreach (var card in _cards)
       {
           _cachedTotal += card.Value;
           if (card.IsAce)
               aceCount++;
       }
       while (_cachedTotal > 21 && aceCount > 0)
       {
           _cachedTotal -= 10;
           aceCount--;
       }
   }
   return _cachedTotal;
}
```

这个方法返回 Hand 中所有牌的总点数，如果 Hand 中牌的数量不是 2，或者 Hand 中没有牌，那么这个方法将返回 0。

- IsBlackjack(检查是否为黑杰克)

```
public bool IsBlackjack(int count)
{
   return Total() == 21 && Count == 2;
}
```

这个方法返回是否为黑杰克，如果 Hand 中牌的数量等于 2，并且总点数等于 21，那么这个方法返回 true，否则返回 false。

- IsBusted(检查是否被击败)

```
public bool IsBusted(int count)
{
   return Total() > 21;
}
```

这个方法返回是否被击败，如果 Hand 中牌的数量大于 21，那么这个方法返回 true，否则返回 false。


```
using System;
using System.Collections.Generic;

namespace Blackjack
{
    public class Hand
    {
        private readonly List<Card> _cards = new List<Card>(12);
        private int _cachedTotal = 0;

        public Card AddCard(Card card)
        {
            _cards.Add(card);
            _cachedTotal = 0;
            return card;
        }

        public void Discard(Deck deck)
        {
            deck.Discard(_cards);
            _cards.Clear();
            _cachedTotal = 0;
        }

        public void SplitHand(Hand secondHand)
        {
            if (Count != 2 || secondHand.Count != 0)
                throw new InvalidOperationException();
            secondHand.AddCard(_cards[1]);
            _cards.RemoveAt(1);
            _cachedTotal = 0;
        }

        public IReadOnlyList<Card> Cards => _cards;

        public int Count => _cards.Count;

        public bool Exists => _cards.Count > 0;

        public int Total
        {
            get
            {
                if (_cachedTotal == 0)
                {
                    var aceCount = 0;
                    foreach (var card in _cards)
                    {
                        _cachedTotal += card.Value;
                        if (card.IsAce)
                            aceCount++;
                    }
                    while (_cachedTotal > 21 && aceCount > 0)
                    {
                        _cachedTotal -= 10;
                        aceCount--;
                    }
                }
                return _cachedTotal;
            }
        }

        public bool IsBlackjack => Total == 21 && Count == 2;

        public bool IsBusted => Total > 21;
    }
}

```