# BasicComputerGames源码解析 33

# `27_Civil_War/javascript/civilwar.js`

这段代码的作用是实现了一个简单的交互式输入文本的功能。它由两个函数组成：`input()` 和 `print()`。

1. `input()` 函数的作用是获取用户输入的文本内容。它通过创建一个 `<INPUT>` 元素，设置其 `type` 属性为 "text"，并设置其 `length` 属性为 "50"。然后将该元素添加到页面上，并设置其 focus 属性。接着，当用户按键时，该函数会监听 `keydown` 事件，当用户按下回车键时，它会在控制台输出输入的文本内容，并将其存储在 `input_str` 变量中。

2. `print()` 函数的作用是在控制台上打印字符串。它接受一个字符串参数 `str`，并将其添加到页面上文档中的 `#output` 元素中。

该代码的实现基于 Web 标准，使用了 HTML、CSS 和 JavaScript。


```
// CIVIL WAR
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

This appears to be a list of towns or cities with their names, one for each entry, separated by commas. The names are all written in lowercase with an underscore between them.

The first entry has the name "PULL RUN", followed by a comma and then the numbers 18000, 18500, and 1967. It suggests a location with those coordinates.

The second entry has the name "SHILOH", followed by a comma and then the numbers 40000, 44894, and 10699. It suggests a location with those coordinates.

The third entry has the name "SEVEN DAYS", followed by a comma and then the numbers 95000, 115000, and 20614. It suggests a location with those coordinates.

The fourth entry has the name "SECOND BULL RUN", followed by a comma and then the numbers 54000, 63000, and 10000. It suggests a location with those coordinates.

The fifth entry has the name "ANTIETAM", followed by a comma and then the numbers 40000, 50000, and 10000. It suggests a location with those coordinates.

The sixth entry has the name "FREDERICKSBURG", followed by a comma and then the numbers 75000, 120000, and 5377. It suggests a location with those coordinates.

The seventh entry has the name "MURFREESBORO", followed by a comma and then the numbers 38000, 45000, and 11000. It suggests a location with those coordinates.

The eighth entry has the name "CHANCELLORSVILLE", followed by a comma and then the numbers 32000, 90000, and 13000. It suggests a location with those coordinates.

The ninth entry has the name "VICKSBURG", followed by a comma and then the numbers 50000, 70000, and 12000. It suggests a location with those coordinates.

The tenth entry has the name "GETTYSBURG", followed by a comma and then the numbers 72500, 85000, and 20000. It suggests a location with those coordinates.

The eleventh entry has the name "CHICKAMAUGA", followed by a comma and then the numbers 66000, 60000, and 18000. It suggests a location with those coordinates.

The twelfth entry has the name "CHATTANOOGA", followed by a comma and then the numbers 37000, 60000, and 36700. It suggests a location with those coordinates.

The thirteenth entry has the name "SPOTSYLVANIA", followed by a comma and then the numbers 62000, 110000, and 17723. It suggests a location with those coordinates.

The fourteenth entry has the name "ATLANTA", followed by a comma and then the numbers 65000, 100000, and 8500. It suggests a location with those coordinates.

There are more entries but I don't have enough information to provide the full list.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// Historical data...can add more (strat., etc) by inserting
// data statements after appro. info, and adjusting read
//                      0 - C$     1-M1  2-M2  3-C1 4-C2 5-D
var historical_data = [,
                       ["BULL RUN",18000,18500,1967,2708,1],
                       ["SHILOH",40000.,44894.,10699,13047,3],
                       ["SEVEN DAYS",95000.,115000.,20614,15849,3],
                       ["SECOND BULL RUN",54000.,63000.,10000,14000,2],
                       ["ANTIETAM",40000.,50000.,10000,12000,3],
                       ["FREDERICKSBURG",75000.,120000.,5377,12653,1],
                       ["MURFREESBORO",38000.,45000.,11000,12000,1],
                       ["CHANCELLORSVILLE",32000,90000.,13000,17197,2],
                       ["VICKSBURG",50000.,70000.,12000,19000,1],
                       ["GETTYSBURG",72500.,85000.,20000,23000,3],
                       ["CHICKAMAUGA",66000.,60000.,18000,16000,2],
                       ["CHATTANOOGA",37000.,60000.,36700.,5800,2],
                       ["SPOTSYLVANIA",62000.,110000.,17723,18000,2],
                       ["ATLANTA",65000.,100000.,8500,3700,1]];
```

以下是 Python 代码实现：

```python
p1, p2, t1, t2, w, l, y, r1, bs

print("THE CONFEDERACY HAS WON " + w + " BATTLES AND LOST " + l + "\n")
print("THE UNION HAS WON THE WAR")

if (r1) {
   print("FOR THE " + (w + l + w0) + " BATTLES FOUGHT (EXCLUDING RERUNS)\n")
   print("    % OF ORIGINAL\t" + Math.floor(t1 + 0.5) + "\t" + Math.floor(t2 + 0.5) + "\n")
   print("SIMULATED LOSSES\t" + Math.floor(t1 + 0.5) + "\t" + Math.floor(t2 + 0.5) + "\n")
   print("    % OF ORIGINAL\t" + Math.floor(100 * (t1 / p1) + 0.5) + "\t" + Math.floor(100 * (t2 / p2) + 0.5) + "\n")
   if (bs != "YES") {
       print("\n");
       print("UNION INTELLIGENCE SUGGEST THAT THE SOUTH USED \n");
       print("STRATEGIES 1, 2, 3, 4 IN THE FOLLOWING PERCENTAGES\n");
       print(sa[1] + " " + sa[2] + " " + sa[3] + " " + sa[4] + "\n");
   }
} else {
   print("THE CONFEDERACY HAS WON THE WAR")
}
```

请注意，这段代码需要 `simu` 包支持。如果没有，请使用 `minicom` 包或终端模拟器来进行 `simu` 交互模式。

在 `simu` 包中运行 `simu` 命令时，需要提供 `-s` 参数指定 `simu` 服务器 IP 地址和端口号。例如，如果服务器在本地计算机上，可以使用以下命令：

```sql
simu -s 127.0.0.1:8080 -t
```

这将启动一个 `simu` 服务器，等待来自客户端的连接。

另外，您还可以使用 `graphviz` 包来绘制战争地图。首先，使用 `graphviz` 命令绘制 ` confederacy` 和 ` union` 之间的线：

```
graphviz
dot -Tpng2 -o confederacy.dot -p 0.01w -p 0.01l -p 0.01h -p 0.01z -p 0.01o
   -o union.dot -p 0.01w -p 0.01l -p 0.01h -p 0.01z -p 0.01o
```

然后在 `simu` 服务器上运行以下命令：

```sql
simu -s 127.0.0.1:8080 -t -p 0.01w -p 0.01l -p 0.01h -p 0.01z -p 0.01o
```

这将启动 `simu` 服务器并等待客户端连接。然后，在客户端连接后，客户端将可以使用 `graphviz` 命令绘制战争地图。

请注意，如果 `graphviz` 包在您的系统上不可用，您可能需要使用 `xdot` 包来绘制图形。


```
var sa = [];
var da = [];
var fa = [];
var ha = [];
var ba = [];
var oa = [];

// Main program
async function main()
{
    print(tab(26) + "CIVIL WAR\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // Original game design: Cram, Goodie, Hibbard Lexington H.S.
    // Modifications: G. Paul, R. Hess (Ties), 1973
    // Union info on likely confederate strategy
    sa[1] = 25;
    sa[2] = 25;
    sa[3] = 25;
    sa[4] = 25;
    d = Math.random();
    print("\n");
    print("DO YOU WANT INSTRUCTIONS");
    while (1) {
        str = await input();
        if (str == "YES" || str == "NO")
            break;
        print("YES OR NO -- \n");
    }
    if (str == "YES") {
        print("\n");
        print("\n");
        print("\n");
        print("\n");
        print("THIS IS A CIVIL WAR SIMULATION.\n");
        print("TO PLAY TYPE A RESPONSE WHEN THE COMPUTER ASKS.\n");
        print("REMEMBER THAT ALL FACTORS ARE INTERRELATED AND THAT YOUR\n");
        print("RESPONSES COULD CHANGE HISTORY. FACTS AND FIGURES USED ARE\n");
        print("BASED ON THE ACTUAL OCCURRENCE. MOST BATTLES TEND TO RESULT\n");
        print("AS THEY DID IN THE CIVIL WAR, BUT IT ALL DEPENDS ON YOU!!\n");
        print("\n");
        print("THE OBJECT OF THE GAME IS TO WIN AS MANY BATTLES AS ");
        print("POSSIBLE.\n");
        print("\n");
        print("YOUR CHOICES FOR DEFENSIVE STRATEGY ARE:\n");
        print("        (1) ARTILLERY ATTACK\n");
        print("        (2) FORTIFICATION AGAINST FRONTAL ATTACK\n");
        print("        (3) FORTIFICATION AGAINST FLANKING MANEUVERS\n");
        print("        (4) FALLING BACK\n");
        print(" YOUR CHOICES FOR OFFENSIVE STRATEGY ARE:\n");
        print("        (1) ARTILLERY ATTACK\n");
        print("        (2) FRONTAL ATTACK\n");
        print("        (3) FLANKING MANEUVERS\n");
        print("        (4) ENCIRCLEMENT\n");
        print("YOU MAY SURRENDER BY TYPING A '5' FOR YOUR STRATEGY.\n");
    }
    print("\n");
    print("\n");
    print("\n");
    print("ARE THERE TWO GENERALS PRESENT ");
    while (1) {
        print("(ANSWER YES OR NO)");
        bs = await input();
        if (bs == "YES") {
            d = 2;
            break;
        } else if (bs == "NO") {
            print("\n");
            print("YOU ARE THE CONFEDERACY.   GOOD LUCK!\n");
            print("\n");
            d = 1;
            break;
        }
    }
    print("SELECT A BATTLE BY TYPING A NUMBER FROM 1 TO 14 ON\n");
    print("REQUEST.  TYPE ANY OTHER NUMBER TO END THE SIMULATION.\n");
    print("BUT '0' BRINGS BACK EXACT PREVIOUS BATTLE SITUATION\n");
    print("ALLOWING YOU TO REPLAY IT\n");
    print("\n");
    print("NOTE: A NEGATIVE FOOD$ ENTRY CAUSES THE PROGRAM TO \n");
    print("USE THE ENTRIES FROM THE PREVIOUS BATTLE\n");
    print("\n");
    print("AFTER REQUESTING A BATTLE, DO YOU WISH ");
    print("BATTLE DESCRIPTIONS ");
    while (1) {
        print("(ANSWER YES OR NO)");
        xs = await input();
        if (xs == "YES" || xs == "NO")
            break;
    }
    l = 0;
    w = 0;
    r1 = 0;
    q1 = 0;
    m3 = 0;
    m4 = 0;
    p1 = 0;
    p2 = 0;
    t1 = 0;
    t2 = 0;
    for (i = 1; i <= 2; i++) {
        da[i] = 0;
        fa[i] = 0;
        ha[i] = 0;
        ba[i] = 0;
        oa[i] = 0;
    }
    r2 = 0;
    q2 = 0;
    c6 = 0;
    f = 0;
    w0 = 0;
    y = 0;
    y2 = 0;
    u = 0;
    u2 = 0;
    while (1) {
        print("\n");
        print("\n");
        print("\n");
        print("WHICH BATTLE DO YOU WISH TO SIMULATE");
        a = parseInt(await input());
        if (a < 1 || a > 14)
            break;
        if (a != 0 || r == 0) {
            cs = historical_data[a][0];
            m1 = historical_data[a][1];
            m2 = historical_data[a][2];
            c1 = historical_data[a][3];
            c2 = historical_data[a][4];
            m = historical_data[a][5];
            u = 0;
            // Inflation calc
            i1 = 10 + (l - w) * 2;
            i2 = 10 + (w - l) * 2;
            // Money available
            da[1] = 100 * Math.floor((m1 * (100 - i1) / 2000) * (1 + (r1 - q1) / (r1 + 1)) + 0.5);
            da[2] = 100 * Math.floor(m2 * (100 - i2) / 2000 + 0.5);
            if (bs == "YES") {
                da[2] = 100 * Math.floor((m2 * (100 - i2) / 2000) * (1 + (r2 - q2) / (r2 + 1)) + 0.5);
            }
            // Men available
            m5 = Math.floor(m1 * (1 + (p1 - t1) / (m3 + 1)));
            m6 = Math.floor(m2 * (1 + (p2 - t2) / (m4 + 1)));
            f1 = 5 * m1 / 6;
            print("\n");
            print("\n");
            print("\n");
            print("\n");
            print("\n");
            print("THIS IS THE BATTLE OF " + cs + "\n");
            if (xs != "NO") {
                switch (a) {
                    case 1:
                        print("JULY 21, 1861.  GEN. BEAUREGARD, COMMANDING THE SOUTH, MET\n");
                        print("UNION FORCES WITH GEN. MCDOWELL IN A PREMATURE BATTLE AT\n");
                        print("BULL RUN. GEN. JACKSON HELPED PUSH BACK THE UNION ATTACK.\n");
                        break;
                    case 2:
                        print("APRIL 6-7, 1862.  THE CONFEDERATE SURPRISE ATTACK AT\n");
                        print("SHILOH FAILED DUE TO POOR ORGANIZATION.\n");
                        break;
                    case 3:
                        print("JUNE 25-JULY 1, 1862.  GENERAL LEE (CSA) UPHELD THE\n");
                        print("OFFENSIVE THROUGHOUT THE BATTLE AND FORCED GEN. MCCLELLAN\n");
                        print("AND THE UNION FORCES AWAY FROM RICHMOND.\n");
                        break;
                    case 4:
                        print("AUG 29-30, 1862.  THE COMBINED CONFEDERATE FORCES UNDER LEE\n");
                        print("AND JACKSON DROVE THE UNION FORCES BACK INTO WASHINGTON.\n");
                        break;
                    case 5:
                        print("SEPT 17, 1862.  THE SOUTH FAILED TO INCORPORATE MARYLAND\n");
                        print("INTO THE CONFEDERACY.\n");
                        break;
                    case 6:
                        print("DEC 13, 1862.  THE CONFEDERACY UNDER LEE SUCCESSFULLY\n");
                        print("REPULSED AN ATTACK BY THE UNION UNDER GEN. BURNSIDE.\n");
                        break;
                    case 7:
                        print("DEC 31, 1862.  THE SOUTH UNDER GEN. BRAGG WON A CLOSE BATTLE.\n");
                        break;
                    case 8:
                        print("MAY 1-6, 1863.  THE SOUTH HAD A COSTLY VICTORY AND LOST\n");
                        print("ONE OF THEIR OUTSTANDING GENERALS, 'STONEWALL' JACKSON.\n");
                        break;
                    case 9:
                        print("JULY 4, 1863.  VICKSBURG WAS A COSTLY DEFEAT FOR THE SOUTH\n");
                        print("BECAUSE IT GAVE THE UNION ACCESS TO THE MISSISSIPPI.\n");
                        break;
                    case 10:
                        print("JULY 1-3, 1863.  A SOUTHERN MISTAKE BY GEN. LEE AT GETTYSBURG\n");
                        print("COST THEM ONE OF THE MOST CRUCIAL BATTLES OF THE WAR.\n");
                        break;
                    case 11:
                        print("SEPT. 15, 1863. CONFUSION IN A FOREST NEAR CHICKAMAUGA LED\n");
                        print("TO A COSTLY SOUTHERN VICTORY.\n");
                        break;
                    case 12:
                        print("NOV. 25, 1863. AFTER THE SOUTH HAD SIEGED GEN. ROSENCRANS'\n");
                        print("ARMY FOR THREE MONTHS, GEN. GRANT BROKE THE SIEGE.\n");
                        break;
                    case 13:
                        print("MAY 5, 1864.  GRANT'S PLAN TO KEEP LEE ISOLATED BEGAN TO\n");
                        print("FAIL HERE, AND CONTINUED AT COLD HARBOR AND PETERSBURG.\n");
                        break;
                    case 14:
                        print("AUGUST, 1864.  SHERMAN AND THREE VETERAN ARMIES CONVERGED\n");
                        print("ON ATLANTA AND DEALT THE DEATH BLOW TO THE CONFEDERACY.\n");
                        break;
                }
            }
        } else {
            print(cs + " INSTANT REPLAY\n");
        }
        print("\n");
        print(" \tCONFEDERACY\t UNION\n"),
        print("MEN\t  " + m5 + "\t\t " + m6 + "\n");
        print("MONEY\t $" + da[1] + "\t\t$" + da[2] + "\n");
        print("INFLATION\t " + (i1 + 15) + "%\t " + i2 + "%\n");
        print("\n");
        // ONLY IN PRINTOUT IS CONFED INFLATION = I1 + 15%
        // IF TWO GENERALS, INPUT CONFED, FIRST
        for (i = 1; i <= d; i++) {
            if (bs == "YES" && i == 1)
                print("CONFEDERATE GENERAL---");
            print("HOW MUCH DO YOU WISH TO SPEND FOR\n");
            while (1) {
                print(" - FOOD......");
                f = parseInt(await input());
                if (f < 0) {
                    if (r1 == 0) {
                        print("NO PREVIOUS ENTRIES\n");
                        continue;
                    }
                    print("ASSUME YOU WANT TO KEEP SAME ALLOCATIONS\n");
                    print("\n");
                    break;
                }
                fa[i] = f;
                while (1) {
                    print(" - SALARIES..");
                    ha[i] = parseInt(await input());
                    if (ha[i] >= 0)
                        break;
                    print("NEGATIVE VALUES NOT ALLOWED.\n");
                }
                while (1) {
                    print(" - AMMUNITION");
                    ba[i] = parseInt(await input());
                    if (ba[i] >= 0)
                        break;
                    print("NEGATIVE VALUES NOT ALLOWED.\n");
                }
                print("\n");
                if (fa[i] + ha[i] + ba[i] > da[i]) {
                    print("THINK AGAIN! YOU HAVE ONLY $" + da[i] + "\n");
                } else {
                    break;
                }
            }
            if (bs != "YES" || i == 2)
                break;
            print("UNION GENERAL---");
        }
        for (z = 1; z <= d; z++) {
            if (bs == "YES") {
                if (z == 1)
                    print("CONFEDERATE ");
                else
                    print("      UNION ");
            }
            // Find morale
            o = ((2 * Math.pow(fa[z], 2) + Math.pow(ha[z], 2)) / Math.pow(f1, 2) + 1);
            if (o >= 10) {
                print("MORALE IS HIGH\n");
            } else if (o >= 5) {
                print("MORALE IS FAIR\n");
            } else {
                print("MORALE IS POOR\n");
            }
            if (bs != "YES")
                break;
            oa[z] = o;
        }
        o2 = oa[2];
        o = oa[1];
        print("CONFEDERATE GENERAL---");
        // Actual off/def battle situation
        if (m == 3) {
            print("YOU ARE ON THE OFFENSIVE\n");
        } else if (m == 1) {
            print("YOU ARE ON THE DEFENSIVE\n");
        } else {
            print("BOTH SIDES ARE ON THE OFFENSIVE \n");
        }
        print("\n");
        // Choose strategies
        if (bs != "YES") {
            print("YOUR STRATEGY ");
            while (1) {
                y = parseInt(await input());
                if (Math.abs(y - 3) < 3)
                    break;
                print("STRATEGY " + y + " NOT ALLOWED.\n");
            }
            if (y == 5) {
                print("THE CONFEDERACY HAS SURRENDERED.\n");
                break;
            }
            // Union strategy is computer choesn
            print("UNION STRATEGY IS ");
            if (a == 0) {
                while (1) {
                    y2 = parseInt(await input());
                    if (y2 > 0 && y2 < 5)
                        break;
                    print("ENTER 1, 2, 3, OR 4 (USUALLY PREVIOUS UNION STRATEGY)\n");
                }
            } else {
                s0 = 0;
                r = Math.random() * 100;
                for (i = 1; i <= 4; i++) {
                    s0 += sa[i];
                    // If actual strategy info is in program data statements
                    // then r-100 is extra weight given to that strategy.
                    if (r < s0)
                        break;
                }
                y2 = i;
                print(y2 + "\n");
            }
        } else {
            for (i = 1; i <= 2; i++) {
                if (i == 1)
                    print("CONFEDERATE STRATEGY ");
                while (1) {
                    y = parseInt(await input());
                    if (Math.abs(y - 3) < 3)
                        break;
                    print("STRATEGY " + y + " NOT ALLOWED.\n");
                }
                if (i == 2) {
                    y2 = y;
                    y = y1;
                    if (y2 != 5)
                        break;
                } else {
                    y1 = y;
                }
                print("UNION STRATEGY ");
            }
            // Simulated losses - North
            c6 = (2 * c2 / 5) * (1 + 1 / (2 * (Math.abs(y2 - y) + 1)));
            c6 = c6 * (1.28 + (5 * m2 / 6) / (ba[2] + 1));
            c6 = Math.floor(c6 * (1 + 1 / o2) + 0.5);
            // If loss > men present, rescale losses
            e2 = 100 / o2;
            if (Math.floor(c6 + e2) >= m6) {
                c6 = Math.floor(13 * m6 / 20);
                e2 = 7 * c6 / 13;
                u2 = 1;
            }
        }
        // Calculate simulated losses
        print("\n");
        print("\n");
        print("\n");
        print("\t\tCONFEDERACY\tUNION\n");
        c5 = (2 * c1 / 5) * (1 + 1 / (2 * (Math.abs(y2 - y) + 1)));
        c5 = Math.floor(c5 * (1 + 1 / o) * (1.28 + f1 / (ba[1] + 1)) + 0.5);
        e = 100 / o;
        if (c5 + 100 / o >= m1 * (1 + (p1 - t1) / (m3 + 1))) {
            c5 = Math.floor(13 * m1 / 20 * (1 + (p1 - t1) / (m3 + 1)));
            e = 7 * c5 / 13;
            u = 1;
        }
        if (d == 1) {
            c6 = Math.floor(17 * c2 * c1 / (c5 * 20));
            e2 = 5 * o;
        }
        print("CASUALTIES\t" + c5 + "\t\t" + c6 + "\n");
        print("DESERTIONS\t" + Math.floor(e) + "\t\t" + Math.floor(e2) + "\n");
        print("\n");
        if (bs == "YES") {
            print("COMPARED TO THE ACTUAL CASUALTIES AT " + cs + "\n");
            print("CONFEDERATE: " + Math.floor(100 * (c5 / c1) + 0.5) + "% OF THE ORIGINAL\n");
            print("UNION:       " + Math.floor(100 * (c6 / c2) + 0.5) + "% OF THE ORIGINAL\n");
        }
        print("\n");
        // 1 Who one
        if (u == 1 && u2 == 1 || (u != 1 && u2 != 1 && c5 + e == c6 + e2)) {
            print("BATTLE OUTCOME UNRESOLVED\n");
            w0++;
        } else if (u == 1 || (u != 1 && u2 != 1 && c5 + e > c6 + e2)) {
            print("THE UNION WINS " + cs + "\n");
            if (a != 0)
                l++;
        } else  {
            print("THE CONFEDERACY WINS " + cs + "\n");
            if (a != 0)
                w++;
        }
        // Lines 2530 to 2590 from original are unreachable.
        if (a != 0) {
            t1 += c5 + e;
            t2 += c6 + e2;
            p1 += c1;
            p2 += c2;
            q1 += fa[1] + ha[1] + ba[1];
            q2 += fa[2] + ha[2] + ba[2];
            r1 += m1 * (100 - i1) / 20;
            r2 += m2 * (100 - i2) / 20;
            m3 += m1;
            m4 += m2;
            // Learn present strategy, start forgetting old ones
            // present startegy of south gains 3*s, others lose s
            // probability points, unless a strategy falls below 5%.
            s = 3;
            s0 = 0;
            for (i = 1; i <= 4; i++) {
                if (sa[i] <= 5)
                    continue;
                sa[i] -= 5;
                s0 += s;
            }
            sa[y] += s0;
        }
        u = 0;
        u2 = 0;
        print("---------------");
        continue;
    }
    print("\n");
    print("\n");
    print("\n");
    print("\n");
    print("\n");
    print("\n");
    print("THE CONFEDERACY HAS WON " + w + " BATTLES AND LOST " + l + "\n");
    if (y == 5 || (y2 != 5 && w <= l)) {
        print("THE UNION HAS WON THE WAR\n");
    } else {
        print("THE CONFEDERACY HAS WON THE WAR\n");
    }
    print("\n");
    if (r1) {
        print("FOR THE " + (w + l + w0) + " BATTLES FOUGHT (EXCLUDING RERUNS)\n");
        print(" \t \t ");
        print("CONFEDERACY\t UNION\n");
        print("HISTORICAL LOSSES\t" + Math.floor(p1 + 0.5) + "\t" + Math.floor(p2 + 0.5) + "\n");
        print("SIMULATED LOSSES\t" + Math.floor(t1 + 0.5) + "\t" + Math.floor(t2 + 0.5) + "\n");
        print("\n");
        print("    % OF ORIGINAL\t" + Math.floor(100 * (t1 / p1) + 0.5) + "\t" + Math.floor(100 * (t2 / p2) + 0.5) + "\n");
        if (bs != "YES") {
            print("\n");
            print("UNION INTELLIGENCE SUGGEST THAT THE SOUTH USED \n");
            print("STRATEGIES 1, 2, 3, 4 IN THE FOLLOWING PERCENTAGES\n");
            print(sa[1] + " " + sa[2] + " " + sa[3] + " " + sa[4] + "\n");
        }
    }
}

```

这道题的代码是 `main()`，这是一个程序的入口函数。在大多数程序中，它用于将程序从命令行或控制台接收开始，并提供一些必要的说明和选项。

`main()` 函数通常包含以下内容：

1. 程序的路径：告诉操作系统程序在哪个目录下如何找到它。
2. 程序的参数：如果用户提供了参数，这个函数可以用作参数传递给程序。
3. 程序的欢迎消息：告诉用户程序的名称和版本，或者提供一些关于程序的更多信息。
4. 程序的退出码：告诉操作系统程序成功执行完度的状态码。

`main()` 函数的实现是程序的重要组成部分，因为它可以作为程序和用户的交互点，使程序更加易于使用和维护。


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

Conversion to [Perl](https://www.perl.org/)


# `27_Civil_War/python/Civilwar.py`

这段代码定义了一个名为 "AttackState" 的枚举类型，它有三种状态：DEFENSIVE、BOTH_OFFENSIVE 和 OFFENSIVE。

AttackState的定义是为了描述游戏中攻击的类型，以帮助开发人员更好地理解和设计游戏。

此外，import了两个类：enum 和 math，以及两个函数：random 和 None， random用于生成随机数，math用于数学计算。

import还可能导入其他模块或库，具体取决于游戏的设计和需要。


```
"""
Original game design: Cram, Goodie, Hibbard Lexington H.S.
Modifications: G. Paul, R. Hess (Ties), 1973
"""
import enum
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple


class AttackState(enum.Enum):
    DEFENSIVE = 1
    BOTH_OFFENSIVE = 2
    OFFENSIVE = 3


```

这段代码定义了一个名为PlayerStat的类，用于表示游戏中的玩家数据。这个类包括玩家的属性(如食物、薪水、弹药、逃跑人数、伤亡人数、士气、策略、可用人数、可用金钱)以及一些计算结果(如军队数量、军队士气、伤亡人数、累计成本)，用于游戏中的策略决策和游戏统计。

下面是PlayerStat类中一些重要的函数和类：

- `__init__`函数：初始化玩家的属性，包括食物、薪水、弹药、逃跑人数、伤亡人数、士气、策略、可用人数、可用金钱，以及军队数量、军队士气、伤亡人数、累计成本。其中，军队数量和士气是根据军队策略计算出来的。
- `set_available_money`函数：设置玩家的可用金钱。这个函数考虑了通货膨胀对可用金钱的影响，根据通货膨胀率和可用军队数量计算出可用金钱。
- `get_cost`函数：计算玩家损伤的总成本。这个函数根据食物、薪水、弹药、逃跑人数、伤亡人数、可用人数、可用金钱计算出损伤的总成本。
- `get_army_factor`函数：计算军队所需的因子。这个函数根据死亡人数、平均死亡人数、剩余人数计算出军队所需的因子。
- `get_present_men`函数：计算当前军队的人数。这个函数根据军队策略计算出当前军队的人数。
- `excessive_losses`变量：判断当前损失是否超过可承受范围，用于控制游戏中的策略决策。这个变量可以根据食物的剩余数量判断。
- `is_player`变量：标记玩家身份，值为False。这个变量可以在游戏中根据玩家行动决定是否执行某些操作。
- `excessive_losses`变量：判断当前损失是否超过可承受范围，用于控制游戏中的策略决策。这个变量可以根据食物的剩余数量判断。
- `inflation`变量：记录当前通货膨胀率。这个变量可以根据玩家行为和游戏设置进行更新。

PlayerStat类表示了一个游戏中的玩家，可以进行各种操作来控制游戏进程，包括移动、探索、攻击、防守等。这个类提供了很多游戏中的重要变量和计算结果，可以帮助玩家制定游戏策略并进行游戏统计。


```
CONF = 1
UNION = 2


@dataclass
class PlayerStat:
    food: float = 0
    salaries: float = 0
    ammunition: float = 0

    desertions: float = 0
    casualties: float = 0
    morale: float = 0
    strategy: int = 0
    available_men: int = 0
    available_money: int = 0

    army_c: float = 0
    army_m: float = 0  # available_men ????
    inflation: float = 0

    r: float = 0
    t: float = 0  # casualties + desertions
    q: float = 0  # accumulated cost?
    p: float = 0
    m: float = 0

    is_player = False
    excessive_losses = False

    def set_available_money(self):
        if self.is_player:
            factor = 1 + (self.r - self.q) / (self.r + 1)
        else:
            factor = 1
        self.available_money = 100 * math.floor(
            (self.army_m * (100 - self.inflation) / 2000) * factor + 0.5
        )

    def get_cost(self) -> float:
        return self.food + self.salaries + self.ammunition

    def get_army_factor(self) -> float:
        return 1 + (self.p - self.t) / (self.m + 1)

    def get_present_men(self) -> float:
        return self.army_m * self.get_army_factor()


```

这段代码定义了一个名为 `simulate_losses` 的函数和一个名为 `update_army` 的函数。这两个函数的主要目的是模拟玩家 1 和玩家 2之间的战争，并计算出玩家 1的伤亡人数。

函数 `simulate_losses(player1: PlayerStat, player2: PlayerStat) -> float` 接收两个 `PlayerStat` 类型的参数，并返回一个浮点数类型的参数。这个参数代表玩家 1的伤亡人数。函数通过以下步骤计算出玩家的伤亡人数：

1. 将玩家的军队数量除以 5，并加上 1。这样就可以将军队数量加倍。
2. 将玩家的军队数量乘以一个系数，再加上一个系数。这个系数加上了 5，因为每场比赛玩家只能失去 5% 的军队。
3. 将每一步计算出来的值乘以一个系数，再加上一个系数。这个系数在玩家拥有粮食时进行了修改，使得玩家可以有更多的人去战斗。
4. 使用 math.floor() 函数将计算结果向下取整，得到一个整数。
5. 将整数和玩家的士气值相加，得到了玩家的伤亡人数。

函数 `update_army(player: PlayerStat, enemy: PlayerStat, use_factor=False) -> None` 接收两个 `PlayerStat` 类型的参数，并返回一个 None 类型的参数。这个参数代表玩家在面对敌人时如何调整军队。

如果 `use_factor` 为真，则函数会使用一个因素来调整玩家的军队数量。否则，函数会直接将玩家的军队数量设置为玩家的人数。


```
def simulate_losses(player1: PlayerStat, player2: PlayerStat) -> float:
    """Simulate losses of player 1"""
    tmp = (2 * player1.army_c / 5) * (
        1 + 1 / (2 * (abs(player1.strategy - player2.strategy) + 1))
    )
    tmp = tmp * (1.28 + (5 * player1.army_m / 6) / (player1.ammunition + 1))
    tmp = math.floor(tmp * (1 + 1 / player1.morale) + 0.5)
    return tmp


def update_army(player: PlayerStat, enemy: PlayerStat, use_factor=False) -> None:
    player.casualties = simulate_losses(player, enemy)
    player.desertions = 100 / player.morale

    loss = player.casualties + player.desertions
    if not use_factor:
        present_men: float = player.available_men
    else:
        present_men = player.get_present_men()
    if loss >= present_men:
        factor = player.get_army_factor()
        if not use_factor:
            factor = 1
        player.casualties = math.floor(13 * player.army_m / 20 * factor)
        player.desertions = 7 * player.casualties / 13
        player.excessive_losses = True


```

这两段代码定义了两个函数 `get_choice` 和 `get_morale`。

函数 `get_choice` 的作用是询问用户在给定的提示语句（`prompt`）下，从一组选项（`choices`）中选择一个答案（在函数内部发生）。然后返回用户的选择（如果用户选择的是正确答案，函数将返回该答案，否则继续询问）。

函数 `get_morale` 的作用是计算两个 `PlayerStat` 对象（`stat` 和 `enemy`）之间的心理健康值（`stat` 的军队实力是 `enemy` 的军队实力的 5 倍）。心理健康值基于以下公式：`enemy_strength = 5 * enemy.army_m / 6`，其中 `army_m` 是 `enemy` 的军队数量。然后返回一个浮点数，该值将大于 `enemy_strength`，因为心理健康值是一个比例。


```
def get_choice(prompt: str, choices: List[str]) -> str:
    while True:
        choice = input(prompt)
        if choice in choices:
            break
    return choice


def get_morale(stat: PlayerStat, enemy: PlayerStat) -> float:
    """Higher is better"""
    enemy_strength = 5 * enemy.army_m / 6
    return (2 * math.pow(stat.food, 2) + math.pow(stat.salaries, 2)) / math.pow(
        enemy_strength, 2
    ) + 1


```

This appears to be a Python code that simulates the effects of different military strategies on the outcome of a historical battle. The code has a few different parts:

1. A if statement that checks whether the strategy is 5 or if it's one of the following cases:
* 5: The UNION has won the war
* The CONFEDERACY has won the war
* Both the UNION and the CONFEDERACY have won the war, but there are multiple confederations
* None of the above
2. If the strategy is 5 or one of the other cases, it prints a message about the outcome of the battle.
3. If the strategy is 4 or 5, it prints a table showing the historical losses and simulated losses for each side, and then calculates a distribution of these losses.
4. If the strategy is 4, it prints a table showing the percentage of original statistics for each side, and then suggests that the UNION used strategies 1, 2, or 3.
5. If the strategy is 5 or 4, it prints a table showing the percentage of original statistics for each side, and then suggests that the CONFEDERACY used strategies 3 or 4.
6. If the strategy is 5 or 4, it prints a table showing the percentage of original statistics for each side, and then suggests that both the UNION and the CONFEDERACY used strategies 3 or 4.
7. If the strategy is 6, it prints a table showing the percentage of captured territory for each side, and then prints a message about the缠斗会战。
8. If the strategy is 7 or 8, it prints a table showing the percentage of retreats for each side, and then prints a message about the Battle of White's River.

It is based on the data provided in the stats and math tables it seems to be very complex and there are different strategies and different versions of it.


```
def main() -> None:
    battles = [
        [
            "JULY 21, 1861.  GEN. BEAUREGARD, COMMANDING THE SOUTH, MET",
            "UNION FORCES WITH GEN. MCDOWELL IN A PREMATURE BATTLE AT",
            "BULL RUN. GEN. JACKSON HELPED PUSH BACK THE UNION ATTACK.",
        ],
        [
            "APRIL 6-7, 1862.  THE CONFEDERATE SURPRISE ATTACK AT",
            "SHILOH FAILED DUE TO POOR ORGANIZATION.",
        ],
        [
            "JUNE 25-JULY 1, 1862.  GENERAL LEE (CSA) UPHELD THE",
            "OFFENSIVE THROUGHOUT THE BATTLE AND FORCED GEN. MCCLELLAN",
            "AND THE UNION FORCES AWAY FROM RICHMOND.",
        ],
        [
            "AUG 29-30, 1862.  THE COMBINED CONFEDERATE FORCES UNDER LEE",
            "AND JACKSON DROVE THE UNION FORCES BACK INTO WASHINGTON.",
        ],
        [
            "SEPT 17, 1862.  THE SOUTH FAILED TO INCORPORATE MARYLAND",
            "INTO THE CONFEDERACY.",
        ],
        [
            "DEC 13, 1862.  THE CONFEDERACY UNDER LEE SUCCESSFULLY",
            "REPULSED AN ATTACK BY THE UNION UNDER GEN. BURNSIDE.",
        ],
        ["DEC 31, 1862.  THE SOUTH UNDER GEN. BRAGG WON A CLOSE BATTLE."],
        [
            "MAY 1-6, 1863.  THE SOUTH HAD A COSTLY VICTORY AND LOST",
            "ONE OF THEIR OUTSTANDING GENERALS, 'STONEWALL' JACKSON.",
        ],
        [
            "JULY 4, 1863.  VICKSBURG WAS A COSTLY DEFEAT FOR THE SOUTH",
            "BECAUSE IT GAVE THE UNION ACCESS TO THE MISSISSIPPI.",
        ],
        [
            "JULY 1-3, 1863.  A SOUTHERN MISTAKE BY GEN. LEE AT GETTYSBURG",
            "COST THEM ONE OF THE MOST CRUCIAL BATTLES OF THE WAR.",
        ],
        [
            "SEPT. 15, 1863. CONFUSION IN A FOREST NEAR CHICKAMAUGA LED",
            "TO A COSTLY SOUTHERN VICTORY.",
        ],
        [
            "NOV. 25, 1863. AFTER THE SOUTH HAD SIEGED GEN. ROSENCRANS'",
            "ARMY FOR THREE MONTHS, GEN. GRANT BROKE THE SIEGE.",
        ],
        [
            "MAY 5, 1864.  GRANT'S PLAN TO KEEP LEE ISOLATED BEGAN TO",
            "FAIL HERE, AND CONTINUED AT COLD HARBOR AND PETERSBURG.",
        ],
        [
            "AUGUST, 1864.  SHERMAN AND THREE VETERAN ARMIES CONVERGED",
            "ON ATLANTA AND DEALT THE DEATH BLOW TO THE CONFEDERACY.",
        ],
    ]

    historical_data: List[Tuple[str, float, float, float, int, AttackState]] = [
        ("", 0, 0, 0, 0, AttackState.DEFENSIVE),
        ("BULL RUN", 18000, 18500, 1967, 2708, AttackState.DEFENSIVE),
        ("SHILOH", 40000.0, 44894.0, 10699, 13047, AttackState.OFFENSIVE),
        ("SEVEN DAYS", 95000.0, 115000.0, 20614, 15849, AttackState.OFFENSIVE),
        ("SECOND BULL RUN", 54000.0, 63000.0, 10000, 14000, AttackState.BOTH_OFFENSIVE),
        ("ANTIETAM", 40000.0, 50000.0, 10000, 12000, AttackState.OFFENSIVE),
        ("FREDERICKSBURG", 75000.0, 120000.0, 5377, 12653, AttackState.DEFENSIVE),
        ("MURFREESBORO", 38000.0, 45000.0, 11000, 12000, AttackState.DEFENSIVE),
        ("CHANCELLORSVILLE", 32000, 90000.0, 13000, 17197, AttackState.BOTH_OFFENSIVE),
        ("VICKSBURG", 50000.0, 70000.0, 12000, 19000, AttackState.DEFENSIVE),
        ("GETTYSBURG", 72500.0, 85000.0, 20000, 23000, AttackState.OFFENSIVE),
        ("CHICKAMAUGA", 66000.0, 60000.0, 18000, 16000, AttackState.BOTH_OFFENSIVE),
        ("CHATTANOOGA", 37000.0, 60000.0, 36700.0, 5800, AttackState.BOTH_OFFENSIVE),
        ("SPOTSYLVANIA", 62000.0, 110000.0, 17723, 18000, AttackState.BOTH_OFFENSIVE),
        ("ATLANTA", 65000.0, 100000.0, 8500, 3700, AttackState.DEFENSIVE),
    ]
    confederate_strategy_prob_distribution = {}

    # What do you spend money on?
    stats: Dict[int, PlayerStat] = {
        CONF: PlayerStat(),
        UNION: PlayerStat(),
    }

    print(" " * 26 + "CIVIL WAR")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    # Union info on likely confederate strategy
    confederate_strategy_prob_distribution[1] = 25
    confederate_strategy_prob_distribution[2] = 25
    confederate_strategy_prob_distribution[3] = 25
    confederate_strategy_prob_distribution[4] = 25
    print()
    show_instructions = get_choice(
        "DO YOU WANT INSTRUCTIONS? YES OR NO -- ", ["YES", "NO"]
    )

    if show_instructions == "YES":
        print()
        print()
        print()
        print()
        print("THIS IS A CIVIL WAR SIMULATION.")
        print("TO PLAY TYPE A RESPONSE WHEN THE COMPUTER ASKS.")
        print("REMEMBER THAT ALL FACTORS ARE INTERRELATED AND THAT YOUR")
        print("RESPONSES COULD CHANGE HISTORY. FACTS AND FIGURES USED ARE")
        print("BASED ON THE ACTUAL OCCURRENCE. MOST BATTLES TEND TO RESULT")
        print("AS THEY DID IN THE CIVIL WAR, BUT IT ALL DEPENDS ON YOU!!")
        print()
        print("THE OBJECT OF THE GAME IS TO WIN AS MANY BATTLES AS ")
        print("POSSIBLE.")
        print()
        print("YOUR CHOICES FOR DEFENSIVE STRATEGY ARE:")
        print("        (1) ARTILLERY ATTACK")
        print("        (2) FORTIFICATION AGAINST FRONTAL ATTACK")
        print("        (3) FORTIFICATION AGAINST FLANKING MANEUVERS")
        print("        (4) FALLING BACK")
        print(" YOUR CHOICES FOR OFFENSIVE STRATEGY ARE:")
        print("        (1) ARTILLERY ATTACK")
        print("        (2) FRONTAL ATTACK")
        print("        (3) FLANKING MANEUVERS")
        print("        (4) ENCIRCLEMENT")
        print("YOU MAY SURRENDER BY TYPING A '5' FOR YOUR STRATEGY.")

    print()
    print()
    print()
    print("ARE THERE TWO GENERALS PRESENT ", end="")
    two_generals = get_choice("(ANSWER YES OR NO) ", ["YES", "NO"]) == "YES"
    stats[CONF].is_player = True
    if two_generals:
        party: Literal[1, 2] = 2  # number of players in the game
        stats[UNION].is_player = True
    else:
        party = 1
        print()
        print("YOU ARE THE CONFEDERACY.   GOOD LUCK!")
        print()

    print("SELECT A BATTLE BY TYPING A NUMBER FROM 1 TO 14 ON")
    print("REQUEST.  TYPE ANY OTHER NUMBER TO END THE SIMULATION.")
    print("BUT '0' BRINGS BACK EXACT PREVIOUS BATTLE SITUATION")
    print("ALLOWING YOU TO REPLAY IT")
    print()
    print("NOTE: A NEGATIVE FOOD$ ENTRY CAUSES THE PROGRAM TO ")
    print("USE THE ENTRIES FROM THE PREVIOUS BATTLE")
    print()
    print("AFTER REQUESTING A BATTLE, DO YOU WISH ", end="")
    print("BATTLE DESCRIPTIONS ", end="")
    xs = get_choice("(ANSWER YES OR NO) ", ["YES", "NO"])
    confederacy_lost = 0
    confederacy_win = 0
    for i in [CONF, UNION]:
        stats[i].p = 0
        stats[i].m = 0
        stats[i].t = 0
        stats[i].available_money = 0
        stats[i].food = 0
        stats[i].salaries = 0
        stats[i].ammunition = 0
        stats[i].strategy = 0
        stats[i].excessive_losses = False
    confederacy_unresolved = 0
    random_nb: float = 0
    while True:
        print()
        print()
        print()
        simulated_battle_index = int(
            get_choice(
                "WHICH BATTLE DO YOU WISH TO SIMULATE? (0-14) ",
                [str(i) for i in range(15)],
            )
        )
        if simulated_battle_index < 1 or simulated_battle_index > 14:
            break
        if simulated_battle_index != 0 or random_nb == 0:
            loaded_battle = historical_data[simulated_battle_index]
            battle_name = loaded_battle[0]
            stats[CONF].army_m = loaded_battle[1]
            stats[UNION].army_m = loaded_battle[2]
            stats[CONF].army_c = loaded_battle[3]
            stats[UNION].army_c = loaded_battle[4]
            stats[CONF].excessive_losses = False

            # Inflation calc
            stats[CONF].inflation = 10 + (confederacy_lost - confederacy_win) * 2
            stats[UNION].inflation = 10 + (confederacy_win - confederacy_lost) * 2

            # Money and Men available
            for i in [CONF, UNION]:
                stats[i].set_available_money()
                stats[i].available_men = math.floor(stats[i].get_army_factor())
            print()
            print()
            print()
            print()
            print()
            print(f"THIS IS THE BATTLE OF {battle_name}")
            if xs != "NO":
                print("\n".join(battles[simulated_battle_index - 1]))

        else:
            print(f"{battle_name} INSTANT REPLAY")

        print()
        print("          CONFEDERACY\t UNION")
        print(f"MEN       {stats[CONF].available_men}\t\t {stats[UNION].available_men}")
        print(
            f"MONEY    ${stats[CONF].available_money}\t${stats[UNION].available_money}"
        )
        print(f"INFLATION {stats[CONF].inflation + 15}%\t\t {stats[UNION].inflation}%")
        print()
        # ONLY IN PRINTOUT IS CONFED INFLATION = I1 + 15 %
        # IF TWO GENERALS, INPUT CONFED, FIRST
        for player_index in range(1, party + 1):
            if two_generals and player_index == 1:
                print("CONFEDERATE GENERAL---", end="")
            print("HOW MUCH DO YOU WISH TO SPEND FOR")
            while True:
                food_input = int(input(" - FOOD...... ? "))
                if food_input < 0:
                    if stats[CONF].r == 0:
                        print("NO PREVIOUS ENTRIES")
                        continue
                    print("ASSUME YOU WANT TO KEEP SAME ALLOCATIONS")
                    print()
                    break
                stats[player_index].food = food_input
                while True:
                    stats[player_index].salaries = int(input(" - SALARIES.. ? "))
                    if stats[player_index].salaries >= 0:
                        break
                    print("NEGATIVE VALUES NOT ALLOWED.")
                while True:
                    stats[player_index].ammunition = int(input(" - AMMUNITION ? "))
                    if stats[player_index].ammunition >= 0:
                        break
                    print("NEGATIVE VALUES NOT ALLOWED.")
                print()
                if stats[player_index].get_cost() > stats[player_index].available_money:
                    print(
                        f"THINK AGAIN! YOU HAVE ONLY ${stats[player_index].available_money}"
                    )
                else:
                    break

            if not two_generals or player_index == 2:
                break
            print("UNION GENERAL---", end="")

        for player_index in range(1, party + 1):
            if two_generals:
                if player_index == 1:
                    print("CONFEDERATE ", end="")
                else:
                    print("      UNION ", end="")
            morale = get_morale(stats[player_index], stats[1 + player_index % 2])

            if morale >= 10:
                print("MORALE IS HIGH")
            elif morale >= 5:
                print("MORALE IS FAIR")
            else:
                print("MORALE IS POOR")
            if not two_generals:
                break
            stats[player_index].morale = morale  # type: ignore

        stats[UNION].morale = get_morale(stats[UNION], stats[CONF])
        stats[CONF].morale = get_morale(stats[CONF], stats[UNION])
        print("CONFEDERATE GENERAL---")
        # Actual off/def battle situation
        if loaded_battle[5] == AttackState.OFFENSIVE:
            print("YOU ARE ON THE OFFENSIVE")
        elif loaded_battle[5] == AttackState.DEFENSIVE:
            print("YOU ARE ON THE DEFENSIVE")
        else:
            print("BOTH SIDES ARE ON THE OFFENSIVE")

        print()
        # Choose strategies
        if not two_generals:
            while True:
                stats[CONF].strategy = int(input("YOUR STRATEGY "))
                if abs(stats[CONF].strategy - 3) < 3:
                    break
                print(f"STRATEGY {stats[CONF].strategy} NOT ALLOWED.")
            if stats[CONF].strategy == 5:
                print("THE CONFEDERACY HAS SURRENDERED.")
                break
            # Union strategy is computer chosen
            if simulated_battle_index == 0:
                while True:
                    stats[UNION].strategy = int(input("UNION STRATEGY IS "))
                    if stats[UNION].strategy > 0 and stats[UNION].strategy < 5:
                        break
                    print("ENTER 1, 2, 3, OR 4 (USUALLY PREVIOUS UNION STRATEGY)")
            else:
                s0 = 0
                random_nb = random.random() * 100
                for player_index in range(1, 5):
                    s0 += confederate_strategy_prob_distribution[player_index]
                    # If actual strategy info is in program data statements
                    # then r-100 is extra weight given to that strategy.
                    if random_nb < s0:
                        break
                stats[UNION].strategy = player_index
                print(stats[UNION].strategy)
        else:
            for player_index in [1, 2]:
                if player_index == 1:
                    print("CONFEDERATE STRATEGY ? ", end="")
                while True:
                    stats[CONF].strategy = int(input())
                    if abs(stats[CONF].strategy - 3) < 3:
                        break
                    print(f"STRATEGY {stats[CONF].strategy} NOT ALLOWED.")
                    print("YOUR STRATEGY ? ", end="")
                if player_index == 2:
                    stats[UNION].strategy = stats[CONF].strategy
                    stats[CONF].strategy = previous_strategy  # type: ignore # noqa: F821
                    if stats[UNION].strategy != 5:
                        break
                else:
                    previous_strategy = stats[CONF].strategy  # noqa: F841
                print("UNION STRATEGY ? ", end="")

            update_army(stats[UNION], stats[CONF], use_factor=False)

        # Calculate simulated losses
        print()
        print()
        print()
        print("\t\tCONFEDERACY\tUNION")
        update_army(stats[CONF], stats[UNION], use_factor=True)

        if party == 1:
            stats[UNION].casualties = math.floor(
                17
                * stats[UNION].army_c
                * stats[CONF].army_c
                / (stats[CONF].casualties * 20)
            )
            stats[CONF].desertions = 5 * morale

        print(
            "CASUALTIES\t"
            + str(stats[CONF].casualties)
            + "\t\t"
            + str(stats[UNION].casualties)
        )
        print(
            "DESERTIONS\t"
            + str(math.floor(stats[CONF].desertions))
            + "\t\t"
            + str(math.floor(stats[UNION].desertions))
        )
        print()
        if two_generals:
            print("COMPARED TO THE ACTUAL CASUALTIES AT " + str(battle_name))
            print(
                "CONFEDERATE: "
                + str(
                    math.floor(
                        100 * (stats[CONF].casualties / stats[CONF].army_c) + 0.5
                    )
                )
                + "% OF THE ORIGINAL"
            )
            print(
                "UNION:       "
                + str(
                    math.floor(
                        100 * (stats[UNION].casualties / stats[UNION].army_c) + 0.5
                    )
                )
                + "% OF THE ORIGINAL"
            )

        print()
        # Find who won
        if (
            stats[CONF].excessive_losses
            and stats[UNION].excessive_losses
            or (
                not stats[CONF].excessive_losses
                and not stats[UNION].excessive_losses
                and stats[CONF].casualties + stats[CONF].desertions
                == stats[UNION].casualties + stats[CONF].desertions
            )
        ):
            print("BATTLE OUTCOME UNRESOLVED")
            confederacy_unresolved += 1
        elif stats[CONF].excessive_losses or (
            not stats[CONF].excessive_losses
            and not stats[UNION].excessive_losses
            and stats[CONF].casualties + stats[CONF].desertions
            > stats[UNION].casualties + stats[CONF].desertions
        ):
            print(f"THE UNION WINS {battle_name}")
            if simulated_battle_index != 0:
                confederacy_lost += 1
        else:
            print(f"THE CONFEDERACY WINS {battle_name}")
            if simulated_battle_index != 0:
                confederacy_win += 1

        # Lines 2530 to 2590 from original are unreachable.
        if simulated_battle_index != 0:
            for i in [CONF, UNION]:
                stats[i].t += stats[i].casualties + stats[i].desertions
                stats[i].p += stats[i].army_c
                stats[i].q += stats[i].get_cost()
                stats[i].r += stats[i].army_m * (100 - stats[i].inflation) / 20
                stats[i].m += stats[i].army_m
            # Learn present strategy, start forgetting old ones
            # present strategy of south gains 3*s, others lose s
            # probability points, unless a strategy falls below 5 % .
            s = 3
            s0 = 0
            for player_index in range(1, 5):
                if confederate_strategy_prob_distribution[player_index] <= 5:
                    continue
                confederate_strategy_prob_distribution[player_index] -= 5
                s0 += s
            confederate_strategy_prob_distribution[stats[CONF].strategy] += s0

        stats[CONF].excessive_losses = False
        stats[UNION].excessive_losses = False
        print("---------------")
        continue

    print()
    print()
    print()
    print()
    print()
    print()
    print(
        f"THE CONFEDERACY HAS WON {confederacy_win} BATTLES AND LOST {confederacy_lost}"
    )
    if stats[CONF].strategy == 5 or (
        stats[UNION].strategy != 5 and confederacy_win <= confederacy_lost
    ):
        print("THE UNION HAS WON THE WAR")
    else:
        print("THE CONFEDERACY HAS WON THE WAR")
    print()
    if stats[CONF].r > 0:
        print(
            f"FOR THE {confederacy_win + confederacy_lost + confederacy_unresolved} BATTLES FOUGHT (EXCLUDING RERUNS)"
        )
        print(" \t \t ")
        print("CONFEDERACY\t UNION")
        print(
            f"HISTORICAL LOSSES\t{math.floor(stats[CONF].p + 0.5)}\t{math.floor(stats[UNION].p + 0.5)}"
        )
        print(
            f"SIMULATED LOSSES\t{math.floor(stats[CONF].t + 0.5)}\t{math.floor(stats[UNION].t + 0.5)}"
        )
        print()
        print(
            f"    % OF ORIGINAL\t{math.floor(100 * (stats[CONF].t / stats[CONF].p) + 0.5)}\t{math.floor(100 * (stats[UNION].t / stats[UNION].p) + 0.5)}"
        )
        if not two_generals:
            print()
            print("UNION INTELLIGENCE SUGGEST THAT THE SOUTH USED")
            print("STRATEGIES 1, 2, 3, 4 IN THE FOLLOWING PERCENTAGES")
            print(
                f"{confederate_strategy_prob_distribution[CONF]} {confederate_strategy_prob_distribution[UNION]} {confederate_strategy_prob_distribution[3]} {confederate_strategy_prob_distribution[4]}"
            )


```

这段代码是一个if语句，判断当前脚本是否作为主程序运行。如果当前脚本作为主程序运行，则执行if语句中的代码块。

在这个例子中，if语句块中只有一条语句，即“main()”。这条语句的作用是定义一个函数名为“main”，如果当前脚本作为主程序运行，则会执行这个函数。因此，这个代码的作用是定义一个函数名为“main”，如果当前脚本作为主程序运行，则会执行这个函数。


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


### Combat

In this game, you are fighting a small-scale war with the computer. You have 72,000 troops which you first ust distribute among your Army, Navy, and Air Force. You may distribute them in any way you choose as long as you don’t use more than 72,000.

You then attack your opponent (the computer) and input which service and the number of men you wish to use. The computer then tells you the outcome of the battle, gives you the current statistics and allows you to determine your next move.

After the second battle, it is decided from the total statistics whether you win or lose or if a treaty is signed.

This program was created by Bob Dores of Milton, Massachusetts.

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=50)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=65)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- The original game misspells "unguarded" on line 1751.
- In an initial army attack, the program claims that the computer loses 2/3 of its army, but it actually loses its entire army (lines 150-155).

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `28_Combat/csharp/ArmedForces.cs`

这段代码定义了一个名为ArmedForces的类，用于表示一个国家的军队人数。

ArmedForces类包含三个成员变量，分别是陆军、海军和空军，都通过getter和初始化器初始化，并且可以从中获取军队总人数。

另外，还有一个名为TotalTroops的成员函数，用于计算军队总人数，它是将陆军、海军和空军人数相加得到的结果。

最后，有一个名为this的成员变量，用于存储当前军队中某个分支部门的人员数量，当需要使用当前部门人员数量时，可以从ArmedForces类中获取该部门的ArmedForces对象，然后使用branch属性来访问该属性的成员变量。如果没有正确的军事分支，则会抛出ArgumentException异常。


```
﻿using System;

namespace Game
{
    /// <summary>
    /// Represents the armed forces for a country.
    /// </summary>
    public record ArmedForces
    {
        /// <summary>
        /// Gets the number of men and women in the army.
        /// </summary>
        public int Army { get; init; }

        /// <summary>
        /// Gets the number of men and women in the navy.
        /// </summary>
        public int Navy { get; init; }

        /// <summary>
        /// Gets the number of men and women in the air force.
        /// </summary>
        public int AirForce { get; init; }

        /// <summary>
        /// Gets the total number of troops in the armed forces.
        /// </summary>
        public int TotalTroops => Army + Navy + AirForce;

        /// <summary>
        /// Gets the number of men and women in the given branch.
        /// </summary>
        public int this[MilitaryBranch branch] =>
            branch switch
            {
                MilitaryBranch.Army     => Army,
                MilitaryBranch.Navy     => Navy,
                MilitaryBranch.AirForce => AirForce,
                _                       => throw new ArgumentException("INVALID BRANCH")
            };
    }
}

```

# `28_Combat/csharp/Ceasefire.cs`

This is a class called `Ceasefire` that represents a new instance of the `Ceasefire` class. It takes in three arguments:

- `computerForces`: The computer's forces, represented by an instance of the `ArmedForces` class.
- `playerForces`: The player's forces, represented by an instance of the `ArmedForces` class.
- `absoluteVictory`: A boolean indicating whether the player achieved absolute victory by defeating the computer without destroying its military.

The `Ceasefire` class initializes a new instance of the class with the given arguments and sets the `IsAbsoluteVictory` flag to `false`.

The `AttackWithArmy`, `AttackWithNavy`, and `AttackWithAirForce` methods can be used to attack the computer with different types of forces. These methods each throw an `InvalidOperationException` if the attack is not valid.

To end the war, you can call the `Ceasefire`'s `CancelAttack` method.


```
﻿using System;

namespace Game
{
    /// <summary>
    /// Represents the state of the game after reaching a ceasefire.
    /// </summary>
    public sealed class Ceasefire : WarState
    {
        /// <summary>
        /// Gets a flag indicating whether the player achieved absolute victory.
        /// </summary>
        public override bool IsAbsoluteVictory { get; }

        /// <summary>
        /// Gets the outcome of the war.
        /// </summary>
        public override WarResult? FinalOutcome
        {
            get
            {
                if (IsAbsoluteVictory || PlayerForces.TotalTroops > 3 / 2 * ComputerForces.TotalTroops)
                    return WarResult.PlayerVictory;
                else
                if (PlayerForces.TotalTroops < 2 / 3 * ComputerForces.TotalTroops)
                    return WarResult.ComputerVictory;
                else
                    return WarResult.PeaceTreaty;
            }
        }

        /// <summary>
        /// Initializes a new instance of the Ceasefire class.
        /// </summary>
        /// <param name="computerForces">
        /// The computer's forces.
        /// </param>
        /// <param name="playerForces">
        /// The player's forces.
        /// </param>
        /// <param name="absoluteVictory">
        /// Indicates whether the player acheived absolute victory (defeating
        /// the computer without destroying its military).
        /// </param>
        public Ceasefire(ArmedForces computerForces, ArmedForces playerForces, bool absoluteVictory = false)
            : base(computerForces, playerForces)
        {
            IsAbsoluteVictory = absoluteVictory;
        }

        protected override (WarState nextState, string message) AttackWithArmy(int attackSize) =>
            throw new InvalidOperationException("THE WAR IS OVER");

        protected override (WarState nextState, string message) AttackWithNavy(int attackSize) =>
            throw new InvalidOperationException("THE WAR IS OVER");

        protected override (WarState nextState, string message) AttackWithAirForce(int attackSize) =>
            throw new InvalidOperationException("THE WAR IS OVER");
    }
}

```