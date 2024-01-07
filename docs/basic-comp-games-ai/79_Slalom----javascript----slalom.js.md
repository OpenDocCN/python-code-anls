# `basic-computer-games\79_Slalom\javascript\slalom.js`

```

// SLALOM
// 
// 由 Oscar Toledo G. (nanochess) 将 BASIC 转换为 Javascript
//

// 打印函数，将字符串输出到指定的元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 输入函数，返回一个 Promise 对象，当用户输入完成时解析该 Promise
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

// 生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 速度数组
var speed = [,14,18,26,29,18,
             25,28,32,29,20,
             29,29,25,21,26,
             29,20,21,20,18,
             26,25,33,31,22];

// 显示游戏说明的函数
function show_instructions()
{
    print("\n");
    print("*** SLALOM: THIS IS THE 1976 WINTER OLYMPIC GIANT SLALOM.  YOU ARE\n");
    print("            THE AMERICAN TEAM'S ONLY HOPE OF A GOLD MEDAL.\n");
    print("\n");
    print("     0 -- TYPE THIS IS YOU WANT TO SEE HOW LONG YOU'VE TAKEN.\n");
    print("     1 -- TYPE THIS IF YOU WANT TO SPEED UP A LOT.\n");
    print("     2 -- TYPE THIS IF YOU WANT TO SPEED UP A LITTLE.\n");
    print("     3 -- TYPE THIS IF YOU WANT TO SPEED UP A TEENSY.\n");
    print("     4 -- TYPE THIS IF YOU WANT TO KEEP GOING THE SAME SPEED.\n");
    print("     5 -- TYPE THIS IF YOU WANT TO CHECK A TEENSY.\n");
    print("     6 -- TYPE THIS IF YOU WANT TO CHECK A LITTLE.\n");
    print("     7 -- TYPE THIS IF YOU WANT TO CHECK A LOT.\n");
    print("     8 -- TYPE THIS IF YOU WANT TO CHEAT AND TRY TO SKIP A GATE.\n");
    print("\n");
    print(" THE PLACE TO USE THESE OPTIONS IS WHEN THE COMPUTER ASKS:\n");
    print("\n");
    print("OPTION?\n");
    print("\n");
    print("                GOOD LUCK!\n");
    print("\n");
}

// 显示速度的函数
function show_speeds()
{
    print("GATE MAX\n");
    print(" #  M.P.H.\n");
    print("----------\n");
    for (var b = 1; b <= v; b++) {
        print(" " + b + "  " + speed[b] + "\n");
    }
}

// 主程序
async function main()
}

main();

```