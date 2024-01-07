# `basic-computer-games\03_Animal\javascript\animal.js`

```

// 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，解析输入的值并返回
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

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 打印标题
print(tab(32) + "ANIMAL\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");
print("PLAY 'GUESS THE ANIMAL'\n");
print("\n");
print("THINK OF AN ANIMAL AND THE COMPUTER WILL TRY TO GUESS IT.\n");
print("\n");

var k;
var n;
var str;
var q;
var z;
var c;
var t;

// 定义动物数组
var animals = [
               "\\QDOES IT SWIM\\Y1\\N2\\",
               "\\AFISH",
               "\\ABIRD",
               ];

n = animals.length;

// 打印已知的动物
function show_animals() {
    var x;

    print("\n");
    print("ANIMALS I ALREADY KNOW ARE:\n");
    str = "";
    x = 0;
    for (var i = 0; i < n; i++) {
        if (animals[i].substr(0, 2) == "\\A") {
            while (str.length < 15 * x)
                str += " ";
            for (var z = 2; z < animals[i].length; z++) {
                if (animals[i][z] == "\\")
                    break;
                str += animals[i][z];
            }
            x++;
            if (x == 4) {
                x = 0;
                print(str + "\n");
                str = "";
            }
        }
    }
    if (str != "")
        print(str + "\n");
}

// 主控制部分，使用异步函数
async function main()
{
    while (1) {
        while (1) {
            print("ARE YOU THINKING OF AN ANIMAL");
            str = await input();
            if (str == "LIST")
                show_animals();
            if (str[0] == "Y")
                break;
        }

        k = 0;
        do {
            // 子程序打印问题
            q = animals[k];
            while (1) {
                str = "";
                for (z = 2; z < q.length; z++) {
                    if (q[z] == "\\")
                        break;
                    str += q[z];
                }
                print(str);
                c = await input();
                if (c[0] == "Y" || c[0] == "N")
                    break;
            }
            t = "\\" + c[0];
            x = q.indexOf(t);
            k = parseInt(q.substr(x + 2));
        } while (animals[k].substr(0,2) == "\\Q") ;

        print("IS IT A " + animals[k].substr(2));
        a = await input();
        if (a[0] == "Y") {
            print("WHY NOT TRY ANOTHER ANIMAL?\n");
            continue;
        }
        print("THE ANIMAL YOU WERE THINKING OF WAS A ");
        v = await input();
        print("PLEASE TYPE IN A QUESTION THAT WOULD DISTINGUISH A\n");
        print(v + " FROM A " + animals[k].substr(2) + "\n");
        x = await input();
        while (1) {
            print("FOR A " + v + " THE ANSWER WOULD BE ");
            a = await input();
            a = a.substr(0, 1);
            if (a == "Y" || a == "N")
                break;
        }
        if (a == "Y")
            b = "N";
        if (a == "N")
            b = "Y";
        z1 = animals.length;
        animals[z1] = animals[k];
        animals[z1 + 1] = "\\A" + v;
        animals[k] = "\\Q" + x + "\\" + a + (z1 + 1) + "\\" + b + z1 + "\\";
    }
}

// 调用主函数
main();

```