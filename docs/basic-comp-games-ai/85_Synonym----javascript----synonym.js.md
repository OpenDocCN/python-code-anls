# `basic-computer-games\85_Synonym\javascript\synonym.js`

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
                       // 监听键盘事件，当按下回车键时，将输入的值添加到输出元素中，并解析 Promise
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

// 定义一些变量和数组
var ra = [, "RIGHT", "CORRECT", "FINE", "GOOD!", "CHECK"];
var la = [];
var tried = [];

var synonym = [[5,"FIRST","START","BEGINNING","ONSET","INITIAL"],
               // ... 其他同类别的词组
               [7,"PAIN","SUFFERING","HURT","MISERY","DISTRESS","ACHE","DISCOMFORT"]
               ];

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "SYNONYM\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化 tried 数组
    for (c = 0; c <= synonym.length; c++)
        tried[c] = false;
    // 打印提示信息
    print("A SYNONYM OF A WORD MEANS ANOTHER WORD IN THE ENGLISH\n");
    print("LANGUAGE WHICH HAS THE SAME OR VERY NEARLY THE SAME");
    print(" MEANING.\n");
    print("I CHOOSE A WORD -- YOU TYPE A SYNONYM.\n");
    print("IF YOU CAN'T THINK OF A SYNONYM, TYPE THE WORD 'HELP'\n");
    print("AND I WILL TELL YOU A SYNONYM.\n");
    print("\n");
    c = 0;
    while (c < synonym.length) {
        c++;
        do {
            n1 = Math.floor(Math.random() * synonym.length + 1);
        } while (tried[n1]) ;
        tried[n1] = true;
        n2 = synonym[n1][0];    // Length of synonym list
        // This array keeps a list of words not shown
        for (j = 1; j <= n2; j++)
            la[j] = j;
        la[0] = n2;
        g = 1;  // Always show first word
        print("\n");
        la[g] = la[la[0]];  // Replace first word with last word
        la[0] = n2 - 1; // Reduce size of list by one.
        print("\n");
        while (1) {
            print("     WHAT IS A SYNONYM OF " + synonym[n1][g]);
            str = await input();
            if (str == "HELP") {
                g1 = Math.floor(Math.random() * la[0] + 1);
                print("**** A SYNONYM OF " + synonym[n1][g] + " IS " + synonym[n1][la[g1]] + ".\n");
                print("\n");
                la[g1] = la[la[0]];
                la[0]--;
                continue;
            }
            for (k = 1; k <= n2; k++) {
                if (g == k)
                    continue;
                if (str == synonym[n1][k])
                    break;
            }
            if (k > n2) {
                print("     TRY AGAIN.\n");
            } else {
                print(synonym[n1][Math.floor(Math.random() * 5 + 1)] + "\n");
                break;
            }
        }
    }
    print("\n");
    print("SYNONYM DRILL COMPLETED.\n");
}

// 调用主程序
main();

```