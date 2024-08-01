# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\pages\knowledge\graph-f0b2c9d145d2c446.js`

```py
(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[7917],{99790:function(e,t,r){  // 定义一个 webpack 模块函数，接受 e、t、r 三个参数
    (window.__NEXT_P=window.__NEXT_P||[]).push(["/knowledge/graph",function(){  // 将函数推入 __NEXT_P 数组，表示预加载指定路径的模块
        return r(15219);  // 调用 r 函数加载模块 15219
    }])
},15219:function(e,t,r){  // 定义模块 15219
    "use strict";  // 启用严格模式

    r.r(t);  // 导出默认模块

    var l=r(85893),  // 加载模块 85893
        a=r(67294),  // 加载模块 67294
        n=r(1387),   // 加载模块 1387
        o=r(13840),  // 加载模块 13840
        i=r.n(o),    // 使用模块 13840 的默认导出
        s=r(71577),  // 加载模块 71577
        c=r(71965),  // 加载模块 71965
        u=r(89182),  // 加载模块 89182
        d=r(11163);  // 加载模块 11163

    n.Z.use(i());  // 使用模块 13840 的默认导出

    let f={  // 定义对象 f，包含图形布局的参数设置
        name:"euler",  // 图形布局算法名称
        springLength:340,  // 弹簧长度
        fit:!1,  // 不自适应视口大小
        springCoeff:8e-4,  // 弹簧系数
        mass:20,  // 质量
        dragCoeff:1,  // 拖拽系数
        gravity:-20,  // 重力
        pull:.009,  // 拉力
        randomize:!1,  // 不随机布局
        padding:0,  // 边距
        maxIterations:1e3,  // 最大迭代次数
        maxSimulationTime:4e3  // 最大模拟时间
    };

    t.default=function(){  // 默认导出函数
        let e=(0,a.useRef)(null),  // 创建 ref 对象 e
            t=(0,d.useRouter)(),  // 使用 useRouter 钩子函数
            r=async()=>{  // 异步函数定义
                let[t,r]=await (0,u.Vx)((0,u.FT)(p,{limit:500}));  // 执行异步数据请求
                e.current&&r&&i(o(r))  // 当 ref 对象存在且返回结果有效时，调用模块 13840 处理数据
            },
            o=e=>{  // 定义函数 o 处理图形数据
                let t=[],  // 定义空数组 t 存储节点数据
                    r=[];  // 定义空数组 r 存储边数据
                return e.nodes.forEach(e=>{  // 遍历传入的节点数据
                    let r={data:{id:e.vid,displayName:e.vid}};  // 创建节点对象
                    t.push(r)  // 将节点对象添加到数组 t 中
                }),
                e.edges.forEach(e=>{  // 遍历传入的边数据
                    let t={data:{id:e.src+"_"+e.dst+"_"+e.label,source:e.src,target:e.dst,displayName:e.label}};  // 创建边对象
                    r.push(t)  // 将边对象添加到数组 r 中
                }),
                {nodes:t,edges:r}  // 返回处理后的节点和边数组
            },
            i=t=>{  // 定义函数 i 初始化图形渲染
                let r=e.current,  // 获取当前 ref 对象
                    l=(0,n.Z)({  // 调用 cytoscape 渲染函数
                        container:e.current,  // 指定渲染容器
                        elements:t,  // 传入图形元素数据
                        zoom:.3,  // 缩放比例
                        pixelRatio:"auto",  // 像素比例自动调整
                        style:[  // 渲染样式设置
                            {selector:"node",style:{width:60,height:60,color:"#fff","text-outline-color":"#37D4BE","text-outline-width":2,"text-valign":"center","text-halign":"center","background-color":"#37D4BE",label:"data(displayName)"}},  // 节点样式设置
                            {selector:"edge",style:{width:1,color:"#fff",label:"data(displayName)","line-color":"#66ADFF","font-size":14,"target-arrow-shape":"vee","control-point-step-size":40,"curve-style":"bezier","text-background-opacity":1,"text-background-color":"#66ADFF","target-arrow-color":"#66ADFF","text-background-shape":"roundrectangle","text-border-color":"#000","text-wrap":"wrap","text-valign":"top","text-halign":"center","text-background-padding":"5"}}  // 边样式设置
                        ]
                    });
                l.layout(f).run(),  // 运行图形布局算法
                l.pan({x:r.clientWidth/2,y:r.clientHeight/2})  // 将视角中心移动到容器中心
            },
            s=(0,d.useRouter)();  // 使用 useRouter 钩子函数获取路由信息
        return(0,a.useEffect)(()=>{  // 使用 useEffect 钩子函数
            p&&r()  // 当条件 p 成立时执行异步函数 r
        }),
        (0,l.jsxs)("div",{className:"p-4 h-full overflow-y-scroll relative px-2",children:[  // 返回 JSX 结构
            (0,l.jsx)("div",{children:(0,l.jsx)(s.ZP,{onClick:()=>{t.push("/knowledge")},icon:(0,l.jsx)(c.Z,{}),children:" Back "})}),  // 返回返回按钮 JSX 结构
            (0,l.jsx)("div",{className:"h-full w-full",ref:e})  // 返回包含 ref 的 div JSX 结构
        ]})  // 返回整体 JSX 结构
    }
},function(e){e.O(0,[9209,193,9774,2888,179],function(){return e(e.s=99790)}),_N_E=e.O()}]);  // 导出 webpack 模块
```