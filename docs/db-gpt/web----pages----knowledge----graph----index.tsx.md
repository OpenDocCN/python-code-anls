# `.\DB-GPT-src\web\pages\knowledge\graph\index.tsx`

```py
// 引入 React 中的必要 Hook 和依赖库
import React, { useEffect, useRef, useState } from 'react';
// 引入 cytoscape 及其布局插件 euler
import cytoscape from 'cytoscape';
import euler from 'cytoscape-euler';
// 引入 antd 中的 Button 组件及 RollbackOutlined 图标
import { Button } from 'antd';
import { RollbackOutlined } from '@ant-design/icons';
// 使用 cytoscape 的 euler 布局插件
cytoscape.use(euler);
// 引入 API 相关函数和组件路由工具
import { apiInterceptors, getGraphVis } from '@/client/api';
import { useRouter } from 'next/router';

// 定义布局配置常量对象
const LAYOUTCONFIG = {
  name: 'euler',
  springLength: 340,
  fit: false,
  springCoeff: 0.0008,
  mass: 20,
  dragCoeff: 1,
  gravity: -20,
  pull: 0.009,
  randomize: false,
  padding: 0,
  maxIterations: 1000,
  maxSimulationTime: 4000,    
}

// 定义 React 函数组件 GraphVis
function GraphVis() {
  // 使用 useRef 创建 DOM 元素的引用
  const myRef = useRef<HTMLDivElement>(null);
  // 定义并初始化 LIMIT 常量
  const LIMIT = 500
  // 使用 useRouter 获取路由对象
  const router = useRouter();

  // 定义异步函数 fetchGraphVis，获取并处理图形数据
  const fetchGraphVis = async () => {
    // 发起 API 请求，并解构响应数据
    const [_, data] =  await apiInterceptors(getGraphVis(spaceName as string, {limit: LIMIT}))
    // 检查当前组件是否已挂载及数据是否有效
    if (myRef.current && data) {
      // 处理获取到的数据
      let processedData = processResult(data)
      // 渲染图形数据
      renderGraphVis(processedData)
    }
  }

  // 定义函数 processResult，处理 API 返回的节点和边数据
  const processResult = (data: { nodes: Array<any>, edges: Array<any> }) => {
    let nodes:any[] = []
    let edges:any[] = []
    // 遍历节点数据，构造符合 Cytoscape 要求的节点对象
    data.nodes.forEach((node:any)=>{
      let n = {
        data:{
          id: node.vid,
          displayName: node.vid,
        }
      }
      nodes.push(n)
    })
    // 遍历边数据，构造符合 Cytoscape 要求的边对象
    data.edges.forEach((edge:any)=>{
      let e = {
        data:{
          id: edge.src + '_' + edge.dst + '_' + edge.label,
          source: edge.src,
          target: edge.dst,
          displayName: edge.label
        }
      }
      edges.push(e)
    })
    // 返回处理后的节点和边数据对象
    return {
      nodes,
      edges
    }
  }

  // 定义函数 renderGraphVis，渲染图形数据到指定的 DOM 元素
  const renderGraphVis = (data: any) => {
    // 获取当前组件引用的 DOM 元素
    let dom = myRef.current as HTMLDivElement
    // 创建 cytoscape 实例，将其渲染到指定的 DOM 元素中
    let cy = cytoscape(
      {
        container: myRef.current,  // 指定 cytoscape 实例的容器为 myRef 引用的 DOM 元素
        elements: data,  // 设置 cytoscape 实例的元素数据
        zoom: 0.3,  // 设置初始缩放级别
        pixelRatio: 'auto',  // 自动适配像素比例
        style: [  // 设置 cytoscape 实例的样式
          {
            selector: 'node',  // 选择节点元素
            style: {
              width: 60,  // 设置节点宽度
              height: 60,  // 设置节点高度
              color: '#fff',  // 设置节点文本颜色
              'text-outline-color': '#37D4BE',  // 设置节点文本轮廓颜色
              'text-outline-width': 2,  // 设置节点文本轮廓宽度
              'text-valign': 'center',  // 设置节点文本垂直对齐方式
              'text-halign': 'center',  // 设置节点文本水平对齐方式
              'background-color': '#37D4BE',  // 设置节点背景颜色
              'label': 'data(displayName)'  // 设置节点显示的标签为 displayName 属性的值
            }
          },
          {
            selector: 'edge',  // 选择边元素
            style: {
              'width': 1,  // 设置边的宽度
              color: '#fff',  // 设置边的颜色
              'label': 'data(displayName)',  // 设置边的标签为 displayName 属性的值
              'line-color': '#66ADFF',  // 设置边线的颜色
              'font-size': 14,  // 设置文本字体大小
              'target-arrow-shape': 'vee',  // 设置目标箭头形状为 vee
              'control-point-step-size': 40,  // 设置控制点步长大小
              'curve-style': 'bezier',  // 设置曲线样式为贝塞尔曲线
              'text-background-opacity': 1,  // 设置文本背景透明度
              'text-background-color': '#66ADFF',  // 设置文本背景颜色
              'target-arrow-color': '#66ADFF',  // 设置目标箭头颜色
              'text-background-shape': 'roundrectangle',  // 设置文本背景形状为圆角矩形
              'text-border-color': '#000',  // 设置文本边框颜色
              'text-wrap': 'wrap',  // 设置文本换行
              'text-valign': 'top',  // 设置文本垂直对齐方式
              'text-halign': 'center',  // 设置文本水平对齐方式
              'text-background-padding': '5',  // 设置文本背景填充大小
            }
          }
        ]
      }
    )
    // 运行 cytoscape 实例的布局算法
    cy.layout(LAYOUTCONFIG).run()
    // 平移 cytoscape 实例，使其居中显示在 DOM 元素中央
    cy.pan({
      x: dom.clientWidth / 2,
      y: dom.clientHeight / 2
     })
  }
  // 返回按钮点击事件处理函数，跳转回知识页面
  const back = ()=>{
    router.push(`/knowledge`);
  }
  // 从路由钩子中获取 spaceName 参数
  const {
    query: { spaceName },
  } = useRouter();
  // 使用 useEffect 钩子，在 spaceName 参数存在时触发 fetchGraphVis 函数
  useEffect(()=>{
    spaceName && fetchGraphVis()
  })
  // 返回页面的 JSX 结构
  return (
    <div className="p-4 h-full overflow-y-scroll relative px-2">
       <div>
        {/* 返回按钮，点击触发 back 函数 */}
        <Button onClick={back} icon={<RollbackOutlined />}> Back </Button>
       </div>
        {/* cytoscape 实例的容器 */}
        <div className='h-full w-full' ref={myRef}></div>
    </div>
  );
}

export default GraphVis;
```