# `.\DB-GPT-src\web\components\chart\helpers\downloadChartImage.ts`

```py
import { ChartRef as G2Chart } from "@berryv/g2-react";

/** 
 * 获取图表的画布元素（canvas DOM）
 * @param chart G2Chart 实例
 * @returns 返回图表的 canvas DOM 元素，如果不存在则返回 undefined
 */
const getChartCanvas = (chart: G2Chart) => {
    // 如果 chart 不存在，则返回 undefined
    if (!chart) return;
    
    // 获取 chart 的容器元素
    const chartContainer = chart.getContainer();
    
    // 从容器元素中获取第一个 canvas 元素
    const canvasNode = chartContainer.getElementsByTagName('canvas')[0];
    
    return canvasNode;
}

/** 
 * 将图表转换为 Data URL 字符串
 * @param chart G2Chart 实例
 * @returns 返回图表的 Data URL，如果无法获取则返回 undefined
 */
function toDataURL(chart: G2Chart) {
    // 获取图表的 canvas DOM 元素
    const canvasDom = getChartCanvas(chart);
    
    // 如果 canvasDom 存在，则将其转换为 PNG 格式的 Data URL
    if (canvasDom) {
        const dataURL = canvasDom.toDataURL('image/png');
        return dataURL;
    }
}

/**
 * 下载图表图片
 * @param chart G2Chart 实例
 * @param name 图片名称，默认为 'Chart'
 */
export function downloadImage(chart: G2Chart, name: string = 'Chart') {
    // 创建一个新的 <a> 元素
    const link = document.createElement('a');
    
    // 设置下载文件的名称
    const filename = `${name}.png`;

    // 在下一个事件循环中执行
    setTimeout(() => {
        // 获取图表的 Data URL
        const dataURL = toDataURL(chart);
        
        // 如果成功获取到 Data URL，则设置 <a> 元素的下载属性和链接
        if (dataURL) {
            link.addEventListener('click', () => {
                link.download = filename;
                link.href = dataURL;
            });
            
            // 创建并初始化一个模拟的鼠标事件，并分发给 <a> 元素
            const e = document.createEvent('MouseEvents');
            e.initEvent('click', false, false);
            link.dispatchEvent(e);
        }
    }, 16);
}
```