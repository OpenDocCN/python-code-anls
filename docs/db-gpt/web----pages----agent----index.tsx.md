# `.\DB-GPT-src\web\pages\agent\index.tsx`

```py
# 导入需要的组件和库
import MarketPlugins from '@/components/agent/market-plugins';  # 导入市场插件组件
import MyPlugins from '@/components/agent/my-plugins';  # 导入我的插件组件
import { Tabs } from 'antd';  # 从antd库中导入Tabs组件
import { useMemo, useState } from 'react';  # 从react库中导入useMemo和useState钩子
import { useTranslation } from 'react-i18next';  # 从react-i18next库中导入useTranslation钩子

function Agent() {
  const { t } = useTranslation();  # 使用i18n国际化钩子，获取t函数来进行翻译

  const [activeKey, setActiveKey] = useState('market');  # 定义并初始化activeKey状态为'market'，并提供修改状态的函数setActiveKey

  # 使用useMemo来计算和缓存items数组，该数组包含Tabs组件所需的数据
  const items: Required<Parameters<typeof Tabs>[0]['items']> = useMemo(
    () => [
      {
        key: 'market',  # 标签页的键为'market'
        label: t('Market_Plugins'),  # 标签页的显示文本由t函数翻译'Market_Plugins'得到
        children: <MarketPlugins />,  # 如果是'market'标签页，则展示MarketPlugins组件
      },
      {
        key: 'my',  # 标签页的键为'my'
        label: t('My_Plugins'),  # 标签页的显示文本由t函数翻译'My_Plugins'得到
        children: activeKey === 'market' ? null : <MyPlugins />,  # 如果不是'market'标签页，则根据activeKey来决定是否展示MyPlugins组件
      },
    ],
    [t, activeKey],  # useMemo的依赖数组，当t或activeKey发生变化时重新计算items
  );

  # 渲染Agent组件的主体部分
  return (
    <div className="h-screen p-4 md:p-6 overflow-y-auto">  # 设置div元素的类名和样式
      <Tabs activeKey={activeKey} items={items} onChange={setActiveKey} />  # 渲染Tabs组件，并传入activeKey、items和onChange属性
    </div>
  );
}

export default Agent;  # 导出Agent组件作为默认输出
```