# `.\DB-GPT-src\web\components\chat\header\agent-selector.tsx`

```py
import { ChatContext } from '@/app/chat-context';
import { apiInterceptors, getDbgptsList } from '@/client/api';
import { useRequest } from 'ahooks';
import { Select } from 'antd';
import { useContext } from 'react';
import { useTranslation } from 'react-i18next';

function AgentSelector() {
  // 使用 useTranslation hook 获取翻译函数 t
  const { t } = useTranslation();
  // 使用 useContext hook 获取 ChatContext 中的 agent 状态和 setAgent 函数
  const { agent, setAgent } = useContext(ChatContext);

  // 使用 useRequest hook 发起异步请求获取数据
  const { data = [] } = useRequest(async () => {
    // 调用 apiInterceptors 和 getDbgptsList 函数获取数据
    const [, res] = await apiInterceptors(getDbgptsList());
    // 返回获取的数据，如果 res 为 null 或 undefined，则返回空数组
    return res ?? [];
  });

  // 返回一个 Select 组件，用于选择 agent
  return (
    <Select
      className="w-60"  // 设置组件的样式类
      value={agent}  // 设置 Select 组件的值为当前 agent 状态
      placeholder={t('Select_Plugins')}  // 设置占位符文本，使用 t 函数翻译
      options={data.map((item) => ({ label: item.app_name, value: item.app_code }))}  // 使用 data 映射出选项数组，每项包括 label 和 value
      allowClear  // 允许清除已选中的值
      onChange={(val) => {
        setAgent?.(val);  // 设置选中项变化时的回调，调用 setAgent 函数更新 agent 状态
      }}
    />
  );
}

export default AgentSelector;
```