# `.\DB-GPT-src\web\components\chat\mode-tab\index.tsx`

```py
import './index.css';
import { useContext } from 'react';  // 导入React库中的useContext钩子函数
import { ChatContext } from '@/app/chat-context';  // 从应用的聊天上下文中导入ChatContext
import { Radio } from 'antd';  // 从antd库中导入Radio组件
import Icon, { AppstoreFilled } from '@ant-design/icons';  // 从Ant Design图标库中导入Icon和AppstoreFilled图标
import { StarsSvg } from '@/components/icons';  // 从自定义组件中导入StarsSvg图标

export default function ModeTab() {  // 定义一个名为ModeTab的React函数组件
  const { isContract, setIsContract, scene } = useContext(ChatContext);  // 使用ChatContext中的useContext钩子函数，获取isContract、setIsContract和scene

  const isShow = scene && ['chat_with_db_execute', 'chat_dashboard'].includes(scene as string);  // 根据scene变量判断当前是否显示组件

  if (!isShow) {  // 如果不需要显示组件，则返回null
    return null;
  }

  return (  // 返回Radio.Group组件，用于渲染单选按钮组
    <Radio.Group
      value={isContract}  // 设置Radio.Group的选中值为isContract
      defaultValue={true}  // 设置Radio.Button的默认值为true
      buttonStyle="solid"  // 设置Radio.Button的样式为实心
      onChange={() => {  // 当Radio.Group的值发生变化时的回调函数，切换isContract的值
        setIsContract(!isContract);
      }}
    >
      <Radio.Button value={false}>  // 渲染一个值为false的Radio.Button
        <Icon component={StarsSvg} className="mr-1" />  // 在Radio.Button内部渲染StarsSvg图标，带有样式类名"mr-1"
        Preview  // Radio.Button的文本内容为Preview
      </Radio.Button>
      <Radio.Button value={true}>  // 渲染一个值为true的Radio.Button
        <AppstoreFilled className="mr-1" />  // 在Radio.Button内部渲染AppstoreFilled图标，带有样式类名"mr-1"
        Editor  // Radio.Button的文本内容为Editor
      </Radio.Button>
    </Radio.Group>
  );
}
```