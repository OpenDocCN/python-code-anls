# `.\DB-GPT-src\web\pages\index.tsx`

```py
import { useRequest } from 'ahooks';
import { useContext, useState } from 'react';
import { Divider, Spin, Tag } from 'antd';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import { NextPage } from 'next';
import { apiInterceptors, newDialogue, postScenes } from '@/client/api';
import ModelSelector from '@/components/chat/header/model-selector';
import { ChatContext } from '@/app/chat-context';
import { SceneResponse } from '@/types/chat';
import CompletionInput from '@/components/common/completion-input';
import { useTranslation } from 'react-i18next';
import { STORAGE_INIT_MESSAGE_KET } from '@/utils';
import Icon from '@ant-design/icons/lib/components/Icon';
import { ColorfulDB, ColorfulPlugin, ColorfulDashboard, ColorfulData, ColorfulExcel, ColorfulDoc, ColorfulChat } from '@/components/icons';
import classNames from 'classnames';

const Home: NextPage = () => {
  const router = useRouter(); // 获取路由对象，用于页面跳转
  const { model, setModel } = useContext(ChatContext); // 使用上下文获取和设置聊天模型
  const { t } = useTranslation(); // 获取用于国际化的翻译函数

  const [loading, setLoading] = useState(false); // 状态钩子，用于加载状态管理
  const [chatSceneLoading, setChatSceneLoading] = useState<boolean>(false); // 状态钩子，用于聊天场景加载状态管理

  // 使用 ahooks 提供的 useRequest 钩子发起网络请求，获取聊天场景列表数据
  const { data: scenesList = [] } = useRequest(async () => {
    setChatSceneLoading(true); // 设置聊天场景加载状态为 true
    const [, res] = await apiInterceptors(postScenes()); // 发起网络请求获取聊天场景列表
    setChatSceneLoading(false); // 设置聊天场景加载状态为 false
    return res ?? []; // 返回聊天场景列表数据或空数组
  });

  // 异步函数，用于提交对话消息
  const submit = async (message: string) => {
    setLoading(true); // 设置加载状态为 true
    const [, res] = await apiInterceptors(newDialogue({ chat_mode: 'chat_normal' })); // 发起网络请求创建新对话
    if (res) {
      localStorage.setItem(STORAGE_INIT_MESSAGE_KET, JSON.stringify({ id: res.conv_uid, message })); // 将初始消息存储到本地存储
      router.push(`/chat/?scene=chat_normal&id=${res.conv_uid}${model ? `&model=${model}` : ''}`); // 跳转到聊天页面
    }
    setLoading(false); // 设置加载状态为 false
  };

  // 异步函数，处理新的聊天场景
  const handleNewChat = async (scene: SceneResponse) => {
    if (scene.show_disable) return; // 如果场景被禁用，则返回
    const [, res] = await apiInterceptors(newDialogue({ chat_mode: 'chat_normal' })); // 发起网络请求创建新对话
    if (res) {
      router.push(`/chat?scene=${scene.chat_scene}&id=${res.conv_uid}${model ? `&model=${model}` : ''}`); // 跳转到指定场景的聊天页面
    }
  };

  // 根据场景名称渲染对应的图标组件
  function renderSceneIcon(scene: string) {
    switch (scene) {
      case 'chat_knowledge':
        return <Icon className="w-10 h-10 mr-4 p-1" component={ColorfulDoc} />;
      case 'chat_with_db_execute':
        return <Icon className="w-10 h-10 mr-4 p-1" component={ColorfulData} />;
      case 'chat_excel':
        return <Icon className="w-10 h-10 mr-4 p-1" component={ColorfulExcel} />;
      case 'chat_with_db_qa':
        return <Icon className="w-10 h-10 mr-4 p-1" component={ColorfulDB} />;
      case 'chat_dashboard':
        return <Icon className="w-10 h-10 mr-4 p-1" component={ColorfulDashboard} />;
      case 'chat_agent':
        return <Icon className="w-10 h-10 mr-4 p-1" component={ColorfulPlugin} />;
      case 'dbgpt_chat':
        return <Icon className="w-10 h-10 mr-4 p-1" component={ColorfulChat} />;
      default:
        return null;
    }
  }

  return (
    <div className="px-4 h-screen flex flex-col justify-center items-center overflow-hidden">
      {/* 整个页面的主要容器，设置了左右内边距、屏幕高度、垂直居中和溢出隐藏 */}
      <div className="max-w-3xl max-h-screen overflow-y-auto">
        {/* 限制内容最大宽度和屏幕最大高度，并允许垂直方向上的滚动 */}
        <Image
          src="/LOGO.png"
          alt="Revolutionizing Database Interactions with Private LLM Technology"
          width={856}
          height={160}
          className="w-full mt-4"
          unoptimized
        />
        {/* 显示图片，设置了宽度、高度和外边距，并禁用了优化 */}
        <Divider className="!text-[#878c93] !my-6" plain>
          {t('Quick_Start')}
        </Divider>
        {/* 分隔线组件，使用了特定的文本颜色和垂直间距 */}
        <Spin spinning={chatSceneLoading}>
          {/* 加载中动画，根据 chatSceneLoading 变量来决定是否显示 */}
          <div className="flex flex-wrap -m-1 md:-m-2">
            {/* 使用 Flex 布局进行包裹，并设置了响应式间距 */}
            {scenesList.map((scene) => (
              {/* 遍历场景列表，对每个场景生成一个组件 */}
              <div
                key={scene.chat_scene}
                className="w-full sm:w-1/2 p-1 md:p-2"
                onClick={() => {
                  handleNewChat(scene);
                }}
              >
                {/* 每个场景卡片容器，设置了宽度、点击事件和响应式内边距 */}
                <div
                  className={classNames(
                    'flex flex-row justify-center h-[102px] min-h-min bg-white dark:bg-[#232734] dark:text-white rounded p-4 cursor-pointer hover:-translate-y-1 transition-[transform_shadow] duration-300 hover:shadow-[0_14px_20px_-10px_rgba(100,100,100,.1)]',
                    { 'grayscale !cursor-no-drop': scene.show_disable },
                  )}
                >
                  {/* 场景卡片内容，根据场景属性设置不同的样式 */}
                  {renderSceneIcon(scene.chat_scene)}
                  {/* 渲染场景图标 */}
                  <div className="flex flex-col flex-1">
                    {/* 垂直布局，占据剩余空间 */}
                    <h2 className="flex items-center text-lg font-sans font-semibold">
                      {scene.scene_name}
                      {/* 显示场景名称 */}
                      {scene.show_disable && <Tag className="ml-2">Comming soon</Tag>}
                      {/* 如果场景禁用，显示"Comming soon"标签 */}
                    </h2>
                    <p className="opacity-80 line-clamp-2">{scene.scene_describe}</p>
                    {/* 显示场景描述，设置了透明度和文本截断 */}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Spin>
        {/* 加载中动画的包裹，根据 loading 状态来决定是否显示 */}
        <div className="mt-8 mb-2">
          {/* 设置上下外边距 */}
          <ModelSelector
            onChange={(newModel: string) => {
              setModel(newModel);
            }}
          />
          {/* 模型选择器组件，设置了选择变更时的回调函数 */}
        </div>
        <div className="flex flex-1 w-full mb-4">
          {/* 使用 Flex 布局，占据剩余空间 */}
          <CompletionInput loading={loading} onSubmit={submit} />
          {/* 补全输入组件，设置了加载状态和提交回调函数 */}
        </div>
      </div>
    </div>
};

export default Home;
```