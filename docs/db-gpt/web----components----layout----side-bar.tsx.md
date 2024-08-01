# `.\DB-GPT-src\web\components\layout\side-bar.tsx`

```py
// 导入 ChatContext 对象从 '@/app/chat-context' 中
import { ChatContext } from '@/app/chat-context';
// 导入 apiInterceptors 和 delDialogue 方法从 '@/client/api' 中
import { apiInterceptors, delDialogue } from '@/client/api';
// 导入 STORAGE_LANG_KEY 和 STORAGE_THEME_KEY 从 '@/utils' 中
import { STORAGE_LANG_KEY, STORAGE_THEME_KEY } from '@/utils';
// 导入 DarkSvg, SunnySvg, ModelSvg 从 '@/components/icons' 中
import { DarkSvg, SunnySvg, ModelSvg } from '@/components/icons';
// 导入 IChatDialogueSchema 类型从 '@/types/chat' 中
import { IChatDialogueSchema } from '@/types/chat';
// 导入 Icon 和多个 ant-design/icons 中的图标
import Icon, {
  ConsoleSqlOutlined,
  PartitionOutlined,
  DeleteOutlined,
  MessageOutlined,
  GlobalOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  PlusOutlined,
  ShareAltOutlined,
  MenuOutlined,
  SettingOutlined,
  BuildOutlined,
  ForkOutlined,
  AppstoreOutlined,
} from '@ant-design/icons';
// 导入 Modal, message, Tooltip, Dropdown 从 'antd' 中
import { Modal, message, Tooltip, Dropdown } from 'antd';
// 导入 ItemType 从 'antd/es/menu/hooks/useItems' 中
import { ItemType } from 'antd/es/menu/hooks/useItems';
// 导入 copy 方法从 'copy-to-clipboard' 中
import copy from 'copy-to-clipboard';
// 导入 Image 和 Link 从 'next/image' 和 'next/link' 中
import Image from 'next/image';
import Link from 'next/link';
// 导入 useRouter 从 'next/router' 中
import { useRouter } from 'next/router';
// 导入 ReactNode, useCallback, useContext, useEffect, useMemo, useState 从 'react' 中
import { ReactNode, useCallback, useContext, useEffect, useMemo, useState } from 'react';
// 导入 useTranslation 从 'react-i18next' 中
import { useTranslation } from 'react-i18next';

// 定义 SettingItem 类型，包含 key, name, icon, noDropdownItem, onClick 属性
type SettingItem = {
  key: string;
  name: string;
  icon: ReactNode;
  noDropdownItem?: boolean;
  onClick: () => void;
};

// 定义 RouteItem 类型，包含 key, name, icon, path 属性
type RouteItem = {
  key: string;
  name: string;
  icon: ReactNode;
  path: string;
};

// 定义 menuItemStyle 函数，返回菜单项样式的字符串
function menuItemStyle(active?: boolean) {
  return `flex items-center h-10 hover:bg-[#F1F5F9] dark:hover:bg-theme-dark text-base w-full transition-colors whitespace-nowrap px-4 ${
    active ? 'bg-[#F1F5F9] dark:bg-theme-dark' : ''
  }`;
}

// 定义 smallMenuItemStyle 函数，返回小菜单项样式的字符串
function smallMenuItemStyle(active?: boolean) {
  return `flex items-center justify-center mx-auto rounded w-14 h-14 text-xl hover:bg-[#F1F5F9] dark:hover:bg-theme-dark transition-colors cursor-pointer ${
    active ? 'bg-[#F1F5F9] dark:bg-theme-dark' : ''
  }`;
}

// 定义 SideBar 组件
function SideBar() {
  // 使用 useContext 获取 ChatContext 提供的 chatId, scene, isMenuExpand, dialogueList, queryDialogueList,
  // refreshDialogList, setIsMenuExpand, setAgent, mode, setMode 等属性
  const { chatId, scene, isMenuExpand, dialogueList, queryDialogueList, refreshDialogList, setIsMenuExpand, setAgent, mode, setMode } =
    useContext(ChatContext);
  // 使用 useRouter 获取当前页面的 pathname 和 replace 方法
  const { pathname, replace } = useRouter();
  // 使用 useTranslation 获取当前语言和翻译函数
  const { t, i18n } = useTranslation();

  // 定义并初始化 logo 状态，初始值为 '/LOGO_1.png'
  const [logo, setLogo] = useState<string>('/LOGO_1.png');

  // 使用 useMemo 创建 routes 常量，用于存储路由信息
  const routes = useMemo(() => {
  const dropDownSettings: ItemType[] = useMemo(() => {
    // 使用 useMemo 来缓存设置项数组，避免不必要的重复计算
    return settings.map<ItemType>((item) => ({
      // 将设置项映射为 ItemType 类型的对象
      key: item.key,
      label: (
        // 设置标签为一个带链接的组件，显示图标和名称
        <Link href={item.onClick ? undefined : '#'} className="text-base">
          {item.icon}
          <span className="ml-2 text-sm">{item.name}</span>
        </Link>
      ),
      onClick: item.onClick,
      // 如果设置项有 noDropdownItem 属性，则不显示在下拉菜单中
      noDropdownItem: item.noDropdownItem,
    }));
  }, [settings]);
  return settings
    .filter((item) => !item.noDropdownItem)  // 过滤掉标记为 noDropdownItem 的项
    .map<ItemType>((item) => ({  // 将 settings 数组中的每个项映射为 ItemType 类型的对象
      key: item.key,  // 使用 item 的 key 作为新对象的 key
      label: (  // 定义新对象的 label 属性，这是一个 JSX 元素
        <div className="text-base" onClick={item.onClick}>
          {item.icon}  // 在 label 中显示 item 的 icon
          <span className="ml-2 text-sm">{item.name}</span>  // 在 label 中显示 item 的 name
        </div>
      ),
    }));
  }, [settings]);

const handleDelChat = useCallback(
  (dialogue: IChatDialogueSchema) => {
    Modal.confirm({  // 调用 Ant Design 的 Modal.confirm 方法，显示确认对话框
      title: 'Delete Chat',  // 对话框标题为 'Delete Chat'
      content: 'Are you sure delete this chat?',  // 对话框内容为 'Are you sure delete this chat?'
      width: '276px',  // 对话框宽度为 276px
      centered: true,  // 对话框居中显示
      onOk() {  // 点击确认按钮时的回调函数
        return new Promise<void>(async (resolve, reject) => {
          try {
            const [err] = await apiInterceptors(delDialogue(dialogue.conv_uid));  // 调用 delDialogue 接口删除对话
            if (err) {  // 如果删除出错
              reject();  // 拒绝 Promise
              return;
            }
            message.success('success');  // 显示成功消息
            refreshDialogList();  // 刷新对话列表
            // 如果对话的 chat_mode 是 scene 并且 conv_uid 等于 chatId，则跳转到 '/'
            dialogue.chat_mode === scene && dialogue.conv_uid === chatId && replace('/');
            resolve();  // 解决 Promise
          } catch (e) {
            reject();  // 捕获到异常时拒绝 Promise
          }
        });
      },
    });
  },
  [refreshDialogList],  // 依赖于 refreshDialogList 函数
);

const handleClickChatItem = (item: IChatDialogueSchema) => {
  if (item.chat_mode === 'chat_agent' && item.select_param) {  // 如果对话模式是 'chat_agent' 并且有 select_param
    setAgent?.(item.select_param);  // 调用 setAgent 函数，并传入 select_param 参数
  }
};

const copyLink = useCallback((item: IChatDialogueSchema) => {
  const success = copy(`${location.origin}/chat?scene=${item.chat_mode}&id=${item.conv_uid}`);  // 复制当前对话的链接到剪贴板
  message[success ? 'success' : 'error'](success ? 'Copy success' : 'Copy failed');  // 根据复制成功与否显示消息
}, []);

useEffect(() => {
  queryDialogueList();  // 组件挂载后立即查询对话列表
}, []);

useEffect(() => {
  setLogo(mode === 'dark' ? '/WHITE_LOGO.png' : '/LOGO_1.png');  // 根据 mode 设置 Logo 的路径
}, [mode]);

if (!isMenuExpand) {  // 如果菜单没有展开
    return (
      // 返回一个 JSX 元素，表示整个应用程序的主界面
      <div className="flex flex-col justify-between h-screen bg-white dark:bg-[#232734] animate-fade animate-duration-300">
        {/* 应用程序的顶部链接 */}
        <Link href="/" className="px-2 py-3">
          {/* 应用程序的 logo */}
          <Image src="/LOGO_SMALL.png" alt="DB-GPT" width={63} height={46} className="w-[63px] h-[46px]" />
        </Link>
        <div>
          {/* 应用程序的添加按钮 */}
          <Link href="/" className="flex items-center justify-center my-4 mx-auto w-12 h-12 bg-theme-primary rounded-full text-white">
            <PlusOutlined className="text-lg" />
          </Link>
        </div>
        {/* 聊天列表 */}
        <div className="flex-1 overflow-y-scroll py-4 space-y-2">
          {dialogueList?.map((item) => {
            const active = item.conv_uid === chatId && item.chat_mode === scene;

            return (
              // 每个聊天项的提示框和链接
              <Tooltip key={item.conv_uid} title={item.user_name || item.user_input} placement="right">
                <Link
                  href={`/chat?scene=${item.chat_mode}&id=${item.conv_uid}`}
                  className={smallMenuItemStyle(active)}
                  onClick={() => {
                    handleClickChatItem(item);
                  }}
                >
                  <MessageOutlined />
                </Link>
              </Tooltip>
            );
          })}
        </div>
        {/* 应用程序底部的设置和菜单按钮 */}
        <div className="py-4">
          {/* 设置菜单下拉 */}
          <Dropdown menu={{ items: dropDownRoutes }} placement="topRight">
            <div className={smallMenuItemStyle()}>
              <MenuOutlined />
            </div>
          </Dropdown>
          {/* 设置下拉菜单 */}
          <Dropdown menu={{ items: dropDownSettings }} placement="topRight">
            <div className={smallMenuItemStyle()}>
              <SettingOutlined />
            </div>
          </Dropdown>
          {/* 显示在设置中的个性化设置 */}
          {settings
            .filter((item) => item.noDropdownItem)
            .map((item) => (
              <Tooltip key={item.key} title={item.name} placement="right">
                <div className={smallMenuItemStyle()} onClick={item.onClick}>
                  {item.icon}
                </div>
              </Tooltip>
            ))}
        </div>
      </div>
    );
  }
    <div className="flex flex-col h-screen bg-white dark:bg-[#232734]">
      {/* 页面布局的主体部分，包括整体的背景色和布局方向 */}
      <Link href="/" className="p-2">
        {/* 点击链接返回首页 */}
        <Image src={logo} alt="DB-GPT" width={239} height={60} className="w-full h-full" />
      </Link>
      <Link href="/" className="flex items-center justify-center mb-4 mx-4 h-11 bg-theme-primary rounded text-white">
        {/* 带图标的新聊天链接 */}
        <PlusOutlined className="mr-2" />
        <span>{t('new_chat')}</span>
      </Link>
      {/* 聊天列表 */}
      <div className="flex-1 overflow-y-scroll">
        {dialogueList?.map((item) => {
          const active = item.conv_uid === chatId && item.chat_mode === scene;
    
          return (
            <Link
              key={item.conv_uid}
              href={`/chat?scene=${item.chat_mode}&id=${item.conv_uid}`}
              className={`group/item ${menuItemStyle(active)}`}
              onClick={() => {
                handleClickChatItem(item);
              }}
            >
              {/* 聊天列表中的每一项 */}
              <MessageOutlined className="text-base" />
              <div className="flex-1 line-clamp-1 mx-2 text-sm">{item.user_name || item.user_input}</div>
              <div
                className="group-hover/item:opacity-100 cursor-pointer opacity-0 mr-1"
                onClick={(e) => {
                  e.preventDefault();
                  copyLink(item);
                }}
              >
                {/* 分享按钮 */}
                <ShareAltOutlined />
              </div>
              <div
                className="group-hover/item:opacity-100 cursor-pointer opacity-0"
                onClick={(e) => {
                  e.preventDefault();
                  handleDelChat(item);
                }}
              >
                {/* 删除按钮 */}
                <DeleteOutlined />
              </div>
            </Link>
          );
        })}
      </div>
      {/* 设置区域 */}
      <div className="pt-4">
        <div className="max-h-52 overflow-y-auto scrollbar-default">
          {/* 菜单项 */}
          {routes.map((item) => (
            <Link key={item.key} href={item.path} className={`${menuItemStyle(pathname === item.path)} overflow-hidden`}>
              <>
                {item.icon}
                <span className="ml-3 text-sm">{item.name}</span>
              </>
            </Link>
          ))}
        </div>
        {/* 设置按钮区域 */}
        <div className="flex items-center justify-around py-4 mt-2">
          {settings.map((item) => (
            <Tooltip key={item.key} title={item.name}>
              <div className="flex-1 flex items-center justify-center cursor-pointer text-xl" onClick={item.onClick}>
                {item.icon}
              </div>
            </Tooltip>
          ))}
        </div>
      </div>
    </div>
}

export default SideBar;
```