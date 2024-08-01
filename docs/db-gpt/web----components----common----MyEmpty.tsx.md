# `.\DB-GPT-src\web\components\common\MyEmpty.tsx`

```py
// 引入需要的组件和工具函数
import { Button, Empty } from 'antd';
import classNames from 'classnames';  // 用于动态生成组件的 class 名称
import { useTranslation } from 'react-i18next';  // 用于实现国际化文本的翻译

// 定义组件 Props 的接口，包括可选的 className、error、description 和 refresh 回调函数
interface Props {
  className?: string;  // 可选的 CSS 类名
  error?: boolean;  // 是否显示错误状态
  description?: string;  // 描述信息文本
  refresh?: () => void;  // 刷新按钮的回调函数
}

// 定义 MyEmpty 组件，接收 Props 作为参数
function MyEmpty({ className, error, description, refresh }: Props) {
  const { t } = useTranslation();  // 获取国际化翻译函数

  // 返回 Empty 组件，根据 error 属性决定显示错误信息或描述信息
  return (
    <Empty
      image="/empty.png"  // 设置空状态时显示的图片路径
      imageStyle={{ width: 320, height: 196, margin: '0 auto', maxWidth: '100%', maxHeight: '100%' }}  // 设置图片的样式
      className={classNames('flex items-center justify-center flex-col h-full w-full', className)}  // 动态生成组件的 CSS 类名
      description={
        error ? (  // 如果有错误，显示带有刷新按钮的错误信息
          <Button type="primary" onClick={refresh}>
            {t('try_again')}  // 使用国际化翻译函数 t() 获取“try_again”对应的文本
          </Button>
        ) : (
          description ?? t('no_data')  // 否则显示描述信息或默认的“no_data”文本，使用国际化翻译函数 t() 获取
        )
      }
    />
  );
}

export default MyEmpty;  // 导出 MyEmpty 组件作为默认导出
```