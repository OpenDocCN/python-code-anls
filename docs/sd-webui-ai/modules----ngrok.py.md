# `stable-diffusion-webui\modules\ngrok.py`

```py
import ngrok

# 连接到 ngrok 用于入口
def connect(token, port, options):
    # 如果 token 为 None，则将其设置为字符串 'None'
    account = None
    if token is None:
        token = 'None'
    else:
        # 如果 token 中包含冒号，则将其拆分为 authtoken、username 和 password
        if ':' in token:
            token, username, password = token.split(':', 2)
            account = f"{username}:{password}"

    # 设置 ngrok 连接的选项，具体选项请参考链接: https://github.com/ngrok/ngrok-py/blob/main/examples/ngrok-connect-full.py
    if not options.get('authtoken_from_env'):
        options['authtoken'] = token
    if account:
        options['basic_auth'] = account
    if not options.get('session_metadata'):
        options['session_metadata'] = 'stable-diffusion-webui'

    try:
        # 尝试连接到 ngrok，获取公共 URL
        public_url = ngrok.connect(f"127.0.0.1:{port}", **options).url()
    except Exception as e:
        # 如果连接出现异常，则打印错误信息和提示获取正确的 ngrok authtoken
        print(f'Invalid ngrok authtoken? ngrok connection aborted due to: {e}\n'
              f'Your token: {token}, get the right one on https://dashboard.ngrok.com/get-started/your-authtoken')
    else:
        # 如果连接成功，则打印成功信息和可用的公共 URL
        print(f'ngrok connected to localhost:{port}! URL: {public_url}\n'
               'You can use this link after the launch is complete.')
```