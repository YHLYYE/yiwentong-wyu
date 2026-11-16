# 邑问通公网部署说明

## 一、推荐方案：Streamlit Community Cloud

这是最容易拿到稳定公网链接的方案。需要一个 GitHub 账号，并用该账号登录 Streamlit Community Cloud。

1. 把 `05-项目源码` 推到 GitHub 仓库；
2. 打开 https://share.streamlit.io ，用 GitHub 登录；
3. 选择仓库，主文件填 `ui/wyu_app.py`；
4. 在 Advanced settings 的 Secrets 中填写：

```toml
WYU_USE_DENSE = "0"
WYU_OFFLINE = "1"
WYU_DEMO_PASSWORD = "WYU-demo-2026"
WYU_COVERAGE_MIN = "0.5"
```

5. 点击 Deploy，得到 `https://<应用名>.streamlit.app` 链接。

公开演示默认不调用大模型，且不加载 bge-m3，启动快、成本为零。
如果需要展示完整 LLM 生成，把 `WYU_OFFLINE` 改为 `"0"`，并在 Secrets 中增加：

```toml
DEEPSEEK_API_KEY = "你的密钥"
DEEPSEEK_MODEL = "deepseek-chat"
```

公开环境不建议把调用额度较大的真实 key 直接暴露给匿名用户，至少保留演示口令。

## 二、测试账号

当前页面没有用户名，只使用统一演示口令。正式提交时填写：

- 测试账号：无需用户名
- 访问口令：`WYU-demo-2026`

也可以在部署 Secrets 中改成学校要求的测试口令。

## 三、临时公网链接

如果只需要临时给老师或同学演示，可在本机运行：

```powershell
cd 05-项目源码
$env:WYU_USE_DENSE='0'
$env:WYU_OFFLINE='1'
$env:WYU_DEMO_PASSWORD='WYU-demo-2026'
python -m streamlit run ui/wyu_app.py --server.port 8502
```

再用 Cloudflare Tunnel 暴露本地端口：

```powershell
cloudflared tunnel --url http://127.0.0.1:8502
```

命令会生成一个 `trycloudflare.com` 临时链接，关闭电脑或停止进程后链接失效，不适合作为最终长期提交链接。

## 四、学校服务器或普通云服务器

```bash
cd 05-项目源码
docker build -t yiwentong .
docker run -d --name yiwentong -p 8501:8501 \
  -e WYU_USE_DENSE=0 \
  -e WYU_OFFLINE=1 \
  -e WYU_DEMO_PASSWORD=WYU-demo-2026 \
  yiwentong
```

然后由学校域名或服务器反向代理到 8501 端口。
