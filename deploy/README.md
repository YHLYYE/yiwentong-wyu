# 国内云服务器部署

本项目可以部署到阿里云、腾讯云、华为云或学校 Linux 服务器。
公网演示默认使用 BM25-only / 离线原文摘录，不加载 2.2GB 模型，1 核 2GB 内存即可。

## 1. 准备服务器

- Ubuntu 22.04 或 Debian 12；
- 安全组放行 8501；如果使用 Nginx，放行 80/443；
- 安装 Docker 和 Docker Compose。

## 2. 拉取代码

```bash
git clone https://github.com/YHLYYE/yiwentong-wyu.git
cd yiwentong-wyu
```

## 3. 配置演示口令

```bash
cp deploy/.env.production.example deploy/.env.production
```

编辑 `deploy/.env.production`，至少修改：

```dotenv
WYU_USE_DENSE=0
WYU_OFFLINE=1
WYU_DEMO_PASSWORD=换成正式口令
```

## 4. 启动

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

访问：

```text
http://服务器公网IP:8501
```

## 5. 如果使用 Nginx

把 `deploy/nginx.conf` 放到服务器的 Nginx 配置目录，重载 Nginx：

```bash
sudo cp deploy/nginx.conf /etc/nginx/conf.d/yiwentong.conf
sudo nginx -t
sudo systemctl reload nginx
```

只暴露 80/443，不直接暴露 8501。

## 6. 域名与 HTTPS

国内云服务器绑定域名通常需要完成 ICP 备案。
没有域名时，使用 `http://服务器IP:8501` 也可以作为演示地址；
有域名时可用 Nginx + Certbot 配置 HTTPS。

## 7. 更新版本

```bash
git pull
docker compose -f deploy/docker-compose.yml up -d --build
```
