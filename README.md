# Cat Breath Rate Monitor v1.2

非接触式猫咪呼吸频率监测。安卓手机 + 云服务器，YOLOv8n 猫检测 + Farneback 光流 + 峰计数 + FFT 交叉验证，超阈值飞书群告警。

## 架构

```
安卓手机(IP Webcam) → frp 穿透 → 腾讯云(Docker) → 飞书群告警
```

| 组件 | 位置 | 作用 |
|------|------|------|
| IP Webcam | 安卓手机 | 摄像头推流（局域网 8080） |
| frpc (Termux) | 安卓手机 | TCP 隧道，8080 → 云:18080 |
| frps | 云服务器 Docker | 接收 frpc 连接 |
| cat_monitor | 云服务器 Docker | 抓帧 → 检测 → 分析 → 告警 |

## 检测流程

```
每5分钟: 抓30秒快照(1.3fps) → YOLOv8n 猫检测 → 静止判定(35%漂移)
                                    ↓ 静止
                              Farneback 光流(躯干ROI 30-70%)
                                    ↓
                              峰计数(主) + FFT 交叉验证(0.17-1.2Hz)
                                    ↓
                              呼吸频率 → CSV + 飞书告警
```

## 告警规则

- **呼吸急促**：> 35 次/分钟 → 飞书群推送（同类 30 分钟冷却）
- **每小时汇总**：推送过去 1 小时检测次数和每次呼吸频率；0 次检测时报警
- **飞书接入**：App ID + Secret 方式，调用 `im/v1/messages` API

## 部署

### 1. 视频源

安卓手机安装 IP Webcam + Termux + frpc：

```
# frpc.toml
serverAddr = "你的云服务器IP"
serverPort = 7000

[[proxies]]
name = "ipwebcam"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8080
remotePort = 18080
```

### 2. 云服务器

```bash
cp deploy/.env.example deploy/.env
# 编辑 .env 填入飞书 App ID / Secret / Chat ID

docker compose up -d
```

### 3. 本地开发

```bash
cp .env.example .env
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python cat_monitor.py
```

## 配置

| 变量 | 说明 |
|------|------|
| `CAMERA_URL` | 摄像头地址（云端默认 `http://127.0.0.1:18080/video`） |
| `FEISHU_APP_ID` | 飞书应用 App ID |
| `FEISHU_APP_SECRET` | 飞书应用 Secret |
| `FEISHU_CHAT_ID` | 飞书群 Chat ID |

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CAPTURE_DURATION` | 30s | 每轮采集时长 |
| `CYCLE_INTERVAL` | 300s | 检测间隔 |
| `BREATH_RATE_MAX` | 35 | 呼吸告警阈值（次/分钟） |
| `ALERT_COOLDOWN` | 1800s | 同类告警冷却 |

## 技术栈

YOLOv8n · Farneback Optical Flow · Peak Counting + FFT · OpenCV · PyTorch · 飞书 Bot API · frp

## 实测数据

二胖（家猫），静息状态：**21.6 次/分钟**（人工计数 21-22 bpm，系统测量吻合）

## 许可

MIT
