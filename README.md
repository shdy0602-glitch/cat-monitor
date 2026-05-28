# Cat Breath Rate Monitor v2.0

非接触式猫咪呼吸频率监测。安卓手机 + 云服务器，4×4 网格光流 + 峰计数 + FFT + 自相关三方法共识，超阈值飞书群告警。**不再依赖 YOLO 物体检测。**

## 架构

```
安卓手机(IP Webcam) → frp 穿透 → 腾讯云(Docker) → 飞书群告警
```

| 组件 | 位置 | 作用 |
|------|------|------|
| IP Webcam | 安卓手机 | 摄像头推流（局域网 8080） |
| frpc (Termux) | 安卓手机 | TCP 隧道，8080 → 云:18080 |
| frps | 云服务器 Docker | 接收 frpc 连接 |
| cat_monitor | 云服务器 Docker | 网格光流检测 → 三方法共识 → 告警 |

## 检测流程

```
每5分钟: 抓30秒快照(~1.3fps) → 4×4网格全帧光流
                                    ↓
                              找到运动最低的5个网格
                                    ↓
                              FFT 扫描呼吸频段(0.25-0.8Hz, 15-48bpm)
                                    ↓
                              最佳网格 → 峰计数+FFT+自相关三方法共识
                                    ↓
                              呼吸频率 → CSV + 飞书告警
```

## 核心特性

- **无 YOLO**：纯光流网格检测，不受猫姿势、光线影响
- **网格锁定**：3 轮热身期后锁定最佳检测区域，稳定性提升
- **三方法共识**：峰计数 + FFT + 自相关加权平均，谐波/次谐波自动抑制
- **历史权重**：FFT 峰靠近近期均值时获得 1.5× 加成
- **次谐波校正**：检测到半频信号时自动修正到真实频率

## 告警规则

- **呼吸急促**：> 35 次/分钟 → 飞书群推送（同类 30 分钟冷却）
- **每小时汇总**：推送过去 1 小时每次检测结果和信号质量（逐条列出，不做平均）
- **飞书接入**：App ID + Secret 方式，调用 `im/v1/messages` API

## 部署

### 1. 视频源

安卓手机安装 IP Webcam + Termux + frpc：

```toml
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
| `CYCLE_INTERVAL` | 300s | 检测间隔（5 分钟） |
| `BREATH_RATE_MAX` | 35 | 呼吸告警阈值（次/分钟） |
| `ALERT_COOLDOWN` | 1800s | 同类告警冷却 |
| `GRID_ROWS × GRID_COLS` | 4×4 | 网格密度 |
| `GRID_BREATH_BAND` | 0.25-0.8 Hz | 呼吸频段（15-48 bpm） |

## 技术栈

OpenCV Farneback Optical Flow · Grid Motion Detection · Peak Counting + FFT + Autocorrelation · scipy · 飞书 Bot API · frp

## 实测数据

二胖（家猫），静息状态：**22-25 次/分钟**（人工计数 21-25 bpm，系统测量吻合）

## 许可

MIT
