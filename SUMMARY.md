# 猫咪呼吸监控系统 — 项目总结 v1.2

## 项目目标

搭建 7×24 小时猫咪呼吸频率监测系统：摄像头采集 → 猫检测 → 呼吸频率分析 → 飞书告警。

## 最终架构

```
安卓手机(IP Webcam + Termux/frpc)
        ↓ frp TCP 穿透
腾讯云轻量服务器(Docker)
  ├── frps (接收 frpc 连接，映射 18080 端口)
  └── cat_monitor (每5分钟: 抓帧30s → YOLO检测 → 光流分析 → 峰计数+FFT → CSV+飞书)
        ↓
飞书群告警
```

## 验证结果

| 指标 | 数值 |
|------|------|
| 帧率 | ~40 帧 / 30 秒（FRAME_SKIP=2，帧间隔 0.75s） |
| 猫检测 | 8-20 帧命中（依赖光线条件） |
| 呼吸频率 | 21.6 次/分钟（人工计数 21-22，吻合） |
| 飞书推送 | 正常 |

## 关键技术

- **YOLOv8n**：猫检测 + teddy bear/dog 备选类别（昏暗光线下猫易误识别），conf=0.08
- **Farneback 光流**：躯干区域 ROI 30-70% 运动幅值提取，median 聚合
- **峰计数 + FFT 交叉验证**：峰计数为主（prominence=0.08σ），FFT 频段 0.17-1.2Hz 验证
- **谐波抑制**：检测 2× 谐波关系，避免吸+呼产生倍频
- **快照轮询**：HTTP GET `/shot.jpg`，~1.3fps，比 MJPEG 更抗断流
- **bbox 中值平滑**：3 帧窗口平滑 YOLO 抖动，静止判定阈值 35%
- **frp 穿透**：TCP 隧道将手机局域网 8080 暴露到公网

## 依赖

- 安卓手机（IP Webcam + Termux + frpc）
- 云服务器（Ubuntu 22.04，Docker，≥2GB 可用内存）
- 飞书应用（App ID + Secret，群机器人权限）

## 已知限制

- frp 隧道偶发断连（运营商 CGNAT / Android 后台策略），检测窗口与断连重合时丢帧
- cat_monitor 容器占用 ~1.1GB 内存（PyTorch），轻量服务器上不能并行跑多个模型推理服务

## 相关仓库

- [cat-monitor](https://github.com/shdy0602-glitch/cat-monitor) — 本项目
- [micam-xiaomi-rtsp](https://github.com/shdy0602-glitch/micam-xiaomi-rtsp) — 小米摄像头 RTSP 方案（已废弃，备份）
