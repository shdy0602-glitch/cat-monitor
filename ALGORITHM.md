# Cat Monitor 呼吸频率算法 v2.0

## 一、帧采集

```
HTTP GET /shot.jpg 轮询，30秒
  ├── 每轮循环: HTTP请求(~0.27s) + sleep(0.5s)
  ├── 原始帧率: ~2.6 fps（约78帧/30s）
  ├── FRAME_SKIP=2: 每隔1帧取1帧 → ~1.3 fps（约35帧/30s）
  └── 记录每帧真实时间戳 time.time()
```

## 二、网格光流检测

```
帧缩放到 640px 宽 → Farneback 光流(全帧, 每对连续帧)
  ├── 4×4 网格，16 个 cell
  ├── 每 cell 累积 median 光流幅值 → 16 条时间序列
  └── 按 mean(|signal|) 排序，取最低 5 个 cell

候选 cell FFT 扫描:
  ├── 频段: 0.25–0.8 Hz (15–48 bpm)
  ├── SNR > 1.25 → 有效呼吸 cell
  └── score = SNR / mean_motion → 选最佳 cell
```

## 三、网格锁定

```
前 3 轮: 正常扫描选最佳 cell，记录 cell_history
第 4 轮起: 若同一 cell 3 轮中命中 ≥2 次 → 锁定
锁定后: 跳过扫描，直接用锁定 cell
解锁条件: 连续 2 次 SNR < 历史均值 × 60%
```

## 四、时域峰计数

```
smoothed = uniform_filter1d(signal, size=max(3, len//12))
find_peaks(smoothed, distance=1, prominence=std*0.04)
需要 ≥2 个峰
intervals = diff(peaks) / fps
IQR 过滤: 保留 1.5× IQR 内
需要 ≥2 个有效间隔
peak_rate = 60 / mean(valid)
```

## 五、FFT 分析

```
rFFT, 频段 0.25-1.2 Hz (15-72 bpm)
取 top-6 功率峰, SNR > 1.2× median → "strong"
fft_rate = 最强峰 × 60
谐波抑制: fft > 25 且 fft/2 ≈ peak_rate 或 ac_rate → weight = 0
```

## 六、自相关分析

```
ac = np.correlate(signal, signal) / ac[0]
搜索呼吸范围 lag (10-40 bpm)
取最强峰 → ac_rate, ac_strength (0-1)
```

## 七、三方法共识

```
候选构建 (method, rate, weight):
  ├── peaks: len(valid)≥3 时, weight = valid/10
  ├── fft:   SNR/5 换算 weight, 谐波抑制可置零
  ├── ac:    strength>0.15 时, weight = strength×2
  └── fft_hist: FFT 峰距历史均值 ±5bpm 内, weight × 1.5

谐波过滤:
  ├── 2×/3× 关系 → 剔除高频
  └── ÷2/÷3 关系 → 剔除低频
  容差: ±8 bpm

生理加权:
  ├── 12-35 bpm: × 1.3
  ├── <10 或 >50: × 0.4

加权平均: breath_rate = Σ(rate × weight) / Σ(weight)

次谐波校正 (在共识后, 仅 10-17 bpm):
  ├── 扫描 strong FFT 峰
  ├── |fft_peak - rate×2| < 8 且 18 ≤ fft_peak ≤ 35
  ├── 选最接近 21 bpm 的峰
  └── 校正为 fft_peak

历史更新: 成功检测后追加到 _rate_history (保持最近 4 次)
```

## 八、关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| FRAME_SKIP | 2 | 帧间隔采样 |
| CAPTURE_DURATION | 30s | 每轮采集 |
| CYCLE_INTERVAL | 300s | 检测间隔 |
| GRID | 4×4 | 网格密度 |
| GRID_SCALE | 640px | 光流缩放宽度 |
| GRID_BREATH_BAND | 0.25-0.8 Hz | 呼吸频段 |
| GRID_SNR_THRESHOLD | 1.25 | cell 有效性门槛 |
| GRID_CANDIDATE | 5 | 候选 cell 数 |
| prominence | 0.04 × std | 峰检测敏感度 |
| FFT band | 0.25-1.2 Hz | 分析频段 |
| FFT SNR | 1.2 × median | 强峰门槛 |
| autocorr height | 0.05 | 自相关峰最小值 |
| harmonic tolerance | ±8 bpm | 谐波判定宽限 |
| sub-harmonic range | 10-17 bpm | 触发校正范围 |
| sub-harmonic target | 18-35 bpm | 校正目标范围 |
| physio range | 12-35 bpm | 猫静息呼吸 |
| history size | 4 | 历史均值窗口 |
| history bonus | 1.5× | 历史匹配加成 |
| lock warmup | 3 cycles | 网格锁定热身 |
| lock threshold | 2/3 | 锁定命中比例 |
| unlock SNR ratio | 60% | 解锁门槛 |
| sanity check | 10-80 bpm | 合理范围 |
