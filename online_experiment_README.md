# Online SSVEP 实验使用说明

本实现保留原刺激生成、目标布局和 `get_data()`＋逐样本写入；实现三个可选冻结模型及完整阶段控制。
默认 M=4，算法 Aug_FB_TRCA；可选 M=2、Aug_FB_LDA、Aug_FB_eTRCA。一次实验只选择一种算法。

## 启动

在 `D:\experiment\ssvep_online` 的 PowerShell 中运行：

```powershell
uv run --locked python .\online_experiment.py --config .\online_config.json --method Aug_FB_TRCA --task-m 4
```

无提示使用：添加 `--online-mode uncued`；时长：`--online-seconds 120`。
正式采集会查找 USB8 放大器，设置并核实 2000 Hz；无需输入 COM3。
只有一台设备时自动连接；多台时必须用 `--device-name` 选择 SDK 返回的设备名称。
全屏显示使用主显示器；必须检测到配置的刷新率（默认120 Hz，600 Hz配置见README.md），不符则停止。
放大器准备完毕后，第一次按 SPACE 开始训练。训练结束后黑屏，模型拟合、冻结并追上实时数据后显示 `PRESS SPACE TO START`；准备好后再次按 SPACE，启动测试刺激。ESC 在任意阶段退出并保存已有数据。

没有放大器的软件联调：

```powershell
uv run --locked python .\online_experiment.py --simulate --headless --training-blocks 1 --train-cue-seconds 1 1 --skip-seconds 0.2 --min-samples 5 --online-seconds 2
```

`--simulate` 是合成 EEG，`--headless` 不打开显示器；去掉 `--headless` 可测试真实绘制＋合成 EEG。
这两种模式均不能用于验证真实 EEG 分类效果。仅测试时缩短训练，正式默认仍为 10 block。
仅 `--headless` 自动模拟第二次空格，方便自动测试；有真实窗口时（包括 `--simulate`）必须手动按第二次空格。

## 参数与计时

配置集中在 `online_config.json`，命令行指定的选项覆盖 JSON；不指定 JSON 时使用 `online_core.Config` 默认值。

| 配置项 | 默认值及含义 |
|---|---|
| fs | 固定 2000 Hz，启动时核实 SDK 设置 |
| window_size | 200 个原始样本＝100 ms；`[edge-100, edge+100)`，末端不含 |
| decimation | 2；原始采样率滤波后全局隔点保留，窗口为 100 点 |
| eeg_channels | `[0,2,3,4,5,6]`，从 0 开始，不包含第 9 列 trigger |
| enable_rereference | true；false 禁用重参考 |
| reference_channels | `[1,7]`；逐采样点减去这些通道的均值 |
| enable_notch / notch_hz / notch_q | true / 50 / 30；因果 IIR 陷波，与带通独立配置 |
| filter_order / bands | 4 阶 Butterworth 因果带通，10 个子带与当前离线脚本一致 |
| training_blocks | 10，每个 block 每个目标各提示一次，块边界也无相邻重复 |
| train_cue_seconds | `[3,5]` 秒，随机提示时长 |
| skip_seconds | 每段开始后跳过实际 0.5 秒；整个窗口必须处于有效区间 |
| min_samples | 每一个触发边沿×注视类别至少 20 个有效窗口；不是 20 次注视 |
| max_extra_blocks | 2，10 block 后仍不足才补充；上限仍不足则停止报错 |
| online_mode / online_seconds | cued / 120 秒；无提示选 uncued |
| online_cue_seconds | `[3,5]` 秒，仅限制相邻提示不重复；M=2 必然交替 |
| test_warmup_seconds | 0.5秒；第二次空格后从初始相位恢复闪烁，先缓冲，再开始正式online计时；可用 `--test-warmup-seconds` 覆盖 |
| feedback | true，仅 online 显示绿色预测标记，训练不预测 |
| refresh_rate / frequency | 120 Hz / 10 Hz；保留原按帧生成的方波及相位规则 |
| catchup_max_seconds / catchup_hold_seconds | 模型冻结且解码读指针落后写指针≤20 ms、接收数据新鲜，持续至少250 ms后显示第二次空格提示；按键时再检查 |
| max_prediction_lag_seconds | 100 ms：窗口末端相对最新已接收数据落后超过此值，则记录跳过，不发布陈旧结果 |
| ring_seconds / fit_timeout_seconds | 60 / 45 秒；冻结拟合期间继续采集，保存到缓冲区，之后顺序处理追赶 |

子带为 `[5,95], [12,95], [19,95], [27,95], [35,95], [43,95], [51,95], [59,95], [67,95], [75,95]` Hz。
本版使用与当前离线分析一致的因果 IIR 配置，不提供 FIR/非因果滤波切换；训练和预测使用同一滤波实例，阶段切换不清空滤波状态。
改变滤波、参考通道、窗口或分析通道，需要重新采集/训练，不能把旧模型直接用于新配置。

提示时长、跳过时长、online 总时长使用 `perf_counter()`；首帧 swap 返回时刻作为提示开始的软件估计。
截止时间在下一次绘制循环检查，实际结束通常相差一帧，卡顿时可能更大；保存实际持续时间。
闪烁按段内刺激帧编号计算：训练和测试各自连续闪烁，提示切换不重置相位。训练结束黑屏，第二次空格后测试刺激从各目标原先的初始相位重新开始；启动缓冲转入计分时不再次重置。全局记录帧号及EEG样本索引始终连续。按时间切换提示不会修复闪烁丢帧。

## 完整流程与训练/预测

`设备准备 → 第一次SPACE → 10 block有提示训练 → 黑屏排空边界窗口并检查样本 → 黑屏拟合、冻结、追赶 → 黑底PRESS SPACE TO START → 第二次SPACE → 初始相位启动闪烁＋缓冲 → 有提示/无提示online计分 → 保存`

训练结束后额外处理一个窗口长度的数据，排空已经触发且需要未来半窗的候选窗口；只接受整个窗口所有原始采样点都处于同一有效训练注视段的数据。
包括降采样丢弃的奇数位置，也要满足区间有效性，避免边界检查漏掉原始样本。
排空、拟合和追赶期间黑屏；等待第二次空格时只显示文字，目标intensity记录为0。持续采集、保存和因果滤波，不追加训练样本、不预测。缺少任意类别、退化模板或拟合失败时明确报错，不悄悄输出类别1。

若样本不足，在排空检查后恢复闪烁并补充训练block；这是一个新的刺激段，会重新启动相位，但不会重新初始化因果滤波器。训练每段的skip及整窗检查继续有效。

等待测试期间若数据不新鲜或解码又落后，会重新回到黑屏追赶，恢复后重新显示提示并要求新的按键。训练开始的空格、拟合时提前按的空格，以及按住空格产生的重复事件不会自动启动测试。

测试刺激从段内帧0开始，M4对应各目标0、π/2、π、3π/2的原始相位配置。有提示模式在启动缓冲时即显示第一个测试目标，并在进入计分后继续提示同一目标。第一段正式提示时长从计分首帧重新计时；无提示模式的缓冲和测试均无目标提示。0.5秒是可配置的初始缓冲值，不是已验证的生理稳态阈值。

每个边沿编号 r 都有一个模型组，区分全部 M 个注视类别 c。训练目标 c 时，所有目标的上升沿都能产生 `D[r,c]`。
直接按各目标上升沿构建窗口，不使用离线 global_reference/tau1 动态重切和重训。

| 方法 | 数据结构和计算 |
|---|---|
| 公共训练输入 | 每个 r：`X[N,10,6,100]`，`y[N]`；内部类别、边沿编号从 0 开始，界面和预测日志从 1 开始 |
| Aug_FB_LDA | 每个 r、每个子带的输入展平为 `[N,600]`；基础 LDA 为 lsqr＋auto shrinkage；拼接其训练集 decision scores，拟合最终 LDA。M2 拼接后 `[N,10]`，M4 为 `[N,40]`；保持离线的训练集内 stacking 规则 |
| Aug_FB_TRCA | 每个 r/b/c 的样本 `[N_rc,6,100]`；模板 `[6,100]`，空间滤波器 `[6]`；同类投影相关后按 `b**(-1.25)+0.25`（b 从 1 开始）加权子带有符号相关 |
| Aug_FB_eTRCA | 相同模板；每个 r/b 拼接各类别空间滤波器为 `[6,M]`；测试和各模板投影为 `[M,100]`，展平相关，再按相同规则加权 |

所有上升沿均可触发，不再设置原来的全局 100 ms 最小解码间隔。10 Hz 下 M2/M4 理想候选速率约 20/40 次每秒。
预测等待后半窗数据到齐，只选触发边沿 r 的冻结模型，比较所有类别。online 可跨提示切换取窗，不读取真实提示标签参与预测，不随提示重置旧预测。
每个窗口还必须属于同一个非零stimulus_id。黑屏转亮的首次状态只初始化边沿检测，不能触发正常周期解码；跨黑屏或跨刺激段的窗口被排除。online窗口中心必须属于正式online，全部原始样本须位于该次测试的缓冲或online阶段，因此第一窗可使用缓冲末尾的数据，但不能使用训练或黑屏数据。即使缓冲设为0，仍保留这些边界约束。滤波状态和全局降采样位置不随黑屏/重启清零。
瞬时陈旧结果会跳过并记录；持续跟不上实时、缓冲区溢出或采集无数据时停止，避免给出看似正常的失效记录。

## 时间准确率

仅 cued 模式计算 `正确输出累计时间 / 实际 online 总时间`。
使用主机单调时钟：提示在软件 swap 返回时切换；模型输出在实际发布时切换；两者分段积分。
初始无预测和提示切换过渡均计入分母，切换后保留旧输出。窗口时间不能替代结果发布时间。
黑屏、等待第二次空格、启动缓冲均不计入online时长或准确率分母；进入正式计分后没有输出的时间才计入分母。测试重启时预测清零，之后提示切换不清零。
主指标是解码器当前输出的时间准确率；反馈标记下一帧才出现，因此 `frames.csv` 另存显示的预测，不把两种时刻混同。
uncued 不生成准确率，不要求提供注视真值。

## 采集与原处理的区别

- 保留 SDK 队列模式 `get_data()`，一次取回一批，对每行分别写入并更新索引；没有改为 callback 或块写入。
- 一批数据仍使用一次状态快照。逐行写入不意味着批次内部每个样本的 intensity 时刻已被精确恢复。
- 增加整体状态快照锁，主线程在 swap 后发布完整状态；缓冲区读写使用短锁，防止竞争。它们的开销需要在完整硬件负载下观察。
- 同一批次记录 get_data 返回、状态快照前后及写入完成时间；逐帧记录 swap 前后、帧间隔和 intensity。没有安装解析回调探针、没有重分配批次内部的边沿。
- 数据从预热完成后开始连续保存，包括 SPACE 前等待和拟合/追赶阶段；索引 0 对应保存文件第 0 行，可直接回查训练窗口与预测窗口。
- 启动预热期间 SDK 异常计数另存。正式阶段出现新的序号/校验异常，保存异常批次并停止；事件数不解释为精确丢样本数。当前不能定位异常的精确样本位置，因此不继续假装时间轴连续。
- 这些时间都是软件时间，不能单独证明 ADC 采样时刻、USB 之前的延迟或屏幕真实发光时刻。

## 输出文件

默认 `项目目录/online_results/时间戳_随机后缀/`；显式 `--output` 必须是尚不存在的目录。

| 文件 | 内容 |
|---|---|
| config.json / summary.json | 完整配置、完成/错误状态、训练block数、实际online时长、准确率；测试就绪、空格接受、刺激恢复、计分开始时刻和实际缓冲时长；阶段编号映射 |
| EEG.mat / States.mat | 原始9列EEG（含trigger）、兼容原SharedState字段的States、Fs；States.mat另存metadata及metadata_columns |
| recording.npz | 同上及额外metadata；列顺序 `label,train_valid,stage,segment,valid,intensity_1...M,stimulus_id,stimulus_frame`，并存列名 |
| frames.csv / events.json | 全局连续frame、每帧软件swap时间、间隔、intensity、显示预测；刺激段stimulus_id及段内stimulus_frame；阶段/提示切换事件 |
| batches.csv | 原始采样索引范围、get_data 返回/状态快照/写入完成时间、SDK 异常计数 |
| decoder_batches.csv | 解码处理块时长和采集写指针落后量（不等于物理端到端延迟） |
| training.json | M×M 样本计数、实际取窗索引、拟合时长；取窗表内 edge/class 从 0 开始 |
| frozen_model.joblib | 完整配置与冻结模型，包含 LDA 或空间滤波器/模板；只加载可信的模型文件 |
| predictions.json | 触发编号、原始窗口起止、分数、实际发布时间、输出类别，以及陈旧结果跳过记录 |
| *_error.json | 发生异常时的错误和栈，检查后再开始新实验 |

输出时间为同一主机 `perf_counter` 秒值，不是 UTC；用 events 中的时刻相减得到相对时间。
黑屏/等待时stimulus_id=0、stimulus_frame=-1；每次恢复闪烁使用新的正整数stimulus_id、段内帧从0递增。训练提示切换以及测试缓冲→正式计分不改变stimulus_id。原有阶段编号0～6保留，新增加7=waiting_test、8=test_warmup。旧记录没有这两个metadata末尾字段，分析时应按保存的列名/版本区分。
程序有序退出时保存内存记录，异常强制杀进程或断电可能丢失尚未落盘的数据。

## 验证

```powershell
uv run --locked python -m unittest test_online_experiment -v
uv run --locked python .\verify_online_runs.py
```

第一条检查滤波连续性、全部原始点的训练窗口边界、模型公式/维度、冻结规则、时间积分、缓冲区覆盖与时钟调度。
同时检查黑屏不绘制目标、就绪前不能接受第二次空格、丢失就绪状态后重新追赶、启动边沿排除、跨刺激段窗口排除、相位帧重置和因果滤波连续性。
第二条真实启动采集/解码子进程，以合成 EEG 覆盖 3 算法×M2/M4 和无提示分支，并回查保存文件。其合成准确率不代表真人表现。

已有真实 EEG 的顺序回放示例（数据不随项目分发，请自行提供或替换数据路径）：

```powershell
uv run --locked python .\online_experiment.py --task-m 4 --method Aug_FB_TRCA --replay-eeg .\Dataset\sub10\VEP_Exp2_M4_Active_EEG_20261008_201507.mat --replay-states .\Dataset\sub10\VEP_Exp2_M4_Active_States_20261008_201507.mat --replay-train-segments 10
```

回放按每批 10 点顺序送入同一滤波/取窗逻辑；前 10 个完整注视段训练，后续段仅预测，边界处冻结，不重置滤波。测试窗口必须完整位于划分边界之后，避免训练/测试窗口共享原始样本；因果滤波状态自然延续历史数据。
这不是 10 个 block 的正式实验，也不重现当年 USB 接收或软件显示时刻；只输出重叠窗口诊断准确率，**不输出 online 时间准确率**。
回放参数不按测试结果自动调优。软件通过后，仍需放大器空采＋120 Hz 窗口联调，再用真实注视验证分类性能。
该MAT回放入口用于旧的全程连续刺激记录，不会人为插入黑屏或相位重启，也不支持直接把新范式含黑屏的整场States当作旧的连续有提示记录读取。
