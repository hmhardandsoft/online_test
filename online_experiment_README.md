# SSVEP 在线实验

## 运行

在本项目目录安装依赖后，根据实际显示器选择配置文件。600 Hz 示例：

```powershell
uv run --locked python .\online_experiment.py --config .\online_config_600hz.json --task-m 4
```

120 Hz 使用 `online_config.json`。必须在显示器实际刷新率与配置相同时运行。软件联调：

```powershell
uv run --locked python .\online_experiment.py --config .\online_config_600hz.json --simulate --headless --training-blocks 1 --train-cue-seconds 1 1 --skip-seconds 0.2 --min-samples 5 --online-seconds 2
```

`--simulate` 生成合成 EEG，不测量真人分类效果；`--headless` 不打开刺激窗口。真实窗口运行需按四次空格：一次启动训练，三次分别启动测试。ESC 中断后仍保存已记录数据，`summary.json` 标记未完成。

命令行 `--method` 仅供旧 MAT 单模型回放使用；真实在线运行固定测试全部三种方法。配置中的 `method` 也只决定 MAT 回放选择，与三段在线测试的随机顺序无关。

## 一次运行的流程

1. 放大器连接、开始采集并完成预热；确认接收数据后，黑底出现第一次 `PRESS SPACE TO START`。就绪前的空格不会启动训练。
2. 按空格开始 10 block 有提示训练。每个目标每 block 一段，提示 3–5 秒；每段前 0.5 秒不用于训练。训练中没有预测反馈。如果每个触发边沿 × 注视类别不足 20 个有效窗口，最多补 2 block。
3. 训练结束后黑屏，排空边界窗口，从**同一批有效训练窗口**拟合 `Aug_FB_LDA`、`Aug_FB_TRCA`、`Aug_FB_eTRCA`，一次性冻结。采集持续进行；解码读指针追上实时数据并稳定后，显示第一段测试的空格提示。
4. 三种方法各测试一次，顺序在每次运行时随机排列。各段有独立随机目标序列和提示时长，保存随机种子。相邻提示目标不重复。每段开始前黑屏等待模型已冻结、数据接收新鲜、解码已追上；就绪后显示空格提示，接受新的空格按键。
5. 每次按空格后，刺激从初始相位、局部帧号 0 重新开始；先进行可配置的 0.5 秒缓冲，再进入正式计分。缓冲与正式计分之间不断闪、不更换首个目标。段间黑屏；上一段预测清零，不可进入下一段。
6. 第三段结束后保存 EEG、States、逐帧、事件、预测、各方法准确率及定位记录。

训练提示顺序仍由 `seed` 控制。三段测试的随机算法顺序和各自提示序列由新的运行随机种子控制，种子写入 `summary.json`，便于复现。如果比较跨受试者算法表现，应在实验设计中平衡六种算法顺序，避免学习和疲劳效应与算法混淆。

## 停止方式

`online_stop_mode` 或 `--online-stop-mode` 选择：

| 值 | 结束规则 | 配套参数 |
|---|---|---|
| `time` | 每段正式在线计分达到设定实际经过时间后，在下一帧切黑。最后一个提示可以不完整。 | `online_seconds` / `--online-seconds`，默认 120 秒 |
| `segments` | 每段完成设定数量的完整提示后，在下一帧切黑。实际总时长随提示时长变化。 | `online_cue_count` / `--online-cue-count`，默认 30 段 |

无提示模式 `online_mode=uncued` 只允许 `time`，不计算注视准确率。固定完整提示段数仍按实际时间计算准确率，不转换为试次准确率。三段实际时长、完整段数、最后一段是否部分呈现分别记录；不同算法的总时间准确率按各自实际时长归一化。若比较算法，仍应检查每种目标的总暴露时长是否相近。

## 解码和计分

`fs=2000`，`window_size=200` 个原始采样点，即 100 ms，窗口为上升沿前后各 50 ms。所有目标上升沿都能触发候选窗口，每个边沿的模型分别比较全部 M 个注视类别。M 可选 2 或 4；训练/测试使用同一连续因果滤波器和滤波参数。10 个滤波子带为 `[5,95]`、`[12,95]`、`[19,95]`、`[27,95]`、`[35,95]`、`[43,95]`、`[51,95]`、`[59,95]`、`[67,95]`、`[75,95]` Hz。可配置是否工频陷波、重参考开关和重参考通道。方法间共用训练窗口，但各自保存独立参数。

完整训练窗口必须处于同一有效训练注视段。在线窗口可跨同一测试段内的提示切换，但不能跨黑屏、缓冲开始边界或不同 `stimulus_id` / `test_id`。窗口后半部分数据到齐才解码；过时或跨段结果跳过。预测在实际发布时更新，发布之间保持上次标签；同一测试段内提示切换不清空预测。每段测试开始时才清空上一段结果。

有提示时间准确率 = 当前保持的输出与提示目标一致的累计时间 / 该段正式在线实际总时间。首个预测发布前和提示切换后的错误保持时间均计入分母。计时依据主机 `perf_counter()`：提示按软件画面交换返回时刻，预测按解码进程发布时刻；不是屏幕实际发光或 ADC 精确采样时刻。无提示测试保存预测，不给准确率。

## 主要输出与异常定位

- `summary.json`：三段测试各自的方法、顺序、随机种子、准备/按键/缓冲/正式开始/结束时间、完整提示数、实际时长和准确率。`config.json` 保存配置。
- `recording.npz`、`EEG.mat`、`States.mat`：原始采集与状态。元数据列由文件内 `metadata_columns` 给出；新版在末尾增加 `test_id`。不要按旧版固定列数解析。
- `frames.csv`、`events.json`、`predictions.json`：逐帧及事件时间、局部刺激帧号、测试编号、方法和预测发布/跳过原因；全局帧号和 EEG 样本号全程连续。
- `training.json`、`frozen_model.joblib`：训练窗口索引与计数、三种冻结模型。新版模型文件使用 `models` 字典，旧单模型文件使用 `model` 字段。
- `batches.csv`、`decoder_batches.csv`、`frame_timing.csv`、`gc_events.json`、`window_events.json`、`timing_diagnostics_summary.json`：USB 接收、解码积压和主线程帧间隔定位记录。长间隔同时记当前及前一帧的测试编号，便于判断是否在段间切换。

详细时序定位字段见 `timing_diagnostics_README.md`。记录是软件侧观测，不直接测量设备 ADC 采样时刻或屏幕真实发光时刻。异常定位记录默认开启；可以用 `--no-timing-diagnostics` 关闭。

## 验证

```powershell
uv run --locked python -m unittest test_online_experiment test_timing_diagnostics -v
uv run --locked python .\verify_online_runs.py
```

第一条覆盖滤波、模型、窗口、状态机与计分；第二条分别跑 M2/M4、定时/定段数和无提示的完整合成流程，并回算每段结果。软件通过后仍需在目标电脑上使用实际放大器和显示器检查帧间隔、丢包、解码落后与真人表现。

绘制卡顿根因定位：新增进程/线程信息、ETW时间锚点、Windows系统跟踪脚本及分析步骤，见 [render_trace_README.md](render_trace_README.md)。系统跟踪需单独开启；普通实验默认只增加启动与退出时的轻量上下文记录。
