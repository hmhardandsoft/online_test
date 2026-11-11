# 主线程定位记录

`timing_diagnostics` 默认为 `true`，120 Hz 和600 Hz配置均开启。用于定位两次画面交换之间出现的长间隔；正式采集时建议保留，经目标电脑空采确认开销后保持配置一致。

```powershell
uv run --locked python online_experiment.py --config online_config_600hz.json --method Aug_FB_eTRCA
# 仅在开销对照时关闭新增定位记录；原有frames/batches等记录仍保留
uv run --locked python online_experiment.py --config online_config_600hz.json --method Aug_FB_eTRCA --no-timing-diagnostics
```

## 新增文件

|文件|用途|
|---|---|
|frame_timing.csv|每帧循环、阶段控制、预测锁访问、绘制、glFlush、swap、事件处理、共享状态发布、事件日志、帧日志和定位记录存储的时间边界|
|gc_events.json|主进程Python垃圾回收开始/结束、代数、线程编号及当时帧号|
|window_events.json|初始焦点/最小化状态，以及焦点、最小化、窗口重绘和帧缓冲尺寸变化事件|
|timing_diagnostics_summary.json|超过1.5个理想帧周期的全部间隔，按时长排序；各步骤耗时、最长步骤及重叠GC/附近窗口事件索引|

600 Hz时阈值为2.5 ms，120 Hz时为12.5 ms。阈值只用于输出异常列表，不更改刺激、提示、训练窗口有效性或准确率。

## 如何阅读

先查看`timing_diagnostics_summary.json`中的`long_intervals`，然后按`frame`与原有`frames.csv`连接。对间隔结束于第n帧的异常，耗时包含**第n-1帧交换后的工作，以及第n帧交换前的工作**：

- `previous_poll_events`：上一帧swap返回后至事件处理结束，包含少量计时存储开销。
- `previous_stage_bookkeeping`：阶段时间设置及阶段切换打印。
- `previous_present_state`：共享状态快照发布，包含锁等待。
- `previous_event_log`：事件记录和训练排空边界设置。
- `previous_frame_log`：原有帧字典/亮度JSON记录。
- `previous_diagnostic_store`：新增定位记录尾部存储。
- `between_iterations`：上一轮尾部到下一轮开始的间隔。
- `control_and_schedule`：下一轮健康检查、阶段调度、积压检查，以及定位缓冲分配。
- `cue_and_phase_setup`、`prediction_lock_and_read`、`intensity_setup`：提示、相位与当前预测读取。
- `draw_commands`：清屏、目标、提示和反馈绘制命令。
- `gl_flush`：提交绘制命令。
- `before_swap_gap`、`swap_call`：交换前剩余间隔与交换调用。

上述分段之和等于该次完整swap返回间隔。一次异常可同时包含多个慢步骤；`largest_component`只指耗时最长的区间，不能单独证明该函数的CPU计算慢，因为测量也包含系统调度或驱动等待。

`overlapping_gc`是gc_events.json中的索引；`nearby_window_events`是window_events.json中位于异常前后50 ms范围的事件索引。时间重合是定位线索，不等于因果结论。垃圾回收的记录范围是主进程，不能据此推断采集或解码子进程是否发生GC。

最后一次循环可能在阶段结束/退出检查处停止，因此CSV可比已呈现帧多一行，未执行到的时间边界为空；异常汇总只使用完整行。无窗口合成测试中`swap_call`包含模拟等待，绘制区间为空操作，不能拿它评价真实显卡或显示器。

## 开销与正式实验

运行期间只写内存，不逐帧写磁盘或打印，也不增加采集回调线程。逐帧数据采用按块分配的数值数组，约128字节/帧；600 Hz运行5分钟约23 MB数值数据，加上最后一个未填满块和少量事件。实验结束、采集停止后，才导出CSV和异常摘要。长实验/长时间等待会继续增长，结束时文件保存也会更久。

它不修改EEG、滤波、训练/预测规则或准确率，但新增计时、内存写入和回调总有开销；主线程和采集进程共享机器资源，所以不能声称绝对不影响数据采集。建议在实际600 Hz电脑上做一次开启/关闭的完整空采对照，查看长帧尾部、接收间隔、SDK异常和解码积压，随后正式实验固定配置。

纯记录微基准（本开发电脑，每模式两次、每次100000帧；无刺激/放大器）：

|模式|每帧记录耗时中位数|P95|
|---|---:|---:|
|原有记录模式|1.3微秒|2.2–2.3微秒|
|新增记录关闭|1.8微秒|2.7–2.8微秒|
|新增记录开启|3.4微秒|4.4–4.5微秒|

开启相比原模式的P95约增加2.2微秒，约占600 Hz的1.667 ms帧周期的0.13%。这是记录代码本身的测试，未覆盖真实OpenGL、窗口事件分发和设备负载，不能视为目标电脑零影响证明。关闭模式仍保留轻量函数调用，因此并非完全等同原始程序。

可重复测试：

```powershell
uv run --locked python -m unittest test_online_experiment test_timing_diagnostics -v
uv run --locked python benchmark_timing_diagnostics.py --frames 100000
```

初版定位记录的验证：18项单元测试通过；600 Hz计划节奏的合成流程在开启/关闭两种配置下均完成训练、冻结、等待、缓冲与online，各发布78条预测；开启时4287个已呈现帧与定位记录逐行匹配，长间隔分段之和一致。没有运行真实放大器或物理600 Hz显示器。

目前只增加定位信息，尚未根据长帧自动排除训练窗口。可先积累异常及其归因证据，再单独确定质量阈值和保护规则。

# 三段测试的归属

新版 `frames.csv` 与 `events.json` 含 `test_id`、`method`；`predictions.json` 还记录触发窗口所属的测试编号和算法。`timing_diagnostics_summary.json` 的长帧间隔同时含当前帧和前一帧的 `test_id`/`method`，并给出 `long_interval_count_by_test`。段间切黑帧可能属于 `test_id=0`，所以定位跨边界停顿时应查看 `previous_test_id`。记录仍按全局帧号关联，不会因每段刺激局部帧号重置而冲突。

# 绘制步骤与主线程 CPU 时间（记录格式 v2）

`timing_diagnostics=true` 时，`frame_timing.csv` 在原有列之后增加：

- `clear_end`、`identity_end`：`glClear` 与 `glLoadIdentity` 返回后的主机时间。
- `target_1_end` 至 `target_4_end`：各目标绘制返回后的时间；M2 的后两列留空。
- `cue_end`、`feedback_end`、`prompt_end`：提示标记、预测反馈、空格提示文字绘制返回后的时间。未执行的步骤留空。
- `draw_cpu_start`、`draw_cpu_end`：在刺激主线程读取的 `time.thread_time()` 累计 CPU 秒数。两者之差覆盖整个绘制区间及少量测量开销，不含 `glFlush` 和交换缓冲。

原有 `draw_begin` 至 `draw_end` 仍表示整个绘制调用范围。各子步骤使用相邻时间点相减，包含调用间少量 Python 控制与测量开销；这些数据测量主机调用返回耗时，不是显卡执行耗时。绘图命令和调用顺序保持原样，没有增加强制 GPU 同步。

`timing_diagnostics_summary.json` 的每个长间隔新增 `draw_components_ms`、`largest_draw_component`、`draw_wall_ms`、`draw_cpu_ms`、`draw_cpu_fraction`。子步骤加上 `draw_tail` 的和等于该帧的 `draw_commands`，不能再把子步骤与整体重复相加。如果长间隔实际出现在前一帧日志或当前交换缓冲，应优先看原有 `largest_component`；新增绘制数据只描述当前帧。

若绘制实际经过 58 ms，线程 CPU 只增加很少，说明期间该线程没有持续消耗 CPU；仍不能直接区分驱动等待和 Windows 暂停调度。CPU 时间较大也可能包含底层库或驱动忙等待，不能直接归因为 Python 算法。系统时钟的精度会影响短区间的 CPU 差值；零值不能作为“没有计算”的证据，比例也不做强制截断或自动归因。

汇总保存 `clock_info`（时钟实现、报告分辨率等）和 `draw_thread_native_id`，供后续系统线程跟踪使用。`--headless` 没有真实 GL 绘制，因此这些子步骤及 CPU 字段留空；不会伪造 GPU 测量。未执行步骤不影响正常帧纳入汇总；退出循环时已开始但未完成的帧计入 `incomplete_frame_rows`。

测量值仍写入分块预分配的数值内存，停止实验后统一保存。关闭原有 `--no-timing-diagnostics` 开关也会关闭新增时钟读取。M4 每个真实绘制帧增加约 7 次实际时钟读取和 2 次线程 CPU 时钟读取；包含预测反馈时再增加一次。27 列缓冲在 600 Hz、5 分钟下约占 37 MiB，实际按 8192 帧分块分配。

运行 `benchmark_timing_diagnostics.py` 可比较 `legacy`（仅原有帧日志）、`coarse`（原分段时钟读取，使用新版缓冲布局）、`enabled`（增加细分与 CPU 记录）、`disabled`。这是纯记录开销检查；真实显示器上的最大帧间隔仍需在相同 600 Hz/M4 条件下对照测量。
本轮验证：24 项单元测试通过；M4 合成三段测试完成，7721 帧与完整计时记录逐行匹配，发布 139 个预测。每模式两轮、每轮 100000 帧的纯记录基准中，`coarse` 的 P95 为 4.5–4.6 微秒，`enabled` 为 5.7–5.8 微秒，约增加 1.2 微秒/帧。这个差值是本机记录开销测量，不是实际 600 Hz 显示器上的卡顿改善结果。
