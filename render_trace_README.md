# 绘制停顿的系统跟踪

目的：把 `frame_timing.csv` 中的 `glClear` 等异常区间对应到 Windows 线程调度、CPU调用栈和图形活动。新增记录不是卡顿修复；定位根因后再做单因素修改验证。

## 自动记录什么

`timing_diagnostics=true` 时，每次 live run 自动保存 `render_trace_context.json`：

- run_id、绘图进程 PID/主线程 native TID、采集和解码进程 PID。
- Python/Windows版本、GL vendor/renderer/version、GLFW版本、实际窗口帧缓冲大小、显示模式刷新率、程序请求的 swap_interval。
- workers_started/run_stop 两个时间锚点、QPC原始计数和频率、UTC对应关系。
- ETW provider是否被启用、写入返回码、标记前后 perf_counter 时间及不确定区间。

新模块只在窗口初始化后、初次就绪提示前和退出帧循环后工作，不在每个刺激帧调用，不改变glClear、glFlush、swap顺序，也不增加glFinish。上下文在初次等待前保存一次，结束时更新；即使运行崩溃，首次保存仍可能保留。`--no-timing-diagnostics` 同时关闭此功能。

ETW使用固定GUID `8c88351d-6ab9-480a-83a3-96a7df776154`，字符串前缀 `SSVEP_RENDER`。这里仅用 EventWriteString 写两个简单诊断标记，未建立通用事件模型。返回0不证明ETL已收集事件；必须同时检查provider_enabled并在ETL确认相同run_id。未启动WPR时，程序仍照常实验，不自动启动系统级采集。

## 一次实际空采跟踪

在**实际出现卡顿的600 Hz电脑**操作。开发电脑检测到的工具安装状态不能代替该电脑的检查。第一次只需要训练阶段；如要查看训练与测试同步位置的差异，再完成一段测试。

1. 安装/确认 Windows Performance Analyzer（WPA）。WPR是录制工具，WPA是查看ETL的工具，不是实验Python依赖，无需加入uv。可从微软Windows ADK安装 Windows Performance Toolkit。先运行 `Get-Command wpr.exe,wpa.exe -ErrorAction SilentlyContinue`；WPA也可能安装在自定义目录或Store应用中，未在PATH中找到不等于没有安装。
2. 管理员PowerShell进入仓库，执行下面的Start命令。目录必须是新目录；Start先保存显卡驱动/系统/电源方案快照，再启动GeneralProfile、GPU和自定义标记配置。不会自动取消其他跟踪。

```powershell
cd D:\experiment\ssvep_online
powershell -NoProfile -ExecutionPolicy Bypass -File .\capture_render_trace.ps1 -Action Start -TraceDirectory D:\experiment\ssvep_online\render_traces\capture_01
```

3. 在另一个普通终端运行原有online命令，使用已经确认的600 Hz配置（仓库默认配置可能仍为120 Hz，不能直接假设它适合本机）。保留当前采集/滤波/绘制设置。出现明显卡顿后正常退出实验，待结果保存完成。
4. 回到管理员终端，停止并保存系统跟踪：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\capture_render_trace.ps1 -Action Stop -TraceDirectory D:\experiment\ssvep_online\render_traces\capture_01
```

输出 `render.etl`、`system_snapshot.json`、`capture_state.json`。同时保留对应online_results目录。需要查看录制状态时用 `-Action Status`。若实验异常退出，仍需要执行Stop。若WPR报冲突，先查看已有录制，脚本不会擅自取消它。文件模式会占用磁盘，适合一次有限时长训练；不要整天运行。ETL可能包含系统上其他进程路径/活动，分析前保留本地，不提交Git。

## 分析步骤

1. WPA打开ETL，检查是否有事件丢失。确认有CPU Usage (Sampled)、CPU Usage (Precise)/线程切换，以及GPU/图形事件；不同Windows版本的图形表内容不同。缺失对应数据时不能根据空表认定无等待。
2. 在Generic Events中按上述provider GUID过滤，找到字符串 `SSVEP_RENDER`、正确run_id和name=workers_started，记下此事件在WPA时间轴中的秒数。必要时展开String/Message类字段。字符串事件不需要项目清单注册；若当前查看器不显示文本，可用tracerpt将ETL导出XML，按GUID、PID、线程号核对。不能拿另一run的标记对齐。
3. 生成异常列表的WPA时间范围（将路径和12.345替换为实际值）：

```powershell
uv run --locked python export_render_trace_windows.py .\online_results\YOUR_RUN --marker-seconds 12.345
```

输出 `render_trace_windows.csv`，含异常帧、PID/TID、起止时间上下界。未提供marker-seconds时输出相对标记的偏移。上下界来自EventWriteString调用前后包围区间，并非物理显示误差；若锚点本身被暂停而区间过宽，可用 `--marker-name run_stop` 并提供该结束事件在WPA中的秒数核对，或重新录制。不要直接把perf_counter大数当成WPA相对秒数。
4. 在异常区间定位绘图主线程：CPU采样看它运行时的调用栈；CPU Precise看Running/Ready/Waiting及唤醒关系；图形事件看同步/队列/其他进程负载。同时对照frame_timing的clear区间、采集与解码PID。加载微软公共符号；显卡厂商无公开符号时可能只能定位到驱动模块，不能承诺具体内部函数。
5. Running且驱动栈占主导，检查驱动内部工作/忙等；Ready长，检查调度竞争、DPC/ISR；Waiting长，检查等待栈和图形事件。GPU繁忙或时间重合本身不是因果证明。以同一位置附近正常帧为对照。
6. 根据证据一次改一项，再比较每分钟长帧数、严重程度和采集/解码积压。对照一次WPR开/关，判断系统跟踪自身开销。ETW适合定位轮次，不建议默认在全部正式实验开启；轻量上下文可保留。

## PATH是什么

PATH是Windows搜索可执行程序的目录列表。直接输入 `wpa.exe` 时终端仅在这些目录等搜索位置查找。已安装但未加入PATH时，可以使用开始菜单，或显式路径：

```powershell
& 'C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\wpa.exe' 'D:\experiment\ssvep_online\render_traces\capture_01\render.etl'
```

这只是常见路径，先确认文件存在。没有WPA不会影响刺激或EEG采集，只影响查看系统跟踪结果。

参考：[WPR命令行](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/wpr-command-line-options)、[CPU分析](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/cpu-analysis)、[Windows Performance Toolkit](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/)。

## 本次验证范围

29项单元测试通过，包括时间轴上下界、禁用开关、ETW故障降级、Windows真实provider注册/写入/注销以及既有解码和阶段控制检查。一次M4无窗口合成流程完成训练与三段测试，新增文件正确保存3个进程编号、主线程编号、两个锚点，errors为空。PowerShell语法检查与WPR自定义配置解析通过。

当前执行终端不是Windows管理员，因此未运行系统级WPR采集，也未验证ETL在WPA中的实际呈现。provider API测试不能替代ETL链路测试；需在目标电脑按上述Start→实验→Stop步骤检查两个标记、CPU线程切换和GPU事件。未连接放大器或打开刺激窗口，合成测试不能证明600 Hz卡顿已经修复。
