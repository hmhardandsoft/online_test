## 安装

在本项目目录中执行，需要 uv、Git、可访问 SDK Git 仓库的网络，以及放大器和显卡驱动。

```powershell
uv sync --locked --python 3.12.13
uv run --locked python -m unittest test_online_experiment -v
uv run --locked python .\verify_online_runs.py
```

使用 pyproject.toml 和 uv.lock 管理依赖，SDK 固定到已测试的 Git 提交。不要复制旧 .venv，也不要用另一个 requirements.txt 覆盖环境。完整流程验证使用合成信号，不打开硬件，结果位于 online_results/software_checks。

## 运行

先将目标显示器设为 Windows 主显示器，并在系统中选择相应刷新率。

```powershell
# 600 Hz，M4，TRCA
uv run --locked python .\online_experiment.py --config .\online_config_600hz.json --method Aug_FB_TRCA --task-m 4
# 120 Hz，M4，TRCA
uv run --locked python .\online_experiment.py --config .\online_config.json --method Aug_FB_TRCA --task-m 4
```

第一次 SPACE 开始训练；训练结束黑屏拟合、冻结、追赶，然后显示 PRESS SPACE TO START。第二次 SPACE 从初始相位启动测试，缓冲结束后连续进入计分。ESC 退出并保存。

600 Hz 配置仅改变刷新率，FS=2000、窗口200样本、刺激10 Hz和训练时长保持一致。实际600 Hz帧间隔及硬件采集仍需在目标电脑上验证。

无放大器、无窗口的短流程：

```powershell
uv run --locked python .\online_experiment.py --simulate --headless --training-blocks 1 --train-cue-seconds 1 1 --skip-seconds 0.2 --min-samples 5 --online-seconds 2
```

## 文件与 Git


- online_core.py：滤波、取窗、模型和时间准确率。
- SSVEP_stimulator.py、util.py：刺激界面及共享状态。刺激模块仍导入 SSVEP_decoder.py 和 SSVEP_recorder.py，因此随项目保留。
- usb_acquisition_benchmark.py：采集连接辅助函数，也是可单独使用的空采工具。
- switch_event/label_switch_accuracy_analysis_1.py：离线算法参考快照，仅一致性测试读取；不是运行实验所需的数据集。
- source_manifest.json：迁移时原始代码 SHA256，记录初始来源，后续改动不要求与其相同。
- online_experiment_README.md：完整范式、参数和输出说明。

原始数据、模型、虚拟环境和测试输出由 .gitignore 排除。保留源码、配置、测试、uv.lock 和文档。实际运行参数另外保存于每次结果的 config.json。


## 刺激停顿定位

默认开启主线程分段计时、垃圾回收和窗口状态记录。使用说明、输出文件及开销测试见 [timing_diagnostics_README.md](timing_diagnostics_README.md)。通过 --no-timing-diagnostics 可关闭新增定位记录。
