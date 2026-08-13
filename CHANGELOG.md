# Changelog

本项目遵循 [Semantic Versioning](https://semver.org/spec/v2.0.0.html)。
在 1.0 之前，补丁版本用于机制修正与文档补充，次版本用于新增能力。

## [0.1.1] - 2026-10-14

### Added

- 评测套件 `echoscribe.eval`：edit-distance 对齐与操作重建、WER/CER/语料 micro 平均、
  code-switch hybrid 打分（按 script 归因）、RTF bench、确定性 SVG 报告（golden digest
  固定字节）与带 overwrite 保护的报告写盘。
- 噪声工具 `echoscribe.noise`：归一化 overlap-add 的 STFT/iSTFT 变换对、spectral
  subtraction 与 decision-directed Wiener 增益、lead-in 噪声剖面估计，以及基于 planted
  参考的 SNR 测量（含"没有安静前导就会误减"的失败模式测试）。
- 可训练脚手架 `echoscribe.model`（可选 torch extra，仅 CPU）：版本化 bundle 文件、
  frame stacking、frame-stacked BiLSTM encoder、CTC head 与 lattice loss、transducer
  predictor/joint、精确 RNN-T 前向损失、种子化训练循环（真实 loss 下降）与训练历史导出。
- `echoscribe` CLI 十个子命令：synth / feats / decode / stream / bias / eval / bench /
  report / train / demo，统一 workdir 契约、一行式错误输出与 0/1/2 退出码。
- 五个离线 examples（含子进程 smoke test）：语料与特征、解码对比、流式会话与确定性回放、
  code-switch SVG 报告、训练 demo。
- docs/ 五份指南：architecture、usage、API reference、honesty、references，配防漂移测试。
- CI 矩阵（ubuntu-latest × Python 3.11/3.12/3.13，目标与 Makefile 一一对应）、release
  workflow（仅 GitHub release 产物）、issue/PR 模板、dependabot、built-wheel 安装冒烟脚本。

## [0.1.0] - 2025-01-06

### Added

- 项目骨架：`src` 布局的 `echoscribe` 包、Makefile 开发入口、MIT 许可证。
- 版本一致性测试：`__version__` 与已安装包元数据必须相符。

[0.1.1]: https://github.com/mouse2p/echoscribe/releases/tag/v0.1.1
[0.1.0]: https://github.com/mouse2p/echoscribe/releases/tag/v0.1.0
