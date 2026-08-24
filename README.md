# echoscribe

[![CI](https://github.com/mouse2p/echoscribe/actions/workflows/ci.yml/badge.svg)](https://github.com/mouse2p/echoscribe/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

面向**流式语音识别机制**的离线工作台：transducer (RNN-T) 与 CTC 解码数学、带延迟预算的
chunk 流式运行时、hotword biasing、endpointing、English-Chinese code-switch 评测、
谱减/Wiener 噪声工具，以及一个完全合成的声学实验室（planted transcript 即 ground truth）。
核心只依赖 NumPy；可选的 `torch` extra（仅 CPU）驱动可训练的 encoder/joint 演示。

## 这个项目验证什么，不验证什么

echoscribe 实现并测试的是流式语音识别的**机制**：特征流水线、tokenizer、CTC/transducer
解码数学、延迟受限的 chunk 调度、biasing、endpointing、code-switch 感知的打分和确定性
SVG 报告。

它**不**声称任何真实声学系统的精度。默认测试与示例全部离线运行：音频由带种子的 formant
合成器从脚本化 lexicon 渲染，转写文本是植入的（planted）ground truth。没有预训练权重、
没有真实语音数据集、没有模型下载。所有出现在文档和输出里的数字都来自这个合成实验室，
并在打印处明确标注。详见 [docs/honesty.md](docs/honesty.md)。

## 安装

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

可选 torch extra（仅 CPU，永远不会下载 GPU 构建或权重）：

```bash
.venv/bin/pip install -e ".[dev,torch]" \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

## 快速开始

一条命令跑完整离线链路（合成语料 → 特征 → 批量/流式解码 → 打分 → RTF → SVG 报告），
并打印本次运行实测的数字：

```bash
make demo        # 等价于 echoscribe demo --workdir /tmp/echoscribe-demo
```

加上 `--train`（需要 torch extra）后，demo 还会训练一个小 CTC 模型并通过
`echoscribe.model.infer` 的接缝用它解码同一批合成语料，打印训练损失与
trained-bundle 的 UER——并明确标注：训练与解码使用同一合成实验室时，低错误率
是对 planted 数据的记忆，而不是识别精度。

或者按阶段执行，每个阶段读上一阶段写进同一 workdir 的文件：

```bash
echoscribe synth  --workdir run --utterances 24 --seed 2025   # wav/ + manifest.jsonl
echoscribe feats  --workdir run --num-mel-bins 40             # features.npz
echoscribe decode --workdir run --method greedy               # decode.jsonl
echoscribe stream --workdir run --frames-per-chunk 16         # session.jsonl
echoscribe eval   --workdir run --level text --metric hybrid  # scores.json
echoscribe bench  --workdir run --repeats 3                   # bench.json
echoscribe report --workdir run                               # report.svg
```

注意：默认管线解码的是由 planted transcript 构造的 **scripted fixture posteriors**。
它验证的是特征、解码器、流式调度、打分与报告这些机制本身；零错误率证明的是流水线
算术正确，而不是"识别准确"。

库接口同样直接：

```python
from echoscribe.eval import format_summary, wer

format_summary(wer("the cat sat", "the cat"), label="WER")
# 'WER 33.33% | S0 D1 I0 | N3'
```

## 架构

```
audio front end → synthetic laboratory → text/units → decoding math
→ streaming runtime → evaluation → noise utilities → model (torch extra) → CLI
```

层间只向下依赖；NumPy 核心不知道 torch 的存在；离线解码器与可训练模型只在
`(frames, symbols)` 的 posterior/logit 矩阵这一处接缝相遇（blank 固定在 index 0）。
完整分层图与设计规则见 [docs/architecture.md](docs/architecture.md)。

## 示例

五个可离线运行的示例（`tests/test_examples.py` 会以子进程真实执行每一个）：

| 示例 | 内容 | 需要 torch |
| --- | --- | --- |
| `examples/corpus_and_features` | 种子化语料渲染、manifest digest、log-mel + CMVN + delta | 否 |
| `examples/decoding_comparison` | 同一 scripted posterior 上的 CTC greedy / prefix beam / transducer | 否 |
| `examples/streaming_session` | chunk 流式、延迟预算、endpointing、确定性回放 | 否 |
| `examples/codeswitch_report` | English-Chinese hybrid 打分与确定性 SVG 报告 | 否 |
| `examples/training_demo` | 小模型 CTC lattice loss 在合成语音上真实下降 | 是 |

## CLI 命令

`synth` `feats` `decode` `stream` `bias` `eval` `bench` `report` `train` `demo`，
统一退出码：0 成功、1 任务失败（stderr 一行说明）、2 用法错误。
逐个说明见 [docs/usage.md](docs/usage.md)。

## 文档

- [docs/architecture.md](docs/architecture.md) — 分层与设计规则
- [docs/usage.md](docs/usage.md) — 安装、CLI、库用法、文件格式、CI
- [docs/api.md](docs/api.md) — 公开 API 一览（有防漂移测试）
- [docs/honesty.md](docs/honesty.md) — 每个数字允许声称什么
- [docs/references.md](docs/references.md) — 机制出处与引用政策

## 开发

```bash
make build          # 构建 wheel
make test           # 快速测试（不含 slow 标记，约 15 秒）
make test-all       # 全量测试，含 slow 与 model 标记
make format-check   # ruff format --check + ruff check
make typecheck      # mypy
```

CI 在 Python 3.11/3.12/3.13 矩阵上执行与 Makefile 一一对应的目标；release workflow
只发布 GitHub release 产物（wheel、sdist、SHA256SUMS），不涉及任何 registry 凭据。
贡献指南见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License

MIT，见 [LICENSE](LICENSE)。
