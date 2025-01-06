# echoscribe

面向**流式语音识别机制**的离线工作台：transducer (RNN-T) 与 CTC 解码、带延迟预算的 chunk 流式运行时、hotword biasing、endpointing、English-Chinese code-switch 评测，以及一个完全合成的声学实验室（planted transcript 即 ground truth）。

## 这个项目验证什么，不验证什么

echoscribe 实现并测试的是流式语音识别的**机制**：特征流水线、tokenizer、CTC/transducer 解码数学、延迟受限的 chunk 调度、biasing、endpointing 和 code-switch 感知的打分。

它**不**声称任何真实声学系统的精度。默认测试与示例全部离线运行，音频由带种子的 formant 合成器从脚本化 lexicon 生成，转写文本是植入的（planted）ground truth。没有预训练权重、没有真实语音数据集、没有模型下载。所有出现在文档里的数字都来自这个合成实验室，并明确标注为合成结果。

## 安装

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

可选的 `torch` extra（仅 CPU）用于可训练的 encoder/joint 演示：

```bash
.venv/bin/pip install -e ".[dev,torch]" \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

NumPy 是唯一必需的运行时依赖。

## 快速开始

```bash
make demo
```

`demo` 会跑完整条离线链路：合成语料 → 提取特征 → 解码 → 打分 → 生成 SVG 报告，并打印在合成集上实测的 WER / CER / RTF。

## 开发

```bash
make build          # 构建 wheel
make test           # 快速测试（不含 slow 标记）
make test-all       # 全量测试，含 slow 与 model 标记
make format-check   # ruff format --check + ruff check
make typecheck      # mypy
```

## License

MIT，见 [LICENSE](LICENSE)。
