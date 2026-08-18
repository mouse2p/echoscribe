# API reference

One line per public surface, by subpackage. Everything listed is importable
from the module named; a test (`tests/test_docs.py`) fails when this page
names a module that does not exist.

## Audio front end

| module | what it holds |
| --- | --- |
| `echoscribe.waveform` | `Audio`, `silence`, `concatenate` |
| `echoscribe.audio` | `read_wav`, `write_wav` (stdlib `wave` + NumPy) |
| `echoscribe.resample` | `ResampleConfig`, polyphase and linear resampling |
| `echoscribe.framing` | `FrameConfig`, `frame_signal`, pre-emphasis, DC removal |
| `echoscribe.windows` | `window_vector`: rectangular, hamming, hann, blackman, povey |
| `echoscribe.spectrum` | `power_spectrum`, `log_power_spectrum`, `complex_spectrum` |
| `echoscribe.mel` | mel filterbanks, HTK and Slaney scales |
| `echoscribe.features` | `FeatureConfig`, `extract_log_mel`, `extract_mfcc` |
| `echoscribe.cmvn` | `CmvnStatistics`, `apply_cmvn`, `sliding_window_cmvn` |
| `echoscribe.delta` | `DeltaConfig`, `delta_features`, `stack_features` |
| `echoscribe.numeric` | `logsumexp`, `log_softmax`, `safe_log` |
| `echoscribe.types` | dtype aliases and array validators |
| `echoscribe.seeds` | `Seed` and its derivation rules |
| `echoscribe.errors` | the error hierarchy and `require` helpers |
| `echoscribe.meta` | `RunMetadata`, `run_metadata`, config fingerprints |

## Synthetic laboratory (`echoscribe.synth`)

| module | what it holds |
| --- | --- |
| `echoscribe.synth.phones` | the invented English-style phone inventory |
| `echoscribe.synth.lexicon` | `Lexicon`: spellings to phone/syllable sequences |
| `echoscribe.synth.tone` | `ToneContour`, the four lexical tones |
| `echoscribe.synth.source` | glottal-source and excitation signals |
| `echoscribe.synth.formant` | the seeded formant synthesizer |
| `echoscribe.synth.render` | `RenderConfig`, `render_utterance`, `render_corpus` |
| `echoscribe.synth.noise` | `generate_noise`, `mix_at_snr` |
| `echoscribe.synth.room` | convolutional room impulse responses |
| `echoscribe.synth.perturb` | speed and volume perturbation |
| `echoscribe.synth.corpus` | `CorpusConfig`, `generate_corpus`, `Corpus` |
| `echoscribe.synth.manifest` | `ManifestRecord`, checksummed JSONL manifests |

## Text and units (`echoscribe.text`)

| module | what it holds |
| --- | --- |
| `echoscribe.text.normalize` | Unicode normalization and text cleanup |
| `echoscribe.text.script` | `script_of`, `script_runs` (han/latin/digit/...) |
| `echoscribe.text.segment` | script-aware word/character segmentation |
| `echoscribe.text.bpe` | `learn_bpe`, `BpeModel` (deterministic, seeded) |
| `echoscribe.text.unigram` | `learn_unigram`, `UnigramModel` |
| `echoscribe.text.vocab` | `Vocabulary` with reserved tokens and digests |
| `echoscribe.text.units` | tone units, `utterance_units`, `unit_vocabulary` |

## Decoding math (`echoscribe.decode`)

| module | what it holds |
| --- | --- |
| `echoscribe.decode.posteriors` | `scripted_posteriors`, posterior validation |
| `echoscribe.decode.tokens` | `DecodedToken`, emission timing |
| `echoscribe.decode.ctc_greedy` | `ctc_greedy`: argmax, collapse, drop blanks |
| `echoscribe.decode.ctc_prefix` | `ctc_prefix_search`: merge semantics, n-best |
| `echoscribe.decode.transducer` | `transducer_greedy`, `TableJoint`, emission spans |
| `echoscribe.decode.chunked` | latency-bounded monotonic chunk decoding |
| `echoscribe.decode.cache` | joint-result caching across chunks |
| `echoscribe.decode.bias` | hotword trie, shallow-fusion bonus, `bias_greedy` |
| `echoscribe.decode.confidence` | posterior-marginal confidence and calibration |
| `echoscribe.decode.alignment` | token-to-word time spans |

## Streaming runtime (`echoscribe.stream`)

| module | what it holds |
| --- | --- |
| `echoscribe.stream.chunks` | `ChunkSchedule`, latency budget arithmetic |
| `echoscribe.stream.buffer` | `FrameBuffer` growth across pushes |
| `echoscribe.stream.joint` | `AppendableJoint`, the growing decode table |
| `echoscribe.stream.protocol` | `Hypothesis`, `StreamEvent`, partial-vs-final |
| `echoscribe.stream.runtime` | `StreamDecoder`, `StreamConfig`, barge-in reset |
| `echoscribe.stream.endpoint` | energy and blank-ratio endpointers with hangover |
| `echoscribe.stream.replay` | `record`, `replay`, `trace_digest` |
| `echoscribe.stream.session` | `Session`, JSONL persistence, `latency_report` |

## Evaluation (`echoscribe.eval`)

| module | what it holds |
| --- | --- |
| `echoscribe.eval.edit` | `align`, `edit_distance`, operation breakdown |
| `echoscribe.eval.wer` | `wer`, `cer`, `score_corpus`, fixed summary lines |
| `echoscribe.eval.codeswitch` | `score_mixed`, per-script attribution, switch counts |
| `echoscribe.eval.bench` | `measure_rtf`, `BenchResult`, timing statistics |
| `echoscribe.eval.report` | deterministic SVG sections, `render_report`, golden digests |

## Noise utilities (`echoscribe.noise`)

| module | what it holds |
| --- | --- |
| `echoscribe.noise.stft` | `stft`/`istft` transform pair, `power_spectrogram` |
| `echoscribe.noise.gains` | spectral-subtraction and decision-directed Wiener gains |
| `echoscribe.noise.denoise` | lead-in noise profiles, the end-to-end `denoise` |
| `echoscribe.noise.snr` | `snr_db`, `snr_improvement_db` against planted references |

## Trainable scaffolding (`echoscribe.model`, torch extra)

| module | what it holds |
| --- | --- |
| `echoscribe.model.schema` | versioned bundle files, `save_bundle`/`load_bundle` |
| `echoscribe.model.stack` | `stack_frames` |
| `echoscribe.model.encoder` | `Encoder`, a compact frame-stacked BiLSTM |
| `echoscribe.model.infer` | bundle -> posterior bridge to the NumPy decoders |
| `echoscribe.model.ctc` | `CtcHead`, `ctc_loss` |
| `echoscribe.model.transducer` | `Predictor`, `Joint`, the exact RNN-T lattice loss |
| `echoscribe.model.train` | seeded CTC and transducer loops over the laboratory |

## Command line (`echoscribe.cli`)

| module | what it holds |
| --- | --- |
| `echoscribe.cli.common` | exit codes, JSON/JSONL helpers |
| `echoscribe.cli.scripted` | planted-transcript fixture posteriors |
| `echoscribe.cli.synth` | render a seeded corpus into a work directory |
| `echoscribe.cli.feats` | log-mel extraction over a rendered corpus |
| `echoscribe.cli.decode` | CTC greedy and prefix-beam decoding |
| `echoscribe.cli.stream` | chunked streaming decode into a session file |
| `echoscribe.cli.bias` | hotword biasing over an ambiguous fixture joint |
| `echoscribe.cli.evaluate` | unit- and text-level scoring of hypotheses |
| `echoscribe.cli.bench` | decode real-time-factor measurement |
| `echoscribe.cli.report` | the SVG report over a scored work directory |
| `echoscribe.cli.train` | tiny CTC/transducer training (torch extra) |
| `echoscribe.cli.demo` | the whole offline loop in one call |
