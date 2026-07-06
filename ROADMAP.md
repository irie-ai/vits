# VITS C++ 移植ロードマップ

## プロジェクト概要

| 項目      | 内容            |
| ------- | ------------- |
| プロジェクト名 | VITS          |
| 元プロジェクト | Python版 VITS  |
| 使用言語    | C++17         |
| 推論ライブラリ | LibTorch      |
| IDE     | Visual Studio |
| ビルド     | CMake         |
| テスト     | CTest + standalone test executables |
| 標準ビルド構成 | RelWithDebInfo |

---

# 全体進捗

| モジュール           | 状態 | テスト | 備考 |
| --------------- | -- | --- | -- |
| 環境構築            | ✅  | -   | 完了 |
| CMake           | ✅  | -   | 完了 |
| commons         | ✅  | ✅   | 基本関数・mask・path・grad clip まで実装 |
| utils           | 🟡  | ✅   | HParams/config/latest checkpoint/wav load-save/checkpoint save-load/plot tensor/file summary を実装。Python checkpoint 完全互換は未検証 |
| text            | 🟡  | ✅   | symbols/cleaners/numbers の主要部を移植。IPA/phonemizer は保留 |
| data_utils      | 🟡  | ✅   | filelist/text/audio/spec cache/collate/bucket sampler を移植 |
| monotonic_align | 🟡  | ✅   | CPU/LibTorch 版 maximum_path を実装。Cython/CUDA 相当の高速化は未着手 |
| attentions      | 🟡  | ✅   | Encoder/Decoder/relative attention 実装。proximal init は保留 |
| modules         | 🟡  | ✅   | flow/resblock/conv 系を実装。weight_norm 等は保留 |
| models          | 🟡  | ✅   | TextEncoder から SynthesizerTrn まで主要構造を実装。重み互換は未完 |
| losses          | ✅  | ✅   | generator/discriminator/feature/kl loss を実装 |
| mel_processing  | ✅  | ✅   | spectrogram/mel/dynamic range 処理を実装 |
| inference       | 🟡  | ✅   | config/text から SynthesizerTrn::infer へ接続する C++ ユーティリティを実装。実 checkpoint 推論は未検証 |
| examples        | 🟡  | ✅   | hello/load_model/inference CLI を追加。vits_inference は wav 保存まで対応。実 checkpoint 推論は未検証 |
| benchmark       | 🟡  | ✅   | ランダム重みの軽量 infer benchmark CLI を追加。実 checkpoint/GPU benchmark は未検証 |
| training        | 🟡  | ✅   | 合成データで SynthesizerTrn + MultiPeriodDiscriminator の 1 step train を確認。実 dataset 学習は未検証 |
| docs            | ⬜  | -   |    |

---

# commons

## 対応ファイル

Python

* commons.py

C++

* include/commons.hpp
* src/commons.cpp

---

## 関数一覧

| Python                          | C++                         | 実装 | テスト | リファクタリング |
| ------------------------------- | --------------------------- | -- | --- | -------- |
| init_weights                    | initWeights                 | ✅  | ✅   | 🟡        |
| get_padding                     | getPadding                  | ✅  | ✅   | ✅        |
| intersperse                     | intersperse                 | ✅  | ✅   | ✅        |
| kl_divergence                   | klDivergence                | ✅  | ✅   | ✅        |
| convert_pad_shape               | convertPadShape             | ✅  | ✅   | ✅        |
| slice_segments                  | sliceSegments               | ✅  | ✅   | ✅        |
| rand_slice_segments             | randSliceSegments           | ✅  | ✅   | ✅        |
| rand_gumbel                     | randGumbel                  | ✅  | ✅   | ✅        |
| rand_gumbel_like                | randGumbelLike              | ✅  | ✅   | ✅        |
| get_timing_signal_1d            | getTimingSignal1D           | ✅  | ✅   | ✅        |
| add_timing_signal_1d            | addTimingSignal1D           | ✅  | ✅   | ✅        |
| cat_timing_signal_1d            | catTimingSignal1D           | ✅  | ✅   | ✅        |
| subsequent_mask                 | subsequentMask              | ✅  | ✅   | ✅        |
| fused_add_tanh_sigmoid_multiply | fusedAddTanhSigmoidMultiply | ✅  | ✅   | ✅        |
| shift_1d                        | shift1D                     | ✅  | ✅   | ✅        |
| sequence_mask                   | sequenceMask                | ✅  | ✅   | ✅        |
| generate_path                   | generatePath                | ✅  | ✅   | ✅        |
| clip_grad_value_                | clipGradValue               | ✅  | ⬜   | 🟡        |

---

# utils

| 関数 | 実装 | テスト | リファクタリング |
| -- | -- | --- | -------- |
| load_filepaths_and_text | ✅ | ✅ | ✅ |
| HParams / config.json | ✅ | ✅ | 🟡 |
| latest_checkpoint_path | ✅ | ✅ | ✅ |
| load_wav_to_torch | ✅ | ✅ | 🟡 |
| save_wav_from_torch | ✅ | ✅ | PCM16 wav 書き出し。`[-1, 1]` 系 tensor に `max_wav_value` を掛けて保存 |
| checkpoint save/load | 🟡 | ✅ | LibTorch archive 形式で model/optimizer/iteration/lr に対応。Python `.pth` 完全互換は未検証 |
| plot_spectrogram_to_numpy | 🟡 | ✅ | matplotlib ではなく HWC RGB `uint8` tensor を返す |
| plot_alignment_to_numpy | 🟡 | ✅ | alignment を転置して HWC RGB `uint8` tensor を返す |
| summarize / TensorBoard logging | 🟡 | ✅ | `FileSummaryWriter` で scalar CSV / image PPM / audio wav 出力に対応。TensorBoard event 互換は未着手 |

※ `load_filepaths_and_text` は `data_utils::loadFilepathsAndText` として実装。

---

# text

## 対応ファイル

Python

* text/\_\_init\_\_.py
* text/symbols.py
* text/cleaners.py
* text/numbers.py

C++

* include/text_processing.hpp
* src/text_processing.cpp

| 機能 | 実装 | テスト | 備考 |
| -- | -- | --- | -- |
| symbols / SPACE_ID | 🟡 | ✅ | ASCII と VITS 句読点は対応。IPA 全量は未対応 |
| text_to_sequence | ✅ | ✅ | cleaner 適用と ID 化 |
| cleaned_text_to_sequence | ✅ | ✅ | 既クリーニング文字列の ID 化 |
| sequence_to_text | ✅ | ✅ | ID から文字列へ戻す |
| basic_cleaners | ✅ | ✅ | lowercase + whitespace collapse |
| transliteration_cleaners | 🟡 | ✅ | common accent の簡易 ASCII 化 |
| english_cleaners | 🟡 | ✅ | 数値・略語展開まで。phonemizer は未接続 |
| english_cleaners2 | 🟡 | ✅ | phonemizer hook のみ |
| numbers.py | 🟡 | ✅ | cardinal/ordinal/decimal/currency を実装。inflect 完全互換は未検証 |

---

# data_utils

## 対応ファイル

Python

* data_utils.py

C++

* include/data_utils.hpp
* src/data_utils.cpp

| 機能 | 実装 | テスト | 備考 |
| -- | -- | --- | -- |
| load_filepaths_and_text | ✅ | ✅ | delimiter 指定対応 |
| TextAudioLoader.get_text | ✅ | ✅ | `textToTensor` として実装 |
| TextAudioCollate | ✅ | ✅ | spec 長降順 sort + padding |
| TextAudioSpeakerCollate | ✅ | ✅ | speaker id 付き padding |
| DistributedBucketSampler._bisect | ✅ | ✅ | `bucketIndex` |
| DistributedBucketSampler._create_buckets | ✅ | ✅ | `createBucketSamplerState` |
| DistributedBucketSampler.__iter__ | ✅ | ✅ | `createBucketBatches` |
| load_wav_to_torch / get_audio | ✅ | ✅ | wav 読み込みは `utils`、spec 生成は `data_utils::getAudio` |
| TextAudioLoader.get_audio | ✅ | ✅ | `makeTextAudioItem` / `makeTextAudioSpeakerItem` として実装 |
| spec cache | ✅ | ✅ | `useSpecCache` 有効時に `<audio>.spec.pt` を save/load |

---

# monotonic_align

## 対応ファイル

Python

* monotonic_align/\_\_init\_\_.py
* monotonic_align/core.pyx

C++

* include/monotonic_align.hpp
* src/monotonic_align.cpp

| 機能 | 実装 | テスト | 備考 |
| -- | -- | --- | -- |
| maximum_path | 🟡 | ✅ | CPU 上で dynamic programming + backtrace を実行し、入力 tensor の device/dtype に戻す |
| Cython/CUDA optimized path | ⬜ | ⬜ | Python 版の Cython 実装相当の高速化は未着手 |

---

# attentions

| クラス                | 実装 | テスト | リファクタリング |
| ------------------ | -- | --- | -------- |
| FFN                | ✅  | ✅   | 🟡        |
| MultiHeadAttention | ✅  | ✅   | 🟡        |
| Encoder            | ✅  | ✅   | 🟡        |
| Decoder            | ✅  | ✅   | 🟡        |

保留:

* `proximal_init` 時の `conv_q` -> `conv_k` weight copy
* Python 版との厳密な重みロード互換検証

---

# modules

| クラス       | 実装 | テスト | リファクタリング |
| --------- | -- | --- | -------- |
| LayerNorm | ✅  | ✅   | ✅        |
| ConvReluNorm | ✅  | ✅   | 🟡        |
| ResBlock1 | ✅  | ✅   | 🟡        |
| ResBlock2 | ✅  | ✅   | 🟡        |
| DDSConv   | ✅  | ✅   | 🟡        |
| WN        | ✅  | ✅   | 🟡        |
| Flip      | ✅  | ✅   | ✅        |
| Log       | ✅  | ✅   | ✅        |
| ElementwiseAffine | ✅ | ✅ | ✅ |
| ResidualCouplingLayer | ✅ | ✅ | 🟡 |
| ConvFlow | ✅ | ✅ | 🟡 |

保留:

* `weight_norm` / `spectral_norm`
* 一部 submodule の完全な `register_module` 化
* Python 初期化との厳密一致

---

# models

| クラス                   | 実装 | テスト | リファクタリング |
| --------------------- | -- | --- | -------- |
| TextEncoder           | ✅  | ✅   | 🟡        |
| DurationPredictor     | ✅  | ✅   | 🟡        |
| StochasticDurationPredictor | ✅ | ✅ | 🟡 |
| PosteriorEncoder      | ✅  | ✅   | 🟡        |
| Generator             | ✅  | ✅   | 🟡        |
| ResidualCouplingBlock | ✅  | ✅   | 🟡        |
| SynthesizerTrn        | ✅  | ✅   | 🟡        |
| DiscriminatorP        | ✅  | ✅   | 🟡        |
| DiscriminatorS        | ✅  | ✅   | 🟡        |
| MultiPeriodDiscriminator | ✅ | ✅ | 🟡 |

保留:

* checkpoint からの重みロード互換
* GPU/CPU 両対応の monotonic alignment
* inference example での実音声生成検証

---

# losses

| 関数 | 実装 | テスト | リファクタリング |
| -- | -- | --- | -------- |
| feature_loss | ✅ | ✅ | ✅ |
| discriminator_loss | ✅ | ✅ | ✅ |
| generator_loss | ✅ | ✅ | ✅ |
| kl_loss | ✅ | ✅ | ✅ |

---

# mel_processing

| 関数 | 実装 | テスト | リファクタリング |
| -- | -- | --- | -------- |
| dynamic_range_compression | ✅ | ✅ | ✅ |
| dynamic_range_decompression | ✅ | ✅ | ✅ |
| spectral_normalize | ✅ | ✅ | ✅ |
| spectral_de_normalize | ✅ | ✅ | ✅ |
| spectrogram_torch | ✅ | ✅ | 🟡 |
| spec_to_mel_torch | ✅ | ✅ | 🟡 |
| mel_spectrogram_torch | ✅ | ✅ | 🟡 |

---

# inference

## 対応ファイル

C++

* include/inference.hpp
* src/inference.cpp

| 機能 | 実装 | テスト | 備考 |
| -- | -- | --- | -- |
| text_to_sequence + add_blank | ✅ | ✅ | `text_processing` と `commons::intersperse` を使用 |
| prepare_text | ✅ | ✅ | `tokens` と `lengths` を batch 形状で生成 |
| cleaner/add_blank hparams 読み取り | ✅ | ✅ | `data.text_cleaners` / `data.add_blank` |
| SynthesizerTrn hparams 構築 | 🟡 | ✅ | 標準 VITS config の主要 model/data/train 項目に対応 |
| infer_text | 🟡 | ✅ | `NoGradGuard` + `eval()` + `SynthesizerTrn::infer` 接続。実 checkpoint 音声生成は未検証 |

---

# examples

| ファイル           | 状態 |
| -------------- | -- |
| hello_vits.cpp | ✅  |
| inference.cpp  | ✅  |
| load_model.cpp | 🟡  |

保留:

* Python `.pth` checkpoint の完全互換ロード検証
* 実 checkpoint を使った音声生成確認

---

# benchmark

| ファイル | 状態 | 備考 |
| -- | -- | -- |
| examples/benchmark.cpp | 🟡 | `vits_benchmark` としてビルド。`--warmup` / `--runs` / `--max-length` / `--text` に対応 |

保留:

* 実 checkpoint を使った CPU/GPU benchmark
* wav 保存込みの end-to-end benchmark
* PyTorch Python 版との速度比較

---

# training

| 項目 | 状態 | テスト | 備考 |
| -- | -- | --- | -- |
| synthetic 1-step GAN smoke | ✅ | ✅ | 合成 text/spec/audio で generator/discriminator の forward/backward/AdamW step を確認 |
| real dataset training loop | ⬜ | ⬜ | filelist/config/checkpoint/logging を接続した学習ループは未実装 |
| resume/eval/export loop | ⬜ | ⬜ | checkpoint resume、定期 eval、成果物 export は未実装 |

---

# テスト進捗

| テスト          | 状態 |
| ------------ | -- |
| test_commons | ✅  |
| test_modules | ✅  |
| test_attentions | ✅ |
| test_models  | ✅  |
| test_losses | ✅ |
| test_mel_processing | ✅ |
| test_text_processing | ✅ |
| test_data_utils | ✅ |
| test_utils | ✅ |
| test_monotonic_align | ✅ |
| test_inference | ✅ |
| test_training | ✅ |
| example_hello_vits | ✅ |
| vits_benchmark build/run | ✅ |

最終確認:

* 2026-07-07 時点で RelWithDebInfo の `ctest` は 13/13 passed

---

# リファクタリング

* [ ] テンプレート化
* [ ] 共通処理整理
* [ ] C++20対応
* [ ] コメント追加
* [ ] Doxygen生成
* [ ] clang-format
* [ ] clang-tidy

---

# 最終目標

* [ ] Python版との出力一致
* [ ] 全CTest成功
* [ ] CPU推論成功
* [ ] GPU推論成功
* [ ] README完成
* [ ] GitHub公開
* [ ] Release版作成
