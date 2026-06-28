# VITS C++ 移植ロードマップ

## プロジェクト概要

| 項目      | 内容            |
| ------- | ------------- |
| プロジェクト名 | VITS          |
| 元プロジェクト | Python版 VITS  |
| 使用言語    | C++20         |
| 推論ライブラリ | LibTorch      |
| IDE     | Visual Studio |
| ビルド     | CMake         |
| テスト     | GoogleTest    |

---

# 全体進捗

| モジュール           | 状態 | テスト | 備考 |
| --------------- | -- | --- | -- |
| 環境構築            | ✅  | -   | 完了 |
| CMake           | ✅  | -   | 完了 |
| commons         | ⬜  | ⬜   |    |
| utils           | ⬜  | ⬜   |    |
| text            | ⬜  | ⬜   |    |
| monotonic_align | ⬜  | ⬜   |    |
| attentions      | ⬜  | ⬜   |    |
| modules         | ⬜  | ⬜   |    |
| models          | ⬜  | ⬜   |    |
| inference       | ⬜  | ⬜   |    |
| examples        | ⬜  | ⬜   |    |
| benchmark       | ⬜  | ⬜   |    |
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
| init_weights                    | initWeights                 | ⬜  | ⬜   | ⬜        |
| get_padding                     | getPadding                  | ⬜  | ⬜   | ⬜        |
| intersperse                     | intersperse                 | ⬜  | ⬜   | ⬜        |
| convert_pad_shape               | convertPadShape             | ⬜  | ⬜   | ⬜        |
| slice_segments                  | sliceSegments               | ⬜  | ⬜   | ⬜        |
| rand_slice_segments             | randSliceSegments           | ⬜  | ⬜   | ⬜        |
| rand_gumbel                     | randGumbel                  | ⬜  | ⬜   | ⬜        |
| rand_gumbel_like                | randGumbelLike              | ⬜  | ⬜   | ⬜        |
| get_timing_signal_1d            | getTimingSignal1D           | ⬜  | ⬜   | ⬜        |
| add_timing_signal_1d            | addTimingSignal1D           | ⬜  | ⬜   | ⬜        |
| cat_timing_signal_1d            | catTimingSignal1D           | ⬜  | ⬜   | ⬜        |
| subsequent_mask                 | subsequentMask              | ⬜  | ⬜   | ⬜        |
| fused_add_tanh_sigmoid_multiply | fusedAddTanhSigmoidMultiply | ⬜  | ⬜   | ⬜        |
| shift_1d                        | shift1D                     | ⬜  | ⬜   | ⬜        |
| sequence_mask                   | sequenceMask                | ⬜  | ⬜   | ⬜        |
| generate_path                   | generatePath                | ⬜  | ⬜   | ⬜        |
| clip_grad_value_                | clipGradValue               | ⬜  | ⬜   | ⬜        |

---

# utils

| 関数 | 実装 | テスト | リファクタリング |
| -- | -- | --- | -------- |
|    | ⬜  | ⬜   | ⬜        |

---

# attentions

| クラス                | 実装 | テスト | リファクタリング |
| ------------------ | -- | --- | -------- |
| MultiHeadAttention | ⬜  | ⬜   | ⬜        |
| Encoder            | ⬜  | ⬜   | ⬜        |
| Decoder            | ⬜  | ⬜   | ⬜        |

---

# modules

| クラス       | 実装 | テスト | リファクタリング |
| --------- | -- | --- | -------- |
| ResBlock1 | ⬜  | ⬜   | ⬜        |
| ResBlock2 | ⬜  | ⬜   | ⬜        |
| DDSConv   | ⬜  | ⬜   | ⬜        |
| WN        | ⬜  | ⬜   | ⬜        |
| Flip      | ⬜  | ⬜   | ⬜        |

---

# models

| クラス                   | 実装 | テスト | リファクタリング |
| --------------------- | -- | --- | -------- |
| TextEncoder           | ⬜  | ⬜   | ⬜        |
| PosteriorEncoder      | ⬜  | ⬜   | ⬜        |
| Generator             | ⬜  | ⬜   | ⬜        |
| ResidualCouplingLayer | ⬜  | ⬜   | ⬜        |
| SynthesizerTrn        | ⬜  | ⬜   | ⬜        |

---

# examples

| ファイル           | 状態 |
| -------------- | -- |
| hello_vits.cpp | ⬜  |
| inference.cpp  | ⬜  |
| load_model.cpp | ⬜  |

---

# テスト進捗

| テスト          | 状態 |
| ------------ | -- |
| test_commons | ⬜  |
| test_utils   | ⬜  |
| test_modules | ⬜  |
| test_models  | ⬜  |

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
* [ ] 全GoogleTest成功
* [ ] CPU推論成功
* [ ] GPU推論成功
* [ ] README完成
* [ ] GitHub公開
* [ ] Release版作成
