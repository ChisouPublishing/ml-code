# ml-code — 『Pythonで学ぶ 機械学習の基礎と実践』サンプルコード

教科書の各章に載せたコードを、**検証単位ごとに 1 つの Notebook** にまとめたもの。
フォルダ番号は**印刷版（PDF）の章番号**と一致している（第 1 章＝`01_overview`）。

## 使い方

- **どのノートも単体で完結する。** 他のノートを先に実行する必要はない。
  必要な import・データ生成・前処理はすべてそのノートの中に入っている。
- 各ノートの先頭に **出典（章・節）／このノートのパス／必要なパッケージ／実行時間の目安** がある。
- 先頭の `# !pip install ...` セルは、未導入の環境（Colab など）でコメントを外して実行する。

```bash
git clone https://github.com/ChisouPublishing/ml-code.git
cd ml-code
pip install -r requirements.txt     # まとめて入れる場合
jupyter lab                         # あるいは VS Code / Colab で開く
```

## 一括実行による検証

このフォルダをカレントディレクトリにして実行する。

```bash
zsh run_all_notebooks.sh . "$(command -v python3)" 1800
```

対象環境に `nbconvert` と `ipykernel` が必要。スクリプトは指定した Python から一時 kernel を作るため、既存の `python3` kernel 設定には依存しない。
第 3 引数は 1 セルあたりのタイムアウト秒数。全ノートブックを順に実行し、成功・失敗を最後に集計する。
出力は各 `.ipynb` に保存され、学習モデルと Optuna の探索履歴もコードの設定に従って更新される。

macOS で `divide by zero encountered in matmul` などの数値警告が出る場合は、抑制する前に次を実行する。

```bash
python check_numeric_backend.py
```

行列積、線形回帰予測、有限値、別実装との一致、実ゼロ除算の検出をまとめて確認する。

## 章の一覧

### `01_overview/` — 第 1 章 総論

機械学習とは何か

| ファイル | 内容 |
|---|---|
| `01_rule_vs_learning.ipynb` | ルールを書く vs 例から学ばせる |

### `02_regression/` — 第 2 章 1-A 回帰と正則化

線形回帰・過学習・Ridge/Lasso

| ファイル | 内容 |
|---|---|
| `01_linear_regression.ipynb` | 線形回帰 — 重みを読む |
| `02_polynomial_overfit.ipynb` | 多項式回帰 — 過学習を自分で起こす |
| `03_regularization.ipynb` | 正則化 — 係数に罰を与える |

### `03_classification/` — 第 3 章 1-B 分類

ロジスティック回帰・手法比較

| ファイル | 内容 |
|---|---|
| `01_logistic_regression.ipynb` | ロジスティック回帰と閾値 |
| `02_classifier_comparison.ipynb` | 分類器 6 種を同じデータで比較 |

### `04_ensemble_eval/` — 第 4 章 1-C アンサンブル・評価・前処理

RF・GBDT・リーク対策

| ファイル | 内容 |
|---|---|
| `01_random_forest.ipynb` | ランダムフォレストと OOB スコア・不純度ベースと permutation の重要度比較 |
| `02_gbdt_early_stopping.ipynb` | LightGBM と早期終了 |
| `03_leakproof_pipeline.ipynb` | リークしないパイプライン |
| `04_leakage_patterns.ipynb` | リークの 4 パターン（演習の答え合わせ） |
| `05_cross_validation.ipynb` | 交差検証 — hold-out のばらつき、Stratified / Group / TimeSeries 分割 |
| `06_classification_metrics.ipynb` | 評価指標 — 不均衡データ、コストで決める閾値、ROC と PR、R² の水増し |

### `05_hpo/` — 第 5 章 1-D 探索の自動化とモデル選定

探索・閾値・校正・HITL

| ファイル | 内容 |
|---|---|
| `01_model_benchmark.ipynb` | 3 実装の直接比較 — 精度だけを見ない |
| `02_threshold_optimization.ipynb` | 閾値最適化 — 0.5 は「決めなかった」の意味 |
| `03_pure_search_loop.ipynb` | 自前で書く探索ループ（30 行） |
| `04_sklearn_search.ipynb` | scikit-learn だけで完結する探索 |
| `05_pure_tpe.ipynb` | TPE を自分で書く（40 行） |
| `06_optuna_model_selection.ipynb` | Optuna — モデル選択そのものを探索対象にする |
| `07_search_comparison.ipynb` | 同一条件での探索比較 — 自前 vs Optuna vs Pruning |
| `08_sampler_overhead.ipynb` | サンプラー単体の処理時間 — Rustuna が効く条件を判定する |
| `09_calibration_triage.ipynb` | 確率の校正と LOW/REVIEW/HIGH の 3 段振り分け |

### `06_nn_basics/` — 第 6 章 2-A 基礎と学習の安定化

train / eval モード

| ファイル | 内容 |
|---|---|
| `01_train_eval_mode.ipynb` | model.train() と model.eval() の違いを実測する |
| `02_mlp_backprop_numpy.ipynb` | NumPy だけで MLP の順伝播・逆伝播を書き、勾配チェックする |

### `07_cnn_rnn/` — 第 7 章 2-B CNN と RNN

SmallCNN・双方向 LSTM

| ファイル | 内容 |
|---|---|
| `01_small_cnn.ipynb` | SmallCNN と学習ループ |
| `02_sentiment_lstm.ipynb` | 双方向 LSTM と勾配クリッピング |

### `08_transformer/` — 第 8 章 2-C Transformer と生成モデル

MLP / CNN / Transformer

| ファイル | 内容 |
|---|---|
| `01_mlp_cnn_transformer.ipynb` | MLP → CNN → 簡易 Transformer |
| `02_self_attention_numpy.ipynb` | Self-Attention を NumPy で計算する（√d_k・マスク・O(n²)） |

### `09_clustering/` — 第 9 章 3-A クラスタリング

4 手法比較・k の決め方・GMM

| ファイル | 内容 |
|---|---|
| `01_four_methods.ipynb` | クラスタリング 4 手法を同じデータで比較 |
| `02_choose_k.ipynb` | k をどう決めるか — エルボー法とシルエット |
| `03_gmm_bic.ipynb` | GMM の k と共分散型を BIC で選ぶ |

### `10_dimreduction_anomaly/` — 第 10 章 3-B 次元削減と異常検知

PCA・t-SNE・異常検知

| ファイル | 内容 |
|---|---|
| `01_pca.ipynb` | PCA — 何次元まで落とせるか |
| `02_tsne.ipynb` | t-SNE による可視化 — PCA で前処理してから掛ける |
| `03_anomaly_detection.ipynb` | 異常検知 — 正常データだけで学習する |

### `11_mdp_q/` — 第 11 章 4-A MDP と価値ベース

表形式 Q 学習・DQN

| ファイル | 内容 |
|---|---|
| `01_tabular_q_learning.ipynb` | 表形式 Q 学習（FrozenLake） |
| `02_dqn_skeleton.ipynb` | DQN の骨格 — ReplayBuffer と ターゲットネットワーク |

### `12_policy_gradient/` — 第 12 章 4-B 方策ベース・発展・実践

Gymnasium・PPO・SAC

| ファイル | 内容 |
|---|---|
| `01_gymnasium_basics.ipynb` | Gymnasium の基本 API とランダム方策のベースライン |
| `02_ppo_cartpole.ipynb` | PPO で CartPole を解く |
| `03_sac_pendulum.ipynb` | SAC で連続行動を扱う（Pendulum） |

## 教科書との対応

本書『Pythonで学ぶ 機械学習の基礎と実践』の各節に、対応するファイルのパスを記載してある。
パスはこのフォルダ（`ml-code/`）を起点とする。

## 注意

- **数値は環境で変わる。** とくに秒数は CPU 数で大きく動く。
  教科書に載せた実測値は Python 3.10.17 / scikit-learn 1.7.2 / LightGBM 4.7 / XGBoost 3.2 /
  CatBoost 1.2 / Optuna 5.0、macOS 26.6.2（Apple Silicon、CPU 実行）で測ったもの。**比率と桁を読むこと。**
- **ネットワークが要るノートがある。**
  `02_regression/01_linear_regression.ipynb`（California Housing）と
  `07_cnn_rnn/01_small_cnn.ipynb`（MNIST）は、取得に失敗すると合成データへ自動で切り替わる。
- 合成データでの精度は**手順の確認用**であって、実務の根拠にはならない。
- 探索系のノートは、書籍より**試行数・データ件数を小さくして**ある（CPU で試せるように）。
  各セルのコメントに書籍の値を併記してあるので、必要なら戻すこと。

## ローカルの生成物

探索履歴のSQLite DB、学習ログ、取得済みデータ、保存したモデルは実行時に生成され、Git対象から除外する。
`tmp/` は内部資料・退避ファイル用で、一括実行の対象にも含めない。
Notebookの警告出力に含まれる実行環境のローカルパスは、公開用に伏せている。
