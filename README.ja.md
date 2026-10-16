🇰🇷 [한국어](README.md) | 🇺🇸 [English](README.en.md) | 🇯🇵 **日本語**

# DL_Basic — ディープラーニング入門 講義資料

ここに来てくださった皆さん、大歓迎です。👍 内容が役に立ったら、そのまま帰らずに~~
**右上の「Star」ボタン** を押していただけると大きな励みになります。^^

ディープラーニング入門からMNIST・CNN・RNN・GAN・転移学習（YOLO）・PyTorch実践・Kaggle参加まで、
**TensorFlow/Keras** と **PyTorch** の2つのフレームワークで構成した講義資料室です。実習ノートブックをHTML・PDFにまとめているので、誰でもすぐに閲覧できます。

> **ご注意：** 教材（PDF/HTML）はすべて **韓国語** で書かれています。コードはそのまま実行でき、HTMLページはブラウザ（Chromeなど）の自動翻訳で読むことができます。

![GitHub stars](https://img.shields.io/github/stars/LDJWJ/DL_Basic?style=flat-square)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Basics%20to%20Project-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![Keras](https://img.shields.io/badge/Keras-red?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x%2B-EE4C2C?style=flat-square)
![Google Colab](https://img.shields.io/badge/Colab-Open%20in%20Colab-blueviolet?style=flat-square)
![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Hosted-green?style=flat-square)
![Last update](https://img.shields.io/badge/Last%20update-2025%2F10-success?style=flat-square)

**Webページ**: [https://ldjwj.github.io/DL_Basic/README.ja.html](https://ldjwj.github.io/DL_Basic/README.ja.html) — GitHub Pagesですべての資料をすぐに閲覧できます。

---

## 総合目次

- [クイックスタート](#クイックスタート)
- [1. ディープラーニングの基礎理論](#1-ディープラーニングの基礎理論)
- [2. 実習コードとNotion](#2-実習コードとnotion)
- [3. ディープラーニングを始めよう](#3-ディープラーニングを始めよう)
- [4. MNIST 画像分類](#4-mnist-画像分類)
- [5. 映画レビューとニュースの分類](#5-映画レビューとニュースの分類)
- [6. CNN 画像分類](#6-cnn-画像分類)
- [7. 犬と猫の分類](#7-犬と猫の分類)
- [8. TensorFlow 2.x 実習](#8-tensorflow-2x-実習)
- [9. PyTorch 実習](#9-pytorch-実習)
- [10. 自然言語処理](#10-自然言語処理)
- [11. RNN 再帰型ニューラルネットワーク](#11-rnn-再帰型ニューラルネットワーク)
- [12. GAN](#12-gan)
- [13. 転移学習 物体検出 OCR](#13-転移学習-物体検出-ocr)
- [14. Kaggle実戦](#14-kaggle実戦)
- [代表的な実習のプレビュー](#代表的な実習のプレビュー)
- [参考サイト](#参考サイト)

---

## クイックスタート

1. **フレームワークを選びます。** どのスタックでディープラーニングを始めるか決めます。
   - **TensorFlow/Keras** → ディープラーニングの基本 · MNIST · CNN · RNN · GAN（part 04）
   - **PyTorch** → テンソル · モデル構築 · 線形回帰 · Wine · Kaggle実戦（part 06）
2. **実習環境をインストールします。**（約5分）
   - Windows: [TensorFlow 2.9 インストールPDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01A_딥러닝기본환경만들기_tf29_window.pdf) · [インストールマニュアル（HTML）](https://ldjwj.github.io/DL_Basic/딥러닝_기본_환경_만들기_window_3_8_13(tf2.9).html)
   - macOS: [TensorFlow 2.9 インストールPDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01A_딥러닝기본환경만들기_tf29_mac.pdf)
3. **最初の実習を実行します。**
   - Google Colabですぐに実行 → [`part01_dl_basic_V10_2510.ipynb`](https://github.com/LDJWJ/DL_Basic/blob/main/class_code/part01_dl_basic_V10_2510.ipynb)
     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LDJWJ/DL_Basic/blob/main/class_code/part01_dl_basic_V10_2510.ipynb)
   - または [numpy · Kerasを始めよう](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_02_KerasStart_numpy_2211.html) → [Titanic実習](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_03_Neural_Net_Titanic_2207.html) の順に進めてください

> **状態マークの説明**
> ✅ 推奨（最新版） · ⚠️ 旧版・参考用 · 🚧 動作確認が必要（バージョン更新が必要）

---

## 1. ディープラーニングの基礎理論

### 主な講義PDF

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| AI倫理（DL00） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL00_AI윤리_v10.pdf) | ✅ |
| 開発環境のインストール（Windows、TF 2.9） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01A_딥러닝기본환경만들기_tf29_window.pdf) | ✅ |
| 開発環境のインストール（macOS、TF 2.9） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01A_딥러닝기본환경만들기_tf29_mac.pdf) | ✅ |
| 開発環境のインストール（Windows、TF 2.6 旧版） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01A_딥러닝기본환경만들기_tf26.pdf) | ⚠️ |
| ディープラーニングを始めよう（v13、2023.05） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01_딥러닝입문_v13_2305.pdf) | ✅ |
| ディープラーニングの基本用語（v11、2022.07） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL02_딥러닝기본용어_v11_2207.pdf) | ✅ |
| ディープラーニングの基本（v12、2022.07） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL02_딥러닝기본_v12_2207.pdf) | ✅ |
| CNNを始めよう（アルゴリズム v11） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL03_CNN01_알고리즘_v11.pdf) | ✅ |
| CNNのアップグレード（モデル改善、2207） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝입문_DL03_CNN03_모델UP_2207_v02.pdf) | ✅ |
| TensorFlowの基本紹介（TF 2.0、2002） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL04_tf2.0소개_2002_v02.pdf) | ⚠️ |
| モデル改善（Dropout） | [PDF](https://ldjwj.github.io/DL_Basic/part04_07_dl_tf2x_modelUp/DL03_02_DROPOUT.pdf) | ✅ |
| GAN 基本入門（202011） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝입문_DL04_기본_GAN_202011_v05.pdf) | ✅ |
| RNN 基本入門 | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝입문_DL05_기본_RNN_v11.pdf) | ✅ |
| YOLOの紹介 | [PDF](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/Yolo_Intro_V12.pdf) | ✅ |

> RNNに関する補足：2024年9月時点で、自然言語処理などの分野ではTransformerモデルに多くの部分が置き換えられています。ただしTransformerの登場後も、時系列予測や音声認識などの分野では引き続き重要な役割を担うと考えられます。

### 旧版 · 参考用PDF

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| ディープラーニング入門（v10、2021.11） | [PDF](https://ldjwj.github.io/DL_Basic/01C_딥러닝입문_v10_2111.pdf) | ⚠️ |
| ディープラーニング入門（v12、2022.12） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01B_딥러닝입문_v12_2212.pdf) | ⚠️ |
| CNNアルゴリズム（v02、2022.05） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL03_CNN01_알고리즘_v02_2205.pdf) | ⚠️ |
| CNNアルゴリズム（v10） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL03_CNN01_알고리즘_v10.pdf) | ⚠️ |
| CNNアルゴリズム（v02、2022.05） | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL03_CNN02_알고리즘_v02_2205.pdf) | ⚠️ |

---

## 2. 実習コードとNotion

- ディープラーニングの基本と実習 **Notion学習ページ**: [Link](https://www.notion.so/_GP510-2817d480b593802bbf3ec228ad2cafc4?source=copy_link)
- 実習コード（`class_code/part01_dl_basic_V10_2510.ipynb`）:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LDJWJ/DL_Basic/blob/main/class_code/part01_dl_basic_V10_2510.ipynb)

---

## 3. ディープラーニングを始めよう

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| Pythonの基本関数と基本ライブラリ | [HTML](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_01_Python_Library_V10_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_01_Python_Library_V10_2211.pdf) | ✅ |
| numpyの基本 · Kerasを始めよう | [HTML](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_02_KerasStart_numpy_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_02_KerasStart_numpy_2211.pdf) | ✅ |
| Bike Sharing Demand（回帰） | [HTML](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_02_Neural_Net_Bike_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_02_Neural_Net_Bike_2211.pdf) | ✅ |
| Titanic（分類） | [HTML](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_03_Neural_Net_Titanic_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_03_Neural_Net_Titanic_2207.pdf) | ✅ |
| Titanic with PyTorch（2411） | [HTML](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_03_NNet_Titanic_pytorch_V01_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_03_NNet_Titanic_pytorch_V01.pdf) | ✅ |

---

## 4. MNIST 画像分類

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| MNIST 01 — 最初のモデル（WC） | [HTML](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_02_DL_mnist01_firstmodel_WC_V11.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_02_DL_mnist01_firstmodel_WC_V11.pdf) | ✅ |
| MNIST 02 — モデルの性能を上げる | [HTML](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_02_DL_mnist02_up_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_02_DL_mnist02_up_2207.pdf) | ✅ |
| MNIST 03 — 性能改善 | [HTML](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_03_DL_mnist03_up_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_03_DL_mnist03_up_2207.pdf) | ✅ |
| MNIST 04 — EarlyStopping | [HTML](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_04_DL_mnist04_EarlyStop_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_04_DL_mnist04_EarlyStop_2207.pdf) | ✅ |

---

## 5. 映画レビューとニュースの分類

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| 映画レビューの肯定/否定分類（IMDB） | [HTML](https://ldjwj.github.io/DL_Basic/part04_03_dl_pratice/ch03_01_01_movie_imdb_classification_2208.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_03_dl_pratice/ch03_01_01_movie_imdb_classification_2208.pdf) | ✅ |
| ニュース記事の分類 | [HTML](https://ldjwj.github.io/DL_Basic/part04_03_dl_pratice/ch03_05_01_news_classification_2205.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_03_dl_pratice/ch03_05_01_news_classification_2205.pdf) | ✅ |

---

## 6. CNN 画像分類

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| CNN FirstModel（LAB01） | [HTML](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_01_LAB01_CNN_A1_V11.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_01_LAB01_CNN_A1_V11.pdf) | ✅ |
| CNN SecondModel（LAB02） | [HTML](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_02_keras_LAB02_CNN_A2_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_01_Keras_LAB02_CNN_A2_2211.pdf) | ✅ |
| CNN cs231n 参照 | [HTML](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_02/ch05_04_CNN_D_add_cs231n.html) | ✅ |
| CNN Upgrade — LeakyReLU（LAB03） | [HTML](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_02_keras_LAB03_CNN_add_B1.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_02_keras_LAB03_CNN_add_B1.pdf) | ✅ |
| CNN Conv3（LAB04） | [HTML](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_02_keras_LAB04_CNN_add_B2.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_02_keras_LAB04_CNN_add_B2.pdf) | ✅ |
| Dogs vs Cats（ローカル） | [HTML](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_11_tf2x_CNN_CatvsDogs(1)_wlocal_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/ch05_11_tf2x_CNN_CatvsDogs(1)_wlocal_2211.pdf) | ✅ |
| CIFAR-10 (1) | [HTML](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_03_tf2x_LAB07_CNN_cifar10_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_03_tf2x_LAB07_CNN_cifar10_2211.pdf) | ✅ |
| CIFAR-10 (3) — モデルの保存 | [.py](https://github.com/LDJWJ/DL_Basic/blob/main/part05_01_flask/model_cifar.py) | ✅ |

---

## 7. 犬と猫の分類

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| CNN Colab環境（V10、2411） | [HTML](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_CatvsDogs(1)_wcolab_V10_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_CatvsDogs(1)_wcolab_V10_2411.pdf) | ✅ |
| CNN Colab環境 — PyTorch（V10） | [HTML](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_CatvsDogs(1)_PyTorch_V10_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_CatvsDogs(1)_PyTorch_V10_2411.pdf) | ✅ |
| CNN Kaggleノートブック — 基本モデル | [HTML](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_catvsdogs_withkaggle(1).html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_catvsdogs_withkaggle(1).pdf) | 🚧 |
| CNN (2) — 画像ジェネレーター | [HTML](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_CatvsDogs(2)_wcolab_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_CatvsDogs(2)_wcolab_2207.pdf) | 🚧 |
| CNN (3) — VGG16 | [HTML](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_VGG_CatvsDogs(3)_wColab_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_VGG_CatvsDogs(3)_wColab_2207.pdf) | 🚧 |
| CNN (4) — VGG16 | [HTML](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_VGG_CatvsDogs(4)_wcolab_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_VGG_CatvsDogs(4)_wcolab_2207.pdf) | 🚧 |

> 上の (2)、(3)、(4) の資料はバージョンの更新が必要です。旧版のため実行できない可能性があります。（24/11時点）

---

## 8. TensorFlow 2.x 実習

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| MNIST — TF 2.0 を始めよう | [HTML](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_09_tf20_start.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_09_tf20_start.pdf) | ✅ |
| Fashion MNIST — DNN | [HTML](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_10_fashionMNIST01_DNN_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_10_fashionMNIST01_DNN_2211.pdf) | ✅ |
| Fashion MNIST — CNN | [HTML](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_10_fashionMNIST02_CNN_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_10_fashionMNIST02_CNN_2211.pdf) | ✅ |
| Kaggle — Dogs vs Cats | [HTML](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_11_tf20_CatsandDog(2)_onkaggle.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_11_tf20_CatsandDog(2)_onkaggle.pdf) | ✅ |

---

## 9. PyTorch 実習

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| テンソルの理解と基本関数 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_01_pytorch_tensor_V10_colab.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_01_pytorch_tensor_V10_colab.pdf) | ✅ |
| ディープラーニングモデルを構築してみよう | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_02_pytorch_start_colab_V10.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_02_pytorch_start_colab_V10.pdf) | ✅ |
| 線形回帰モデルの基本 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_03_pytorch_LinearRegression_wc_V10.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_03_pytorch_LinearRegression_wc_V10.pdf) | ✅ |
| Wine ディープラーニングモデルの構築 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_04_pytorch_wine_v10_colab.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_04_pytorch_wine_v10_colab.pdf) | ✅ |
| タイタニックのディープラーニングモデルを作る | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_03_PyTorch_Simple_wColab_V01_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_03_PyTorch_Simple_wColab_V01_2411.pdf) | ✅ |
| Fashion MNIST ディープラーニングモデルの構築 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_05_pt_fashionMnist_V10_wc.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_05_pt_fashionMnist_V10_wc.pdf) | ✅ |
| RNN 基本実習 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch06_04_pytorch_rnn_v10.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch06_04_pytorch_rnn_v10.pdf) | ✅ |
| Kaggle — Exploring Mental Health Data | [コンペページ](https://www.kaggle.com/competitions/playground-series-s4e11) | — |
| Kaggle — 基本1 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_06_pytorch_kaggle_Health01_V01_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_06_pytorch_kaggle_Health01_V01_2411.pdf) | ✅ |
| Kaggle — 基本2 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_06_pytorch_kaggle_Health02_V01_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_06_pytorch_kaggle_Health02_V01_2411.pdf) | ✅ |
| Kaggle — 基本3 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_06_pytorch_kaggle_Health03_V01_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_06_pytorch_kaggle_Health03_V01_2411.pdf) | ✅ |

---

## 10. 自然言語処理

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| テキストデータの処理 (1) — 前処理 · ワンホット | [HTML](https://ldjwj.github.io/DL_Basic/part04_08_text/ch06_textA_preprocessing_onehot_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_08_text/ch06_textA_preprocessing_onehot_2207.pdf) | ✅ |
| テキストデータの処理 (2) — Word Embedding | [HTML](https://ldjwj.github.io/DL_Basic/part04_08_text/ch06_textB_wordembedding_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_08_text/ch06_textB_wordembedding_2207.pdf) | ✅ |
| テキストデータの処理 (3) — 事前学習モデル | [HTML](https://ldjwj.github.io/DL_Basic/part04_08_text/ch06_textC_pretrained.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_08_text/ch06_textC_pretrained.pdf) | ✅ |

---

## 11. RNN 再帰型ニューラルネットワーク

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| RNN 実習 (1) — 順序のある値の予測 | [HTML](https://ldjwj.github.io/DL_Basic/part04_09_RNN/ch06_01_RNN_simpleRNN_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_09_RNN/ch06_01_RNN_simpleRNN_2207.pdf) | ✅ |
| RNN 実習 (2) — IMDB · SimpleRNN | [HTML](https://ldjwj.github.io/DL_Basic/part04_09_RNN/ch06_02_RNN_SimpleRNN_IMDB_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_09_RNN/ch06_02_RNN_SimpleRNN_IMDB_2207.pdf) | ✅ |
| RNN 実習 (3) — IMDB · LSTM | [HTML](https://ldjwj.github.io/DL_Basic/part04_09_RNN/ch06_03_RNN_LSTM_IMDB_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_09_RNN/ch06_03_RNN_LSTM_IMDB_2207.pdf) | ✅ |

---

## 12. GAN

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| GAN — 敵対的生成ネットワーク（MNIST） | [HTML](https://ldjwj.github.io/DL_Basic/part04_09_GAN/ch08_03_gan_mnist_start_2208.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_09_GAN/ch08_03_gan_mnist_start_2208.pdf) | ✅ |
| GAN — Neural Style Transfer | [HTML](https://ldjwj.github.io/DL_Basic/part04_09_GAN/ch08_03_keras_GAN_NeuralStyleTransfer.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_09_GAN/ch08_03_keras_GAN_NeuralStyleTransfer.pdf) | ✅ |

---

## 13. 転移学習 物体検出 OCR

| 資料 | 形式 | 状態 |
|:---|:---:|:---:|
| ResNet50 による転移学習 | [HTML](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/resnet50_Basic_01_202305.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/resnet50_Basic_01_202305.pdf) | ✅ |
| YOLOv8を始めよう — 物体検出（"You Only Look Once"） | [HTML](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/yolov8_start_01_v10.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/yolov8_start_01_v10.pdf) | ✅ |
| YOLOv8 — モデル評価 · yamlファイルの作成 | [HTML](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/yolov8_start_02_v10.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/yolov8_start_02_v10.pdf) | ✅ |
| テキストOCR — 画像からテキストを抽出 | [HTML](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/TextExt_Basic_01.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/TextExt_Basic_01.pdf) | ✅ |

---

## 14. Kaggle実戦

### 追加データセットの実習

| 資料 | 内容 | リンク |
|:---|:---|:---:|
| Surface Crack Detection | コンクリート表面のサンプル画像（損傷予測） | [データセット](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) |
| 自動車ドライバー画像の分析 | 画像の確認とGIFの作成 | [HTML](https://ldjwj.github.io/DL_Basic/part04_10_kaggle/kaggle-driver01-imageshow-220625.html) |
| 自動車ドライバー画像の分析 | 画像のEDAとCNNモデルの構築 | [HTML](https://ldjwj.github.io/DL_Basic/part04_10_kaggle/kaggle-driver02-imageshow-220625.html) |
| Surface Crack Detection | CNN 基本モデルの構築 | [HTML](https://ldjwj.github.io/DL_Basic/part04_10_kaggle/ch05_07_CNN_crackDetection_CNN.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_10_kaggle/ch05_07_CNN_crackDetection_CNN.pdf) |

### おすすめのKaggleコンペ

| コンペ | リンク |
|:---|:---:|
| Titanic — Machine Learning from Disaster | [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic) |
| Bike Sharing Demand | [https://www.kaggle.com/c/bike-sharing-demand](https://www.kaggle.com/c/bike-sharing-demand) |
| Dogs vs. Cats | [https://www.kaggle.com/c/dogs-vs-cats/overview](https://www.kaggle.com/c/dogs-vs-cats/overview) |
| Spaceship Titanic | [https://www.kaggle.com/competitions/spaceship-titanic](https://www.kaggle.com/competitions/spaceship-titanic) |
| PUBG Finish Placement Prediction | [https://www.kaggle.com/competitions/pubg-finish-placement-prediction](https://www.kaggle.com/competitions/pubg-finish-placement-prediction) |
| American Express — Default Prediction | [https://www.kaggle.com/competitions/amex-default-prediction](https://www.kaggle.com/competitions/amex-default-prediction) |
| ASHRAE — Great Energy Predictor II | [https://www.kaggle.com/competitions/ashrae-energy-prediction](https://www.kaggle.com/competitions/ashrae-energy-prediction) |

---

## 代表的な実習のプレビュー

サムネイルをクリックすると、実際の実習ページ（HTML）に移動します。

| 実習 | プレビュー |
|:---|:---:|
| [Titanic (Keras)](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_03_Neural_Net_Titanic_2207.html) | ![Titanic実習](assets/gallery/titanic.jpg) |
| [MNIST 最初のモデル](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_02_DL_mnist01_firstmodel_WC_V11.html) | ![MNIST実習](assets/gallery/mnist.jpg) |
| [Dogs vs Cats (TF 2.x)](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_11_tf2x_CNN_CatvsDogs(1)_wlocal_2211.html) | ![Dogs vs Cats実習](assets/gallery/catdog.jpg) |
| [GAN — MNIST](https://ldjwj.github.io/DL_Basic/part04_09_GAN/ch08_03_gan_mnist_start_2208.html) | ![GAN実習](assets/gallery/gan.jpg) |
| [PyTorch — Wine](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_04_pytorch_wine_v10_colab.html) | ![PyTorch Wine実習](assets/gallery/pytorch_wine.jpg) |
| [YOLOv8を始めよう](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/yolov8_start_01_v10.html) | ![YOLO実習](assets/gallery/yolo.jpg) |

---

## 参考サイト

- [Neural Network Playground](https://playground.tensorflow.org/) — ニューラルネットワークを目で見て、自分で実験できるサイト（https://playground.tensorflow.org/）
- 内容の一部は『PythonとKerasによるディープラーニング』（François Chollet著）を参考にしています。

---

資料が役に立ったら **GitHub Star** で教えてください。実習ノートブックと講義資料は、バージョンアップに合わせて随時更新しています。
