# DL_Basic — 딥러닝 입문 강의 자료

이곳에 오신 분들 많이 많이 환영합니다. 👍 아래 내용이 도움이 되시면 그냥 가지 마시고~~
**오른쪽 상단 'Star 버튼'** 꾹 눌러주시면 큰 힘이 됩니다.^^

딥러닝 입문부터 MNIST·CNN·RNN·GAN·전이학습(YOLO)·PyTorch 실전·캐글 참여까지,
**TensorFlow/Keras**와 **PyTorch** 두 프레임워크로 구성한 강의 자료실입니다. 실습 노트북을 HTML·PDF로 정리해 누구나 바로 열람할 수 있습니다.

![GitHub stars](https://img.shields.io/github/stars/LDJWJ/DL_Basic?style=flat-square)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Basics%20to%20Project-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![Keras](https://img.shields.io/badge/Keras-red?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x%2B-EE4C2C?style=flat-square)
![Google Colab](https://img.shields.io/badge/Colab-Open%20in%20Colab-blueviolet?style=flat-square)
![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Hosted-green?style=flat-square)
![Last update](https://img.shields.io/badge/Last%20update-2025%2F10-success?style=flat-square)

**웹 페이지**: [https://ldjwj.github.io/DL_Basic/](https://ldjwj.github.io/DL_Basic/) — GitHub Pages에서 모든 자료를 바로 열람할 수 있습니다.

---

## 종합 목차

- [빠른 시작](#빠른-시작)
- [1. 딥러닝 기초 이론](#1-딥러닝-기초-이론)
- [2. 실습 코드 및 Notion](#2-실습-코드-및-notion)
- [3. 딥러닝 시작하기](#3-딥러닝-시작하기)
- [4. MNIST 이미지 분류](#4-mnist-이미지-분류)
- [5. 영화 리뷰와 뉴스 분류](#5-영화-리뷰와-뉴스-분류)
- [6. CNN 이미지 분류](#6-cnn-이미지-분류)
- [7. 강아지 고양이 분류](#7-강아지-고양이-분류)
- [8. TensorFlow 2.x 실습](#8-tensorflow-2x-실습)
- [9. PyTorch 실습](#9-pytorch-실습)
- [10. 자연어 처리](#10-자연어-처리)
- [11. RNN 순환신경망](#11-rnn-순환신경망)
- [12. GAN](#12-gan)
- [13. 전이학습 객체인식 OCR](#13-전이학습-객체인식-ocr)
- [14. 캐글 실전](#14-캐글-실전)
- [대표 실습 미리보기](#대표-실습-미리보기)
- [참고 사이트](#참고-사이트)

---

## 빠른 시작

1. **프레임워크를 고릅니다.** 어떤 스택으로 딥러닝을 시작할지 정합니다.
   - **TensorFlow/Keras** → 딥러닝 기본 · MNIST · CNN · RNN · GAN (part 04)
   - **PyTorch** → 텐서 · 모델 구축 · 선형회귀 · Wine · 캐글 실전 (part 06)
2. **실습 환경을 설치합니다.** (5분 소요)
   - Windows: [TensorFlow 2.9 설치 PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01A_딥러닝기본환경만들기_tf29_window.pdf) · [설치 매뉴얼(HTML)](https://ldjwj.github.io/DL_Basic/딥러닝_기본_환경_만들기_window_3_8_13(tf2.9).html)
   - macOS: [TensorFlow 2.9 설치 PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01A_딥러닝기본환경만들기_tf29_mac.pdf)
3. **첫 실습을 실행합니다.**
   - Google Colab에서 바로 실행 → [`part01_dl_basic_V10_2510.ipynb`](https://github.com/LDJWJ/DL_Basic/blob/main/class_code/part01_dl_basic_V10_2510.ipynb)
     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LDJWJ/DL_Basic/blob/main/class_code/part01_dl_basic_V10_2510.ipynb)
   - 또는 [numpy · Keras 시작하기](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_02_KerasStart_numpy_2211.html) → [Titanic 실습](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_03_Neural_Net_Titanic_2207.html) 순서로 진행

> **상태 표기 안내**
> ✅ 사용 권장 (최신 버전) · ⚠️ 구버전 참고용 · 🚧 실행 확인 필요 (버전 최신화 필요)

---

## 1. 딥러닝 기초 이론

### 주요 강의 PDF

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| AI 윤리 (DL00) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL00_AI윤리_v10.pdf) | ✅ |
| 기본개발 환경설치 (Windows, TF 2.9) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01A_딥러닝기본환경만들기_tf29_window.pdf) | ✅ |
| 기본개발 환경설치 (macOS, TF 2.9) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01A_딥러닝기본환경만들기_tf29_mac.pdf) | ✅ |
| 기본개발 환경설치 (Windows, TF 2.6 구버전) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01A_딥러닝기본환경만들기_tf26.pdf) | ⚠️ |
| 딥러닝 시작하기 (v13, 2023.05) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01_딥러닝입문_v13_2305.pdf) | ✅ |
| 딥러닝 기본용어 (v11, 2022.07) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL02_딥러닝기본용어_v11_2207.pdf) | ✅ |
| 딥러닝 기본 (v12, 2022.07) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL02_딥러닝기본_v12_2207.pdf) | ✅ |
| CNN 시작하기 (알고리즘 v11) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL03_CNN01_알고리즘_v11.pdf) | ✅ |
| CNN 업그레이드 (모델 개선, 2207) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝입문_DL03_CNN03_모델UP_2207_v02.pdf) | ✅ |
| 텐서플로우 기본 소개 (TF 2.0, 2002) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL04_tf2.0소개_2002_v02.pdf) | ⚠️ |
| 모델 개선 (Dropout) | [PDF](https://ldjwj.github.io/DL_Basic/part04_07_dl_tf2x_modelUp/DL03_02_DROPOUT.pdf) | ✅ |
| GAN 기본입문 (202011) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝입문_DL04_기본_GAN_202011_v05.pdf) | ✅ |
| RNN 기본입문 | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝입문_DL05_기본_RNN_v11.pdf) | ✅ |
| YOLO 소개 | [PDF](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/Yolo_Intro_V12.pdf) | ✅ |

> RNN 관련 참고: 2024년 9월 기준 자연어 처리 분야 등에서는 Transformer 모델로 많은 부분이 대체되고 있습니다. 다만 Transformer 모델의 등장에도 시계열 예측, 음성 인식 등의 분야에서 여전히 중요한 역할로 사용될 것으로 보입니다.

### 구버전 · 참고용 PDF

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| 딥러닝입문 (v10, 2021.11) | [PDF](https://ldjwj.github.io/DL_Basic/01C_딥러닝입문_v10_2111.pdf) | ⚠️ |
| 딥러닝입문 (v12, 2022.12) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL01B_딥러닝입문_v12_2212.pdf) | ⚠️ |
| CNN 알고리즘 (v02, 2022.05) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL03_CNN01_알고리즘_v02_2205.pdf) | ⚠️ |
| CNN 알고리즘 (v10) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL03_CNN01_알고리즘_v10.pdf) | ⚠️ |
| CNN 알고리즘 (v02, 2022.05) | [PDF](https://ldjwj.github.io/DL_Basic/딥러닝기본_DL03_CNN02_알고리즘_v02_2205.pdf) | ⚠️ |

---

## 2. 실습 코드 및 Notion

- 딥러닝 기본 및 실습 **Notion 학습 페이지**: [Link](https://www.notion.so/_GP510-2817d480b593802bbf3ec228ad2cafc4?source=copy_link)
- 실습 코드 (`class_code/part01_dl_basic_V10_2510.ipynb`):
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LDJWJ/DL_Basic/blob/main/class_code/part01_dl_basic_V10_2510.ipynb)

---

## 3. 딥러닝 시작하기

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| Python 기본 함수 및 기본 라이브러리 | [HTML](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_01_Python_Library_V10_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_01_Python_Library_V10_2211.pdf) | ✅ |
| numpy 기본 · Keras 시작 | [HTML](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_02_KerasStart_numpy_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_02_KerasStart_numpy_2211.pdf) | ✅ |
| Bike Sharing Demand (회귀) | [HTML](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_02_Neural_Net_Bike_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_02_Neural_Net_Bike_2211.pdf) | ✅ |
| Titanic (분류) | [HTML](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_03_Neural_Net_Titanic_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_03_Neural_Net_Titanic_2207.pdf) | ✅ |
| Titanic with PyTorch (2411) | [HTML](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_03_NNet_Titanic_pytorch_V01_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_03_NNet_Titanic_pytorch_V01.pdf) | ✅ |

---

## 4. MNIST 이미지 분류

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| MNIST 01 — 첫 모델 (WC) | [HTML](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_02_DL_mnist01_firstmodel_WC_V11.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_02_DL_mnist01_firstmodel_WC_V11.pdf) | ✅ |
| MNIST 02 — 모델 성능 올리기 | [HTML](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_02_DL_mnist02_up_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_02_DL_mnist02_up_2207.pdf) | ✅ |
| MNIST 03 — 성능 개선 | [HTML](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_03_DL_mnist03_up_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_03_DL_mnist03_up_2207.pdf) | ✅ |
| MNIST 04 — EarlyStopping | [HTML](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_04_DL_mnist04_EarlyStop_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_04_DL_mnist04_EarlyStop_2207.pdf) | ✅ |

---

## 5. 영화 리뷰와 뉴스 분류

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| 영화 리뷰 긍정/부정 분류 (IMDB) | [HTML](https://ldjwj.github.io/DL_Basic/part04_03_dl_pratice/ch03_01_01_movie_imdb_classification_2208.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_03_dl_pratice/ch03_01_01_movie_imdb_classification_2208.pdf) | ✅ |
| 뉴스 기사 분류 | [HTML](https://ldjwj.github.io/DL_Basic/part04_03_dl_pratice/ch03_05_01_news_classification_2205.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_03_dl_pratice/ch03_05_01_news_classification_2205.pdf) | ✅ |

---

## 6. CNN 이미지 분류

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| CNN FirstModel (LAB01) | [HTML](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_01_LAB01_CNN_A1_V11.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_01_LAB01_CNN_A1_V11.pdf) | ✅ |
| CNN SecondModel (LAB02) | [HTML](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_02_keras_LAB02_CNN_A2_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_01_Keras_LAB02_CNN_A2_2211.pdf) | ✅ |
| CNN cs231n 참조 | [HTML](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_02/ch05_04_CNN_D_add_cs231n.html) | ✅ |
| CNN Upgrade — LeakyReLU (LAB03) | [HTML](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_02_keras_LAB03_CNN_add_B1.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_02_keras_LAB03_CNN_add_B1.pdf) | ✅ |
| CNN Conv3 (LAB04) | [HTML](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_02_keras_LAB04_CNN_add_B2.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_04_dl_cnn_01/ch05_02_keras_LAB04_CNN_add_B2.pdf) | ✅ |
| Dogs vs Cats (로컬) | [HTML](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_11_tf2x_CNN_CatvsDogs(1)_wlocal_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/ch05_11_tf2x_CNN_CatvsDogs(1)_wlocal_2211.pdf) | ✅ |
| CIFAR-10 (1) | [HTML](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_03_tf2x_LAB07_CNN_cifar10_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_03_tf2x_LAB07_CNN_cifar10_2211.pdf) | ✅ |
| CIFAR-10 (3) — 모델 save | [.py](https://github.com/LDJWJ/DL_Basic/blob/main/part05_01_flask/model_cifar.py) | ✅ |

---

## 7. 강아지 고양이 분류

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| CNN Colab 환경 (V10, 2411) | [HTML](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_CatvsDogs(1)_wcolab_V10_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_CatvsDogs(1)_wcolab_V10_2411.pdf) | ✅ |
| CNN Colab 환경 — PyTorch (V10) | [HTML](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_CatvsDogs(1)_PyTorch_V10_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_CatvsDogs(1)_PyTorch_V10_2411.pdf) | ✅ |
| CNN 캐글 노트북 — 기본 모델 | [HTML](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_catvsdogs_withkaggle(1).html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_catvsdogs_withkaggle(1).pdf) | 🚧 |
| CNN (2) — 이미지 제너레이터 | [HTML](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_CatvsDogs(2)_wcolab_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_CNN_CatvsDogs(2)_wcolab_2207.pdf) | 🚧 |
| CNN (3) — VGG16 | [HTML](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_VGG_CatvsDogs(3)_wColab_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_VGG_CatvsDogs(3)_wColab_2207.pdf) | 🚧 |
| CNN (4) — VGG16 | [HTML](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_VGG_CatvsDogs(4)_wcolab_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_05_dl_cnn_catvsdog/ch05_07_VGG_CatvsDogs(4)_wcolab_2207.pdf) | 🚧 |

> 아래 (2),(3),(4) 자료는 버전 최신화가 필요합니다. 구버전으로 실행이 안 될 가능성이 있습니다. (24/11 기준)

---

## 8. TensorFlow 2.x 실습

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| MNIST — TF 2.0 시작 | [HTML](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_09_tf20_start.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_09_tf20_start.pdf) | ✅ |
| Fashion MNIST — DNN | [HTML](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_10_fashionMNIST01_DNN_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_10_fashionMNIST01_DNN_2211.pdf) | ✅ |
| Fashion MNIST — CNN | [HTML](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_10_fashionMNIST02_CNN_2211.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_10_fashionMNIST02_CNN_2211.pdf) | ✅ |
| 캐글 — Dogs vs Cats | [HTML](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_11_tf20_CatsandDog(2)_onkaggle.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_11_tf20_CatsandDog(2)_onkaggle.pdf) | ✅ |

---

## 9. PyTorch 실습

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| 텐서의 이해 및 기본 함수 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_01_pytorch_tensor_V10_colab.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_01_pytorch_tensor_V10_colab.pdf) | ✅ |
| 딥러닝 모델 구축해 보기 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_02_pytorch_start_colab_V10.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_02_pytorch_start_colab_V10.pdf) | ✅ |
| 선형회귀 모델 기본 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_03_pytorch_LinearRegression_wc_V10.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_03_pytorch_LinearRegression_wc_V10.pdf) | ✅ |
| Wine 딥러닝 모델 구축 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_04_pytorch_wine_v10_colab.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_04_pytorch_wine_v10_colab.pdf) | ✅ |
| 타이타닉 딥러닝 모델 만들기 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_03_PyTorch_Simple_wColab_V01_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_03_PyTorch_Simple_wColab_V01_2411.pdf) | ✅ |
| Fashion MNIST 딥러닝 모델 구축 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_05_pt_fashionMnist_V10_wc.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_05_pt_fashionMnist_V10_wc.pdf) | ✅ |
| RNN 기본 실습 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch06_04_pytorch_rnn_v10.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch06_04_pytorch_rnn_v10.pdf) | ✅ |
| 캐글 — Exploring Mental Health Data | [대회 링크](https://www.kaggle.com/competitions/playground-series-s4e11) | — |
| 캐글 — 기본1 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_06_pytorch_kaggle_Health01_V01_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_06_pytorch_kaggle_Health01_V01_2411.pdf) | ✅ |
| 캐글 — 기본2 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_06_pytorch_kaggle_Health02_V01_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_06_pytorch_kaggle_Health02_V01_2411.pdf) | ✅ |
| 캐글 — 기본3 | [HTML](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_06_pytorch_kaggle_Health03_V01_2411.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_06_pytorch_kaggle_Health03_V01_2411.pdf) | ✅ |

---

## 10. 자연어 처리

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| 텍스트 데이터 처리 (1) — 전처리 · 원핫 | [HTML](https://ldjwj.github.io/DL_Basic/part04_08_text/ch06_textA_preprocessing_onehot_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_08_text/ch06_textA_preprocessing_onehot_2207.pdf) | ✅ |
| 텍스트 데이터 처리 (2) — Word Embedding | [HTML](https://ldjwj.github.io/DL_Basic/part04_08_text/ch06_textB_wordembedding_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_08_text/ch06_textB_wordembedding_2207.pdf) | ✅ |
| 텍스트 데이터 처리 (3) — 사전훈련모델 | [HTML](https://ldjwj.github.io/DL_Basic/part04_08_text/ch06_textC_pretrained.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_08_text/ch06_textC_pretrained.pdf) | ✅ |

---

## 11. RNN 순환신경망

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| RNN 실습 (1) — 순서가 있는 값 예측 | [HTML](https://ldjwj.github.io/DL_Basic/part04_09_RNN/ch06_01_RNN_simpleRNN_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_09_RNN/ch06_01_RNN_simpleRNN_2207.pdf) | ✅ |
| RNN 실습 (2) — IMDB · SimpleRNN | [HTML](https://ldjwj.github.io/DL_Basic/part04_09_RNN/ch06_02_RNN_SimpleRNN_IMDB_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_09_RNN/ch06_02_RNN_SimpleRNN_IMDB_2207.pdf) | ✅ |
| RNN 실습 (3) — IMDB · LSTM | [HTML](https://ldjwj.github.io/DL_Basic/part04_09_RNN/ch06_03_RNN_LSTM_IMDB_2207.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_09_RNN/ch06_03_RNN_LSTM_IMDB_2207.pdf) | ✅ |

---

## 12. GAN

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| GAN — 적대적 생성 신경망 (MNIST) | [HTML](https://ldjwj.github.io/DL_Basic/part04_09_GAN/ch08_03_gan_mnist_start_2208.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_09_GAN/ch08_03_gan_mnist_start_2208.pdf) | ✅ |
| GAN — Neural Style Transfer | [HTML](https://ldjwj.github.io/DL_Basic/part04_09_GAN/ch08_03_keras_GAN_NeuralStyleTransfer.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_09_GAN/ch08_03_keras_GAN_NeuralStyleTransfer.pdf) | ✅ |

---

## 13. 전이학습 객체인식 OCR

| 자료 | 형태 | 상태 |
|:---|:---:|:---:|
| ResNet50 전이학습 | [HTML](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/resnet50_Basic_01_202305.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/resnet50_Basic_01_202305.pdf) | ✅ |
| YOLOv8 시작 — 객체 인식 ("You Only Look Once") | [HTML](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/yolov8_start_01_v10.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/yolov8_start_01_v10.pdf) | ✅ |
| YOLOv8 — 모델 평가 · yaml 파일 만들기 | [HTML](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/yolov8_start_02_v10.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/yolov8_start_02_v10.pdf) | ✅ |
| 텍스트 OCR — 이미지에서 텍스트 추출 | [HTML](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/TextExt_Basic_01.html) · [PDF](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/TextExt_Basic_01.pdf) | ✅ |

---

## 14. 캐글 실전

### 추가 데이터셋 실습

| 자료 | 내용 | 링크 |
|:---|:---|:---:|
| Surface Crack Detection | 콘크리트 표면 샘플 이미지 (손상 예측) | [데이터셋](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) |
| 자동차 드라이버 이미지 분석 | 이미지 확인 및 GIF 만들기 | [HTML](https://ldjwj.github.io/DL_Basic/part04_10_kaggle/kaggle-driver01-imageshow-220625.html) |
| 자동차 드라이버 이미지 분석 | 이미지 EDA 및 CNN 모델 구축 | [HTML](https://ldjwj.github.io/DL_Basic/part04_10_kaggle/kaggle-driver02-imageshow-220625.html) |
| Surface Crack Detection | CNN 기본 모델 구축 | [HTML](https://ldjwj.github.io/DL_Basic/part04_10_kaggle/ch05_07_CNN_crackDetection_CNN.html) · [PDF](https://ldjwj.github.io/DL_Basic/part04_10_kaggle/ch05_07_CNN_crackDetection_CNN.pdf) |

### 추천 캐글 대회

| 대회 | 링크 |
|:---|:---:|
| Titanic — Machine Learning from Disaster | [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic) |
| Bike Sharing Demand | [https://www.kaggle.com/c/bike-sharing-demand](https://www.kaggle.com/c/bike-sharing-demand) |
| Dogs vs. Cats | [https://www.kaggle.com/c/dogs-vs-cats/overview](https://www.kaggle.com/c/dogs-vs-cats/overview) |
| Spaceship Titanic | [https://www.kaggle.com/competitions/spaceship-titanic](https://www.kaggle.com/competitions/spaceship-titanic) |
| PUBG Finish Placement Prediction | [https://www.kaggle.com/competitions/pubg-finish-placement-prediction](https://www.kaggle.com/competitions/pubg-finish-placement-prediction) |
| American Express — Default Prediction | [https://www.kaggle.com/competitions/amex-default-prediction](https://www.kaggle.com/competitions/amex-default-prediction) |
| ASHRAE — Great Energy Predictor II | [https://www.kaggle.com/competitions/ashrae-energy-prediction](https://www.kaggle.com/competitions/ashrae-energy-prediction) |

---

## 대표 실습 미리보기

썸네일을 클릭하면 실제 실습 페이지(HTML)로 이동합니다.

| 실습 | 미리보기 |
|:---|:---:|
| [Titanic (Keras)](https://ldjwj.github.io/DL_Basic/part04_01_dl_start/ch01_03_Neural_Net_Titanic_2207.html) | ![Titanic 실습](assets/gallery/titanic.jpg) |
| [MNIST 첫 모델](https://ldjwj.github.io/DL_Basic/part04_02_dl_mnist/ch01_04_02_DL_mnist01_firstmodel_WC_V11.html) | ![MNIST 실습](assets/gallery/mnist.jpg) |
| [Dogs vs Cats (TF 2.x)](https://ldjwj.github.io/DL_Basic/part04_06_dl_tf2x/ch05_11_tf2x_CNN_CatvsDogs(1)_wlocal_2211.html) | ![Dogs vs Cats 실습](assets/gallery/catdog.jpg) |
| [GAN — MNIST](https://ldjwj.github.io/DL_Basic/part04_09_GAN/ch08_03_gan_mnist_start_2208.html) | ![GAN 실습](assets/gallery/gan.jpg) |
| [PyTorch — Wine](https://ldjwj.github.io/DL_Basic/part06_01_pytorch/ch01_04_pytorch_wine_v10_colab.html) | ![PyTorch Wine 실습](assets/gallery/pytorch_wine.jpg) |
| [YOLOv8 시작](https://ldjwj.github.io/DL_Basic/part06_02_ocr_yolo_resnet/yolov8_start_01_v10.html) | ![YOLO 실습](assets/gallery/yolo.jpg) |

---

## 참고 사이트

- [Neural Network Playground](https://playground.tensorflow.org/) — 신경망을 눈으로 보고 직접 실험해볼 수 있는 사이트 (https://playground.tensorflow.org/)
- 내용 중 일부는 『케라스 창시자에게 배우는 딥러닝』을 참조하였습니다.

---

자료가 도움이 되셨다면 **GitHub Star**로 알려주세요. 실습 노트북·강의 자료는 버전이 올라가면서 계속 갱신됩니다.