---
title: "아이펠 2주차 목요일- 성능 지표와 Linear , Conv Layer"
date: 2023-11-03
description: "아이펠 회고"
categories: [개념 공부]
tags: [아이펠, AI]
slug: /fundamenta-9-aiffel
showLikes: true
showViews: true
authors: 
    - tripleyoung
showAuthor: article.showAuthor
draft: true
series: ["AI 공부"]
series_order: 5
showTable: true
---



## 9-2. Loss와 Metric
- Loss : 모델 학습시 **학습데이터(train data)** 를 바탕으로 계산되어, 모델의 파라미터 업데이트에 활용되는 함수
- Metric : 모델 학습 종료 후 **테스트데이터(test data)** 를 바탕으로 계산되어, 학습된 모델의 성능을 평가하는데 활용되는 함수
## 9-3. Confusion Matrix 와 Precision/Recall
![image.png](https://eraser.imgix.net/workspaces/Us1Mhk1o4yCbUVAVk5tx/CMBaOZ93q6OB4U95U5ediQejHO32/1kvkXN6V_abfToCOVKWIJ.png?ixlib=js-3.7.0 "image.png")

### F-score
아마도 F1 score라는 개념도 익숙하실 것입니다. 이 개념은 아래의 F score에서 β가 1이 될 때를 말합니다. F1 score는 Precision과 Recall의 조화평균이 되는 값으로서, Precision과 Recall 둘 다 고려할 수 있어서 분류 모델의 훌륭한 척도가 됩니다.

## 9-4. Threshold의 변화에 따른 모델 성능
Threshold가 달라지는 것을 고려하여 전체적인 모델의 성능을 평가하는 방법으로 PR(Precision and Recall) 커브와 ROC(Receiver Operating Characteristic) 커브를 그려보는 두가지 방법



PR(Precision-Recall) 커브: Recall을 X축, Precision을 Y축에 놓고 Threshold 변화에 따른 두 값의 변화를 그래프로 그린 것입니다.

Precision과 Recall 사이의 트레이드오프 관계를 확인할 수 있습니다. 



- **불균형 데이터셋**: 한 클래스의 사례 수가 다른 클래스에 비해 매우 적은 경우, 예를 들어 사기 거래 탐지나 희귀 질병 진단 같은 문제에서 PR 곡선은 모델이 소수 클래스를 얼마나 잘 탐지하는지 더 잘 보여줍니다.
- **성능 트레이드오프 분석**: 다양한 임계값에서 모델의 TPR과 FPR 또는 Precision과 Recall의 트레이드오프를 이해하고, 비즈니스 요구 사항에 맞게 모델의 임계값을 조정할 수 있습니다.
- **모델 비교**: 서로 다른 모델이나 모델 구성을 비교할 때, AUC를 기준으로 어떤 모델이 더 나은 성능을 나타내는지를 결정할 수 있습니다.


이 파란 점선보다는 위쪽에 그려져야 하며, 가급적 (0, 1)에 가깝게 그려질 수록 우수한 분류기가 됩니다. 그러므로 ROC AUC가 클수록 상대적으로 좋은 성능의 모델이라고 할 수 있을 것입니다.





### 회귀 모델의 평가척도
**MSE, RMSE 등 square 계열 Metric과 MAE, MAPE**

### 랭킹 모델의 평가척도
**NDCG,  MRR, MAP**

### 이미지 생성 모델의 평가척도
**MSE,PSNR,  SSIM**

### 기계번역 모델의 평가척도
   **BLEU score**

## 10-2. 데이터의 형태
 다차원 데이터
## 10-3. 레이어는 어렵다?
여러 개의 논리적인 개체가 층을 이루어서 하나의 물체를 구성하는 경우, 이러한 각각의 객체를 하나의 레이어라 한다.

## 10-4. 딥러닝의 근본! Linear 레이어
 여기서 각각의 행렬들이 **Weight** 입니다

여기서 모든 Weight의 모든 요소를 **Parameter** 

지나치게 많은 Parameter는 **과적합(Overfitting) 이건 아닌듯**

**편향(Bias)**

## 10-5. 정보를 집약시키자! Convolution 레이어
 **Convolution 레이어**

"손"이라는 목적이 있음에도 모든 픽셀을 한 줄씩 살펴야 하기 때문에 비효율적이죠.

 **필터(커널)** 

 두 칸, 세 칸씩 이동하며 훑을 수도 있습니다. 그것을 결정하는 값을 **Stride**

 Convolution 연산이 **입력의 형태를 변형시킨다**

방지하기 위한 개념이 **Padding**입니다.



**손을 찾는 데에 적합한 필터**도 존재할 수 있지 않을까요? 목적에 도움이 되는 정보는 선명하게, 그렇지 않은 정보는 흐리게 만드는 필터를 상상할 수 있겠군요! 그런 멋진 필터를 훈련을 통해 찾아주는 것이 바로 Convolution 레이어가 하는 일입니다. 심지어는 단 하나의 필터가 아닌 **수십 개의 필터를 중첩**해서요



## 10-6. 핵심만 추려서 더 넓게! Pooling 레이어
### Max Pooling 레이어의 의미 :가장 값이 큰 대표 선수 하나를 뽑고 나머지는 무시하는 역할
#### translational invariance 효과
이미지는 약간의 상하좌우 시프트가 생긴다고 해도 내용상 동일한 특징이 있는데, Max Pooling을 통해 인접한 영역 중 가장 특징이 두드러진 영역 하나를 뽑는 것은 오히려 약간의 시프트 효과에도 불구하고 동일한 특징을 안정적으로 잡아낼 수 있는 긍정적 효과가 있어서 오히려 object 위치에 대한 오버피팅을 방지하고 안정적인 특징 추출 효과를 가져온다고 합니다.



#### Non-linear 함수와 동일한 피처 추출 효과
Relu와 같은 Non-linear 함수도 마찬가지로 많은 하위 레이어의 연산 결과를 무시하는 효과를 발생시키지만, 그 결과 중요한 피처만을 상위 레이어로 추출해서 올려줌으로써 결과적으로 분류기의 성능을 증진시키는 효과를 가집니다. Min/Max Pooling도 이와 동일한 효과를 가지게 됩니다.



#### Receptive Field 극대화 효과
Max Pooling이 없이도 Receptive Field를 크게 하려면 Convolutional 레이어를 아주 많이 쌓아야 합니다. 그 결과 큰 파라미터 사이즈로 인한 오버피팅, 연산량 증가, Gradient Vanishing 등의 문제를 감수해야 합니다. 이런 문제를 효과적으로 해결하는 방법으로 꼽히는 두 가지 중 하나가 Max Pooling 레이어 사용입니다. 다른 하나로는 Dilated Convolution이 있습니다. 상세한 내용은 다음 [﻿링크](https://m.blog.naver.com/sogangori/220952339643)를 참고하세요.

## 10-7. 집약된 정보의 복원! Transpose Convolution 레이어


#### Upsampling 레이어
- Nearest Neighbor : 복원해야 할 값을 가까운 값으로 복제한다.
- Bed of Nails : 복원해야 할 값을 0으로 처리한다.
- Max Unpooling : Max Pooling 때 버린 값을 실은 따로 기억해 두었다가 그 값으로 복원한다.
#### Transposed Convolution


