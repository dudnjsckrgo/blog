---
title: "아이펠 3주차 월요일"
date: 2023-11-03
description: "아이펠 회고"
categories: [개념 공부]
tags: [아이펠, AI]
slug: /fundamenta-5-aiffel
showLikes: true
showViews: true
authors: 
    - tripleyoung
showAuthor: article.showAuthor
draft: true
series: ["AI 공부"]
series_order: 3
showTable: true
---



# 13. 딥네트워크, 서로 뭐가 다른 거죠?
### 13-2 **이미지넷(ImageNet)**
**ImageNet Pretrained Model의 Accuracy 2가지**

Top-1 accuracy : 예측 값이 일반적으로 생각하는 정답을 맞힌 정확도입니다.
Top-5 accruacy : 예측한 확률이 높은 순서로 5개 내에서 정답이 있을 경우 맞힌 것으로 간주한 정확도입니다.



### 13-3 딥네트워크
AlexNet : Le-Net 모델을 활용,  CNN 모델, ReLU 활성화 함수와 드롭아웃(Dropout), 그리고 오버래핑 풀링(Overlapping pooling) 등이 적용



### 13-4 CNN을 잘쓰자
- output = (input image size-filter size)/stride+1)


- parameter = filter size x filter size x channel


- VGG = Visual Geometry Group
VGG 이전의 모델은 비교적 큰 filter를 사용했지만 VGG모델은 3*3 filter를 사용하면서 깊이를 늘렸다. -> 간단한 구조로 깊이에 따라 높은 성능을 보일 수 있다는 것이 특징

장점 : 결정 함수의 비선형성 증가(모델의 특징 식별성 증가), 학습 파라미터 수의 감소



### 13-5 ~ 6 멀리 있으면 잘 안들려서 지름길을 만들어줌..
- 기울기 소실(vanishing gradient)
발생하는 원인 : vanishing 또는 exploding gradient의 문제가 발생하는 원인은, 레이어가 깊어지면서 gradient가 매우 커지거나 작아지기 때문입니다.

 레이어의 가중치가 여러 번 곱해지다 보면, 가중치가 1보다 작을 때에는 gradient가 0에 너무 가까워져 버리고, 1보다 클 때에는 그 값이 기하급수적으로 커지게 됩니다.

-> ResNet에서 skip connection이 추가된 구조를 사용해서 vanishing gradient 문제를 해결했습니다.



### 13-8 VGG-16
- max pooling과 softmax 등의 활성화 함수(activation function)를 제외하면 CNN 레이어와 완전 연결 레이어(fully connected layer)를 합쳐 16개의 레이어로 구성되어 있습니다.


### 13-9 RasNet-50
- VGG 구조에서 무작정 늘린다고 성능은 좋아지지 않는데  기울기 소실(vanishing gradient)의 문제가 있기 때문으로 이러한 문제를 해결하기 위해서 Skip connection이 추가된 구조
[﻿wikidocs.net/137252](https://wikidocs.net/137252) 



# 14. 활성화 함수의 이해
### 14-2 활성화 함수(activation function)
- 노드에 입력으로 들어오는 값이 어떤 '임계치'를 넘어가면 활성화(activated) 되고, 넘어가지 않으면 비활성화(deactivated)


- 실제로 딥러닝에서 활성화 함수를 쓰는 결정적 이유는 따로 있습니다. 바로 딥러닝 모델의 표현력을 향상시켜주기 위해 사용


### 14-3 퍼셉트론
- 딥러닝 모델은 보통 여러 개의 층으로 이루어져 있습니다. 그중에 하나의 층을 가져와 다시 쪼갠다면 보통 '노드'라고 불리는 것으로 쪼개지게 되는데, 이것이 바로 **퍼셉트론(Perceptron)** 입니다.


- ** **신경세포(뉴런)에서의 가지돌기, 축삭돌기, 시냅스, 세포 활성화는 각각 퍼셉트론(노드)에서의 어떤 개념에 해당할까요?
신경세포(뉴런) - 퍼셉트론(노드)
가지돌기 - 입력 신호
축삭돌기 - 출력 신호
시냅스 - 가중치
세포 활성화 - 활성화 함수



### 14-4 선형
- 선형
**선형 변환 정의**V 와 W가 어떤 (1)벡터 공간이고 둘 모두 (2)실수 집합 (3)상에 있다고 가정하겠습니다. 이때 함수 (4) T:V→W 가 다음 두 조건을 만족할 때,

가산성(Additivity) : 모든 x,y∈V 에 대해, T(x+y)=T(x)+T(y)

동차성(Homogeneity) : 모든 x∈V,c∈R 에 대해, T(cx)=cT(x)

우리는 함수 T를 **선형 변환(linear transformation)** 이라고 부릅니다.

(1) : 간단하게 말해서 벡터를 그릴 수 있는 공간입니다. 영상에서의 좌표 평면이라고 생각하시면 됩니다.
(2) : 정확히 표현하면 같은 **체(field)에 속해 있다**고 해야 하나, 이 글에선 실수만 다루기 때문에 실수 집합 상에 있다고 표현했습니다. 체의 예로는 실수 집합 R, 유리수 집합 Q, 복소수 집합 C 등이 있습니다.
(3) : **실수 집합 상에 있다**는 말은 V를 이루는 **원소들이 실수**라는 의미입니다. 예를 들어 실수 집합 상의 V가 어떤 벡터들의 집합이라고 했을 때, 그 벡터는 실수 벡터(벡터의 각 원소가 실수)가 됩니다.
(4) : 정의역(domain)이 V이고 공역(codomain)이 W인 함수 T라는 의미입니다.



### 14-5 비선형
선형이 아닌 함수를 **비선형(Non-linear)** 함수라고 합니다.



- 활성화 함수로 비선형 함수를 쓰는 이유
1. 모델의 표현력을 향상시키기 위해서
2. 선형의 경우 여러개의 퍼셉트론이 있어도 아래의 사진과 같이f(f(f(Wx)))를 f⋆(Wx)f⋆(Wx)로 표현될 수 있기 때문에 결과가 달라지지 않기 때문
![image.png](https://eraser.imgix.net/workspaces/gather%3Ad2GFsiGRWyVvXKZqtable9/0Bjf68dM44cOhXAGqgQF5XKC38C3/anie4zr-WJhVm1FfhyE8y.png?ixlib=js-3.7.0 "image.png")



### 14-6 비선형 함수를 쓰는 이유
 활성화 함수가 **선형**이라면 모델의 표현력은 증가하지 않는다는 것을 증명하는 부분



### 14-7 이진 계단 함수
들어온 입력이 특정 임계점을 넘으면 1(혹은 True)를 출력하고 그렇지 않을 때는 0을 출력 -> OR gate와 AND gate 구현

![image.png](https://eraser.imgix.net/workspaces/gather%3Ad2GFsiGRWyVvXKZqtable9/0Bjf68dM44cOhXAGqgQF5XKC38C3/NFljXPHcY1neNTUgQAuj8.png?ixlib=js-3.7.0 "image.png")

 

이진 계단 함수의 큰 단점

1. **역전파 알고리즘(backpropagation algorithm)을 사용하지 못하는 것**
이진 계단 함수는 0에서는 미분이 안 될뿐더러 0인 부분을 제외하고 미분을 한다고 해도 미분 값이 전부 0이 나옵니다. 때문에 역전파에서 가중치들이 업데이트되지 않습니다.



2. **다중 출력은 할 수 없다**

이진 계단 함수는 출력을 1 또는 0으로 밖에 주지 못하기 때문에 다양한 클래스를 구분해야 하는 문제는 해결할 수 없습니다.



### 14-8 선형 활성화 함수
- 선형 활성화 함수의 한계
모델에 선형 활성화 함수를 사용한다면 **비선형적 특성을 지닌 데이터를 예측하지 못한다는 것**



### 14-9 비선형 활성화 함수-시그모이드, Softmax
- 시그모이드
시그모이드 함수를 쓰는 가장 주된 이유가 바로 치역이 0과 1사이라는 것입니다. 특히 확률을 예측해야 하는 모델에서 자주 사용됩니다.



![image.png](https://eraser.imgix.net/workspaces/gather%3Ad2GFsiGRWyVvXKZqtable9/0Bjf68dM44cOhXAGqgQF5XKC38C3/LLPTroKleAwnXqEnUBrwI.png?ixlib=js-3.7.0 "image.png")



- 시그모이드 함수의 단점 2가지
1. 시그모이드 함수는 0 또는 1에서 **포화(saturate)** 됩니다.
이 말은 입력값이 아무리 커져도 함수의 출력은 1에 더 가까워져 갈 뿐 1 이상으로 높아지지 않고, 입력값이 아무리 작아져도 함수의 출력은 0에 더 가까워져 갈 뿐 0 이하로 떨어지지 않는다는 의미입니다. - **그래디언트를 죽인다(kill the gradient)**



 2. 시그모이드 함수의 출력은 **0이 중심(zero-centered)이 아니다.**

여기서 발생하는 문제는 **훈련의 시간이 오래 걸리게 된다는 것**



- Softmax
1. Softmax는 10가지, 100가지 class 등 class의 수에 제한 없이 "각 class의 확률"을 구할 때 쓰입니다.
2. Softmax는 모델의 마지막 layer에서 활용이 됩니다.


### 14-10 하이퍼볼릭 탄젠트
- 하이퍼볼릭 탄젠트(tanh, Hyperbolic tangent)
1. 하이퍼볼릭 탄젠트 함수는 쌍곡선 함수 중 하나입니다.
![image.png](https://eraser.imgix.net/workspaces/gather%3Ad2GFsiGRWyVvXKZqtable9/0Bjf68dM44cOhXAGqgQF5XKC38C3/-U5T2FEw9zu5BEtEToYtJ.png?ixlib=js-3.7.0 "image.png")

 2. 0을 중심으로 하기 때문에 일반적으로 하이퍼볼릭 탄젠트 함수를 사용한 모델이 시그모이드 함수를 사용한 모델보다 더 빨리 훈련됩니다.



- 하이퍼볼릭 탄젠트 함수의 단점
하이퍼볼릭 탄젠트 함수는 -1 또는 1에서 포화됩니다.



### 14-11 ReLU(rectified linear unit)
최근 가장 많이 사용되고 있는 활성화 함수

![image.png](https://eraser.imgix.net/workspaces/gather%3Ad2GFsiGRWyVvXKZqtable9/0Bjf68dM44cOhXAGqgQF5XKC38C3/Zf7vDe5ecj8TYIeuaOuwi.png?ixlib=js-3.7.0 "image.png")



- '시그모이드 함수처럼 곡선이 포함되어 있지 않은데 어떻게 비선형적 데이터의 특징을 잡아낼까?'
'그' 논문(도전하시고 싶으시다면... 화이팅!) : [﻿ReLU Deep Neural Networks and Linear Finite Elements](https://arxiv.org/abs/1807.03973) 



- 장점
1. 계산이 빠르고 간단합니다.
ReLU 함수는 입력이 양수일 경우, 입력값을 그대로 출력하고, 음수일 경우에는 0을 출력하는 함수입니다. 이러한 단순한 계산 구조로 계산 속도가 빠르고, 복잡한 수식이나 계산 과정이 없어서 구현이 간단합니다.

2.비선형 함수입니다.
ReLU 함수는 입력이 양수일 경우에는 비선형 함수이므로, 신경망이 비선형 모델로 작동할 수 있습니다. 따라서, ReLU 함수를 활성화 함수로 사용하면 신경망이 다양한 문제를 해결하는 데 더욱 유연하게 대응할 수 있습니다.

3.Gradient Vanishing 문제를 완화할 수 있습니다.
Gradient Vanishing 문제는 역전파 알고리즘에서 층이 깊어질수록 기울기가 점점 작아져 학습이 제대로 이루어지지 않는 문제입니다. ReLU 함수는 입력이 양수일 경우에 기울기가 1로 일정하기 때문에, 층이 깊어져도 Gradient Vanishing 문제가 덜 발생할 수 있습니다.



- 단점
1. ReLU 함수의 출력값이 0이 중심이 아닙니다.
2. Dying ReLU
이것의 의미는 모델에서 ReLU를 사용한 노드가 비활성화되며 출력을 0으로만 하게 되는 것입니다.

만약, 이전 훈련 스텝에서 이 노드의 가중치가 업데이트되어 ∑i​wi​xi​+b가 가중치 w값에 의해 입력값 x에 상관없이 0이하로 나오게 되었다면, 이 이후의 업데이트에서는 그래디언트가 항상 0이 되어 가중치 업데이트가 일어나지 않게 됩니다. 즉, 이 노드의 출력값과 그래디언트가 0이 되어 **노드가 죽어** 버립니다.



### 14-12 ReLU의 단점을 극복하기 위한 시도
- Leaky ReLU
'Dying ReLU'를 발생시켰던 0을 출력하던 부분을 아주 작은 음수값을 출력하게 만들어 주어 해당 문제를 해결

```
def leaky_relu(x):
return max(0.01*x,x)
```


- PReLU
**PReLU(parametric ReLU)** 는 Leaky ReLU와 유사하지만 새로운 파라미터를 추가하여 0 미만일 때의 '기울기'가 훈련되게 했습니다. 식으로 표현하면 다음과 같습니다.

```
def prelu(x, alpha):
return max(alpha*x,x)
```


- ELU
ELU(exponential linear unit)은 ReLU의 모든 장점을 포함하며, 0이 중심점이 아니었던 단점과, 'Dying ReLU'문제를 해결한 활성화 함수입니다. 식은 다음과 같습니다.

