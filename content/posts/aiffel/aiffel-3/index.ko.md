---
title: "아이펠 2주차 수요일- 다양한 데이터 전처리 기법"
date: 2023-11-03
description: "아이펠 회고"
categories: [개념 공부]
tags: [아이펠, AI]
slug: /fundamenta-7-aiffel
showLikes: true
showViews: true
authors: 
    - tripleyoung
showAuthor: article.showAuthor
draft: false
series: ["AI 공부"]
series_order: 1
showTable: true
---


## 7-2 결측치(Missing Data)

처리방법 크게 2가지 

1. 결측치 제거 
2. 결측치를 어떤 값으로 대체
.isnull() 결측치가 있는 데이터는 true 없으면 false로 결과값 알려줌 

.isnull().any(axis=1)  축에 결측치가 하나라도 있으면 불리언 값으로 결과를 알려줌.

dropna 결측치를 삭제 시켜주기위한 메서드 how로 옵션 설정과 subset으로 컬럼선택 

loc 로 값을 변경할수 있음. 



## 7-3 중복된 데이터 

`DataFrame.duplicated()`는 데이터 중복 여부를 불리언 값으로 반환해 줍니다

`DataFrame.drop_duplicates`를 통해 중복된 데이터를 손쉽게 삭제할 수 있습니다

`DataFrame.drop_duplicates`의 `subset`, `keep` 옵션을 통해 손쉽게 중복을 제거할 수 있습니다. subset은 컬럼 선택 keep은 first와 last로 어느것을 남길지 선택 



## 7-4 이상치(Outlier)

가장 먼저 생각해 볼 수 있는 간단하고 자주 사용되는 방법은 평균과 표준편차를 이용하는 `z score` 방법입니다.

평균을 빼주고 표준편차로 나눠 z score(σX−μ​)를 계산합니다. 

z score가 특정 기준을 넘어서는 데이터에 대해 이상치라고 판단합니다.

기준을 작게 하면 이상치라고 판단하는 데이터가 많아지고, 기준을 크게 하면 이상치라고 판단하는 데이터가 적어집니다.

이상치를 판단한 뒤 어떻게 해야 할까요?

가장 간단한 방법으로 **이상치를 삭제**할 수 있습니다. 이상치를 원래 데이터에서 삭제하고, 이상치끼리 따로 분석하는 방안도 있습니다.

**이상치를 다른 값으로 대체**할 수 있습니다. 데이터가 적으면 이상치를 삭제하기보다 다른 값으로 대체하는 것이 나을 수 있습니다. 예를 들어 최댓값, 최솟값을 설정해 데이터의 범위를 제한할 수 있습니다.

혹은 결측치와 마찬가지로 다른 데이터를 활용하여 **예측 모델을 만들어 예측값**을 활용할 수도 있습니다.

아니면** binning을 통해 수치형 데이터를 범주형으로 바꿀 수**도 있습니다.

abs 절대값 std 표준편차 

z-score의 단점 

Robust하지 못합니다. 왜나하면 **평균과 표준편차 자체가 이상치의 존재에 크게 영향을 받기 때문**입니다.
**작은 데이터셋의 경우 z-score의 방법으로 이상치를 알아내기 어렵습니다**. 특히 item이 **12개 이하인 데이터셋에서는 불가능**합니다.

### IQR method 4분위수 
하지만 이상치를 찾는 방법에는 위에 설명한 z-score 방법만 있는 것은 아닙니다. 그리고 z-score 방법은 몇 가지 뚜렷한 한계점을 가지고 있습니다. z-score 방법의 대안으로 사분위 범위수 IQR(Interquartile range) 로 이상치를 알아내는 방법을 알아보겠습니다.

 3->75% 1->25%

즉, IQR은 제 3사분위수에서 제 1사분위 값을 뺀 값으로 데이터의 중간 50%의 범위라고 생각하시면 됩니다. Q1​−1.5∗IQR보다 왼쪽에 있거나, Q3​+1.5∗IQR 보다 오른쪽에 있는 경우 우리는 이상치라고 판단합니다.

아래 그림을 보면 이 식의 의미가 좀 더 와닿을 겁니다.

![content img](https://d3s0tskafalll9.cloudfront.net/media/images/F-19-1.max-800x600.jpg "")

7-5  정규화 (Normalization)

trade 데이터를 보면 수입건수, 수출건수와 수입금액, 수출금액, 무역수지는 단위가 다르다는 것을 알 수 있습니다.

-> 정규화로 문제해결  스케일을 맞추기 위해 

정규화를 하는 방법은 다양하지만, 가장 잘 알려진 표준화(Standardization)와 Min-Max Scaling을 알아보도록 하겠습니다.

- **Standardization**
    - 데이터의 평균은 0, 분산은 1로 변환합니다.
    - Standardization은 보통 평균이 0이고 표준편차가 1일 때 사용합니다. 그렇기에 데이터가 가우시안 분포를 따를 경우 유용합니다.
- **Min-Max Scaling**
    - 데이터의 최솟값은 0, 최댓값은 1로 변환합니다.
    - Min-Max Scaling은 피처의 범위가 다를 때 주로 사용하며 확률 분포를 모를 때 유용합니다.   X−Xmin/Xmax−Xmin
    - 
scikit-learn의 `StandardScaler`, `MinMaxScaler`를 사용하는 방법도 있습니다.



##  7-6 원핫인코딩 


머신러닝이나 딥러닝 프레임워크에서 범주형을 지원하지 않는 경우 원-핫 인코딩을 해야 합니다.

원-핫 인코딩이란 카테고리별 이진 특성을 만들어 해당하는 특성만 1, 나머지는 0으로 만드는 방법입니다

pandas에서 `get_dummies` 함수를 통해 손쉽게 원-핫 인코딩을 할 수 있습니다.

`pd.concat` 함수 데이터프레임 합치기 



## 7-7.구간화
데이터를 구간별로 나누고자 합니다. 이러한 기법을 구간화(`Data binning` 혹은 `bucketing`)이라고 부릅니다.



pd.cut(data, bins=bins)   bins범주 

`qcut`은 구간을 일정하게 나누는 것이 아니라 데이터의 분포를 비슷한 크기의 그룹



## 8-2 **탐색적 데이터 분석(Exploratory Data Analysis, 이하 EDA)** 

 EDA는 더 좋은 데이터 분석과 더 좋은 머신러닝 모델을 만들기 위해 필수적인 과정



##  8-4
우리가 타겟으로 두고 확인할 데이터는 `Legendary (전설의 포켓몬인지 아닌지의 여부)`이므로, `Legendary == True` 값을 가지는 레전드 포켓몬 데이터셋은 `legendary` 변수에, `Legendary == False` 값을 가지는 일반 포켓몬 데이터셋은 `ordinary` 변수에 저장해두겠습니다

```
.reset_index(drop=True)
```
앞의 인덱스 번호를 초기화 





```
len(set(pokemon["#"]))
#set으로 칼럼에 해당되는 종류 알수 있음.
```
```
set(pokemon["Type 2"]) - set(pokemon["Type 1"])
#차집합구하기
isna()
#데이터가 비어있는 nan값의 개수 확인 
sort_value
# 높은 것부터 낮은 순으로 정렬
```
`countplot`은 말 그대로 데이터의 개수를 표시하는 플롯

```
n1, n2, n3, n4, n5 = legendary[3:6], legendary[14:24], legendary[25:29], legendary[46:50], legendary[52:57]
names = pd.concat([n1, n2, n3, n4, n5]).reset_index(drop=True)
names
#pd.concat으로 합침
```
#### **이름에 알파벳이 아닌 문자가 들어간 경우 전처리하기**
이 중 가장 먼저 '알파벳이 아닌 문자'를 포함하는 경우를 처리하도록 하겠습니다.
어떤 문자열이 알파벳으로만 이루어져 있는지를 확인하고 싶을 때는 `isalpha()` 함수를 사용하면 편리합니다.



문자열을 원하는 다른 문자열로 바꾸고 싶을 때는 pandas의 `replace` 함수를 사용하면 됩니다



#### **이름을 띄어쓰기 & 대문자 기준으로 분리해 토큰화하기**
그러면 이제 모든 이름은 세 가지 타입으로 나뉘므로 토큰화(tokenizing)할 수 있습니다.
이름에 있는 토큰을 추출하기 위해 이름을 토큰화 (모든 토큰으로 분리) 할 수 있는 함수를 생성해 주겠습니다.

문자열을 처리할 때는 주로 **정규표현식(RegEx: Regular Expression)** 이라는 기법이 사용됩니다.
정규표현식이란 문자열을 처리하는 방법의 하나로, 특정한 조건을 만족하는 문자를 검색하거나 치환하는 등의 작업을 하고 싶을 때 간편하게 처리할 수 있게 해주는 도구입니다.



붙어있는 이름을 우리는 `대문자로 시작해서 소문자로 끝나는 토큰들`로 분리하고 싶습니다.
그러기 위해서는 `대문자로 시작해서 소문자로 끝나는 패턴`을 찾으면 됩니다.

패턴을 찾는 것은 앞서 import 한 `re` 패키지의 `findall` 함수를 이용해서 실행



list 또는 set의 자료형에서 각 요소의 개수를 다루고 싶을 때에는 파이썬의 `collection`이라는 패키지를 사용하면 편리합니다.
`collection`은 순서가 있는 딕셔너리인 `OrderedDict`, 요소의 개수를 카운트하는 `Counter` 등 여러 다양한 모듈을 제공

`most_common`을 활용하면 가장 많은 요소와 등장 횟수가 무엇인지를 정렬



- 18가지의 모든 Type를 모두 원-핫 인코딩(One-Hot Encoding)합니다.
- 두 가지 속성을 가진 포켓몬은 두 가지 Type에 해당하는 자리에서 1 값을 가지도록 합니다.
여기에서 원-핫 인코딩이란, 주어진 카테고리 중 단 하나만 1(True), 나머지는 모두 0(False)으로 나타나도록 인코딩하는 방식을 말합니다.

```
confusion_matrix(y_test, y_pred)
---------------------
array([[144,   3],
[  5,   8]])              위 값은 왼쪽 위부터 순서대로 TN, FP, FN, TP

```


- TN (True Negative) : 옳게 판단한 Negative, 즉 일반 포켓몬을 일반 포켓몬이라고 알맞게 판단한 경우입니다.
- FP (False Positive) : 틀리게 판단한 Positive, 즉 일반 포켓몬을 전설의 포켓몬이라고 잘못 판단한 경우입니다.
- FN (False Negative) : 틀리게 판단한 Negative, 즉 전설의 포켓몬을 일반 포켓몬이라고 잘못 판단한 경우입니다.
- TP (True Positive) : 옳게 판단한 Positive, 즉 전설의 포켓몬을 전설의 포켓몬이라고 알맞게 판단한 경우입니다.
```
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
--------
precision    recall  f1-score   support

       False       0.97      0.98      0.97       147
        True       0.73      0.62      0.67        13

    accuracy                           0.95       160
   macro avg       0.85      0.80      0.82       160
weighted avg       0.95      0.95      0.95       160
```
Recall=TP/(FN+TP)

Recall이 낮다는 것은 분모에 있는 FN값이 높다는 것을 뜻



![image.png](https://eraser.imgix.net/workspaces/gather%3APZwSUGN3GtuF8R9htable4/6cg3FlQZenfnW15jWx53mpr0UGn1/6tbFZ6q7OL1lZyYUU5rbg.png?ixlib=js-3.7.0 "image.png")



