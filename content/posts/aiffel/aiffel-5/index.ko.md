---
title: "아이펠 2주차 요일- 성능 지표와 Linear , Conv Layer"
date: 2023-11-03
description: "아이펠 회고"
categories: [개념 공부]
tags: [아이펠, AI]
slug: /fundamenta-4-aiffel
showLikes: true
showViews: true
authors: 
    - tripleyoung
showAuthor: article.showAuthor
draft: true
series: ["AI 공부"]
series_order: 2
showTable: true
---





# 4. Data 어떻게 표현하면 좋을까? 배열(array)과 표(table)


- List와 array의 차이
    - list는 다양한 자료형을 저장할 때 유용하며, 형태가 유동적일 경우에 유용, 별도의 모듈이 필요하지 않습니다.
    - array는 동일한 자료형의 데이터를 처리할 때, 빠른 처리 속도가 필요할 때 사용, 파이썬의 표준 모듈인 array 모듈에서 제공
## NumPy
---

- Numerical Python의 줄임말
- 과학 계산용 고성능 컴퓨팅과 데이터 분석에 필요한 파이썬 패키지
- pip로 설치 가능
- 빠르고 메모리를 효율적으로 사용하여 벡터의 산술 연산과 브로드캐스팅 연산을 지원하는 다차원 배열 `**ndarray**`  데이터 타입을 지원한다.
- 반복문을 작성할 필요 없이 전체 데이터 배열에 대해 빠른 연산을 제공하는 다양한 표준 수학 함수를 제공한다.
- 배열 데이터를 디스크에 쓰거나 읽을 수 있다. (즉 파일로 저장한다는 뜻입니다)
- 선형대수, 난수발생기, 푸리에 변환 가능, C/C++ 포트란으로 쓰여진 코드를 통합한다.
- `**ndarray.size**` : 행렬 내 원소의 개수
- `**ndarray.shape**` :행렬의 모양
- `**ndarray.ndim**` :행렬의 축(axis)의 개수
- `**reshape()**` :행렬의 모양을 바꿈
- `dtype` 은 NumPy ndarray의 "원소"의 데이터 타입을 반환
- `type(A)` 을 이용하면 행렬 A의 자료형이 반환
- np.eye() : 단위 행렬
- np.zeros(): 0 행렬
- np.ones(): 1 행렬
- 브로드캐스팅 지원
- 슬라이스 인덱싱 연산 제공
- random 지원 
    - `**np.random.randint()**` : 범위 내 난수 하나 생성
    - `**np.random.choice()**` :주어진 값 중 하나를 랜덤하게 골라
    - `**np.random.permutation()**` : 무작위로 섞인 배열을 만들어 줍니다
    - `**np.random.normal()**` 
        - np.random.normal(loc=0, scale=1, size=5) # 평균(loc), 표준편차(scale), 추출개수(size)
    - `**np.random.uniform()**` 
        - np.random.uniform(low=-1, high=1, size=5)  # 최소(low), 최대(high), 추출개수(size)
- 전치행렬
    - `**arr.T**` : 행렬의 행과 열 맞바꾸기
    - `**np.transpose**` : 축을 기준으로 행렬의 행과 열 바꾸기
- 기본 통계 데이터 계산
    -  `sum()` , 평균 `mean()` , 표준편차 `std()` , 중앙값 `median()`  등을 구현하지 않고도 손쉽게 통계작업을 진행가능


### Data Representation
---

- 이미지
    - 이미지는 수많은 점(픽셀)들로 구성
    - 각각의 픽셀은 R, G, B 값 3개 요소의 튜플로 색상이 표시
    - 이미지와 관련된 파이썬 라이브러리: matplotlib, PIL
        - 두 라이브러리는 이미지 파일을 열고, 자르고, 복사하고, rgb 색상 값을 가져오는 등 이미지 파일과 관련된 몇 가지 작업을 수행
    - **이미지 조작에 쓰이는 메서드**
        - open : `**Image.open()**` 
        - size : `**Image.size**` 
        - filename : `**Image.filename**` 
        - crop : `**Image.crop((x0, y0, xt, yt))**` 
        - resize : `**Image.resize((w,h))**` 
        - save : `**Image.save()**` 


- 데이터의 값을 찾을 때 인덱스가 아닌 "한국", "미국" 등의 키(key)를 사용해 데이터에 접근하는 데이터 구조를 해시(hash)라고 합니다.
- 구조화된 데이터란  **데이터 내부에 자체적인 서브 구조를 가지는 데이터**
    - 테이블(table) 형태로 전개


## pandas
---

- NumPy기반에서 개발되어 NumPy를 사용하는 애플리케이션에서 쉽게 사용 가능
- 축의 이름에 따라 데이터를 정렬할 수 있는 자료 구조
- 다양한 방식으로 인덱싱(indexing)하여 데이터를 다룰 수 있는 기능
- 통합된 시계열 기능과 시계열 데이터와 비시계열 데이터를 함께 다룰 수 있는 통합 자료 구조
- 누락된 데이터 처리 기능
- 데이터베이스처럼 데이터를 합치고 관계 연산을 수행하는 기능
- 
- series 
    - 일련의 객체를 담을 수 있는, 1차원 배열과 비슷한 자료 구조
    - 리스트, 튜플, 딕셔너리를 통해서 만들거나 NumPy 자료형(정수형, 실수형 등)으로도 만들 수 있음
    - `index` 와 `value` 가 있음
    - 인덱스는 기본적으로 정수 형태로 설정되고, 사용자가 원하면 값을 할당할 수 있음
    - 파이썬 딕셔너리 타입의 데이터를 `Series`  객체로 손쉽게 나타낼 수 있음
    - `Series`  객체와 `Series`  인덱스는 모두 `name`  속성이 있습니다. 
- DataFrame
    - , DataFrame은 여러 개의 컬럼을 나타낼 수 있음 
        - csv 파일이나 excel 파일에 적용
## pandas와 함께 EDA 시작하기 
---

**CSV 파일 읽기 : **read.csv

**head(), tail(): data 앞, 뒤 살펴보기**

**.columns: 컬럼명 확인**

**.info(): 데이터에 대한 대략적인 설명, null값 ,자료형 등 확인 가능**

**.describe(): 데이터의 **기본적 통계 데이터(평균, 표준편차 등) 확인 가능 

**.isnull().sum(): null 값 총 개수 확인**

**.value_counts(): 범주형 데이터에 범주별로 값이 몇개 있는지 확인**

**.value_counts().sum(): 합 확인 **

**.sum(): **`sum()`을 컬럼에 단독으로 적용해서 해당 컬럼 값의 총합을 구할 수 있습니다.

**.corr(): 상관관계 확인**



## pandas에서 제공하는 통계 관련 메서드
---

- `**count()**` : NA를 제외한 수를 반환합니다.
- `**describe()**` : 요약 통계를 계산합니다.
- `**min(), max()**` : 최소, 최댓값을 계산합니다.
- `**sum()**` : 합을 계산합니다.
- `**mean()**` : 평균을 계산합니다.
- `**median()**` : 중앙값을 계산합니다.
- `**var()**` : 분산을 계산합니다.
- `**std()**` : 표준편차를 계산합니다.
- `**argmin()**` , `**argmax()**` : 최소, 최댓값을 가지고 있는 값을 반환합니다.
- `**idxmin()**` , `**idxmax()**` : 최소, 최댓값을 가지고 있는 인덱스를 반환합니다.
- `**cumsum()**` : 누적 합을 계산합니다.
- `**pct_change()**` : 퍼센트 변화율을 계산합니다.




# 5.  데이터를 한눈에! Visualization
- import matplotlib.pyplot as plt
- import seaborn as sns
- 
- **그래프 안에 추가적으로 글자나 화살표 등 주석을 그릴 때는 **`**annotate()**` ** 메서드를 이용**
- **plt.plot()의 인자로 x 데이터, y 데이터, 마커 옵션, 색상 등을 지정할 수 있음**
    - **pandas.plot 메서드 인자**
        - **label: 그래프의 범례 이름**
        - **ax: 그래프를 그릴 matplotlib의 서브플롯 객체**
        - **style: matplotlib에 전달할 'ko--'같은 스타일의 문자열**
        - **alpha: 투명도 (0 ~1)**
        - **kind: 그래프의 종류: line, bar, barh, kde**
        - **logy: Y축에 대한 로그 스케일**
        - **use_index: 객체의 색인을 눈금 이름으로 사용할지의 여부**
        - **rot: 눈금 이름을 로테이션(0 ~ 360)**
        - **xticks, yticks: x축, y축으로 사용할 값**
        - **xlim, ylim: x축, y축 한계**
        - **grid: 축의 그리드 표시할지 여부**
- 
    - **pandas의 data가 DataFrame 일 때 plot 메서드 인자**
        - **subplots: 각 DataFrame의 칼럼(column)을 독립된 서브플롯에 그립니다.**
        - **sharex: **`**subplots=True**` **면 같은 X축을 공유하고 축의 범위와 눈금을 연결합니다.**
        - **sharey: **`**subplots=True**` **면 같은 Y축을 공유합니다.**
        - **figsize: 그래프의 크기를 지정합니다. (튜플)**
        - **title: 그래프의 제목을 지정합니다. (문자열)**
        - **sort_columns: 칼럼을 알파벳 순서로 그립니다.**
## (정리) 그래프 그리는 방법
**그래프를 그리는 과정을 다시 정리해 봅시다.**

1. `**fig = plt.figure()**`**: figure 객체를 선언해 '도화지를 펼쳐'줍니다.**
2. ****`**ax1 = fig.add_subplot(1,1,1)**`**: 축을 그립니다.**
3. ****`**ax1.bar(x, y)**`** 축 안에 어떤 그래프를 그릴지 메서드를 선택한 다음, 인자로 데이터를 넣어줍니다.**
4. **그래프 타이틀 축의 레이블 등을 plt의 여러 메서드 **`**grid**`**, **`**xlabel**`**, **`**ylabel**`** 을 이용해서 추가해 주고 **`**plt.savefig**`** 메서드를 이용해 저장해줍니다.**


**Heatmap**은 방대한 양의 데이터와 현상을 수치에 따른 색상으로 나타내는 것으로, 데이터 차원에 대한 제한은 없으나 모두 2차원으로 시각화하여 표현.

**kde** = 확률 밀도 그래프 > 연속된 확률 분포를 나타냄, 가우시안으로 나타낼 수 있음

**히스토그램** : 도수분포표를 그래프로 나타낸 것

- 가로: 변수의 구간, bin(or bucket)
- 세로: 빈도수, frequency
- 전체 총량 : n


**범주형 데이터**

- 주로 막대 그래프를 이용해서 수치를 요약 
    - plt.bar(x = x, height = y)
    - sns.barplot(data=df, x='sex', y='tip')


**수치형 데이터**

- 산점도 혹은 선 그래프으로 표현
    - sns.scatterplot(data=df, x='total_bill', y='tip', hue='day')
    - sns.lineplot(x=x, y=np.sin(x)ㅌ)


## 6. 사이킷런으로 구현해 보는 머신러닝
---

- 사이킷런에서의 알고리즘 TASK : 4개
    - 분류 : 7개 / SGD Classifier, KNeighborsClassifier, LinearSVC, NaiveBayes, SVC, Kernel approximation, EnsembleClassifiers
    - 회귀: 7개/  SGD Regressor, Lasso, ElasticNet, RidgeRegression, SVR(kernel='linear'), SVR(kernel='rbf'), EnsembelRegressor
    - 차원축소
    - 클러스터링
- 위와 같이 나누는 기준은 데이터 수량라벨의 유무(정답의 유무)데이터의 종류 (수치형 데이터(quantity), 범주형 데이터(category) 등)
- Ensemble은 Classification, Regression에 속해있음
- SGD는 Classification, Regression에 속해있음


- Estimator
    - **사이킷런에서 데이터셋을 기반으로 머신러닝 모델의 파라미터를 추정하는 객체**
    - ** 사이킷런의 모든 머신러닝 모델은 이 객체로 구현되어 있음**
    - fit(), prediction()


- train_test_split()
    - **훈련 데이터와 테스트 데이터를 편하게 분리할 수 있도록 해주는 사이킷런의 함수**
    - from sklearn.model_selection import train_test_split
    - X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


- Pipeline, meta-estimator
    - ** Estimator와 transformer() 2가지 기능을 수행하는 scikit-learn의 API**


- transformer()
    - **scikit-Learn에서 ETL(Extract Transform Load) 기능을 수행하는 함수특성 행렬(Feature Matrix)**


#### 특성 행렬(Feature Matrix)
- 입력 데이터를 의미
- 특성(feature): 특성 행렬의 열에 해당하는 값
- 표본(sample): 각 입력 데이터, 특성 행렬에서는 행에 해당하는 값
- `**n_samples**` : 행의 개수(표본의 개수)
- `**n_features**` : 열의 개수(특성의 개수)
- `**X**` : 통상 특성 행렬은 변수명 X로 표기합니다.
- `**[n_samples, n_features]**` 은 [행, 열] 형태의 2차원 배열 구조를 사용하며 이는 _NumPy의 ndarray, Pandas의 DataFrame, SciPy의 Sparse Matrix_를 사용하여 나타낼 수 있습니다.


#### 타겟 벡터 (Target Vector)
- 입력 데이터의 **라벨(정답)** 을 의미
- 목표(Target): 라벨, 타겟값, 목표값이라고도 부르며 특성 행렬(Feature Matrix)로부터 예측하고자 하는 것을 말합니다.
- `**n_samples**` : 벡터의 길이(라벨의 개수)
- 타겟 벡터에서 `**n_features**` 는 없습니다.
- `**y**` : 통상 타겟 벡터는 변수명 y로 표기합니다.
- 타겟 벡터는 보통 1차원 벡터로 나타내며, 이는 _NumPy의 ndarray, Pandas의 Series_를 사용하여 나타낼 수 있습니다.
- (단, 타겟 벡터는 경우에 따라 1차원으로 나타내지 않을 수도 있습니다. 이 노드에서 사용되는 예제는 모두 1차원 벡터입니다.)


! **특성 행렬 X의 n_samples와 타겟 벡터 y의 n_samples는 동일해야 함.**



---

****



list와 array 구별

- List와 array의 차이
    - list는 다양한 자료형을 저장할 때 유용하며, 형태가 유동적일 경우에 유용, 별도의 모듈이 필요하지 않습니다.
    - array는 동일한 자료형의 데이터를 처리할 때, 빠른 처리 속도가 필요할 때 사용, 파이썬의 표준 모듈인 array 모듈에서 제공(import array as arr)
        - array는 element들이 연속된 메모리 공간에 배치되며, 모든 element들이 동일한 크기와 타입을 가져야 합니다. 그러므로 위에서 `myarray.append('4')` 는 허용되지 않습니다.
데이터 출력 방법들

.head() 앞에 5행 

.tail() 뒤에 5행 



브로드캐스트 연산(4-5)

브로드캐스팅 조건

**1. 원소가 하나인 배열은 어떤 배열이나 브로드캐스팅이 가능**

**2. 하나의 배열이 1차원 배열인 경우, 브로드캐스팅이 가능**

**3. 차원의 짝이 맞을때 브로드캐스팅이 가능**



transpose를 하고 어떤 숫자를 넣을지?

index

데이터 행렬변(4-7)

dataframe 과 series의 차이

Series는 한 개의 인덱스 컬럼과 값 컬럼, 딕셔너리는 키 컬럼과 값 컬럼과 같이 2개의 컬럼만 존재하는데 비해, DataFrame은 여러 개의 컬럼을 나타낼 수 있습니다



4대천왕(5-7 ~ 5-9)

막대그래프 / 산점도 / 선그래프 / 히스토그램



어떠한 데이터를 주고 어떤 그래프가 적절할지?

ex) 범주형 데이터를 주고 어떤 그래프가 적절할지?



**Q. 사이킷런에서 알고리즘의 Task는 몇 가지이며 각각 무엇인가요?(6-3)**

4가지. Classification, Regression, Clustering, Dimensionality Reduction



estimator 객체(6-9)

추정을 하는 과정 즉, 훈련은 `Estimator`의 `fit()`메서드를 통해 이루어지고, 예측은 `predict()`메서드를 통해 이루어집니다.



> 데이터셋 로드하기

훈련용 데이터셋 나누기

훈련하기

예측하기

정답률 출력하기





train_test_split









