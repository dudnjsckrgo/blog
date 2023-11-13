---
title: "6D Object Pose Estimation"
date: 2023-11-09
description: "6D Object Pose Estimation 논문 분석 시작하기"
categories: [논문 분석]
tags: [iros20-6d-pose-tracking , 6D Object Pose Estimation]
slug: /6d-pose-estimation
showLikes: true
showViews: true
authors: 
    - tripleyoung
showAuthor: article.showAuthor
draft: false
series: ["6D Object Pose Estimation 논문 분석"]
series_order: 1
showTable: true
---
## 논문 분석를 시작해야겠다는 결심
제가 XR 회사를 다닐때, 타 업체의 AR 오브젝트 트랙킹 솔루션들을 리서치를 해봤습니다.
vuforia, wikitude 등 쟁쟁한 AR 업체들의 솔루션들을 보았는데, 괜찮은 오브젝트 트랙킹이 없었습니다.
그래서 **저는 이때 생각을 했습니다.**

딥러닝을 활용한 

성능 좋은 

- 움직임에 Robust한
- 물체가 가렸을때 Robust한

오브젝트 트랙킹 AR 솔루션이 없구나!

그래서 이걸 만들어야 겠다고 생각 했습니다.

그래서 저는 최신 오브젝트 트랙킹 딥러닝 논문을 리서치하기로 했습니다.


## 첫번째로 

국제 지능형 로봇 및 시스템 컨퍼런스(IROS) 2020에서 채택 된 

### se(3)-TrackNet 

#### 합성 도메인에서 이미지 잔여물을 보정하여 데이터 기반 6D 포즈 추적

의 공식 구현을 분석할려고 합니다.

{{< link_block3 url="https://github.com/wenbowen123/iros20-6d-pose-tracking" >}}
