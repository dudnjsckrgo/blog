---
title: "iros20-6d-pose-tracking 논문 리뷰"
date: 2023-11-03
description: "iros20-6d-pose-tracking 논문 리뷰"
categories: [논문 리뷰]
tags: [iros20-6d-pose-tracking , 6D Object Pose Estimation]
slug: /iros20-6d-pose-tracking
showLikes: true
showViews: true
authors: 
    - tripleyoung
showAuthor: article.showAuthor
draft: false
series: ["6D Object Pose Estimation 논문 리뷰"]
series_order: 1
showTable: true
---
## 논문 리뷰에 앞서
- ### 논문 리뷰 선택 이유
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



