---
title: "IROS20 6D Pose Tracking 전체적인 간단한 코드 리뷰"
date: 2023-11-09
description: "iros20-6d-pose-tracking 논문 코드 리뷰 시작하기"
categories: [논문 리뷰]
tags: [iros20-6d-pose-tracking , 6D Object Pose Estimation]
slug: /iros20-6d-pose-tracking-2
showLikes: true
showViews: true
authors: 
    - tripleyoung
showAuthor: article.showAuthor
draft: false
series: ["6D Object Pose Estimation 논문 리뷰"]
series_order: 3
showTable: true
---
{{< link_block3 url="https://github.com/wenbowen123/iros20-6d-pose-tracking" >}}

## 전체적인 코드리뷰 
***
### config.yml 분석

이 코드는 컴퓨터 비전이나 딥러닝 모델 학습을 위한 설정 파일의 일부로 보입니다. 각 섹션을 분석해보겠습니다:

1. **data_augmentation**:
   - `hsv_noise`: 이것은 색상(Hue), 채도(Saturation), 밝기(Value)에 대한 노이즈를 추가합니다. `[15,15,15]`는 각각의 HSV 채널에 적용되는 노이즈의 정도를 나타냅니다.
   - `bright_mag`: 이미지의 밝기를 조절합니다. `[0.5,1.5]` 범위는 밝기를 50% 감소시키거나 150% 증가시키는 것을 의미합니다.
   - `gaussian_noise`: 가우시안 노이즈를 추가합니다. `rgb`는 RGB 채널에 적용되는 노이즈의 정도, `depth`는 깊이 채널에 적용되는 노이즈의 정도입니다.
   - `gaussian_blur_kernel`: 이미지에 가우시안 블러를 적용합니다. 여기서 6은 블러 효과의 크기를 나타냅니다.
   - `depth_missing_percent`: 깊이 채널에서 누락된 데이터의 비율을 나타냅니다. 여기서는 40%입니다.

2. **학습 매개변수**:
   - `learning_rate`: 학습률로, 여기서는 0.001입니다.
   - `weight_decay`: 가중치 감소는 모델이 과적합을 방지하는 데 도움을 줍니다. 여기서는 0.000001입니다.
   - `epochs`: 학습 에포크 수, 즉 전체 데이터 세트를 몇 번 반복할지 지정합니다. 여기서는 300입니다.
   - `loss_weights`: 손실 가중치를 설정합니다. `trans`와 `rot` 모두 1로 설정되어 있어, 이 두 손실이 동일하게 중요하다는 것을 의미합니다.

3. **데이터 경로**:
   - `data_path`: 훈련 데이터의 경로입니다.
   - `validation_path`: 검증 데이터의 경로입니다.

4. **기타 설정**:
   - `batch_size`: 한 번에 처리할 데이터의 크기입니다. 여기서는 200입니다.
   - `n_workers`: 데이터 로딩에 사용될 워커(병렬 처리를 위한 스레드 또는 프로세스)의 수입니다. 여기서는 20입니다.

이 설정은 딥러닝 모델의 학습 과정에서 데이터 증강, 학습 매개변수, 데이터 경로 등을 정의하는 데 사용됩니다. 이러한 설정은 모델이 실제 환경에서 더 잘 일반화하고, 과적합을 방지하며, 효율적으로 학습되도록 돕습니다.

### utils.py 분석

`Utils.py` 파일은 컴퓨터 비전이나 로보틱스 응용 프로그램, 특히 6D 포즈 추정 및 추적과 관련된 다양한 구성 요소를 포함하고 있습니다. 이 파일의 내용을 요약하면 다음과 같습니다:

1. **임포트(Imports)**:
   - 표준 라이브러리와 패키지들을 임포트합니다. 예를 들어 `os`, `sys`, `time`, `numpy`, `math`, `random`, `scipy` 등이 있습니다.
   - 3D 데이터 처리를 위한 `open3d`, 이미지 처리를 위한 `cv2`, 이미지 조작을 위한 `PIL`도 사용합니다.
   - 기하학적 변환과 회전을 처리하는 데 필요한 `transformations`도 임포트합니다.

2. **함수(Functions)**:
   - 이미지 및 포인트 클라우드 처리, 기하학적 변환, 뷰 조작에 중점을 둔 다양한 함수를 정의합니다. 주요 함수로는:
     - `rgbd2PointCloud`, `toOpen3dCloud`: RGB-D 이미지를 포인트 클라우드 및 Open3D 클라우드 형식으로 변환하는 함수입니다.
     - `compute_cloud_diameter`, `compute_obj_max_width`: 객체의 크기를 계산하는 데 사용될 것으로 보입니다.
     - `normalize_rotation_matrix`, `random_direction`, `sph2cart`: 회전 및 좌표 조작과 관련된 함수입니다.
     - `fill_depth`: 이미지나 포인트 클라우드에서 깊이 정보를 처리하는 함수로 보입니다.
   - `add`, `makeCanvas`, `crop_bbox` 같은 기본 이미지 조작 작업에 중점을 둔 유틸리티 함수도 포함되어 있습니다.

3. **클래스(Classes)**:
   - `Compose(object)`라는 단일 클래스가 정의되어 있으며, 여러 변환 또는 연산을 결합하는 데 사용될 가능성이 있습니다. 이미지 처리 파이프라인에서 흔히 볼 수 있는 패턴입니다.

4. **메인 실행 블록(Main Execution Block)**:
   - 파일에 메인 실행 블록(`if __name__ == "__main__"`)이 포함되어 있지 않아, 독립 실행 프로그램으로 실행되기보다는 다른 스크립트에서 임포트하여 사용되도록 의도된 모듈로 보입니다.

전반적으로, `Utils.py`는 3D 객체 추적 및 포즈 추정을 위한 기본 기능을 제공하는 유틸리티 모듈로 보입니다. 이 모듈은 합성 또는 실제 환경에서 3D 객체 추적 및 포즈 추정과 관련된 더 큰 시스템의 중요한 부분일 것입니다.