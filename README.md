# Data_Capstone_2021_2

## 주제 선정 배경 
일반적인 얼굴 인식의 경우 얼굴 사진 데이터에 방해요소 없이 온전한 상태의 얼굴 형태가 많다. 하지만 실세계에서 우리가 얼굴을 인식할 때에는 마스크나 선글라스로 얼굴의 부분이 가려져 있거나, 혹은 사진 자체적으로 얼굴 전체를 담지 못하는 경우가 많다. 이런 부분을 감안한 얼굴 인식을 수행하기 위해 손상된 얼굴 사진으로부터 온전한 얼굴 사진을 얻을 필요가 있다.

## 과제 주요 내용
Face dataset 으로부터 occluded face dataset 을 생성 혹은 occluded face dataset 을 수집
dataset 구성
- Paired dataset(synthetic occlusion) : Face-de-occlusion dataset (https://github.com/xweiyuan/Face-de-occlusion-using-3D-morphable-model-and-generative-adversarial-network)
- non-paired dataset(natural occlusion) : ROF-Real world Occluded Faces (https://github.com/ekremerakin/RealWorldOccludedFaces)

선정된 GAN을 이용한 Occluded face recovery
- 사용 모델-Occlusion Aware GAN (https://github.com/swordcheng/OA-GAN)

PSNR, SSIM 을 통한 원본 이미지와의 비교

모델 구조 변경을 통한 성능 향상

## 과제 세부 내용

### 가이드라인 논문 선정
얼굴 이미지 복원에 관한 여러 논문들을 찾아보고, 조작된 occlusion과 자연적으로 생긴 occlusion을 번갈아가며 학습해, 자연적으로 생긴 occlusion에 대한 일반화 성능을 높인 OA-GAN에 흥미가 생겼고, 해당 모델에 대해 공부 및 성능을 높여보고 싶어 선정했다.

### 가이드라인 논문 이해 
![image](https://user-images.githubusercontent.com/33544078/146284695-4513ddae-9d09-4002-96b7-01016820a3f0.png)
OA-GAN의 구조이다. WGAN의 loss 함수와 Gradient penalty regularization을 적용했다.

#### Generator
Generator는 Occlusion-aware module과 Face-completion module로 나뉜다.

##### Occlusion aware module 
convolution, transpose convolution과 Residual Block을 이용해 Occluded image의 feature map을 얻는다.
이 feature map에 convolution + sigmoid 를 통해 Occlusion mask(Occlusion area를 시각화한 mask)를 얻는다.
Occlusion mask와 occluded face image에 대해 element wise 곱을 하게 되면 occlsion area가 제거된 face image를 얻는다.

##### face completion module


