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
- OA-GAN의 구조이다. WGAN의 loss 함수와 Gradient penalty regularization을 적용했다.
- 간단하게 설명하자면, occluded face image 에서 face feature와 occluded area를 얻어 이를 이용해 복원된 얼굴을 생성하는 모델이다.
- 자세한 설명은 https://ieeexplore.ieee.org/document/9195444 본문에서 확인할 수 있다.

### 데이터셋 선정
- 가이드라인 모델을 적용하기 위해 두 가지의 데이터셋이 필요했다.
- 얼굴 이미지와 그 이미지에 인공적으로 occlusion 을 생성시킨 Paired dataset
- 자연적으로 occlusion이 생성되어 있는 Non-paired dataset 

#### Paired dataset 
- 처음 시도는 LFW dataset (http://vis-www.cs.umass.edu/lfw/) 에 10x10, 15x10, 20x20 크기의 black image를 위에 덮어 씌워 생성 해보려 했다.
- ![Aaron_Eckhart_0001](https://user-images.githubusercontent.com/33544078/146288521-a8078277-e044-426b-8117-c8a4fbad9180.jpg) ![Aaron_Eckhart_0001_occ_15](https://user-images.githubusercontent.com/33544078/146288415-f170a92a-4897-4957-981d-7eeb9878217e.jpg) 
- an example of synthetic occlusion(15 x 15 black image)
- 위 사진과 같이 black image로 occluded image를 생성하려 했으나, natural occluded image 복원을 위해서는 일반화 측면에서 안 좋을 것이라고 생각해 다른 dataset을 이용했다.

- Face De-occlusion dataset (https://github.com/xweiyuan/Face-de-occlusion-using-3D-morphable-model-and-generative-adversarial-network)
- ![image_train_0001](https://user-images.githubusercontent.com/33544078/146289299-5432fcbc-00b8-40e5-bc1a-0d6eed2a1102.jpg) ![image_train_0001_cup1](https://user-images.githubusercontent.com/33544078/146289314-92842e4c-6883-403b-8d84-cd7655f21587.jpg)
- 위 사진과 같이 한 사람의 얼굴에 대해 여러가지 종류의 synthetic occlusion으로 구성되어 있다. 
- training set은 400명의 Ground Truth(400명의 얼굴)에 대해 24633장의 occluded face로 이루어져 있고, testset은 222명의 Ground truth에 대해 13217장의 occluded face로 구성되어있다.

#### Non-paired dataset
- ROF-Real world Occluded Faces (https://github.com/ekremerakin/RealWorldOccludedFaces)
- ![780](https://user-images.githubusercontent.com/33544078/146290551-da8c0f58-0f7b-40aa-a645-42aa85bf6c1f.jpg) ![40](https://user-images.githubusercontent.com/33544078/146290574-8f968458-441f-4d1d-be98-613f5c219ef3.jpg)
- 위 사진과 같이 깨끗한 이미지와 occluded 이미지가 짝을 이루고 있진 않다. occlusion은 sunglasses와 mask 두 가지의 occlusion이 존재한다.
- Training set은 2768 장의 occluded face, 4698장의 non-occluded face로 이루어져 있고, testset은 692장의 occluded face, 1174장의 non-occluded face로 이루어져 있다.

### 모델 학습

#### OA-GAN 학습
- 가이드라인 논문에서 구조를 변경하지 않고 그대로 학습시켰다.
- 학습시킬 때 사용한 hyperparameter는 hyperparameter 폴더에 업로드 했다. 
- ![image](https://user-images.githubusercontent.com/33544078/146293094-b677524f-e207-4949-a40c-04a121fe95e3.png) ![image](https://user-images.githubusercontent.com/33544078/146293105-f27e5649-2303-46e0-b2e0-16d71dbd3c1f.png) ![image](https://user-images.githubusercontent.com/33544078/146293109-04891d09-81ab-4223-a8d9-b3b8dcc274e8.png)










