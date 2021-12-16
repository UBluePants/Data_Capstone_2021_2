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

### 모델 학습 및 

#### OA-GAN 학습
- 가이드라인 논문에서 구조를 변경하지 않고 그대로 학습시켰다.
- 학습시킬 때 사용한 hyperparameter는 hyperparameter 폴더에 업로드 했다. 

##### recovery of synthetic occluded face 
- ![image](https://user-images.githubusercontent.com/33544078/146293094-b677524f-e207-4949-a40c-04a121fe95e3.png) ![image](https://user-images.githubusercontent.com/33544078/146293105-f27e5649-2303-46e0-b2e0-16d71dbd3c1f.png) ![image](https://user-images.githubusercontent.com/33544078/146293109-04891d09-81ab-4223-a8d9-b3b8dcc274e8.png)
- 차례대로 synthetic occluded face, recovered face, original face이다. 
- synthetic data에 대해서는 좋은 복원 능력을 보여줬다.

##### recovery of natural occluded face 
- ![image](https://user-images.githubusercontent.com/33544078/146293918-bb3e8ccd-6227-4615-8359-f3e7e13cc054.png) ![image](https://user-images.githubusercontent.com/33544078/146293924-48d6fb6a-cf06-4abb-883f-94d36013c290.png)
- natural occluded face, recovered face 이다. 
- paired dataset에 비해 좋지 않은 복원 능력이 보여졌다. 

##### PSNR, SSIM
- PSNR : 26.35
- SSIM : 0.93
- 가이드라인 논문보다 더 좋은 점수가 나왔지만, 이는 복원하기 전 데이터셋에 대한 PSNR, SSIM 자체가 높을 것이라 생각했고 따라서 복원 전 PSNR, SSIM에 비해 얼마나 올랐는지를 다시 구했다.
- PSNR : 7.6 향상
- SSIM : 0.029 향상을 확인할 수 있었다.

#### DRAGAN algorithm 도입

##### DRAGAN (On Convergence and Stability of GANs, https://arxiv.org/abs/1705.07215)
- DRAGAN은 WGAN의 Lipschiptz 제약 조건을 해결하기 위한 방법 중 하나인 Gradient penalty regularization을 발전시킨 알고리즘이다.
- Gradient Penalty는 실제 데이터와 생성된 데이터 모두를 고려하여 gradient에 penalty를 주지만, DRAGAN algorithm은 실제 데이터만을 고려하기 때문에 이미지 복원 시 실제 얼굴과 더 가깝게 복원될 것이라 생각하여 적용해봐았다.
- 학습시킬 때 사용한 hyperparameter는 hyperparameter 폴더에 업로드 했다. 

##### recovery of synthetic occluded face
- ![image](https://user-images.githubusercontent.com/33544078/146297694-a15784ba-a786-4111-bdec-c92eca2964d3.png) ![image](https://user-images.githubusercontent.com/33544078/146297699-1f0861e5-fa76-4a91-9c03-9d163d4a417b.png) ![image](https://user-images.githubusercontent.com/33544078/146297713-2aaa5858-d7a7-4c54-966d-e06f5bafdca4.png)
- 차례대로 synthetic occluded face, recovered face, original face이다. 
- synthetic data에 대해서 시각적으로는 gradient penalty를 사용한 것과 비슷한 수준으로 복원되었다.

##### recovery of natural occluded face 
- ![image](https://user-images.githubusercontent.com/33544078/146298720-92809ac1-425c-44f0-bc29-29dcf3e74292.png) ![image](https://user-images.githubusercontent.com/33544078/146298731-11adf6b7-a8d7-4a74-9dc6-404ea82ec083.png)
- 마찬가지로 gradient penalty 를 사용한 것과 비슷한 수준으로 복원되었다.

##### PSNR, SSIM
- PSNR : 7.3 향상
- SSIM : 0.022 향상
- gradient penalty를 사용했을 때 보다 소폭 하락한 모습을 확인할 수 있었다.

#### DRAGAN algorithm, Instance normalization in discriminator
- OA-GAN에서는 discriminator에 instance normalization 을 사용하지 않았는데 discriminator에 instance normalization을 추가하면 효과적이지 않을까란 생각으로 적용했다.
- 학습시킬 때 사용한 hyperparameter는 hyperparameter 폴더에 업로드 했다. 

##### recovery of synthetic occluded face
-![image](https://user-images.githubusercontent.com/33544078/146299733-d6fee9cc-3e98-400d-b17f-c39bd91ed62b.png) ![image](https://user-images.githubusercontent.com/33544078/146299746-6043a29c-9044-4978-ace0-2bc07f88a39b.png) ![image](https://user-images.githubusercontent.com/33544078/146299755-f17a5b79-25d5-49fb-80b4-596df343e031.png)
- 차례대로 synthetic occluded face, recovered face, original face이다.
- 위의 두 방법에 비해 훨씬 안 좋은 복원 능력을 보여줬다.
- ![image](https://user-images.githubusercontent.com/33544078/146300091-d9bb8bf6-9b13-4294-a43b-fbbe15e2977c.png)
- 위 사진은 occlusion area를 시각화한 occluson mask 인데, occlusion area를 잘 잡아내지 못하는 모습을 보여줬다(occlusion 부분이 검은색). 

##### recovery of natural occluded face 
- ![image](https://user-images.githubusercontent.com/33544078/146300344-ffb3e7a7-b15c-4daf-8cca-c1b4f7dd9196.png) ![image](https://user-images.githubusercontent.com/33544078/146300354-1f1d2443-5725-4f1a-ba11-c805421d5f44.png)
- 마찬가지로 위의 두 방법에 비해 안 좋은 복원 능력을 보여줬다.

##### PSNR, SSIM
- PSNR : 0.54 향상
- SSIM : 0.26 감소
- gradient penalty, DRAGAN만을 사용했을 때보다 PSNR의 향상 정도는 급격히 감소했고, SSIM은 오히려 감소하는 모습을 보여줬다.

#### Spectral normalization (Spectral Normalization for Generative Adversarial Networks, https://arxiv.org/abs/1802.05957)
- Lipschitz 제약을 해결하기 위한 방법 중 하나인 Spectral normalization을 discriminator에 적용해봤다.
- Spectral normalization 은 discriminator의 weight에 weight의 Singular value 중 가장 큰 값(spectral norm)을 나눠줌으로써 네트워크의 lipschitz norm을 1보다 작게 만들어주는 normalization 기법이다.
- 학습 중간에 생성되는 이미지들을 확인해본 결과, occlusion area를 잘 찾아내지 못하는 모습을 보여줘서 끝까지 학습시키진 않았다.

#### Spectral Nomralization with Gradient penalty
- Spectral nomralization에 gradient penalty를 추가하면 어떨까 싶어서 적용해봤다.
- 학습시킬 때 사용한 hyperparameter는 hyperparameter 폴더에 업로드 했다. 

##### recovery of synthetic occluded face
- ![image](https://user-images.githubusercontent.com/33544078/146302458-e4767250-521e-4afa-bdd5-745a390e050b.png) ![image](https://user-images.githubusercontent.com/33544078/146302463-a66c2bfd-f217-42da-bf8a-f8e70f9666c6.png) ![image](https://user-images.githubusercontent.com/33544078/146302467-8409aa1c-8ea3-4421-9dd0-8d77dde18768.png)
- 시각적으로는 상당히 잘 복원된 모습을 보여줬다.

##### recovery of natural occluded face 
- ![image](https://user-images.githubusercontent.com/33544078/146302554-923da7b6-5f67-49d0-9edf-1c2dad9b9267.png) ![image](https://user-images.githubusercontent.com/33544078/146302561-e6825315-f0fe-4da4-80bc-a4ae76511af8.png)
- 마찬가지로 완벽하진 않지만 잘 복원된 모습을 보여줬다.

##### PSNR, SSIM
- PSNR :  7.22 향상
- SSIM :  0.027 향상
- 수치적으로는 어느 정도 향상되었지만, 수정하지 않은 OA-GAN에 비해서는 약간 낮은 모습을 확인할 수 있었다.

#### Resnet Feature extractor, label smoothing 
Feature extractor로 사용된 VGG16 대신, Resnet34의 layer들로 feature extractor를 대신하였고, lipschitz constraint 해결 방법으로는 Spectral normalization, Gradient penalty를 동시에 사용하였다. 또한 paired dataset의 label을 one-hot encoding 이 아닌 0.9로 smoothing을 해주었다.

##### recovery of synthetic occluded face
- ![image](https://user-images.githubusercontent.com/33544078/146391909-b079eef8-3743-495f-9c05-27fff7e509ad.png) ![image](https://user-images.githubusercontent.com/33544078/146391937-c9090102-4bfc-4091-ab53-8e10fcc3277c.png) ![image](https://user-images.githubusercontent.com/33544078/146391948-46586342-113e-4b9d-b08d-6d7a0f37fdc2.png)
- occlusion area의 잔상이 남아있지만 어느 정도 잘 복원된 모습을 보여줬다.

##### recovery of natural occluded face 
- ![image](https://user-images.githubusercontent.com/33544078/146392193-cccfc2ff-917e-49bd-a9b9-84b534734d8b.png) ![image](https://user-images.githubusercontent.com/33544078/146392215-caaba625-a976-4408-afd9-36d8774c312f.png)
- 눈 코 입이 생성되려는 모습은 보이지만, 매우 흐리고 얼굴 형태만 알아볼 수 있어 복원이 잘 되었다고 보긴 어렵다.

##### PSNR, SSIM
- PSNR :  0.69 향상
- SSIM :  0.022 향상
- OA-GAN(원본)은 물론, DRAGAN algorithm과 Spectral normalization + Gradient penalty를 사용한 모델에 비해 적은 수치를 보였다.

### 결론 및 아쉬운 점 
- OA-GAN에 여러 기법을 적용해봤지만 원본 모델보다 좋은 성능이 나오는 기법을 찾지 못했다.
- 더 많은 데이터셋으로 학습시켜 natural occlusion에 대해서도 좋은 복원 능력을 보여줬으면 좋았을 것 이라고 생각한다.
- GAN은 데이터의 분포를 학습하는 신경망이고 이를 학습하는 과정에서 목적 함수로 사용되는 데이터 분포간의 거리를 어떤 거리(distance)를 사용하고 해당 거리에 대한 깊은 이해가 필요하다는 것을 깨달았다. 적용한 기법들이 왜 원 모델보다 성능이 더 잘 나오지 않았는가에 대해서 추가적인 공부를 진행할 예정이다.




