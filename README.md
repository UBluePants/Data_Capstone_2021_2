# Data_Capstone_2021_2

### 주제 선정 배경 
일반적인 얼굴 인식의 경우 얼굴 사진 데이터에 방해요소 없이 온전한 상태의 얼굴 형태가 많다. 하지만 실세계에서 우리가 얼굴을 인식할 때에는 마스크나 선글라스로 얼굴의 부분이 가려져 있거나, 혹은 사진 자체적으로 얼굴 전체를 담지 못하는 경우가 많다. 이런 부분을 감안한 얼굴 인식을 수행하기 위해 손상된 얼굴 사진으로부터 온전한 얼굴 사진을 얻을 필요가 있다.

### 과제 주요 내용
Face dataset 으로부터 occluded face dataset 을 생성 혹은 occluded face dataset 을 수집
dataset 구성
- Paired dataset(synthetic occlusion) : Face-de-occlusion dataset https://github.com/xweiyuan/Face-de-occlusion-using-3D-morphable-model-and-generative-adversarial-network
- non-paired dataset(natural occlusion) : Real world Occluded Faces https://github.com/ekremerakin/RealWorldOccludedFaces
- 
선정된 GAN을 이용한 Occluded face recovery
- 사용한 모델 : https://github.com/swordcheng/OA-GAN (source code for "Semi-Supervised Natural Face De-Occlusion") 
- 
PSNR, SSIM 을 통한 원본 이미지와의 비교

모델 구조 변경을 통한 성능 향상
