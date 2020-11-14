# 텐서플로2를  활용한 딥러닝 컴퓨터 비전

- 예제 코드, 개념 설명 주피터 노트북 링크
  - 책에 있는 건 잘못 표기되어 있음.
  - https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2



### 2장 텐서플로 기초와 모델훈련

- 텐서플로 주요 아키텍쳐
  - c++ -> 저수준 api (python wrapper)-> 고수준 api(keras, estimator api)순서로 이뤄짐.(아래에서 위로 탑 모양)
  - 딥러닝 계산은 c++로 코딩되어 있음. 근데 gpu에서 이 계산을 하기 위해서 텐서플로는 NVIDIA에서 개발한 CUDA 라이브러리를 사용함. 따라서 gpu를 사용하기 위해 CUDA를 설치해야하고 다른 제조사의 gpu는 사용할 수 없음. 
- 딥러닝에서 모델은 일반적으로 데이터에서 훈련된 신경망을 말함. 모델은 매개변수와 함께 아키텍쳐, 가중치로 구성됌.
- keras 
  - 2017년 이후로 텐서플로가 케라스를 완전히 통합함. 그래서 텐서플로만 설치해도 케라스 함께 사용할 수 있게 됌. 
  - 이 책에서 `독립형 케라스 `대신에 `tf.keras`를 사용함. (모듈 호환성, 모델 저장방식 등 차이가 있음.)
    - 코드에서 keras가 아니라 tf.keras를 import 해야함.
    - keras.io 문서가 아니라 텐서플로 웹사이트에서 tf.keras 문서를 참고해야 함. 
    - 일부 저장된 모델은 케라스 버전 간에 호환 안될 수 있음.



#### keras를 사용한 간단한 컴퓨터 비전 모델

##### 데이터 전처리

```python
import tensorflow as tf

num_classes = 10
img_rows, img_cols = 28, 28
num_channels = 1
input_shape = (img_rows, img_cols, num_channels)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
```

- mnist 데이터 활용함, 크기: 28 x 28, class(lable): 10개 
- 일반적으로 입력데이터를 x, 레이블을 y로 표기함.
- 아래처럼, class, rows, cols, channels 를 변수처럼 쓸 것. 
- 데이터를 import 하고 나면 꼭 `정규화` 해줘야하는 데 보통 [0,1]로 변환하거나 [-1,1]로 변환함.
  - 여기서는 배열을 255.0으로 나눠 [0, 255.0] 범위를 [0, 1]로 변환해줌. 
- num_channels은 color channels 을 의미한다.
  - 0: input data의 color channel을 그대로 따름.
  - 1: grayscale
  - 3: RGB
- tensorflow 에서 image 는  4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]` 인데 Conv2D를 사용할때는 무조건  4-D Tensor of shape인듯.
  - batch가 batch size인데 이미지 전처리할때 미리 해주면 batch 빼고 3개만 적어도 되는 듯.

##### 모델 구성

```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
```

- 두개의 `완전 연결(== 밀집) `계층(layer)으로 구성된 아키텍처

- 이 모델은 layer을 `선형으로 쌓아` 구성하므로 Sequential 함수를 이용. (`순차형 API == 순차형 모델` 이라 부름.)

- `Flatten` layer(평면화 계층): 이미지 픽셀을 표현하는 2차원 행렬을 취해서 1차원 배열로 전환함.  즉, 28 x 28 이미지는 784 크기의 벡터로 변환됌. `이 계층은 dense layer을 추가하기전에 이뤄져야함.`

- 크기가 128인 Dense layer(밀집 계층): 여기서는 784 픽셀 값을 128 x 784 크기의 가중치(weight) 행렬과 128 크기의 편향치(bias) 행렬을 사용해 128개의 활성화 값으로 전환한다. 다 합치면 100,480 개의 매개변수.

-  크기가 10인 Dense layer:  이 층에서는 128개의 활성화 값을 최종 예측 값으로 전환함. `softmax` 함수는 전체 더하면 1이 되는 확률을 반환함. `분류 모델 마지막 layer `에서 사용되는 활성화 함수.

  - 맨 앞에 인자는 분류할 `labels` 즉, `class의 개수`를 받음.

- `model.summary()`를 사용하면 모델의 설명, 출력, 가중치를 확인 할 수 있음.

  - 전체 params가 trainable params 이고,  여기서는 잘려서 안보이는데 non-trainable params가  0으로 되어 있는데 이게 transfer learning에서 freeze 했을때 나오는 내용인듯.

  <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow study\computer vision with tensorflow2.assets\책1.jpg" alt="책1" style="zoom:80%;" />



##### 모델 훈련

```python
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_test,y_test)
```

- `.comple()`을 실행할 때 `optimizer`, `loss`, `metrics`를 꼭 지정해줘야함.
  - optimizer(최적화): 신경망의 가중치를 갱신하는 알고리즘. 경사 하강법 알고리즘에 기반해 동작함.
  - loss(손실): 예측이 얼마나 틀렸는지, 정확도를 평가하는 함수.
  - metrics(메트릭): 모니터링 할  메트릭을 선택(모델 성능을 더 시각적으로 보여주기 위해 평가되는 추가적은 거리 함수)
  
- 여기서 `optimizer ='sgd' `는 `tf.keras.optimizers.SGD() `와 같음. 후자처럼 쓰면 읽기는 편하지만 학습률 같은 매개변수를 직접 지정할 수 없음. `전자는 직접 지정가능`

- `sparse_categorical_crossentropy` 는 `categorical_crossentropy` 와 동일한 교차 엔트로피의 연산을 수행함. 전자는 실제 레이블을 입력(`integer type`)으로 직접 받지만 후자는 그전에 실제 레이블을 `one-hot `레이블로 인코딩 되어 받음.  

  - 두개를 비교하면 y_true가 다름. sparse categoical 은 y_true 가 일반 정수인 반면,  categorical은 one-shot vector 임. (label을 그대로 받냐 아니면 label를 인덱스로 하고 그 인덱스에 해당되면 1로 바꿔주느냐 이 차이인듯.)
- 모두 `categorical문제`를 풀때 `loss 설정`에 사용.
  
  - 둘이 계산 수식에 차이가 없기 때문에 정확도에 영향을 미치지는 않음.
- sparse_categorical_crossentropy 사용하려면 `flow_from_directory` 이용해서 data를 labling 할때 `class_mode='sparse'` 로 해줘야함.
  
- 참고
  
  - https://ahnjg.tistory.com/88, https://crazyj.tistory.com/153
  
  ```python
    # sparse_categorical_crossentropy
  # y는 여기서 label
    y_true = [1, 2]
    y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    
    # categorical_crossentropy
    y_true = [[0, 1, 0], [0, 0, 1]]
    y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  ```
  
- `.fit()`을 통해서 훈련시킴.

  - epoch: 5이면 전체 데이터 셋을 5번 반복해서 학습한다는 의미.
  
  - validation set은 저렇게 묶어서 넣어줘야하는 듯.
  
  - verbose: 학습 진행 상황을 보여줄 것인지 정함. 0이면 안보여주고, 1이면 보여줌.
    - 보여주는 건 `loss`, `metrics에서 정해준 것`(여기서는 accuracy), ETA(Estimated Time of Arrival)
    - ETA는 1 epoch가 끝날 때까지 남은 시간
    
    



#### Tensorflow2 와 keras 자세히 알아보기

##### 순차형 API(Sequential model), 함수형 API(Functional API)

- 순차형 API

  - 순차형 모델이라고도 함. 영어로는 api보다는 model로 부르는 게 맞는 듯. 
  - 위에서 모델 작성한 코드는 순차형 모델임. 그 결과 얻은 model은 여러가지 메서드와 속성을 포함.
    - `.inputs`, `outputs`: 이를 통해서 모델의 입력과 출력에 접근함.
    - `.layers`: 모델 layer와 형상 목록.
    - `.summary()`: 모델 아키텍쳐를 출력.
    - `.save()`: 훈련에서 모델, 아키텍처의 현 상태를 저장함. 나중에 저장된 모델은 `tf.keras.models.load_model()` 을 사용해서 인스턴스화 될 수 있음.
    - `.save_weights()`: 모델의 가중치만 저장.
  - 참고 링크
    - https://www.tensorflow.org/guide/keras/sequential_model

- 함수형 API

  ```python
  # 함수형 API 코드 예시, 위에서 순차형 모델로 짠 것과 같은 모델.
  
  model_input = tf.keras.layers.Input(shape=input_shape)
  output = tf.keras.layers.Flatten()(model_input)
  output = tf.keras.layers.Dense(128, activation='relu')(output)
  output = tf.keras.layers.Dense(num_classes, activation='softmax')(output)
  model = tf.keras.Model(model_input, output)
  ```

  - 코드는 좀 길어졌으나 함수형 API가 순차형 보다 훨씬 더 범용적으로 사용되며 다양한 정보를 표현할 수 있음.
  - 함수형 API는 모델을 분기할 수 있음. 예를 들어 여러 병렬 계층으로 아키텍처를 구성할 수 있음.
  - 하지만 순차형 API는 선형 모델에서만 사용될 수 있음.

  - 참고 링크
    - https://www.tensorflow.org/guide/keras/functional

- Model 객체는 어떤 방식으로 구성되든 간에 layer로 구성됌. 

  - `.get_weights()`: 해당 layer의 가중치를 조회할 수 있음. 
  - `.set_weights()`: 해당 layer의 가중치를 설정할 수 있음.
  - 좀더 복잡한 모델을 위해 `tf.keras.layers.Layer`를 통해 서브클래스도 만들수 있음. 
  - 참고 링크
    - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer



##### 콜백

- 케라스 모델의 `.fit()` 메서드에 전달할 수 있는 유틸리티 함수.
- 각 배치 반복, 각 epoch, 전체 훈련 절차 전후로 사용할 콜백을 여러 개 정의 할 수 있으며 미리 정의된 콜백의 종류는 아래와 같음.
  - `CSVLogger`: 훈련 정보를 CSV 파일에 로그로 남긴다.
  - `EarlyStopping`: 손실 혹은 메트릭이 더이상 개선되지 않으면 훈련을 중지. `over-fitting을 피할때 유용함.`
  - `LearningRateSchduler`: 스케줄에 따라 epoch마다 학습률을 변경함.
  - `ReduceLROnPlateau`: 손실이나 메트릭이 더이상 개선되지 않으면 학습률을 자동으로 감소시킴.
- `tf.keras.callbacks.Callback`의 서브클래스를 생성하여 맞춤형 콜백도 생성 가능.





##### 분산 전략(Distributed training)

- 큰 모델과 많은 데이터를 이용해 학습할 경우 많은 컴퓨팅 파워가 필요함.

- `tf.distribute.Strategy` API는 모델을 효과적으로 훈련시키기 위해 여러 컴퓨터가 통신하는 방법을 정의.

- Tensorflow에서 정의한 전략 종류

  - MirroredStrategy: 한 서버 내의 여러 GPU에서 훈련시키는 경우, 모델 가중치는 각 기기 사이에 싱크를 유지한다.
  - MultiWorkerMirroredStrategy: 여러 서버에서 훈련시킨다는 점을 제외하면 MirroredStategy와 유사함.
  - ParameterServerStrategy: 여러 서버에서 훈련시킬 때 사용함. 각 기기에 가중치를 동기화하는 대신 매개변수 서버에 가중치를 저장한다.
  - TPUStrategy: 구글 텐서 처리 장치칩에서 훈련시킬때 사용. TPU는 구글에서 신경망 연산을 위해 특별히 설계 제작한 맞춤형 칩으로 구글 클라우드를 통해 사용할 수 있음.

- 예시 코드

  ```python
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
      model= make_model() # 모델을 여기에서 생성
      model.compile([...]) # ... 여러 속성을 지정.
  ```

  - 분산 학습을 시키려면 모델을 그 분산 전략의 범위에서 생성하고 컴파일 해줘야함. 위의 코드가 그 과정.
  - 각 기기가 각 배치의 작은 하위 집합을 받기때문에 배치 크기를 키워야 할 수도 있음. 
  - 또는 모델에 따라 학습률을 변경해야 할 수 있음. 

- 참고 링크

  - https://www.tensorflow.org/guide/distributed_training
  - https://www.tensorflow.org/tutorials/distribute/keras



##### 텐서보드(Tensorboard)

- 강력한 모니터링 도구로 우리가 보고자 하는 정보를 실시간으로 시각화에 보여줌.

- 텐서플로를 설치하면 기본으로 설치됌. keras의 콜백과 결합해 사용하기에 좋음.

  ```python
  callbacks = [tf.keras.callbacks.TensorBoard('./logs_keras')]
  model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_test,y_test), callbacks=callbacks)
  ```

  - 텐서보드 콜백을 `model.fit()`에 전달하여 사용. 텐서플로는 기본적으로 지정된 폴더에 손실과 메트릭을 자동으로 기록함.

- `$ tensorboard --logdir ./logs_keras`:  텐서보드 인터페이스를 표시하기 위한 URL을 출력함. `Scalars` 탭에서 손실과 정확도를 표시하는 그래프를 확인할 수 있음.

- 딥러닝 모델은 여러 차례의 미세 조정이 필요함. 따라서 모델의 성능을 모니터링하는 것이 매우 중요함. 따라서 시간에 따라 모델 손실이 어떻게 개선되고 있는 모니터링 하기 위해 텐서보드를 주로 사용. 

- 주로 하는 작업 

  - 메트릭(예: 정확도) 그래프 그리기
  - 입출력 이미지 표시
  - 실행 시간 표시
  - 모델의 그래프 표현 그리기.

- 각 정보는 tf.summary에 저장되며 그 정보는 스칼라일수도 있고 이미지, 히스토그램, 텍스트 일 수도 있음.

  ```python
  writer = tf.summary.create_file_writer('./model_logs')
  with writer.as_default():
      tf.summary.scalar('custom_log', 10, stpe=3)
  ```

  - 스칼라를 기록하려면 먼저 위의 코드처럼 요약 작성자(summary writer)를 생성하고 정보를 기록해야함.

    - 이때 step을 지정해주는 데 epochs 또는 배치 번호 일 수 있으며 그래프의 x축에 해당.

  - 직접 정확도를 기록할때

    ```python
    accuracy = tf.keras.metrics.Accuracy()
    ground_truth, predictions = [1,0,1], [1, 0, 0] # 이게 모델에서 온다는데 무슨말이지?
    accuracy.update_state(ground_truth, predictions)
    tf.summary.scalar('accuracy', accuracy.result(), step=4)
    ```

- 텐서보드에서 `메트릭 로그`를 남기도록 설정하는 것은 다소 `복잡`할수 있지만 `텐서플로 툴킷의 필수 도구`임.



### 3장 현대 신경망

#### CNN(합성곱 신경망)

##### 컴퓨터 비전 작업에서 CNN이 언제나 사용되는 이유

- CNN은 초기 신경망(완전 연결 네트워크)의 단점을 해결하기 위해 도입됌.
  - 단점1: 매개변수의 폭발적인 증가
  - 단점2: 공간 추론의 부족

- 매개변수의 폭발적인 증가

  - 이미지 H(높이) x W(폭) x D(깊이 혹은 채널 개수) 개의 엄청난 숫자로 구성된 복합적인 구조.
    - RGB 이미지의 경우 D=3
  - 2장에서 예제로 사용한 단일 채널 이미지는 28 * 28 * 1 = 784개의 값으로 이뤄진 벡터임.
  - 앞에서 만든 기초 신경망의 첫번째층의 경우 (764,64)형상의 가중치 행렬을 가짐. 이는 최적화해야할 매개변수의 값이 784*64=50,176 이라는 의미.

  - 따라서 `이미지가 커지거나 네트워크가 깊어질수록 매개변수의 개수는 매우 급격히 증가함.`

- 공간추론의 부족(이 부분 정확하게 이해안됌.)
  - 뉴런이 어떤 구분없이 이전 계층의 모든 값을 받기 때문에(== 뉴런이 모두 연결되어 있다.)이 신경망은 '거리/공간성'이 없다. 
  - 즉, 이미지 같은 다차원 데이터는 Dense층에 전달되기 전에 Flatten층을 통해 1차원으로 변환되기 때문에 그 연산은 데이터 차원이나 입력값의 위치를 고려하지 않음. 따라서 모든 픽셀값이 계층(layer)별로 원래 위치와 상관없이 결합되므로 픽셀 사이의 `근접성 개념`이 `완전연결(Fully-connected == dense) 계층`에서 `손실됌`.
    - Flatten 층을 통과하면서 다차원 데이터가 `1차원으로 변환되는 것`과 같은 과정을 `칼럼 벡터`로 형상을 바꾼다고도 말함.
  - 따라서 입력값이 동일한 채널 값 또는 동일한 이미지 영역(이웃 픽셀)에 속하는 `공간 정보를 고려할 수 있다면 신경망 계층이 훨씬 똑똑해질 것.`



##### CNN이 기존 네트워크 단점을 해결하는 과정

- CNN은 2장에서의 네트워크와 같은 방식(전방전달, 역전파 등)으로 작동하지만, 아키텍쳐를 약간 변경함.

  <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow study\computer vision with tensorflow2.assets\책2.jpg" alt="책2" style="zoom: 50%;" />

- 뉴런이 이전 계층의 모든 요소에 연결된 완전 연결 네트워크와 달리 CNN의 각 뉴런은 이전 계층에서 `이웃한 영역에 속한 일부 요소`에만 접근함.  이 영역을 뉴런의 `수용 영역` 또는 `필터 크기` 라고 함.(일반적으로 `정사각형`이며 `모든 채널에 걸쳐있음.` 모든 채널에 걸쳐있다는 말이 여러 계층을 거쳐도 같은 깊이를 가진다는 말 같음. 사진 보면 그렇게 되어 있는듯.)
- 뉴런을 이전 계층의 `이웃한 뉴런과만 연결함`으로써 CNN은 `훈련시킬 매개변수 개수를 급격히 줄이고`, `이미지 특징의 위치 정보를 보존함.`
  - 완전 연결 계층에서 매개변수는 데이터 차원에 영향을 받는 반면, 합성곱 계층에서는 `데이터 차원이 매개변수 개수에 영향을 주지 않음`. 
    - CNN에서는 입력 크기가 다양하더라도 별도의 조정이나 재 훈련 과정을 거칠 필요가 없음.
      - 이 부분 잘 이해가 안된다. 이미지 resize하거나 target size 정해주지 않나?
  - CNN에서는 입력 이미지 크기가 커져도 튜닝해야할 매개변수 개수에 영향을 주지않고 네트워크 훈련 가능. 



#### CNN 작업(기본 구조)

#### 1. 합성곱 계층(convolution layer)

##### 속성

- 합성곱 계층은 `N개`의 뉴런의 집합을 가짐. (즉, 같은 매개변수를 공유하는 `N개`의 뉴런 집합.)
- 즉, 형상이 `D x k x k` (필터가 정사각형인 경우)인 `N개`의 가중치 행렬과 `N개`의 편향값으로 정의됌.
  - 여기서 `가중치 행렬`  == `필터` == `커널` 로 같은 걸 의미.
  - `필터`는 주로 `정사각형`을 사용하지만 필터의 높이와 너비가 다를 수 있음.
- CNN을 다양한 크기의 이미지에 적용하는 경우, 입력 배치를 샘플링할때 주의해야함.
  - 이미지 하위 집합은 모두 `동일한 차원`을 가질 때만 함께 쌓여서 일반 배치 텐서가 될 수 있음.
  - 따라서 `이미지 전처리`를 하여 이미지가 모두 `동일한 크기(차원)`을 갖게 한다. 
    - 이를 위해 이미지 크기를 조정하거나 자른다.
      - `resize`를 해서 이미지 크기를 맞추거나, `targetsize`를 정해줘서 신경망이 보는 size를 지정하거나 하는 듯함.



##### 초매개변수(Hyper-parameters)

- 합성곱 계층은 필터 개수 N, 입력 깊이 D(즉, 입력 채널의 개수), 필터(커널)의 크기(k)로 정의됌.

  - 필터는 주로 정사각형이므로 k로 표기.

- 입력과 필터사이의 연산은 몇 가지 추가적인 `초매개변수`를 취해 필터가 이미지 위에서 `움직이는 방식`을 결정.

  초매개변수는 아래와 같음.

  - 보폭(Stride)
    - 필터가 움직이는 보폭을 다양한게 적용할 수 있음. 
    - 필터가 움직일때 이미지와 필터 상의 내적을 위치마다 계산할지(`stride=1`), s 위치마다 계산할지(`stride=s`) 정의함. 
    - 보폭이 커지면 결과 특징 맵은 희소해짐. (희소 행렬처럼 된다는 의미인듯.)
  - 패딩(Padding)
    - 이미지가 합성곱을 적용하기 전에 패딩을 적용해 원본 컨텐츠 주변에 `0으로 된 행과 열`을 추가해 인위적으로 키울 수 있음.
    - 이를 통해서 필터가 이미지를 차지할 수 있는 `위치의 수를 증가`시킴. 예를 들어서 3x3인 이미지에 3x3 필터를 적용하면 1번만 움직일 수 있음. 근데 패딩을 모든 방향으로  1줄씩만 적용하면 5x5로 커지고 3x3 필터는 최소 1번이상 움직일 수 있음. 
    - 입력 이미지 주변에 추가할 빈 행과 열의 개수를 지정할 수 있음. 주로 같은 행과 열의 개수를 지정.
    - tensorflow에서는 두가지 모드만을 옵션으로 줘서 사용자가 어떤 p값을 사용할지 자동으로 설정해줌.
      - `VALID`: 이미지에 패딩을 더하지 않음.(p=0)
      - `SAME`: 합성곱 출력이 보폭이 1인 입력과 `동일한` 높이와 너비를 갖게 p를 계산해 패딩을 적용함.

- `필터(커널)의 개수 N` , `필터(커널)의 크기 k`, `보폭 s`, `패딩 p`로 표기.

- tensorflow에서 합성곱 예시 코드

  ```python
  conv = tf.keras.layers.Conv2D(filters=N, kernel_size=(k, k), strides=s, padding='valid', activation='relu')
  ```

  - 참고링크: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D




#### 2. 풀링 계층(pooling layer)

##### 개념 및 초매개변수

- 각 뉴런이 자기 윈도우(수용 영역)의 값을 사전에 정의된 함수로 계산한 하나의 출력을 반환함.
  
- 이를 통해서 `데이터 공간 차원을 줄여서` 네트워크에서 필요한 `매개변수의 전체 개수를 줄이고 계산 시간을 단축함.`
  
- `최대 풀링(max pooling)`과 `평균 풀링(average pooling)`이 있음. 보편적으로 이 두개가 자주 사용됌.

  - 최대 풀링: 풀링된 영역의 깊이 마다 최댓값만 반환.
  - 평균 풀링: 풀링된 영역의 깊이 마다 평균을 계산해서 반환.

- tensorflow에서 pooling 예시 코드

  ```python
  tf.keras.layers.MaxPool2D(
      pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs
  )
  
  tf.keras.layers.AveragePooling2D(
      pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs
  )
  ```

  

#### 3. 완전 연결 계층(Fully-connected layer)

- CNN에서도 일반 네트워크와 같은 방식으로 FC 계층이 사용됌.

- FC 계층은 밀집(dense) 연결된 계층, 단순히 밀집 계층이라고도 함.

- 다른 CNN계층과는 다르게 밀집 계층은 입력과 출력 크기에 의해 정의됌. 특정 밀집 계층은 그 계층의 설정과 다른 형상을 갖는 입력에는 동작하지 않음. 따라서 밀집 계층만을 사용하면 다양한 크기의 이미지에 적용할 수 없음.

- 하지만 CNN에서 보편적으로 사용됌. 주로 네트워크의 마지막 계층으로 사용되는 데 다차원 특징을 1차원 분류 벡터로 변환하기 위해 사용.

- 다만 `다차원 텐서`를 `dense` 층에 입력하기 앞서 `평면화(flattening)`을 거쳐야함. 이를 위해 tensorflow에서는 `Flatten()` 를 사용하여 높이, 너비, 깊이 차원을 단차원으로 평면화함. (칼럼 벡터로 만듬.)

  - `단차원으로 평면화` == `칼럼 벡터로 만듬` == `높이 * 너비 * 깊이`로 모두 같은 의미.

- tensorflow에서 pooling 예시 코드

  ```python
  tf.keras.layers.Dense(units, acitvation='relu')
  # 여기서 units은 output size(출력 개수)를 의미함.
  ```

##### 참고

- 수용 영역(Receptive Field)
  - 한 계층의 필터 크기나 윈도우 크기 == 뉴런이 연결된 이전 계층의 로컬 영역.



#### Tensorflow로 CNN 구현하기.

- LeNet-5 구현함. 7계층으로 이뤄짐.
- MNIST 데이터셋 활용.
- 참고 링크
  - https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2/blob/master/Chapter03/ch3_nb2_build_and_train_first_cnn_with_tf2.ipynb
  - https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2/blob/master/Chapter03/ch3_nb3_experiment_with_optimizers.ipynb

##### LeNet-5 아키텍처

![책3](C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow study\computer vision with tensorflow2.assets\책3.jpg)

- LeNet-5은 두개의 블록(conv#1, conv#2)으로 구성되어 있음. 

  - 각 블록은 합성곱 계층(커널 크기 k=5, 보폭 s=1)과 최대 풀링 계층(k=2, s=2)을 포함.
  - 첫번째 블록은 합성곱 계층에 전달되기 전에 2만큼 0으로 패딩을 하고(p=2) 합성곱 계층에는 6개의 필터가 있음(N=6).
    - 패딩을 2만큼 줬기때문에 실제 입력 크기는 32x32임.

- 두 블록 다음에 세 개의 완전 연결 계층은 특징을 함께 합쳐 최종 클래스 추정값(10개 숫자 클래스)을 도출함.

  - 첫번째 밀집 계층 전에 5x5x16개의 특징 볼륨이 400개의 벡터로 평면화됌.(위에서 dense층에 넣기 전에 flatten 써서 칼럼 벡터로 1차원화 해야한다고 했음.)

  - 마지막 계층을 제외하면 각 합성곱 계층과 밀집 계층은 활성화 함수로 ReLU를 사용함.

  - 마지막 계층은 softmax 함수를 사용함.

    - 분류 작업을 하는 신경망에서 네트워크 예측을 클래스별 확률로 변환하기 위해 신경망의 끝에서 사용함.  각 값은 네트워크가 입력 데이터를 해당 클래스로 얼마나 확신하는지 나타냄.

    - 즉, softmax는 벡터를 정규화해 전체합이 1이 되도록 0과 1사이의 값으로 만든다. 이는 입력이 네트워크에 따라 클래스에 해당할 확률을 칼럼 벡터 형태로 반환함. ex) y=[y0,y1,y2...y9]



##### LeNet-5를 Tensorflow와 keras로 구현

- 구현 코드

```python
import tensorflow as tf
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
```

```python
## Preparing the Data

num_classes = 10
img_rows, img_cols, img_ch = 28, 28, 1
input_shape = (img_rows, img_cols, img_ch)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

# 이렇게 이미지 크기 조정하는 방법도 있음.
# size = (150, 150)
# train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
# validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
# test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

# print('Training data: {}'.format(x_train.shape))
# print('Testing data: {}'.format(x_test.shape))
```



- 모델 구현 방식에는 순차형과 함수형이 있음.

- 순차형 모델 방식

  ```python
  # Preparing the Mode 여기서는 순차형 모델 방식으로 만듬.
  
  def lenet(name='lenet'):
      model = Sequential(name=name)
      # 1st block:
      model.add(Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      # 2nd block:
      model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      # Dense layers:
      model.add(Flatten())
      model.add(Dense(120, activation='relu'))
      model.add(Dense(84, activation='relu'))
      model.add(Dense(num_classes, activation='softmax'))
      return model
  # creating the model
  model = lenet()
  ```
  - 모델은 계층 하나씩 인스턴스화 하고 `순차적으로` 추가해 생성됌.

- 함수형 API 방식

  ```python
  # Preparing the Mode 여기서는 함수형 API 방식으로 만듬.
  
  from tensorflow.keras import Model
  # 이 방식을 사용하기 위해서는 추가적으로 Model을 import 해줘야함.
  class LeNet5(Model):
      # Model은 Layer와 동일한 API를 가지고 있으며 이를 더 확장해줌.
      def __init__(self, num_classes):
          """
          Initialize the model.
          :param num_classes:     Number of classes to predict from
          """
          super(LeNet5, self).__init__()
          # We instantiate the various layers composing LeNet-5:
          # self.conv1 = SimpleConvolutionLayer(6, kernel_size=(5, 5))
          # self.conv2 = SimpleConvolutionLayer(16, kernel_size=(5, 5))
          # ... or using the existing and (recommended) Conv2D class:
          self.conv1 = Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu')
          self.conv2 = Conv2D(16, kernel_size=(5, 5), activation='relu')
          self.max_pool = MaxPooling2D(pool_size=(2, 2))
          self.flatten = Flatten()
          self.dense1 = Dense(120, activation='relu')
          self.dense2 = Dense(84, activation='relu')
          self.dense3 = Dense(num_classes, activation='softmax')
          
      def call(self, inputs):
          """
          Call the layers and perform their operations on the input tensors
          :param inputs:  Input tensor
          :return:        Output tensor
          """
          x = self.max_pool(self.conv1(inputs))        # 1st block
          x = self.max_pool(self.conv2(x))             # 2nd block
          x = self.flatten(x)
          x = self.dense3(self.dense2(self.dense1(x))) # dense layers
          return x
      
  # creating the model.
  model = LeNet5(num_classes)
  ```

  - 케라스 계층은 `함수`처럼 작동해 원하는 결과를 얻을 떄까지 연결될 수 있음. 
  - 네트워크 내부에서 특정 계층을 `여러 회 재사용하는 경우`나 `계층에 여러 입력 또는 여러 출력이 있을 경우`에 함수형 API를 사용해 더 `복합적인 신경망`을 구성할 수 있음. 

- 모델 컴파일

  ```python
  model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  ```

  - 모델 훈련을 시작하기 전 optimizer를 인스턴스화 하고, loss를 정의 해줌.

- callback 인스턴스화 및 모델 훈련

  ```python
  callbacks = [
      # 3 epochs가 지나도 'val_loss'가 개선되지 않으면 훈련을 중단함.
      tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
      #  graph, losses and metrics을 TensorBoard에 기록(log files을 `./logs`에 저장)
      tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)]
  
  # 모델 훈련
  model.fit(x_train, y_train, batch_size=32, epochs=80, 
            validation_data=(x_test, y_test), verbose=1, callbacks=callbacks)
  ```

  - Before launching the training, we also instantiate some Keras callbacks, i.e., utility functions automatically called at specific points during the training (before/after batch training, before/after a full epoch, etc.), in order to monitor it:

    - 즉, 훈련하는 동안 모니터링을 위해 몇몇 지점에서 자동으로 호출되는 `유틸리티 함수`를 `인스턴스화`하는 `케라스 콜백`을 `인스턴스화` 해줌.

    

#### 훈련 프로세스 개선

- 네트워크 아키텍쳐뿐만아니라 네트워크 훈련 방법도 개선되어 옴. 

- 아래에서 이를 위한 여러 optimizer를 소개함.

  ```python
  optimizers_examples = {
      'sgd': optimizers.SGD(),
      'momentum': optimizers.SGD(momentum=0.9),
      'nag': optimizers.SGD(momentum=0.9, nesterov=True),
      'adagrad': optimizers.Adagrad(),
      'adadelta': optimizers.Adadelta(),
      'rmsprop': optimizers.RMSprop(),
      'adam': optimizers.Adam()
  }
  ```

   #### 여기 부분은 1장 내용을 읽고와서 다시 봐야할 듯.(p.30~p.39, p.89~p.95)

- 어느 optimizer가 최선인지에 대해 합의된 바는 없지만 부족한 데이터에서의 효과성때문에 전문가들은 `adam`을 선호함.

- `RMSprop`는 대체로 순환신경망(RNN)에 사용하기 적절한 것으로 간주됌.

- LeNet-5에서 optimizer에 따른 training, validation 의 loss와 accuracy.

  <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow study\computer vision with tensorflow2.assets\image-20201025145708933.png" alt="image-20201025145708933" style="zoom:67%;" />

- 참고 링크

  - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam



#### 정규화 기법

- 신경망을 효율적으로 가르쳐서 손실을 최소화하는 것만으로 충분하지 않음. 훈련 데이터 셋에 over-fitting 되지 않아야함.
- 네트워크 일반화에 성공하려면 `풍부한 훈련 데이터 셋`(즉, 가능한 테스트 시나리오를 다루기에 충분한 다양성을 갖춘)과 `잘 정의된 아키텍처`(과소적합을 피하기에 너무 얕거나 과적합을 방지하기에 너무 복잡하지 않은)가 필요함.
- 이와 별개로 over-fitting을 피하기 위한 최적화 단계를 정교화하는 프로세스인 `정규화(Regularization)`을 위한 다른 기법도 개발 되어 옴.
- 참고 링크
  - https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2/blob/master/Chapter03/ch3_nb4_apply_regularization_methods_to_cnns.ipynb

##### 조기 중단(Early-stopping)

- 훈련 epochs는 네트워크가 과적합되기 시작하기 전에 종료시킬 수 있을 만틈 충분히 낮아야하지만, 그 훈련 데이터에서 모든 것을 배울 수 있을 만큼 높아야함.
- `epoch 마다`네트워크를 검증함으로써 훈련을 계속해야 할지, 중단해야할지(즉, `검증 정확도가 정체되거나 떨어지는 경우`)를 정할 수 잇는데 이를 `조기 중단`이라고 함.
- `tf.keras.callbacks.EarlyStopping` 를 통해 조기 중단 할 수있음.
  - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping?hl=en

##### L1, L2 정규화

- 손실 함수를 수정함을써 과적합을 방지함.

- L1, L2가 있음.
- `tf.keras.regularizers`를 통해서 사용할 수 있음.
  - https://www.tensorflow.org/api_docs/python/tf/keras/regularizers

##### 드롭아웃(Dropout)

- 위의 두가지 정규화 기법은 네크워크의 훈련 방식에 영향을 주는 방법인데 `드롭아웃`은 신경망 아키텍처에 영향을 주는 방법.

- 드롭아웃은 훈련이 반복될 때마다 `타깃 계층의 일부 뉴런의 연결`을 `임의로 끊음`. 초매개변수로 훈련 단계마다 뉴런이 꺼질 확률을 나타내는 `비율 Ρ`를 취함.(Ρ는 주로 `0.1`에서 `0.5` 사이의 값으로 설정.)

  <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow study\computer vision with tensorflow2.assets\책4.jpg" alt="책4" style="zoom:67%;" />

- 핵심 특징을 담당하는 뉴런을 비활성화할 수 있으므로 네트워크는 동일한 예측에 도달하기 위해 다른 중요한 특징을 알아내야 함. 이를 통해 데이터 분류를 위한 중복 표현을 개발하는 효과를 얻을 수 있음. 이런 방법을 통해 견고한 공동의 특징을 학습할 수 있게  함.

- 드롭아웃은 `테스트 단계에서는 네트워크에 적용되지 않음`. 따라서 네트워크의 예측은 부분적인 모델이 제공하는 결과의 조합으로 볼 수 있음. 즉, 이 정보를 평균하면 네트워크가 `과적합되는 것을 방지할 수 있음`.

- `tf.keras.layers.Dropout()` 은 훈련하는 동안에만 적용되고 그 외에는 비활성됌. (training=True으로 되어 있을때 이와 같은 방식으로 작동하는 데 model.fit을 사용할때 자동으로 True로 설정이 됌.)

  - 참고로 `trainable=False`은 적용되지 않음. Dropout에는 훈련 동안 frozen 될 variables이나 weights를 가지고 있지 않기 때문임.
  - 드롭아웃 계층은 과적합을 방지할 계층 `바로 뒤에 추가`돼야함.(드롭아웃 계층은 `앞 계층에서 반환한 값 중 일부를 임의로 누락시키고` 적응하게 만듬, 주로 dense 층 뒤에 사용되는 것 같음.)

##### 배치정규화(Batch normalization)

- 드롭아웃처럼 신경망에 삽입되어 훈련에 영향을 줄 수 있는 연산임.
- 이전 계층의 배치 결과를 취해 `정규화(normalize)`함. 즉, 배치 평균을 빼서 배치 표준편차로 나눔.

- 참고 링크
  - https://arxiv.org/abs/1502.03167
  - https://hcnoh.github.io/2018-11-27-batch-normalization
  - https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalizat

##### LeNet-5에서 정규화 방법들에 따른 training, validation의 loss와 accuracy

<img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow study\computer vision with tensorflow2.assets\image-20201025145920062.png" alt="image-20201025145920062" style="zoom:67%;" />



 

### 4장 분류를 위한 신경망 아키텍처

#### 전이학습(Transfer learning)

##### 전이학습의 필요성

- CNN은 특정 특징을 추출해 해석하도록 훈련됐기 때문에 특징 분포가 바뀌면 그 성능은 떨어지게 됌. 
  - 훈련된 모델을 다른 데이터셋에 바로 적용하면, 특히 두 데이터 샘플이 동일한 의미론적 컨텐츠(MNIST 숫자 이미지와 ImageNet 사진) 또는 동일한 이미지 품질/분포(스마트폰 사진 데이터셋과 해상도 높은 사진 데이터셋)를 공유하고 있지 않으면 형편없는 결과를 얻음.
- 따라서 네트워크를 새로운 작업에 적용하려면 어느 정도의 변환 작업이 필요하다.
- 머신러닝에서 `작업(Task)`는 주어진 입력(사진)과 예상 출력( 특정 클래스에 대한 예측 결과)에 의해 정의됌.
  - ImageNet에서 분류, 탐지하는 작업은 입력 이미지는 동일하지만 출력이 다르기 때문에 서로 다른 작업으로 봐야함. 
- 전이학습은 이미 갖춘 지식을 하나의 작업(Task)에서 다른 작업으로 또는 한 `도메인(데이터 분포)`에서 다른 도메인으로 적용하는 것을 목표로 함. 
  - 도메인에서 다른 도메인으로 적용하는 전이학습을 `도메인 적응(domain adaptation)` 이라고 함.
- 전이학습은 `새로운 작업을 학습하기에 충분한 데이터가 없을 경우` 꼭 필요함. (즉, 분포를 추정하기에 이미지 샘플이 충분하지 않을때)



##### CNN 지식 전이

- CNN을 위한 전이학습은 주로 다른 작업을 위한 새로운 모델을 인스턴스화하기 위해 풍부한 데이터셋에서 훈련된 성능 좋은 네트워크의 아키텍처 전체 혹은 일부와 가중치를 재사용하는 것으로 구성됌.
- 새로운 모델은 이 조건에 따라 인스턴스화 한 다음 `미세 조정(fine-tunning)` 될 수 있음.  이후 새로운 작업/도메인에 대해 활용할 수 잇는 데이터에서 더 훈련될 수 있음.

- 네트워크의 첫번째 계층은 저차원 `특징`(선, 테두리, 색 변화 등)을 `추출`하는 경향이 있지만 마지막 합섭공 계층은 더 복잡한 특징(특정 형태나 패턴)에 반응함. 이후 마지막 풀링 또는 완전 연결 계층에서 클래스를 예측하기 위해 이 고차원 특징 맵(==병목 특징,Bottleneck feature)을 처리함.
- 이런 전이학습에 대한 구성에서 다양한 전이학습 전략이 도출됌.
  - 작업/도메인이 유사한 경우
    - 마지막 예측 계층을 제거한 사전 훈련된 CNN은 효율적으로 `특징을 추출하는 용도(Feature Extraction)` 로 사용되기 시작함. 새로운 작업이 이 추출기가 훈련된 목적과 충분히 비슷한 경우 적절한 특징을 출력하기 위해 바로 사용될 수 있음.
    - 추출기에서 나온 특징은 작업(Task)과 관련된 예측을 출력하도록 훈련된 한두개의 `새로 추가된 밀집 계층`에서 처리될 수 있음. 추출된 특징의 품질을 유지하기 위해 대체로 훈련 단계동안 `특징 추출기의 계층`을 `고정(Freeze)`시킴. (즉, 고정된 계층의 매개변수는 업데이트되지 않음.)
  - 작업/도메인이 유사하지 않은 경우
    - 특징 추출기의 마지막 계층 중 일부 또는 전체를 `미세조정(Fine-tunning)`함. 즉, 이 계층들을 작업 데이터에서 새로운 예측 계층과 함께 훈련시킴.

##### CNN 전이학습 사례

- 사전 훈련된 모델 중 어떤 모델을 재사용해야할까?, 어느 계층을 고정시키고 어느 계층을 미세 조정해야 하나?
  - 목표한 작업과 사용하려는 모델이 훈련한 `작업이 얼마나 비슷한지`, 새로운 훈련 샘플이 `얼마나 풍부한지`에 따라 달라짐.

1. 제한적 훈련 데이터로 유사한 작업 수행
   - 전이학습은 특정 작업을 해결하고 싶고 성능 좋은 모델을 제대로 훈련시킬 만큼 데이터가 충분하지 않지만 유사한 훈련 데이터셋을 이용할 때 특히 유용함.
     - 사전 훈련된 모델을 가져온다. 
     - 목표한 작업이 다른 경우. 즉, 출력이 사전 훈련의 목표 작업의 출력과 다른 경우에 마지막 계층을 제거하고 목표한 작업에 맞춰 조정한 계층으로 교체암.
   - 예시 (벌 사진과 말벌 사진을 구분하는 모델을 훈련시키고 싶지만 사진 데이터가 충분치 않은 상황)
     - ImageNet 데이터셋에서 1000개의 카테고리로 분류하도록 이 네트워크를 훈련시킴. 
     - 이 사전 훈련 과정을 거친 다음 마지막 밀집 계층을 제거함.
     - 목표한 두 개의 클래스에 대해 예측을 출력하도록 상단에 설정된 계층으로 교체함.
     - 미리 훈련된 계층을 고정시킴. (데이터셋이 적어 특징 추출기의 구성 요소를 고정시키지 않으면 과적합이될 수 있음. 또한 네트워크가 풍부한 데이터넷에서 개발한 표현력을 유지할 수 있게함.)
     - 상단에 위치한 밀집 계층만 훈련시킴.
2.  풍부한 훈련 데이터로 유사한 작업 수행
   - 훈련 데이터가 풍부하면 네트워크가 과적합될 가능성이 낮음.
   - 일반적으로 특징 추출기의 최신 계층을 고정시켰던것을 해체함. 즉, 데이터셋이 크면 안전하게 미세 조정할 수 있는 계층이 많아짐.
   - 새로운 작업과 관련성이 높은 특징을 추출하고 새로운 작업을 수행하는 방법을 더 잘 학습할 수 있음.
   - 이때 유사한 데이터셋에서 첫번째 훈련 단계를 거쳤기때문에 이미 수렴에 가까워져 있을 것. 따라서 미세 조정하는 단계에서는 학습률을 낮게 설정함.
3. 풍부한 훈련 데이터로 유사하지 않은 작업 수행
   - 원래 작업과 목표 작업 사이의 유사성이 매우 낮은 경우 모델을 사전에 훈련 시키거나 사전 훈련된 가중치를 내려받는 일이 비용이 높을 수 있음.
   - 하지만 다양한 실험을 통해 대부분의 경우 무작위로 선정된 가중치보다 사전 훈련된 가중치(비슷하지 않은 용도로 훈련됐더라도)를 사용해 네트워크를 초기화하는 것이 더 낫다는 사실이 확인됌.
     - 하지만 작업이나 도메인이 어느정도 기초적인 유사성을 공유해야함. 
     - 예를 들어 이미지, 오디오 파일 모두 2차원 텐서로 저장될 수 있고 CNN은 공통적으로 두 파일 모두에 적용될 수 있음. 하지만 이 모델은 시각 인식과 오디오 인식을 위해 전혀 다른 특징에 의거해 훈련되므로 오디오 관련 작업을 위해 훈련된 네트워크의 가충치를 받는 것은 시각 인식을 위한 모델에 도움이 되지 않음.
4.  제한적 훈련 데이터로 유사하지 않은 작업 수행
   - 이럴 경우에는 데이터셋이 작아 사전 훈련시킬 모델이 과적합이 발생할 것임. 따라서 의도와는 다른, 매우 무관한 특징을 반환하게 됌. 
   - 하지만 CNN의 첫 계층이 저차원특징에 반응한다는 점을 감안하면 전이학습 혜택을 볼 수 있음. 
   - 사전 훈련된 모델의 최종 예측 계층만 제거하는 것이 아니라, 작업에 너무 특화된 최종 합성곱 블록의 일부도 제거할 수 있음. 그 이후 남은 계층 위에 얕은 분류기를 추가해 모델을 미세 조정할 수 있음.



##### 텐서플로와 케라스로 전이학습 구현

- 모델 선택

  - Tensorflow Hub나 keras application을 통해 제공되는 사전 훈련된  표준 모델을 가져와 새로운 작업을 위한 특징 추출기로 사용.

  - 또는 더 특수한 용도의 최신 CNN이나 일부 이전 작업을 위해 이미 훈련된 맞춤형 모델 같은 비표준 네트워크를 재활용하기도 함.

- 계층 제거

  1. 사전 훈련된 모델의 마지막 계층을 제거해 특징 추출기로 변환. 

     - 여기서 케라스를 이용함. 

     - Sequential 모델의 경우 model.layers 속성을 통해 계층 리스트에 접근할 수 있음.  그리고 마지막 계층을 제거하기 위해 pop() 메서드를 사용함. 

     - 네트워크를 특징 추출기로 변환하기 위해 제거해야 할 마지막 계층의 개수를 안다면 아래 코드를 통해 제거할 수 있음.(ResNet의 경우 두 개 계층)

       ```python 
       for i in range(num_layers_to_remove):
           model.layers.pop()
       ```

  2. 다른 방법으로 계층을 제거하는 대신 이전 모델에서 유지하고자 하는 마지막 계층/연산을 정확히 다른 것으로 특정하면 됌. 

     - 먼저 유지할 최종 계층의 이름을 알아내야함. 이를 위해 `model.summary()`로 이름 및 구조 출력.

     - 아래 코드를 통해 특징 추출기 모델을 구성.

       ```python
       bottleneck_feats = model.get_layer(last_name_name).output
       feature_extractor = Model(input=model.input, outputs=bottleneck_feats)
       ```

- 계층 이식

  - 케라스 api 이용해서 할 수 있음.

    ```python
    dense1 = Dense(...)(feature_extractor.output)
    new_model = Model(model.input, dense1)
    ```

- 선택적 훈련

  - 전이학습을 사용하려면 먼저 사전 훈련된 계층을 복원하고 어느 계층을 고정할지 정의해야함. 이를 위한 과정을 아래서 소개함.

    - 일부 사전 훈련된 매개변수 복원하기.

    - 전체 모델을 복원.

      ```python
      model = tf.keras.models.load_model('/path/to/pretrained/model.h5')
      ```

      - 일부 계층을 제거하기 위해 전체 모델을 복원하는 것이 최적은 아님. 하지만 간결함.

- 계층 고정하기.

  - 일부 계층 고정하기.

    - optimizer에 전달되는 매개변수 리스트에서 tf.Variable 특성을 제거함.

      ```python
      # 예를 들어 이름에 "conv"가 포함된 모델 계층을 고정하고자 함.
      
      vars_to_train = model.trainable_variables
      vars_to_train = [ v for vin vars_to_train if "conv" in v.name]
      
      # optimizer를 남은 모델 변수에 적용함
      optimizer.apply_gradients(zip(gradient, vars_to_train))
      ```

  - 전체 계층 고정하기.

    - 케라스의 계층에는 `.trainable`특성이 있고 그 계층을 고정하기 위해 이 속성을 False로 설정하면 됌.

      ```python
      for layer in feature_extractor_model.layers:
          layer.trainable = False
      ```

      









### 모델 저장 및 로드 (공식 문서 참고함.)

- 참고 링크:  https://www.tensorflow.org/guide/keras/save_and_serialize

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
```

#### 전체 모델 저장 및 로딩

- 전체 모델을 단일 아티팩트(artifact, which is an object, such as a tool that was made in the past) 로 저장할 수 있음.
  - 모델의 아키텍처(architecture) 및 구성(config)
  - 훈련 중에 학습된 모델의 가중치 값
  - 모델의 컴파일 정보(compile()이 호출된 경우)
  - 존재하는 옵티마이저와 그 상태(훈련을 중단한 곳에서 다시 시작할 수 있게 해줌.)
- APIs
  - `model.save()` 또는 `tf.keras.models.save_model()`
  - `tf.keras.models.load_model()`
- 저장 형식
  - 모델을 저장하는 데 사용할 수 있는 두 형식은 `TensorFlow SavedModel` 형식과 `이전 Keras H5 `형식임.

  - 하지만 SavedModel을 권장하며 model.save() 를 사용할 때의 기본값임.
    - H5 형식으로 전환하기 위한 방법. 
      - `save_format='h5'` 을 `save()`로 전달할 것.
      - `.h5` 또는 `.keras`로 끝나는 파일명을 `save()` 로 전달함.
    
  - SavedModel이 포함하는 것.

    - 저장 방법: `model.save('path/to/location')` 
    - 로드 방법: `model = tf.keras.models.load_model('path/to/location')`

    - `assets`, `saved_model.pb`,  `variables`,
    - 모델 아키텍처 및 훈련 구성(optimizer, loss, metrics 포함.)은 saved_model.pb에 저장됌.
    - 가중치는 variables/ 디렉토리에 저장됌.