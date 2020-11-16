# Tensorflow 강의

- 내가 들은 건 `Laurence Moroney` 강의

### Intro to Machine Learning (ML Zero to Hero - Part 1)

- 가위 바위 보 게임을 만들것임.

  - 사람이 인식하는 건 아주 쉬움.

  - 어떻게 프로그래밍해야 컴퓨터가 이를 인식할까?

    - 손 색상이 다르고 가위, 바위 보를 내는 방식이 모두 다름. (누구는 엄지와 검지를 이용해 가위를 새끼나와 약지를 이용해 가위를 냄.) 
    - 사람이 인식하는 것과 같은 방식으로 컴퓨터가 인식하면 어떨까가 머신러닝과 인공지능의 방향임.

    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922150949635.png" alt="image-20200922150949635" style="zoom:80%;" />



- 기존 프로그래밍 방식

  1. 웹캠에서 들어온 데이터와 이를 처리하는 규칙이 있음. (규칙은 프로그래밍 된 엄청난 양의 코드)

  2. 이러한 규칙이 데이터를 처리하고 우리에게 답을 줌. (가위,바위, 보를 인식)

     <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922151531042.png" alt="image-20200922151531042" style="zoom: 67%;" />

- 머신러닝

  1. 프로그래머인 우리가 규칙을 알아내는 것 대신에 데이터로 답을 주는 것. 

  2. 이를 통해서 컴퓨터가 알아서 규칙을 알아냄.

     <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922151633471.png" alt="image-20200922151633471" style="zoom:67%;" />

  3. 데이터와 라벨을 통해서 가위,바위,보가 뭔지 알려줌. 그리고 컴퓨터가 패턴을 알아내도록 해서 서로 일치하는 것을 찾도록 함. 그러면 컴퓨터에서 가위, 바위, 보를 인식하는 걸 배움.

     - 즉, 패턴을 담고 있는 일련의 데이터를 제공하고 컴퓨터가 그러한 패턴을 배우도록 함.

     <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922152134459.png" alt="image-20200922152134459" style="zoom:67%;" />

     - 더 간단한 예시

       - x와 y사이에는 관계가 있음.  y=2x-1 이라는 걸 숫자들을 통해서 알게됌. 그리고 이를 다른 숫자에도 적용해보니 예상이 맞다는 걸 알 수 있음.

       <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922152349653.png" alt="image-20200922152349653" style="zoom:67%;" />

       - 아래에 있는 코드는 머신러닝 모델을 만드는 데 사용할 수 있음.

         - 1번째 줄: 모델 자체를 정의함. 모델은 훈련된 신경망임. 여기에는 아주 간단한 신경망이 있는데 이번에는 단일 레이어로 표시된 `keras.layers.Dense` 코드임.
- units=1의 의미: 레이어는 단일 신경을 가지고 있으며  단위는 1로 표시하고 있음.
  
- input_shape=[1] : 신경망에 `단일` 값을 입력함. 여기서는 x값을 의미. 이를 통해서 신경망에서는 x에 맞는 y값을 예측할 것.  그래서 input_shape=[]에 1을 넣어준 것.
  
- 2번째 줄:  모델을 편집할때 두가지 함수가 있음. `loss`와 `optimizer`임. 머신러닝의 핵심 파트로 머신러닝은 모델이 숫자 사이의 관계에 대해서 추측하며 작동함.
         - 예를 들어 Y=2X-1 라고 예측했다고 가정했을때 `loss 함수`를 통해서 추측이 좋은지 나쁜지 판단함.(여기서는 loss='mean_squared_error')
  - `optimizer 함수`를 사용해서 `새로운 추측(수식)`을 생성함.
           - 이 두가지 함수의 결합으로 점점 더 올바른 공식으로 바꿀것. 
           
         - 5번째 줄: `epochs=500` 을 설정해 500번 반복함.  추측하고, loss함수를 통해 추측이 맞는지 계산, optimizer 함수를 사용해서 추측을 개선함. 이를 500번 반복하는 것. 

         - 3번째, 4번째 줄: 여기 코드를 통해 데이터를 input함.  여기서 데이터는 x와 y 배열로 이루어짐. x와 y를 서로 맞추는 과정은 모델의 `fit` 메서드를 이용. 

           - 이를 x를 y에 fit한다고 표현하고 여기서는 이것을 500번 반복하는 것. 

         - 6번째 줄: 완료가 되면 훈련된 모델이 나옴. 이를 확인해보는 코드

           - predict 메소드를 이용해서 주어지는 x에 대한 y를 예측함 . 
  - 답은 19가 아닌 18.9998 쯤으로 출력됌. 
           - 답인 19에는 가깝지만 맞는 답은 아님. 그 이유는 훈련시킨 케이스가 6개밖에 안 되기 때문인데 이 케이스만으로는 일직선 관계로 보이지만 그 밖에 있는 값은 일직선 관계가 아닐 수 있음. 일직선이라는 매우 높은 확률은 있지만 확신할 수 없음.
               - 이럴 경우 확률이 예측으로 이어짐. 값이 19에 매우 가깝지만 정확히 19는 아님. 머신러닝에서 많이 볼 수 있는 예시임. (예측이기때문에  완벽한 정답이 아닌 정답에 가까울 확률로 예측을 한다? 이런 느낌인듯?)
       
         <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922152629059.png" alt="image-20200922152629059" style="zoom:80%;" />





### 머신러닝을 이용한 기초 컴퓨터 비전(2화)

- 컴퓨터에게 각기 다른 대상을 식별하는 방법을 훈련하는게 목표

- 강의에서 다룬 내용은 심층 신경망(DNN)을 이용한 이미지처리(이미지의 픽셀을 라벨에 매치하는 방법)

  - 앵클 부츠 사진을 보여주고 - 9라는 라벨을 매치시킴.

- 우리가 신발을 구분할 수 있는 이유는 그동안 많은 신발을 보면서 신발이라고 배웠기 때문임. 

  따라서 컴퓨터도 많은 신발을 보여주면 무엇이 신발인지 식별할 수 있을 것.

- 이때 사용되는 게 `Fashion MNIST` 라는 `dataset`이 유용하게 사용됌. 

  - 10개의 각기 다른 항목에 70,000가지의 이미지를 가지고 있음.

  - 여기 이미지들은 모두 28x28 픽셀로만 저장됌. (신발인지 식별할 수 있다면 이렇게 작은 파일이 좋음. ->데이터 사용량 감소 -> 컴퓨터의 처리속도 증가 )

  - 아래처럼 화질이 저래도 신발인걸 알 수 있으니 문제없음.

    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922201607585.png" alt="image-20200922201607585" style="zoom:80%;" />



- 아래는 컴퓨터를 훈련시키는 방법을 알려주는 코드1

  - 이전 강의에서 작성했던 코드와 유사함. 이점이 tensorflow의 장점인데 일관된 프로그래밍 api 를 바탕으로 다양한 작업을 위한 신경망 설계를 가능하게 해줌. 

  - Fashion MNIST 데이터 세트는 TensorFlow에 내장되어 있어서 쉽게 불러 올 수 있음.

  - training image는 60,000장임. 나머지 10,000개는 신경망 테스트용.

  - 여기서는 숫자 9이 앵클부츠를 의미하는 라벨임. -> 

    - 왜 글자가 아닌 숫자일까?: 

      1. 컴퓨터가 더 숫자를 잘 처리하기 때문

      2. 컴퓨터가 잘못 판단하는 걸 막기 위함.  특히 영어로 ankle boots 라고 달아놓으면 영어에 편향된 경향을 나타내게 됌. 하지만 숫자를 사용하면 나타나는 모든 언어를 이용해서 텍스트를 설명할 수 있음.

    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922203852027.png" alt="image-20200922203852027" style="zoom:80%;" />

    ```python
    import tensorflow as tf
    from tensorflow import keras
    # fashion_mnist.load_data()를 사용하기 위해서 keras를 import 해줌.(확실하진 않음 강의에서는 언급하지 않고 표시만 함.)
    
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()   
    # fashion_mnist.load_data()를 통해서 Fashion MNIST 데이터 세트 불러옴.
    ```

- 아래는 컴퓨터를 훈련시키는 방법을 알려주는 코드2

  - `keras.layers.Flatten(input_shape=(28, 28))` : 28x28이라는 사진크기를 입력.

    - 여기처럼 사진 크기와 같이 데이터의 shape을 지정해 줄 필요는 없지만 Flatten() layer를 없애면 shape of the data에 대한 에러뜸.

    - 사진의 크기가 28x28인데 이를 위해서 784x1로 flatten 해줌.

      > Right now our data is 28x28 images, and 28 layers of 28 neurons would be infeasible, so it makes more sense to 'flatten' that 28,28 into a 784x1.

  - `keras.layers.Dense(10, activation=tf.nn.softmax)`: 

    - 여기서 10의 의미는 의류의 개수. 즉, 미리 지정한 카테고리의 개수임. 

    - 10의 개수는 꼭 내가 분류하려는 카테로리의 개수와 일치해야함. 아니면 에러뜸.

      > the number of neurons in the last layer should match the number of classes you are classifying for.

    - 여기가 final(output) layer임. 모델 정의할때 마지막에 output layer 만들어 주면 되는 것 같음.

    - 신경망은 필터의 역할을 수행하여 28x28의 픽셀의 세트(이미지)를 받아서 10개 값 중에서 1개를 출력함.

    - 여기서 nn은 neural network의 약자인 것 같음.

    - 각 layer의 neurons 에게 activation function을 사용해서 뭘 해야할지 말해줘야함. 많은 옵션들이 있음.

  - `keras.layers.Dense(128, activation=tf.nn.relu)`: 128의 의미는 함수의 개수= `뉴런의 개수`

    - 128개의 함수가 있다고 가정, 각 함수는 자기만의 매개 변수를 가짐. (여기서는 f0 ~ f127)

    - 우리가 원하는 결과는 신발의 픽셀이 하나씩 그 함수에 입력될 때 모든 함수의 조합을 통해서 올바른 값을 출력함. 여기 예시에서는 9

    - 올바른 결과를 위해서 컴퓨터가 각 함수들 안의 매개 변수들이 어떤 값을 가지고 있는지를 알아내야 함.

    - 그리고 데이터 세트의 다른 의류 항목으로 확장해 나감. 그리고 이런 작업이 완료되면 의류 항목을 식별할 수 있게 됌.

    - 뉴런의 개수를 증가시키면 학습하는 데 좀 더 오래 걸리지만 더 정확한 예측 결과를 얻을 수 있음.

      <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922210118070.png" alt="image-20200922210118070" style="zoom:80%;" />

  - `activation=tf.nn.relu` 의미: relu라고 부르며 128개 함수의 레이어에 있음. `정류 선형 유닛(rectified linear unit)`이라고도 불림. 

    - relu의 역할은 0보다 크면 해당 값을 return 하고 0보다 작으면 0을 return 함. 즉, 0보다 작은 값을 필터링 하는 역할을 함.

    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922213218828.png" alt="image-20200922213218828" style="zoom:67%;" />

  - `activation=tf.nn.softmax`의미: 아까 열개의 카테고리(의류의 종류)로 나눴는데 10개중에서 속할 확률 중에서 가장 큰 확률을 뽑아냄. 즉, `softmax 함수`는 한 세트안에서 가장 큰 숫자를 골라냄.

    - 이 신경망 내의 출력 레이어는 10개 항목을 포함하는데 의류의 특정 항목에 속할 확률을 나타내는 역할을 함. 이 경우에는 9번 항목(앵클 부츠)에 가장 큰 확률을 보이고 있음. softmax 함수는 가장 큰 확률을 골라내서 해당 항목을 1로 설정하고 나머지는 0으로 설정함.

      <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922214145981.png" alt="image-20200922214145981" style="zoom:80%;" />

  - `model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy')` : 이 코드를 통해서 신경망은 임의의 값으로 초기화 됌. 

    - 이후 `loss 함수`를 통해서 결과가 얼마나 좋은지 나쁜지를 측정. (우리가 원하는 값에 얼마나 가까운지 먼지)
    - `optimizer 함수`를 통해서 함수의 새로운 매개변수를 생성함(더 좋게 예측하기 위해서)

- 아래는 컴퓨터를 훈련시키는 방법을 알려주는 코드3

  - 훈련하는 코드
    - `model.fit(train_images, train_labels, epochs=5)` : 훈련 라벨에  훈련 이미지를 대입하면 됌. 여기서는 5번만 반복함.

    - `test_loss, test_acc = model.evaluate(test_images, test_labels)`: `evaluate()` 메서드에 이미지들을 보내서 테스트를 할 수 있음. 이를 통해서 모델이 얼마나 예측하는지, 잘 맞추는지 봄. 

      모델이 한번도 안 본 케이스들도 잘 구분하는 지 보기 위함.

    - `predictions = model.predict(my_images)`: `model.predict()`를 통해서 새로운 이미지를 받아 예측(구분)함.

- 이번 코드에서의 `단점`은 사진 속에 `신발만` 있고 `중앙`에 있음. 그리고 그레이스케일임. 따라서 여러 객체가 있는 사진에서도 AI가 똑같이 분류하고 인별하기 위해서는  특징을 잡아주는 과정과 나선형 신경망(convolutional neural networks, CNN)의 도구가 필요함.

- 전체 코드 사진 및 코드

  - 내가 이해한 것.
    - 2번째줄에서 input의 shape을 정해주고 받음. 
    - 3번째줄은 128개의 함수를 만드는 데 그 함수들은 relu임. 이 128개의 함수의 return값을 조합해서 10개의 어떤 카테고리에 해당하는 지 확률을 만듬.
    - 4번째줄에서 그 확률 중에서 가장 큰 확률을 뽑고 해당 카테고리에느 1을 주고 나머지에는 0을 주는 역할을 함.
    - 6번째줄에서는 loss 즉, 현재까지 만들어진 모델이 정확하게 예측(구분)했는지 정도를 판단함. 
    - 7번째줄에서는 6번째줄의 loss를 바탕으로 함수안에 새로운 파라미터를 생성하여 해당 모델을 개선함. 

  <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922211236860.png" alt="image-20200922211236860" style="zoom:80%;" />

```python
#1
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
#2
model.compile(optimizer=tf.train.AdamOptimizer(), 			     				                 loss='sparse_categorical_crossentropy')
#3
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)

predictions = model.predict(my_images)
```

- 아래처럼 256개의 뉴런을 가진 layer를 처음과 final layer에 추가한다면?
  - 지금같은 간단한 데이터를 분석할 경우에는 별차이가 없지만 색이 들어간 사진과 같은 좀 더 복잡한 데이터의 경우에는 layer를 추가시켜야함. 

```
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
```

- list max 값 찾기

  ```python 
  # 최대값 찾기
  a = [1,2,3]
  max(a)
  
  # 최대값을 가지는 index 출력.
  a.index(max(a))
  
  # numpy에서 최대값 찾기
  np.amax(array 이름)
  np.amin(array 이름)
  # numpy에서 해당하는 값의 인덱스 찾기
  np.where('찾으려는 원소' == array이름)
  ```

  

### Introducing convolutional neural networks (ML Zero to Hero - Part 3)

- 이전 영상에서는 사진속에 하나의 객체가 사진 중앙에 있었음. 그리고 이런 사진들만 분류할 수 있었음. 하지만 현실 사진은 이렇지 않음. 따라서 이런 한계을 보안하기 위해 CNN(Convolutional Neural Network, 나선형 신경망)을 이번 강의에서 사용할 것임.

- CNN의 원리는 심층 신경망(DNN)을 사용하기 전에 이미지를 필터링 하는 것. 먼저 이미지를 필터링 하게 되면 이미지 내 특징들이 나타나고  그 특징들을 하나하나 인식할 수 있게 됌. 

- 여기서 필터는 단순히 곱한 수들의 집합(set)임.

  - 예시

    1. 예를 들어서 192라는 값을 가진 픽셀 하나를 본다고 한다면 필터는 빨강색 상자에 있는 값들임.

       <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922230808581.png" alt="image-20200922230808581" style="zoom:67%;" />

    2.  그리고 사진과 필터(빨강색 상자)의 같은 행과 열에 있는 값들을 서로 곱함. 그리고 모두 더해서 픽셀에 대한 새로운 값을 구함.

       <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922230940635.png" alt="image-20200922230940635" style="zoom: 67%;" />

    3. 다른 필터에 대한 예시를 보면 첫번째 예시에서 앞의 과정처럼 `필터가 적용된(필터와 곱한)` 사진을 보면 `세로선`만 제외하고 거의 모든 선이 지워진걸 확인할 수 있음. 두번째 예시는 필터가 적용되었을때 `수평선`만 제외하고 거의 모든 선이 지워진걸 확인할 수 있음.

      <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922231456179.png" alt="image-20200922231456179" style="zoom: 67%;" />

      <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922231708515.png" alt="image-20200922231708515" style="zoom:67%;" />

  - 다음으로 풀링(pooling)이라는 것과 결합될 수 있는데 이는 이미지 픽셀과 필터를 묶어 부분 집합으로 만들어 주는 기능임.

    - 풀링 예시
      - 예를 들어서 2x2 맥스 풀링을 하게 되면 이미지를 2x2 픽셀로 묶어주고 그 픽셀 중 최대값을 선택해줌.
      - 그리고 다시 합침. 따라서 이미지는 원본의 1/4로 줄어들지만 가장 눈에 띄는  특징은 그대로 남아있게 됌.

    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922232202399.png" alt="image-20200922232202399" style="zoom:67%;" />

    - 앞서 필터를 적용했던 이미지에 맥스 풀링을 하면 아래와 같이 됌.

      - 오른쪽 이미지는 왼쪽의 1/4 사이즈이지만 수직선 특징은 그대로 남아 있으면서 오히려 더 강조 됌.

      <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922232359001.png" alt="image-20200922232359001" style="zoom:67%;" />

- 해당 필터는 어떻게 만들까?

  - CNN의 주요 기능으로 해당 필터는 배울 수 있음.

  - 지난 시간의 신경망의 뉴런내에 있는 매개변수와 동일함.

  - 이미지를 CNN에 넣으면 임의로 초기화 된 여러 필터가 이미지에 적용됌. 이후 그 결과가 다음 레이어에 입력되고 신경망은 매칭을 수행함. 이런 작업이 반복되면서 최상의 매치를 보인 이미지 필터들을 학습하게 됌. 이런 과정을 `특징 추출`이라고 함.

    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922233215382.png" alt="image-20200922233215382" style="zoom:50%;" />

    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922233239677.png" alt="image-20200922233239677" style="zoom:50%;" />

    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922233310214.png" alt="image-20200922233310214" style="zoom:50%;" />

  - 아래는 나선형 필터 레이어가 어떻게 컴퓨터가 물체를 visualize 하는 것을 돕는 지에 대한 예시

  - 신발 사진들이 있는데 가장 윗줄을 보면 신발 이미지가 있는데 신발의 밑창과 실루엣만 남아있는 것을 볼 수 있음. 필터는 신발이 어떻게 생겼는지 학습한 상태임.

    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200922233659800.png" alt="image-20200922233659800" style="zoom:67%;" />

- 위의 과정을 코드로 구현하는 방법(즉, 나선형 신경망을 만드는 코드)

  - 이전 강의에서 사용된 코드와 아주 유사함.

  - `flatten` 된 `input` 값이 `dense layer`로 들어가고 이는 최종 `dense layer`을 지나서 결과물로 나옴.

  - 이전 강의 코드와의 유일한 차이점은 입력값 형태(28x28)를 설정하지 않았다는 것. 왜냐하면 추가로 나선형 레이어를 넣어줄것이기 때문에

    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200923014416099.png" alt="image-20200923014416099" style="zoom:67%;" />

  	```python
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  	```

  - 아래와 같이 나선형 레이어 추가함. 

    - `tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1))`
  - 두번째 코드줄에서 입력값의 형태를 설정해주기 때문에 Flatten에서 형태를 설정해주기 않아도 됌.
      - 첫번째 인자로 64를 받는 데 이는 64개의 필터를 만들라는 의미. 이를 이미지 전체에 개별적으로 곱해서(각 필터를 이미지에 적용) 각각의 epoch 에서 어떤 필터가 최상의 결과를 내는지 파악함.(즉, 어떤 필터가 라벨과 가장 잘 맞춰지는 이미지를 제공하는지 파악함. 라벨과 이미지 매칭을 잘하는지 파악한다는 의미인듯)
    - 위의 과정은 dense layer 에서 어떤 매개변수(인자)가 가장 효과가 좋았는지 알게되는 것과 매우 유사한 방식을 통해서 이뤄짐.
    - `tf.keras.layers.MaxPooling2D(2,2)`
      - 이미지를 압축하고 특징을 강조하는 맥스 풀링을 함.
    
    ```python
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    ```
    
    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200923020009200.png" alt="image-20200923020009200" style="zoom: 80%;" />
    
  - 나선형 레이어를 계속해서 쌓아 올림. 

    - 이를 통해서 이미지를 완전 해체하여 추상적인 특징도 학습할 수 있음.
    - 이런 방법을 사용하면 단순한 픽셀의 패턴을 기반으로 하는게 아니라 이미지의 특징을 기반으로 신경망이 학습을 하게 됌.  소매가 두 개면 셔츠고 짧은 소매가 두 개면 티셔츠로 인식하는 것처럼

    ```python
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    ```

![image-20200923020403417](C:\Users\multicampus\Documents\s03p31c203\Document\KJW\Tensorflow 강의 정리\Tensorflow lecture.assets\image-20200923020403417.png)



### Build an image classifier (ML Zero to Hero - Part 4)

- 3편까지 복습 후에 볼것.
- 과적합(over-fitting) 내용 
  - 하나의 카테고리의 양이 많을 경우 잘못 학습하게 됌. 예를 들어서 평생 하이일만 봤다면 다른 신발을 신발로 인식 못 할 수 있음.
  - 이를 해결하기 위한 방법 중 하나가 `image augmentation`

- image augmentation 실습해보는 코드 있음.



### 꼭 아래 코드 링크 참고할 것 훨씬 자세하게 알려줌.

- 이때 실습 및 코드 실행은 구글드라이브에서 colab 파일 생성해서 할것. 



- 링크

  - 영상
    
    - 1편: https://www.youtube.com/watch?v=KNAWp2S3w94&list=PLQY2H8rRoyvwLbzbnKJ59NkZvQAW9wLbx&index=15&ab_channel=TensorFlow
    - 2편: https://www.youtube.com/watch?v=bemDFpNooA8&ab_channel=TensorFlow
    - 3편: https://www.youtube.com/watch?v=x_VrgWTKkiM&ab_channel=TensorFlow
    - 4편: https://www.youtube.com/watch?v=u2TjZzNuly8&list=PLQY2H8rRoyvwLbzbnKJ59NkZvQAW9wLbx&index=12&ab_channel=TensorFlow
    
  - 예시 코드

    - 1편 코드: https://codelabs.developers.google.com/codelabs/tensorflow-lab1-helloworld/#6
    - 2편 코드: https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab2-Computer-Vision.ipynb#scrollTo=OgQSIfDSOWv6
    - 3편 코드: https://codelabs.developers.google.com/codelabs/tensorflow-lab3-convolutions/#0
    - 4편 코드: https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb

  - 참고

    - 위키 백과 

      https://ko.wikipedia.org/wiki/%EB%94%A5_%EB%9F%AC%EB%8B%9D#%EC%8B%AC%EC%B8%B5_%EC%8B%A0%EA%B2%BD%EB%A7%9D(Deep_Neural_Network,_DNN)

    - ANN,DNN,CNN, RNN 개념 차이

      https://ebbnflow.tistory.com/119

    - image aumentation

      https://github.com/aleju/imgaug

      http://blog.naver.com/PostView.nhn?blogId=4u_olion&logNo=221437862590&parentCategoryNo=&categoryNo=45&viewDate=&isShowPopularPosts=true&from=search

      https://hoya012.github.io/blog/albumentation_tutorial/





# Tensorflow 강의 코드 실습

### 1강

### 2강

- list  max값 찾기

  ```python
  # 최대값 찾기
  a = [1,2,3]
  max(a)
  
  # 최대값을 가지는 index 출력.
  a.index(max(a))
  
  # numpy에서 최대값 찾기
  np.amax(array 이름)
  np.amin(array 이름)
  # numpy에서 해당하는 값의 인덱스 찾기
  np.where('찾으려는 원소' == array이름)
  ```

  

### 3강(part3)

- Convolution이란?

  - 합성이라는 의미
  - Convolution은 전자공학에서 linear time invaiant system 상에서 이전값과 현재값을 연산하기 위해 주로 사용하던 연산이였으나 CNN 에서는 이미지내에서 feature을 뽑기 위한연산으로 활용.
  -  https://talkingaboutme.tistory.com/entry/DL-Convolution%EC%9D%98-%EC%A0%95%EC%9D%98

- 2강에서 한계

  - 실제로 어느 정도 정확하게 분류해주지만 grayscale과 물체가 사진 한 가운데, 홀로 있을때만 작동함.

  - CNN을 써야하는 이유

    <img src="C:\Users\multicampus\Desktop\수업 markdown\Tensorflow 공식채널 강의.assets\image-20200926142150449.png" alt="image-20200926142150449" style="zoom:67%;" />

- 사용되는 사진 
  
  - scipy에서 가져옴. 다양한 각도와 장면이 있어서 
- convolutions은 AI가 분류, 인식을 잘 할 수 있도록 feature을 강조하고 pooling은 이 강조된 feature을 유지하면서 data의 크기를 줄이는 게 목표인듯.
  
- 원문: As well as using convolutions, pooling helps us greatly in detecting features. The goal is to reduce the overall amount of information in an image while maintaining the features that are detected as present.
  
- 필터 적용

  - 필터 중심을 움직이면서 필터적용(곱해주려고)하려고 그런듯.
  - 예시를 보면 if문 이전까지 다 더한 값이 255가 넘음. 근데 if문 통해서 0보다 작은 건 0으로 255 보다 큰건 255를 넘어줌. 따라서 강조될 건 강조 되고 약한 부분은 연하게 처리되는 듯. 그 사이 값은 그대로 넣어주고
  - 근데 이렇게 하면 상하좌우 1줄씩은 적용이 안되는데 상관이 없는 건가? 이것도 사진에 따라서 사용자가 필터를 어떻게 사용할 것인가에 따라서 그냥 다르게 사용하면 되는 건가.
  - 그리고 가중치도 여기서는 1로 줬는데 확실히 가중치를 몇으로 주는가에 따라서 강조될 수 있겠음. 예를 들어서 다 던한 값이 64인데 가중치가 2이면 128로 되어서 필터를 거치고 나면 좀 더 그 부분은 강조될 수 있음.

  ```python
  for x in range(1,size_x-1):     # 필터의 중심을 기준으로 
    for y in range(1,size_y-1):
        output_pixel = 0.0
        output_pixel = output_pixel + (i[x - 1, y-1] * filter[0][0])
        output_pixel = output_pixel + (i[x, y-1] * filter[0][1])
        output_pixel = output_pixel + (i[x + 1, y-1] * filter[0][2])
        output_pixel = output_pixel + (i[x-1, y] * filter[1][0])
        output_pixel = output_pixel + (i[x, y] * filter[1][1])
        output_pixel = output_pixel + (i[x+1, y] * filter[1][2])
        output_pixel = output_pixel + (i[x-1, y+1] * filter[2][0])
        output_pixel = output_pixel + (i[x, y+1] * filter[2][1])
        output_pixel = output_pixel + (i[x+1, y+1] * filter[2][2])
        output_pixel = output_pixel * weight
        if(output_pixel<0):
          output_pixel=0
        if(output_pixel>255):
          output_pixel=255
        i_transformed[x, y] = output_pixel
  ```

- 많은 종류의 pooling이 있는데 여기서는 max pooling을 사용함. 



# Tensorflow 이용하면서 공부한 것 정리

- 가상환경구성
  - tensorflow-gpu=2.0.0,  numpy=1.18.1, pillow(PIL)
  - tensorflow-gpu, tensorflow-cpu의 차이는 연산시 어떤걸 이용하나 차이인듯하고 gpu가 더 빠른듯.
- 체크포인트
- 모델 저장하기
  - 참고 링크
    1. https://www.tensorflow.org/guide/checkpoint?hl=ko
    2. https://www.tensorflow.org/tutorials/keras/save_and_load
- 모델 불러오기, 사용
- matplotlib 사용해서 정확도 시각화해서 보여주기.