# Tensorflow gpu 사용을 위한 설정

### Tensorflow-gpu 2.3.0 사용을 위한 설정.

- 이미지 학습시 gpu로 학습하는 게 훨씬 빠름. 따라서 Tensorflow-gpu를 사용.

- gpu 사용을 위한 소프트웨어 요구사항

  - 참고 링크: https://www.tensorflow.org/install/gpu

  <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\tensorflow gpu 사용을 위한 설정\tensorflow gpu 사용을 위한 설정.assets\image-20201112023102070.jpg" alt="image-20201112023102070" style="zoom: 67%;" />

- tensorflow-gpu-2.3.0 버전을 지원하는 CUDA, cuDNN 버전

  - 참고링크: https://www.tensorflow.org/install/source_windows

  <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\tensorflow gpu 사용을 위한 설정\tensorflow gpu 사용을 위한 설정.assets\image-20201112022926195.jpg" alt="image-20201112022926195" style="zoom:80%;" />
  
  - 따라서 해당 버전의 NVIDIA GPU 드라이버, CUDA Tookit, cuDNN 을 설치해줘야함. 
    - NVIDIA GPU 드라이버 `418.x 이상`
    - CUDA Tookit은 `10.1 update 2` 로 설치할 것.
    - cuDNN은  `v7.6.5 (November 5th, 2019), for CUDA 10.1`

#### 설치 전 확인사항

- 다른 버전의 CUDA가 설치되어 있는지 확인.

- 설치 되어 있다면 프로그램 추가/제거에 들어가서 NVIDIA가 적힌 모든 앱을 제거할 것. (`NVIDIA 그래픽 드라이버의 버전이 418.x이상이면 제외.`)

- 아래는 삭제 내역임.(`NVIDIA CUDA Development10.1은 사진에서 짤렸지만 삭제함.`)

  <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\tensorflow gpu 사용을 위한 설정\tensorflow gpu 사용을 위한 설정.assets\설치 내역.JPG" alt="설치 내역" style="zoom:50%;" />

- 이후 C:\Program Files, C:\Program Files (x86) 에 설치되어 있는 `NVIDIA Corporation`, `NVIDIA GPU Computing Toolkit` 폴더 모두 삭제함.



#### 설치하기

- 그래픽 드라이버를 삭제했다면 그래픽 드라이버 설치하기. (`457.30 버전으로 설치함.`)

  <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\tensorflow gpu 사용을 위한 설정\tensorflow gpu 사용을 위한 설정.assets\image-20201112024553108.jpg" alt="image-20201112024553108" style="zoom: 67%;" />

  <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\tensorflow gpu 사용을 위한 설정\tensorflow gpu 사용을 위한 설정.assets\image-20201112024455417.jpg" alt="image-20201112024455417" style="zoom:67%;" />

  - 그래픽 드라이버 설치가 완료됐다면 버전 확인하기

    - cmd에서 `nvidia-smi` 입력, `Driver version` 확인 가능함.

    - 아래 사진처럼 나오는데 `CUDA Version`은 `현재 설치되어 있는 버전이 아니라` 설치된 그래픽 드라이버가 지원할 수 있는 최대 버전임.

      <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\tensorflow gpu 사용을 위한 설정\tensorflow gpu 사용을 위한 설정.assets\image-20201112025327196.jpg" alt="image-20201112025327196" style="zoom: 67%;" />

- `CUDA Toolkit 10.1 update2 (Aug 2019)` 을 다운 받고 설치함.

  <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\tensorflow gpu 사용을 위한 설정\tensorflow gpu 사용을 위한 설정.assets\설치 1.JPG" alt="설치 1" style="zoom:67%;" />

  - 다운 링크: https://developer.nvidia.com/cuda-toolkit-archive , 회원가입 필요함
  - 설치옵션 - 사용자 정의 설치에서 NVidia GeForce Experience 설치로 체크되어 있다면 필요없으니 체크 취소 하고 설치

- cuDNN을 설치함.  `v7.6.5 (November 5th, 2019), for CUDA 10.1` 을 선택해서 다운.

  - https://developer.nvidia.com/rdp/cudnn-archive

    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\tensorflow gpu 사용을 위한 설정\tensorflow gpu 사용을 위한 설정.assets\설치3.JPG" alt="설치3" style="zoom:67%;" />

  - 압축을 풀면 3개의 폴더가 있는데 안에 있는 파일을 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1` 으로 `복사`함.

    - cuda\bin 폴더 안의 모든 파일 => C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin
    - cuda\include 폴더 안의 모든 파일 => C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include
    - cuda\lib 폴더 안의 모든 파일 => C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib

- CUDA_PATH 확인하기.

  - window + R 누르고 control sysdm.cpl을 실행함. 

  - 시스템 속성 - 고급 - 환경변수 클릭.

  - 시스템 변수에  아래와 같이 되어 있는지 확인

    <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\tensorflow gpu 사용을 위한 설정\tensorflow gpu 사용을 위한 설정.assets\image-20201112030304925.jpg" alt="image-20201112030304925" style="zoom:67%;" />

- 잘 실행될 경우의 터미널 창

  <img src="C:\Users\multicampus\Documents\s03p31c203\Document\KJW\tensorflow gpu 사용을 위한 설정\tensorflow gpu 사용을 위한 설정.assets\image-20201112031029579.jpg" alt="image-20201112031029579"  />

- 참고 링크
  - https://mellowlee.tistory.com/entry/Windows-Tensorflow-Keras-%EC%82%AC%EC%A0%84-%EC%A4%80%EB%B9%84
  - https://teddylee777.github.io/colab/tensorflow-gpu-install-windows



### 가상환경

아나콘다 설치

- anaconda 홈페이지에서 3.8로 설치함.

- 기본 명령어

  - `conda update -n base -c defaults conda` : 아나콘다 버전 최신화

  - `conda create -n AI python=3.7.6` : AI 라는 이름의 가상환경 생성

    (n은 name의 약자, 뒤에 설치할 파이썬 버전 선택가능)

  - `conda activate AI`: 가상환경 실행 

    (anaconda prompt에서는 이 명령어로 하면 되는 데 VSC에서 bash 창에 입력하면 안됌. `source activate AI` 하니 실행됌. 위치와 상관없음.)

  - `conda deactivate` : 가상환경 끄기 

    (위와 마찬가지로 source deactivate 하니 실행됌. )

  - `conda info --envs` : 현재 가상환경 목록 환인

  - `conda remove -n 가상환경이름 --all` : 해당 가상환경 삭제

  - `conda list`:  패키지 확인 (pip list로 되기도 함.)

  - `conda install 패키지명` : 패키지 설치 

    (설치할때 버전 지정해줄것 아니면 최신버전 설치됌.)

    버전지정은 `패키지명=version`

  - `conda remove 패키지명`: 패키지 삭제 (아나콘다 가상환경 활성화 상태)

  - `conda update 패키지명=version`

  - `conda remove -n 가상환경이름 패키지명`: 패키지 삭제 

    (아나콘다 가상환경 비활성화 상태)

  - `python --version`: 파이썬 버전확인 (또는 python -V)

- 사용 버전

  - `tensorflow-gpu=2.3.0`
  - `scipy=1.4.1`
  - `scikit-learn=0.23.2 `
  - `numpy=1.18.1`
  - `matplotlib=3.1.3`
  - `pyqt5=5.15.1`

- 아나콘다 가상환경 export

  - `$ conda env export -n 가상환경이름 > 파일명.yml`

- export된 가상환경 설치

  - 가상환경 정보가 저장된 디렉토리로 이동하고 아래 명령어 입력.
  - `$ conda env create -f 파일명.yml`

