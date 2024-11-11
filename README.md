# yolov8_colab_image
<b> 사진하나를 다운 받아서 1.jpg로 저장한 후 yolov8에서 이미지 detect를 하는 과정


``` bash
# 필요한 패키지 설치
%pip install ultralytics opencv-python-headless matplotlib

# 구글 코랩 파일 업로드 기능 임포트
from google.colab import files

# 파일 업로드
uploaded = files.upload()
filename = next(iter(uploaded))

# 디렉토리 생성 및 파일 이동
!mkdir -p /content/sample_data
!mv {filename} /content/sample_data/{filename}

# YOLO와 이미지 표시 기능 임포트
from ultralytics import YOLO
from IPython.display import Image, display  # 수정된 부분: lPython -> IPython, lmage -> Image

# YOLO 모델 로드
model = YOLO('yolov8n.pt')

# 이미지 경로 설정
image_path = f'/content/sample_data/{filename}'

# 객체 감지 수행
results = model.predict(image_path)
first_result = results[0]

# 결과 저장
first_result.save()  # 수정된 부분: save() 메서드 호출 방식 수정

# 저장된 이미지 경로
saved_image_path = '/content/results_1.jpg'
# 결과 이미지 표시
display(Image(saved_image_path))
```
![results_1](https://github.com/user-attachments/assets/d59236eb-df49-4f83-bb19-b9b2d8f307f5)
