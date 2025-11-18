from ultralytics import YOLO
import os

# 1. 학습된 모델 불러오기
# trained_models 폴더에 저장된 모델 경로를 지정하세요.
model_path = "./trained_models/yolo_food_detector.pt" 
model = YOLO(model_path)

# 2. 테스트할 이미지 경로 (파일 하나 혹은 폴더 전체)
source_path = "./test_image.jpg"  # 혹은 "./test_images/"

# 3. 추론 실행 (Inference)
# save=True: 결과 이미지 저장
# conf=0.5: 확신(Confidence)이 50% 이상인 것만 표시 (수치 조절 가능)
results = model.predict(source=source_path, save=True, conf=0.5)

# 4. 결과 확인 (콘솔 출력 및 경로 안내)
for result in results:
    boxes = result.boxes  # 탐지된 박스 정보
    print(f"탐지된 객체 수: {len(boxes)}")
    # 탐지된 클래스 이름 출력
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        conf = float(box.conf[0])
        print(f" - {class_name} (정확도: {conf:.2f})")

print(f"\n결과 이미지는 {results[0].save_dir} 에 저장되었습니다.")