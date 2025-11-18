import torch
import timm
from PIL import Image
import os
import argparse
import sys
from torchvision import transforms

# Food-101 클래스 리스트 (데이터셋 없이 실행할 경우를 위해 하드코딩)
# 만약 데이터셋이 있다면 torchvision에서 불러올 수도 있지만, 독립 실행을 위해 리스트를 포함함
FOOD101_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 
    'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 
    'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 
    'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 
    'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 
    'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 
    'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 
    'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 
    'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 
    'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 
    'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def predict_image(image_path, model_path):
    # 1. 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using the training script.")
        return

    device = get_device()
    print(f"Using device: {device}")

    # 2. 모델 구조 생성 (학습 때와 동일해야 함)
    # num_classes=101 (Food-101 데이터셋 클래스 개수)
    HF_MODEL_NAME = 'timm/efficientnet_b0.ra_in1k'
    try:
        model = timm.create_model(HF_MODEL_NAME, pretrained=False, num_classes=101)
    except ImportError:
        print("Error: 'timm' library is missing. Install it via 'pip install timm'")
        return

    # 3. 가중치 로드
    print(f"Loading model weights from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # 저장 방식에 따라 state_dict 키가 다를 수 있음 처리
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.to(device)
    model.eval() # 평가 모드 전환

    # 4. 이미지 전처리 (timm 설정 활용)
    # 학습 코드와 동일하게 모델에 맞는 설정을 가져옴
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config)

    # 5. 이미지 로드 및 변환
    try:
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device) # 배치 차원 추가 (1, C, H, W)
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # 6. 추론 실행
    print(f"Analyzing image: {image_path}")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # 7. 결과 출력 (Top 3)
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    
    print("\n--- Inference Results ---")
    for i in range(top3_prob.size(0)):
        idx = top3_catid[i].item()
        prob = top3_prob[i].item() * 100
        class_name = FOOD101_CLASSES[idx]
        print(f"{i+1}. {class_name}: {prob:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EfficientNet Inference Script")
    parser.add_argument("--source", type=str, default="./test_image.jpg", help="Path to the image file")
    parser.add_argument("--model", type=str, default="./trained_models/efficientnet_food_classifier.pth", help="Path to the trained model file")
    
    args = parser.parse_args()
    
    predict_image(args.source, args.model)