import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import logging

# 로그 설정
logging.basicConfig(
    filename='mae_results.log',  # 로그 파일 이름
    level=logging.INFO,          # 로그 레벨 설정
    format='%(message)s'  # 로그 포맷
)

# 모델 로드
model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_lite0', pretrained=True)
model.eval()

# ImageNet 클래스 라벨 로딩
imagenet_labels = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.strip().split('\n')

# 전처리 정의
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 정답 태그 그룹핑
label_mapping = {
    "gun": ["rifle", "revolver", "assault rifle", "air rifle"],
    "shoe": ["cowboy boot", "sneaker", "running shoe"],
    "cooking": ["frying pan", "wok"],
    "music": ["oboe", "saxophone"],
    "camera": ["reflex camera", "digital camera"],
    "toy": ["teddy", "stuffed toy"],
    "kitchen": ["strainer", "colander"]
}

# 이미지 URL과 정답 태그 (10개)
image_data = [
    ("{filePath}/{fileName}.jpeg", "{label_mapping}")
]

# 평가 결과 저장
absolute_errors = []

for idx, (url, ground_truth) in enumerate(image_data):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"❗ Error loading image: {e}")
        continue

    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    # Top-10 예측
    topk = torch.topk(probs, 10)
    topk_dict = {imagenet_labels[idx]: val.item() for idx, val in zip(topk.indices, topk.values)}

    # 정답에 해당하는 실제 후보 태그들
    target_tags = label_mapping.get(ground_truth, [ground_truth])

    # 예측된 태그들 중 일부가 target_tags 안에 있으면 해당 확률 사용
    gt_prob = 0.0
    for label, prob in topk_dict.items():
        if any(tag.lower() in label.lower() for tag in target_tags):
            gt_prob = prob
            break

    # 절대 오차 계산
    error = abs(1.0 - gt_prob)
    absolute_errors.append(error)

    # 출력
    logging.info(f"\n이미지 {idx+1}")
    logging.info(f" - 예측 확률: {gt_prob:.4f}")
    logging.info(f" - 절대 오차: {error:.4f}")
    logging.info(" - Top-5 예측:")
    for i in range(5):
        label = imagenet_labels[topk.indices[i]]
        conf = topk.values[i].item()
        logging.info(f"   • {label}: {conf:.4f}")

# 전체 MAE 출력
mae = np.mean(absolute_errors)
logging.info(f"\n✅ 최종 MAE (정답 태그 매핑 기준): {round(mae, 4)}")
