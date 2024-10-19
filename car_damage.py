import os
import random
import shutil
import cv2
import glob
import json
import numpy as np
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.draw import polygon2mask

# 데이터 불러오기
data_root = '데이터 경로'
file_root = os.path.join(data_root, 'data')
cls_list = ['Separated', 'Breakage', 'Separated', 'Crushed'] # 클래스 리스트

project_name = 'cd'
train_root = os.path.join(data_root, project_name, 'train')
valid_root = os.path.join(data_root, project_name, 'valid')
test_root = os.path.join(data_root, project_name, 'test')

# 학습, 검증, 테스트 데이터 폴더 생성
for folder in [train_root, valid_root, test_root]:
    if not os.path.exists(folder):
        os.makedirs(folder)
    for s in ['images', 'labels']:
        s_folder = os.path.join(folder, s)
        if not os.path.exists(s_folder):
            os.makedirs(s_folder)

# 이미지의 가로, 세로 크기를 기준으로 폴리곤의 각 점의 좌표를 0과 1사이의 값으로 변환
def json_to_yolo_polygon(polygon, w, h):
    yolo_list = []
    for p in polygon:
        yolo_list.append(p[0]/w) # x 좌표를 이미지 너비 w로 나눠 정규화
        yolo_list.append(p[1]/h) # y 좌표를 이미지 높이 h로 나눠 정규화
    return " ".join([str(x) for x in yolo_list])

# JSON 파일을 YOLO 형식으로 변환
file_list = glob.glob(os.path.join(file_root, 'annotations', '*.json'))
random.seed(2024)
random.shuffle(file_list)
print(len(file_list))

if not os.path.isdir(os.path.join(file_root, 'labels')):
    os.mkdir(os.path.join(file_root, 'labels'))

for file in tqdm(file_list):
    result = []
    with open(file, 'r') as json_file:
        data = json.load(json_file)
        h = data['images']['height']
        w = data['images']['width']
        for ann in data['annotations']:
            label = ann['damage']
            if label in cls_list:
                polygon_cood = ann['segmentation'][0][0][:-1]
                cood_string = json_to_yolo_polygon(polygon_cood, w, h)
                yolo_string = f'{cls_list.index(label)} {cood_string}'
                result.append(yolo_string)

    if result:
        save_path = file.replace('annotations', 'labels').replace('json', 'txt')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(result))

# 데이터셋 분할
file_list = glob.glob(os.path.join(file_root, 'labels', '*.txt'))
random.shuffle(file_list)
test_ratio = 0.1
num_file = len(file_list)

test_list = file_list[:int(num_file * test_ratio)]
valid_list = file_list[int(num_file * test_ratio):int(num_file * test_ratio) * 2]
train_list = file_list[int(num_file * test_ratio) * 2:]

for i in test_list:
    label_name = os.path.basename(i)
    shutil.copyfile(i, os.path.join(test_root, 'labels', label_name))
    img_name = label_name.replace('txt', 'jpg')
    img_path = os.path.join(file_root, 'images', img_name)
    shutil.copyfile(img_path, os.path.join(test_root, 'images', img_name))

for i in valid_list:
    label_name = os.path.basename(i)
    shutil.copyfile(i, os.path.join(valid_root, 'labels', label_name))
    img_name = label_name.replace('txt', 'jpg')
    img_path = os.path.join(file_root, 'images', img_name)
    shutil.copyfile(img_path, os.path.join(valid_root, 'images', img_name))

for i in train_list:
    label_name = os.path.basename(i)
    shutil.copyfile(i, os.path.join(train_root, 'labels', label_name))
    img_name = label_name.replace('txt', 'jpg')
    img_path = os.path.join(file_root, 'images', img_name)
    shutil.copyfile(img_path, os.path.join(train_root, 'images', img_name))

# YOLO 모델 학습 및 검증 설정 / 클래스 정보를 담은 YAML 설정 파일을 생성
data = dict()
data['train'] = train_root
data['val'] = valid_root
data['test'] = test_root
data['nc'] = len(cls_list)
data['names'] = cls_list

# 위에서 설정한 정보를 YAML 형식으로 파일에 저장
# YAML 파일 생성 경로를 절대 경로로 설정
yaml_file_path = os.path.join(data_root, 'car_damage.yaml')

# YAML 파일을 절대 경로로 저장
with open(yaml_file_path, 'w') as f:
    yaml.dump(data, f)

# YOLOv8s 세그멘테이션 모델 학습 / device='cpu'
model = YOLO('yolov8s-seg.yaml')
results = model.train(data=yaml_file_path, epochs=10, batch=16, device=0, patience=5, name='yolo_s')

# 저장된 모델 불러오기
model = YOLO('runs/segment/yolo_s5/weights/best.pt')

# 테스트 및 결과 시각화
test_file_list = glob.glob(f'{test_root}/images/*')
random.shuffle(test_file_list)

test_img = cv2.imread(test_file_list[10])
img_src = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# 모델 예측
result = model(img_src)[0]

if result.masks is not None:
    result_mask = np.zeros(test_img.shape[:2])
    masks = result.masks

    for m in masks:
        polygon_coord = m.xy[0]
        mask = polygon2mask(test_img.shape[:2], polygon_coord)
        result_mask = np.maximum(mask, result_mask)

    result_mask = np.repeat(result_mask[:, :, np.newaxis], 3, -1)

    plt.subplot(1, 2, 1)
    plt.imshow(img_src)
    plt.subplot(1, 2, 2)
    plt.imshow(result_mask)
    plt.show()
else:
    print("No masks detected.")