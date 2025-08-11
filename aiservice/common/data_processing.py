from ultralytics import settings, YOLO
from roboflow import Roboflow
import os, json, shutil, itertools, csv
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import optuna
import albumentations as A
from collections import Counter, defaultdict
import random
import itertools
import yaml


def augment_to_target_counts_v2(name_and_bbox,
                                target_per_class,
                                output_dir):
    """
    사용자 업로드 이미지 데이터 증강 V2 : 모든 이미지가 균등하게 증강되도록 분배
    """
    os.makedirs(output_dir, exist_ok=True)

    class_items = defaultdict(list)
    for item in name_and_bbox:
        if item["xyxy"]:
            class_items[item["class"]].append(item)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=15, p=0.3),
        A.GaussNoise(p=0.2),
        A.MotionBlur(p=0.2),
        A.RandomCrop(width=800, height=800, p=0.3),
        A.CoarseDropout(p=0.3),
        A.ColorJitter(p=0.4),
        A.CLAHE(p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    aug_name_and_bbox = []
    augment_counter = defaultdict(int)
    class_aug_counts = defaultdict(int)

    for cls, items in class_items.items():
        # 중복 제거된 고유 이미지 경로 기준
        unique_items = {}
        for item in items:
            unique_items[item["origin_img_path"]] = item
        unique_list = list(unique_items.values())

        original_count = len(unique_list)
        need_aug = max(target_per_class - original_count, 0)

        per_image_aug = [need_aug // original_count] * original_count
        for i in range(need_aug % original_count):
            per_image_aug[i] += 1

        for item, aug_times in zip(unique_list, per_image_aug):
            img_path = item["origin_img_path"]
            bboxes = item["xyxy"]

            image = cv2.imread(img_path)
            if image is None:
                print(f"이미지 없음 : {img_path}")
                continue

            class_labels = [cls] * len(bboxes)
            bboxes_pascal = [[x1, y1, x2, y2] for x1, y1, x2, y2 in bboxes]

            for _ in range(aug_times):
                transformed = transform(image=image, bboxes=bboxes_pascal, class_labels=class_labels)
                if not transformed["bboxes"]:
                    # print(f"bbox 없음 : {img_path}")
                    continue

                aug_img = transformed["image"]
                aug_bboxes = transformed["bboxes"]

                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                augment_counter[base_filename] += 1
                aug_idx = augment_counter[base_filename]

                filename = f"aug_{base_filename}_{aug_idx}.jpg"
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, aug_img)

                aug_name_and_bbox.append({
                    "origin_img_path": save_path,
                    "class": cls,
                    "xyxy": [[x1, y1, x2, y2] for x1, y1, x2, y2 in aug_bboxes]
                })

                class_aug_counts[cls] += 1

    print(f"\n총 유효 원본 수: {sum(len(set(i['origin_img_path'] for i in v)) for v in class_items.values())}")
    print("클래스별 원본 수:")
    for cls, items in class_items.items():
        print(f"  - {cls}: {len(set(i['origin_img_path'] for i in items))}장")
    print("클래스별 증강 수:")
    for cls, count in class_aug_counts.items():
        print(f"  - {cls}: {count}장 증강됨")

    return aug_name_and_bbox


def config_train_datasets(base_url,               # /app
                          user_id,                # your_user_id
                          name_and_bbox, 
                          ai_datasets_base_path): # /app/tmp/datasets/your_user_id
    """
    YOLO모델(2번) 학습데이터셋 구성
    """

    # 데이터 증강(클래스별로 균등하게)
    aug_name_and_bbox = augment_to_target_counts_v2(name_and_bbox=name_and_bbox,
                                                target_per_class=300, # 각 클래스당 300장
                                                output_dir=os.path.join(base_url, f'tmp/augmented/{user_id}'))

    # 원본 + 증강 병합
    combined_data = name_and_bbox + aug_name_and_bbox

    # 클래스별로 샘플 모으기
    class_to_samples = defaultdict(list)
    for item in combined_data:
        class_to_samples[item["class"]].append(item)

    # 클래스별로 train/valid 나누기
    train_data, valid_data = [], []
    for cls, samples in class_to_samples.items():
        random.shuffle(samples)
        split_idx = int(len(samples) * 0.7)  # 7:3 비율
        train_data.extend(samples[:split_idx])
        valid_data.extend(samples[split_idx:])

    # 클래스 -> ID 매핑
    class_names = sorted(class_to_samples.keys())
    class_to_id = {cls: i for i, cls in enumerate(class_names)}

    # YOLO 라벨 저장 함수 (정규화 포함)
    def save_yolo_label(label_path, bboxes, cls_id, img_width, img_height):
        with open(label_path, 'w') as f:
            for x1, y1, x2, y2 in bboxes:
                cx = (x1 + x2) / 2 / img_width
                cy = (y1 + y2) / 2 / img_height
                w = (x2 - x1) / img_width
                h = (y2 - y1) / img_height
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # 이미지/라벨 저장
    for split_name, dataset in [('train', train_data), ('valid', valid_data)]:
        for item in dataset:
            img_path = item['origin_img_path']
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'

            # 이미지 저장
            dst_img_path = os.path.join(ai_datasets_base_path, split_name, 'images', img_name)
            shutil.copy(img_path, dst_img_path)

            # 라벨 저장
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            dst_label_path = os.path.join(ai_datasets_base_path, split_name, 'labels', label_name)
            save_yolo_label(dst_label_path, item['xyxy'], class_to_id[item['class']], w, h)

    # data.yaml 생성 또는 갱신
    yaml_path = os.path.join(ai_datasets_base_path, 'data.yaml')

    # 새 데이터 기본값
    new_data = {
        "train": os.path.join(ai_datasets_base_path, 'train/images'),
        "val": os.path.join(ai_datasets_base_path, 'valid/images'),
        # nc와 names는 아래에서 최종 결정
    }

    merged_names = list(class_names)  # 기본은 이번에 생성된 클래스들

    if os.path.exists(yaml_path):
        # 기존 yaml 읽어서 names 병합 (중복 제거, 기존 순서 우선 유지)
        try:
            with open(yaml_path, 'r') as f:
                existing = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            print("yaml 읽기 오류")
            existing = {}

        existing_names = existing.get("names", [])
        if not isinstance(existing_names, list):
            existing_names = []

        # 순서 유지하며 중복 제거: 기존 -> 신규 순
        name_seen = set()
        merged_names = []
        for nm in existing_names + class_names:
            if nm not in name_seen:
                merged_names.append(nm)
                name_seen.add(nm)

        # train/val 경로는 최신 값으로 보정(경로 변경 가능성 대비)
        new_data["train"] = existing.get("train", new_data["train"])
        new_data["val"]   = existing.get("val",   new_data["val"])

    new_data["names"] = merged_names
    new_data["nc"] = len(merged_names)

    with open(yaml_path, 'w') as f:
        yaml.safe_dump(new_data, f, sort_keys=False, allow_unicode=True)

    return yaml_path, len(train_data), len(valid_data)