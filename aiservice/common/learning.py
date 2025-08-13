from ultralytics import settings, YOLO
from roboflow import Roboflow
import os, json, shutil, itertools, csv
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
# import optuna
import albumentations as A
from collections import Counter, defaultdict
import random
import itertools
from common.utils import use_korean_font


def GridSearch_YOLO(epochs,
                    lr0s,
                    models,
                    base_dir, # /app/tmp
                    dataset_name, # datasets/your_user_id
                    train_output_name): # yolo2_train/your_user_id

    # 한글 폰트 사용 설정 (한 번만 호출해도 충분하지만, 안전하게 매 진입 시 보장)
    use_korean_font("/app/font/NanumSquareR.ttf")

    data_path = os.path.join(base_dir, dataset_name, 'data.yaml') # /app/tmp/datasets/your_user_id/data.yaml
    base_project = os.path.join(base_dir, train_output_name, 'gridsearch') # /app/tmp/yolo2_train/your_user_id/gridsearch

    results = []
    for epoch, lr0, model_name in itertools.product(epochs, lr0s, models):
        run_name = f"{model_name}_ep{epoch}_lr{lr0}".replace('.','p') # yolo11s_ep35_lr0p00725
        model_path = os.path.join(base_dir, f"{model_name}.pt")

        # Train
        model = YOLO(model=f"{model_name}.pt", task='detect')
        model.train(
            data=data_path,
            epochs=epoch,
            seed=20,
            pretrained=True,
            project=base_project,
            name=run_name,
            lr0=lr0,
            optimizer='Adam'
        ) # /app/tmp/yolo2_train/gridsearch/yolo11s_ep35_lr0p00725

        # Validate (val split)
        model_val = model.val(
            data=data_path,
            split="val",
            save=True,
            project=base_project,
            name=f"val_ep{epoch}_lr{lr0}"
        )

        # Train 중 생성된 results.csv에서 best row 추출
        metrics_path = os.path.join(base_project, run_name, "results.csv")
        if not os.path.exists(metrics_path):
            # 스킵(필요시 예외처리로 바꿔도 됨)
            continue

        with open(metrics_path) as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            if not rows:
                continue

            # mAP50(B) 최고인 row (best.pt 대응)
            best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0.0)))
            map50 = float(best_row.get("metrics/mAP50(B)", 0.0))
            precision = float(best_row.get("metrics/precision(B)", 0.0))
            recall = float(best_row.get("metrics/recall(B)", 0.0))

        # val 측정치
        mp, mr, map50_val, _ = model_val.box.mean_results()
        val_precision = float(mp)
        val_recall = float(mr)
        val_map50 = float(map50_val)

        # best.pt 경로
        best_file_path = os.path.join(base_project, run_name, "weights", "best.pt")

        results.append({
            "model": model_name,
            "epochs": epoch,
            "lr0": lr0,
            "run_name": run_name,
            "map50": map50,
            "precision": precision,
            "recall": recall,
            "val_map50": val_map50,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "best_file_path": best_file_path
        })

    if not results:
        raise RuntimeError("GridSearch 결과가 비어 있습니다. results.csv 생성 여부를 확인하세요.")

    # val_map50 기준으로 정렬 후 베스트 1개 반환
    results.sort(key=lambda x: x["val_map50"], reverse=True)
    best = results[0]
    return best
