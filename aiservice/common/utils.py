from ultralytics import settings, YOLO
from roboflow import Roboflow
import os, json, shutil, itertools, csv
import time
from azure.storage.blob import BlobServiceClient
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
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from urllib.parse import urljoin
import requests
from tqdm import tqdm



def customer_upload_image_download_via_api(
    api_base_url: str,            # 예: "http://localhost:8000"
    x_api_key: str,                 # Depends(get_api_key)에서 요구하는 값
    container_name: str,          # 예: "origin"
    blob_prefix: str,             # 예: "your_user_id"
    cat_name: str,                # 예: "나비"
    download_dir: str,            # 예: "BASE_URL/tmp/origin/your_user_id/"
    exts: tuple = (".jpg", ".jpeg", ".png"),
    timeout_sec: int = 60,
):
    """
    사용자가 업로드한 특정 고양이 사진을 Azure Blob Storage로부터 다운로드 받는 함수

    1) /api/blobs/list 로 prefix별 blob 목록을 가져온다.
    2) /api/sas/generate 로 파일별 SAS URL을 받아 GET으로 다운로드한다.
    3) download_dir/name_{idx}.<ext> 형식으로 저장한다.

    서버 쪽(collectionservice) SAS 권한에 read=True로 수정함. (그래야, 조회가능)
    """
    os.makedirs(download_dir, exist_ok=True)

    # 엔드포인트
    list_url = urljoin(api_base_url.rstrip("/") + "/", "api/blobs/list")
    sas_url_endpoint = urljoin(api_base_url.rstrip("/") + "/", "api/sas/generate")

    headers_json = {
        "X-API-Key": x_api_key,
        "Content-Type": "application/json",
    }
    headers_get = {"X-API-Key": x_api_key}

    with requests.Session() as s:
        prefix = f"{blob_prefix.rstrip('/')}/{cat_name}/"

        # 1) 목록 조회
        resp = s.get(
            list_url,
            params={"container": container_name, "prefix": prefix},
            headers=headers_get,
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        blobs = resp.json().get("blobs", [])

        # 확장자 필터
        target_blobs = [b for b in blobs if b.lower().endswith(exts)]

        file_idx = 1
        for blob_name in target_blobs:
            # 2) SAS 생성
            payload = {"fileName": blob_name, "containerName": container_name}
            r = s.post(sas_url_endpoint, json=payload, headers=headers_json, timeout=timeout_sec)
            r.raise_for_status()
            sas_info = r.json()
            sas_url = sas_info["sasUrl"]

            # 3) 다운로드
            ext = os.path.splitext(blob_name)[1].lower() or ".jpg"
            dst = os.path.join(download_dir, f"{cat_name}_{file_idx}{ext}")

            with s.get(sas_url, stream=True, timeout=timeout_sec) as dl:
                dl.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in dl.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            file_idx += 1

    print(f"[저장완료] 고양이명: {cat_name} / 개수: {file_idx - 1}")
    return file_idx - 1


def bbox_predict_and_points(model, # best.pt (yolo(0)모델)
                            base_dir, # /app/tmp
                            img_dir, # origin/your_user_id
                            project_name, # bbox_predict/your_user_id
                            conf, # 0.5
                            iou): # 0.7
    """
    For YOLO모델(2번) : 사용자 업로드 사진 bbox 추론 및 좌표 얻기
    """

    img_path = os.path.join(base_dir, img_dir) # /app/tmp/origin/your_user_id
    project_path = os.path.join(base_dir, project_name, f"conf{conf}_iou{iou}") # /app/tmp/bbox_predict/your_user_id/conf0.5_iou0.7
    # print(img_path)
    results = []
    origin_img_paths = []
    class_names = []
    for file in os.listdir(img_path):
        img_source = os.path.join(img_path, file)
        result = model.predict(source=img_source,
                                save=True,
                                project=project_path,
                                name="",
                                exist_ok=True,
                                conf=conf,
                                iou=iou)
        results.append(result[0])
        origin_img_paths.append(img_source)
        class_names.append(file.split('_')[0])

    print(f"[bbox 추론 완료] results: {len(results)}개 / origin_img_paths: {len(origin_img_paths)}개 / class_names: {len(class_names)}개")

    name_and_bbox = [] # {origin_img_path:"/app/tmp/origin/your_user_id/기쁨_1.jpg" ,class:"기쁨", xyxy:[]}

    for path, class_name, result in zip(origin_img_paths, class_names, results):
        item = {
            'origin_img_path' : path,
            'class' : class_name,
            'xyxy' : result.boxes.xyxy.tolist()
        }
        name_and_bbox.append(item)

    return name_and_bbox


# def customer_upload_image_download_via_api(
#     api_base_url: str,            # 예: "http://localhost:8000"
#     x_api_key: str,                 # Depends(get_api_key)에서 요구하는 값
#     container_name: str,          # 예: "origin"
#     blob_prefix: str,             # 예: "your_user_id"
#     cat_name: str,                # 예: "나비"
#     download_dir: str,            # 예: "BASE_URL/tmp/origin/your_user_id/"
#     exts: tuple = (".jpg", ".jpeg", ".png"),
#     timeout_sec: int = 60,
# ):


def download_datasets_via_api(api_base_url,   # 'http://collectionservice:8000'
                              x_api_key,      # Depends(get_api_key)에서 요구하는 값
                              container_name, # 'datasets'
                              blob_name,      # 'your_user_id'
                              download_dir,   # "BASE_URL/tmp/datasets/your_user_id"
                              timeout_sec):   # 14400
    """
    Azure Blob Storage에 'container_name/blob_name' (예: datasets/your_user_id) 존재 여부를 확인하고,
    있으면 prefix 하위 전체 파일을 로컬로 동기화, 없으면 Roboflow 포맷 기본 디렉토리를 생성한다.

    작업 순서
    1) /api/blobs/list 로 prefix별 blob 목록을 가져온다.
    2) 목록이 있으면 /api/sas/generate 로 파일별 SAS URL을 받아 GET으로 다운로드한다.
       (원격 상대경로를 보존하여 download_dir/ 아래에 동일한 구조로 저장)
    3) 목록이 없으면 download_dir/train|valid 의 images, labels 디렉토리를 생성한다.
    """

    # 로컬 루트 보장
    os.makedirs(download_dir, exist_ok=True)

    # 엔드포인트
    list_url = urljoin(api_base_url.rstrip("/") + "/", "api/blobs/list")
    sas_url_endpoint = urljoin(api_base_url.rstrip("/") + "/", "api/sas/generate")

    headers_json = {
        "X-API-Key": x_api_key,
        "Content-Type": "application/json",
    }
    headers_get = {"X-API-Key": x_api_key}

    with requests.Session() as s:
        prefix = f"{blob_name.rstrip('/')}/"

        # 1) 목록 조회
        resp = s.get(
            list_url,
            params={"container": container_name, "prefix": prefix},
            headers=headers_get,
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        blobs = resp.json().get("blobs", []) or []
    
        # 3) 존재하지 않으면 로컬 디렉토리만 생성
        if not blobs:
            base_dirs = [
                os.path.join(download_dir, "train", "images"),
                os.path.join(download_dir, "train", "labels"),
                os.path.join(download_dir, "valid", "images"),
                os.path.join(download_dir, "valid", "labels"),
            ]
            for d in base_dirs:
                os.makedirs(d, exist_ok=True)
            print("[생성완료] 원격에 prefix가 없어 로컬 기본 디렉토리만 생성")
            return 0
        
        # 2) 존재하면 전체 파일 다운로드 (상대경로 유지)
        file_list = [b for b in blobs if not b.endswith("/")]
        total_files = len(file_list)
        file_cnt = 0

        # 전체 진행률 tqdm
        with tqdm(total=total_files, unit="file", desc="Downloading Azure BloB Storage-dataset") as pbar_all:
            for remote_blob in file_list:
                if remote_blob.startswith(prefix):
                    rel_path = remote_blob[len(prefix):]
                else:
                    continue

                local_path = os.path.join(download_dir, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                # SAS URL 생성
                payload = {"fileName": remote_blob, "containerName": container_name}
                r = s.post(sas_url_endpoint, json=payload, headers=headers_json, timeout=timeout_sec)
                r.raise_for_status()
                sas_url = r.json().get("sasUrl")
                if not sas_url:
                    print(f"[경고] SAS URL 생성 실패: {remote_blob}")
                    continue

                # 파일 다운로드 (개별 tqdm, 상대경로 표시)
                with s.get(sas_url, stream=True, timeout=timeout_sec) as dl:
                    dl.raise_for_status()
                    total_size = int(dl.headers.get("Content-Length", 0))
                    chunk_size = 1024 * 1024  # 1MB

                    file_bar = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"{rel_path}",
                        leave=False
                    )

                    with open(local_path, "wb") as f:
                        for chunk in dl.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                file_bar.update(len(chunk))

                    file_bar.close()

                file_cnt += 1
                pbar_all.update(1)

    print(f"[저장완료] prefix: {blob_name} / 파일 개수: {file_cnt} / 로컬 경로: {download_dir}")
    return file_cnt