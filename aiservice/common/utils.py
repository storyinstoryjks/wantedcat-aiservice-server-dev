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
    1) /api/blobs/list 로 prefix별 blob 목록을 가져온다.
    2) /api/sas/generate 로 파일별 SAS URL을 받아 GET으로 다운로드한다.
    3) download_dir/name_{idx}.<ext> 형식으로 저장합니다.

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