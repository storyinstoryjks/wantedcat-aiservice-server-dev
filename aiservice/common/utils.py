from ultralytics import settings, YOLO
from roboflow import Roboflow
import os, json, shutil, itertools, csv
import time
from azure.storage.blob import BlobServiceClient
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import optuna
import albumentations as A
from collections import Counter, defaultdict
import random
import itertools
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile
from urllib.parse import urlparse, urljoin, unquote
import requests
from tqdm import tqdm
import mimetypes
from glob import glob


def use_korean_font(ttf_path: str = "/app/font/NanumSquareR.ttf"):
    """
    주어진 TTF를 Matplotlib 기본 폰트로 등록/사용한다.
    - 한글 글리프 경고 제거
    - 음수 기호 깨짐 방지(axes.unicode_minus=False)
    """
    if not os.path.isfile(ttf_path):
        print(f"[FONT][WARN] Not found: {ttf_path}")
        return

    try:
        # 폰트 등록
        fm.fontManager.addfont(ttf_path)
        # 내부 폰트 패밀리명 추출 (예: 'NanumSquare')
        font_name = fm.FontProperties(fname=ttf_path).get_name()

        # 전역 기본 폰트 설정
        plt.rcParams["font.family"] = font_name
        plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

        print(f"[FONT][OK] Using font '{font_name}' from {ttf_path}")
    except Exception as e:
        print(f"[FONT][ERR] Failed to load {ttf_path}: {e}")


def download_blob_via_api(api_base_url: str,
                          x_api_key: str,
                          container_name: str,
                          blob_path: str,           # 예: "yolo0/best.pt"
                          download_dir: str,        # 예: "/app/best_model/yolo0"
                          timeout_sec: int = 14400) -> str:
    """
    Azure Blob Storage의 {container_name}/{blob_path}를 download_dir로 다운로드한다.
    반환값: 로컬 저장 경로
    """
    os.makedirs(download_dir, exist_ok=True)

    # 엔드포인트
    api_base = api_base_url.rstrip("/") + "/"
    sas_url_endpoint = urljoin(api_base, "api/sas/generate")
    headers_json = {"X-API-Key": x_api_key, "Content-Type": "application/json"}

    with requests.Session() as s:
        # 1) 파일별 SAS URL 생성
        payload = {"fileName": blob_path, "containerName": container_name, "permission": "r"}
        r = s.post(sas_url_endpoint, json=payload, headers=headers_json, timeout=timeout_sec)
        r.raise_for_status()
        sas_url = r.json().get("sasUrl")
        if not sas_url:
            raise RuntimeError(f"SAS URL 생성 실패: {container_name}/{blob_path}")

        # 2) 다운로드 (스트리밍)
        with s.get(sas_url, stream=True, timeout=timeout_sec) as dl:
            dl.raise_for_status()
            total_size = int(dl.headers.get("Content-Length", 0))
            chunk_size = 1024 * 1024  # 1MB

            local_path = os.path.join(download_dir, os.path.basename(blob_path))
            with tqdm(total=total_size if total_size > 0 else None,
                      unit="B", unit_scale=True, unit_divisor=1024,
                      desc=os.path.basename(blob_path), leave=True) as pbar, \
                 open(local_path, "wb") as f:
                for chunk in dl.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    print(f"[저장완료] {container_name}/{blob_path} -> {local_path}")
    return local_path


import os
import requests
from urllib.parse import urlparse, urljoin, unquote
from tqdm import tqdm

def download_video_by_url(api_base_url: str, # http://collectionservice:8000
                          x_api_key: str, 
                          video_url: str,
                          download_root: str = "/app/videos",
                          subdir: str | None = None,
                          timeout_sec: int = 14400) -> str:
    """
    전체 Blob URL(예: https://<account>.blob.core.windows.net/<container>/<blob_path>)
    을 받아, SAS 생성 API로 읽기 권한 URL을 발급받아 로컬에 스트리밍 다운로드
    """
    # 1) URL 파싱하여 container / blob_path 추출
    parsed = urlparse(video_url)
    # path 예: "/video/your_user_id_af556a...a2c.mp4" 또는 "/video/folder/a.mp4"
    # split 후 첫 요소는 빈 문자열이므로 제외
    path_parts = [p for p in parsed.path.split('/') if p]
    if len(path_parts) < 2:
        raise ValueError(f"URL에서 container/blob_path를 파싱하지 못했습니다: {video_url}")

    container_name = path_parts[0]
    blob_path = "/".join(path_parts[1:])
    # URL 인코딩된 경로 복원
    blob_path = unquote(blob_path)
    file_name = os.path.basename(blob_path)

    # 2) 저장 디렉터리 결정
    # subdir 미지정 시: 파일명에서 첫 '_' 앞부분을 폴더명으로 사용 (예: your_user_id_abc.mp4 -> your_user_id)
    if subdir is None:
        if "_" in file_name:
            subdir = file_name.split("_", 1)[0]
        else:
            # '_'가 없으면 확장자 없는 파일명 사용
            subdir = os.path.splitext(file_name)[0]

    download_dir = os.path.join(download_root, subdir)
    os.makedirs(download_dir, exist_ok=True)

    # 3) SAS 생성 엔드포인트 구성
    api_base = api_base_url.rstrip("/") + "/"
    sas_url_endpoint = urljoin(api_base, "api/sas/generate")
    headers_json = {"X-API-Key": x_api_key, "Content-Type": "application/json"}

    with requests.Session() as s:
        # 원래 URL에 이미 SAS 쿼리가 있더라도, 보안/통일성을 위해 API로 새로 발급
        payload = {"fileName": blob_path, "containerName": container_name, "permission": "r"}
        r = s.post(sas_url_endpoint, json=payload, headers=headers_json, timeout=timeout_sec)
        r.raise_for_status()
        sas_url = r.json().get("sasUrl")
        if not sas_url:
            raise RuntimeError(f"SAS URL 생성 실패: {container_name}/{blob_path}")

        # 4) 스트리밍 다운로드
        with s.get(sas_url, stream=True, timeout=timeout_sec) as dl:
            dl.raise_for_status()
            total_size = int(dl.headers.get("Content-Length", 0))
            chunk_size = 1024 * 1024  # 1MB

            local_path = os.path.join(download_dir, file_name)
            with tqdm(total=total_size if total_size > 0 else None,
                      unit="B", unit_scale=True, unit_divisor=1024,
                      desc=file_name, leave=True) as pbar, \
                 open(local_path, "wb") as f:
                for chunk in dl.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    print(f"[저장완료] {container_name}/{blob_path} -> {local_path}")
    return local_path



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
                                iou=iou,
                                verbose=False)
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


class TqdmFileReader:
    """requests가 read()로 읽을 때마다 tqdm를 갱신하는 래퍼 (Content-Length는 헤더로 고정)"""
    def __init__(self, f, pbar):
        self.f = f
        self.pbar = pbar
    def read(self, amt=None):
        chunk = self.f.read(amt)
        if chunk:
            self.pbar.update(len(chunk))
        return chunk
    def __getattr__(self, name):
        return getattr(self.f, name)

def upload_datasets_via_api(api_base_url, x_api_key, container_name, blob_name, local_dir,
                            timeout_sec=14400, overwrite=True):
    api_base = api_base_url.rstrip("/") + "/"
    sas_url_endpoint = urljoin(api_base, "api/sas/generate")
    headers_json = {"X-API-Key": x_api_key, "Content-Type": "application/json"}

    # 업로드 대상 수집 (data.yaml + train/** + valid/**)
    targets = []
    yaml_path = os.path.join(local_dir, "data.yaml")
    if os.path.isfile(yaml_path):
        targets.append(("data.yaml", yaml_path))
    for split in ("train", "valid"):
        split_dir = os.path.join(local_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for root, _, files in os.walk(split_dir):
            for fname in files:
                ap = os.path.join(root, fname)
                rp = os.path.relpath(ap, start=local_dir).replace(os.sep, "/")
                targets.append((rp, ap))
    if not targets:
        print(f"[경고] 업로드할 파일이 없습니다: {local_dir}")
        return 0

    # 총 바이트
    total_bytes = sum((os.path.getsize(ap) for _, ap in targets if os.path.exists(ap)), 0)
    uploaded_files = 0

    with requests.Session() as s, tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                                       desc="Uploading Azure Blob Storage-dataset") as pbar_total:
        for rel_path, abs_path in targets:
            remote_blob = f"{blob_name.rstrip('/')}/{rel_path.lstrip('/')}"
            ctype, _ = mimetypes.guess_type(abs_path)
            if ctype is None:
                if abs_path.lower().endswith(".txt"):
                    ctype = "text/plain"
                elif abs_path.lower().endswith((".yaml", ".yml")):
                    ctype = "text/yaml"
                else:
                    ctype = "application/octet-stream"

            # SAS URL 생성 (쓰기)
            payload = {
                "fileName": remote_blob,
                "containerName": container_name,
                "permission": "w",
                "overwrite": bool(overwrite),
            }
            r = s.post(sas_url_endpoint, json=payload, headers=headers_json, timeout=timeout_sec)
            r.raise_for_status()
            sas_url = r.json().get("sasUrl")
            if not sas_url:
                print(f"[경고] SAS URL 생성 실패: {remote_blob}")
                continue

            file_size = os.path.getsize(abs_path)

            # === 중요 변경점 시작 ===
            # 1) Content-Length 명시
            # 2) PUT 헤더에서 커스텀 'X-API-Key' 제거
            # 3) data는 파일 객체(비 generator)로 전달
            put_headers = {
                "x-ms-blob-type": "BlockBlob",
                "Content-Type": ctype,
                "Content-Length": str(file_size),
                # 선택: "x-ms-version": "2021-12-02",  # 버전 강제 지정이 필요할 때만
            }

            with open(abs_path, "rb") as f, \
                 tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
                      desc=rel_path, leave=False) as pbar_file:
                reader = TqdmFileReader(f, pbar_file)  # 진행률 갱신용 래퍼
                resp = s.put(sas_url, data=reader, headers=put_headers, timeout=timeout_sec)
                try:
                    resp.raise_for_status()
                except Exception as e:
                    print(f"[오류] 업로드 실패: {remote_blob} -> {e}")
                    continue
                pbar_total.update(file_size)
            # === 중요 변경점 끝 ===

            uploaded_files += 1

    print(f"[업로드완료] prefix: {blob_name} / 파일 개수: {uploaded_files} / 총 바이트: {total_bytes}")
    return uploaded_files


def upload_bestpt_via_api(api_base_url,
                          x_api_key,
                          result_dict,
                          user_id,
                          model_family="yolo2",
                          container_name="best_model",
                          timeout_sec=14400,
                          overwrite=True):
    """
    result_dict['best_file_path']의 best.pt를
    Azure Blob Storage: best_model/{model_family}/{user_id}/best.pt 로 업로드.

    Returns: 1 (성공 시 업로드 파일 수), 실패 시 예외 발생
    """
    # 0) 로컬 best.pt 확인
    local_best = result_dict.get("best_file_path")
    print(local_best)
    if not local_best or not os.path.isfile(local_best):
        raise FileNotFoundError(f"best_file_path가 없거나 파일이 아님: {local_best}")

    # 1) 원격 blob 경로
    blob_name = f"{model_family.rstrip('/')}/{user_id}/best.pt"

    # 2) 엔드포인트 & 헤더
    api_base = api_base_url.rstrip("/") + "/"
    sas_url_endpoint = urljoin(api_base, "api/sas/generate")
    headers_json = {"X-API-Key": x_api_key, "Content-Type": "application/json"}

    # 3) MIME (pt는 일반적으로 octet-stream)
    ctype, _ = mimetypes.guess_type(local_best)
    if not ctype:
        ctype = "application/octet-stream"

    file_size = os.path.getsize(local_best)

    with requests.Session() as s:
        # 4) 쓰기 SAS URL 생성
        payload = {
            "fileName": blob_name,
            "containerName": container_name,
            "permission": "w",
            "overwrite": bool(overwrite),
        }
        r = s.post(sas_url_endpoint, json=payload, headers=headers_json, timeout=timeout_sec)
        r.raise_for_status()
        sas_url = r.json().get("sasUrl")
        if not sas_url:
            raise RuntimeError(f"SAS URL 생성 실패: {container_name}/{blob_name}")

        # 5) PUT 업로드 (BlockBlob), Content-Length 명시
        put_headers = {
            "x-ms-blob-type": "BlockBlob",
            "Content-Type": ctype,
            "Content-Length": str(file_size),
        }

        with open(local_best, "rb") as f, \
             tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
                  desc=f"uploading {os.path.basename(local_best)} → {container_name}/{blob_name}",
                  leave=True) as pbar:
            reader = TqdmFileReader(f, pbar)
            resp = s.put(sas_url, data=reader, headers=put_headers, timeout=timeout_sec)
            resp.raise_for_status()

    print(f"[업로드완료] {local_best} -> {container_name}/{blob_name} ({file_size:,} bytes)")
    print()
    return 1



def build_predict_dir(user_id: str,
                      base_dir: str = "/app/tmp/bbox_predict",
                      conf: float = 0.5,
                      iou: float = 0.7) -> str:
    """
    예) /app/tmp/bbox_predict/your_user_id/conf0.5_iou_0.7/predict
    """
    return os.path.join(base_dir, user_id, f"conf{conf}_iou_{iou}", "predict")


def purge_directory(path: str,
                    remove_root: bool = False,
                    dry_run: bool = False,
                    safety_prefix: str = "/app/tmp/bbox_predict"):
    """
    path 하위의 모든 파일/폴더를 삭제한다.
    - remove_root=True 이면 path 폴더 자체도 삭제
    - dry_run=True 이면 실제 삭제 대신 삭제 예정 항목만 출력
    - safety_prefix: 안전장치(이 prefix 밖 경로는 삭제 거부)

    Returns: (files_deleted, dirs_deleted, bytes_freed)
    """
    apath = os.path.abspath(path)
    if not apath.startswith(os.path.abspath(safety_prefix)):
        raise ValueError(f"Safety check failed: {apath} is outside {safety_prefix}")

    if not os.path.isdir(apath):
        print(f"[정보] 대상 디렉토리가 없습니다: {apath}")
        return (0, 0, 0)

    files_deleted = 0
    dirs_deleted = 0
    bytes_freed = 0

    # 하위부터 지우기
    for root, dirs, files in os.walk(apath, topdown=False):
        for fname in files:
            fp = os.path.join(root, fname)
            try:
                size = os.path.getsize(fp)
            except OSError:
                size = 0

            if dry_run:
                print("[DRY-RUN] 파일 삭제:", fp)
            else:
                try:
                    os.remove(fp)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"[경고] 파일 삭제 실패: {fp} -> {e}")
                    continue

            bytes_freed += size
            files_deleted += 1

        for dname in dirs:
            dp = os.path.join(root, dname)
            if dry_run:
                print("[DRY-RUN] 폴더 삭제:", dp)
                dirs_deleted += 1
                continue

            try:
                os.rmdir(dp)  # 비어있으면 OK
                dirs_deleted += 1
            except OSError:
                # 잔여물이 있거나 권한 이슈: 강제 삭제 시도
                try:
                    shutil.rmtree(dp, ignore_errors=True)
                    dirs_deleted += 1
                except Exception as e:
                    print(f"[경고] 폴더 삭제 실패: {dp} -> {e}")

    if remove_root:
        if dry_run:
            print("[DRY-RUN] 루트 폴더 삭제:", apath)
        else:
            try:
                os.rmdir(apath)
            except OSError:
                shutil.rmtree(apath, ignore_errors=True)

    print(f"[정리완료] files: {files_deleted}, dirs: {dirs_deleted}, freed: {bytes_freed:,} bytes")
    print()
    return files_deleted, dirs_deleted, bytes_freed


# ImageFile.LOAD_TRUNCATED_IMAGES = True  # 잘린 JPEG 복구에 도움

def recovery_corrupt_jpeg(base_url, # /app
                          img_dir): # tmp/datasets/your_user_id 
    """데이터증강으로 인한 corrupt JPEG 복구 """

    for type_ in ['train', 'valid']:
        image_dir = os.path.join(base_url, img_dir, type_, 'images') # /app/tmp/datasets/your_user_id/train|valid/images
        image_paths = glob(os.path.join(image_dir, "*.jpg"))

        for img_path in tqdm(image_paths):
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")  # RGB 보장
                    img.save(img_path, "JPEG", quality=95, optimize=False) # optimize=True : 저장 용량 최적화 (단점: 메모리 사용 UP)
                                                                           # False는 빠른 처리, True는 저장 용량 감소
            except Exception as e:
                print(f"[ERROR] {img_path} - {e}")



def videos_extractor(output_dir, # /app/frames/origin/your_user_id
                     video_path): # /app/videos/origin/your_user_id/stream_2025-07-29_19-03.mp4
    """
    동영상 프레임별 이미지 추출
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if success:
            frame_filename = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

    cap.release()

    print(f"총 {frame_count}개의 프레임이 저장")


def crop_bboxes_and_save(origin_img_path, bbox_list, save_dir):
    """
    한 장 이미지에 그려진 bbox 좌표별로 crop해 로컬에 저장
    """
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(origin_img_path)

    frame_name = os.path.splitext(os.path.basename(origin_img_path))[0]

    cropped_paths = []
    for idx, (x1, y1, x2, y2) in enumerate(bbox_list):
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        crop_path = os.path.join(save_dir, f'{frame_name}_crop_{idx:02d}.jpg')
        cv2.imwrite(crop_path, crop)
        cropped_paths.append(crop_path)

    return cropped_paths


# 화질개선 : 샤프닝 기법 v1
def enhance_image_opencv_v1(image_path, output_path):
    """
    enhance_all_images_v1에서 사용
    """
    img = cv2.imread(image_path)

    # 샤프닝 필터 적용
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, sharpening_kernel)

    # 히스토그램 평활화 (대비 향상) - YUV 공간
    yuv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # 저장
    cv2.imwrite(output_path, enhanced)

def enhance_all_images_v1(input_dir, output_dir):
    """
    화질개선 : 샤프닝 기법 v1
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            enhance_image_opencv_v1(in_path, out_path)