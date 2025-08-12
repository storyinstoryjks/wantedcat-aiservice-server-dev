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
import re


# =========================
# Utilities
# =========================
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _unique_path(path):
    base, ext = os.path.splitext(path)
    i, new_path = 1, path
    while os.path.exists(new_path):
        new_path = f"{base}_{i}{ext}"
        i += 1
    return new_path

def _link_then_copy(src, dst, overwrite=False):
    """원본 이미지는 하드링크 시도 → 실패 시 copy2로 폴백. 충돌 시 기본은 유니크 저장."""
    _ensure_dir(os.path.dirname(dst))
    if os.path.abspath(src) == os.path.abspath(dst):
        return "same"
    if os.path.exists(dst):
        try:
            if os.path.samefile(src, dst):
                return "same"
        except FileNotFoundError:
            pass
        if overwrite:
            os.remove(dst)
        else:
            dst = _unique_path(dst)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"

def _save_yolo_label(label_path, bboxes_xyxy, cls_id, img_w, img_h):
    _ensure_dir(os.path.dirname(label_path))
    with open(label_path, 'w', encoding='utf-8') as f:
        for x1, y1, x2, y2 in bboxes_xyxy:
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def _count_class_from_dir(lbl_dir, id2name):
    counts = defaultdict(int)
    if not os.path.isdir(lbl_dir):
        return counts
    for fn in os.listdir(lbl_dir):
        if not fn.lower().endswith(".txt"):
            continue
        p = os.path.join(lbl_dir, fn)
        try:
            with open(p, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                if not line:
                    continue
                cid = int(line.split()[0])
                cname = id2name.get(cid)
                if cname is not None:
                    counts[cname] += 1
        except Exception:
            continue
    return counts

def _build_albu_transform():
    return A.Compose([
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

# =========================
# Aug direct-write helpers (이름: aug_{원본명}_{seq:int}.jpg)
# =========================
def _next_aug_seq_by_base(dst_dir, base):
    """
    dst_dir 내에서 aug_{base}_{NNN}.jpg 의 다음 NNN을 반환 (없으면 1)
    """
    prefix = f"aug_{base}_"
    if not os.path.isdir(dst_dir):
        return 1
    max_seq = 0
    for fn in os.listdir(dst_dir):
        if not fn.startswith(prefix):
            continue
        m = re.match(rf"^{re.escape(prefix)}(\d+)\.jpg$", fn)
        if m:
            max_seq = max(max_seq, int(m.group(1)))
    return max_seq + 1

def _write_aug_by_bbox(item, bbox_idx, cls_name, cls_id, dst_img_dir, dst_lbl_dir, transform):
    """
    item: {"origin_img_path": ..., "class": ..., "xyxy": [[x1,y1,x2,y2], ...]}
    bbox_idx: 원본의 bbox 좌표 인덱스(int)
    저장 파일명: aug_{원본명}_{seq:int}.jpg  (seq는 동일 원본명 기준으로만 증가)
    """
    img = cv2.imread(item["origin_img_path"])
    if img is None:
        return False

    bboxes = item["xyxy"]
    if not (0 <= bbox_idx < len(bboxes)):
        return False

    bbox = bboxes[bbox_idx]
    t = transform(image=img, bboxes=[bbox], class_labels=[cls_name])
    if not t["bboxes"]:
        return False

    aug_img = t["image"]
    ah, aw = aug_img.shape[:2]

    base = os.path.splitext(os.path.basename(item["origin_img_path"]))[0]
    _ensure_dir(dst_img_dir)
    seq = _next_aug_seq_by_base(dst_img_dir, base)
    dst_img = os.path.join(dst_img_dir, f"aug_{base}_{seq}.jpg")

    cv2.imwrite(dst_img, aug_img)

    aug_bboxes = [[x1, y1, x2, y2] for x1, y1, x2, y2 in t["bboxes"]]
    _ensure_dir(dst_lbl_dir)
    dst_lbl = os.path.join(dst_lbl_dir, os.path.splitext(os.path.basename(dst_img))[0] + ".txt")
    _save_yolo_label(dst_lbl, aug_bboxes, cls_id, aw, ah)
    return True

# =========================
# Main: Direct write + 7:3 rebalance
# =========================
def config_train_datasets_v4(
    base_url,                 # "/app"
    user_id,                  # "your_user_id"
    name_and_bbox,            # 신규 데이터(원본) 리스트 [{origin_img_path, class, xyxy}, ...]
    ai_datasets_base_path,    # "/app/tmp/datasets/your_user_id"
    target_per_class=300,
    train_ratio=0.7
):
    """
    - 증강을 임시폴더 없이 곧바로 train/valid 목적지에 '직접 저장'
    - 클래스별 최종 개수(원본 unique + 증강)가 target_per_class가 되도록 증강 개수 산정
    - 기존 데이터는 그대로 두고, 전체(기존+신규) 기준으로 train:valid = 7:3 되도록 신규를 배치
    - data.yaml(names/nc) 갱신
    - 증강 파일명: aug_{원본명}_{seq:int}.jpg (동일 원본명 내에서 번호 증가)
    """
    # 목적지 폴더
    train_img_dir = os.path.join(ai_datasets_base_path, "train", "images")
    train_lbl_dir = os.path.join(ai_datasets_base_path, "train", "labels")
    valid_img_dir = os.path.join(ai_datasets_base_path, "valid", "images")
    valid_lbl_dir = os.path.join(ai_datasets_base_path, "valid", "labels")
    # for d in [train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir]:
    #     _ensure_dir(d)

    # 기존 data.yaml
    yaml_path = os.path.join(ai_datasets_base_path, 'data.yaml')
    existing_names = []
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                existing = yaml.safe_load(f) or {}
                if isinstance(existing.get("names", []), list):
                    existing_names = existing["names"]
        except Exception:
            existing_names = []

    # 신규 클래스 그룹
    class_to_items = defaultdict(list)
    for it in name_and_bbox:
        if it.get("xyxy"):
            class_to_items[it["class"]].append(it)
    new_class_names = sorted(class_to_items.keys())

    # names 병합
    seen, merged_names = set(), []
    for nm in existing_names + new_class_names:
        if nm not in seen:
            merged_names.append(nm); seen.add(nm)
    class_to_id = {c: i for i, c in enumerate(merged_names)}
    id2name = {i: n for i, n in enumerate(merged_names)}  # id->name

    # 기존 분포 스캔
    current_train = _count_class_from_dir(train_lbl_dir, id2name)
    current_valid = _count_class_from_dir(valid_lbl_dir, id2name)

    transform = _build_albu_transform()
    total_added_train = total_added_valid = 0

    for cls in new_class_names:
        items = class_to_items[cls]

        # 원본 중복 제거(파일 경로 기준)
        uniq = {}
        for it in items:
            uniq[it["origin_img_path"]] = it
        uniq_list = list(uniq.values())

        # 최종 개수 목표: original_count + need_aug == target_per_class
        original_count = len(uniq_list)
        need_aug = max(target_per_class - original_count, 0)

        # bbox 소스 풀(증강 1장 당 1개 bbox 사용)
        bbox_sources = []
        for it in uniq_list:
            for idx in range(len(it["xyxy"])):
                bbox_sources.append((it, idx))
        if not bbox_sources and need_aug > 0:
            need_aug = 0  # bbox 전무 → 증강 불가

        # need_aug 장을 만들기 위해 bbox_sources 라운드로빈
        aug_plan = []
        if need_aug > 0:
            i = 0
            for _ in range(need_aug):
                it, bidx = bbox_sources[i % len(bbox_sources)]
                aug_plan.append((it, bidx))
                i += 1

        # 이번 실행에서 생성될 신규 샘플 수
        n_new = original_count + len(aug_plan)

        # 최종 7:3 계산
        cur_tr = current_train.get(cls, 0)
        cur_va = current_valid.get(cls, 0)
        cur_tot = cur_tr + cur_va
        target_tr_after = int((cur_tot + n_new) * train_ratio + 1e-9)
        need_tr = max(0, min(n_new, target_tr_after - cur_tr))
        need_va = n_new - need_tr

        # 실행 플랜: 원본(uniq_list) + 증강(aug_plan)
        plan = [("orig", it) for it in uniq_list] + [("aug", it, bidx) for it, bidx in aug_plan]
        random.shuffle(plan)
        train_plan = plan[:need_tr]
        valid_plan = plan[need_tr:]

        # ---- 쓰기: train ----
        for item in train_plan:
            if item[0] == "orig":
                it = item[1]
                base = os.path.splitext(os.path.basename(it["origin_img_path"]))[0]
                dst_img = os.path.join(train_img_dir, base + ".jpg")
                if os.path.exists(dst_img):
                    dst_img = _unique_path(dst_img)
                _link_then_copy(it["origin_img_path"], dst_img, overwrite=False)

                img = cv2.imread(dst_img)
                if img is None:
                    continue
                h, w = img.shape[:2]
                dst_lbl = os.path.join(train_lbl_dir, os.path.splitext(os.path.basename(dst_img))[0] + ".txt")
                _save_yolo_label(dst_lbl, it["xyxy"], class_to_id[cls], w, h)
                total_added_train += 1
            else:
                _, it, bidx = item
                ok = _write_aug_by_bbox(
                    it, bidx, cls, class_to_id[cls],
                    dst_img_dir=train_img_dir,
                    dst_lbl_dir=train_lbl_dir,
                    transform=transform
                )
                if ok:
                    total_added_train += 1

        # ---- 쓰기: valid ----
        for item in valid_plan:
            if item[0] == "orig":
                it = item[1]
                base = os.path.splitext(os.path.basename(it["origin_img_path"]))[0]
                dst_img = os.path.join(valid_img_dir, base + ".jpg")
                if os.path.exists(dst_img):
                    dst_img = _unique_path(dst_img)
                _link_then_copy(it["origin_img_path"], dst_img, overwrite=False)

                img = cv2.imread(dst_img)
                if img is None:
                    continue
                h, w = img.shape[:2]
                dst_lbl = os.path.join(valid_lbl_dir, os.path.splitext(os.path.basename(dst_img))[0] + ".txt")
                _save_yolo_label(dst_lbl, it["xyxy"], class_to_id[cls], w, h)
                total_added_valid += 1
            else:
                _, it, bidx = item
                ok = _write_aug_by_bbox(
                    it, bidx, cls, class_to_id[cls],
                    dst_img_dir=valid_img_dir,
                    dst_lbl_dir=valid_lbl_dir,
                    transform=transform
                )
                if ok:
                    total_added_valid += 1

        # 다음 클래스 계산에 반영
        current_train[cls] = cur_tr + need_tr
        current_valid[cls] = cur_va + need_va

    # data.yaml 갱신
    new_yaml = {
        "train": train_img_dir,
        "val":   valid_img_dir,
        "names": merged_names,
        "nc":    len(merged_names),
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(new_yaml, f, sort_keys=False, allow_unicode=True)

    return yaml_path, total_added_train, total_added_valid



def count_dataset_files_by_class(ai_datasets_base_path: str, mode: str = "image"):
    """
    YOLO 데이터셋 경로에서
    - 전체 파일 개수(train/valid images|labels)
    - 클래스별 개수(train/valid 각각 및 합계)
    를 계산/출력한다.

    mode:
      - "image": 이미지 단위 카운트 (라벨에 해당 클래스가 하나라도 있으면 1로 셈)
      - "object": 객체 단위 카운트 (라벨 라인 수를 그대로 합산)
    """
    assert mode in ("image", "object"), "mode는 'image' 또는 'object'만 가능."

    train_img_dir = os.path.join(ai_datasets_base_path, "train", "images")
    train_lbl_dir = os.path.join(ai_datasets_base_path, "train", "labels")
    valid_img_dir = os.path.join(ai_datasets_base_path, "valid", "images")
    valid_lbl_dir = os.path.join(ai_datasets_base_path, "valid", "labels")
    yaml_path = os.path.join(ai_datasets_base_path, "data.yaml")

    # 전체 파일 개수
    def count_files(p):
        return len([f for f in os.listdir(p)]) if os.path.isdir(p) else 0

    totals = {
        "train_images": count_files(train_img_dir),
        "train_labels": count_files(train_lbl_dir),
        "valid_images": count_files(valid_img_dir),
        "valid_labels": count_files(valid_lbl_dir),
    }
    totals["all_files"] = sum(totals.values())

    # 클래스 매핑 로드
    id2name = {}
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                names = data.get("names", [])
                if isinstance(names, list):
                    for i, nm in enumerate(names):
                        id2name[i] = str(nm)
        except Exception:
            pass
    # id2name이 비어있어도 동작은 하게(미상 클래스는 "cls_{id}")
    def id_to_name(cid):
        return id2name.get(cid, f"cls_{cid}")

    # 라벨 파싱
    def parse_label_file(path):
        """각 라벨 파일에서 클래스 ID 목록(모든 라인)을 반환"""
        ids = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cid = int(parts[0])
                    ids.append(cid)
        except Exception:
            pass
        return ids

    def count_by_class_in_dir(lbl_dir):
        """dir 내부 라벨 파일들을 읽어 클래스별 카운트 반환"""
        counts = {}
        if not os.path.isdir(lbl_dir):
            return counts
        for fn in os.listdir(lbl_dir):
            if not fn.lower().endswith(".txt"):
                continue
            cids = parse_label_file(os.path.join(lbl_dir, fn))
            if not cids:
                continue

            if mode == "image":
                # 이미지 단위: 파일 내 등장한 클래스들을 중복 제거 후 1씩 증가
                for cid in set(cids):
                    cname = id_to_name(cid)
                    counts[cname] = counts.get(cname, 0) + 1
            else:
                # 객체 단위: 등장한 라인 수만큼 증가
                for cid in cids:
                    cname = id_to_name(cid)
                    counts[cname] = counts.get(cname, 0) + 1
        return counts

    train_class_counts = count_by_class_in_dir(train_lbl_dir)
    valid_class_counts = count_by_class_in_dir(valid_lbl_dir)

    # 합계
    all_class_names = sorted(set(train_class_counts.keys()) | set(valid_class_counts.keys()))
    total_class_counts = {
        cname: train_class_counts.get(cname, 0) + valid_class_counts.get(cname, 0)
        for cname in all_class_names
    }

    # 출력
    print("=== 전체 파일 개수 ===")
    print(f"Train Images : {totals['train_images']}")
    print(f"Train Labels : {totals['train_labels']}")
    print(f"Valid Images : {totals['valid_images']}")
    print(f"Valid Labels : {totals['valid_labels']}")
    print(f"총 파일 개수 : {totals['all_files']}")
    print()
    print(f"=== 클래스별 개수 (mode={mode}) ===")
    print("[Train]")
    for cname in all_class_names:
        print(f"  {cname}: {train_class_counts.get(cname, 0)}")
    print("[Valid]")
    for cname in all_class_names:
        print(f"  {cname}: {valid_class_counts.get(cname, 0)}")
    print("[Total]")
    for cname in all_class_names:
        print(f"  {cname}: {total_class_counts.get(cname, 0)}")

    return {
        "totals": totals,
        "per_class": {
            "train": train_class_counts,
            "valid": valid_class_counts,
            "total": total_class_counts,
        }
    }