from flask import Flask, request, jsonify
import os, logging, datetime
from common.utils import *
from common.data_processing import *
from common.learning import *
from dotenv import load_dotenv
from ultralytics import YOLO
from threading import Lock


# --- 전역 설정(앱, 경로, ...) ---
load_dotenv()
_model_lock = Lock()

def ensure_yolo0_loaded():
    global YOLO0_BEST_MODEL
    if YOLO0_BEST_MODEL is None:
        with _model_lock:
            if YOLO0_BEST_MODEL is None:
                path = download_blob_via_api(
                    api_base_url="http://collectionservice:8000",
                    x_api_key=X_API_KEY,
                    container_name="bestmodel",
                    blob_path="yolo0/best.pt",
                    download_dir=os.path.join(BASE_URL, "best_model/yolo0"),
                    timeout_sec=14400
                )
                YOLO0_BEST_MODEL = YOLO(path)
                print("YOLO0 준비 완료")

app = Flask(__name__)
BASE_URL = app.root_path
X_API_KEY = os.getenv("X_API_KEY")
YOLO0_BEST_MODEL = None


# --- API 엔드포인트 정의 ---
@app.route('/api/aiservice/prepare/upload', methods=['POST'])
def prepare_upload_image():
    """
    사용자 업로드 이미지 로컬에 저장
    """

    ### 1. (React->수집서버)에서 고양이 프로필 정보 받기 ###
    data = request.get_json()
    cat_name=data.get('cat_name') # 단일 고양이 이름
    user_id=data.get('user_id') # 고객id

    ### 2. 사용자 업로드 사진을 다운로드 받기 ###
    cat_cnt = customer_upload_image_download_via_api(api_base_url="http://collectionservice:8000",
                                           x_api_key=X_API_KEY,
                                           container_name='origin',
                                           blob_prefix=user_id,
                                           cat_name=cat_name,
                                           download_dir=os.path.join(BASE_URL,f'tmp/origin/{user_id}'),
                                           exts=(".jpg", ".jpeg", ".png"),
                                           timeout_sec=14400 )# Blob storage에서 다운로드 받아, 해당 폴더에 저장
    
    ### 3. 수집서버에게 리턴 ###
    return jsonify(status_code=200, message=f"{user_id} : ['{cat_name}'고양이] 사용자 업로드 이미지 저장 완료. (최종 x)"),200


@app.route('/api/aiservice/prepare/model', methods=['POST'])
def prepare_yolo_model():
    """
    고양이 등록 플로우 : yolo모델(2번) 제작 및 blob storage에 저장.
    """

    ###  1. (React->수집서버)에서 훈련 대상 고객id를 받아오기 ###
    data = request.get_json()
    # final_flag=data.get('final_flag')
    # cat_name=data.get('cat_name')
    user_id=data.get('user_id')

    # YOLO(0번) 모델 준비
    ensure_yolo0_loaded()
    if YOLO0_BEST_MODEL==None:
        return jsonify(status_code=500, message="YOLO0모델의 best.pt가 로컬에 없음. Blob Storage에서 다운로드 필요."),500


    ### 2. [1번/2-1번] 사용자 업로드 이미지 bbox 예측 ###
    print("[1번/2-1번] 사용자 업로드 이미지 bbox 예측")
    print("="*100)
    name_and_bbox = bbox_predict_and_points(model=YOLO0_BEST_MODEL,
                                            base_dir=os.path.join(BASE_URL, 'tmp'), 
                                            img_dir=f"origin/{user_id}", 
                                            project_name=f"bbox_predict/{user_id}",
                                            conf=0.5,
                                            iou=0.7)
    print(f"name_and_bbox 구성 확인하기")
    print(name_and_bbox[0])
    print()

    ### 3-1. [2-2번] Azure Blob Storage에 your_user_id에 해당되는 학습데이터셋 다운로드 ### #(테스트 2개씩: 기쁨/흰둥, 은애/이안)
    print("[2-2번] Azure Blob Storage에 your_user_id에 해당되는 학습데이터셋 다운로드")
    print("="*100)
    file_cnt = download_datasets_via_api(api_base_url="http://collectionservice:8000",
                                        x_api_key=X_API_KEY, 
                                        container_name='datasets',
                                        blob_name=user_id, 
                                        download_dir=os.path.join(BASE_URL, f"tmp/datasets/{user_id}"),
                                        timeout_sec=14400)
    print()

    ### 3-2. [2-1번+2-2번] 학습데이터셋 구성 : 원본 + 데이터 증강 (로컬 누적) ###
    print("[2-1번+2-2번] YOLO모델(2번)을 위한 학습데이터셋 구성")
    print("="*100)
    # yaml_path, train_data_cnt, valid_data_cnt = config_train_datasets_v1(base_url=BASE_URL,
    #                                                                   user_id=user_id, 
    #                                                                   name_and_bbox=name_and_bbox, 
    #                                                                   ai_datasets_base_path=os.path.join(BASE_URL,f'tmp/datasets/{user_id}'))

    # yaml_path, total_added_train, total_added_valid = config_train_datasets_v2(base_url=BASE_URL, 
    #                                                                            user_id=user_id,     
    #                                                                            name_and_bbox=name_and_bbox, 
    #                                                                            ai_datasets_base_path=os.path.join(BASE_URL,f'tmp/datasets/{user_id}'),
    #                                                                            keep_existing=True, 
    #                                                                            rebalance_append=True, 
    #                                                                            train_ratio=0.7) # v1 + class별 rebalancing
        
    yaml_path, total_added_train, total_added_valid = config_train_datasets_v4(base_url=BASE_URL,        
                                                                                user_id=user_id, 
                                                                                name_and_bbox=name_and_bbox, 
                                                                                ai_datasets_base_path=os.path.join(BASE_URL,f'tmp/datasets/{user_id}'),
                                                                                target_per_class=300,
                                                                                train_ratio=0.7) # v2 + 용량 최적화(aug파일 move to train/valid)
    print(f"추가 Train, Valid = {total_added_train}, {total_added_valid}")
    print("data.yaml 경로:", yaml_path)
    print()
    result_datasets = count_dataset_files_by_class(ai_datasets_base_path=os.path.join(BASE_URL,f'tmp/datasets/{user_id}'),
                                                    mode="image")
    print()

    ### 3-3. [2-3번] Azure Blob Storage에 학습데이터셋 덮어쓰기 (blob storage 누적) ###
    print("[2-3번] 학습데이터셋 Azure Blob Storage에 업로드 (덮어쓰기)")
    uploaded = upload_datasets_via_api(api_base_url="http://collectionservice:8000",
                                        x_api_key=X_API_KEY,
                                        container_name="datasets",
                                        blob_name=f"{user_id}",
                                        local_dir=os.path.join(BASE_URL, f"tmp/datasets/{user_id}"),
                                        timeout_sec=14400,
                                        overwrite=True)
    print()

    ### [중간 폴더 정리] ###
    # /app/tmp/bbox_predict/your_user_id/conf0.5_iou_0.7/predict
    target = os.path.join(BASE_URL, f"tmp/bbox_predict/{user_id}")
    purge_directory(target, safety_prefix=os.path.join(BASE_URL, "tmp/bbox_predict")) # 경로 직접 지정해서 비우기(폴더는 유지)
    print()

    # /app/tmp/origin/your_user_id
    target = os.path.join(BASE_URL, f"tmp/origin/{user_id}")
    purge_directory(target, safety_prefix=os.path.join(BASE_URL, "tmp/origin"))

    ### 4. [2-4번] corrupt JPEG 복구 및 yolo모델(2번) 학습 ###
    # corrupt JPEG 복구
    print("[2-4번] corrupt JPEG 복구")
    print("="*100)
    recovery_corrupt_jpeg(base_url=BASE_URL, # /app
                          img_dir=f"tmp/datasets/{user_id}") # tmp/datasets/your_user_id 
    print()

    # YOLO모델(2번) 학습 : Hybrid -> 실험 결과 Best HyperParameter로 바로 학습하도록 변경 (yolo11s, epoch=35, lr=0.00725)
    print("[2-4번] YOLO모델(2번) 학습 시작")
    print("="*100)
    path = download_blob_via_api(
                    api_base_url="http://collectionservice:8000",
                    x_api_key=X_API_KEY,
                    container_name="font",
                    blob_path="NanumSquareR.ttf",
                    download_dir=os.path.join(BASE_URL, "font"),
                    timeout_sec=14400
                )
    print("NanumSquareR.ttf 폰트 다운로드 완료:")
    print()
    
    result = GridSearch_YOLO(epochs=[1], # cpu 8코어 테스트용 (실전: 35)
                             lr0s=[0.00725],
                             models=['yolo11s'],
                             base_dir=os.path.join(BASE_URL, 'tmp'), 
                             dataset_name=f"datasets/{user_id}", 
                             train_output_name=f'yolo2_train/{user_id}') 
    print(f"학습결과")
    print(result)
    print()

    ### 5. [3번] YOLO모델(2번) 베스트 모델을 Azure Blob Storage에 저장 ###
    print("[3번] YOLO모델(2번) 베스트 모델을 Azure Blob Storage에 저장")
    print("="*100)
    uploaded_flag = upload_bestpt_via_api(api_base_url="http://collectionservice:8000",
                                     x_api_key=X_API_KEY,
                                     result_dict=result, 
                                     user_id=user_id,  
                                     model_family="yolo2",  
                                     container_name="bestmodel",
                                     timeout_sec=14400,
                                     overwrite=True)

    ### [최종 폴더 정리] ###
    # /app/tmp/datasets/your_user_id
    target = os.path.join(BASE_URL, f"tmp/datasets/{user_id}")
    purge_directory(target, safety_prefix=os.path.join(BASE_URL, "tmp/datasets"))

    # /app/tmp/yolo2_train/your_user_id/gridsearch
    target = os.path.join(BASE_URL, f"tmp/yolo2_train/{user_id}")
    purge_directory(target, safety_prefix=os.path.join(BASE_URL, "tmp/yolo2_train"))


    return jsonify(
        status_code=200, 
        message=f"{user_id} : 학습데이터셋 구성 및 YOLO모델(2번) 준비 완료",
        user_id=user_id,
        result=result # 이 내용을 수집서버가 DB에 저장
    ),200



@app.route('/api/aiservice/upload/bbox', methods=['POST'])
def make_and_upload_bbox_video():
    """
    실제 환경 플로우 : 홈캠 영상 -> bbox 영상 만들기
    """
    # 1. 수집서버에서 특정 고양이의 식사 이벤트 정보를 넘겨받는다.
    data = request.get_json()

    user_id = data.get('user_id')
    video_url = data.get('video_url')
    event_time = datetime.datetime.fromisoformat(data.get('event_time'))
    weight_info = data.get('weight_info')
    duration_seconds = data.get('duration_seconds')

    print(f"User ID: {user_id}")
    print(f"Video URL: {video_url}")
    print(f"Event Time: {event_time}")
    print(f"Weight Info: {weight_info}")
    print(f"Duration (seconds): {duration_seconds}")

    return "OK",200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)