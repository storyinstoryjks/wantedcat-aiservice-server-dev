from flask import Flask, request, jsonify
import os, logging, datetime
from common.utils import *
from dotenv import load_dotenv


# --- 전역 설정(앱, 경로, ...) ---
load_dotenv()

app = Flask(__name__)
BASE_URL = app.root_path
X_API_KEY = os.getenv("X_API_KEY")


@app.route('/hello', methods=['GET'])
def hello():
    print("[INFO 테스트] /hello")
    return "OK", 200

# --- API 엔드포인트 정의 ---
@app.route('/api/aiservice/prepare', methods=['POST'])
def prepare_yolo_model():
    """
    고양이 등록 플로우 : yolo모델(2번) 제작 및 blob storage에 저장.
    """
    ###  1. (React->Spring-Boot)에서 고양이 프로필 정보 받기 ###
    data = request.get_json()
    final_flag=data.get('final_flag')
    cat_name=data.get('cat_name')
    user_id=data.get('user_id')

    # 2-1. final_flag=='true' : 모든 고양이 등록 완료라는 의미
    if final_flag=='true':
        # 3. [1번/2-1번] 사용자 업로드 이미지 bbox 예측
        # 4. [2-2번/2-3번] 학습데이터셋 구성 : 원본 + 데이터 증강 (blob storage 누적)
        # 5. [2-4번] corrupt JPEG 복구 및 yolo모델(2번) 학습
        # 6. [3번] YOLO모델(2번) 베스트 모델을 Azure Blob Storage에 저장
        return {
            status_code:200,
            message:f"{user_id} : 학습데이터셋 구성 및 yolo모델(2번) 준비 완료"
        }

    # 2-2. 아직 고양이 등록 중이라면(final_flag=='false'), 사용자 업로드 사진만 우선 로컬에 저장시켜둠.
    cat_cnt = customer_upload_image_download_via_api(api_base_url="http://collectionservice:8000",
                                           x_api_key=X_API_KEY,
                                           container_name='origin',
                                           blob_prefix=user_id,
                                           cat_name=cat_name,
                                           download_dir=os.path.join(BASE_URL,f'tmp/origin/{user_id}'),
                                           exts=(".jpg", ".jpeg", ".png"),
                                           timeout_sec=300 )# Blob storage에서 다운로드 받아, 해당 폴더에 저장
    return jsonify(status_code=200, 
                   message=f"{user_id} : ['{cat_name}'고양이] 사용자 업로드 이미지 저장 완료. (최종 x)")


@app.route('/api/aiservice/upload', methods=['POST'])
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