import os
from datetime import datetime, timedelta
import crud
import models
import schemas
from database import SessionLocal, engine, get_db

from azure.storage.blob import (BlobSasPermissions, BlobServiceClient,
                                  generate_blob_sas)
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.orm import Session
import httpx

# --- 초기 설정 ---
load_dotenv()

# (앱 최초 실행 시) SQLAlchemy 모델을 기반으로 DB 테이블 생성 (이미 있다면 아무 작업 안 함)
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# --- API 키 인증 함수 ---
async def get_api_key(api_key_from_header: str = Security(api_key_header)):
    if api_key_from_header == API_KEY:
        return api_key_from_header
    else:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

# --- API 엔드포인트 정의 ---
@app.post("/api/sas/generate", response_model=schemas.SasResponse, dependencies=[Depends(get_api_key)])
async def generate_sas_url(request: schemas.SasRequest):
    blob_name = request.fileName
    container_name = request.containerName
    
    print(f"SAS 토큰 생성 요청 수신: 컨테이너='{container_name}', 파일명='{blob_name}'")

    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(write=True, read=True), # write:쓰기 권한, read:읽기 권한(이게 있어야 다운로드 가능)
        expiry=datetime.utcnow() + timedelta(minutes=10)
    )

    sas_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}" # 업로드용 주소
    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}" # 다운로드용 주소

    return schemas.SasResponse(sasUrl=sas_url, blobUrl=blob_url)


############################################ AI 관련 ################################################
@app.get("/api/blobs/list")
async def list_blobs(container: str, prefix: str, _: str = Depends(get_api_key)):
    """
    prefix로 특정 container의 blob 리스트를 반환하는 컨트롤러
    ex) origin(container)/고객id(prefix)/기쁨|나비|...
    """
    container_client = blob_service_client.get_container_client(container)
    names = [b.name for b in container_client.list_blobs(name_starts_with=prefix)]
    return {"blobs": names}

@app.post("/api/model", dependencies=[Depends(get_api_key)])
async def prepare_user_yolo_model(req: schemas.ModelRequest, db: Session = Depends(get_db)):
    """
    고객별 YOLO모델(2번) 준비 컨트롤러
    """
    user_id = req.user_id

    # Flask 서버(aiservice 컨테이너)로 POST 요청
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(18000.0)) as client:
            resp = await client.post(
                "http://aiservice:8001/api/aiservice/prepare/model",
                json={"user_id": user_id},                   # 바디
                headers={"Content-Type": "application/json"} # 헤더
            )
            resp.raise_for_status()
            data = resp.json()  # {"status_code": int, "message": str, "result": dict, "user_id": str}

            ai_schemas = schemas.AimodelCreate(
                user_id = user_id,
                model_status = "정상",
                model_name = data['result']['run_name'],
                val_precision = data['result']['val_precision'],
                val_recall = data['result']['val_recall'],
                val_map50 = data['result']['val_map50']
            )

            created_ai = crud.create_aimodel(db=db, ai=ai_schemas)
            print("DB에 성공적으로 저장:", created_ai.__dict__)

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"aiservice의 prepare_yolo_model() 호출 실패: {e}") from e
    
    return {"status":"success", "message":"YOLO모델이 준비되었습니다."}


@app.post("/api/events/bbox/video", dependencies=[Depends(get_api_key)])
async def receive_event_for_bbox_video(event: schemas.EventRequest, db: Session = Depends(get_db)):
    """
    BBOX 영상 제작 컨트롤러
    1. 라즈베리파이(실시간 요청)로부터 이벤트 정보 받기(video_url=특정고양이가식사하는crop된영상주소 by Azure Blob Storage)
    2. AI서버에게 요청 : BBOX 제작
    3. AI서버로부터 BBOX 영상 blob주소 + 고양이라벨 정보를 받음
    4. db의 event 테이블에 레코드 삽입
    """
    
    user_id = event.user_id
    origin_video_url = event.origin_video_url
    bowl_where_cell = event.bowl_where_cell
    cat_name = ""
    bbox_video_url = ""

    # Flask 서버(aiservice 컨테이너)로 POST 요청
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(18000.0)) as client:
            resp = await client.post(
                "http://aiservice:8001/api/aiservice/upload/bbox",
                json={"user_id": user_id, "origin_video_url": origin_video_url, "bowl_where_cell": bowl_where_cell}, # 바디
                headers={"Content-Type": "application/json"} # 헤더
            )
            resp.raise_for_status()
            data = resp.json()  # {"cat_name": str, "bbox_video_url": str}

            cat_name = data['cat_name']
            bbox_video_url = data['bbox_video_url']

            event_schemas = schemas.EventCreate(
                user_id = user_id,
                event_time = event.event_time,
                duration_seconds = event.duration_seconds,
                weight_info = event.weight_info,
                origin_video_url = origin_video_url,
                bbox_video_url = data['bbox_video_url'],
                event_type = event.event_type,
                cat_name = data['cat_name']
            )

            created_event = crud.create_event(db=db, event=event_schemas)
            print("DB에 성공적으로 저장:", created_event.__dict__)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"aiservice의 make_and_upload_bbox_video() 호출 실패: {e}") from e
    

    return {
        "status":"success", "message":"BBOX 영상 완료", "user_id":user_id,
        "event_time":event.event_time, "duration_seconds":event.duration_seconds, "weight_info":event.weight_info,
        "origin_video_url":origin_video_url, "bbox_video_url":bbox_video_url, "event_type":event.event_type,
        "cat_name":cat_name
    }

#####################################################################################################


@app.post("/api/events", dependencies=[Depends(get_api_key)])
async def receive_event_data(
    event: schemas.EventCreate, # 수정된 Pydantic 모델 사용
    db: Session = Depends(get_db) # DB 세션 주입
):
    print("Received event data:", event.dict())

    # CRUD 함수를 호출하여 DB에 이벤트 저장
    created_event = crud.create_event(db=db, event=event)

    print("Successfully saved to DB:", created_event.__dict__)
    
    return {"status": "success", "message": "Event data received and saved."}

@app.get("/api/events/{user_id}", dependencies=[Depends(get_api_key)])
def list_events_by_user(user_id: str, db: Session = Depends(get_db)):
    return crud.get_events_by_user(db, user_id)

@app.post("/api/internal/auth/stream")
def authenticate_stream(auth_request: schemas.StreamAuthRequest, db: Session = Depends(get_db)):
    # MediaMTX는 경로 맨 앞에 '/'를 붙여서 보내므로, 이를 제거합니다.
    stream_key = auth_request.path.lstrip('/')
    
    print(f"스트림 인증 요청 수신: 키='{stream_key}', 동작='{auth_request.action}'")

    # 방송 시작(publish) 요청일 때만 인증을 수행합니다.
    if auth_request.action == "publish":
        user = crud.get_user_by_stream_key(db, stream_key=stream_key)
        
        if user and user.is_active:
            print(f"인증 성공: 사용자='{user.username}'")
            return {"status": "success"} # 인증 성공 시 HTTP 200 OK 반환
        else:
            print("인증 실패: 유효하지 않은 스트림 키")
            raise HTTPException(status_code=404, detail="Invalid stream key")
    
    # 시청(read) 요청 등은 별도 인증 없이 허용
    return {"status": "success"}
