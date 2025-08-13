from sqlalchemy.orm import Session
import models
import schemas

def create_event(db: Session, event: schemas.EventCreate):
    # Pydantic 모델을 SQLAlchemy 모델 인스턴스로 변환
    db_event = models.Event(**event.dict())
    
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    
    return db_event

def get_events_by_user(db: Session, user_id: str):
    # user_id 로 필터링해서 리스트를 반환
    return db.query(models.Event).filter(models.Event.user_id == user_id).all()

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate):
    # 실제로는 password를 해싱해야 합니다. 예: get_password_hash(user.password)
    fake_hashed_password = user.password + "notreallyhashed"
    db_user = models.User(username=user.username, hashed_password=fake_hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def generate_and_set_stream_key(db: Session, user_id: int):
    """(기능 1) 고유 스트림 키를 생성하여 DB에 저장"""
    db_user = get_user(db=db, user_id=user_id)
    if db_user:
        new_key = f"{db_user.username}-{uuid.uuid4()}"
        db_user.stream_key = new_key
        db.commit()
        db.refresh(db_user)
        return db_user
    return None

def get_user_by_stream_key(db: Session, stream_key: str):
    """(기능 3) 스트림 키로 사용자를 조회"""
    return db.query(models.User).filter(models.User.stream_key == stream_key).first()


############################################ AI 관련 ################################################
def create_aimodel(db: Session, ai: schemas.AimodelCreate):
    db_aimodel = models.Aimodel(**ai.dict())

    db.add(db_aimodel)
    db.commit()
    db.refresh(db_aimodel)

    return db_aimodel


#####################################################################################################