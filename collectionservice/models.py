from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from database import Base

class Event(Base): #Pydantic 모델 구
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255)) # 클라이언트 코드의 USER_ID를 저장할 컬럼 추가
    event_time = Column(DateTime)
    duration_seconds = Column(Float)
    weight_info = Column(String(255))
    video_url = Column(String(255))

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=True)
    stream_key = Column(String(255), unique=True, index=True, nullable=True)


# aimodel
class Aimodel(Base):
    __tablename__ = "aimodel"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255)) 
    model_status = Column(String(255))
    model_name = Column(String(255))
    val_precision = Column(Float)
    val_recall = Column(Float)
    val_map50 = Column(Float)