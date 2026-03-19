from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

#Databse setup --------------------
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_URL=f"sqlite:///{os.path.join(BASE_DIR, 'merchants.db')}"
engine=create_engine(DATABASE_URL, connect_args={"check_same_thread":False})
SessionLocal=sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base=declarative_base()

#Database model -------------------
class MerchantScore(Base):
    """stores every merchant risk score with timestamp"""
    __tablename__="merchant_scores"
    
    id                       = Column(Integer, primary_key=True, index=True)
    merchant_id              = Column(String, index=True)
    risk_score               = Column(Integer)
    risk_label               = Column(String)
    fraud_probability        = Column(Float)

    transaction_velocity     = Column(Float)
    avg_transaction_value    = Column(Float)
    refund_rate              = Column(Float)
    chargeback_rate          = Column(Float)
    business_age_days        = Column(Integer)
    category_mismatch_score  = Column(Float)
    night_txn_ratio          = Column(Float)
    unique_customer_ratio    = Column(Float)
    geographic_spread        = Column(Integer)
    incomplete_profile_score = Column(Float)

    score_at                 = Column(DateTime, default=datetime.utcnow)

#Create tables ----------------
def init_db():
    Base.metadata.create_all(bind=engine)

#Database session --------------
def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()