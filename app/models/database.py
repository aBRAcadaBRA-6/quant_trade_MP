
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

# User Profile Model
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    email = Column(String(100), unique=True)
    risk_tolerance = Column(Float)  # 0.0 to 1.0
    capital = Column(Float)
    max_assets = Column(Integer, default=20)
    drawdown_limit = Column(Float, default=0.25)  # 25% max drawdown
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="user")
    trades = relationship("Trade", back_populates="user")

# Market Data Model
class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), index=True)
    date = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)

# Portfolio Model
class Portfolio(Base):
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    assets = Column(JSON)  # {"AAPL": 0.15, "MSFT": 0.20, ...}
    method = Column(String(50))  # "sparse_mr", "ml_enhanced", etc.
    backtest_sharpe = Column(Float)
    backtest_max_drawdown = Column(Float)
    
    user = relationship("User", back_populates="portfolios")

# Trade Model
class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(10))
    side = Column(String(4))  # 'BUY' or 'SELL'
    quantity = Column(Integer)
    price = Column(Float)
    slippage = Column(Float)
    commission = Column(Float)
    pnl = Column(Float)
    
    user = relationship("User", back_populates="trades")

# Database connection
DATABASE_URL = "postgresql://cherry:asdfuiop@localhost/quant_trading_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Create all tables
def init_db():
    Base.metadata.create_all(bind=engine)
