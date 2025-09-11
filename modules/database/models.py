"""
SQLAlchemy models for Container Analytics database.

Defines tables for:
- Images: Downloaded camera images
- Detections: YOLO object detections 
- Containers: Container tracking data
- Metrics: Aggregated analytics metrics
"""

import os
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    Boolean, ForeignKey, Text, Index, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.engine import Engine

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/database.db')
Base = declarative_base()

# Global engine and session factory
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


class Image(Base):
    """
    Table for storing downloaded camera images metadata.
    """
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    filepath = Column(String(500), nullable=False, unique=True)
    camera_id = Column(String(100), nullable=False)
    processed = Column(Boolean, default=False, nullable=False)
    file_size = Column(Integer)  # File size in bytes
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    detections = relationship("Detection", back_populates="image", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_image_timestamp', 'timestamp'),
        Index('idx_image_camera_id', 'camera_id'),
        Index('idx_image_processed', 'processed'),
        Index('idx_image_camera_timestamp', 'camera_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Image(id={self.id}, camera_id='{self.camera_id}', timestamp={self.timestamp})>"


class Detection(Base):
    """
    Table for storing YOLO object detections from images.
    """
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    object_type = Column(String(50), nullable=False)  # 'container', 'truck', 'person', etc.
    confidence = Column(Float, nullable=False)
    bbox_x = Column(Float, nullable=False)  # Bounding box coordinates
    bbox_y = Column(Float, nullable=False)
    bbox_width = Column(Float, nullable=False) 
    bbox_height = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    tracking_id = Column(Integer)  # For object tracking across frames
    
    # Relationships
    image = relationship("Image", back_populates="detections")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_detection_image_id', 'image_id'),
        Index('idx_detection_object_type', 'object_type'),
        Index('idx_detection_timestamp', 'timestamp'),
        Index('idx_detection_tracking_id', 'tracking_id'),
        Index('idx_detection_confidence', 'confidence'),
    )
    
    @property
    def bbox(self) -> dict:
        """Return bounding box as dictionary."""
        return {
            'x': self.bbox_x,
            'y': self.bbox_y,
            'width': self.bbox_width,
            'height': self.bbox_height
        }
    
    def __repr__(self):
        return f"<Detection(id={self.id}, object_type='{self.object_type}', confidence={self.confidence:.2f})>"


class Container(Base):
    """
    Table for tracking individual containers over time.
    """
    __tablename__ = 'containers'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    container_number = Column(String(20), unique=True, index=True)  # OCR extracted number
    first_seen = Column(DateTime, nullable=False)
    last_seen = Column(DateTime, nullable=False)
    dwell_time = Column(Float)  # Hours between first and last seen
    total_detections = Column(Integer, default=0)
    avg_confidence = Column(Float)
    status = Column(String(20), default='active')  # 'active', 'departed', 'unknown'
    camera_id = Column(String(100))  # Primary camera where detected
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_container_number', 'container_number'),
        Index('idx_container_first_seen', 'first_seen'),
        Index('idx_container_last_seen', 'last_seen'),
        Index('idx_container_status', 'status'),
        Index('idx_container_camera_id', 'camera_id'),
    )
    
    def calculate_dwell_time(self):
        """Calculate dwell time in hours."""
        if self.first_seen and self.last_seen:
            delta = self.last_seen - self.first_seen
            self.dwell_time = delta.total_seconds() / 3600.0
        return self.dwell_time
    
    def __repr__(self):
        return f"<Container(id={self.id}, number='{self.container_number}', dwell_time={self.dwell_time})>"


class Metric(Base):
    """
    Table for storing aggregated analytics metrics.
    """
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)  # Date of metric
    hour = Column(Integer, nullable=False)   # Hour of day (0-23)
    throughput = Column(Integer, default=0)  # Containers processed in this hour
    avg_dwell_time = Column(Float)           # Average dwell time in hours
    container_count = Column(Integer, default=0)  # Total containers present
    total_detections = Column(Integer, default=0)  # Total detections in hour
    avg_confidence = Column(Float)           # Average detection confidence
    camera_id = Column(String(100))          # Camera ID for this metric
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_metric_date', 'date'),
        Index('idx_metric_hour', 'hour'),
        Index('idx_metric_camera_id', 'camera_id'),
        Index('idx_metric_date_hour', 'date', 'hour'),
        Index('idx_metric_camera_date_hour', 'camera_id', 'date', 'hour'),
    )
    
    def __repr__(self):
        return f"<Metric(date={self.date}, hour={self.hour}, throughput={self.throughput})>"


def get_engine() -> Engine:
    """Get or create database engine."""
    global _engine
    if _engine is None:
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        _engine = create_engine(
            DATABASE_URL,
            echo=False,  # Set to True for SQL debugging
            connect_args={'check_same_thread': False} if 'sqlite' in DATABASE_URL else {}
        )
    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal


def get_session() -> Session:
    """Get a new database session."""
    SessionLocal = get_session_factory()
    return SessionLocal()


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_tables():
    """Create all database tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


def init_database():
    """Initialize database with tables and default data."""
    print("Initializing Container Analytics database...")
    
    # Create tables
    create_tables()
    
    # Add any default data if needed
    with session_scope() as session:
        # Check if we need to add any seed data
        image_count = session.query(Image).count()
        print(f"Database initialized. Current images: {image_count}")
    
    print("Database initialization complete.")


def migrate_database():
    """Run database migrations (placeholder for future use)."""
    print("Running database migrations...")
    
    # For now, just ensure all tables exist
    create_tables()
    
    print("Database migrations complete.")


def get_database_stats():
    """Get basic database statistics."""
    with session_scope() as session:
        stats = {
            'total_images': session.query(Image).count(),
            'processed_images': session.query(Image).filter(Image.processed == True).count(),
            'total_detections': session.query(Detection).count(),
            'total_containers': session.query(Container).count(),
            'active_containers': session.query(Container).filter(Container.status == 'active').count(),
            'total_metrics': session.query(Metric).count(),
        }
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Database management')
    parser.add_argument('--init', action='store_true', help='Initialize database')
    parser.add_argument('--migrate', action='store_true', help='Run migrations')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    
    args = parser.parse_args()
    
    if args.init:
        init_database()
    elif args.migrate:
        migrate_database()
    elif args.stats:
        stats = get_database_stats()
        print("\nDatabase Statistics:")
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    else:
        print("Use --init, --migrate, or --stats")