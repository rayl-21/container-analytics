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
    image_path = Column(String(500), nullable=False, unique=True)  # Renamed for consistency
    filepath = Column(String(500))  # Keep for backward compatibility
    camera_id = Column(String(100), nullable=False)
    processed = Column(Boolean, default=False, nullable=False)
    file_size = Column(Integer)  # File size in bytes
    detection_count = Column(Integer, default=0)  # Number of detections in this image
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    detections = relationship("Detection", back_populates="image", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_image_timestamp', 'timestamp'),
        Index('idx_image_camera_id', 'camera_id'),
        Index('idx_image_processed', 'processed'),
        Index('idx_image_camera_timestamp', 'camera_id', 'timestamp'),
        Index('idx_image_path', 'image_path'),
    )
    
    def __repr__(self):
        return f"<Image(id={self.id}, camera_id='{self.camera_id}', timestamp={self.timestamp}, detections={self.detection_count})>"


class Detection(Base):
    """
    Table for storing YOLO object detections from images.
    """
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    class_id = Column(Integer, nullable=False)  # YOLO class ID (0=container, etc.)
    confidence = Column(Float, nullable=False)
    x1 = Column(Float, nullable=False)  # Bounding box coordinates (xyxy format)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False) 
    y2 = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    track_id = Column(Integer)  # For object tracking across frames
    object_type = Column(String(50))  # Human-readable type ('container', 'truck', etc.)
    
    # Relationships
    image = relationship("Image", back_populates="detections")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_detection_image_id', 'image_id'),
        Index('idx_detection_class_id', 'class_id'),
        Index('idx_detection_timestamp', 'timestamp'),
        Index('idx_detection_track_id', 'track_id'),
        Index('idx_detection_confidence', 'confidence'),
        Index('idx_detection_object_type', 'object_type'),
    )
    
    @property
    def bbox(self) -> dict:
        """Return bounding box as dictionary."""
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2
        }
    
    @property 
    def bbox_xywh(self) -> dict:
        """Return bounding box in xywh format."""
        return {
            'x': self.x1,
            'y': self.y1,
            'width': self.x2 - self.x1,
            'height': self.y2 - self.y1
        }
    
    def set_bbox(self, x1: float, y1: float, x2: float, y2: float):
        """Set bounding box coordinates."""
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def __repr__(self):
        return f"<Detection(id={self.id}, class_id={self.class_id}, confidence={self.confidence:.2f}, track_id={self.track_id})>"


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
    entry_camera_id = Column(String(100))  # Camera where first detected (IN gate)
    exit_camera_id = Column(String(100))   # Camera where last seen (OUT gate)
    current_camera_id = Column(String(100))  # Current camera location
    entry_type = Column(String(20))  # 'truck_entry', 'rail_entry', etc.
    exit_type = Column(String(20))   # 'truck_exit', 'rail_exit', etc.
    ocr_confidence = Column(Float)   # Best OCR confidence score
    track_ids = Column(Text)         # JSON array of associated track IDs
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    movements = relationship("ContainerMovement", back_populates="container", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_container_number', 'container_number'),
        Index('idx_container_first_seen', 'first_seen'),
        Index('idx_container_last_seen', 'last_seen'),
        Index('idx_container_status', 'status'),
        Index('idx_container_entry_camera', 'entry_camera_id'),
        Index('idx_container_exit_camera', 'exit_camera_id'),
        Index('idx_container_current_camera', 'current_camera_id'),
        Index('idx_container_entry_type', 'entry_type'),
        Index('idx_container_created_at', 'created_at'),
    )
    
    def calculate_dwell_time(self):
        """Calculate dwell time in hours."""
        if self.first_seen and self.last_seen:
            delta = self.last_seen - self.first_seen
            self.dwell_time = delta.total_seconds() / 3600.0
        return self.dwell_time
    
    def get_track_ids_list(self) -> List[int]:
        """Parse track IDs from JSON string."""
        if not self.track_ids:
            return []
        try:
            import json
            return json.loads(self.track_ids)
        except (json.JSONDecodeError, TypeError):
            return []
    
    def set_track_ids_list(self, track_ids: List[int]):
        """Store track IDs as JSON string."""
        import json
        self.track_ids = json.dumps(track_ids)
    
    def add_track_id(self, track_id: int):
        """Add a track ID to the list."""
        current_ids = self.get_track_ids_list()
        if track_id not in current_ids:
            current_ids.append(track_id)
            self.set_track_ids_list(current_ids)
    
    @property
    def is_active(self) -> bool:
        """Check if container is currently active."""
        return self.status == 'active'
    
    @property
    def has_moved(self) -> bool:
        """Check if container has moved between cameras."""
        return (self.entry_camera_id != self.current_camera_id or 
                len(self.movements) > 1)
    
    def __repr__(self):
        return f"<Container(id={self.id}, number='{self.container_number}', status='{self.status}', dwell_time={self.dwell_time})>"


class ContainerMovement(Base):
    """
    Table for tracking container movements between cameras/locations.
    """
    __tablename__ = 'container_movements'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    container_id = Column(Integer, ForeignKey('containers.id'), nullable=False)
    from_camera_id = Column(String(100))    # Source camera (can be null for initial entry)
    to_camera_id = Column(String(100), nullable=False)  # Destination camera
    movement_type = Column(String(20), nullable=False)  # 'entry', 'movement', 'exit'
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    track_id = Column(Integer)  # Associated tracking ID
    confidence = Column(Float)  # Detection confidence
    bbox_x1 = Column(Float)     # Bounding box coordinates
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)
    ocr_confidence = Column(Float)  # OCR confidence for this detection
    duration_from_last = Column(Float)  # Minutes since last movement
    
    # Relationships
    container = relationship("Container", back_populates="movements")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_movement_container_id', 'container_id'),
        Index('idx_movement_timestamp', 'timestamp'),
        Index('idx_movement_from_camera', 'from_camera_id'),
        Index('idx_movement_to_camera', 'to_camera_id'),
        Index('idx_movement_type', 'movement_type'),
        Index('idx_movement_track_id', 'track_id'),
        Index('idx_movement_container_timestamp', 'container_id', 'timestamp'),
    )
    
    @property
    def bbox(self) -> dict:
        """Return bounding box as dictionary."""
        return {
            'x1': self.bbox_x1,
            'y1': self.bbox_y1,
            'x2': self.bbox_x2,
            'y2': self.bbox_y2
        }
    
    def set_bbox(self, x1: float, y1: float, x2: float, y2: float):
        """Set bounding box coordinates."""
        self.bbox_x1 = x1
        self.bbox_y1 = y1
        self.bbox_x2 = x2
        self.bbox_y2 = y2
    
    def __repr__(self):
        return (f"<ContainerMovement(id={self.id}, container_id={self.container_id}, "
                f"type='{self.movement_type}', from='{self.from_camera_id}', "
                f"to='{self.to_camera_id}', timestamp={self.timestamp})>")


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