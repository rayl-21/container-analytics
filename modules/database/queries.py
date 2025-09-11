"""
Database query functions for Container Analytics.

Provides high-level functions for:
- Inserting new detections and images
- Updating container tracking
- Fetching metrics by date range
- Aggregating statistics
- Getting unprocessed images
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.orm import Session

from .models import Image, Detection, Container, Metric, session_scope


def insert_image(filepath: str, camera_id: str, timestamp: Optional[datetime] = None, 
                file_size: Optional[int] = None) -> int:
    """
    Insert a new image record.
    
    Args:
        filepath: Path to the image file
        camera_id: Camera identifier
        timestamp: When the image was taken (defaults to now)
        file_size: File size in bytes
        
    Returns:
        The ID of the inserted image
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    with session_scope() as session:
        image = Image(
            filepath=filepath,
            camera_id=camera_id,
            timestamp=timestamp,
            file_size=file_size,
            processed=False
        )
        session.add(image)
        session.flush()  # Get the ID
        return image.id


def insert_detection(image_id: int, object_type: str, confidence: float, 
                    bbox: Dict[str, float], tracking_id: Optional[int] = None) -> int:
    """
    Insert a new detection record.
    
    Args:
        image_id: ID of the associated image
        object_type: Type of object detected (e.g., 'container', 'truck')
        confidence: Detection confidence score (0.0 to 1.0)
        bbox: Bounding box dictionary with keys: x, y, width, height
        tracking_id: Optional tracking ID for multi-frame tracking
        
    Returns:
        The ID of the inserted detection
    """
    with session_scope() as session:
        detection = Detection(
            image_id=image_id,
            object_type=object_type,
            confidence=confidence,
            bbox_x=bbox['x'],
            bbox_y=bbox['y'],
            bbox_width=bbox['width'],
            bbox_height=bbox['height'],
            tracking_id=tracking_id
        )
        session.add(detection)
        session.flush()
        return detection.id


def mark_image_processed(image_id: int):
    """Mark an image as processed."""
    with session_scope() as session:
        image = session.query(Image).filter(Image.id == image_id).first()
        if image:
            image.processed = True


def update_container_tracking(container_number: str, detection_time: datetime, 
                            camera_id: str, confidence: float) -> int:
    """
    Update container tracking information.
    
    Args:
        container_number: Container identification number
        detection_time: When the container was detected
        camera_id: Camera that detected the container
        confidence: Detection confidence
        
    Returns:
        The ID of the container record
    """
    with session_scope() as session:
        # Find existing container or create new one
        container = session.query(Container).filter(
            Container.container_number == container_number
        ).first()
        
        if container:
            # Update existing container
            container.last_seen = detection_time
            container.total_detections += 1
            
            # Update average confidence
            total_conf = container.avg_confidence * (container.total_detections - 1) + confidence
            container.avg_confidence = total_conf / container.total_detections
            
            # Recalculate dwell time
            container.calculate_dwell_time()
            
        else:
            # Create new container
            container = Container(
                container_number=container_number,
                first_seen=detection_time,
                last_seen=detection_time,
                total_detections=1,
                avg_confidence=confidence,
                camera_id=camera_id,
                status='active'
            )
            session.add(container)
        
        session.flush()
        return container.id


def get_unprocessed_images(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get unprocessed images for detection processing.
    
    Args:
        limit: Maximum number of images to return
        
    Returns:
        List of image dictionaries
    """
    with session_scope() as session:
        images = session.query(Image).filter(
            Image.processed == False
        ).order_by(Image.timestamp).limit(limit).all()
        
        return [
            {
                'id': img.id,
                'filepath': img.filepath,
                'camera_id': img.camera_id,
                'timestamp': img.timestamp,
                'file_size': img.file_size
            }
            for img in images
        ]


def get_metrics_by_date_range(start_date: datetime, end_date: datetime, 
                            camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get aggregated metrics for a date range.
    
    Args:
        start_date: Start of date range
        end_date: End of date range
        camera_id: Optional camera filter
        
    Returns:
        List of metric dictionaries
    """
    with session_scope() as session:
        query = session.query(Metric).filter(
            and_(
                Metric.date >= start_date,
                Metric.date <= end_date
            )
        )
        
        if camera_id:
            query = query.filter(Metric.camera_id == camera_id)
        
        metrics = query.order_by(Metric.date, Metric.hour).all()
        
        return [
            {
                'id': m.id,
                'date': m.date,
                'hour': m.hour,
                'throughput': m.throughput,
                'avg_dwell_time': m.avg_dwell_time,
                'container_count': m.container_count,
                'total_detections': m.total_detections,
                'avg_confidence': m.avg_confidence,
                'camera_id': m.camera_id
            }
            for m in metrics
        ]


def get_container_movements(start_date: datetime, end_date: datetime, direction: str = 'in') -> int:
    """
    Get the count of container movements in or out for a time period.
    
    Args:
        start_date: Start of date range
        end_date: End of date range  
        direction: 'in' or 'out' to filter by movement direction
        
    Returns:
        Count of container movements
    """
    with session_scope() as session:
        query = session.query(Container)
        
        if direction == 'in':
            # Containers that first appeared in this period
            count = query.filter(
                and_(
                    Container.first_seen >= start_date,
                    Container.first_seen <= end_date
                )
            ).count()
        else:
            # Containers that departed in this period  
            count = query.filter(
                and_(
                    Container.status == 'departed',
                    Container.last_seen >= start_date,
                    Container.last_seen <= end_date
                )
            ).count()
            
        return count

def get_container_statistics(start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Get container statistics for a date range.
    
    Args:
        start_date: Start date (defaults to 7 days ago)
        end_date: End date (defaults to now)
        
    Returns:
        Dictionary with container statistics
    """
    if end_date is None:
        end_date = datetime.utcnow()
    if start_date is None:
        start_date = end_date - timedelta(days=7)
    
    with session_scope() as session:
        # Base query for containers in date range
        base_query = session.query(Container).filter(
            or_(
                and_(Container.first_seen >= start_date, Container.first_seen <= end_date),
                and_(Container.last_seen >= start_date, Container.last_seen <= end_date)
            )
        )
        
        total_containers = base_query.count()
        active_containers = base_query.filter(Container.status == 'active').count()
        
        # Average dwell time
        avg_dwell = session.query(func.avg(Container.dwell_time)).filter(
            Container.dwell_time.isnot(None)
        ).scalar() or 0
        
        # Throughput (containers that departed in period)
        departed_containers = base_query.filter(Container.status == 'departed').count()
        
        return {
            'total_containers': total_containers,
            'active_containers': active_containers,
            'departed_containers': departed_containers,
            'average_dwell_time': round(avg_dwell, 2),
            'date_range': {'start': start_date, 'end': end_date}
        }


def get_throughput_data(start_date: datetime, end_date: datetime, 
                       camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get hourly throughput data for visualization.
    
    Args:
        start_date: Start date
        end_date: End date
        camera_id: Optional camera filter
        
    Returns:
        List of throughput data points
    """
    with session_scope() as session:
        query = session.query(
            Metric.date,
            Metric.hour,
            func.sum(Metric.throughput).label('throughput'),
            func.avg(Metric.avg_dwell_time).label('avg_dwell_time')
        ).filter(
            and_(
                Metric.date >= start_date,
                Metric.date <= end_date
            )
        )
        
        if camera_id:
            query = query.filter(Metric.camera_id == camera_id)
        
        results = query.group_by(Metric.date, Metric.hour).order_by(
            Metric.date, Metric.hour
        ).all()
        
        return [
            {
                'datetime': datetime.combine(r.date.date(), datetime.min.time()) + timedelta(hours=r.hour),
                'throughput': r.throughput,
                'avg_dwell_time': r.avg_dwell_time
            }
            for r in results
        ]


def get_dwell_time_data(start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    """
    Get dwell time distribution data.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of dwell time data
    """
    with session_scope() as session:
        containers = session.query(Container).filter(
            and_(
                Container.first_seen >= start_date,
                Container.first_seen <= end_date,
                Container.dwell_time.isnot(None)
            )
        ).all()
        
        return [
            {
                'container_number': c.container_number,
                'dwell_time': c.dwell_time,
                'first_seen': c.first_seen,
                'last_seen': c.last_seen,
                'status': c.status
            }
            for c in containers
        ]


def aggregate_hourly_metrics(date: datetime, hour: int, camera_id: str):
    """
    Calculate and store hourly metrics.
    
    Args:
        date: Date to aggregate
        hour: Hour to aggregate (0-23)
        camera_id: Camera ID
    """
    start_time = datetime.combine(date.date(), datetime.min.time()) + timedelta(hours=hour)
    end_time = start_time + timedelta(hours=1)
    
    with session_scope() as session:
        # Get images in this hour
        images = session.query(Image).filter(
            and_(
                Image.camera_id == camera_id,
                Image.timestamp >= start_time,
                Image.timestamp < end_time,
                Image.processed == True
            )
        ).all()
        
        if not images:
            return
        
        image_ids = [img.id for img in images]
        
        # Count detections
        total_detections = session.query(Detection).filter(
            Detection.image_id.in_(image_ids)
        ).count()
        
        # Average confidence
        avg_confidence = session.query(func.avg(Detection.confidence)).filter(
            Detection.image_id.in_(image_ids)
        ).scalar() or 0
        
        # Container metrics
        containers_in_hour = session.query(Container).filter(
            or_(
                and_(Container.first_seen >= start_time, Container.first_seen < end_time),
                and_(Container.last_seen >= start_time, Container.last_seen < end_time)
            )
        ).all()
        
        container_count = len(containers_in_hour)
        throughput = len([c for c in containers_in_hour if c.status == 'departed'])
        
        avg_dwell = None
        if containers_in_hour:
            dwell_times = [c.dwell_time for c in containers_in_hour if c.dwell_time is not None]
            if dwell_times:
                avg_dwell = sum(dwell_times) / len(dwell_times)
        
        # Check if metric already exists
        existing_metric = session.query(Metric).filter(
            and_(
                Metric.date == date,
                Metric.hour == hour,
                Metric.camera_id == camera_id
            )
        ).first()
        
        if existing_metric:
            # Update existing
            existing_metric.throughput = throughput
            existing_metric.avg_dwell_time = avg_dwell
            existing_metric.container_count = container_count
            existing_metric.total_detections = total_detections
            existing_metric.avg_confidence = avg_confidence
        else:
            # Create new metric
            metric = Metric(
                date=date,
                hour=hour,
                throughput=throughput,
                avg_dwell_time=avg_dwell,
                container_count=container_count,
                total_detections=total_detections,
                avg_confidence=avg_confidence,
                camera_id=camera_id
            )
            session.add(metric)


def get_recent_detections(limit: int = 100, object_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get recent detections for monitoring.
    
    Args:
        limit: Maximum number of detections to return
        object_type: Optional filter by object type
        
    Returns:
        List of recent detection data
    """
    with session_scope() as session:
        query = session.query(Detection, Image).join(Image).order_by(desc(Detection.timestamp))
        
        if object_type:
            query = query.filter(Detection.object_type == object_type)
        
        results = query.limit(limit).all()
        
        return [
            {
                'detection_id': d.id,
                'image_id': i.id,
                'filepath': i.filepath,
                'camera_id': i.camera_id,
                'object_type': d.object_type,
                'confidence': d.confidence,
                'bbox': d.bbox,
                'timestamp': d.timestamp,
                'tracking_id': d.tracking_id
            }
            for d, i in results
        ]


def cleanup_old_data(days_to_keep: int = 30):
    """
    Clean up old data to maintain database performance.
    
    Args:
        days_to_keep: Number of days of data to retain
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    
    with session_scope() as session:
        # Delete old detections (cascade will handle relationships)
        old_images = session.query(Image).filter(Image.timestamp < cutoff_date)
        
        deleted_images = old_images.count()
        old_images.delete()
        
        # Delete old metrics
        old_metrics = session.query(Metric).filter(Metric.date < cutoff_date)
        deleted_metrics = old_metrics.count()
        old_metrics.delete()
        
        # Update container status for old containers
        old_containers = session.query(Container).filter(
            and_(
                Container.last_seen < cutoff_date,
                Container.status == 'active'
            )
        )
        
        for container in old_containers:
            container.status = 'departed'
        
        print(f"Cleanup complete: {deleted_images} images, {deleted_metrics} metrics deleted")


def get_detection_summary(start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """
    Get detection summary statistics.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Summary statistics dictionary
    """
    with session_scope() as session:
        # Total detections by type
        detection_counts = session.query(
            Detection.object_type,
            func.count(Detection.id).label('count'),
            func.avg(Detection.confidence).label('avg_confidence')
        ).join(Image).filter(
            and_(
                Image.timestamp >= start_date,
                Image.timestamp <= end_date
            )
        ).group_by(Detection.object_type).all()
        
        # Images processed
        total_images = session.query(Image).filter(
            and_(
                Image.timestamp >= start_date,
                Image.timestamp <= end_date
            )
        ).count()
        
        processed_images = session.query(Image).filter(
            and_(
                Image.timestamp >= start_date,
                Image.timestamp <= end_date,
                Image.processed == True
            )
        ).count()
        
        return {
            'detection_counts': [
                {
                    'object_type': dc.object_type,
                    'count': dc.count,
                    'avg_confidence': round(dc.avg_confidence, 3)
                }
                for dc in detection_counts
            ],
            'total_images': total_images,
            'processed_images': processed_images,
            'processing_rate': round((processed_images / total_images * 100), 1) if total_images > 0 else 0,
            'date_range': {'start': start_date, 'end': end_date}
        }