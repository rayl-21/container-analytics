#!/usr/bin/env python
"""
Script to purge all image records from database and download new images from Dray Dog.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.database.models import session_scope, Image, Detection, Container, Metric
from modules.downloader.selenium_client import DrayDogDownloader

def purge_all_image_records():
    """Purge all image records and related data from database."""
    print("Purging all image records from database...")
    
    with session_scope() as session:
        # Count records before deletion
        image_count = session.query(Image).count()
        detection_count = session.query(Detection).count()
        
        print(f"Found {image_count} image records and {detection_count} detection records")
        
        # Delete all detections first (they reference images)
        session.query(Detection).delete()
        
        # Delete all images
        session.query(Image).delete()
        
        # Also clean up any orphaned containers and metrics
        session.query(Container).delete()
        session.query(Metric).delete()
        
        session.commit()
        print(f"Successfully purged {image_count} image records and all related data")

def download_images_for_date_range(start_date, end_date):
    """Download images from Dray Dog for specified date range."""
    print(f"\nDownloading images from {start_date} to {end_date}...")
    
    # Initialize downloader
    downloader = DrayDogDownloader()
    
    try:
        total_downloaded = 0
        # Download images for in_gate stream
        for stream in ['in_gate']:
            print(f"\nDownloading {stream} images for date range...")
            
            try:
                # Download images for this date range and stream
                results = downloader.download_images_for_date_range(
                    start_date=start_date,
                    end_date=end_date,
                    stream_name=stream
                )
                
                # Count total downloaded images
                stream_total = sum(len(files) for files in results.values())
                total_downloaded += stream_total
                print(f"  Downloaded {stream_total} images for {stream}")
                
                # Show breakdown by date
                for date, files in results.items():
                    if files:
                        print(f"    {date}: {len(files)} images")
                
            except Exception as e:
                print(f"  Error downloading {stream} images: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\nTotal images downloaded: {total_downloaded}")
        
    finally:
        # Clean up
        downloader.cleanup()

def main():
    """Main execution function."""
    print("="*60)
    print("Image Database Purge and Download Script")
    print("="*60)
    
    # Step 1: Purge all existing image records
    purge_all_image_records()
    
    # Step 2: Download new images for date range
    start_date = "2025-09-01"
    end_date = "2025-09-07"
    
    download_images_for_date_range(start_date, end_date)
    
    print("\n" + "="*60)
    print("Process completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()