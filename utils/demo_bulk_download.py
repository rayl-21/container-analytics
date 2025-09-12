#!/usr/bin/env python3
"""
Demonstration script for BulkImageDownloader usage.

This script shows various ways to use the BulkImageDownloader class
for downloading images from Dray Dog cameras in bulk.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.bulk_download import BulkImageDownloader


def demo_basic_usage():
    """Demonstrate basic bulk download usage."""
    print("=== Basic Bulk Download Demo ===")
    
    # Initialize downloader
    downloader = BulkImageDownloader(
        download_dir="demo_downloads",
        headless=True,
        use_direct_download=True
    )
    
    try:
        # Download a small date range for demonstration
        results = downloader.download_date_range(
            start_date="2025-09-06",  # Single day for demo
            end_date="2025-09-06",
            streams=["in_gate"],
            max_images_per_date=5  # Limit for demo
        )
        
        # Generate report
        report = downloader.generate_download_report()
        
        # Print summary
        print(f"Downloads completed: {report['summary']['successful_downloads']}")
        print(f"Success rate: {report['summary']['success_rate_percent']:.1f}%")
        print(f"Total size: {report['summary']['total_file_size_mb']:.2f} MB")
        
        return results
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return None
    
    finally:
        downloader.cleanup()


def demo_with_database():
    """Demonstrate database integration."""
    print("\n=== Database Integration Demo ===")
    
    # Sample metadata (would normally come from actual downloads)
    sample_metadata = [
        {
            'filepath': '/demo_downloads/2025-09-06/in_gate/image1.jpg',
            'camera_id': 'in_gate',
            'timestamp': datetime(2025, 9, 6, 10, 0, 0),
            'file_size': 1024
        },
        {
            'filepath': '/demo_downloads/2025-09-06/in_gate/image2.jpg',
            'camera_id': 'in_gate',
            'timestamp': datetime(2025, 9, 6, 10, 10, 0),
            'file_size': 2048
        }
    ]
    
    with BulkImageDownloader() as downloader:
        try:
            # Note: This would fail without proper database setup
            # saved_count = downloader.save_to_database(sample_metadata)
            # print(f"Saved {saved_count} records to database")
            print("Database integration available (requires database setup)")
            
        except Exception as e:
            print(f"Database demo failed (expected): {e}")


def demo_file_organization():
    """Demonstrate file organization features."""
    print("\n=== File Organization Demo ===")
    
    # Create demo directory structure
    demo_dir = Path("demo_source")
    demo_dir.mkdir(exist_ok=True)
    
    # Create some sample files
    sample_files = [
        "20250906100000_in_gate.jpg",
        "20250906101000_out_gate.jpg",
        "20250907120000_in_gate.jpg"
    ]
    
    for filename in sample_files:
        (demo_dir / filename).write_text("fake image data")
    
    with BulkImageDownloader() as downloader:
        try:
            # Organize files
            success = downloader.organize_files(
                source_path=str(demo_dir),
                target_path="demo_organized"
            )
            
            if success:
                print("File organization completed successfully")
                
                # Show organized structure
                organized_dir = Path("demo_organized")
                if organized_dir.exists():
                    print("Organized structure:")
                    for item in organized_dir.rglob("*"):
                        if item.is_file():
                            print(f"  {item}")
            else:
                print("File organization failed")
                
        except Exception as e:
            print(f"Organization demo failed: {e}")


def demo_reporting():
    """Demonstrate reporting functionality."""
    print("\n=== Reporting Demo ===")
    
    with BulkImageDownloader() as downloader:
        # Simulate some download statistics
        downloader.stats.start_time = datetime(2025, 9, 12, 10, 0, 0)
        downloader.stats.end_time = datetime(2025, 9, 12, 10, 15, 0)
        downloader.stats.successful_downloads = 42
        downloader.stats.failed_downloads = 8
        downloader.stats.total_file_size = 1024 * 1024 * 25  # 25 MB
        
        # Generate report
        report = downloader.generate_download_report()
        
        # Save report
        report_path = downloader.save_report(report, "demo_report.json")
        
        # Display summary
        print("Download Report Summary:")
        print(f"  Successful downloads: {report['summary']['successful_downloads']}")
        print(f"  Failed downloads: {report['summary']['failed_downloads']}")
        print(f"  Success rate: {report['summary']['success_rate_percent']:.1f}%")
        print(f"  Total size: {report['summary']['total_file_size_mb']:.2f} MB")
        print(f"  Duration: {report['summary']['duration_seconds']:.0f} seconds")
        print(f"  Report saved to: {report_path}")


def main():
    """Run all demonstration examples."""
    print("BulkImageDownloader Demonstration")
    print("=" * 40)
    
    # Basic usage demo
    demo_basic_usage()
    
    # Database integration demo
    demo_database_integration()
    
    # File organization demo
    demo_file_organization()
    
    # Reporting demo
    demo_reporting()
    
    print("\n=== Demo Complete ===")
    print("Note: Some demos may fail without proper environment setup")
    print("(database, network access, etc.)")


if __name__ == "__main__":
    main()