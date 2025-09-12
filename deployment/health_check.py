#!/usr/bin/env python3
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

def check_database_health():
    try:
        db_path = Path("/data/database.db")
        if not db_path.exists():
            return False, "Database file does not exist"
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        one_hour_ago = datetime.now() - timedelta(hours=1)
        cursor.execute(
            "SELECT COUNT(*) FROM images WHERE created_at > ?",
            (one_hour_ago.isoformat(),)
        )
        recent_count = cursor.fetchone()[0]
        conn.close()
        
        if recent_count > 0:
            return True, f"Found {recent_count} recent images"
        else:
            return True, "No recent images yet (service may be starting)"
            
    except Exception as e:
        return False, f"Database error: {str(e)}"

def main():
    healthy, message = check_database_health()
    
    if healthy:
        print(f"Health check passed: {message}")
        sys.exit(0)
    else:
        print(f"Health check failed: {message}")
        sys.exit(1)

if __name__ == "__main__":
    main()