#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NEXUS SYSTEM MANAGER - IMPROVED LEGITIMATE VERSION
Android-Compatible System Monitor and Task Manager
Version 4.0 - Honest Implementation with Real Functionality
"""

import os
import sys
import time
import platform
import logging
import signal
import threading
import subprocess
import json
import hashlib
import psutil  # For real system monitoring
import sqlite3  # For proper data storage
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock, Event
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import configparser

# Try to import real ML libraries (optional)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Android-specific imports with proper error handling
try:
    import android
    ANDROID_API_AVAILABLE = True
except ImportError:
    ANDROID_API_AVAILABLE = False

# Thread safety
_state_lock = RLock()
_db_lock = RLock()

# ======================== CONFIGURATION ========================

APP_NAME = "NEXUS_SYSTEM_MANAGER"
APP_VERSION = "4.0 - Legitimate Implementation"

def get_safe_base_path():
    """Get a safe base path that's writable on Android."""
    potential_paths = [
        os.path.expanduser("~/NexusSystem"),
        os.path.join(os.getcwd(), "NexusSystem"),
        os.path.join(os.path.expanduser("~"), "Documents", "NexusSystem"),
        "/storage/emulated/0/NexusSystem"
    ]
    
    for path in potential_paths:
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return path
        except (OSError, PermissionError):
            continue
    
    return os.path.join(os.getcwd(), "NexusSystem")

BASE_DIR = get_safe_base_path()
LOG_DIR = os.path.join(BASE_DIR, "logs")
CONFIG_FILE = os.path.join(BASE_DIR, "config.ini")
DATABASE_FILE = os.path.join(BASE_DIR, "nexus.db")

# ======================== COLORS ========================

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

# ======================== CONFIGURATION MANAGER ========================

class ConfigManager:
    """Manages system configuration with proper validation."""
    
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config_file = CONFIG_FILE
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or create defaults."""
        defaults = {
            'system': {
                'max_log_size_mb': '10',
                'log_retention_days': '7',
                'max_concurrent_tasks': '5',
                'scan_interval_seconds': '60'
            },
            'security': {
                'allow_script_execution': 'false',
                'whitelist_enabled': 'true',
                'scan_depth_limit': '3'
            },
            'monitoring': {
                'cpu_threshold': '80',
                'memory_threshold': '80',
                'disk_threshold': '90'
            }
        }
        
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            self.config.read_dict(defaults)
            self.save_config()
    
    def save_config(self):
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                self.config.write(f)
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def get(self, section: str, key: str, fallback=None):
        """Get configuration value with fallback."""
        return self.config.get(section, key, fallback=fallback)
    
    def getint(self, section: str, key: str, fallback=0):
        """Get integer configuration value."""
        return self.config.getint(section, key, fallback=fallback)
    
    def getboolean(self, section: str, key: str, fallback=False):
        """Get boolean configuration value."""
        return self.config.getboolean(section, key, fallback=fallback)

# ======================== DATABASE MANAGER ========================

class DatabaseManager:
    """Manages SQLite database for persistent storage."""
    
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        try:
            os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # System monitoring table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        cpu_percent REAL,
                        memory_percent REAL,
                        disk_percent REAL,
                        process_count INTEGER,
                        network_connections INTEGER
                    )
                ''')
                
                # Task execution table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS task_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        task_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        duration_seconds REAL,
                        output_summary TEXT
                    )
                ''')
                
                # File scan results
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS file_scans (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_type TEXT,
                        file_size INTEGER,
                        file_hash TEXT,
                        is_executable BOOLEAN,
                        is_safe BOOLEAN
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
    
    def execute_query(self, query: str, params: tuple = ()):
        """Execute a database query safely."""
        try:
            with _db_lock:
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    conn.commit()
                    return cursor.fetchall()
        except Exception as e:
            logging.error(f"Database query failed: {e}")
            return []
    
    def log_system_metrics(self, metrics: dict):
        """Log system metrics to database."""
        query = '''
            INSERT INTO system_metrics 
            (timestamp, cpu_percent, memory_percent, disk_percent, process_count, network_connections)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        params = (
            datetime.now().isoformat(),
            metrics.get('cpu_percent', 0),
            metrics.get('memory_percent', 0),
            metrics.get('disk_percent', 0),
            metrics.get('process_count', 0),
            metrics.get('network_connections', 0)
        )
        self.execute_query(query, params)
    
    def log_task_execution(self, task_name: str, status: str, duration: float, output: str = ""):
        """Log task execution to database."""
        query = '''
            INSERT INTO task_executions (timestamp, task_name, status, duration_seconds, output_summary)
            VALUES (?, ?, ?, ?, ?)
        '''
        params = (datetime.now().isoformat(), task_name, status, duration, output[:1000])
        self.execute_query(query, params)
    
    def get_recent_metrics(self, hours: int = 24) -> List[dict]:
        """Get recent system metrics."""
        query = '''
            SELECT * FROM system_metrics 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours)
        
        results = self.execute_query(query)
        return [
            {
                'timestamp': row[1],
                'cpu_percent': row[2],
                'memory_percent': row[3],
                'disk_percent': row[4],
                'process_count': row[5],
                'network_connections': row[6]
            }
            for row in results
        ]

# ======================== REAL SYSTEM MONITOR ========================

class SystemMonitor:
    """Real system monitoring using psutil."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.monitoring_active = False
        self.monitor_thread = None
    
    def get_system_metrics(self) -> dict:
        """Get real system metrics."""
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
                'process_count': len(psutil.pids()),
                'network_connections': len(psutil.net_connections()),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add more detailed info
            memory = psutil.virtual_memory()
            metrics.update({
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'cpu_count': psutil.cpu_count(),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
            })
            
            return metrics
            
        except Exception as e:
            logging.error(f"Failed to get system metrics: {e}")
            return {}
    
    def start_monitoring(self, interval: int = 60):
        """Start continuous system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_worker, args=(interval,), daemon=True)
        self.monitor_thread.start()
        logging.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logging.info("System monitoring stopped")
    
    def _monitor_worker(self, interval: int):
        """Worker thread for continuous monitoring."""
        while self.monitoring_active:
            try:
                metrics = self.get_system_metrics()
                if metrics:
                    self.db_manager.log_system_metrics(metrics)
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(interval)

# ======================== SECURE FILE SCANNER ========================

class SecureFileScanner:
    """Secure file scanning with proper validation."""
    
    def __init__(self, config_manager: ConfigManager, db_manager: DatabaseManager):
        self.config = config_manager
        self.db_manager = db_manager
        self.safe_extensions = {'.txt', '.log', '.json', '.xml', '.csv', '.md'}
        self.executable_extensions = {'.py', '.sh', '.exe', '.bat', '.cmd', '.jar'}
        self.whitelist_paths = set()
        self.load_whitelist()
    
    def load_whitelist(self):
        """Load whitelist of safe paths."""
        whitelist_file = os.path.join(BASE_DIR, "whitelist.txt")
        if os.path.exists(whitelist_file):
            try:
                with open(whitelist_file, 'r') as f:
                    self.whitelist_paths = {line.strip() for line in f if line.strip()}
            except Exception as e:
                logging.error(f"Failed to load whitelist: {e}")
    
    def scan_directory(self, directory: str, max_depth: int = 3) -> List[dict]:
        """Safely scan directory for files."""
        results = []
        depth_limit = self.config.getint('security', 'scan_depth_limit', 3)
        
        try:
            for root, dirs, files in os.walk(directory):
                # Check depth
                current_depth = root.replace(directory, '').count(os.sep)
                if current_depth >= min(max_depth, depth_limit):
                    dirs[:] = []  # Don't go deeper
                    continue
                
                # Skip hidden and system directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'cache']]
                
                for file in files:
                    if file.startswith('.'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    file_info = self.analyze_file(file_path)
                    if file_info:
                        results.append(file_info)
        
        except Exception as e:
            logging.error(f"Directory scan failed: {e}")
        
        return results
    
    def analyze_file(self, file_path: str) -> Optional[dict]:
        """Analyze a single file safely."""
        try:
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                return None
            
            stat = os.stat(file_path)
            file_ext = Path(file_path).suffix.lower()
            
            # Calculate file hash for integrity
            file_hash = self.calculate_file_hash(file_path)
            
            # Determine if file is executable
            is_executable = (
                file_ext in self.executable_extensions or
                os.access(file_path, os.X_OK)
            )
            
            # Determine if file is safe
            is_safe = (
                file_ext in self.safe_extensions or
                file_path in self.whitelist_paths or
                stat.st_size == 0
            )
            
            file_info = {
                'path': file_path,
                'name': os.path.basename(file_path),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'extension': file_ext,
                'is_executable': is_executable,
                'is_safe': is_safe,
                'hash': file_hash,
                'analyzed_at': datetime.now().isoformat()
            }
            
            # Log to database
            self.db_manager.execute_query(
                '''INSERT INTO file_scans 
                   (timestamp, file_path, file_type, file_size, file_hash, is_executable, is_safe)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (file_info['analyzed_at'], file_path, file_ext, stat.st_size, 
                 file_hash, is_executable, is_safe)
            )
            
            return file_info
            
        except Exception as e:
            logging.error(f"File analysis failed for {file_path}: {e}")
            return None
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return "unknown"

# ======================== SIMPLE ML ANALYTICS ========================

class SimpleAnalytics:
    """Simple analytics using real ML libraries (if available)."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.ml_available = NUMPY_AVAILABLE and SKLEARN_AVAILABLE
    
    def analyze_system_patterns(self) -> dict:
        """Analyze system metrics patterns using simple ML."""
        if not self.ml_available:
            return {"error": "ML libraries not available", "ml_enabled": False}
        
        try:
            # Get recent metrics
            metrics = self.db_manager.get_recent_metrics(hours=168)  # 1 week
            
            if len(metrics) < 10:
                return {"error": "Insufficient data for analysis", "ml_enabled": True}
            
            # Prepare data
            data = np.array([
                [m['cpu_percent'], m['memory_percent'], m['disk_percent']]
                for m in metrics
            ])
            
            # Simple clustering to find patterns
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Calculate statistics
            results = {
                "ml_enabled": True,
                "data_points": len(metrics),
                "avg_cpu": float(np.mean(data[:, 0])),
                "avg_memory": float(np.mean(data[:, 1])),
                "avg_disk": float(np.mean(data[:, 2])),
                "cpu_trend": "stable",  # Simple trend analysis
                "memory_trend": "stable",
                "clusters_found": len(set(clusters)),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Simple trend detection
            if len(metrics) >= 20:
                recent_cpu = np.mean([m['cpu_percent'] for m in metrics[:10]])
                older_cpu = np.mean([m['cpu_percent'] for m in metrics[-10:]])
                if recent_cpu > older_cpu * 1.1:
                    results["cpu_trend"] = "increasing"
                elif recent_cpu < older_cpu * 0.9:
                    results["cpu_trend"] = "decreasing"
            
            return results
            
        except Exception as e:
            logging.error(f"Analytics failed: {e}")
            return {"error": str(e), "ml_enabled": True}

# ======================== TASK MANAGER ========================

class TaskManager:
    """Manages and executes tasks safely."""
    
    def __init__(self, config_manager: ConfigManager, db_manager: DatabaseManager):
        self.config = config_manager
        self.db_manager = db_manager
        self.running_tasks = {}
        self.task_history = deque(maxlen=100)
    
    def execute_safe_task(self, task_name: str, task_func, *args, **kwargs) -> bool:
        """Execute a task safely with logging."""
        start_time = time.time()
        task_id = f"{task_name}_{int(start_time)}"
        
        try:
            logging.info(f"Starting task: {task_name}")
            self.running_tasks[task_id] = {
                'name': task_name,
                'start_time': start_time,
                'status': 'running'
            }
            
            result = task_func(*args, **kwargs)
            
            duration = time.time() - start_time
            self.running_tasks[task_id]['status'] = 'completed'
            
            self.db_manager.log_task_execution(task_name, 'success', duration)
            self.task_history.append({
                'name': task_name,
                'status': 'success',
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            })
            
            logging.info(f"Task completed: {task_name} ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.running_tasks[task_id]['status'] = 'failed'
            
            self.db_manager.log_task_execution(task_name, 'failed', duration, str(e))
            self.task_history.append({
                'name': task_name,
                'status': 'failed',
                'duration': duration,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            logging.error(f"Task failed: {task_name} - {e}")
            return False
        
        finally:
            self.running_tasks.pop(task_id, None)

# ======================== MAIN NEXUS SYSTEM ========================

class NexusSystem:
    """Main system class that coordinates all components."""
    
    def __init__(self):
        self.setup_logging()
        self.config = ConfigManager()
        self.db_manager = DatabaseManager(DATABASE_FILE)
        self.system_monitor = SystemMonitor(self.db_manager)
        self.file_scanner = SecureFileScanner(self.config, self.db_manager)
        self.analytics = SimpleAnalytics(self.db_manager)
        self.task_manager = TaskManager(self.config, self.db_manager)
        self.running = False
    
    def setup_logging(self):
        """Setup proper logging."""
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            log_file = os.path.join(LOG_DIR, f"nexus_{datetime.now().strftime('%Y%m%d')}.log")
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            
        except Exception as e:
            logging.basicConfig(level=logging.WARNING)
            logging.warning(f"Failed to setup file logging: {e}")
    
    def start(self):
        """Start the Nexus system."""
        self.running = True
        logging.info("Nexus System starting...")
        
        # Start system monitoring
        interval = self.config.getint('system', 'scan_interval_seconds', 60)
        self.system_monitor.start_monitoring(interval)
        
        # Start main loop
        self.main_loop()
    
    def stop(self):
        """Stop the Nexus system."""
        self.running = False
        self.system_monitor.stop_monitoring()
        logging.info("Nexus System stopped")
    
    def main_loop(self):
        """Main interactive loop."""
        try:
            while self.running:
                self.display_menu()
                choice = input(f"{Colors.CYAN}Select option: {Colors.RESET}").strip()
                
                if choice == '1':
                    self.show_system_status()
                elif choice == '2':
                    self.scan_files()
                elif choice == '3':
                    self.show_analytics()
                elif choice == '4':
                    self.show_recent_activity()
                elif choice == '5':
                    self.system_maintenance()
                elif choice == 'c':
                    self.show_configuration()
                elif choice == 'q':
                    break
                else:
                    print(f"{Colors.RED}Invalid option{Colors.RESET}")
                
                if choice != 'q':
                    input(f"\n{Colors.YELLOW}Press ENTER to continue...{Colors.RESET}")
        
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Shutdown requested{Colors.RESET}")
        finally:
            self.stop()
    
    def display_menu(self):
        """Display the main menu."""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'NEXUS SYSTEM MANAGER':^60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{f'Version {APP_VERSION}':^60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")
        
        print(f"{Colors.GREEN}[1]{Colors.RESET} System Status & Monitoring")
        print(f"{Colors.GREEN}[2]{Colors.RESET} Scan Files & Directories")
        print(f"{Colors.GREEN}[3]{Colors.RESET} Analytics & Patterns")
        print(f"{Colors.GREEN}[4]{Colors.RESET} Recent Activity Log")
        print(f"{Colors.GREEN}[5]{Colors.RESET} System Maintenance")
        print(f"{Colors.CYAN}[c]{Colors.RESET} Configuration")
        print(f"{Colors.RED}[q]{Colors.RESET} Quit")
        print()
    
    def show_system_status(self):
        """Show current system status."""
        print(f"\n{Colors.BOLD}System Status{Colors.RESET}")
        print("="*40)
        
        metrics = self.system_monitor.get_system_metrics()
        if metrics:
            print(f"CPU Usage: {Colors.YELLOW}{metrics['cpu_percent']:.1f}%{Colors.RESET}")
            print(f"Memory Usage: {Colors.YELLOW}{metrics['memory_percent']:.1f}%{Colors.RESET}")
            print(f"Disk Usage: {Colors.YELLOW}{metrics['disk_percent']:.1f}%{Colors.RESET}")
            print(f"Process Count: {metrics['process_count']}")
            print(f"Network Connections: {metrics['network_connections']}")
            print(f"CPU Cores: {metrics['cpu_count']}")
            print(f"Total Memory: {metrics['memory_total_gb']:.1f} GB")
            print(f"Available Memory: {metrics['memory_available_gb']:.1f} GB")
        else:
            print(f"{Colors.RED}Failed to get system metrics{Colors.RESET}")
    
    def scan_files(self):
        """Scan files in a directory."""
        print(f"\n{Colors.BOLD}File Scanner{Colors.RESET}")
        print("="*40)
        
        directory = input("Enter directory to scan (or press ENTER for current): ").strip()
        if not directory:
            directory = os.getcwd()
        
        if not os.path.exists(directory):
            print(f"{Colors.RED}Directory does not exist{Colors.RESET}")
            return
        
        print(f"Scanning: {directory}")
        
        def scan_task():
            return self.file_scanner.scan_directory(directory)
        
        success = self.task_manager.execute_safe_task("file_scan", scan_task)
        
        if success:
            results = scan_task()
            print(f"\n{Colors.GREEN}Scan completed. Found {len(results)} files{Colors.RESET}")
            
            safe_count = sum(1 for r in results if r['is_safe'])
            executable_count = sum(1 for r in results if r['is_executable'])
            
            print(f"Safe files: {safe_count}")
            print(f"Executable files: {executable_count}")
            
            if executable_count > 0:
                print(f"\n{Colors.YELLOW}Executable files found:{Colors.RESET}")
                for result in results:
                    if result['is_executable']:
                        status = "SAFE" if result['is_safe'] else "REVIEW"
                        color = Colors.GREEN if result['is_safe'] else Colors.YELLOW
                        print(f"  {color}{status}{Colors.RESET}: {result['name']}")
    
    def show_analytics(self):
        """Show analytics and patterns."""
        print(f"\n{Colors.BOLD}System Analytics{Colors.RESET}")
        print("="*40)
        
        def analytics_task():
            return self.analytics.analyze_system_patterns()
        
        success = self.task_manager.execute_safe_task("analytics", analytics_task)
        
        if success:
            results = analytics_task()
            
            if "error" in results:
                print(f"{Colors.RED}Analytics Error: {results['error']}{Colors.RESET}")
                print(f"ML Libraries Available: {results.get('ml_enabled', False)}")
            else:
                print(f"ML Analysis: {Colors.GREEN}ENABLED{Colors.RESET}")
                print(f"Data Points Analyzed: {results['data_points']}")
                print(f"Average CPU Usage: {results['avg_cpu']:.1f}%")
                print(f"Average Memory Usage: {results['avg_memory']:.1f}%")
                print(f"Average Disk Usage: {results['avg_disk']:.1f}%")
                print(f"CPU Trend: {results['cpu_trend']}")
                print(f"Memory Trend: {results['memory_trend']}")
                print(f"Usage Patterns Found: {results['clusters_found']}")
    
    def show_recent_activity(self):
        """Show recent system activity."""
        print(f"\n{Colors.BOLD}Recent Activity{Colors.RESET}")
        print("="*40)
        
        # Show recent tasks
        if self.task_manager.task_history:
            print(f"\n{Colors.YELLOW}Recent Tasks:{Colors.RESET}")
            for task in list(self.task_manager.task_history)[-10:]:
                status_color = Colors.GREEN if task['status'] == 'success' else Colors.RED
                print(f"  {status_color}{task['status'].upper()}{Colors.RESET}: {task['name']} ({task['duration']:.2f}s)")
        
        # Show recent metrics
        recent_metrics = self.db_manager.get_recent_metrics(hours=1)
        if recent_metrics:
            print(f"\n{Colors.YELLOW}Recent System Metrics:{Colors.RESET}")
            for metric in recent_metrics[:5]:
                timestamp = datetime.fromisoformat(metric['timestamp']).strftime('%H:%M:%S')
                print(f"  {timestamp}: CPU {metric['cpu_percent']:.1f}%, "
                      f"MEM {metric['memory_percent']:.1f}%, "
                      f"DISK {metric['disk_percent']:.1f}%")
    
    def system_maintenance(self):
        """Perform system maintenance tasks."""
        print(f"\n{Colors.BOLD}System Maintenance{Colors.RESET}")
        print("="*40)
        
        # Clean old logs
        def cleanup_task():
            retention_days = self.config.getint('system', 'log_retention_days', 7)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Clean database records
            self.db_manager.execute_query(
                "DELETE FROM system_metrics WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            
            # Clean log files
            if os.path.exists(LOG_DIR):
                for log_file in os.listdir(LOG_DIR):
                    file_path = os.path.join(LOG_DIR, log_file)
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_time < cutoff_date:
                            os.remove(file_path)
            
            return True
        
        success = self.task_manager.execute_safe_task("maintenance", cleanup_task)
        
        if success:
            print(f"{Colors.GREEN}Maintenance completed successfully{Colors.RESET}")
        else:
            print(f"{Colors.RED}Maintenance failed{Colors.RESET}")
    
    def show_configuration(self):
        """Show and modify configuration."""
        print(f"\n{Colors.BOLD}System Configuration{Colors.RESET}")
        print("="*40)
        
        print(f"Config File: {self.config.config_file}")
        print(f"Database: {DATABASE_FILE}")
        print(f"Base Directory: {BASE_DIR}")
        print(f"Log Directory: {LOG_DIR}")
        print()
        
        print("Current Settings:")
        for section in self.config.config.sections():
            print(f"\n[{section}]")
            for key, value in self.config.config[section].items():
                print(f"  {key} = {value}")

def main():
    """Main entry point."""
    try:
        nexus = NexusSystem()
        nexus.start()
    except Exception as e:
        logging.error(f"System startup failed: {e}")
        print(f"{Colors.RED}Failed to start Nexus System: {e}{Colors.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()


