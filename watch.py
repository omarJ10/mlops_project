#!/usr/bin/env python3
"""
Automatic File Watcher for ML Pipeline
This script monitors changes to model and pipeline files and automatically
re-runs the ML pipeline when changes are detected.

Usage:
    python watch.py
    or
    make watch
"""

import sys
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class PipelineFileHandler(FileSystemEventHandler):
    """Handler for file system events that triggers pipeline re-execution."""

    def __init__(self, files_to_watch, cooldown_seconds=3):
        """
        Initialize the handler.

        Args:
            files_to_watch (list): List of file paths to monitor.
            cooldown_seconds (int): Minimum seconds between pipeline runs.
        """
        super().__init__()
        self.files_to_watch = [Path(f).resolve() for f in files_to_watch]
        self.cooldown_seconds = cooldown_seconds
        self.last_run_time = 0

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        # Check if the modified file is in our watch list
        modified_file = Path(event.src_path).resolve()
        if modified_file in self.files_to_watch:
            # Implement cooldown to avoid multiple rapid runs
            current_time = time.time()
            if current_time - self.last_run_time < self.cooldown_seconds:
                return

            self.last_run_time = current_time
            print("\n" + "=" * 70)
            print(f"[WATCH] File changed: {modified_file.name}")
            print("=" * 70)
            print("[WATCH] Running pipeline...")
            self.run_pipeline()

    def run_pipeline(self):
        """Execute the ML pipeline."""
        try:
            # Run the pipeline using make command
            result = subprocess.run(
                ["make", "pipeline"],
                capture_output=False,
                text=True,
                cwd=Path(__file__).parent
            )

            if result.returncode == 0:
                print("\n" + "=" * 70)
                print("[WATCH] ✓ Pipeline completed successfully!")
                print("=" * 70)
            else:
                print("\n" + "=" * 70)
                print("[WATCH] ✗ Pipeline failed!")
                print("=" * 70)

        except Exception as e:
            print(f"\n[WATCH] Error running pipeline: {str(e)}")


def main():
    """Main function to start the file watcher."""
    # Files to monitor
    files_to_watch = [
        "model_pipeline.py",
        "main.py",
        "drug200.csv"
    ]

    # Verify files exist
    existing_files = []
    for file in files_to_watch:
        file_path = Path(file)
        if file_path.exists():
            existing_files.append(file)
            print(f"[WATCH] Monitoring: {file}")
        else:
            print(f"[WATCH] Warning: {file} not found, skipping...")

    if not existing_files:
        print("[WATCH] Error: No files to watch!")
        sys.exit(1)

    # Set up the observer
    event_handler = PipelineFileHandler(existing_files, cooldown_seconds=3)
    observer = Observer()

    # Watch the current directory
    watch_path = Path(".").resolve()
    observer.schedule(event_handler, path=str(watch_path), recursive=False)

    print("\n" + "=" * 70)
    print(" " * 20 + "FILE WATCHER STARTED")
    print("=" * 70)
    print(f"[WATCH] Watching directory: {watch_path}")
    print(f"[WATCH] Monitoring {len(existing_files)} file(s)")
    print("[WATCH] Press Ctrl+C to stop...")
    print("=" * 70 + "\n")

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("[WATCH] Stopping file watcher...")
        print("=" * 70)
        observer.stop()

    observer.join()
    print("[WATCH] File watcher stopped.")


if __name__ == "__main__":
    main()
