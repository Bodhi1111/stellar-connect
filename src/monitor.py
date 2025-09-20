# src/monitor.py
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.ingestion import process_new_file

# Define the directories relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PATH_TO_WATCH = os.path.join(PROJECT_ROOT, 'incoming_transcripts')
ARCHIVE_PATH = os.path.join(PROJECT_ROOT, 'archive')

class TranscriptHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.txt', '.pdf', '.docx')):
            print(f"\n[Monitor] New file detected: {event.src_path}")
            time.sleep(2) # Wait briefly for file write completion
            try:
                process_new_file(event.src_path)
                self.archive_file(event.src_path)
            except Exception as e:
                print(f"[Monitor] Error processing file {event.src_path}: {e}")

    def archive_file(self, file_path):
        os.makedirs(ARCHIVE_PATH, exist_ok=True)
        try:
            new_path = os.path.join(ARCHIVE_PATH, os.path.basename(file_path))
            # Handle potential file name collisions
            if os.path.exists(new_path):
                base, ext = os.path.splitext(os.path.basename(file_path))
                new_path = os.path.join(ARCHIVE_PATH, f"{base}_{int(time.time())}{ext}")

            os.rename(file_path, new_path)
            print(f"[Monitor] Moved processed file to archive.")
        except OSError as e:
            print(f"[Monitor] Error archiving file: {e}")

def start_monitoring():
    os.makedirs(PATH_TO_WATCH, exist_ok=True)
    event_handler = TranscriptHandler()
    observer = Observer()
    observer.schedule(event_handler, PATH_TO_WATCH, recursive=False)
    observer.start()
    print(f"[Monitor] Watching for new files in: {PATH_TO_WATCH}")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_monitoring()