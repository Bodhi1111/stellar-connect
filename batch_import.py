#!/usr/bin/env python3
"""
Batch Import Script for McAdams Transcripts
Processes all existing transcripts from the McAdams Transcripts folder
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ingestion import process_new_file
from src.config import CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_import.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def batch_import_transcripts(transcripts_dir: str):
    """
    Import all transcript files from the specified directory

    Args:
        transcripts_dir: Path to the directory containing transcript files
    """
    transcripts_path = Path(transcripts_dir)

    if not transcripts_path.exists():
        logger.error(f"Transcripts directory does not exist: {transcripts_dir}")
        return False

    # Find all .txt files in the directory
    transcript_files = list(transcripts_path.glob("*.txt"))

    if not transcript_files:
        logger.warning(f"No .txt files found in {transcripts_dir}")
        return False

    logger.info(f"Found {len(transcript_files)} transcript files to process")

    successful_imports = 0
    failed_imports = 0

    for i, file_path in enumerate(transcript_files, 1):
        try:
            logger.info(f"Processing [{i}/{len(transcript_files)}]: {file_path.name}")

            # Skip files that are too small (likely empty or corrupted)
            file_size = file_path.stat().st_size
            if file_size < 100:  # Less than 100 bytes
                logger.warning(f"Skipping {file_path.name} - file too small ({file_size} bytes)")
                failed_imports += 1
                continue

            # Process the file
            result = process_new_file(str(file_path))

            if result:
                successful_imports += 1
                logger.info(f"✅ Successfully processed: {file_path.name}")
            else:
                failed_imports += 1
                logger.error(f"❌ Failed to process: {file_path.name}")

        except Exception as e:
            failed_imports += 1
            logger.error(f"❌ Error processing {file_path.name}: {str(e)}")
            continue

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"BATCH IMPORT SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total files found: {len(transcript_files)}")
    logger.info(f"Successfully imported: {successful_imports}")
    logger.info(f"Failed imports: {failed_imports}")
    logger.info(f"Success rate: {(successful_imports/len(transcript_files))*100:.1f}%")
    logger.info(f"{'='*50}")

    return successful_imports > 0

def main():
    """Main execution function"""
    logger.info("Starting batch import of McAdams transcripts...")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Path to the transcripts directory
    transcripts_dir = "/Users/joshuavaughan/Documents/McAdams Transcripts"

    # Verify the directory exists
    if not os.path.exists(transcripts_dir):
        logger.error(f"Transcripts directory not found: {transcripts_dir}")
        logger.error("Please verify the path is correct")
        return False

    # Start the import process
    success = batch_import_transcripts(transcripts_dir)

    if success:
        logger.info("🎉 Batch import completed successfully!")
        logger.info("Your McAdams transcripts are now available in Stellar Connect")
    else:
        logger.error("❌ Batch import failed or no files were processed")

    return success

if __name__ == "__main__":
    main()