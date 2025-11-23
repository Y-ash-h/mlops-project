# /opt/airflow/mlops-pipeline/src/data_ingestion/ingest.py
import os
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Minimal ingestion that copies CSV files from RAW_DIR into a central data/raw folder
    and returns a list of file paths saved.
    """
    def __init__(self, raw_dir: str = None, out_dir: str = None):
        if out_dir is None:
            out_dir = os.environ.get("DATA_DIR", "/opt/airflow/mlops-pipeline/data")
        if raw_dir is None:
            raw_dir = os.path.join(out_dir, "raw")

        self.raw_dir = raw_dir
        self.out_dir = out_dir
        # write ingested copies into an 'ingested' directory inside out_dir to avoid copying onto source
        self.raw_out = os.path.join(out_dir, "ingested")
        os.makedirs(self.raw_out, exist_ok=True)

    def load_and_save(self):
        found = []
        if not os.path.exists(self.raw_dir):
            logger.warning("Raw dir %s does not exist", self.raw_dir)
            return found

        for fname in os.listdir(self.raw_dir):
            if fname.lower().endswith(".csv"):
                src = os.path.join(self.raw_dir, fname)
                dst = os.path.join(self.raw_out, fname)
                # guard: if src and dst are same file (path equality), skip copy
                try:
                    src_abspath = os.path.abspath(src)
                    dst_abspath = os.path.abspath(dst)
                except Exception:
                    src_abspath = src
                    dst_abspath = dst

                if src_abspath == dst_abspath:
                    logger.info("Skipping copy for %s because source and destination are identical", src)
                    found.append(dst)
                    continue

                shutil.copyfile(src, dst)
                found.append(dst)
                logger.info("Copied raw %s -> %s", src, dst)
        return found
