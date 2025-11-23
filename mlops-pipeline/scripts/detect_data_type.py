#!/usr/bin/env python3
import argparse
from src.utils.data_router import detect_data_type

def main():
    parser = argparse.ArgumentParser(description="Detect dataset modality")
    parser.add_argument("--data-dir", "-d", required=True, help="Dataset folder")
    args = parser.parse_args()

    dtype = detect_data_type(args.data_dir)
    print(dtype)

if __name__ == "__main__":
    main()
