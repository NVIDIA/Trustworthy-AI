#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ASL Data Postprocessing Script

This script redacts sensitive information (emails) and removes specified fields
from SuperAnnotate JSON/JSONL export files to prepare them for sharing.

Run this script with:
    python scripts/asl_data_postprocessing.py input.json
    python scripts/asl_data_postprocessing.py input.jsonl
    python scripts/asl_data_postprocessing.py input.json -o output.json
    python scripts/asl_data_postprocessing.py input.json --remove-fields field1 field2
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional, Set

# Configure logging
logger = logging.getLogger(__name__)


class DataPostprocessor:
    """Handles postprocessing of data including email redaction and field removal."""

    EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
    REDACTED_TEXT = "[REDACTED]"
    DEFAULT_FIELDS_TO_REMOVE = frozenset(
        ["movemark", "visibility", "qc_reviewed", "classId", "id", "projectId", "groupId"]
    )

    def __init__(self, fields_to_remove: Optional[Set[str]] = None):
        """
        Initialize the data postprocessor.

        Args:
            fields_to_remove: Set of field names to remove. If None, uses DEFAULT_FIELDS_TO_REMOVE.
        """
        self.fields_to_remove = fields_to_remove if fields_to_remove is not None else self.DEFAULT_FIELDS_TO_REMOVE

    def redact_emails(self, data: Any) -> Any:
        """Recursively redact emails in data structures."""
        if isinstance(data, dict):
            return {key: self.redact_emails(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.redact_emails(item) for item in data]
        elif isinstance(data, str) and self.EMAIL_PATTERN.search(data):
            return self.REDACTED_TEXT
        return data

    def remove_fields(self, data: Any) -> Any:
        """Recursively remove specified fields from data structures."""
        if isinstance(data, dict):
            return {key: self.remove_fields(value) for key, value in data.items() if key not in self.fields_to_remove}
        elif isinstance(data, list):
            return [self.remove_fields(item) for item in data]
        return data

    def process(self, data: Any) -> Any:
        """Process data by redacting emails and removing fields."""
        data = self.redact_emails(data)
        if self.fields_to_remove:
            data = self.remove_fields(data)
        return data


def is_jsonl_format(input_path: Path) -> bool:
    """Detect if a file is in JSONL format based on extension."""
    return input_path.suffix.lower() == ".jsonl"


def process_file(
    input_path: Path, output_path: Optional[Path] = None, fields_to_remove: Optional[Set[str]] = None
) -> None:
    """
    Process a JSON or JSONL file by redacting emails and removing specified fields.

    Args:
        input_path: Path to input file.
        output_path: Path to output file. If None, generates default name.
        fields_to_remove: Set of field names to remove. If None, uses defaults.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        json.JSONDecodeError: If input file contains invalid JSON.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_redacted{input_path.suffix}")

    processor = DataPostprocessor(fields_to_remove)

    logger.info(f"Processing file: {input_path}")
    if processor.fields_to_remove:
        logger.info(f"Removing fields: {', '.join(sorted(processor.fields_to_remove))}")

    if is_jsonl_format(input_path):
        _process_jsonl(input_path, output_path, processor)
    else:
        _process_json(input_path, output_path, processor)

    logger.info(f"Output saved to: {output_path}")


def _process_json(input_path: Path, output_path: Path, processor: DataPostprocessor) -> None:
    """Process a regular JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = processor.process(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)


def _process_jsonl(input_path: Path, output_path: Path, processor: DataPostprocessor) -> None:
    """Process a JSONL file (one JSON object per line)."""
    line_count = 0

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                processed_data = processor.process(data)
                json.dump(processed_data, outfile, ensure_ascii=False)
                outfile.write("\n")
                line_count += 1

                if line_count % 100 == 0:
                    logger.info(f"Processed {line_count} records...")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON on line {line_num}: {e}")
                raise

    logger.info(f"Processed {line_count} total records")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_fields_str = ", ".join(sorted(DataPostprocessor.DEFAULT_FIELDS_TO_REMOVE))

    parser = argparse.ArgumentParser(
        description="Redact emails and remove specified fields from SuperAnnotate JSON/JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.json
  %(prog)s input.jsonl
  %(prog)s input.json -o output.json
  %(prog)s input.jsonl -o output.jsonl
  %(prog)s input.json --remove-fields visibility movemark
  %(prog)s input.json --no-remove-fields
  %(prog)s input.json --verbose
        """,
    )

    parser.add_argument("input_file", type=Path, help="Path to the input JSON or JSONL file")

    parser.add_argument(
        "-o", "--output", dest="output_file", type=Path, help="Path to the output file (default: <input>_redacted.json)"
    )

    parser.add_argument(
        "--remove-fields",
        nargs="*",
        metavar="FIELD",
        help=f"Specify fields to remove (default: {default_fields_str})",
    )

    parser.add_argument("--no-remove-fields", action="store_true", help="Skip field removal step, only redact emails")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


def main() -> int:
    """Main entry point for the postprocessing script."""
    args = parse_arguments()
    setup_logging(args.verbose)

    # Determine fields to remove
    if args.no_remove_fields:
        fields_to_remove = set()
    elif args.remove_fields is not None:
        fields_to_remove = set(args.remove_fields)
    else:
        fields_to_remove = None  # Use defaults

    try:
        process_file(input_path=args.input_file, output_path=args.output_file, fields_to_remove=fields_to_remove)
        return 0
    except FileNotFoundError:
        logger.error("File not found", exc_info=True)
        return 1
    except json.JSONDecodeError:
        logger.error("Invalid JSON format", exc_info=True)
        return 1
    except Exception:
        logger.error("Unexpected error", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
