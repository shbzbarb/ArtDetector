import csv
from pathlib import Path


class CSVLogger:
    """Minimal CSV logger that appends rows and writes a header once"""
    def __init__(self, csv_path: Path, fieldnames):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self.fieldnames = list(fieldnames)
        write_header = not csv_path.exists()
        self.file = open(csv_path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if write_header:
            self.writer.writeheader()

    def log(self, row: dict):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass
