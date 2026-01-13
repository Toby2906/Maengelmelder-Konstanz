import pytest
import csv

@pytest.fixture
def sample_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Datum", "Beschreibung"])
        writer.writerow(["1", "2024-01-01", "Test Beschreibung"])
    return str(csv_file)
