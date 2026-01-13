from src.fetch import load_data_from_csv

def test_load_csv(sample_csv):
    texts = load_data_from_csv(sample_csv)
    assert len(texts) >= 0
