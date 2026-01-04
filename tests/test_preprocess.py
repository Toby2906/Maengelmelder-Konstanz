import os
from nlp.preprocess import load_local_stopwords, preprocess_text


def test_load_local_stopwords_returns_iterables():
    s, p = load_local_stopwords()
    assert isinstance(s, set)
    assert isinstance(p, list)


def test_preprocess_text_basic(tmp_path):
    # create a small temporary stopwords file to simulate phrases
    sw_file = tmp_path / 'stopwords_de.txt'
    sw_file.write_text('und\nstadt konstanz\n')

    # load stopwords from that path by temporarily changing cwd
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        s, p = load_local_stopwords(str(sw_file))
    finally:
        os.chdir(cwd)

    text = 'Die Stadt Konstanz und das ist ein Test.'
    out = preprocess_text(text, s, p)
    # expected to remove 'und' and phrase 'stadt konstanz'
    assert 'und' not in out
    assert 'stadt' not in out
