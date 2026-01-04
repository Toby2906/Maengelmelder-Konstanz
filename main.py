"""Orchestrator: Lade Daten, bereinige, vektorisiere und extrahiere Themen."""
import sys
import argparse
from nlp.fetch import load_data_from_csv
from nlp.preprocess import prepare_stopwords_and_lemmatizer, preprocess_text
from nlp.topics import vectorize_texts, extract_topics_lda, extract_ngrams, top_words


def run(filepath='data/maengelmelder_konstanz.csv', num_topics=5):
    print("Lade Daten...")
    texts = load_data_from_csv(filepath)
    print(f"Geladen: {len(texts)} Texte\n")

    print("Initialisiere Vorverarbeitung...")
    stop_words, stop_phrases, lemmatizer = prepare_stopwords_and_lemmatizer()

    print("Vorverarbeite Texte...")
    clean_texts = [preprocess_text(t, stop_words, stop_phrases) for t in texts]
    clean_texts = [t for t in clean_texts if t.strip()]
    print(f"Bereinigt: {len(clean_texts)} Texte\n")

    if len(clean_texts) == 0:
        print("Fehler: Keine Texte zum Verarbeiten. Bitte prüfe die Datei oder den Pfad.")
        return 1

    print("" + "=" * 60)
    print("VEKTORISIERUNG DER TEXTE")
    print("" + "=" * 60)

    count_matrix, tfidf_matrix, feature_names_count, feature_names_tfidf = vectorize_texts(clean_texts)

    print("\nTHEMENEXTRAKTION\n")
    topics = extract_topics_lda(count_matrix, feature_names_count, num_topics=num_topics)
    if topics:
        for idx, words in topics:
            print(f"\n   Thema {idx}: {', '.join(words)}")

    print("\nN-Gramm-Analyse (Bigramme)")
    bigrams = extract_ngrams(clean_texts)
    for i, (bg, cnt) in enumerate(bigrams, 1):
        print(f"   {i:2d}. '{bg}' ({cnt}x)")

    print("\nTop-Wörter:")
    for i, (w, c) in enumerate(top_words(clean_texts), 1):
        print(f"   {i:2d}. '{w}' ({c}x)")

    print("\nANALYSE ABGESCHLOSSEN")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NLP-Analyse: Lade CSV, preprocess, extrahiere Themen')
    parser.add_argument('input', nargs='?', default='data/maengelmelder_konstanz.csv', help='Pfad zur CSV-Datei (default: data/maengelmelder_konstanz.csv)')
    parser.add_argument('-t', '--topics', type=int, default=5, help='Anzahl Themen für LDA')
    args = parser.parse_args()
    sys.exit(run(args.input, num_topics=args.topics))
