#!/bin/bash
# Download necessary corpora/models for TextBlob and spaCy

python -m textblob.download_corpora
python -m spacy download en_core_web_sm

# Then start your Flask app
python server.py
