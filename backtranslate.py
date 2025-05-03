#!/usr/bin/env python3
"""
scripts/backtranslate.py

A robust script to create a back-translated augmented training set for spaCy NER.
Uses placeholder masking to preserve entity spans during translation, with safe truncation and error handling.
"""
import random
import re
import spacy
from transformers import pipeline
from spacy.tokens import DocBin
from spacy.util import filter_spans

def back_translate_spacy(input_spacy: str, output_spacy: str, ratio: float = 0.2, max_length: int = 512):
    """
    Reads a spaCy DocBin, selects `ratio` fraction of docs for augmentation via back-translation,
    masks entity spans to preserve annotations, translates en->zh->en with truncation,
    reconstructs docs with aligned spans, falls back on errors.
    """
    # Load blank English pipeline for DocBin operations
    nlp_blank = spacy.blank("en")
    docbin = DocBin().from_disk(input_spacy)
    docs = list(docbin.get_docs(nlp_blank.vocab))

    # Determine which docs to augment
    num_augment = int(len(docs) * ratio)
    augment_indices = set(random.sample(range(len(docs)), num_augment))

    # Initialize translation pipelines
    en2zh = pipeline(
        "translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh", device=-1
    )
    zh2en = pipeline(
        "translation_zh_to_en", model="Helsinki-NLP/opus-mt-zh-en", device=-1
    )

    new_docbin = DocBin()
    for idx, doc in enumerate(docs):
        if idx in augment_indices and doc.ents:
            text = doc.text
            # Mask entities with placeholders
            masked = ""
            last = 0
            placeholders = []
            for ent_id, ent in enumerate(doc.ents):
                placeholder = f"<ENT{ent_id}>"
                placeholders.append((placeholder, ent.text, ent.label_))
                masked += text[last:ent.start_char] + placeholder
                last = ent.end_char
            masked += text[last:]

            # Try back-translation with truncation
            try:
                zh = en2zh(
                    masked, max_length=max_length, truncation=True
                )[0]["translation_text"]
                back = zh2en(
                    zh, max_length=max_length, truncation=True
                )[0]["translation_text"]
            except Exception as e:
                print(f"Warning: back-translation failed for doc {idx}: {e}. Using original.")
                new_docbin.add(doc)
                continue

            # Restore original entities in back-translated text
            for placeholder, orig_text, _ in placeholders:
                back = back.replace(placeholder, orig_text)

            # Build new doc and align spans
            new_doc = nlp_blank.make_doc(back)
            spans_list = []
            for _, orig_text, label in placeholders:
                for match in re.finditer(re.escape(orig_text), back):
                    span = new_doc.char_span(
                        match.start(), match.end(), label, alignment_mode="contract"
                    )
                    if span is not None:
                        spans_list.append(span)
            # Filter overlapping spans
            new_doc.ents = filter_spans(spans_list)
            new_docbin.add(new_doc)
        else:
            new_docbin.add(doc)

    new_docbin.to_disk(output_spacy)
    print(f"Back-translated {len(augment_indices)} docs; saved to {output_spacy}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Back-translate spaCy training data with robust span alignment"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to train.spacy"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to output train_bt.spacy"
    )
    parser.add_argument(
        "--ratio", "-r", type=float, default=0.2,
        help="Fraction to augment (e.g. 0.2 for 20%)"
    )
    parser.add_argument(
        "--max_length", "-m", type=int, default=512,
        help="Max token length for translation truncation"
    )
    args = parser.parse_args()

    back_translate_spacy(
        args.input, args.output, args.ratio, args.max_length
    )
