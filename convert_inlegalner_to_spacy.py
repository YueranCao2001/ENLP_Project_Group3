import spacy, tqdm
from datasets import load_dataset
from spacy.tokens import DocBin

nlp = spacy.blank("en")

def build_ents(text, results):
    """results: annotations[0]['result'], with start/end/labels"""
    ents = []
    for item in results:
        value = item["value"]
        start_char = value["start"]
        end_char   = value["end"]
        label = value["labels"][0]
        span = nlp.make_doc(text).char_span(start_char, end_char, label=label, alignment_mode="contract")
        if span:
            ents.append(span)
    return spacy.util.filter_spans(ents)

def convert_split(split_name, ds_split):
    db = DocBin(store_user_data=True)
    for rec in tqdm.tqdm(ds_split, desc=split_name):
        text = rec["data"]["text"]
        results = rec["annotations"][0]["result"]
        doc = nlp.make_doc(text)
        doc.ents = build_ents(text, results)
        db.add(doc)
    db.to_disk(f"{split_name}.spacy")
    print(f"{split_name}: {len(db)} docs saved")

if __name__ == "__main__":
    ds = load_dataset("opennyaiorg/InLegalNER")
    convert_split("train",       ds["train"])
    convert_split("dev",         ds["dev"])
    convert_split("test",        ds["test"])
