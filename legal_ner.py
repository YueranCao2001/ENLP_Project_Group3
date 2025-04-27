######### Definitions of spacy components needed for legal NER
import spacy
from spacy import displacy
import re
from data_preparation import seperate_and_clean_preamble,get_text_from_indiankanoon_url,get_sentence_docs
from postprocessing_utils import postprocessing,get_csv
from spacy.tokens import Span
import time
from postprocessing_utils import get_unique_precedent_count,get_unique_statute_count,get_unique_provision_count
from wasabi import msg
import urllib
import urllib.request
import ssl
import certifi

def extract_entities_from_judgment_text(txt, legal_nlp, nlp_preamble_splitting, text_type, do_postprocess):
    ######### Seperate Preamble and judgment text
    seperation_start_time = time.time()
    preamble_text, preamble_end = seperate_and_clean_preamble(txt, nlp_preamble_splitting)
    print("Seperating Preamble took " + str(time.time() - seperation_start_time))

    ########## process main judgement text
    judgement_start_time = time.time()
    judgement_text = txt[preamble_end:]
    #####  replace new lines in middle of sentence with spaces.
    judgement_text = re.sub(r'(\w[ -]*)(\n+)', r'\1 ', judgement_text)
    judgment_doc = nlp_preamble_splitting(judgement_text)
    if text_type == 'doc':
        doc_judgment = legal_nlp(judgement_text)
    else:
        doc_judgment = get_sentence_docs(judgment_doc, legal_nlp)

    print("Creating doc for judgement  took " + str(time.time() - judgement_start_time))
    print(len(doc_judgment.ents))

    ######### process preamable
    preamble_start_time = time.time()
    doc_preamble = legal_nlp(preamble_text)
    print("Creating doc for preamble  took " + str(time.time() - preamble_start_time))

    ######### Combine preamble doc & judgement doc
    combining_start_time = time.time()
    combined_doc = spacy.tokens.Doc.from_docs([doc_preamble, doc_judgment])
    print("Combining took " + str(time.time() - combining_start_time))

    # Preserve entities during merging
    all_ents = []
    if doc_preamble.ents:
        all_ents.extend([Span(combined_doc, e.start, e.end, e.label_) for e in doc_preamble.ents])
    if doc_judgment.ents:
        # Adjust the start position for entities from doc_judgment
        offset = len(doc_preamble)
        all_ents.extend([Span(combined_doc, e.start + offset, e.end + offset, e.label_) 
                         for e in doc_judgment.ents])
    
    # Set the entities on the combined document
    if all_ents:
        combined_doc.ents = spacy.util.filter_spans(all_ents)

    try:
        if do_postprocess:
            combined_doc = postprocessing(combined_doc)
    except Exception as e:
        print(f"Postprocessing error: {str(e)}")
        msg.warn('There was some issue while performing postprocessing, skipping postprocessing...')
    
    # Return the combined document
    return combined_doc




if __name__ == "__main__":
    #indiankanoon_url = 'https://indiankanoon.org/doc/11757180/'
    #txt = get_text_from_indiankanoon_url(indiankanoon_url)

    """changed url to same sample file, but from github
    had to use some extra packages to get files to load correctly for the model
    """
    
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    txt = urllib.request.urlopen(
    'https://raw.githubusercontent.com/OpenNyAI/Opennyai/master/samples/sample_judgment1.txt',
    context=ssl_context
).read().decode()

    legal_nlp = spacy.load('en_legal_ner_trf') ## path of trained model files
    preamble_spiltting_nlp = spacy.load('en_core_web_sm') #### only for splitting the preamble and judgment when keywords are not found
    ########## Extract Entities
    combined_doc = extract_entities_from_judgment_text(txt,legal_nlp,preamble_spiltting_nlp,text_type='doc',do_postprocess=True)

    ########### show the entities
    extracted_ent_labels = list(set([i.label_ for i in combined_doc.ents]))

    ###To save results as csv
    #get_csv(str(indiankanoon_url),combined_doc,save_path='')

    colors = {'COURT': "#bbabf2", 'PETITIONER': "#f570ea", "RESPONDENT": "#cdee81", 'JUDGE': "#fdd8a5",
              "LAWYER": "#f9d380", 'WITNESS': "violet", "STATUTE": "#faea99", "PROVISION": "yellow",
              'CASE_NUMBER': "#fbb1cf", "PRECEDENT": "#fad6d6", 'DATE': "#b1ecf7", 'OTHER_PERSON': "#b0f6a2",
              'ORG': '#a57db5', 'GPE': '#7fdbd4'}
    options = {"ents": extracted_ent_labels, "colors": colors}
    #import pdb;pdb.set_trace()


    displacy.serve(combined_doc, style='ent',auto_select_port = True,options=options)
