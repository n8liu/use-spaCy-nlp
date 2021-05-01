import spacy

nlp = spacy.load("en_core_web_sm")

def read_my_sentence(sentence):
    doc = nlp(sentence)
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
    return None 

read_my_sentence("As of v2.0, spaCy comes with neural network models "
                 "that are implemented in our machine learning library,"
                 "Thinc. For GPU support, we’ve been grateful to use the "
                 "work of Chainer’s CuPy module, which provides a "
                 "numpy-compatible interface for GPU arrays.")