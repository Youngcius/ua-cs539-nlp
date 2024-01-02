# test.py

import spacy
import pandas as pd
table = pd.DataFrame

# Install spacy
#   pip install -U spacy
# Download models (typically looks like this)
#   python -m spacy download en_core_web_sm
#   python -m spacy download en_core_web_lg

# Don't worry about installing the coref right now
#   pip uninstall neuralcoref
#   pip install neuralcoref --no-binary neuralcoref

nlp = spacy.load('en_core_web_lg')

# Document
doc = nlp(u'''Dr. Jennifer Smith visited
    China. She liked the country a lot.''')

# Print Document
print(doc)
# Print sentences
print([sent.text for sent in doc.sents])
# Print tokens
print([[token.text for token in sent] for sent in doc.sents])
# Print a table of tokens and their lemmas
print(table([[token.text, token.lemma_] for token in doc]))

# Print sample embedding vectors
visited = doc[3]
china = doc[5]
country = doc[10]

print(visited.vector)

# Print embedding vector similarities
sim = china.similarity
print("similarity(china, visited): " + str(sim(visited)))
print("similarity(china, country): " + str(sim(country)))
print("similarity(china, India): " + str(sim(nlp("India")[0])))

# Print tokens and their part-of-speech (POS) tags
print(table([[token.text, token.pos_] for token in doc]))

# Print named entities and their types
print(table([[entity.text, entity.label_] for entity in doc.ents]))

# Display named entities (uses web server, 0.0.0.0:5000)
#spacy.displacy.serve(doc, style='ent')

# Print out a table of tokens and their grammatical heads
print(table([[token.text, token.head.text] for token in doc]))

# Display dependency tree (0.0.0.0:5000)
#spacy.displacy.serve(doc.sents, style='dep')
