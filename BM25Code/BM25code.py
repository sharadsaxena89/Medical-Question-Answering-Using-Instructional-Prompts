# Testing pyserini, question from 1.tsv, and 1 should be in top results
from pyserini.search import SimpleSearcher

searcher = SimpleSearcher('indexes/sample_collection_jsonl')
hits = searcher.search('I have constant lower abdominal pain and deep dyspareunia but not post-coital spotting, what is my potential diagnosis?')

for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:15} {hits[i].score:.5f}')
    
hits = searcher.search('I have gradual weight loss and fatigue but not anorexia, what is my potential diagnosis?')

for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:15} {hits[i].score:.5f}')
