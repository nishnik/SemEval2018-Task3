import json
with open('dependency_sentences.txt') as fh:
    data = json.load(fh)

f = open('corpus_split.txt', 'w')

for a in data:
    for b in a:
        if len(a[b]) > 1 :
            f.write(a[b] + "\n")




