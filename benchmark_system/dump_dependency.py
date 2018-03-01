import json
f =  open('dependency_test_sentences.txt', 'r')
data = json.load(f)
f.close()
f = open('dependency_test_sentences.txt.dump', 'w')
for row in data:
    for key in row:
        f.write(row[key]+"\n")

f.close()