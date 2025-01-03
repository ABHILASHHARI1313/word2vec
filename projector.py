from word2vec_scratch import W2W
import csv

with open("data.txt", "r", errors="ignore", encoding="utf-8") as f:
    paragraph = f.read()
    f.close()
w2v = W2W(paragraph)
weight1 = w2v.load_model()
metadata = w2v.words


with open("embedding.tsv", "w", encoding="utf-8") as e:
    csv_writer = csv.writer(e, delimiter="\t")
    csv_writer.writerows(weight1)

with open("metadata.tsv", "w", encoding="utf-8") as m:
    csv_writer = csv.writer(m, delimiter="\n")
    csv_writer.writerow(metadata)

word = "india"
x = w2v.similiar_words(word)
print(x)
