from sklearn.pipeline import Pipeline
from vgram import VGramBuilder, CharTokenizer
import re


def split_seq(seq):
    data = []
    i = 0
    l = 1000000
    while i * l < len(seq):
        start = i * l
        finish = min((i + 1) * l, len(seq))
        if finish - start > 1:
            data.append(seq[start:finish])
        i += 1
    return data


def main():
    f = open("/Users/b.n.o/Desktop/sequence (1).txt")
    seq = f.read()
    seq = re.sub("N|\n", "", seq)
    data = split_seq(seq)

    vgram = Pipeline([
            ("tokenizer", CharTokenizer()),
            # to learn new dict use this line
            ("vgb", VGramBuilder(10000, 10))
            # use this line for existing dict
            #("vgb", VGramBuilder("dict_10000_10.json"))
    ])
    vgram.fit(data)
    # to save dictionary uncomment next line
    vgram.named_steps["vgb"].save("dict_10000_10.json", vgram.named_steps["tokenizer"])

    alpha = vgram.named_steps["tokenizer"].decode(vgram.named_steps["vgb"].alphabet())
    alpha = sorted(alpha, key=lambda x: -len(x))
    out = open("top_vgrams.txt", "w")
    for word in alpha:
        out.write(word + "\n")


if __name__ == "__main__":
    main()
