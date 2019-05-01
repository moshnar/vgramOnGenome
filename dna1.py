from sklearn.pipeline import Pipeline
from vgram import VGramBuilder, CharTokenizer,StreamVGramBuilder
import re


def split_seq(seq):
    data = []
    i = 0
    l = 10000
    while i * l < len(seq):
        start = i * l
        finish = min((i + 1) * l, len(seq))
        if finish - start > 1:
            subseq = seq[start:finish]
            subseq = re.sub("N|\n", "", subseq)
            data.append(subseq)
        i += 1
    return data


def once_fit(filename, vgb, seq_len=10000):
    f = open(filename, 'r')
    data = []
    while True:
        seq = f.read(seq_len)
        if not seq:
            break
        seq = re.sub("N|\n", "", seq)
        data.append(seq)
    vgram = Pipeline([
        ("tok", CharTokenizer()),
        ("vgb", vgb)
    ])
    vgram.fit(data)
    return vgram


def stream_fit(filename, svgb, seq_len=10000):
    f = open(filename, 'r')
    tokenizer = CharTokenizer()
    tokenizer.fit(["actg"])
    while True:
        seq = f.read(seq_len)
        if not seq:
            break
        seq = re.sub("N|\n", "", seq)
        tokenizer.fit([seq])
        sseq = tokenizer.transform([seq])[0]
        svgb.accept(sseq)
    svgb.update()
    return svgb, tokenizer


def main():
    # filename = "/Users/akhvorov/data/mlimlab/vgram/dna/sequence (1).txt"  # small
    filename = "/Users/b.n.o/Downloads/GRCh38_latest_genomic.txt"  # large
    seq_len = 100000
    # for small files
    # vgram = once_fit(filename, VGramBuilder(1000, 1), seq_len=seq_len)
    # for large files^ stream fitting
    svgb, tokenizer = stream_fit(filename, StreamVGramBuilder(1000), seq_len=seq_len)

    # to save dictionary uncomment next line
    svgb.save("dict_100000_s100k.json", tokenizer)
    vgram.named_steps["vgb"].save("dict_10000_100.json", vgram.named_steps["tokenizer"])

    alpha = tokenizer.decode(svgb.alphabet())
    alpha = sorted(alpha, key=lambda x: -len(x))
    print(alpha[:100])
    out = open("top_vgrams.txt", "w")
    for word in alpha:
        out.write(word + "\n")


if __name__ == "__main__":
    main()
