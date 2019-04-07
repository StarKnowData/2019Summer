##!/user/bin/env python
## coding = utf-8

import os
import time
import sys
from gensim.models import word2vec

def main():
    #原始语料路径，已分词
    sentences = word2vec.Text8Corpus('E:\dataset\sogouSegDone.txt')
    #训练代码
    model = word2vec.Word2Vec(sentences, sg=1, size=100, window=5, min_count=1, negative=3, sample=0.001, hs=1, workers=40)
    #save
    model.save("E:\dataset\sogou_word.model")
    model.wv.save_word2vec_format("E:\dataset\sogou_word2vec.txt")

if __name__ == "__main__":
    main()
    print("Done!")