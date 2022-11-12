from gensim.models import FastText
import os
from load_files import tokenize_text, load_text, sentence_seg
import xml.etree.ElementTree as ET


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # path = r'.\CSE6250_project\CSE6250_projectdata\train\100.xml'
    path = r'.\CSE6250_projectdata\train'
    corpus = []
    for filename in os.listdir(path):
        fullname = os.path.join(path, filename)
        data = load_text(fullname)
        tokens = sentence_seg(data)
        corpus.append(tokens)
    
    ft_model = FastText()
    ft_model.build_vocab(corpus)
    ft_model.train(corpus, total_examples=ft_model.corpus_count, epochs=10)
    ft_model.save(r'.trained_embeddings')

    # # use BioWordVec binary to create a gensim model for embeddings
    # filename = './CSE6250_project/CSE6250_projectdata/BioWordVec_PubMed_MIMICIII_d200.vec.bin'
    # model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=int(4E7))
    

if __name__ == "__main__":
    main()





