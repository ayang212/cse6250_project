from gensim.models import FastText
import os
from tokenize_docs import tokenize_text, load_text, sentence_seg, generate_labels
import xml.etree.ElementTree as ET
import json


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    path = r'.\CSE6250_projectdata\train'
    corpus = []
    labels_list = []
    ft_model = FastText()

    for i, filename in enumerate(os.listdir(path)):
        fullname = os.path.join(path, filename)
        data, labels_ET = load_text(fullname)
        labels_dict = generate_labels(labels_ET)
        
        labels_list.append(labels_dict)
        tokens = sentence_seg(data)
        corpus.append(tokens)

        # build vocabulary of fasttext word embeddings
        if i == 0:
            ft_model.build_vocab(tokens)
        else:
            ft_model.build_vocab(tokens, update=True)
        
        ft_model.train(tokens, total_examples=ft_model.corpus_count, epochs=10)

    file_output_list = []
    for i, doc in enumerate(corpus):
            
        temp_dict = {}
        temp_dict["id"] = i

        doc_flat = [item for sublist in doc for item in sublist]
        vec = ft_model.wv[set(doc_flat)]
        temp_dict["data"] = vec.tolist()
        temp_dict["labels"] = labels_list[i]
        file_output_list.append(temp_dict)
    output_data = {}
    output_data["training_data"] = file_output_list

    # generate vector for each doc with labels
    with open("training_data2.json", "w") as outfile:
        json.dump(output_data, outfile)
                
        
    ft_model.save(r'.trained_embeddings')

    # # use BioWordVec binary to create a gensim model for embeddings
    # filename = './CSE6250_project/CSE6250_projectdata/BioWordVec_PubMed_MIMICIII_d200.vec.bin'
    # model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=int(4E7))
    

if __name__ == "__main__":
    main()





