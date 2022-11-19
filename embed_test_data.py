from gensim.models import FastText
import os
from tokenize_docs import tokenize_text, load_text, sentence_seg, generate_labels
import xml.etree.ElementTree as ET
import json


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    path = r'.\CSE6250_projectdata\test'
    corpus = []
    labels_list = []

    ft_model = FastText.load('trained_embeddings')

    for i, filename in enumerate(os.listdir(path)):
        fullname = os.path.join(path, filename)
        data, labels_ET = load_text(fullname)
        labels_dict = generate_labels(labels_ET)
        
        labels_list.append(labels_dict)
        tokens = sentence_seg(data)
        corpus.append(tokens)

    file_output_list = []
    for i, doc in enumerate(corpus):
            
        temp_dict = {}
        temp_dict["id"] = i

        doc_flat = [item for sublist in doc for item in sublist]
        vec = ft_model.wv[doc_flat]
        temp_dict["data"] = vec.tolist()
        temp_dict["labels"] = labels_list[i]
        file_output_list.append(temp_dict)

    output_data = {}
    output_data["testing_data"] = file_output_list

    # generate vector for each doc with labels
    with open("testing_data.json", "w") as outfile:
        json.dump(output_data, outfile)
    

if __name__ == "__main__":
    main()
