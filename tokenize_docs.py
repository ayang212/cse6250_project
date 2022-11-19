import xml.etree.ElementTree as ET
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS


def tokenize_text(data):
    # data: expecting XML item with text attribute
    # token_text() returns doc_token_list, a list of cleaned sentences from the input text data


    # remove extra white spaces and replace with single space
    text = " ".join(data.text.split())

    nlp = English()
    nlp.add_pipe("sentencizer")


    # convert to spacy doc
    my_doc = nlp(text)

    doc_token_list = []

    # loop through each sentence and create tokens to append to sent_token_list
    for sent in my_doc.sents:
        # remove any non-alphabetic characters and extra spaces from the sentence
        cleaned_sent = " ".join([w for w in str(sent).split() if w.isalpha()])
        tokens = []
        for token in nlp(cleaned_sent):
            word = token.text.lower()
            # only tokenize non-stop words and if length of the word is greater than 1
            if (word not in STOP_WORDS) and (len(word) > 1):
                tokens.append(word)
        if tokens:
            doc_token_list.extend(tokens)
    return doc_token_list

def sentence_seg(data):
    # data: expecting XML item with text attribute
    # sentence_seg() returns sent_token_list, a list of lists where each nested list is a tokenized sentence from the input text data

    # remove extra white spaces and replace with single space
    text = " ".join(data.text.split())

    nlp = English()
    nlp.add_pipe("sentencizer")


    # convert to spacy doc
    my_doc = nlp(text)

    sent_token_list = []

    # loop through each sentence and create tokens to append to sent_token_list
    for sent in my_doc.sents:
        # remove any non-alphabetic characters and extra spaces from the sentence
        cleaned_sent = " ".join([w for w in str(sent).split() if w.isalpha()])

        # tokens: temporary list for removing any stop words
        tokens = []
        for token in nlp(cleaned_sent):
            word = token.text.lower()
            # only tokenize non-stop words and if length of the word is greater than 1
            if (word not in STOP_WORDS) and (len(word) > 1):
                tokens.append(word)
        
        # only append if list is not empty
        if tokens:
            sent_token_list.append(tokens)
    return sent_token_list

def load_text(path):
    root = ET.parse(path)
    
    # training text under the 'TEXT' tag
    data = root.find('TEXT')
    labels = root.find('TAGS')

    return data, labels

def generate_labels(labels):
    data_dict = {}
    for child in labels:
        criteria = child.tag.lower()
        if child.attrib['met'] == 'met':
            val = 1
        else:
            val = 0
        data_dict[criteria] = val
    return data_dict

def main():
    # take 100.xml as an initial example to parse
    path = r'CSE6250_projectdata\test\100.xml'

    data, labels = load_text(path)
    tokenize_text(data)


    ##### process response criteria/variables and store results to dictionary 'responses_dict'
    # response criteria under the 'TAGS' tag


if __name__ == "__main__":
    main()