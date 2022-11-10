import xml.etree.ElementTree as ET
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS


def tokenize_text(data):
    # data: expecting XML item with text attribute
    # token_text() returns sent_token_list, a list of lists of tokens from the input text data


    # remove extra white spaces and replace with single space
    text = " ".join(data.text.split())

    nlp = English()
    nlp.add_pipe("sentencizer")


    # convert to spacy doc
    my_doc = nlp(text)

    sent_token_list = []

    # loop through each sentence and create tokens to append to sent_token_list
    for sent in my_doc.sents:
        cleaned_sent = " ".join([w for w in str(sent).split() if w.isalpha()])
        tokens = []
        for token in nlp(cleaned_sent):
            word = token.text.lower()
            # only tokenize non-stop words and if length of the word is greater than 1
            if (word not in STOP_WORDS) and (len(word) > 1):
                tokens.append(word)
        sent_token_list.append(tokens)
    return sent_token_list



def main():
    # take 100.xml as an initial example to parse
    path = r'CSE6250_projectdata\train\100.xml'
    root = ET.parse(path)
    
    # training text under the 'TEXT' tag
    data = root.find('TEXT')
    tokenize_text(data)


    ##### process response criteria/variables and store results to dictionary 'responses_dict'
    # response criteria under the 'TAGS' tag
    responses_parent = root.find('TAGS')
    data_dict = {}
    for child in responses_parent:
        criteria = child.tag.lower()
        if child.attrib['met'] == 'met':
            val = 1
        else:
            val = 0
        data_dict[criteria] = val

main()