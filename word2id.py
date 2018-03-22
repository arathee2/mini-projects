sentence = "Rules and models destroy genius and art."
sentence = sentence.split(" ")
print(sentence)

def word2id(words):
    '''create a word to id mapping of the given string'''
    
    # create an empty list to store word as keys and ids as values
    word2id = {}
    
    # id iterator
    i = 0
    
    # for each new word, assign an id 'i'
    for word in words:
        word = word.lower()
        if word not in word2id.keys():
            i = i + 1
            word2id[word] = i
    return word2id

word2id = word2id(string)
word2id = sorted(word2id.items(), key=lambda item: item[1], reverse=False)
print(word2id)

# reverse mapping: store id as key and word as value
id2word = {value: key for key, value in word2id.items()}
print(id2word)
