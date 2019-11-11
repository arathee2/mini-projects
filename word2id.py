def word2id(string, split_on = " ", reverse=False):
    '''create a word to id mapping of the given list of words'''
    
    # split text
    words = string.split(split_on)
    
    # create an empty list to store word as keys and ids as values
    word2id = {}
    
    # id iterator
    i = 0
    
    # for each new word, assign an id 'i'
    for word in words:
        word = word.lower()
        if words not in word2id.keys():
            i = i + 1
            word2id[word] = i
    
    # sort word2id dictionary by key
    word2id = sorted(word2id.items(), key=lambda item: item[1], reverse=False)
    
    # reverse mapping: id2word
    if reverse:
        id2word = {value: key for key, value in word2id.items()}
        sorted(id2word.items(), key=lambda item: item[1], reverse=False)
        return id2word
    
    return word2id


sentence = "Rules and models destroy genius and art."

# word to id mapping
word2id = word2id(sentence)
print(word2id)

# id to word mapping
id2word = word2id(sentence, reverse=True)
print(id2word)
