import torch
import random
import math


symbols = ["!", "(", ")", "-", "[", "]", "{", "}", ";", ":", "\'", "\"", "—", ",", "<", ">", ".", "/", "?", "@", "#", "$", "%", "^", "&", "*", "_", "~"]

def main():
    input_text = '''There was nothing so very remarkable in that; nor did Alice think it
            so very much out of the way to hear the Rabbit say to itself, “Oh
            dear! Oh dear! I shall be late!” (when she thought it over afterwards,
            it occurred to her that she ought to have wondered at this, but at the
            time it all seemed quite natural but when the Rabbit actually took a
            watch out of its waistcoat-pocket, and looked at it, and then hurried
            on, Alice started to her feet, for it flashed across her mind that she
            had never before seen a rabbit with either a waistcoat-pocket, or a
            watch to take out of it, and burning with curiosity, she ran across the
            field after it, and fortunately was just in time to see it pop down a
            large rabbit-hole under the hedge.'''.split()
    
    output_text = ''
    vocab = list(input_text)

    for i, w in enumerate(vocab):
        word = ''
        punucation = ''
        for l in w:
            if l in symbols and vocab[i-1] != '_':
                word += '_' + l
            else:
                word += l
        vocab[i] = word
    
    #print(vocab)
    vocab_set = sorted(set(vocab))

    wtoix = {word: i  for i, word in enumerate(vocab_set)}
    ixtow = {i: word  for i, word in enumerate(vocab_set)}

    k = 0.00001
    torch.manual_seed(1)

    probablity_tensor = torch.zeros(len(wtoix), len(wtoix)) 

    input_word = input("Word: ")
    input_word = input("Temp: ")
    for i in range(len(vocab)-1):
       probablity_tensor[wtoix[vocab[i]], wtoix[vocab[i + 1]]] += 1

    k_probablity_tensor = probablity_tensor.clone()
    k_probablity_tensor += k
    rows = k_probablity_tensor.shape[0]

    for i in range(rows):
        k_probablity_tensor[i:].div_(k_probablity_tensor[i:].sum())

    if wtoix[input_word] != '':
            output_text += input_word
            for i in range(100):
                i = wtoix[input_word]
                row = k_probablity_tensor[i, :]
                for i in range(len(row.tolist())-1):
                    weight = row.tolist()[i]**t
                word = random.choices(range(rows), weights=weight)[0]
                output_text +=  ('_' + ixtow[word])
                input_word = ixtow[word]

    finished_text = ''
    for i, l in enumerate(output_text):
        #print(l)
        if l == "_":
            if i < len(output_text):
                if output_text[i+1] in symbols:
                    l = ''
                else:
                    l = ' '
        finished_text += l
    print(finished_text)
        

    #print(wtoix)
    print(probablity_tensor.sum())
    #print(rows)
    print(k_probablity_tensor.sum())


    

if __name__ == '__main__':
    main()