import torch
import random
import math


symbols = ["!", "(", ")", "-", "[", "]", "{", "}", ";", ":", "\'", "\"", "—", ",", "<", ">", ".", "/", "?", "@", "#", "$", "%", "^", "&", "*", "_", "~"]
special = ['<pad>', '<unk>']

def split_percent_list(input_list: list, percent: float):  
    if percent <= 0 or percent >= 1:
        print('Main.py [8] variable percent was out of range: ', percent)
        return [], []

    split_idx = math.ceil(len(input_list) * percent)

    first_output_list = input_list[:split_idx]
    second_output_list = input_list[split_idx:]
    
    return first_output_list, second_output_list


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
    input_text,  test_text = split_percent_list(input_text, 0.8)
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
    for id, str in enumerate(special):
        vocab_set.insert(id, str)  
    #print(vocab_set)

    wtoix = {word: i  for i, word in enumerate(vocab_set)}
    ixtow = {i: word  for i, word in enumerate(vocab_set)}
    #print(wtoix)

    
    torch.manual_seed(1)

    probablity_tensor = torch.zeros(len(wtoix), len(wtoix)) 

    input_word = input("Word: ")
    temp = float(input("Temp: "))
    loops = int(input("loop: "))
    k = float(input("K-smoothing: "))
    for i in range(len(vocab)-1):
       probablity_tensor[wtoix[vocab[i]], wtoix[vocab[i + 1]]] += 1

    k_probablity_tensor = probablity_tensor.clone()
    k_probablity_tensor += k
    rows = k_probablity_tensor.shape[0]

    for i in range(rows):
        k_probablity_tensor[i:].div_(k_probablity_tensor[i:].sum())

    vocab_ix = []
    vocab_sentences = []
    for w in test_text:
        if w in vocab_set:
            ix = int(wtoix[w])
        else:
            ix = int(wtoix['<unk>'])
        vocab_ix.append(ix)
        for i, l in enumerate(w):
            if l in symbols:
                if vocab_ix != []:
                    vocab_sentences += [vocab_ix]
                vocab_ix = []
    print(vocab_sentences)

    Test_tensor = torch.zeros(len(vocab_sentences), len(max(vocab_sentences, key=len)), dtype=torch.int16)
    for y, l in enumerate(vocab_sentences):
        for x, n in enumerate(l):
            Test_tensor[y, x] = n

    print(Test_tensor)

    if wtoix[input_word] != '':
            output_text += input_word
            for i in range(loops):
                i = wtoix[input_word]
                row = k_probablity_tensor[i, :]
                weight = row.tolist()
                for i in range(len(weight)-1):
                    weight[i] = weight[i]**temp
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
    #print(probablity_tensor.sum())
    #print(rows)
    #print(k_probablity_tensor)


    

if __name__ == '__main__':
    main()