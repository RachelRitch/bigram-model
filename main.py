import torch
import random
import math
from pathlib import Path

symbols = ["!", "(", ")", "-", "[", "]", "{", "}", ";", ":", "\'", "\"", "—", ",", "<", ">", ".", "/", "?", "@", "#", "$", "%", "^", "&", "*", "_", "~"]
special = ['<pad>', '<unk>']

def openlibrary():
    text = ''
    for path in Path('Books').glob('*.txt'):
        with open(f'Books/{path.name}' , encoding='utf-8-sig') as f:
            text += f.read()
    return text

def split_percent_list(input_list: list, percent: float):  
    if percent <= 0 or percent >= 1:
        print('Main.py [8] variable percent was out of range: ', percent)
        return [], []

    split_idx = math.ceil(len(input_list) * percent)

    first_output_list = input_list[:split_idx]
    second_output_list = input_list[split_idx:]
    
    return first_output_list, second_output_list

def space_placeholder_inserter(list_str: list):
    for i, w in enumerate(list_str):
        word = ''
        punucation = ''
        for l in w:
            if l in symbols and list_str[i-1] != '_':
                word += '_' + l
            else:
                word += l
        list_str[i] = word
    return list_str

def main():
    input_text = openlibrary()
    input_text = input_text.split()
    #input_text = input_text[:1000000]
    input_text,  test_text = split_percent_list(input_text, 0.8)
    output_text = ''
    vocab = list(input_text)

    vocab = space_placeholder_inserter(vocab)
    test_text = space_placeholder_inserter(test_text)

    vocab_set = sorted(set(vocab))
    for id, str in enumerate(special):
        vocab_set.insert(id, str)  


    wtoix = {word: i  for i, word in enumerate(vocab_set)}
    ixtow = {i: word  for i, word in enumerate(vocab_set)}


    
    torch.manual_seed(1)

    probablity_tensor = torch.zeros(len(wtoix), len(wtoix), dtype=torch.float) 
    input_word = input("Word: ")
    temp = float(input("Temp: "))
    loops = int(input("loop: "))
    k = float(input("K-smoothing: "))
    for i in range(len(vocab)-1):
       probablity_tensor[wtoix[vocab[i]], wtoix[vocab[i + 1]]] += 1

    k_probablity_tensor = probablity_tensor
    k_probablity_tensor += k
    rows = k_probablity_tensor.shape[0]

    
    k_probablity_tensor = k_probablity_tensor.div(k_probablity_tensor.sum(dim=1, keepdim=True))

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

    max_test_length = int(len(max(vocab_sentences, key=len)))
    Test_tensor = torch.zeros(len(vocab_sentences), max_test_length, dtype=torch.long)
    for y, l in enumerate(vocab_sentences):
        for x, n in enumerate(l):
            Test_tensor[y, x] = n

    if wtoix[input_word] != '':
            output_text += input_word
            for i in range(loops):
                i = wtoix[input_word]
                row = k_probablity_tensor[i, :]
                weight = row.tolist()
                for i in range(len(weight)-1):
                    weight[i] = weight[i]**(1/temp)
                word = random.choices(range(rows), weights=weight)[0]
                output_text +=  ('_' + ixtow[word])
                input_word = ixtow[word]
    
    test_text_probablity = torch.zeros(len(vocab_sentences), max_test_length, dtype=torch.float)
    number_of_pairs = 0
    for iny, l in enumerate(vocab_sentences):
        for inx, n in enumerate(l):
                if inx > 0:
                    if Test_tensor[iny, inx].item() != 0 and Test_tensor[iny, inx - 1].item() != 0:
                        test_text_probablity[iny, inx] = -math.log(k_probablity_tensor[Test_tensor[iny, inx - 1], Test_tensor[iny, inx]].item())
                        number_of_pairs += 1
    
    avg = test_text_probablity.sum() / number_of_pairs
    perplexity = math.exp(avg)
    geometric_mean = math.exp(-avg)
    print('Perplexity: ', float(perplexity), ", Geometric Mean: ", geometric_mean)
    

    finished_text = ''
    for i, l in enumerate(output_text):
        if l == "_":
            if i < len(output_text):
                if output_text[i+1] in symbols:
                    l = ''
                else:
                    l = ' '
        finished_text += l
    print(finished_text)


    

if __name__ == '__main__':
    main()