import re
from nltk.tokenize import word_tokenize

train_path = "./data/TEST_FILE.txt"#"./data/TRAIN_FILE.txt"
text = []
target_index = []
relation = []
comment = []
with open(train_path, 'r') as f:
    for i, line in enumerate(f):
            print(line)

            temp = line.split("\n")
            temp = temp[0].split("\t")
            index = temp[0]
            #print('index')
            #print(index)
            temp = re.sub('\"', '', temp[1])
            #print('temp')
            #print(temp)
            #exit()
           
            print()
            s_index = re.search("<e1>", temp).start()
            e_index = re.search("</e1>", temp).start()
            e1_word = temp[s_index + 4:e_index]
            if(e1_word.find(' ') != -1):#e.g.  <e1>burger king<\e1>
                e1_word = e1_word.split(' ',1)
                e1_word = e1_word[0]
            if(temp[s_index-1] != ' '):#e.g.   doves<e2>moles</e2>
                temp = temp[:s_index] + ' ' + temp[s_index:] 
            s_index = re.search("<e2>", temp).start()
            e_index = re.search("</e2>", temp).start()
            e2_word = temp[s_index + 4:e_index]
            if(e2_word.find(' ') != -1):
                e2_word = e2_word.split(' ',1)
                e2_word = e2_word[0]
            #if(index == '213'):
                #print('temp')
                #print(temp)
            if(temp[s_index-1] != ' '):#e.g.   doves<e2>moles</e2>
                temp = temp[:s_index] + ' ' + temp[s_index:]
            #if(index == '213'):
                #print('temp')
                #print(temp)
                #exit()       
            #exit()      
            string = re.sub('<e1>','', temp)
            string = re.sub('</e1>','', string)
            string = re.sub('<e2>','', string)
            string = re.sub('</e2>','', string)
            tokens = word_tokenize(string)
            #print('string')
            #print(string)
            #print('tokens')
            #print(tokens)
            #print('e1_word')
            #print(e1_word)
            #print('e2_word')
            #print(e2_word)
            e1_index = tokens.index(e1_word)
            e2_index = tokens.index(e2_word)
            #print(e1_index,"\t", e2_index)
            text.append(string)
            temp_target_index = []
            temp_target_index.append(e1_index)
            temp_target_index.append(e2_index)
            target_index.append(temp_target_index)

relation = []
ans_path = './data/answer_key.txt'
with open(ans_path, 'r') as f:
    for i, line in enumerate(f):
        temp = line.split("\n")
        temp = temp[0].split("\t")
        index = temp[0]
        #temp = re.sub('\"', '', temp[1])
        temp = temp[1]
        relation.append(temp)

print('text[10]')
print(text[10])
#print('relation[10]')
#print(relation[10])
#print('len(relation)')
#print(len(relation))
print('output:')
for i in range(20):
    print(relation[i],"\t",target_index[i][0], "\t",target_index[i][1],"\t" ,text[i])
    #print(target_index[i][0], "\t",target_index[i][1],"\t" ,text[i])
output_path = './test.txt'
with open(output_path, 'w') as f:
    #f.write('id,label\n')
    #for i, v in  enumerate(predictions):
        #f.write('%d,%d\n' %(i, v))
    for i in range(len(relation)):
       # f.write(relation[i],"\t",target_index[i][0], "\t",target_index[i][1],"\t" ,text[i])
        f.write('%s\t%s\t%s\t%s\n' %(relation[i], target_index[i][0], target_index[i][1], text[i]))