"""
Create Train and Test Files for the SemEval 2010 Task 8 experiment
"""
import nltk

trainFile = 'raw_data/TRAIN_FILE.txt'
testFile = 'raw_data/TEST_FILE.txt'
testAnswer = 'raw_data/answer_key.txt'

def createTrainFile(filepath, outputpath):
    fOut = open(outputpath, 'w')
    lines = [line.strip() for line in open(filepath)]
    for idx in range(0, len(lines), 4):
        sentence = lines[idx].split("\t")[1][1:-1]
        label = lines[idx+1]
        
        sentence = sentence.replace("<e1>", " _e1_ ").replace("</e1>", " _/e1_ ")
        sentence = sentence.replace("<e2>", " _e2_ ").replace("</e2>", " _/e2_ ")
        tokens = nltk.word_tokenize(sentence)
        #print tokens
        tokens.remove('_/e1_')    
        tokens.remove('_/e2_')
        
        e1 = tokens.index("_e1_")
        del tokens[e1]
        
        e2 = tokens.index("_e2_")
        del tokens[e2]
        
        #print tokens
        #print tokens[e1], "<->", tokens[e2]
    
        fOut.write("\t".join([label, str(e1), str(e2), " ".join(tokens)]))
        fOut.write("\n")
    fOut.close()

def createTestFile(filepath, outputpath, answerpath):
    fOut = open(outputpath, 'w')
    answers = [line.strip() for line in open(answerpath)] 
    lines = [line.strip() for line in open(filepath)]
    
    for idx in range(0, len(lines)):
        sentence = lines[idx].split("\t")[1][1:-1]
        label = answers[idx].split("\t")[1]
        # print(sentence)
        # print(label)
        
        sentence = sentence.replace("<e1>", " _e1_ ").replace("</e1>", " _/e1_ ")
        sentence = sentence.replace("<e2>", " _e2_ ").replace("</e2>", " _/e2_ ")
        tokens = nltk.word_tokenize(sentence)
        #print tokens
        tokens.remove('_/e1_')    
        tokens.remove('_/e2_')
        
        e1 = tokens.index("_e1_")
        del tokens[e1]
        
        e2 = tokens.index("_e2_")
        del tokens[e2]
        
        #print tokens
        #print tokens[e1], "<->", tokens[e2]
    
        fOut.write("\t".join([label, str(e1), str(e2), " ".join(tokens)]))
        fOut.write("\n")
    fOut.close()

createTrainFile(trainFile, "files/train.txt")
createTestFile(testFile, "files/test.txt", testAnswer)

print("done")