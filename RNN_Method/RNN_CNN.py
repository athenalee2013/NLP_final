"""
This is a CNN for relation classification within a sentence. The architecture is based on:

Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao, 2014, Relation Classification via Convolutional Deep Neural Network

Performance (without hyperparameter optimization):
Accuracy: 0.7943
Macro-Averaged F1 (without Other relation):  0.7612

Performance Zeng et al.
Macro-Averaged F1 (without Other relation): 0.789


Code was tested with:
- Theano 0.8.2
- Keras 1.1.1
- Python 2.7
"""
import numpy as np
np.random.seed(1337)  # for reproducibility

#import cPickle as pkl
import pickle as pkl
import gzip
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.core import Merge
#from keras.layers import concatenate
from keras.layers.merge import concatenate
from keras.layers import Concatenate
#from keras.layers import add
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv1D, Input, Add, add, Flatten
from keras.models import Model, load_model
#from keras.layers import concatenate
from keras.utils import np_utils
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
#from keras.engine import merge
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

batch_size = 64
nb_filter = 200
filter_length = 3
hidden_dims = 256
nb_epoch = 100
position_dims = 50
GRU_hidden = 256
def find_key(input_dict, value):
    return next((k for k, v in input_dict.items() if v == value), None)

labelsMapping = {'Other':0, 
                 'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2, 
                 'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4, 
                 'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6, 
                 'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
                 'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
                 'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12,  
                 'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
                 'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
                 'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}


print ("Load dataset")
f = gzip.open('pkl/sem-relations.pkl.gz', 'rb')
yTrain, sentenceTrain, positionTrain1, positionTrain2 = pkl.load(f)
yTest, sentenceTest, positionTest1, positionTest2  = pkl.load(f)
f.close()

yvalid = yTrain[0:int(yTrain.shape[0]*0.15)]
sentenceValid = sentenceTrain[0:int(sentenceTrain.shape[0]*0.15)]
positionValid1 = positionTrain1[0:int(positionTrain1.shape[0]*0.15)]
positionValid2 = positionTrain2[0:int(positionTrain2.shape[0]*0.15)]

yTrain = yTrain[int(yTrain.shape[0]*0.15):yTrain.shape[0]]
sentenceTrain = sentenceTrain[int(sentenceTrain.shape[0]*0.15):sentenceTrain.shape[0]]
positionTrain1 = positionTrain1[int(positionTrain1.shape[0]*0.15):positionTrain1.shape[0]]
positionTrain2 = positionTrain2[int(positionTrain2.shape[0]*0.15):positionTrain2.shape[0]]


max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain)+1
print('n_out')
print(n_out)
#exit()
train_y_cat = np_utils.to_categorical(yTrain, n_out)
n_valid_out = max(yvalid)+1
valid_y_cat = np_utils.to_categorical(yvalid, n_valid_out)

print ("sentenceTrain: ", sentenceTrain.shape)
print ("positionTrain1: ", positionTrain1.shape)
print ("yTrain: ", yTrain.shape)




print ("sentenceTest: ", sentenceTest.shape)
print ("positionTest1: ", positionTest1.shape)
print ("yTest: ", yTest.shape)


f = gzip.open('pkl/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

print ("Embeddings: ",embeddings.shape)
print("position_dims ",position_dims)
#exit()
save_path = './CNN_Add.h5'
"""
distanceModel1 = Sequential()
distanceModel1.add(Embedding(max_position, position_dims, input_length=positionTrain1.shape[1]))

distanceModel2 = Sequential()
distanceModel2.add(Embedding(max_position, position_dims, input_length=positionTrain2.shape[1]))

wordModel = Sequential()
wordModel.add(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sentenceTrain.shape[1], weights=[embeddings], trainable=False))
"""
###########

distanceModel1_in = Input(shape=(positionTrain1.shape[1],))
distanceModel1_out = Embedding(max_position, position_dims, input_length=positionTrain1.shape[1])(distanceModel1_in)
#distanceModel1 = Model(distanceModel1_in, distanceModel1_out)

distanceModel2_in = Input(shape=(positionTrain2.shape[1],))
distanceModel2_out = Embedding(max_position, position_dims, input_length=positionTrain2.shape[1])(distanceModel2_in)
#distanceModel2 = Model(distanceModel2_in, distanceModel2_out)

wordModel_in = Input(shape=(sentenceTrain.shape[1],))
wordModel_out = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sentenceTrain.shape[1], weights=[embeddings], trainable=False)(wordModel_in)
#wordModel = Model(wordModel_in, wordModel_out)

x = concatenate([wordModel_out, distanceModel1_out, distanceModel2_out])#wordModel_out#add
x = Bidirectional(GRU(GRU_hidden, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
x = Conv1D(filters =nb_filter, kernel_size=filter_length, padding='same', activation='tanh', strides=1)(x)
#avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
#x = concatenate([avg_pool, max_pool])
#x = Dropout(0.25)(max_pool)
preds = Dense(n_out, activation='softmax')(max_pool)
model = Model(inputs=[wordModel_in, distanceModel1_in, distanceModel2_in], outputs = preds)


model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
print(model.summary())
print ("Start training")
#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1)
earlyStopping=EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')

checkpoint = ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=True, save_weights_only=False, monitor='val_acc', mode='max' )
model.fit([sentenceTrain, positionTrain1, positionTrain2],train_y_cat , 
                validation_data=([sentenceValid, positionValid1, positionValid2], valid_y_cat), 
                batch_size=batch_size , epochs=50, callbacks=[checkpoint, earlyStopping])

semantic_classifier = load_model(save_path)
semantic_predict = semantic_classifier.predict([sentenceTest, positionTest1, positionTest2])
semantic_predictions = semantic_predict.argmax(axis=-1)
#print('semantic_predictions')
#for i in range(20):
    #print(semantic_predictions[i])
print('semantic_predictions.shape')
print(semantic_predictions.shape)
predict_relation = []
for i in range(len(semantic_predictions.tolist())):
    temp = find_key(labelsMapping, semantic_predictions[i])
    #temp = labelsMapping.get(semantic_predictions[i])
    predict_relation.append(temp)
#print('predict_relation')
#for i in range(20):
    #print(predict_relation[i])
output_path = './proposed_answer.txt'
with open(output_path, 'w') as f:
        #f.write('id,label\n')
        for i, v in  enumerate(predict_relation):
            #print('i', i)
            #print('v', v)
            #print()
            f.write('%d\t%s\n' %(i+8001, v))

#
exit()

max_prec, max_rec, max_acc, max_f1 = 0,0,0,0

def getPrecision(pred_test, yTest, targetLabel):
    #Precision for non-vague
    targetLabelCount = 0
    correctTargetLabelCount = 0
    
    for idx in range(len(pred_test)):#xrange
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1
            
            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1
    
    if correctTargetLabelCount == 0:
        return 0
    
    return float(correctTargetLabelCount) / targetLabelCount

for epoch in range(nb_epoch):#xrange
    model.fit([sentenceTrain, positionTrain1, positionTrain2], train_y_cat, batch_size=batch_size, verbose=True,epochs=1)   #nb_epoch
    pred_test = model.predict_classes([sentenceTest, positionTest1, positionTest2], verbose=False)
    
    dctLabels = np.sum(pred_test)
    totalDCTLabels = np.sum(yTest)
   
    acc =  np.sum(pred_test == yTest) / float(len(yTest))
    max_acc = max(max_acc, acc)
    #print "Accuracy: %.4f (max: %.4f)" % (acc, max_acc)
    print("Accuracy: %.4f (max: %.4f)" % (acc, max_acc))

    f1Sum = 0
    f1Count = 0
    for targetLabel in range(1, max(yTest)):#xrange        
        prec = getPrecision(pred_test, yTest, targetLabel)
        rec = getPrecision(yTest, pred_test, targetLabel)
        f1 = 0 if (prec+rec) == 0 else 2*prec*rec/(prec+rec)
        f1Sum += f1
        f1Count +=1    
        
        
    macroF1 = f1Sum / float(f1Count)    
    max_f1 = max(max_f1, macroF1)
    #print "Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1)
    print ("Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1))