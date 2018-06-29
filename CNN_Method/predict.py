"""
- Python 3.6
- TensorFlow 1.8
- Keras 2.0.5
"""
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import pickle
import sys
import pickle as pkl

import keras
import gzip

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.regularizers import Regularizer
from keras.preprocessing import sequence
from keras import backend as K
import tensorflow as tf
sess=tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3)))
K.set_session(sess)
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default='model_best.h5', help="the model name")
parser.add_argument("--predict_file_name", type=str, default='Result.txt', help="predicted file name")
args = parser.parse_args()
model_name = args.model_name
file_name = args.predict_file_name

batch_size = 32
nb_filter = 256
filter_length = 3
hidden_dims = 512
nb_epoch = 60
position_dims = 50

print("Load dataset")
f = gzip.open('pkl/sem-relations.pkl.gz', 'rb')
data = pkl.load(f)
f.close()

embeddings = data['wordEmbeddings']
yTrain, sentenceTrain, positionTrain1, positionTrain2 = data['train_set']
yTest, sentenceTest, positionTest1, positionTest2  = data['test_set']

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain)+1
#train_y_cat = np_utils.to_categorical(yTrain, n_out)
max_sentence_len = sentenceTrain.shape[1]

print("sentenceTrain: ", sentenceTrain.shape)
print("positionTrain1: ", positionTrain1.shape)
print("yTrain: ", yTrain.shape)




print("sentenceTest: ", sentenceTest.shape)
print("positionTest1: ", positionTest1.shape)
print("yTest: ", yTest.shape)



print("Embeddings: ",embeddings.shape)

words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)(words_input)

distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
distance1 = Embedding(max_position, position_dims)(distance1_input)

distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
distance2 = Embedding(max_position, position_dims)(distance2_input)




output = concatenate([words, distance1, distance2])


output = Convolution1D(filters=nb_filter,
                        kernel_size=filter_length,
                        padding='same',
                        activation='tanh',
                        strides=1)(output)

# we use standard max over time pooling
output = GlobalMaxPooling1D()(output)

output = Dropout(0.25)(output)
output = Dense(n_out, activation='softmax')(output)

model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
model.load_weights(model_name)
model.summary()

print("Start Prediction...")

max_prec, max_rec, max_acc, max_f1 = 0,0,0,0

def getPrecision(pred_test, yTest, targetLabel):
    #Precision for non-vague
    targetLabelCount = 0
    correctTargetLabelCount = 0
    
    for idx in range(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1
            
            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1
    
    if correctTargetLabelCount == 0:
        return 0
    
    return float(correctTargetLabelCount) / targetLabelCount

def predict_classes(prediction):

 return prediction.argmax(axis=-1)

# =============write predicted result================
pred_test = predict_classes(model.predict([sentenceTest, positionTest1, positionTest2], verbose=False))
dctLabels = np.sum(pred_test)
totalDCTLabels = np.sum(yTest)
   
acc =  np.sum(pred_test == yTest) / float(len(yTest))
max_acc = max(max_acc, acc)
print("Accuracy: %.4f (max: %.4f)" % (acc, max_acc))

f1Sum = 0
f1Count = 0
for targetLabel in range(1, max(yTest)):        
    prec = getPrecision(pred_test, yTest, targetLabel)
    recall = getPrecision(yTest, pred_test, targetLabel)
    f1 = 0 if (prec+recall) == 0 else 2*prec*recall/(prec+recall)
    f1Sum += f1
    f1Count +=1    
   
macroF1 = f1Sum / float(f1Count)    
max_f1 = max(max_f1, macroF1)
print("Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1))


# save the predicted values
f = open("./class_names.pkl", "rb")
labelsMapping = pickle.load(f)
f.close()
labelsMapping_inv = {}
for i in labelsMapping.items():
    labelsMapping_inv[i[1]]=i[0]
    
f = open(file_name,'w')
for idx, i in enumerate(pred_test):
    f.write(str(idx+8001)+'\t'+labelsMapping_inv[i]+'\n')
f.close()
# ============write predicted result=================
