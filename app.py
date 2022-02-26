from flask import Flask,render_template,request
import pickle
import numpy as np
#import os
#import cv2
#from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model
#from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model,load_model
#import random
from keras.preprocessing import image, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
#import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from nltk.translate.bleu_score import SmoothingFunction


caption_path = './Flickr8k.token.txt'
captions = open(caption_path, 'rb').read().decode('utf-8').split('\n')
vocab = np.load('v7_vocab.npy',allow_pickle=True)
vocab = vocab.item()
inv_vocab = {v:k for k,v in vocab.items()}

def getrealcaptions(x):
  realcaptions = []
  realcaptions1 = []
  for i in captions:
    try:
        img_name = i.split('\t')[0][:-2]
        caption = i.split('\t')[1]
        if(x == img_name):
          subcap = caption.lower().split() 
          realcaptions.append(subcap)
          realcaptions1.append(caption)
    except:
      pass
  return realcaptions,realcaptions1

def beam_search_predictions(image, beam_index = 3):
    start = [inv_vocab["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            preds = model.predict([image,par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [vocab[i] for i in start_word]
    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_len):
        sequence = [inv_vocab[w] for w in in_text.split() if w in inv_vocab]
        sequence = pad_sequences([sequence], maxlen=max_len)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = vocab[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

#resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
resnet = load_model('v7_inceptionmodel.h5')
print("inception loaded")

def encode(image):
    image = preprocess(image) 
    fea_vec = resnet.predict(image) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec




embedding_dim = 200
max_len = 38
#vocab_size = len(vocab)
vocab_size = len(vocab)+1

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_len,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)

# model.load_weights("../input/model_weights.h5")
model.compile(loss='categorical_crossentropy', optimizer='adam')
#model.summary()

model.load_weights("v7_model_weights.h5")

print("model loaded for merging image and captions to dense layer")

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    global model,vocab,inv_vocab

    file = request.files['file1']
    print(file)
    print(file.filename)
    file.save('static/file.jpg')

    image = encode('static/file.jpg').reshape((1,2048))
    
    reference,actual = getrealcaptions(file.filename)
    print("reference:",reference)
    greedysearch1 = greedySearch(image)
    beamsearch3 = beam_search_predictions(image, beam_index = 3)
    beamsearch5 = beam_search_predictions(image, beam_index = 5)
    beamsearch7 = beam_search_predictions(image, beam_index = 7)
    beamsearch10 = beam_search_predictions(image, beam_index = 10)
    scores = []
    chencherry = SmoothingFunction()
    #weights = [0.5,0.5,0,0]
    if reference:
        gscore = round(sentence_bleu(reference, greedysearch1.split(),smoothing_function=chencherry.method1),2)
        scores.append(gscore)
        b3score = round(sentence_bleu(reference, beamsearch3.split(),smoothing_function=chencherry.method1),2)
        scores.append(b3score)
        b5score = round(sentence_bleu(reference, beamsearch5.split(),smoothing_function=chencherry.method1),2)
        scores.append(b5score)
        b7score = round(sentence_bleu(reference, beamsearch7.split(),smoothing_function=chencherry.method1),2)
        scores.append(b7score)
        b10score = round(sentence_bleu(reference, beamsearch10.split(),smoothing_function=chencherry.method1),2)
        scores.append(b10score)

    return render_template('predict.html', scores=scores,greedysearch1 = greedysearch1,beamsearch3=beamsearch3,beamsearch5=beamsearch5,beamsearch7=beamsearch7,beamsearch10=beamsearch10,actual=actual)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)