# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:51:50 2024

@author: sadwika sabbella
"""

import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
TF_ENABLE_ONEDNN_OPTS=0
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


BASE_DIR = "E:\image_desc_gen_proj"
WORKING_DIR = "E:\image_desc_gen_proj\models"


batch_size = 64
size = (256, 256)
num_channels = 3


model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output) # restructure the model
print(model.summary()) # summarize



features = {}
directory = os.path.join(r"E:\image_desc_gen_proj\Flicker8k_Dataset")
#print(os.listdir(directory)[:])
for img_name in tqdm(os.listdir(directory)):
    #print(img_name)
    img_path = directory + '/' + img_name # load the image from fil
    #print(img_path)
    image = load_img(img_path, target_size=(224,224))
    image = img_to_array(image) # convert image pixels to numpy array
    #print(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) #resize the image
    image = preprocess_input(image) # preprocess image for vgg
    feature = model.predict(image, verbose=0) # extract features
    image_id = img_name.split('.')[0] # get image ID
    features[image_id] = feature # store feature
    
print(features)


# store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR,'features1.pkl'), 'wb'))

    
with open(os.path.join(WORKING_DIR, 'features1.pkl'), 'rb') as f:
    features = pickle.load(f)
    
    
with open(os.path.join(BASE_DIR,r"E:\image_desc_gen_proj\Flicker8k_txt\Flickr8k.token.txt"), 'r') as f:
    #next(f)
    captions_doc = f.read()
    

print(captions_doc)


# create mapping of image to captions
mapping = {}
for line in tqdm(captions_doc.split('\n')):
    tokens = line.split(',')
    #print(tokens)
    if(len(line)<2):
        continue
    image_id=tokens[0]
    image_id = image_id.split('.')[0]
    tokens=str(tokens)
    parts = tokens.split(maxsplit=1)
    try:
      caption = parts[1].strip()  # Access description if it exists
    except IndexError:
      caption = ""
    caption = "".join(caption) 
    #print(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption[:-2])


print(mapping)
len(mapping)
print(type(mapping))

#sample before cleaning
mapping['979383193_0a542a059d']
    
    
#preprocess clean text
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            print(caption)
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption
    
#sample caption output
clean(mapping)
mapping['979383193_0a542a059d']


#text preprocessing
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)
        
        
len(all_captions)
all_captions[:10]



# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

print(vocab_size)



# get maximum length of the caption available

max_length = max(len(caption.split()) for caption in all_captions)
print(max_length)


#spliting training and testing dataset images
image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]


print(train)
print(test)
    


photos=mapping.keys()
descriptions=mapping.values()


print(type(mapping))

from numpy import array
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


def data_generator(mapping, photos, tokenizer, max_length,vocab_size):
    while True:
        for key, desc_list in mapping.items():
            
            # retrieve the photo feature
            if key in photos:
                photo = photos[key][0]
                in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
                yield [in_img, in_seq], out_word


# feature extractor model
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# sequence model
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
# tie it together [image, seq] [word]
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
    
# summarize model
print(model.summary())


mapping['1149179852_acad4d7300']

epochs = 20
steps = len(train)
for i in range(epochs):
    generator = data_generator(mapping,features, tokenizer, max_length, vocab_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('model_' + str(i) + '.h5')

    
# save the model
model.save(WORKING_DIR+'/best_model.h5')






















def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
     if index == integer:
        return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        print(sequence)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

print(features)
type(features)
from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

#print(mapping['436015762_8d0bae90c3'])

for key in tqdm(test):
    # get actual caption
    print(key in features.keys())
    print(features[key])
    if key in features:
        print(key)
        captions = mapping[key]
        # predict the caption for image
        y_pred = predict_caption(model, features[key], tokenizer, max_length)
        # split into words
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        # append to the list
        actual.append(actual_captions)
        predicted.append(y_pred)
        # calcuate BLEU score
        print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))




from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Flicker8k_Dataset", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
    
    
    
    
generate_caption("1007129816_e794419615.jpg")

print(mapping['50030244_02cd4de372'])



vgg_model = VGG16() 
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs,outputs=vgg_model.layers[-2].output)



image_path = r"E:\image_desc_gen_proj\Flicker8k_Dataset\49553964_cee950f3ba.jpg"
# load image
image = load_img(image_path, target_size=(224, 224))
# convert image pixels to numpy array
image = img_to_array(image)
# reshape data for model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# preprocess image from vgg
image = preprocess_input(image)
img__n = Image.open(image_path)
# extract features
feature = vgg_model.predict(image, verbose=0)
# predict from the trained model
predict_caption(model, feature, tokenizer, max_length)
plt.imshow(img__n)




