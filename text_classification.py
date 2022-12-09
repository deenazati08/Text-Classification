# %% 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Sequential
import numpy as np
import pandas as pd
import os, re, datetime

# %%
# Step 1: Data Loading
path = os.path.join(os.getcwd(), 'dataset', 'True.csv')
df = pd.read_csv(path)
text = df['text']       
subject = df['subject']    

# %% 
# Step 2: Data Visualization/inspection
df.info()
df.describe()
print(df.isna().sum())

print(text[0])

# %% 
# Step 3: Data Cleaning
for index, txt in enumerate(text):
    text[index] = re.sub('[^a-zA-Z]', ' ', txt).lower()
    
print(text[0])

# %% 
# Step 4: Features Selection

# %% 
# Step 5: Data Preprocessing
# Tokenizer
num_words = 5000
oov_token = '<OOV>'
pad_type = 'post'
trunc_type = 'post'

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(text)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(text)

# Padding + Truncating
train_sequences = pad_sequences(train_sequences, maxlen=200, padding=pad_type, truncating=trunc_type)

# %%
# OneHotEncoder
ohe = OneHotEncoder(sparse=False)
train_subject = ohe.fit_transform(subject[::,None])

# Train Test Split
train_sequences = np.expand_dims(train_sequences, -1)
X_train, X_test, y_train, y_test = train_test_split(train_sequences, train_subject)

# %% 
# Model Development
model = Sequential()
embedding_size = 128

model.add(Embedding(num_words, embedding_size))
model.add(LSTM(embedding_size, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# Callbacks
log_dir =  os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=log_dir)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, callbacks=[tb])

# %% Model Evaluation
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print(accuracy_score(y_true, y_pred))

# %% Saving Model
# save tokenizer
import json

with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

# save ohe
import pickle
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe,f)

# save deep learning model
model.save('model.h5')

# %%
