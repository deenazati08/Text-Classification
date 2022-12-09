# %%
from sklearn.metrics import classification_report
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
target = df['subject'] 
 
# %%
# Step 2: Data Visualization/inspection
df.info()
print(df.isna().sum())
print(df.duplicated().sum())

# %%
# Step 3: Data Cleaning
temp = []      # to stre num words
for index, txt in enumerate(text):
    text[index] = re.sub(r'(^[^-]*)|(@[^\s]+)|bit.ly/\d\w{1,10}|(\s+EST)|[^a-zA-Z]', ' ', txt).lower()
    temp.append(len(text[index].split()))

df1 = pd.concat([text,target], axis=1)
df1 = df1.drop_duplicates()

# %%
# Step 4: Features Selection
text = df1['text']
target = df1['subject']

# %%
# Step 5: Data Preprocessing
# Tokenizer
num_words=5000
tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(text)

text_index = tokenizer.word_index

text = tokenizer.texts_to_sequences(text)

# Padding + Truncating
text = pad_sequences(text, maxlen=400, padding='post', truncating='post')

# %%
# OneHotEncoder
ohe = OneHotEncoder(sparse=False)
target = ohe.fit_transform(target[::,None])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(text, target)

# %%
# Model Development
model = Sequential()
model.add(Embedding(num_words, 64))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# Callbacks
log_dir =  os.path.join(os.getcwd(), 'logs_1', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=log_dir)
es = EarlyStopping(patience=5)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, callbacks=[tb, es])

# %% Model Evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))

# %%
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
