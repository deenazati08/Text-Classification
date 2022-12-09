# IMDB Review Text Classification using NLP



 
 



![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Description

Text classification steps:

1. Data Loading:

        • Download the dataset from Kaggle and load it to python by import os and pandas.

2. Data Inspection:
    
        • Check the data if there any duplicates and any other anomalities.

3. Data Cleaning:

        • As this dataset have many special characters, links and some other unneeded words, RegEx is used to remove it all.
    
        • While to drop all the duplicates text, np.drop_duplicates() is used.

4. Data Preprocessing:
    
        • First, tokenization were applied to the train data.
    
        • Then, padding and truncating it.

        • Fit the target with OneHotEncoder.
    
        • Split the data by using train_test_split.

    (a) Model Development:

        • Create layers with Embedding(as input), LSTM and Dense(as output).
    
        • Callbacks function were use. (Earlystopping and TensorBoard)

    (b) Model Evaluation:

       • Predict the data using model.predict and np.argmax

Last but not least, save all the model used 

## Result
Here is y-pred result:

![text_classification](https://user-images.githubusercontent.com/120104404/206621526-9f64c04d-a300-40da-895f-7855fa79d1a7.jpg)

Accuracy and Loss Graph from TensorBoard:

![epoch_acc](https://user-images.githubusercontent.com/120104404/206622686-1b02c0e8-f3c4-4794-b343-7e4fa58566cb.jpg)
![epoch_loss](https://user-images.githubusercontent.com/120104404/206622863-4ade7924-a1f2-4bac-bc7a-5db4e5fbf28e.jpg)


## Credits

- https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
