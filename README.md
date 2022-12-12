# Model Process

1. Clean raw data using the data.py functions. This stores the data in the data/ directory.
2. Preprocess the data in preparation to be fed into a tensorflow.keras model using the functions in preprocessor.py and sequencer.py file. The preprocessing goes as follows:
    1. Split the full dataset using the train_test_val_split function.
    2. MinMax scale all the features (ensuring that the min and max values that are required for the stateful transformation that is MinMax scaling are derived from the training dataset.)
    3. Sequence the data using the SequencerGenerator class defined in the sequencer.py file. This is so that the data is inputted in the required format for a LSTM RNN model to learn from the data.
3. After having properly preprocessed the data, we can now initailise the model and train the model using the defined functions in model.py.
