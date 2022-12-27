# BDI Predict Development Log

See the ongoing development log for this project [here](https://open.substack.com/pub/leibnizianoptimist/p/bdi-prediction-model-development?r=hgyqx&utm_campaign=post&utm_medium=web)

## Model Process

1. Raw data is cleaned using functions in data.py. If the file itself is run, the cleaned data will be stored within a new directory "data/".
2. Preprocess the data. Preprocessing will crucially involve creating sequentialised data. The data is processed by the functions contained in  preprocessor.py and sequencer.py. The preprocessing is done in the following order:
    1. Dataset split using the train_test_val_split function (in a way that prevents data-leakage for timeseries data).
    2. Min-max feature scale/normalise all the features. The inimum and maximum values are zero and one in this case.
    3. Sequence the data using the SequencerGenerator class defined in the sequencer.py file. This is so that the data is in required confirguration.
3. After having properly preprocessed the data, we can now initailise the model and train a model using the defined functions in model.py.
4. Model tuning.
