from tensorflow import keras, data

from keras import Model, Sequential, layers

from keras.callbacks import EarlyStopping

from keras.optimizers import RMSprop
from keras.optimizers.schedules import ExponentialDecay

from data import Dataset

# FOR TYPE HINTS
from typing import Tuple


def init_model() -> Model:
    
    """ 
    Initialize the LSTM Reucrrent Neural Network.
    """
    
    print("\nInitialising model...")
    
    model = Sequential()

    #LSTM LAYERS:
    
    model.add(layers.LSTM(60,
                          activation="tanh",
                          input_shape=(20,1),
                          return_sequences=False))

    #DENSE LAYERS:
    
    model.add(layers.Dense(25, activation="relu"))
    model.add(layers.Dense(1, activation="linear"))
    
    print("\nmodel initialized.")

    #SETTING UP OPTIMIZERS:
    
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-3,
                                   decay_steps=10000,
                                   decay_rate=0.9)
    
    rmsprop = RMSprop(learning_rate=lr_schedule)
    
    #COMPILING MODEL:
    
    model.compile(loss="mse",
                  optimizer=rmsprop,
                  metrics="mae")
    print("\nmodel compiled.")
    

    return model




def train_model(model: Model,
                XandY:data.Dataset,
                patience=10,
                validation_data=data.Dataset) -> Tuple[Model]:
    
    """
    Fit model and return a the tuple (fitted_model, history)
    """
    
    print("\nTraining model...")
    
    #EarlyStopping DEFINITION:
    
    es = EarlyStopping(monitor="val_mae",
                       patience=patience,
                       restore_best_weights=True)
    
    #FITTING MODEL:
    
    history = model.fit(XandY,
                        epochs=100,
                        validation_data=validation_data,
                        shuffle=True,
                        callbacks=es)
    
    
    print(f"\nmodel trained ({len(XandY)} rows).")
     
    return model, history