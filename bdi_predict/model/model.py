import matplotlib.pyplot as plt

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
                          input_shape=(20,2),
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


def plot_history(history):
    """
    Plots the learning curves.
    """
    
    fig, ax = plt.subplots(1,2, figsize=(20,7))
    # Loss:MSE
    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('MSE')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='best')
    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)
    
    # Metrics:MAE
    
    ax[1].plot(history.history['mae'])
    ax[1].plot(history.history['val_mae'])
    ax[1].set_title('MAE')
    ax[1].set_ylabel('MAE')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='best')
    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)
                        
    return ax