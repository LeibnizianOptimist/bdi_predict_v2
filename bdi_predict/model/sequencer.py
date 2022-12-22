import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from bdi_predict.model.params import BASE_PROJECT_PATH
from bdi_predict.model.preprocessor import train_val_test_split, min_max_scaler

#Custom class based off of tf's WindowGenerator. 

class SequenceGenerator():
  """
  Generates the required data structure (Sequence Tensors) on which the LSTM model is trained on.
  """

  
  def __init__(self, 
               input_width:int,
               target_width:int,
               offset:int,
               df_train:pd.DataFrame,
               df_val:pd.DataFrame,
               df_test:pd.DataFrame,
               target_columns=None
               ):
    
    """
    Initilises generator with required variables and elements.
    """ 
    
    #Stores the data required prior to manipulation via instance methods.
    
    self.df_train = df_train
    self.df_val = df_val
    self.df_test = df_test

    # Work out the target column's index in a given generated Sequence.
  
    self.target_columns = target_columns
    
    if target_columns is not None:
      self.target_columns_index = {name: i for i, name in
                                    enumerate(target_columns)}

    self.column_index = {name: i for i, name in
                           enumerate(df_train.columns)}
    # Work out the sequence parameters.
    
    self.input_width = input_width
    self.target_width = target_width
    
    # Otherwise known as shift/horizon. 
    # The offset represents the gap between the last input value and the target value - TARGET INCLUSIVE.
  
    self.offset = offset

    self.total_sequence_size = input_width + offset

  #The first number in the slice() function represents the number batche_size (the number of sequences) that will passed into 
  #the LSTM model that will then try and optimise on.
  
    self.input_slice = slice(0, input_width)
    self.input_index = np.arange(self.total_sequence_size)[self.input_slice]

    self.target_start = self.total_sequence_size - self.target_width
    self.target_slice = slice(self.target_start, None)
    self.target_index = np.arange(self.total_sequence_size)[self.target_slice]



  def __repr__(self):
    return '\n'.join([
        f'Total sequence size: {self.total_sequence_size}',
        f'Input indices: {self.input_index}',
        f'Target indice(s): {self.target_index}',
        f'Target column name(s): {self.target_columns}'])
   
    
    
  def split_sequence(self,
                     features:tf.Tensor
                     ):
    """
    This instance method converts a list of consecutive inputs 
    into a seperate sequence of inputs with a corresponding seperate sequence of targets.
    It takes an an input a tf.Tensor wherein each row represents a single sequence that constitue a batch fed to the 
    LSTM model. 
    """
    
    inputs = features[:, self.input_slice, :]
    targets = features[:, self.target_slice, :]
    
    if self.target_columns != None:
      targets = tf.stack(
        [targets[:, :, self.column_index[name]] for name in self.target_columns], 
        axis= -1
      )
  
  #Slicing doesn't preserve static shape information, so set the shapes manually.
  #This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    targets.set_shape([None, self.target_width, None])
    
    #Shape of the np.ndarray should be (#Number of batches, #Number of observations, #Number of features)
    
    return inputs, targets
  
  
  
  def plot(self, model=None,
           plot_col="target",
           max_subplots=3
           ):
    """
    Visualises a split sequence created by the split_sequence method above.
    """
    inputs, targets = self.example
    
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col} [normalised]')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
              label='Inputs', marker='.', zorder=-10)

      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      
      else:
        label_col_index = plot_col_index
        

      if label_col_index is None:
        continue

      plt.scatter(self.label_indices, targets[n, :, label_col_index],
                  edgecolors='k', label='targets', c='#2ca02c', s=64)
      
      if model is not None:
        predictions = model(inputs)
        
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()

    plt.xlabel('Time [h]')
    
    return None
  
  
  
  def make_dataset(self, 
                   data:pd.DataFrame
                   ):
    """
    Creates datasets to be passed into the Keras model. This is done by calling
    the timeseries_dataset_from_array tf.keras function which "creates a dataset of sliding windows over a timeseries
    provided as an array". The kwarg "data" should take in a timesreies pd.DataFrame and applies the split_sequence method"""

    data = np.array(data, dtype=np.float64)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_sequence_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_sequence)
    
    

    return ds
  
  
  #Next three fucntions are properties that allows us to access the df_train, df_val, and df_test as tf.data.Datasets
  #using the make_dataset method defined above. Final @property decorated method creates a standard example batch for easy access and plotting.
  
  @property
  def train(self):
    return self.make_dataset(self.df_train)

  @property
  def val(self):
    return self.make_dataset(self.df_val)

  @property
  def test(self):
    return self.make_dataset(self.df_test)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result
    

    
if __name__ == "__main__":
  
  #Importing csv and applying train_val_test_split.
  df = pd.read_csv(os.path.join(BASE_PROJECT_PATH, "data", "cleaned_data.csv"))
  df.set_index("time", inplace=True)
  dfs = train_val_test_split(df, (7, 2, 1))
  df_train, df_val, df_test = min_max_scaler(dfs=dfs)
  
  
  #Testing out the SequenceGenerator works by creating an instance of the class
  sequence_sample = SequenceGenerator(input_width=20,
                                      target_width=1,
                                      offset=1,
                                      df_train=df_train,
                                      df_val=df_val,
                                      df_test=df_test, 
                                      target_columns=["target"])
  print(repr(sequence_sample))
  
    # Stack three slices, the length of the total window.
  example_sequence = tf.stack([np.array(df_train[:sequence_sample.total_sequence_size]),
                            np.array(df_train[100:100+sequence_sample.total_sequence_size]),
                            np.array(df_train[200:200+sequence_sample.total_sequence_size])])

  example_inputs, example_labels = sequence_sample.split_sequence(example_sequence)

  print('All shapes are: (batch, time, features)')
  print(f'Sequence shape: {example_sequence.shape}')
  print(f'Inputs shape: {example_inputs.shape}')
  print(f'Labels shape: {example_labels.shape}')
  
  
  #Show off the plots/visualisations of the sequences fucniton here when you run the file!
  


