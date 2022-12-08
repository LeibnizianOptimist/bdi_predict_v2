import os
import numpy as np
import pandas as pd

from bdi_predict.model.params import BASE_PROJECT_PATH
from bdi_predict.model.preprocessor import train_val_test_split, min_max_scaler


class SequenceGenerator():
  """
  
  """
  
  def __init__(self, input_width:int,
               target_width:int,
               offset:int,
               df_train:pd.DataFrame,
               df_val:pd.DataFrame,
               df_test:pd.DataFrame,
               target_columns=None
               ): 
    
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

    self.total_window_size = input_width + offset

  
  
    self.input_slice = slice(0, input_width)
    self.input_index = np.arange(self.total_window_size)[self.input_slice]

    self.target_start = self.total_window_size - self.target_width
    self.target_slice = slice(self.target_start, None)
    self.target_index = np.arange(self.total_window_size)[self.target_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_index}',
        f'Label indices: {self.target_index}',
        f'Label column name(s): {self.target_columns}'])

    
    
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