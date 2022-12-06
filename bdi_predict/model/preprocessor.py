import numpy as np
import pandas as pd
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler


def final_preprocess(df:pd.DataFrame) -> TimeseriesGenerator:
    """
    Function Objective: obtains a set of X_predict values stored in a TimeseriesGenerator ready to be passed into our model.
    """
    
    def train_val_test_split(df:pd.DataFrame,
                        train_val_test_ratio: tuple) -> tuple:
        """ 
        Function Objective: returns a train, validation, and test dataframe (df_train, df_val, df_test).
        From this output one can sample (X,y) sequences using an instance of a SequenceGenerator Class (defined in sequencer.py)
        input_length is the length of the inputted dataframe. 
        df_train should contain all the timesteps until round(train_test_ratio * len(fold))   
        """
        
        #Creating input_length variable:
        
        input_length = len(df)
        
        #Taking the train_va_test_ratio input and converting it into individual ratios to allow for the split.
        
        train_ratio = train_val_test_ratio[0]*0.1
        val_ratio = train_val_test_ratio[1]*0.1
        test_ratio = train_val_test_ratio[2]*0.1
        #val_and_test_ratio = val_ratio + test_ratio
        assert train_ratio + val_ratio + test_ratio == 1
        
        
        # TRAIN SET
        last_train_idex = round(train_ratio * len(df))
        
        df_train = df.iloc[0:last_train_idex, :]
        
        
        # VALDIATION SET
        first_val_index = last_train_index - input_length
        
        magnitude_of_val = round(val_ratio*len(df))
        last_val_index = first_val_index + magnitude_of_val
        
        df_val = df.iloc[first_val_index:last_val_index, :]
        
        # TEST SET
        first_test_idx = last_val_index - input_length
        
        df_test = df.iloc[first_test_idx:, :]

        # CREATING X,Y FOR df_train AND df_test
        
        X_train = df_train.drop(columns="target").copy()
        y_train = df_train["target"]
        
        assert len(y_train) > 0
        
        X_val = df_val.drop(columns="target").copy()
        y_val = df_val["target"]
        
        assert len(y_val) > 0
        

        X_test  = df_test.drop(columns="target").copy()
        y_test = df_test["target"]
        
        assert len(y_test) > 0
        
        Xy_train = (X_train, y_train)
        Xy_val = (X_val, y_val)
        Xy_test = (X_test, y_test)
        
        
        return (Xy_train, Xy_val, Xy_test)

    def min_max_scaler_X(Xy_train:tuple) -> MinMaxScaler:
        """
        MinMaxScale X_inputs (FOR TRAINING AND TEST INPUTS ONLY)
        """
        
        scaler_X = MinMaxScaler()
        scaler_X.fit(Xy_train[0])
        
        #X_train_scaled = scaler_X.transform(Xy_train[0])
        #X_test_scaled = scaler_X.transform(Xy_test[0])
        
        return scaler_X
     
    def prediction_preprocessing(prediction_df:pd.DataFrame, scaler_X:MinMaxScaler) -> tuple:
        '''
        Function objective: Create a TimeseriesGenerator that will serve as the input to the model.predict() function that will called by the APi.
        '''
        X_input_prescaled = prediction_df[["Price", "CIP"]]
        X_input_scaled = scaler_X.transform(X_input_prescaled)
        
        y_true = np.array(prediction_df["log_diff"])

        predict_generator = TimeseriesGenerator(X_input_scaled, y_true, length=20, batch_size=1, sampling_rate=1, stride=1)
        

        return predict_generator
    
    #Actual running of the fuctions above:
    
    
    #Setting up prediction_df correctly: 
    prediction_df = pd.DataFrame(df.tail(21))

    
    Xy_train = train_test_split(df=df, train_test_ratio=0.8, input_length=len(df))
    
    scaler_X = min_max_scaler_X(Xy_train=Xy_train)
    
    predict_generator = prediction_preprocessing(prediction_df,  scaler_X)
    
    X20 = prediction_df.iloc[20,2]
    X20_abs = prediction_df.iloc[20, 1]
    
    return (predict_generator, X20, X20_abs)
    
        
    
    
    

    
 

