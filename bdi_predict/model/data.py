import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta


def clean_data(to_csv=True) -> pd.DataFrame:
    """
    Function Objective: Takes .csv files stored in the raw_data folder and returns either a clearned .csv file or a pd.DataFrame of the cleaned data. 
    
    Note: the target values (log differences of daily BDRY values) are created in this function.
    
    Note: The kwarg 'to_csv' takes a boolean value where True would make the function create a .csv file in a certin directory within the project, 
    whilst the boolean False would return the cleaned data as a pd.DataFrame.
    """
    
    print("Loading data.")
    bdi = pd.read_csv("../raw_data/BDI_daily.csv")
    bdi["time"] = pd.to_datetime(bdi["time"], unit="s", origin="unix")
    

    bdry = pd.read_csv("../raw_data/BDRY_daily.csv")
    bdry["time"] = pd.to_datetime(bdry["time"], unit="s", origin="unix")
    
    
    for rows in bdry.iterrows():
        rows["time"] = rows["time"].normalize()

    bdry.drop(columns=["open",
                       "high",
                       "low",
                       "Volume",
                       "Volume MA"],
              inplace=True)  
     
    bdi.drop(columns=["open",
                      "high",
                      "low",
                      "Volume",
                      "Volume MA"],
             inplace=True)
    
        
    df = pd.merge(bdry,
                  bdi,
                  on="time",
                  how="left")
    
    df.rename(columns={"close_x":"BDRY",
                       "close_y":"BDI"},
              inplace=True)
    
    
    #Creating the Target, the log difference between dialy BDRY values.
    
    df["log_BDRY"] = np.log10(df["BDRY"])
    df["target"] = df["log_BDRY"].diff()
    df = df[df["time"] > "2018-03-22"].copy()
    
    if to_csv == True :
        
        print("Data has been cleaned & exported as a .csv file.")
        
        df.to_csv("../data/cleaned_data.csv")
        
        return None
    
    else:
        
        print("Data has been cleaned.")
        
        return df





