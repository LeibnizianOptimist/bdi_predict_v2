import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta


def clean_data(df: pd.DataFrame, to_csv=True ) -> pd.DataFrame:
    """
    Taking .csv files stored in the raw_data folder and return a clearned .csv file.
    """
    
    print("Load data.")
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
        
        print("Data has been cleaned.")
        
        df.to_csv("../data/cleaned_data.csv")
        
        return None
    
    else:
        
        print("Data has been cleaned.")
        
        return df





