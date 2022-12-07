import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

import os
from bdi_predict.model.params import BASE_PROJECT_PATH


def clean_data(to_csv=True) -> pd.DataFrame:
    """
    Function Objective: Takes .csv files stored in the raw_data folder and returns either a clearned .csv file or a pd.DataFrame of the cleaned data. 
    
    Note: the target values (log differences of daily BDRY values) are created in this function.
    
    Note: The kwarg 'to_csv' takes a boolean value where True would make the function create a .csv file in a certin directory within the project, 
    whilst the boolean False would return the cleaned data as a pd.DataFrame.
    """
    
    print("Loading data.")
    
     
    bdi_path = os.path.join(BASE_PROJECT_PATH, "raw_data", "BDI_daily.csv")
    bdi = pd.read_csv(bdi_path)
    bdi["time"].astype(int)
    bdi["time"] = pd.to_datetime(bdi["time"], unit="s", origin="unix")
    

    bdry_path = os.path.join(BASE_PROJECT_PATH, "raw_data", "BDRY_daily.csv")
    bdry = pd.read_csv(bdry_path)
    bdry["time"].astype(int)
    bdry["time"] = pd.to_datetime(bdry["time"], unit="s", origin="unix")
    
    #BDRY's datetime values are all of the place inter-day, so I will use the .normalize method to convert all the datetimes to midnight.

    midnight_datetime_list = []    
    for row in bdry.iterrows():
        #print(row[1]["time"])
        midnight_datetime = row[1]["time"].normalize()
        #print(midnight_datetime)
        midnight_datetime_list.append(midnight_datetime)
        
    bdry["time"] = pd.DataFrame(midnight_datetime_list)
   
        
 
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
        
        csv_delivery_path = os.path.join(BASE_PROJECT_PATH, "data", "cleaned_data.csv")
        df.to_csv(csv_delivery_path, index=False)
        
        print("Data has been cleaned & exported as a .csv file.")
        return None
    
    else:
        
        print("Data has been cleaned.")
        
        return df
    
     
if __name__ == "__main__":
    clean_data(to_csv=True)




