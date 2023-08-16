#************** Start importing python modules
import pandas as pd
import json  
import zipfile  
#************** End importing python modules
#
# From https://sparkbyexamples.com/pandas/pandas-convert-json-to-dataframe/
#

# Use json_normalize() to convert JSON to DataFrame
#
#********** More info at https://stackoverflow.com/questions/40824807/reading-zipped-json-files ******
# The function "clean_data" forces the data type of each column to correspond to what is asserted in the meta data (sensordata)-json -file
def clean_data(df,meta_dict):
    #
    for var in list(df.columns):
        meta_type = meta_dict['properties'][var]['type']
        cast_func = None
        if meta_type =='string':
            cast_func = str
        elif meta_type == 'integer':
            cast_func = int
        elif meta_type == 'number':
            cast_func = float
        #        
        if cast_func is not None:
            df[var] = df[var].map(lambda x: cast_func(x))               
    #
    return df 
#
def convertJSON2Frame(jsonPath):
    meta_dict = None
    df = None
    d = None  # Contains "dictionary data"
    data = None 
    with zipfile.ZipFile(jsonPath, "r") as z:
        for filename in z.namelist():  
            with z.open(filename) as f: 
                data = f.read()  
                d = json.loads(data.decode("utf-8"))
                if meta_dict is None:
                    meta_dict = d
                else:
                    #Convert dictionary to dataframe
                    next_frame = pd.DataFrame.from_dict(d,orient="index").transpose()
                    if df is None:        
                        df  = next_frame
                    else:
                        df = pd.concat([df,next_frame])
    #
    df = clean_data(df,meta_dict)
    return df,meta_dict
# 
    
  