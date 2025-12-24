import requests
import json
import pandas as pd

def prediction(data_df):
    url = "http://127.0.0.1:5002/invocations"
    headers = {"Content-Type": "application/json"}
    
    # Konversi DataFrame ke format yang dipahami MLflow (split format)
    data_json = {
        "dataframe_split": data_df.to_dict(orient='split')
    }
    
    # Kirim permintaan POST
    response = requests.post(url, data=json.dumps(data_json), headers=headers)
    print(response)
    
    if response.status_code == 200:
        predictions = response.json().get("predictions")
        
        # Logika pemetaan hasil (Mapping)
        # Jika hasil 1 adalah 'Churn' dan 0 adalah 'Not Churn'
        result = ["Churn" if p == 1 else "Not Churn" for p in predictions]
        return result
    else:
        return f"Error: {response.status_code}, {response.text}"