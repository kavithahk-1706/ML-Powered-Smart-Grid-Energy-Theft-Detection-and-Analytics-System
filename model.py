import joblib
import pandas as pd
import json
from pathlib import Path
import numpy as np

class EnergyTheftPredictor:
    def __init__(self):
        model_path=(Path(__file__).parent/"data"/"models"/"random_forest_energy_theft.pkl").resolve()

        if not model_path.exists():
            
            raise FileNotFoundError(f"Model not found at: {model_path}")

        self.model=joblib.load(str(model_path))

        columns_path=Path(__file__).parent/"data"/"models"/"feature_columns.json"
        with open(columns_path, 'r') as f:
            self.feature_columns=json.load(f)

        print(f"Model loaded with {len(self.feature_columns)} features")

    def preprocess_input(self, input_data):
        '''takes input data from frontend and preprocesses it
        into the format that the model can process
        '''

        df=pd.DataFrame(input_data)

        #renaming columns

        col_map={
            "fans_electricity": "Fans:Electricity in kW (Hourly)",
            "cooling_electricity": "Cooling:Electricity in kW (Hourly)",
            "heating_electricity": "Heating:Electricity in kW (Hourly)",
            "interior_lights_electricity": "InteriorLights:Electricity in kW (Hourly)",
            "interior_equipment_electricity": "InteriorEquipment:Electricity in kW (Hourly)",
            "gas_facility": "Gas:Facility in kW (Hourly)",
            "heating_gas": "Heating:Gas in kW (Hourly)",
            "interior_equipment_gas": "InteriorEquipment:Gas in kW (Hourly)",
            "water_heater_gas": "Water Heater:WaterSystems:Gas in kW (Hourly)",
            "class": "Class"
        }

        df.rename(columns=col_map, inplace=True)

        df_encoded=pd.get_dummies(df, columns=["Class"], drop_first=False)

        #verify all expected columns - replace non-existing columns with 0 (in line with one-hot encoding
   
        for col in self.feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col]=0

        df_encoded=df_encoded[self.feature_columns]

        return df_encoded
    
    def predict(self, input_data):

        '''
        makes prediction
        returns 0 or 1 with probability of prediction/confidence
        '''
        
        is_batch=isinstance(input_data,list)


        if not is_batch:
            input_data=[input_data]

        processed_data=self.preprocess_input(input_data)



        predictions=self.model.predict(processed_data)
        probabilities=self.model.predict_proba(processed_data)

        results=[]

        for i,pred in enumerate(predictions):
            result = {
                "prediction": int(pred),
                "prediction_label": "Theft" if pred==1 else "Normal",
                "confidence": float(probabilities[i][pred]),
                "probabilities": {
                    "normal": float(probabilities[i][0]),
                    "theft": float(probabilities[i][1])
                }

            }
            results.append(result)

        if not is_batch:
            return results[0]
        
        return results




        