from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal
from model import EnergyTheftPredictor

app=FastAPI(
    title="Energy Theft Detection API",
    description="API for predicting electricity theft using machine learning",
    version="1.0.0"

)   


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

predictor=EnergyTheftPredictor()

class PredictionInput(BaseModel):
    fans_electricity: float = Field(..., description="Fans electricity consumption in kW")
    cooling_electricity: float = Field(..., description="Cooling electricity consumption in kW")
    heating_electricity: float = Field(..., description="Heating electricity consumption in kW")
    interior_lights_electricity: float = Field(..., description="Interior lights electricity consumption in kW")
    interior_equipment_electricity: float = Field(..., description="Interior equipment electricity consumption in kW")
    gas_facility: float = Field(..., description="Gas facility consumption in kW")
    heating_gas: float = Field(..., description="Heating gas consumption in kW")
    interior_equipment_gas: float = Field(..., description="Interior equipment gas consumption in kW")
    water_heater_gas: float = Field(..., description="Water heater gas consumption in kW")
    class_type: Literal[
        "FullServiceRestaurant", "Hospital", "LargeHotel","LargeOffice",
        "MediumOffice", "MidriseApartment", "OutPatient", "PrimarySchool", 
        "QuickServiceRestaurant", "SecondarySchool", "SmallHotel", "SmallOffice",
        "Stand-aloneRetail", "StripMall", "SuperMarket", "Warehouse"
    ] = Field(..., alias="class", description="Building class type")

    class Config:
        schema_extra={
            "example": {
                "fans_electricity": 1.5,
                "cooling_electricity": 2.3,
                "heating_electricity": 0.8,
                "interior_lights_electricity": 1.2,
                "interior_equipment_electricity": 3.4,
                "gas_facility": 0.5,
                "heating_gas": 0.5,
                "interior_equipment_gas": 0.1,
                "water_heater_gas": 0.2,
                "class": "SmallOffice"
            }
        }


#home
@app.get("/")
def read_root():
    return {
        "message": "Energy Theft Detection API is running",
        "status": "healthy",
        "model": "Random Forest",
        "version": "1.0.0"
    }

#single prediction
@app.post("/predict")
def predict_theft(data: PredictionInput):
    try:
        #convert from pydantic to dict
        input_dict=data.model_dump(by_alias=True)

        result=predictor.predict(input_dict)

        return {
            "success": True,
            "data": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#batch prediction
@app.post("/predict/batch")
def predict_batch(data: list[PredictionInput]):
    try:
        results=[]
        for item in data:
            input_dict=item.model_dump(by_alias=True)
            result=predictor.predict(input_dict)
            results.append(result)

        return {
            "success":True,
            "count": len(results),
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/model/info")
def get_model_info():
    return {
        "model_type": "Random Forest Classifier",
        "features_count": len(predictor.feature_columns),
        "features": predictor.feature_columns,
        "classes": ["Normal","Theft"]
    }


