from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import model_service, preprocessing_service

router = APIRouter()

## request body schema - 
class PredictRequest(BaseModel):
    text: str

## response body schema
class PredictResponse(BaseModel):
    prediction: str 
    confidence: float

@router.post('/predict', response_model=PredictResponse)
async def predict_news(request: PredictRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
        cleaned_text = preprocessing_service.clean_text(request.text)
        
        prediction, confidence = model_service.predict(cleaned_text)

        return PredictResponse(prediction=prediction, confidence=confidence)
    
    except Exception as es:
        raise HTTPException(status_code=500, detail=f"Prediction failed {str(es)}")