from api.routes.predict import predict
from pydantic import BaseModel
import numpy as np

class InputData(BaseModel):
    model_type: str
    features: list

data = InputData(model_type='stroke', features=[67, 0, 1, 228.69, 36.6, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1])
result = predict(data)
print('Result:', result)
