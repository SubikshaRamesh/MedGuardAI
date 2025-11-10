from api.routes.predict import predict
from pydantic import BaseModel

class TestData(BaseModel):
    model_type: str
    features: list

# Test stroke prediction
data = TestData(model_type='stroke', features=[50, 0, 0, 100, 25, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
result = predict(data)
print('Result:', result)
