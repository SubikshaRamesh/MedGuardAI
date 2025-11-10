from api.utils.load_models import stroke_scaler, stroke_model
import numpy as np

test_data = np.array([[67, 0, 1, 228.69, 36.6, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]])
print('Test data shape:', test_data.shape)
print('Expected features for stroke model:', stroke_scaler.n_features_in_)
try:
    scaled = stroke_scaler.transform(test_data)
    print('Scaling successful')
    pred = stroke_model.predict(scaled)
    print('Prediction:', pred)
except Exception as e:
    print('Error:', e)
