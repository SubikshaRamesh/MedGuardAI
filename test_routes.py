from fastapi import FastAPI
from api.routes.predict import router as predict_router
from api.routes.upload import router as upload_router

app = FastAPI()
app.include_router(predict_router, prefix='/api')
app.include_router(upload_router, prefix='/api')

print('Routes:')
for route in app.routes:
    print(route.path, route.methods)
