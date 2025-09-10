from fastapi import FastAPI
from routers.predict import router
from fastapi.staticfiles import StaticFiles


app = FastAPI()



# @app.get("/")
# async def home():
#     return {"Message" : "You're at home page of news predictor application."}


app.include_router(router, prefix="/api", tags=["Prediction"])

app.mount("/", StaticFiles(directory="static", html=True), name="static")





# if __name__ == "__main__":
    # app/

