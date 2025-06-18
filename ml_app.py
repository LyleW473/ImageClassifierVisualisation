import json
import logging.config
import logging
import shutil
import os
import uuid

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.ml.model.utils import load_model
from src.ml.inference.generators import get_imagenet1k_sample_generator, get_answer_generator

from src.ml.backend.logging_config import LOGGING_CONFIG
from src.ml.inference.utils import save_image_locally
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

logging.config.dictConfig(LOGGING_CONFIG)
log = logging.getLogger(__name__)
log.info("ML app started successfully.")

model_name = "maxvit_tiny_tf_224.in1k"
model = load_model(model_name=model_name)
model.eval()

with open("in1k_cls_index.json", "r") as f:
    imagenet1k_cls_index = json.load(f)
imagenet1k_gen = get_imagenet1k_sample_generator(buffer_size=10)
answer_gen = get_answer_generator(
                                model=model,
                                sample_generator=imagenet1k_gen,
                                class_index=imagenet1k_cls_index
                                )
@app.get('/')
async def root():
    return {"message": "Hello World"}

@app.get('/health')
async def health() -> JSONResponse:
    """
    Health check endpoint.
    """
    return JSONResponse(
        content={"status": "healthy"},
        status_code=status.HTTP_200_OK
    )

@app.get('/predict')
async def get_prediction(request:Request) -> JSONResponse:
    """
    Get the next prediction from the answer generator.

    Args:
        request (Request): The incoming request object.
    """
    try:
        answer_json = next(answer_gen)
        status_code = status.HTTP_200_OK
    except Exception as e:
        log.error(f"Error in prediction: {e}")
        answer_json = {"error": str(e)}
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    # Save the image locally at /static endpoint.
    original_image = answer_json["originalImages"][0]  # Take the first image from the list
    shutil.rmtree("backend/static", ignore_errors=True) # Clear the static directory before saving new images
    os.makedirs("backend/static", exist_ok=True)
    image_name = f"predicted_image_{uuid.uuid4()}.jpg"
    save_image_locally(original_image, f"backend/static/{image_name}")
    image_url = request.url_for("static", path=image_name)
    log.info(f"Image saved at: {image_url}")
    answer_json["originalImagePath"] = str(image_url)
    json_response = JSONResponse(
                            content=answer_json,
                            status_code=status_code
                            )

    log.info(f"Response | Status code: {json_response.status_code} | Content: {json_response.body}")
    return json_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ml_app:app", host="127.0.0.1", port=8000)