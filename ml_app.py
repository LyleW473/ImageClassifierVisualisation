import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.ml.model.utils import load_model
from src.ml.inference.generators import get_imagenet1k_sample_generator, get_answer_generator
from fastapi import status

app = FastAPI()

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
        print("Error in prediction:", e)
        answer_json = {"error": str(e)}
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    json_response = JSONResponse(
                            content=answer_json,
                            status_code=status_code
                            )
    return json_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ml_app:app", host="127.0.0.1", port=8000)