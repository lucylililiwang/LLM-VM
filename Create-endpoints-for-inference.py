# We are import the require library
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# We are define request models using Pydantic
class InferenceRequest(BaseModel):
    text: str
    
class FineTuningRequest(BaseModel):
    data: str
    
# We are define Endpoint for inference
@app.post('/infer/')
def inference(request: InferenceRequest):
    processed_text = request.text.upper()
    return {"result": processed_text}

# We are define Endpoint for fine-tuning
@app.post('/finetune/')
def fine_tuning(request: FineTuningRequest):
    processed_data = request.data + " fine-tuned"
    return {"result": processed_data}


if __name__ == "__main__":
    import uvicorn
    
    # We are Run the FastAPI app with Uvicorn server
    unicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
