from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
model = SentenceTransformer("clip-ViT-B-32")

@app.post("/embed")
async def embed(image: UploadFile = File(...)):
    if not image:
        raise HTTPException(status_code=400, detail="No image uploaded")

    try:
        image_data = await image.read()
        img = Image.open(BytesIO(image_data)).convert("RGB")
        embedding = model.encode(img).tolist()
        return JSONResponse(content={"embedding": embedding})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")




