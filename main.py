from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import shutil
import uuid

from image_processor import enhance_image



app = FastAPI(
    title="AI Photo Studio API"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)



UPLOAD_FOLDER = Path("uploads")
RESULT_FOLDER = Path("results")


UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)




@app.get("/")
def home():

    return {
        "status":
        "AI Photo Studio Backend Online"
    }






@app.post("/process")
async def process_image(
    file:UploadFile = File(...)
):


    image_id = str(uuid.uuid4())


    input_file = (
        UPLOAD_FOLDER /
        f"{image_id}.jpg"
    )


    output_file = (
        RESULT_FOLDER /
        f"{image_id}.jpg"
    )



    with open(input_file,"wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )



    enhance_image(
        str(input_file),
        str(output_file)
    )



    return {


        "success":True,


        "result":
        f"/results/{image_id}.jpg"


    }
