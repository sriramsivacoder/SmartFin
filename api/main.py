from fastapi import FastAPI, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import shutil
from core.orchestrator import Orchestrator

app = FastAPI()
orchestrator = Orchestrator()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-data")
async def upload_data(file: UploadFile):
    path = f"data/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "uploaded", "path": path}

@app.post("/analyze")
async def analyze(file_path: str = Query(...)):
    return orchestrator.spending_agent.process(orchestrator.preprocessor.load_csv(file_path))

@app.post("/optimize")
async def optimize(file_path: str = Query(...)):
    df = orchestrator.preprocessor.load_csv(file_path)
    monthly = df.groupby("category")["amount"].sum().to_dict()
    return orchestrator.budget_agent.process(monthly)

@app.post("/full-workflow")
async def full(file_path: str = Query(...)):
    return orchestrator.run_full_pipeline(file_path)