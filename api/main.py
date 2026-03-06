from fastapi import FastAPI, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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




class InvestmentRequest(BaseModel):
    
    monthly_income: float
    total_spend: float
    savings_amount: float




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




@app.post("/risk-analysis")
async def risk_analysis(file_path: str = Query(...)):

    df = orchestrator.preprocessor.load_csv(file_path)
    return orchestrator.risk_agent.analyze_risk(df)


@app.post("/investment-advice")
async def investment_advice(request: InvestmentRequest):
    return orchestrator.investment_agent.advise(
        monthly_income=request.monthly_income,
        total_spend=request.total_spend,
        savings_amount=request.savings_amount,
    )