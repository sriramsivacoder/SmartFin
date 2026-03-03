from core.preprocessor import Preprocessor
from agents.spending_agent import SpendingAnalyzerAgent
from agents.budget_agent import BudgetOptimizationAgent

class Orchestrator:

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.spending_agent = SpendingAnalyzerAgent()
        self.budget_agent = BudgetOptimizationAgent()

    def run_full_pipeline(self, csv_path):
        df = self.preprocessor.load_csv(csv_path)

        # Add anomaly features
        df = self.preprocessor.get_anomaly_features(df)

        # Spending Analysis
        spending = self.spending_agent.process(df)

        # 1) MONTHLY TOTAL SPEND SUMMARY
        monthly_summary = self.preprocessor.get_monthly_summary(df)

        # 2) CATEGORY TOTAL SPEND SUMMARY
        category_totals = self.preprocessor.get_category_summary(df)

        # 3) Send both to Budget Agent
        budget = self.budget_agent.process(
            monthly_spend=monthly_summary,
            category_totals=category_totals
        )

        return {
            "spending": spending,
            "budget": budget
        }