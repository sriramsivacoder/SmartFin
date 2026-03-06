"""
InvestmentAdvisorAgent — Provides investment recommendations based on
spending and savings patterns.

Calculates savings rate and suggests allocation across emergency fund,
mutual funds, index funds, and fixed deposits using tiered strategies.

Explainable AI: Every recommendation includes a clear reasoning chain
so users understand the decision logic behind the advice.
"""

from typing import Any


class InvestmentAdvisorAgent:
    """Agent that generates simple investment advice from income/spending data."""

    def advise(
        self,
        monthly_income: float,
        total_spend: float,
        savings_amount: float,
    ) -> dict[str, Any]:
        """
        Generate investment recommendations with full explanations.

        Returns:
            dict with keys:
                - savings_rate (float): fraction of income saved
                - investment_plan (dict): allocation recommendations
                - advice (str): personalized financial guidance
                - explanations (list[dict]): reasoning chain for each decision
        """
        if monthly_income <= 0:
            return {
                "savings_rate": 0.0,
                "investment_plan": {},
                "advice": "Monthly income must be a positive number.",
                "explanations": [{
                    "factor": "Input Validation",
                    "reasoning": "Monthly income was zero or negative — cannot compute recommendations.",
                    "impact": "none",
                }],
            }

        savings_rate = round(
            (monthly_income - total_spend) / monthly_income, 4
        )
        monthly_expenses = total_spend
        emergency_months = (
            savings_amount / monthly_expenses if monthly_expenses > 0 else 0
        )

        explanations: list[dict] = []

        # --- 1. Savings Rate Analysis ---
        explanations.append(self._explain_savings_rate(
            savings_rate, monthly_income, total_spend
        ))

        # --- 2. Emergency Fund Assessment ---
        explanations.append(self._explain_emergency_fund(
            emergency_months, savings_amount, monthly_expenses
        ))

        # --- 3. Allocation Strategy ---
        investment_plan = self._build_plan(
            savings_rate, savings_amount, monthly_expenses
        )
        explanations.append(self._explain_allocation(
            savings_rate, investment_plan
        ))

        # --- 4. Overall Advice ---
        advice = self._generate_advice(
            savings_rate, savings_amount, monthly_expenses
        )

        return {
            "savings_rate": savings_rate,
            "investment_plan": investment_plan,
            "advice": advice,
            "explanations": explanations,
        }

    # ------------------------------------------------------------------
    # Explainability helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _explain_savings_rate(
        rate: float, income: float, spend: float
    ) -> dict:
        """Explain the savings rate classification."""
        saved = income - spend
        reasoning = (
            f"Monthly income is ₹{income:,.0f} and expenses are "
            f"₹{spend:,.0f}, leaving ₹{saved:,.0f} saved. "
            f"Savings rate = {rate:.1%}. "
        )
        if rate < 0:
            reasoning += (
                "⚠ You are spending MORE than you earn — this is unsustainable. "
                "Deficit spending erodes wealth and increases debt risk."
            )
            impact = "high"
        elif rate < 0.10:
            reasoning += (
                "A rate below 10% is considered low. Financial planners "
                "recommend saving at least 20% of income. Prioritize "
                "cutting discretionary expenses."
            )
            impact = "medium"
        elif rate < 0.20:
            reasoning += (
                "A 10-20% savings rate is moderate. You are saving, "
                "but there is room for improvement to build a stronger "
                "financial cushion."
            )
            impact = "low"
        elif rate < 0.40:
            reasoning += (
                "A 20-40% savings rate is healthy and above the "
                "recommended 20% minimum. You have good financial discipline."
            )
            impact = "none"
        else:
            reasoning += (
                "A savings rate above 40% is excellent! This provides "
                "strong capacity for both emergency reserves and "
                "aggressive investment growth."
            )
            impact = "none"

        return {
            "factor": "Savings Rate",
            "reasoning": reasoning,
            "impact": impact,
        }

    @staticmethod
    def _explain_emergency_fund(
        months: float, savings: float, expenses: float
    ) -> dict:
        """Explain the emergency fund adequacy."""
        reasoning = (
            f"Current savings of ₹{savings:,.0f} can cover "
            f"{months:.1f} months of expenses (₹{expenses:,.0f}/month). "
        )
        if months < 3:
            reasoning += (
                "This is critically low. Financial advisors recommend "
                "at least 3-6 months of expenses in liquid reserves. "
                "Without it, any income disruption could force debt."
            )
            impact = "high"
        elif months < 6:
            reasoning += (
                "This covers basic emergencies but falls short of the "
                "recommended 6-month target. Continue building this "
                "fund before taking on investment risk."
            )
            impact = "medium"
        else:
            reasoning += (
                "Your emergency fund exceeds the 6-month recommendation. "
                "You are well-protected against income disruptions and "
                "can confidently allocate surplus to investments."
            )
            impact = "none"

        return {
            "factor": "Emergency Fund",
            "reasoning": reasoning,
            "impact": impact,
        }

    @staticmethod
    def _explain_allocation(rate: float, plan: dict) -> dict:
        """Explain why a particular allocation strategy was chosen."""
        if rate < 0.10:
            tier = "Conservative (Emergency Priority)"
            reasoning = (
                "Because your savings rate is below 10%, the allocation "
                "heavily favors Fixed Deposits (70%) for capital "
                "preservation. Only 10% goes to Index Funds because "
                "you cannot afford volatility at this stage. "
                "Mutual Funds get 20% as a moderate-risk middle ground."
            )
        elif rate < 0.20:
            tier = "Moderate Conservative"
            reasoning = (
                "With a 10-20% savings rate, the allocation shifts "
                "slightly toward growth: Fixed Deposits 50%, Mutual "
                "Funds 30%, Index Funds 20%. The priority is still "
                "stability, but some growth exposure is introduced."
            )
        elif rate < 0.40:
            tier = "Balanced Growth"
            reasoning = (
                "A 20-40% savings rate allows a balanced portfolio: "
                "Index Funds 40% for long-term equity growth, Mutual "
                "Funds 30% for managed diversification, and Fixed "
                "Deposits 30% as a safety net. This balances risk "
                "and reward effectively."
            )
        else:
            tier = "Aggressive Growth"
            reasoning = (
                "With an excellent savings rate above 40%, the allocation "
                "maximizes growth potential: Index Funds 50% for "
                "low-cost equity exposure, Mutual Funds 30% for "
                "diversification, and only 20% in Fixed Deposits. "
                "Your strong savings buffer supports this higher-risk approach."
            )

        return {
            "factor": f"Allocation Strategy — {tier}",
            "reasoning": reasoning,
            "impact": "none",
        }

    # ------------------------------------------------------------------
    # Core logic (unchanged)
    # ------------------------------------------------------------------

    def _build_plan(
        self,
        savings_rate: float,
        savings_amount: float,
        monthly_expenses: float,
    ) -> dict[str, str]:
        """Determine asset-allocation recommendation based on savings rate."""
        emergency_months = (
            savings_amount / monthly_expenses if monthly_expenses > 0 else 0
        )

        if emergency_months < 6:
            emergency_advice = (
                f"Build emergency fund to cover 6 months of expenses "
                f"(currently {emergency_months:.1f} months covered)"
            )
        else:
            emergency_advice = (
                f"Emergency fund adequate ({emergency_months:.1f} months covered)"
            )

        if savings_rate < 0.10:
            return {
                "emergency_fund": emergency_advice,
                "fixed_deposits": "70%",
                "mutual_funds": "20%",
                "index_funds": "10%",
            }
        elif savings_rate < 0.20:
            return {
                "emergency_fund": emergency_advice,
                "fixed_deposits": "50%",
                "mutual_funds": "30%",
                "index_funds": "20%",
            }
        elif savings_rate < 0.40:
            return {
                "emergency_fund": emergency_advice,
                "fixed_deposits": "30%",
                "mutual_funds": "30%",
                "index_funds": "40%",
            }
        else:
            return {
                "emergency_fund": emergency_advice,
                "fixed_deposits": "20%",
                "mutual_funds": "30%",
                "index_funds": "50%",
            }

    @staticmethod
    def _generate_advice(
        savings_rate: float,
        savings_amount: float,
        monthly_expenses: float,
    ) -> str:
        """Return a single actionable advice string."""
        if savings_rate < 0:
            return (
                "You are spending more than you earn. "
                "Focus on reducing expenses before investing."
            )
        if savings_rate < 0.10:
            return (
                "Your savings rate is below 10%. "
                "Prioritize building an emergency fund and reducing discretionary spending."
            )
        if savings_rate < 0.20:
            return (
                "Moderate savings rate. Consider increasing savings to 20% "
                "before taking on higher-risk investments."
            )
        if savings_rate < 0.40:
            return (
                "Healthy savings rate. A balanced portfolio of index funds, "
                "mutual funds, and fixed deposits is recommended."
            )
        return (
            "Excellent savings rate! You can pursue aggressive growth "
            "with a larger allocation to index funds and equity mutual funds."
        )