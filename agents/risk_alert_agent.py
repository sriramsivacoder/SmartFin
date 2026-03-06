"""
RiskAlertAgent — Detects financial risk patterns from transaction data.

Analyzes spending behavior to identify risky patterns such as spending spikes,
high category concentration, and variance-based instability.

Explainable AI: Every score includes a human-readable explanation of
the reasoning chain so users understand *why* a risk level was assigned.
"""

import pandas as pd
import numpy as np
from typing import Any


class RiskAlertAgent:
    """Agent that evaluates financial risk from transaction history."""

    # Thresholds for risk heuristics
    SPIKE_MULTIPLIER: float = 2.0        # transactions > 2× mean are spikes
    CONCENTRATION_THRESHOLD: float = 0.50  # single category > 50% triggers alert
    HIGH_VARIANCE_CV: float = 1.0         # coefficient of variation > 1.0 is risky
    ANOMALY_RATE_THRESHOLD: float = 0.10  # >10% anomaly rate is concerning

    def analyze_risk(self, transactions_df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze transaction data for financial risk patterns.

        Returns:
            dict with keys:
                - risk_score (float): 0.0–1.0 composite risk score
                - alerts (list[str]): human-readable risk alerts
                - risk_level (str): "low", "medium", or "high"
                - explanations (list[dict]): per-factor reasoning chain
        """
        if transactions_df.empty:
            return {
                "risk_score": 0.0,
                "alerts": ["No transactions to analyze"],
                "risk_level": "low",
                "explanations": [
                    {
                        "factor": "Data",
                        "reasoning": "No transaction data was provided, so risk cannot be evaluated.",
                        "impact": "none",
                    }
                ],
            }

        alerts: list[str] = []
        explanations: list[dict] = []
        sub_scores: list[float] = []

        # --- 1. Spending spike detection ---
        spike_score, spike_alerts, spike_expl = self._detect_spending_spikes(
            transactions_df
        )
        sub_scores.append(spike_score)
        alerts.extend(spike_alerts)
        explanations.append(spike_expl)

        # --- 2. Category concentration ---
        if "category" in transactions_df.columns:
            conc_score, conc_alerts, conc_expl = self._detect_category_concentration(
                transactions_df
            )
            sub_scores.append(conc_score)
            alerts.extend(conc_alerts)
            explanations.append(conc_expl)

        # --- 3. Spending variance / instability ---
        var_score, var_alerts, var_expl = self._detect_high_variance(
            transactions_df
        )
        sub_scores.append(var_score)
        alerts.extend(var_alerts)
        explanations.append(var_expl)

        # --- 4. Anomaly rate (if anomaly column exists) ---
        if "anomaly" in transactions_df.columns:
            anom_score, anom_alerts, anom_expl = self._detect_anomaly_rate(
                transactions_df
            )
            sub_scores.append(anom_score)
            alerts.extend(anom_alerts)
            explanations.append(anom_expl)

        # --- Composite risk score ---
        risk_score = float(np.clip(np.mean(sub_scores), 0.0, 1.0))
        risk_level = self._score_to_level(risk_score)

        # Top-level explanation of how the final score was computed
        explanations.insert(0, {
            "factor": "Overall Risk Score",
            "reasoning": self._explain_overall(risk_score, risk_level, sub_scores),
            "impact": risk_level,
        })

        return {
            "risk_score": round(risk_score, 4),
            "alerts": alerts if alerts else ["No significant risks detected"],
            "risk_level": risk_level,
            "explanations": explanations,
        }

    # ------------------------------------------------------------------
    # Explainability helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _explain_overall(score: float, level: str, sub_scores: list) -> str:
        """Generate a plain-language explanation of the composite score."""
        n = len(sub_scores)
        parts = [f"{s:.2f}" for s in sub_scores]
        reasoning = (
            f"The final risk score of {score:.2%} is the average of "
            f"{n} sub-scores ({', '.join(parts)}). "
        )
        if level == "low":
            reasoning += (
                "Because the composite score is below 30%, the overall "
                "risk level is classified as LOW — your finances appear stable."
            )
        elif level == "medium":
            reasoning += (
                "Because the composite score is between 30% and 60%, the "
                "risk level is MEDIUM — some spending patterns need attention."
            )
        else:
            reasoning += (
                "Because the composite score exceeds 60%, the risk level "
                "is HIGH — immediate corrective action is recommended."
            )
        return reasoning

    # ------------------------------------------------------------------
    # Private heuristic methods (return score, alerts, explanation)
    # ------------------------------------------------------------------

    def _detect_spending_spikes(
        self, df: pd.DataFrame
    ) -> tuple[float, list[str], dict]:
        """Flag transactions significantly above the mean amount."""
        alerts: list[str] = []
        mean_amount = df["amount"].mean()
        threshold = mean_amount * self.SPIKE_MULTIPLIER
        spike_count = int((df["amount"] > threshold).sum())
        spike_ratio = spike_count / len(df)
        score = min(spike_ratio / 0.20, 1.0)

        # --- Explainability ---
        if spike_count == 0:
            reasoning = (
                f"Average transaction is ₹{mean_amount:.2f}. "
                f"No transactions exceed 2× the average (₹{threshold:.2f}), "
                f"so the spike sub-score is 0.00. This is healthy."
            )
            impact = "none"
        elif spike_ratio < 0.10:
            reasoning = (
                f"Average transaction is ₹{mean_amount:.2f}. "
                f"{spike_count} of {len(df)} transactions ({spike_ratio:.1%}) "
                f"exceed 2× the average. This is a minor concern. "
                f"Sub-score: {score:.2f}."
            )
            impact = "low"
            alerts.append(
                f"Spending spike detected: {spike_count} transaction(s) "
                f"exceed 2× the average (₹{mean_amount:.2f})"
            )
        else:
            reasoning = (
                f"Average transaction is ₹{mean_amount:.2f}. "
                f"{spike_count} of {len(df)} transactions ({spike_ratio:.1%}) "
                f"exceed 2× the average — this is a significant spike pattern. "
                f"Sub-score: {score:.2f}."
            )
            impact = "high" if score > 0.6 else "medium"
            alerts.append(
                f"Spending spike detected: {spike_count} transaction(s) "
                f"exceed 2× the average (₹{mean_amount:.2f})"
            )

        explanation = {
            "factor": "Spending Spikes",
            "reasoning": reasoning,
            "impact": impact,
            "details": {
                "mean_amount": round(mean_amount, 2),
                "spike_threshold": round(threshold, 2),
                "spike_count": spike_count,
                "spike_ratio": round(spike_ratio, 4),
                "sub_score": round(score, 4),
            },
        }
        return score, alerts, explanation

    def _detect_category_concentration(
        self, df: pd.DataFrame
    ) -> tuple[float, list[str], dict]:
        """Detect over-reliance on a single spending category."""
        alerts: list[str] = []
        total = df["amount"].sum()

        if total == 0:
            return 0.0, alerts, {
                "factor": "Category Concentration",
                "reasoning": "Total spending is zero — no concentration risk.",
                "impact": "none",
            }

        category_pct = df.groupby("category")["amount"].sum() / total
        top_cat = category_pct.idxmax()
        max_pct = float(category_pct.max())
        score = min(max(max_pct - 0.40, 0.0) / 0.40, 1.0)

        # Build category breakdown for explanation
        cat_breakdown = {
            str(c): f"{p:.1%}" for c, p in category_pct.items()
        }

        for cat, pct in category_pct.items():
            if pct >= self.CONCENTRATION_THRESHOLD:
                alerts.append(
                    f"{str(cat).capitalize()} category exceeds "
                    f"{pct * 100:.0f}% of total spending"
                )

        # --- Explainability ---
        if max_pct < 0.40:
            reasoning = (
                f"Spending is well-diversified. The largest category is "
                f"'{top_cat}' at {max_pct:.1%}, which is below the 40% "
                f"risk-start threshold. Sub-score: {score:.2f}."
            )
            impact = "none"
        elif max_pct < self.CONCENTRATION_THRESHOLD:
            reasoning = (
                f"The largest category '{top_cat}' accounts for {max_pct:.1%} "
                f"of total spending. This is approaching the 50% alert "
                f"threshold. Sub-score: {score:.2f}."
            )
            impact = "low"
        else:
            reasoning = (
                f"Category '{top_cat}' dominates at {max_pct:.1%} of total "
                f"spending — exceeding the 50% threshold. Over-reliance on "
                f"one category increases financial fragility. "
                f"Sub-score: {score:.2f}."
            )
            impact = "high" if score > 0.6 else "medium"

        explanation = {
            "factor": "Category Concentration",
            "reasoning": reasoning,
            "impact": impact,
            "details": {
                "top_category": str(top_cat),
                "max_percentage": round(max_pct, 4),
                "breakdown": cat_breakdown,
                "sub_score": round(score, 4),
            },
        }
        return score, alerts, explanation

    def _detect_high_variance(
        self, df: pd.DataFrame
    ) -> tuple[float, list[str], dict]:
        """Evaluate spending instability via coefficient of variation."""
        alerts: list[str] = []
        mean_val = df["amount"].mean()
        std_val = df["amount"].std()

        if mean_val == 0:
            return 0.0, alerts, {
                "factor": "Spending Variance",
                "reasoning": "Mean spending is zero — variance cannot be computed.",
                "impact": "none",
            }

        cv = std_val / mean_val
        score = min(cv / 2.0, 1.0)

        # --- Explainability ---
        if cv <= 0.5:
            reasoning = (
                f"Coefficient of Variation (CV) is {cv:.2f} (std=₹{std_val:.2f}, "
                f"mean=₹{mean_val:.2f}). CV ≤ 0.5 indicates very stable and "
                f"predictable spending. Sub-score: {score:.2f}."
            )
            impact = "none"
        elif cv <= self.HIGH_VARIANCE_CV:
            reasoning = (
                f"CV is {cv:.2f} — moderate variability. Spending fluctuates "
                f"but remains within manageable bounds. Sub-score: {score:.2f}."
            )
            impact = "low"
        else:
            reasoning = (
                f"CV is {cv:.2f} — this exceeds the 1.0 threshold, meaning "
                f"the standard deviation (₹{std_val:.2f}) is larger than "
                f"the mean (₹{mean_val:.2f}). This signals unpredictable "
                f"spending that makes budgeting difficult. Sub-score: {score:.2f}."
            )
            impact = "high" if cv > 1.5 else "medium"
            alerts.append(
                f"High spending variance detected (CV={cv:.2f}), "
                f"indicating unstable spending patterns"
            )

        explanation = {
            "factor": "Spending Variance",
            "reasoning": reasoning,
            "impact": impact,
            "details": {
                "mean": round(mean_val, 2),
                "std_dev": round(std_val, 2),
                "cv": round(cv, 4),
                "sub_score": round(score, 4),
            },
        }
        return score, alerts, explanation

    def _detect_anomaly_rate(
        self, df: pd.DataFrame
    ) -> tuple[float, list[str], dict]:
        """Check the proportion of flagged anomalies (from IsolationForest)."""
        alerts: list[str] = []
        anomaly_count = int((df["anomaly"] == -1).sum())
        anomaly_rate = anomaly_count / len(df)
        score = min(anomaly_rate / 0.25, 1.0)

        # --- Explainability ---
        if anomaly_count == 0:
            reasoning = (
                "The Isolation Forest model flagged 0 transactions as anomalies. "
                "All spending appears within normal patterns. Sub-score: 0.00."
            )
            impact = "none"
        elif anomaly_rate <= self.ANOMALY_RATE_THRESHOLD:
            reasoning = (
                f"{anomaly_count} of {len(df)} transactions ({anomaly_rate:.1%}) "
                f"were flagged as anomalies by the ML model. This is within "
                f"the acceptable ≤10% range. Sub-score: {score:.2f}."
            )
            impact = "low"
        else:
            reasoning = (
                f"{anomaly_count} of {len(df)} transactions ({anomaly_rate:.1%}) "
                f"are anomalous — exceeding the 10% threshold. Repeated "
                f"anomalies suggest systematic issues like fraud risk or "
                f"lifestyle inflation. Sub-score: {score:.2f}."
            )
            impact = "high" if anomaly_rate > 0.20 else "medium"
            alerts.append(
                f"Repeated anomalies: {anomaly_count} anomalous "
                f"transactions ({anomaly_rate * 100:.1f}% of total)"
            )

        explanation = {
            "factor": "Anomaly Rate",
            "reasoning": reasoning,
            "impact": impact,
            "details": {
                "anomaly_count": anomaly_count,
                "total_transactions": len(df),
                "anomaly_rate": round(anomaly_rate, 4),
                "sub_score": round(score, 4),
            },
        }
        return score, alerts, explanation

    @staticmethod
    def _score_to_level(score: float) -> str:
        """Map a 0–1 risk score to a human-readable level."""
        if score < 0.3:
            return "low"
        elif score < 0.6:
            return "medium"
        return "high"