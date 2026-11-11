"""Rule-based retention insights.

These are heuristic, general-purpose business suggestions derived from simple rules about
common churn-associated attributes. They are NOT causal explanations and NOT guaranteed
interventions; model feature importance does not prove causation.
"""
from typing import List


def generate_insights(customer: dict, risk_category: str) -> List[dict]:
    """Return a list of {factor, observation, suggestion} dicts."""
    out = []

    def add(factor, obs, sug):
        out.append({"factor": factor, "observation": obs, "suggestion": sug})

    if customer.get("Contract") == "Month-to-month":
        add("Month-to-month contract",
            "Customers on month-to-month contracts have no long-term commitment and, in this dataset, churn more often.",
            "Consider offering an incentive to move to a one- or two-year plan.")
    tenure = float(customer.get("tenure", 0) or 0)
    if tenure <= 12:
        add("Short tenure",
            f"The customer has only {tenure:.0f} month(s) of tenure; early-life customers are often less settled.",
            "An onboarding check-in or welcome offer in the first year may help engagement.")
    if float(customer.get("MonthlyCharges", 0) or 0) >= 80:
        add("High monthly charges",
            "Monthly charges are in the upper range, which may increase price sensitivity.",
            "Review whether a better-fitting plan, bundle or loyalty discount could be offered.")
    if customer.get("InternetService") != "No" and customer.get("TechSupport") == "No":
        add("No technical support",
            "The customer has internet service but no tech-support add-on.",
            "Offering a trial of tech support or proactive service checks could improve experience.")
    if customer.get("InternetService") != "No" and customer.get("OnlineSecurity") == "No":
        add("No online security",
            "No online-security add-on is used alongside internet service.",
            "Highlighting bundled security features may increase perceived value.")
    if customer.get("PaymentMethod") == "Electronic check":
        add("Electronic check payment",
            "Electronic-check customers show a higher churn rate in the historical data.",
            "Encourage switching to automatic payment methods, e.g. with a small incentive.")
    if customer.get("InternetService") == "Fiber optic" and risk_category != "Low Risk":
        add("Fiber optic service",
            "Fiber-optic customers show higher churn in this dataset (possibly linked to price or service quality).",
            "Check service-quality feedback and price competitiveness for this segment.")

    if not out:
        out.append({"factor": "No prominent rule-based risk factors",
                    "observation": "None of the simple heuristic rules were triggered for this profile.",
                    "suggestion": "Continue regular engagement; monitor the customer over time."})
    return out
