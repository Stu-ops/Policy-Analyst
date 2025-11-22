from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


@dataclass
class PremiumCalculation:
    base_premium: float
    risk_factor: float
    adjusted_premium: float
    annual_premium: float
    monthly_premium: float
    breakdown: Dict[str, float]


class RiskLevel(Enum):
    LOW = 0.8
    MEDIUM = 1.0
    HIGH = 1.2
    VERY_HIGH = 1.5


class ActuarialCalculator:
    def __init__(self):
        self.base_rates = {
            'auto': 1500,
            'health': 450,
            'home': 1200,
            'life': 50
        }
    
    def calculate_premium(self, policy_type: str, risk_factors: Dict[str, Any]) -> PremiumCalculation:
        """Calculate insurance premium based on risk factors"""
        base = self.base_rates.get(policy_type.lower(), 1000)
        
        risk_multiplier = 1.0
        breakdown = {}
        
        for factor, value in risk_factors.items():
            if isinstance(value, RiskLevel):
                multiplier = value.value
            elif isinstance(value, (int, float)):
                multiplier = value
            else:
                continue
            
            risk_multiplier *= multiplier
            breakdown[factor] = base * multiplier - base
        
        adjusted = base * risk_multiplier
        monthly = adjusted / 12
        
        return PremiumCalculation(
            base_premium=base,
            risk_factor=risk_multiplier,
            adjusted_premium=adjusted,
            annual_premium=adjusted,
            monthly_premium=monthly,
            breakdown=breakdown
        )
    
    def calculate_deductible_impact(self, coverage_amount: float, deductible: float) -> Dict[str, float]:
        """Calculate effective coverage with deductible"""
        effective_coverage = coverage_amount - deductible
        coverage_percentage = (effective_coverage / coverage_amount) * 100 if coverage_amount > 0 else 0
        
        return {
            'total_coverage': coverage_amount,
            'deductible': deductible,
            'effective_coverage': effective_coverage,
            'coverage_percentage': coverage_percentage,
            'out_of_pocket_max': deductible
        }
    
    def calculate_claim_payout(self, claim_amount: float, coverage_limit: float, 
                              deductible: float, coinsurance_percentage: float = 100) -> Dict[str, float]:
        """Calculate actual claim payout"""
        if claim_amount > coverage_limit:
            claim_amount = coverage_limit
        
        after_deductible = max(0, claim_amount - deductible)
        
        coinsurance_multiplier = coinsurance_percentage / 100
        payout = after_deductible * coinsurance_multiplier
        
        out_of_pocket = claim_amount - payout
        
        return {
            'claim_submitted': claim_amount,
            'after_deductible': after_deductible,
            'insurer_payout': payout,
            'member_out_of_pocket': out_of_pocket,
            'coverage_applied': coinsurance_percentage
        }
    
    def estimate_loss_frequency(self, policy_years: int, annual_claim_probability: float) -> Dict[str, Any]:
        """Estimate likelihood of claims over a period"""
        import math
        
        prob_no_claims = (1 - annual_claim_probability) ** policy_years
        prob_at_least_one = 1 - prob_no_claims
        expected_claims = policy_years * annual_claim_probability
        
        return {
            'policy_period_years': policy_years,
            'annual_claim_probability': annual_claim_probability,
            'probability_no_claims': prob_no_claims,
            'probability_at_least_one_claim': prob_at_least_one,
            'expected_number_of_claims': expected_claims
        }
    
    def calculate_net_present_value(self, premiums: List[float], claims: List[float], 
                                   discount_rate: float = 0.05) -> Dict[str, float]:
        """Calculate NPV for an insurance product"""
        npv_premiums = sum(p / ((1 + discount_rate) ** i) for i, p in enumerate(premiums))
        npv_claims = sum(c / ((1 + discount_rate) ** i) for i, c in enumerate(claims))
        
        net_npv = npv_premiums - npv_claims
        loss_ratio = npv_claims / npv_premiums if npv_premiums > 0 else 0
        
        return {
            'total_premiums_npv': npv_premiums,
            'total_claims_npv': npv_claims,
            'net_npv': net_npv,
            'loss_ratio': loss_ratio,
            'combined_ratio': loss_ratio + 0.25  # Add 25% for expenses
        }
    
    def extract_policy_numbers(self, policy_text: str) -> Dict[str, Any]:
        """Extract key numeric data from policy text"""
        import re
        
        results = {
            'coverage_limits': [],
            'deductibles': [],
            'premiums': [],
            'percentages': []
        }
        
        coverage_patterns = [
            r'\$[\d,]+\s*(?:coverage|limit)',
            r'(?:coverage|limit).*?\$[\d,]+'
        ]
        
        for pattern in coverage_patterns:
            matches = re.findall(pattern, policy_text, re.IGNORECASE)
            results['coverage_limits'].extend(matches)
        
        deductible_patterns = [
            r'\$[\d,]+\s*(?:deductible)',
            r'(?:deductible).*?\$[\d,]+'
        ]
        
        for pattern in deductible_patterns:
            matches = re.findall(pattern, policy_text, re.IGNORECASE)
            results['deductibles'].extend(matches)
        
        premium_patterns = [
            r'\$[\d,]+\s*(?:premium|annual|monthly)',
            r'(?:premium|annual).*?\$[\d,]+'
        ]
        
        for pattern in premium_patterns:
            matches = re.findall(pattern, policy_text, re.IGNORECASE)
            results['premiums'].extend(matches)
        
        percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentages = re.findall(percentage_pattern, policy_text)
        results['percentages'] = [float(p) for p in percentages]
        
        return results
