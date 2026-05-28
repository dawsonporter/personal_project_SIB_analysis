import warnings
import base64
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
from datetime import datetime
from scipy import stats
import logging
import json
import os
import io
import ssl
import hashlib
import time
from dash.exceptions import PreventUpdate

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

BASE_URL = "https://banks.data.fdic.gov/api"
DEFAULT_START_DATE = '20081231'
DEFAULT_END_DATE = datetime.today().strftime('%Y%m%d')
REQUESTED_START_DATE_DISPLAY = '12/31/2008'
CACHE_DIR = 'data_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
PRIMARY_BANK_DISPLAY_NAME = "JPMorgan Chase"
PRIMARY_BANK_ABBR = "JPM"
PRIMARY_BANK_FDIC_NAME = "JPMorgan Chase Bank, National Association"
DASHBOARD_TITLE = "JPMorgan Chase — Systemically Important Banks Dashboard"
DASHBOARD_SHORT_TITLE = "SIB Dashboard"
PEER_UNIVERSE_LABEL = "Systemically Important Banks (SIBs)"
PAIRED_GRAPH_HEIGHT = 340
PAIRED_CARD_MIN_HEIGHT = 432
OVERVIEW_GAUGE_SIZE = 78

# 45 days: safely below the ~90-day quarter gap so closest-match prior-period
# lookups can never grab an adjacent quarter. See compute_period_deltas.
PRIOR_PERIOD_TOLERANCE_DAYS = 45

CS = {
    # JPMorgan Chase-inspired blue theme for the primary benchmarked bank.
    'primary': '#005EB8', 'primary_light': '#2F7FD3', 'primary_dark': '#003B73',
    'secondary': '#333333', 'accent': '#0B4F8A', 'accent_light': '#E8F2FC',
    'bg': '#f4f6f9', 'bg_elevated': '#ffffff',
    'text': '#0f172a', 'text2': '#475569', 'text3': '#64748b',
    'light': '#94a3b8', 'lighter': '#cbd5e1',
    'grid': 'rgba(15,23,42,0.05)', 'border': 'rgba(15,23,42,0.06)',
    'border_strong': 'rgba(15,23,42,0.12)',
    'ghb': '#005EB8', 'peer': '#94a3b8', 'peer_op': 0.55,
    'good': '#16a34a', 'good_light': '#dcfce7', 'good_dark': '#166534',
    'warn': '#f59e0b', 'warn_light': '#fef3c7', 'warn_dark': '#b45309',
    'bad': '#ef4444', 'bad_light': '#fee2e2', 'bad_dark': '#b91c1c',
    'neutral': '#64748b', 'neutral_light': '#f1f5f9',
    'peer_band_top': '#475569', 'peer_band_mid': '#64748b', 'peer_band_low': '#94a3b8',
    'peer_band_bg': '#f8fafc', 'peer_tint': 'rgba(100,116,139,0.06)',
    'hover_bg': '#f8fafc', 'ghb2': '#0B4F8A',
    'gold': '#d4a017', 'silver': '#a8a8a8', 'bronze': '#cd7f32',
    'spark': '#005EB8', 'spark_area': 'rgba(0,94,184,0.08)',
}

CACHE_SCHEMA_VERSION = "v8_sib_jpm_20081231_no_comerica"

BANK_INFO = [
    # Primary FDIC-insured bank charters for the selected U.S. systemically
    # important / large-bank peer set. Display names are intentionally concise
    # for dashboard readability; BANK_NAME_MAPPING below preserves exact FDIC names.
    {"cert": "628",   "display": "JPMorgan Chase"},
    {"cert": "3510",  "display": "Bank of America"},
    {"cert": "3511",  "display": "Wells Fargo"},
    {"cert": "7213",  "display": "Citigroup"},
    {"cert": "33124", "display": "Goldman Sachs"},
    {"cert": "32992", "display": "Morgan Stanley"},
    {"cert": "639",   "display": "BNY Mellon"},
    {"cert": "14",    "display": "State Street"},
    {"cert": "6548",  "display": "U.S. Bancorp"},
    {"cert": "6384",  "display": "PNC"},
    {"cert": "9846",  "display": "Truist"},
    {"cert": "4297",  "display": "Capital One"},
    {"cert": "6672",  "display": "Fifth Third"},
    {"cert": "12368", "display": "Regions Financial"},
    {"cert": "6560",  "display": "Huntington"},
    {"cert": "588",   "display": "M&T Bank"},
    {"cert": "57957", "display": "Citizens Financial"},
    {"cert": "17534", "display": "KeyCorp"},
    {"cert": "57803", "display": "Ally Financial"},
]

CERT_TO_DISPLAY = {b["cert"]: b["display"] for b in BANK_INFO}

BANK_NAME_MAPPING = {
    "JPMORGAN CHASE BANK, NATIONAL ASSOCIATION": "JPMorgan Chase",
    "JPMORGAN CHASE BANK, N.A.": "JPMorgan Chase",
    "BANK OF AMERICA, NATIONAL ASSOCIATION": "Bank of America",
    "BANK OF AMERICA, N.A.": "Bank of America",
    "WELLS FARGO BANK, NATIONAL ASSOCIATION": "Wells Fargo",
    "WELLS FARGO BANK, N.A.": "Wells Fargo",
    "CITIBANK, NATIONAL ASSOCIATION": "Citigroup",
    "CITIBANK, N.A.": "Citigroup",
    "GOLDMAN SACHS BANK USA": "Goldman Sachs",
    "MORGAN STANLEY BANK, NATIONAL ASSOCIATION": "Morgan Stanley",
    "MORGAN STANLEY BANK, N.A.": "Morgan Stanley",
    "THE BANK OF NEW YORK MELLON": "BNY Mellon",
    "BANK OF NEW YORK MELLON, THE": "BNY Mellon",
    "STATE STREET BANK AND TRUST COMPANY": "State Street",
    "U.S. BANK NATIONAL ASSOCIATION": "U.S. Bancorp",
    "US BANK NATIONAL ASSOCIATION": "U.S. Bancorp",
    "PNC BANK, NATIONAL ASSOCIATION": "PNC",
    "PNC BANK, N.A.": "PNC",
    "TRUIST BANK": "Truist",
    "CAPITAL ONE, NATIONAL ASSOCIATION": "Capital One",
    "CAPITAL ONE, N.A.": "Capital One",
    "FIFTH THIRD BANK, NATIONAL ASSOCIATION": "Fifth Third",
    "FIFTH THIRD BANK, N.A.": "Fifth Third",
    "REGIONS BANK": "Regions Financial",
    "THE HUNTINGTON NATIONAL BANK": "Huntington",
    "HUNTINGTON NATIONAL BANK, THE": "Huntington",
    "HUNTINGTON NATIONAL BANK": "Huntington",
    "MANUFACTURERS AND TRADERS TRUST COMPANY": "M&T Bank",
    "CITIZENS BANK, NATIONAL ASSOCIATION": "Citizens Financial",
    "CITIZENS BANK, N.A.": "Citizens Financial",
    "KEYBANK NATIONAL ASSOCIATION": "KeyCorp",
    "KEYBANK N.A.": "KeyCorp",
    "ALLY BANK": "Ally Financial",
}

EXECUTIVE_KPIS = [
    ('Return on Assets', 'Profitability'),
    ('Net Interest Margin', 'Margin'),
    ('Efficiency Ratio', 'Efficiency'),
    ('Leverage (Core Capital) Ratio', 'Capital'),
    ('Nonaccrual / Total Loans', 'Credit'),
    ('Net Loan Growth Rate', 'Growth'),
]

DOLLAR_METRICS = {
    'Total Assets', 'Total Deposits', 'Gross Loans & Leases', 'Net Loans & Leases',
    'Total Securities', 'Total Earning Assets', 'Total Equity Capital', 'Tier 1 Capital',
    'Risk-Weighted Assets', 'Net Income (YTD)', 'Net Income (Quarter)',
    'Allowance for Credit Losses', 'Noncurrent Loans',
    'Gross Charge-Offs (YTD)', 'Gross Charge-Offs (Quarter)',
    'Gross Recoveries (YTD)', 'Gross Recoveries (Quarter)',
    'Brokered Deposits',
}

METRIC_ORDER = [
    'Return on Assets', 'Quarterly Return on Assets', 'Pretax Return on Assets',
    'Return on Equity', 'Quarterly Return on Equity',
    'Net Operating Income to Assets', 'Interest Income to Average Assets',
    'Interest Expense to Average Assets', 'Pre-Provision Net Revenue to Average Assets',
    'Provision for Credit Losses to Average Assets',
    'Yield on Earning Assets', 'Net Interest Margin', 'Cost of Funding Earning Assets',
    'Earning Assets / Total Assets', 'Efficiency Ratio',
    'Noninterest Expense to Average Assets', 'Salaries and Benefits to Average Assets',
    'Noninterest Income to Average Assets',
    'Common Equity Tier 1 (CET1) Ratio', 'Tier 1 Risk-Based Capital Ratio',
    'Leverage (Core Capital) Ratio', 'Total Risk-Based Capital Ratio',
    'Net Charge-Offs / Total Loans & Leases', 'ACL / Total Loans & Leases',
    'ACL / Nonaccrual Loans', 'ACL / 90+ DPD & Nonaccrual',
    'Loan Loss Reserve / Noncurrent Loans', 'Nonaccrual & OREO / Total Loans & OREO',
    '30-89 DPD / Total Loans', '90+ DPD / Total Loans',
    'Nonaccrual / Total Loans', '90+ DPD & Nonaccrual / Total Loans',
    'Net Loan Growth Rate', 'Earnings Coverage of Net Loan Charge-Offs',
    'Loan and Lease Loss Provision to Net Charge-Offs', 'Net Charge-Offs / ACL',
    'Net Loans and Leases to Assets',
    'Net Loans and Leases to Deposits', 'Core Deposits to Total Deposits',
    'Noninterest-Bearing Deposits to Total Deposits', 'Brokered Deposits to Total Deposits',
    'Volatile Liabilities to Total Assets',
    'Real Estate Loans to Tier 1 + ACL',
    'RE Construction and Land Development to Tier 1 + ACL',
    '1-4 Family Construction to Tier 1 + ACL',
    'Other Construction & Land Dev to Tier 1 + ACL',
    'Secured by Farmland to Tier 1 + ACL', '1-4 Family Residential to Tier 1 + ACL',
    'Revolving Home Equity to Tier 1 + ACL', 'Closed-End 1st Lien to Tier 1 + ACL',
    'Closed-End Jr Lien to Tier 1 + ACL', 'Multifamily RE to Tier 1 + ACL',
    'Non-Farm Non-Residential RE to Tier 1 + ACL', 'NFNR: Owner Occupied to Tier 1 + ACL',
    'NFNR: Non-Owner Occupied to Tier 1 + ACL', 'Commercial RE to Tier 1 + ACL',
    'Non-Owner Occupied CRE to Tier 1 + ACL',
    'Non-Owner Occupied CRE 3-Year Growth Rate', 'C&I Loans to Tier 1 + ACL',
    'Loans to Individuals to Tier 1 + ACL', 'Credit Cards to Tier 1 + ACL',
    'Auto Loans to Tier 1 + ACL', 'Agriculture Loans to Tier 1 + ACL',
    'Loans to NDFIs and Other to Tier 1 + ACL',
    'Total Asset Growth Rate', 'Tier 1 Capital Growth Rate',
    'Total Assets', 'Total Deposits', 'Gross Loans & Leases', 'Net Loans & Leases',
    'Total Securities', 'Total Earning Assets', 'Total Equity Capital', 'Tier 1 Capital',
    'Risk-Weighted Assets', 'Net Income (YTD)', 'Net Income (Quarter)',
    'Allowance for Credit Losses', 'Gross Charge-Offs (YTD)',
    'Gross Charge-Offs (Quarter)', 'Gross Recoveries (YTD)',
    'Gross Recoveries (Quarter)', 'Noncurrent Loans', 'Brokered Deposits',
]

METRIC_CATEGORIES = [
    ("Earnings & Profitability", METRIC_ORDER[0:10]),
    ("Efficiency & Margin", METRIC_ORDER[10:18]),
    ("Capitalization", METRIC_ORDER[18:22]),
    ("Asset Quality", METRIC_ORDER[22:32]),
    ("Loan & Lease Analysis", METRIC_ORDER[32:37]),
    ("Funding & Liquidity", METRIC_ORDER[37:42]),
    ("Credit Concentration", METRIC_ORDER[42:64]),
    ("Growth", METRIC_ORDER[64:66]),
    ("Key Financials", METRIC_ORDER[66:84]),
]

METRIC_TO_CATEGORY = {}
for cat_name, cat_metrics in METRIC_CATEGORIES:
    for m in cat_metrics:
        METRIC_TO_CATEGORY[m] = cat_name

CATEGORY_ACCENTS = {
    "Key Financials": "#0f172a",
    "Earnings & Profitability": "#16a34a", "Efficiency & Margin": "#2563eb",
    "Capitalization": "#d97706", "Asset Quality": "#dc2626",
    "Loan & Lease Analysis": "#7c3aed", "Funding & Liquidity": "#0891b2",
    "Credit Concentration": "#64748b", "Growth": "#4f46e5",
}
CATEGORY_BG = {
    "Key Financials": "#f1f5f9",
    "Earnings & Profitability": "#f0fdf4", "Efficiency & Margin": "#eff6ff",
    "Capitalization": "#fffbeb", "Asset Quality": "#fef2f2",
    "Loan & Lease Analysis": "#f5f3ff", "Funding & Liquidity": "#ecfeff",
    "Credit Concentration": "#f8fafc", "Growth": "#eef2ff",
}

INVERSE_METRICS = {
    'Efficiency Ratio', 'Interest Expense to Average Assets',
    'Cost of Funding Earning Assets', 'Noninterest Expense to Average Assets',
    'Salaries and Benefits to Average Assets',
    'Provision for Credit Losses to Average Assets',
    'Net Charge-Offs / Total Loans & Leases',
    'Nonaccrual & OREO / Total Loans & OREO',
    '30-89 DPD / Total Loans', '90+ DPD / Total Loans',
    'Nonaccrual / Total Loans', '90+ DPD & Nonaccrual / Total Loans',
    'Net Charge-Offs / ACL',
    'Brokered Deposits to Total Deposits', 'Volatile Liabilities to Total Assets',
    'Gross Charge-Offs (YTD)', 'Gross Charge-Offs (Quarter)', 'Noncurrent Loans',
}

# FIX #1: UBPR "X multiplier" ratios. Previously empty, so these rendered as
# "%" (e.g. "1.50%" / "151.00%"). Now rendered with an "x" suffix. The two
# ACL coverage ratios below are also computed at scale=1.0 in _aq.
NON_PERCENT_RATIO_METRICS = {
    'Earnings Coverage of Net Loan Charge-Offs',
    'ACL / Nonaccrual Loans',
    'ACL / 90+ DPD & Nonaccrual',
}

METRIC_DEFINITIONS = {
    'Total Assets': "Key Financials \xb7 Sum of all assets. FDIC field: ASSET. Call Report Schedule RC line 12. Values in $000s.",
    'Total Deposits': "Key Financials \xb7 Total domestic and foreign deposits. FDIC field: DEP. Schedule RC line 13. Values in $000s.",
    'Gross Loans & Leases': "Key Financials \xb7 Total loans and lease financing receivables before ACL. FDIC field: LNLSGR. Schedule RC-C. Values in $000s.",
    'Net Loans & Leases': "Key Financials \xb7 Total loans and leases NET of allowance for credit losses. FDIC field: LNLSNET. Used in the UBPR Net Loans & Leases / Deposits ratio. Values in $000s.",
    'Total Securities': "Key Financials \xb7 HTM + AFS securities. FDIC field: SC. Schedule RC line 2. Values in $000s.",
    'Total Earning Assets': "Key Financials \xb7 Interest-bearing balances + securities + net loans + fed funds sold + trading assets. FDIC field: ERNAST. Values in $000s.",
    'Total Equity Capital': "Key Financials \xb7 Total equity capital including AOCI. FDIC field: EQ. Schedule RC line 28. Values in $000s.",
    'Tier 1 Capital': "Key Financials \xb7 Regulatory Tier 1 capital (CET1 + AT1). FDIC field: RBCT1J. Schedule RC-R. Values in $000s.",
    'Risk-Weighted Assets': "Key Financials \xb7 Total risk-weighted assets. FDIC field: RWAJ. Schedule RC-R. Values in $000s.",
    'Net Income (YTD)': "Key Financials \xb7 Year-to-date net income after taxes. FDIC field: NETINC. Values in $000s.",
    'Net Income (Quarter)': "Key Financials \xb7 Current-quarter net income after taxes. FDIC field: NETINCQ. Values in $000s.",
    'Allowance for Credit Losses': "Key Financials \xb7 ACL on loans & leases HFI (CECL). FDIC field: LNATRES. RC line 4.c. Values in $000s.",
    'Gross Charge-Offs (YTD)': "Key Financials \xb7 Year-to-date gross charge-offs. FDIC field: DRLNLS. Values in $000s.",
    'Gross Charge-Offs (Quarter)': "Key Financials \xb7 Current-quarter gross charge-offs. FDIC field: DRLNLSQ. Values in $000s.",
    'Gross Recoveries (YTD)': "Key Financials \xb7 Year-to-date gross recoveries. FDIC field: CRLNLS. Values in $000s.",
    'Gross Recoveries (Quarter)': "Key Financials \xb7 Current-quarter recoveries. FDIC field: CRLNLSQ. Values in $000s.",
    'Noncurrent Loans': "Key Financials \xb7 Nonaccrual + 90+ DPD still accruing. Computed: NALNLS + P9LNLS. Values in $000s.",
    'Brokered Deposits': "Key Financials \xb7 All brokered deposits. FDIC field: BRO. Schedule RC-E. Values in $000s.",
    'Return on Assets': "Earnings \xb7 Annualized net income as % of average total assets. FDIC field: ROA. Already annualized by FDIC.",
    'Quarterly Return on Assets': "Earnings \xb7 Current-quarter net income as % of average total assets. FDIC field: ROAQ.",
    'Pretax Return on Assets': "Earnings \xb7 Pretax net operating income (TE) as % of avg assets. FDIC field: ROAPTX. UBPR Pg1 #12 (UBPRE009).",
    'Return on Equity': "Earnings \xb7 Annualized net income as % of avg equity. FDIC field: ROE. Already annualized by FDIC.",
    'Quarterly Return on Equity': "Earnings \xb7 Current-quarter net income as % of average equity. FDIC field: ROEQ.",
    'Net Operating Income to Assets': "Earnings \xb7 Adjusted net operating income as % of avg assets. FDIC field: NOIJY. UBPR Pg1 #13 (UBPRE010).",
    'Interest Income to Average Assets': "Earnings \xb7 Total interest income (TE) as % of avg assets. FDIC field: INTINCR. UBPR Pg1 #1 (UBPRE001).",
    'Interest Expense to Average Assets': "Earnings \xb7 Total interest expense as % of avg assets. FDIC field: EINTEXPR. UBPR Pg1 #2 (UBPRE002).",
    'Pre-Provision Net Revenue to Average Assets': "Earnings \xb7 PPNR = (II \u2212 IE + NII \u2212 NIE) / Avg Assets. UBPR Pg1 #6 (UBPRPG69). Computed from component fields.",
    'Provision for Credit Losses to Average Assets': "Earnings \xb7 Provision for credit losses as % of avg assets. FDIC field: ELNATRR. UBPR Pg1 #7 (UBPRE006).",
    'Yield on Earning Assets': "Margin \xb7 Interest income (TE, annualized) as % of avg earning assets. FDIC field: INTINCY. UBPR Pg1 #19 (UBPRE016).",
    'Net Interest Margin': "Margin \xb7 Net interest income (TE) as % of avg earning assets. Typical: 2.50\u20133.80%. FDIC field: NIMY. UBPR Pg1 #21 (UBPRE018).",
    'Cost of Funding Earning Assets': "Margin \xb7 Interest expense as % of avg earning assets. FDIC field: INTEXPYQ. UBPR Pg1 #20 (UBPRE017).",
    'Earning Assets / Total Assets': "Margin \xb7 Earning assets as % of total assets. FDIC field: ERNASTR. UBPR Pg1 #17 (UBPRE014).",
    'Efficiency Ratio': "Margin \xb7 NIE / (NII + noninterest income). Lower is better. FDIC field: EEFFR. UBPR Pg3 (UBPRE095).",
    'Noninterest Expense to Average Assets': "Margin \xb7 Total noninterest expense as % of avg assets. FDIC field: NONIXR. UBPR Pg1 #5 (UBPRE005).",
    'Salaries and Benefits to Average Assets': "Margin \xb7 Personnel expense as % of avg assets. FDIC field: ESALR. UBPR Pg3.",
    'Noninterest Income to Average Assets': "Margin \xb7 Total fee/noninterest income as % of avg assets. FDIC field: NONIIR. UBPR Pg1 #4 (UBPRE004).",
    'Common Equity Tier 1 (CET1) Ratio': "Capitalization \xb7 CET1 capital to RWA. Well-capitalized: \u22656.5%. FDIC field: IDT1CER. UBPR Pg11.",
    'Tier 1 Risk-Based Capital Ratio': "Capitalization \xb7 Tier 1 capital to RWA. Well-capitalized: \u22658%. FDIC field: IDT1RWAJR. UBPR Pg11.",
    'Leverage (Core Capital) Ratio': "Capitalization \xb7 Tier 1 capital to avg total assets. Well-capitalized: \u22655%. FDIC field: RBC1AAJ. UBPR Pg1 #33 (UBPRD486).",
    'Total Risk-Based Capital Ratio': "Capitalization \xb7 Total capital (T1 + T2) to RWA. Well-capitalized: \u226510%. FDIC field: RBCRWAJ. UBPR Pg1 #34 (UBPRD488).",
    'Net Charge-Offs / Total Loans & Leases': "Asset Quality \xb7 Net charge-offs as % of avg total L&L, annualized. FDIC field: NTLNLSR. UBPR Pg1 #22 (UBPRE019).",
    'ACL / Total Loans & Leases': "Asset Quality \xb7 ACL as % of LN&LS HFI. FDIC field: LNATRESR. UBPR Pg1 #24 (UBPRE022).",
    'ACL / Nonaccrual Loans': "Asset Quality \xb7 ACL as a MULTIPLE of nonaccrual loans. Well-reserved: \u22651.00x. UBPR Pg1 #26 (UBPRE395). Shown as X multiplier (1.51x = 151%).",
    'ACL / 90+ DPD & Nonaccrual': "Asset Quality \xb7 ACL as a MULTIPLE of (nonaccrual + 90+ DPD still accruing). Shown as X multiplier. Dashboard-computed.",
    'Loan Loss Reserve / Noncurrent Loans': "Asset Quality \xb7 ACL as % of noncurrent loans. Below 100% = reserves may not cover problems. FDIC field: LNRESNCR. UBPR Pg1 #36 (UBPRNC98).",
    'Nonaccrual & OREO / Total Loans & OREO': "Asset Quality \xb7 (Nonaccrual + OREO + 90+ DPD) / (total loans + OREO). Broadest NPA measure. UBPR Pg1 #29 (UBPRE549).",
    '30-89 DPD / Total Loans': "Asset Quality \xb7 Early-stage delinquency. UBPR Pg1 #27 (UBPRE544).",
    '90+ DPD / Total Loans': "Asset Quality \xb7 Seriously delinquent but still accruing. UBPR Pg8. Dashboard-computed.",
    'Nonaccrual / Total Loans': "Asset Quality \xb7 Loans where interest recognition suspended. UBPR Pg8. Dashboard-computed.",
    '90+ DPD & Nonaccrual / Total Loans': "Asset Quality \xb7 Total noncurrent as % of portfolio. UBPR Pg1 #28 (UBPR7414).",
    'Net Loan Growth Rate': "Loan & Lease \xb7 YoY growth of net L&L. >20% YoY triggers regulatory scrutiny. UBPR Pg1 #39 (UBPRE027).",
    'Earnings Coverage of Net Loan Charge-Offs': "Loan & Lease \xb7 Times net income covers NCOs. <1.0x means losing money after credit losses. FDIC field: IDERNCVR. UBPR Pg1 #23 (UBPRE020). Shown as X multiplier.",
    'Loan and Lease Loss Provision to Net Charge-Offs': "Loan & Lease \xb7 Provision expense as % of NCOs. >100% = building reserves. FDIC field: ELNANTR. UBPR Pg7.",
    'Net Charge-Offs / ACL': "Loan & Lease \xb7 Rolling 4-quarter NCOs as % of ACL. Dashboard-computed; requires four contiguous quarters, else N/A.",
    'Net Loans and Leases to Assets': "Loan & Lease \xb7 Net L&L as % of total assets. FDIC field: LNLSNTV. UBPR Pg1 #31 (UBPRE024).",
    'Net Loans and Leases to Deposits': "Funding \xb7 Net loans as % of total deposits. Typical: 70\u201395%. FDIC field: LNLSDEPR. UBPR Pg1 #32 (UBPRE600).",
    'Core Deposits to Total Deposits': "Funding \xb7 Core deposits as % of total deposits. Computed: COREDEP / DEP. UBPR Pg4.",
    'Noninterest-Bearing Deposits to Total Deposits': "Funding \xb7 Domestic NIB deposits as % of total deposits. Computed: DEPNIDOM / DEP. UBPR Pg4.",
    'Brokered Deposits to Total Deposits': "Funding \xb7 All brokered deposits as % of total deposits. \u226520% signals vulnerability. Computed: BRO / DEP. UBPR Pg4.",
    'Volatile Liabilities to Total Assets': "Funding \xb7 Volatile liabilities as % of total assets. FDIC field: VOLIABR. UBPR Pg10.",
    'Real Estate Loans to Tier 1 + ACL': "Concentration \xb7 Total RE loans as % of Tier 1 + ACL. UBPR Pg7B 7B.1 (UBPRE884).",
    'RE Construction and Land Development to Tier 1 + ACL': "Concentration \xb7 Construction & land dev as % of Tier 1 + ACL. UBPR Pg7B 7B.2 (UBPRD490).",
    '1-4 Family Construction to Tier 1 + ACL': "Concentration \xb7 1-4 family construction as % of Tier 1 + ACL. UBPR Pg7B 7B.3 (UBPRE632).",
    'Other Construction & Land Dev to Tier 1 + ACL': "Concentration \xb7 Other construction & land dev as % of Tier 1 + ACL. UBPR Pg7B 7B.4.",
    'Secured by Farmland to Tier 1 + ACL': "Concentration \xb7 Farmland RE as % of Tier 1 + ACL. UBPR Pg7B 7B.5.",
    '1-4 Family Residential to Tier 1 + ACL': "Concentration \xb7 Total 1-4 family residential as % of Tier 1 + ACL. UBPR Pg7B 7B.6.",
    'Revolving Home Equity to Tier 1 + ACL': "Concentration \xb7 HELOCs as % of Tier 1 + ACL. UBPR Pg7B 7B.7.",
    'Closed-End 1st Lien to Tier 1 + ACL': "Concentration \xb7 Closed-end first lien 1-4 family as % of Tier 1 + ACL. UBPR Pg7B 7B.8.",
    'Closed-End Jr Lien to Tier 1 + ACL': "Concentration \xb7 Junior lien as % of Tier 1 + ACL. UBPR Pg7B 7B.9.",
    'Multifamily RE to Tier 1 + ACL': "Concentration \xb7 Multifamily (5+ units) as % of Tier 1 + ACL. UBPR Pg7B 7B.10.",
    'Non-Farm Non-Residential RE to Tier 1 + ACL': "Concentration \xb7 NFNR total as % of Tier 1 + ACL. UBPR Pg7B 7B.11.",
    'NFNR: Owner Occupied to Tier 1 + ACL': "Concentration \xb7 Owner-occupied NFNR as % of Tier 1 + ACL. UBPR Pg7B 7B.12.",
    'NFNR: Non-Owner Occupied to Tier 1 + ACL': "Concentration \xb7 Non-owner-occupied NFNR (investor CRE) as % of Tier 1 + ACL. UBPR Pg7B 7B.13.",
    'Commercial RE to Tier 1 + ACL': "Concentration \xb7 UBPR Total CRE = Construction + Multifamily + NFNR total + LNCOMRE, as % of Tier 1 + ACL. UBPR Pg7B 7B.26.",
    'Non-Owner Occupied CRE to Tier 1 + ACL': "Concentration \xb7 NOO CRE = Construction + Multifamily + NFNR NOO + LNCOMRE, as % of Tier 1 + ACL. Interagency CRE guidance metric. UBPR Pg7B 7B.24.",
    'Non-Owner Occupied CRE 3-Year Growth Rate': "Growth \xb7 3-year growth of NOO CRE. >300% concentration AND >36% 3-year growth trigger enhanced risk management. UBPR Pg7B 7B.25.",
    'C&I Loans to Tier 1 + ACL': "Concentration \xb7 C&I loans as % of Tier 1 + ACL. UBPR Pg7B 7B.17 (UBPRE887).",
    'Loans to Individuals to Tier 1 + ACL': "Concentration \xb7 Total consumer loans as % of Tier 1 + ACL. UBPR Pg7B 7B.18 (UBPRE888).",
    'Credit Cards to Tier 1 + ACL': "Concentration \xb7 Credit card loans as % of Tier 1 + ACL. UBPR Pg7B 7B.19.",
    'Auto Loans to Tier 1 + ACL': "Concentration \xb7 Auto loans as % of Tier 1 + ACL. UBPR Pg7B 7B.20.",
    'Agriculture Loans to Tier 1 + ACL': "Concentration \xb7 Agriculture loans (non-RE) as % of Tier 1 + ACL. UBPR Pg7B 7B.16 (UBPRE886).",
    'Loans to NDFIs and Other to Tier 1 + ACL': "Concentration \xb7 Loans to nondepository FIs and other as % of Tier 1 + ACL. UBPR Pg7B 7B.22.",
    'Total Asset Growth Rate': "Growth \xb7 YoY growth of total assets. \u226530% YoY invites regulatory questions. UBPR Pg1 #37 (UBPR7316).",
    'Tier 1 Capital Growth Rate': "Growth \xb7 YoY growth of Tier 1 capital. UBPR Pg1 #38 (UBPR7408).",
}


def normalize_bank_name(n):
    if not n: return n
    if n in BANK_NAME_MAPPING: return BANK_NAME_MAPPING[n]
    u = n.upper().strip()
    for o, d in BANK_NAME_MAPPING.items():
        if o.upper().strip() == u: return d
    return n


def is_dollar_metric(m):
    return m in DOLLAR_METRICS


def is_inverse_metric(m):
    return m in INVERSE_METRICS


def is_multiplier_metric(m):
    return m in NON_PERCENT_RATIO_METRICS


def is_percent_metric(m):
    if m is None: return False
    if m in DOLLAR_METRICS: return False
    if m in NON_PERCENT_RATIO_METRICS: return False
    return True


def safe_div(numerator, denominator, scale=100.0):
    if numerator is None or denominator is None:
        return None
    if pd.isna(numerator) or pd.isna(denominator):
        return None
    try:
        n = float(numerator); d = float(denominator)
    except (TypeError, ValueError):
        return None
    if d == 0 or not np.isfinite(d) or not np.isfinite(n):
        return None
    return (n / d) * scale


def fmt_dollar(v):
    if v is None or pd.isna(v):
        return "N/A"
    try: v = float(v)
    except (TypeError, ValueError): return "N/A"
    av = abs(v); sign = "-" if v < 0 else ""
    if av >= 1_000_000: return f"{sign}${av / 1_000_000:,.1f}B"
    elif av >= 1_000:   return f"{sign}${av / 1_000:,.1f}M"
    elif av >= 1:       return f"{sign}${av:,.0f}K"
    else:               return "$0"


def fmt_val(v, m=None, with_unit=False):
    if v is None or pd.isna(v):
        return "N/A"
    if m and is_dollar_metric(m):
        return fmt_dollar(v)
    try:
        out = f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "N/A"
    if with_unit and m:
        if is_percent_metric(m):
            return f"{out}%"
        if is_multiplier_metric(m):       # FIX #1
            return f"{out}x"
    return out


def fmt_delta(curr, prev, metric=None):
    if curr is None or prev is None or pd.isna(curr) or pd.isna(prev):
        return ("\u2014", CS['neutral'])
    try:
        curr, prev = float(curr), float(prev)
    except (TypeError, ValueError):
        return ("\u2014", CS['neutral'])
    diff = curr - prev
    inverse = is_inverse_metric(metric) if metric else False
    if metric and is_dollar_metric(metric):
        if prev == 0: return ("\u2014", CS['neutral'])
        pct = (diff / abs(prev)) * 100
        display = f"{pct:+.1f}%"
        is_good = (pct > 0) if not inverse else (pct < 0)
    else:
        if metric and is_percent_metric(metric):
            display = f"{diff:+.2f} pp"
        elif metric and is_multiplier_metric(metric):   # FIX #1
            display = f"{diff:+.2f}x"
        else:
            display = f"{diff:+.2f}"
        is_good = (diff > 0) if not inverse else (diff < 0)
    if abs(diff) < 1e-9:
        if metric and is_percent_metric(metric):
            zero_display = "0.00 pp"
        elif metric and is_multiplier_metric(metric):
            zero_display = "0.00x"
        else:
            zero_display = "0.00"
        return (zero_display, CS['neutral'])
    color = CS['good'] if is_good else CS['bad']
    return (display, color)


def calc_trend_change(start_value, end_value, metric=None):
    if start_value is None or end_value is None or pd.isna(start_value) or pd.isna(end_value):
        return np.nan
    try:
        sv, ev = float(start_value), float(end_value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(sv) or not np.isfinite(ev):
        return np.nan
    if metric and is_dollar_metric(metric):
        return ((ev - sv) / abs(sv)) * 100 if sv != 0 else np.nan
    return ev - sv


def fmt_trend_change(value, metric=None):
    if value is None or pd.isna(value):
        return "N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if metric and is_dollar_metric(metric):
        return f"{v:+.2f}%"
    if metric and is_multiplier_metric(metric):   # FIX #1
        return f"{v:+.2f}x"
    return f"{v:+.2f} pp"


def trend_direction_label(slope, eps=1e-9):
    if slope is None or pd.isna(slope):
        return "N/A"
    try:
        sl = float(slope)
    except (TypeError, ValueError):
        return "N/A"
    if abs(sl) <= eps:
        return "\u2192 Flat"
    return "\u2191 Up" if sl > 0 else "\u2193 Down"


def make_sparkline_svg(values, width=90, height=24, color=None, fill_color=None):
    if color is None: color = CS['spark']
    if fill_color is None: fill_color = CS['spark_area']
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if len(clean) < 2:
        return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"><line x1="2" y1="{height/2:.1f}" x2="{width-2:.1f}" y2="{height/2:.1f}" stroke="{CS["lighter"]}" stroke-width="1" stroke-dasharray="2,2"/></svg>'
    vmin, vmax = min(clean), max(clean)
    rng = vmax - vmin if vmax != vmin else max(abs(vmax), 1) * 0.1
    pad = 3
    pts = []
    for i, v in enumerate(clean):
        x = (i / (len(clean) - 1)) * (width - 2) + 1
        y = (height - pad) - ((v - vmin) / rng) * (height - 2 * pad)
        pts.append((x, y))
    line_path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area_path = line_path + f" L{pts[-1][0]:.1f},{height - 0.5} L{pts[0][0]:.1f},{height - 0.5} Z"
    last_x, last_y = pts[-1]
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">'
            f'<path d="{area_path}" fill="{fill_color}" stroke="none"/>'
            f'<path d="{line_path}" fill="none" stroke="{color}" stroke-width="1.4" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="1.8" fill="{color}"/>'
            f'</svg>')


def svg_to_data_url(svg_str):
    encoded = base64.b64encode(svg_str.encode('utf-8')).decode('ascii')
    return f"data:image/svg+xml;base64,{encoded}"


def make_sparkline_img(values, width=90, height=24, color=None, fill_color=None, cls="spark-img"):
    svg = make_sparkline_svg(values, width, height, color, fill_color)
    return html.Img(src=svg_to_data_url(svg), className=cls, style={'display': 'block'})


def make_percentile_arc_svg(pct, size=72):
    has_pct = pct is not None and not pd.isna(pct)
    if has_pct:
        pct = max(0, min(100, float(pct)))
    else:
        pct = 0.0
    if pct >= 75: color = CS['peer_band_top']
    elif pct >= 25: color = CS['peer_band_mid']
    else: color = CS['peer_band_low']
    cx = cy = size / 2; r = size / 2 - 5
    import math
    start_angle = 135; end_angle_full = 45 + 360
    def polar(angle_deg):
        rad = math.radians(angle_deg)
        return (cx + r * math.cos(rad), cy + r * math.sin(rad))
    sx, sy = polar(start_angle)
    ex, ey = polar(end_angle_full)
    large_arc_bg = 1
    bg_path = f"M {sx:.2f} {sy:.2f} A {r} {r} 0 {large_arc_bg} 1 {ex:.2f} {ey:.2f}"
    sweep = (pct / 100) * 270
    val_end_angle = start_angle + sweep
    vex, vey = polar(val_end_angle)
    large_arc_val = 1 if sweep > 180 else 0
    val_path = f"M {sx:.2f} {sy:.2f} A {r} {r} 0 {large_arc_val} 1 {vex:.2f} {vey:.2f}"
    label = f"{pct:.0f}" if has_pct else "\u2014"
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">'
            f'<path d="{bg_path}" fill="none" stroke="{CS["neutral_light"]}" stroke-width="5" stroke-linecap="round"/>'
            f'<path d="{val_path}" fill="none" stroke="{color}" stroke-width="5" stroke-linecap="round"/>'
            f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
            f'font-family="Inter, sans-serif" font-size="{size * 0.32:.1f}" font-weight="700" fill="{CS["text"]}">{label}</text>'
            f'<text x="{cx}" y="{cy + size * 0.22:.1f}" text-anchor="middle" dominant-baseline="central" '
            f'font-family="Inter, sans-serif" font-size="{size * 0.13:.1f}" font-weight="500" fill="{CS["text3"]}">pctl</text>'
            f'</svg>')


def make_percentile_arc_img(pct, size=72, cls="pct-arc"):
    svg = make_percentile_arc_svg(pct, size)
    return html.Img(src=svg_to_data_url(svg), className=cls, style={'display': 'block'})


def compute_period_deltas(df, bank, metric, current_date):
    """Return (qoq_prev_value, yoy_prev_value). YoY uses closest-match within
    PRIOR_PERIOD_TOLERANCE_DAYS so an adjacent quarter can't be mis-matched."""
    bd = df[df['Bank'] == bank].sort_values('Date')
    if bd.empty: return (None, None)
    dates = list(bd['Date'])
    try:
        idx = dates.index(pd.Timestamp(current_date))
    except ValueError:
        return (None, None)
    qoq_val = bd.iloc[idx - 1][metric] if idx >= 1 else None
    target_yoy = pd.Timestamp(current_date) - pd.DateOffset(years=1)
    yoy_val = None
    best_j = None
    best_diff = None
    for j in range(idx):
        d = dates[j]
        diff = abs((d - target_yoy).days)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_j = j
    if best_j is not None and best_diff <= PRIOR_PERIOD_TOLERANCE_DAYS:
        yoy_val = bd.iloc[best_j][metric]
    return (qoq_val, yoy_val)


def compute_peer_rank(df, date, metric, bank):
    slice_ = df[df['Date'] == pd.Timestamp(date)].copy()
    vals = slice_[[metric, 'Bank']].dropna()
    if vals.empty or bank not in vals['Bank'].values:
        return (None, 0, None)
    total = len(vals)
    bank_val = vals.loc[vals['Bank'] == bank, metric].iloc[0]
    other_vals = vals.loc[vals['Bank'] != bank, metric].values
    inverse = is_inverse_metric(metric)
    if len(other_vals) == 0:
        return (1, total, None)
    if inverse:
        better = sum(1 for v in other_vals if v < bank_val)
        worse = sum(1 for v in other_vals if v > bank_val)
    else:
        better = sum(1 for v in other_vals if v > bank_val)
        worse = sum(1 for v in other_vals if v < bank_val)
    ties = len(other_vals) - better - worse
    rank = better + 1
    pct = ((worse + 0.5 * ties) / len(other_vals)) * 100
    return (rank, total, pct)


def get_sparkline_series(df, bank, metric, lookback_quarters=12, end_date=None):
    bd = df[df['Bank'] == bank].sort_values('Date')
    if bd.empty:
        return []
    if end_date is not None:
        bd = bd[bd['Date'] <= pd.Timestamp(end_date)]
    return bd[metric].tail(lookback_quarters).tolist()


class FDICDataUnavailableError(RuntimeError):
    pass


class FDICAPIClient:
    def __init__(self):
        self.base_url = BASE_URL

    def _get(self, ep, params, attempts=3):
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                r = requests.get(
                    f"{self.base_url}/{ep}",
                    params=params,
                    headers={"Accept": "application/json"},
                    verify=False,
                    timeout=45,
                )
                r.raise_for_status()
                payload = r.json()
                if isinstance(payload, dict):
                    return payload
                last_error = f"non-dict JSON payload: {type(payload).__name__}"
            except requests.exceptions.Timeout as e:
                last_error = f"timeout: {e}"
            except requests.exceptions.RequestException as e:
                last_error = f"request error: {e}"
            except (ValueError, KeyError) as e:
                last_error = f"parse error: {e}"
            if attempt < attempts:
                logger.warning(f"FDIC API {ep} attempt {attempt}/{attempts} failed; retrying. Reason: {last_error}")
                time.sleep(min(2 * attempt, 5))
        logger.warning(f"FDIC API failed after {attempts} attempts for {ep}. Last error: {last_error}")
        return {"data": [], "_error": last_error}

    def get_institutions(self, f, fields):
        return self._get("institutions", {"filters": f, "fields": fields, "limit": 10000}).get('data', [])

    def get_financials(self, cert, f, fields):
        flt = f"CERT:{cert}" + (f" AND {f}" if f else "")
        return self._get("financials", {"filters": flt, "fields": fields, "limit": 10000}).get('data', [])


class BankDataRepository:
    FF = ("CERT,REPDTE,ASSET,DEP,BRO,LNLSGR,LNLSNET,SC,ERNAST,RWAJ,"
        "LNRE,LNRECONS,LNRECNFM,LNRECNOT,LNREAG,LNRERES,LNRELOC,LNRERSFM,LNRERSF2,"
        "LNREMULT,LNRENRES,LNRENROW,LNRENROT,LNCOMRE,"
        "LNCI,LNAG,LNCON,LNCRCD,LNAUTO,LNCONOTH,LNOTHER,"
        "LNATRES,NALNLS,OREOTH,P3LNLS,P9LNLS,RBCT1J,CT1BADJ,EQ,EQPP,DRLNLS,DRLNLSQ,"
        "CRLNLS,CRLNLSQ,NTLNLSQ,NETINC,NETINCQ,ERNASTR,NIMY,NTLNLSR,LNATRESR,ROA,ROAQ,"
        "ROE,ROEQ,RBC1AAJ,RBCRWAJ,LNLSDEPR,LNLSNTV,"
        "EEFFR,ELNANTR,IDERNCVR,IDT1CER,IDT1RWAJR,INTEXPYQ,NONIIR,COREDEP,ROAPTX,"
        "NONIXR,DEPNIDOM,LNRESNCR,VOLIABR,NOIJY,ESALR,INTINCR,EINTEXPR,ELNATRR,INTINCY")

    def __init__(self):
        self.api = FDICAPIClient()

    @staticmethod
    def _bank_set_hash(bi):
        certs = sorted(b['cert'] for b in bi)
        payload = CACHE_SCHEMA_VERSION + '|' + ','.join(certs) + '|' + BankDataRepository.FF
        return hashlib.md5(payload.encode()).hexdigest()[:10]

    def _cp(self, s, e):
        h = self._bank_set_hash(BANK_INFO)
        return os.path.join(CACHE_DIR, f"bank_data_{s}_{e}_{h}.json")

    def _lc(self, s, e):
        p = self._cp(s, e)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Cache load failed for {p}: {exc}")
        return None

    def _sc(self, d, s, e):
        # FIX #3: atomic write. Write to a per-PID temp file and os.replace()
        # into place so a reader never sees a half-written cache.
        path = self._cp(s, e)
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, 'w') as f:
                json.dump(d, f)
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning(f"Cache save failed: {exc}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def fetch_data(self, bi, sd, ed):
        expected_certs = {str(b["cert"]).strip() for b in bi}
        expected_count = len(expected_certs)

        c = self._lc(sd, ed)
        if c:
            cached_inst_certs = {
                str(bd['CERT']).strip()
                for bd in c.get('institutions_data', {}).values()
                if isinstance(bd, dict) and bd.get('CERT')
            }
            cached_fin_certs = set()
            for fin_rows in c.get('financials_data', {}).values():
                if not isinstance(fin_rows, list):
                    continue
                for row in fin_rows:
                    if isinstance(row, dict) and row.get('CERT'):
                        cached_fin_certs.add(str(row['CERT']).strip())
            complete_inst = cached_inst_certs >= expected_certs
            complete_fin = cached_fin_certs >= expected_certs
            if complete_inst and complete_fin:
                logger.info(f"Cache hit (pid={os.getpid()}): complete data for all {expected_count} banks.")
                return c
            missing = expected_certs - (cached_inst_certs & cached_fin_certs)
            logger.warning(
                f"Cache present but INCOMPLETE: institutions {len(cached_inst_certs)}/{expected_count}, "
                f"financials {len(cached_fin_certs)}/{expected_count} (missing certs: {sorted(missing)}). "
                f"Discarding and re-fetching. (pid={os.getpid()})")

        inst, fins = {}, {}
        failed = []
        for b in bi:
            cert = b["cert"]
            display = b.get("display", f"CERT {cert}")
            try:
                ii = self.api.get_institutions(f'CERT:{cert}', "NAME,CERT")
                if not ii:
                    failed.append((cert, display, "FDIC API returned no institution record."))
                    continue
                bk = ii[0]
                if not (isinstance(bk, dict) and 'data' in bk):
                    failed.append((cert, display, "FDIC API returned malformed institution response."))
                    continue
                bd = bk['data']
                if 'NAME' not in bd:
                    failed.append((cert, display, "FDIC institution response missing NAME field."))
                    continue
                fdic_name = bd['NAME']
                logger.debug(f"FDIC fetched: cert {cert} -> '{fdic_name}' -> display '{display}'")
                fn = self.api.get_financials(cert, f"REPDTE:[{sd} TO {ed}]", self.FF)
                fin_data = [f['data'] for f in fn if isinstance(f, dict) and 'data' in f]
                if not fin_data:
                    failed.append((cert, display, f"No financial records in date range {sd}-{ed}."))
                    continue
                inst[fdic_name] = bd
                fins[fdic_name] = fin_data
            except (KeyError, TypeError, ValueError) as exc:
                failed.append((cert, display, f"{type(exc).__name__}: {exc}"))

        if not inst:
            failure_summary = '; '.join(f"{d} (cert {c}): {r}" for c, d, r in failed)
            raise FDICDataUnavailableError(
                f"FDIC BankFind API returned no usable data for any of the {len(bi)} requested banks. "
                f"Synthetic fallback has been removed by design. Failures: {failure_summary}")

        result = {'institutions_data': inst, 'financials_data': fins}

        if failed:
            for cert, display, reason in failed:
                logger.warning(f"FDIC fetch failed for {display} (cert {cert}): {reason}")
            failure_summary = '; '.join(f"{d} (cert {c}): {r}" for c, d, r in failed)
            logger.error(f"FDIC fetch INCOMPLETE: {len(failed)} of {len(bi)} banks failed. (pid={os.getpid()})")
            raise FDICDataUnavailableError(
                f"Incomplete FDIC BankFind fetch: {len(failed)} of {len(bi)} configured banks failed "
                f"after retries. Failures: {failure_summary}")

        logger.info(f"FDIC fetch COMPLETE: all {len(bi)} banks. Writing cache. (pid={os.getpid()})")
        self._sc(result, sd, ed)
        return result


class BankMetricsCalculator:
    @staticmethod
    def _sf(v):
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        try:
            f = float(v)
            if not np.isfinite(f):
                return None
            return f
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _z(v):
        return 0.0 if v is None else v

    def calculate_metrics(self, fd):
        rows = []
        pyd = {}
        for bn, fins in fd.items():
            sf = sorted(fins, key=lambda x: x['REPDTE'])
            for i, fin in enumerate(sf):
                cert = str(fin.get('CERT', '')).strip()
                display_name = CERT_TO_DISPLAY.get(cert)
                if display_name is None:
                    display_name = normalize_bank_name(bn)
                row = self._br(display_name, fin)
                cb = self._cb(row, fin)
                self._cc(row, cb)
                self._cg(row, sf, i, fin)
                self._aq(row, sf, i)
                self._gr(row, display_name, fin, pyd)
                self._bk(row)
                self._dp(row)
                self._pp(row)
                self._kf(row, fin)
                rows.append({k: v for k, v in row.items() if not k.startswith('_')})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        return df.sort_values('Date')

    def _br(self, bn, fin):
        s = self._sf
        return {'Bank': bn, 'Date': fin.get('REPDTE'),
            '_ta': s(fin.get('ASSET')), '_td': s(fin.get('DEP')), '_bd': s(fin.get('BRO')),
            '_tl': s(fin.get('LNLSGR')), '_nl': s(fin.get('LNLSNET')),
            '_sc': s(fin.get('SC')), '_ea': s(fin.get('ERNAST')), '_rwa': s(fin.get('RWAJ')),
            '_re': s(fin.get('LNRE')), '_con': s(fin.get('LNRECONS')),
            '_c14': s(fin.get('LNRECNFM')), '_cot': s(fin.get('LNRECNOT')),
            '_fm': s(fin.get('LNREAG')), '_r14': s(fin.get('LNRERES')),
            '_hel': s(fin.get('LNRELOC')), '_f1': s(fin.get('LNRERSFM')), '_jr': s(fin.get('LNRERSF2')),
            '_mul': s(fin.get('LNREMULT')),
            '_nfnr': s(fin.get('LNRENRES')), '_ooc': s(fin.get('LNRENROW')),
            '_noo': s(fin.get('LNRENROT')), '_cre': s(fin.get('LNCOMRE')),
            '_ci': s(fin.get('LNCI')), '_ag': s(fin.get('LNAG')),
            '_csm': s(fin.get('LNCON')), '_crd': s(fin.get('LNCRCD')), '_aut': s(fin.get('LNAUTO')),
            '_ndf': s(fin.get('LNOTHER')),
            '_acl': s(fin.get('LNATRES')), '_na': s(fin.get('NALNLS')), '_oreo': s(fin.get('OREOTH')),
            '_p30': s(fin.get('P3LNLS')), '_p90': s(fin.get('P9LNLS')),
            '_t1': s(fin.get('RBCT1J')), '_ncoq': s(fin.get('NTLNLSQ')),
            '_ct1b': s(fin.get('CT1BADJ')), '_eq': s(fin.get('EQ')), '_eqpp': s(fin.get('EQPP')),
            '_cd': s(fin.get('COREDEP')), '_nib': s(fin.get('DEPNIDOM')),
            '_ni': s(fin.get('NETINC')), '_niq': s(fin.get('NETINCQ')),
            '_gco_ytd': s(fin.get('DRLNLS')), '_gcoq': s(fin.get('DRLNLSQ')),
            '_rec_ytd': s(fin.get('CRLNLS')), '_recq': s(fin.get('CRLNLSQ')),
            'Net Charge-Offs / Total Loans & Leases': s(fin.get('NTLNLSR')),
            'ACL / Total Loans & Leases': s(fin.get('LNATRESR')),
            'Earnings Coverage of Net Loan Charge-Offs': s(fin.get('IDERNCVR')),
            'Loan and Lease Loss Provision to Net Charge-Offs': s(fin.get('ELNANTR')),
            'Net Loans and Leases to Deposits': s(fin.get('LNLSDEPR')),
            'Net Loans and Leases to Assets': s(fin.get('LNLSNTV')),
            'Return on Assets': s(fin.get('ROA')),
            'Quarterly Return on Assets': s(fin.get('ROAQ')),
            'Return on Equity': s(fin.get('ROE')),
            'Quarterly Return on Equity': s(fin.get('ROEQ')),
            'Leverage (Core Capital) Ratio': s(fin.get('RBC1AAJ')),
            'Total Risk-Based Capital Ratio': s(fin.get('RBCRWAJ')),
            'Efficiency Ratio': s(fin.get('EEFFR')), 'Earning Assets / Total Assets': s(fin.get('ERNASTR')),
            'Net Interest Margin': s(fin.get('NIMY')),
            'Common Equity Tier 1 (CET1) Ratio': s(fin.get('IDT1CER')),
            'Tier 1 Risk-Based Capital Ratio': s(fin.get('IDT1RWAJR')),
            'Cost of Funding Earning Assets': s(fin.get('INTEXPYQ')),
            'Noninterest Income to Average Assets': s(fin.get('NONIIR')),
            'Pretax Return on Assets': s(fin.get('ROAPTX')),
            'Noninterest Expense to Average Assets': s(fin.get('NONIXR')),
            'Loan Loss Reserve / Noncurrent Loans': s(fin.get('LNRESNCR')),
            'Volatile Liabilities to Total Assets': s(fin.get('VOLIABR')),
            'Net Operating Income to Assets': s(fin.get('NOIJY')),
            'Salaries and Benefits to Average Assets': s(fin.get('ESALR')),
            'Interest Income to Average Assets': s(fin.get('INTINCR')),
            'Interest Expense to Average Assets': s(fin.get('EINTEXPR')),
            'Provision for Credit Losses to Average Assets': s(fin.get('ELNATRR')),
            'Yield on Earning Assets': s(fin.get('INTINCY'))}

    def _kf(self, r, fin):
        r['Total Assets'] = r['_ta']
        r['Total Deposits'] = r['_td']
        r['Gross Loans & Leases'] = r['_tl']
        r['Net Loans & Leases'] = r['_nl']
        r['Total Securities'] = r['_sc']
        r['Total Earning Assets'] = r['_ea']
        r['Total Equity Capital'] = r['_eq']
        r['Tier 1 Capital'] = r['_t1']
        r['Risk-Weighted Assets'] = r['_rwa']
        r['Net Income (YTD)'] = r['_ni']
        r['Net Income (Quarter)'] = r['_niq']
        r['Allowance for Credit Losses'] = r['_acl']
        r['Gross Charge-Offs (YTD)'] = r['_gco_ytd']
        r['Gross Charge-Offs (Quarter)'] = r['_gcoq']
        r['Gross Recoveries (YTD)'] = r['_rec_ytd']
        r['Gross Recoveries (Quarter)'] = r['_recq']
        if r['_na'] is not None or r['_p90'] is not None:
            r['Noncurrent Loans'] = self._z(r['_na']) + self._z(r['_p90'])
        else:
            r['Noncurrent Loans'] = None
        r['Brokered Deposits'] = r['_bd']

    def _cb(self, r, fin):
        t1 = r.get('_t1')
        acl = r.get('_acl')
        if t1 is None or acl is None:
            return None
        b = t1 + acl
        return b if b > 0 else None

    def _cc(self, r, cb):
        conc_metrics = [
            'Real Estate Loans to Tier 1 + ACL', 'RE Construction and Land Development to Tier 1 + ACL',
            '1-4 Family Construction to Tier 1 + ACL', 'Other Construction & Land Dev to Tier 1 + ACL',
            'Secured by Farmland to Tier 1 + ACL', '1-4 Family Residential to Tier 1 + ACL',
            'Revolving Home Equity to Tier 1 + ACL', 'Closed-End 1st Lien to Tier 1 + ACL',
            'Closed-End Jr Lien to Tier 1 + ACL', 'Multifamily RE to Tier 1 + ACL',
            'Non-Farm Non-Residential RE to Tier 1 + ACL', 'NFNR: Owner Occupied to Tier 1 + ACL',
            'NFNR: Non-Owner Occupied to Tier 1 + ACL', 'Commercial RE to Tier 1 + ACL',
            'Non-Owner Occupied CRE to Tier 1 + ACL',
            'C&I Loans to Tier 1 + ACL', 'Loans to Individuals to Tier 1 + ACL',
            'Credit Cards to Tier 1 + ACL', 'Auto Loans to Tier 1 + ACL',
            'Agriculture Loans to Tier 1 + ACL', 'Loans to NDFIs and Other to Tier 1 + ACL']
        if cb is None or cb <= 0:
            for m in conc_metrics:
                r[m] = None
            return
        r['Real Estate Loans to Tier 1 + ACL'] = safe_div(r.get('_re'), cb)
        r['RE Construction and Land Development to Tier 1 + ACL'] = safe_div(r.get('_con'), cb)
        r['1-4 Family Construction to Tier 1 + ACL'] = safe_div(r.get('_c14'), cb)
        r['Other Construction & Land Dev to Tier 1 + ACL'] = safe_div(r.get('_cot'), cb)
        r['Secured by Farmland to Tier 1 + ACL'] = safe_div(r.get('_fm'), cb)
        r['1-4 Family Residential to Tier 1 + ACL'] = safe_div(r.get('_r14'), cb)
        r['Revolving Home Equity to Tier 1 + ACL'] = safe_div(r.get('_hel'), cb)
        r['Closed-End 1st Lien to Tier 1 + ACL'] = safe_div(r.get('_f1'), cb)
        r['Closed-End Jr Lien to Tier 1 + ACL'] = safe_div(r.get('_jr'), cb)
        r['Multifamily RE to Tier 1 + ACL'] = safe_div(r.get('_mul'), cb)
        r['Non-Farm Non-Residential RE to Tier 1 + ACL'] = safe_div(r.get('_nfnr'), cb)
        r['NFNR: Owner Occupied to Tier 1 + ACL'] = safe_div(r.get('_ooc'), cb)
        r['NFNR: Non-Owner Occupied to Tier 1 + ACL'] = safe_div(r.get('_noo'), cb)
        cre_sum = self._z(r.get('_con')) + self._z(r.get('_mul')) + self._z(r.get('_nfnr')) + self._z(r.get('_cre'))
        r['Commercial RE to Tier 1 + ACL'] = safe_div(cre_sum, cb)
        noo_cre_sum = self._z(r.get('_con')) + self._z(r.get('_mul')) + self._z(r.get('_noo')) + self._z(r.get('_cre'))
        r['Non-Owner Occupied CRE to Tier 1 + ACL'] = safe_div(noo_cre_sum, cb)
        r['C&I Loans to Tier 1 + ACL'] = safe_div(r.get('_ci'), cb)
        r['Loans to Individuals to Tier 1 + ACL'] = safe_div(r.get('_csm'), cb)
        r['Credit Cards to Tier 1 + ACL'] = safe_div(r.get('_crd'), cb)
        r['Auto Loans to Tier 1 + ACL'] = safe_div(r.get('_aut'), cb)
        r['Agriculture Loans to Tier 1 + ACL'] = safe_div(r.get('_ag'), cb)
        r['Loans to NDFIs and Other to Tier 1 + ACL'] = safe_div(r.get('_ndf'), cb)

    def _cg(self, r, sf, idx, cur):
        s = self._sf
        z = self._z
        now = z(s(cur.get('LNRECONS'))) + z(s(cur.get('LNREMULT'))) + \
              z(s(cur.get('LNRENROT'))) + z(s(cur.get('LNCOMRE')))
        try:
            cur_dt = pd.to_datetime(cur.get('REPDTE'), format='%Y%m%d')
        except (ValueError, TypeError):
            r['Non-Owner Occupied CRE 3-Year Growth Rate'] = None
            return
        target = cur_dt - pd.DateOffset(years=3)
        prior_record = None
        best_diff = None
        for j in range(idx):
            try:
                d = pd.to_datetime(sf[j].get('REPDTE'), format='%Y%m%d')
            except (ValueError, TypeError):
                continue
            diff = abs((d - target).days)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                prior_record = sf[j]
        if prior_record is None or best_diff is None or best_diff > PRIOR_PERIOD_TOLERANCE_DAYS:
            r['Non-Owner Occupied CRE 3-Year Growth Rate'] = None
            return
        old = z(s(prior_record.get('LNRECONS'))) + z(s(prior_record.get('LNREMULT'))) + \
              z(s(prior_record.get('LNRENROT'))) + z(s(prior_record.get('LNCOMRE')))
        if old > 0:
            r['Non-Owner Occupied CRE 3-Year Growth Rate'] = ((now / old) - 1) * 100
        else:
            r['Non-Owner Occupied CRE 3-Year Growth Rate'] = None

    def _aq(self, r, sf, idx):
        l = r.get('_tl'); acl = r.get('_acl'); na = r.get('_na')
        oreo = r.get('_oreo'); p30 = r.get('_p30'); p90 = r.get('_p90')
        # FIX #1: UBPR "X multiplier" coverage ratios -> scale=1.0 (a multiple).
        r['ACL / Nonaccrual Loans'] = safe_div(acl, na, scale=1.0)
        if na is None and p90 is None:
            r['ACL / 90+ DPD & Nonaccrual'] = None
        else:
            r['ACL / 90+ DPD & Nonaccrual'] = safe_div(acl, self._z(na) + self._z(p90), scale=1.0)
        if na is None and oreo is None and p90 is None:
            r['Nonaccrual & OREO / Total Loans & OREO'] = None
        else:
            num = self._z(na) + self._z(oreo) + self._z(p90)
            denom = self._z(l) + self._z(oreo)
            r['Nonaccrual & OREO / Total Loans & OREO'] = safe_div(num, denom)
        r['30-89 DPD / Total Loans'] = safe_div(p30, l)
        r['90+ DPD / Total Loans'] = safe_div(p90, l)
        r['Nonaccrual / Total Loans'] = safe_div(na, l)
        if na is None and p90 is None:
            r['90+ DPD & Nonaccrual / Total Loans'] = None
        else:
            r['90+ DPD & Nonaccrual / Total Loans'] = safe_div(self._z(na) + self._z(p90), l)
        # FIX #2: only sum the rolling 4-quarter NCO window when the four
        # records are genuinely contiguous quarters (gaps ~90-92 days). A
        # skipped filing would otherwise silently span 5+ quarters.
        r['Net Charge-Offs / ACL'] = None
        if idx >= 3:
            window = sf[idx - 3:idx + 1]
            window_dates = []
            parse_ok = True
            for w in window:
                try:
                    window_dates.append(pd.to_datetime(w.get('REPDTE'), format='%Y%m%d'))
                except (ValueError, TypeError):
                    parse_ok = False
                    break
            contiguous = False
            if parse_ok and len(window_dates) == 4:
                gaps = [(window_dates[k + 1] - window_dates[k]).days
                        for k in range(len(window_dates) - 1)]
                contiguous = all(75 <= g <= 100 for g in gaps)
            window_vals = [self._sf(w.get('NTLNLSQ')) for w in window]
            if contiguous and all(v is not None for v in window_vals):
                r4 = sum(window_vals)
                r['Net Charge-Offs / ACL'] = safe_div(r4, acl)

    def _gr(self, r, bn, fin, pyd):
        try:
            dt = pd.to_datetime(fin.get('REPDTE'), format='%Y%m%d')
        except (ValueError, TypeError):
            r['Net Loan Growth Rate'] = None
            r['Total Asset Growth Rate'] = None
            r['Tier 1 Capital Growth Rate'] = None
            return
        q, yr = dt.quarter, dt.year
        for sfx, val, m in [('nl', r.get('_nl'), 'Net Loan Growth Rate'),
                            ('ta', r.get('_ta'), 'Total Asset Growth Rate'),
                            ('t1', r.get('_t1'), 'Tier 1 Capital Growth Rate')]:
            k = f"{bn}_{q}_{sfx}"
            if val is not None:
                pyd.setdefault(k, {})[yr] = val
            pv = pyd.get(k, {}).get(yr - 1)
            if pv is not None and pv > 0 and val is not None:
                r[m] = ((val / pv) - 1) * 100
            else:
                r[m] = None

    def _bk(self, r):
        r['Brokered Deposits to Total Deposits'] = safe_div(r.get('_bd'), r.get('_td'))

    def _dp(self, r):
        r['Core Deposits to Total Deposits'] = safe_div(r.get('_cd'), r.get('_td'))
        r['Noninterest-Bearing Deposits to Total Deposits'] = safe_div(r.get('_nib'), r.get('_td'))

    def _pp(self, r):
        ii = r.get('Interest Income to Average Assets')
        ie = r.get('Interest Expense to Average Assets')
        ni = r.get('Noninterest Income to Average Assets')
        nx = r.get('Noninterest Expense to Average Assets')
        if any(v is None or pd.isna(v) for v in (ii, ie, ni, nx)):
            r['Pre-Provision Net Revenue to Average Assets'] = None
        else:
            r['Pre-Provision Net Revenue to Average Assets'] = ii - ie + ni - nx


class BankDataService:
    def __init__(self):
        self.repo = BankDataRepository()
        self.calc = BankMetricsCalculator()

    def get_metrics_data(self, sd=DEFAULT_START_DATE, ed=DEFAULT_END_DATE):
        d = self.repo.fetch_data(BANK_INFO, sd, ed)
        if not d['institutions_data']:
            return pd.DataFrame()
        df = self.calc.calculate_metrics(d['financials_data'])
        if df.empty:
            return df
        df['Bank'] = df['Bank'].apply(normalize_bank_name)
        return df[['Bank', 'Date'] + [m for m in METRIC_ORDER if m in df.columns]]


def build_primary_bank_export(df):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    ghb = df[df['Bank'] == PRIMARY_BANK_DISPLAY_NAME].copy()
    if ghb.empty:
        return None
    ghb = ghb.sort_values('Date')
    metrics = [m for m in METRIC_ORDER if m in ghb.columns]
    dates = ghb['Date'].sort_values().unique()
    wb = Workbook()
    ws = wb.active
    ws.title = "JPM Metrics"

    hdr_fill = PatternFill('solid', fgColor='005EB8')
    cat_fills = {
        "Key Financials": PatternFill('solid', fgColor='E2E8F0'),
        "Earnings & Profitability": PatternFill('solid', fgColor='E8F0EB'),
        "Efficiency & Margin": PatternFill('solid', fgColor='E0ECF4'),
        "Capitalization": PatternFill('solid', fgColor='FFF8E1'),
        "Asset Quality": PatternFill('solid', fgColor='FBE9E7'),
        "Loan & Lease Analysis": PatternFill('solid', fgColor='F3E5F5'),
        "Funding & Liquidity": PatternFill('solid', fgColor='E0F2F1'),
        "Credit Concentration": PatternFill('solid', fgColor='F5F5F5'),
        "Growth": PatternFill('solid', fgColor='E8EAF6')}
    white_fill = PatternFill('solid', fgColor='FFFFFF')
    alt_fill = PatternFill('solid', fgColor='F8FAFB')
    thin_border = Border(bottom=Side(style='hair', color='D0D0D0'),
                         right=Side(style='hair', color='D0D0D0'))
    hdr_font = Font(name='Arial', bold=True, color='FFFFFF', size=9)
    cat_font = Font(name='Arial', bold=True, color='005EB8', size=8)
    metric_font = Font(name='Arial', bold=True, color='333333', size=8)
    date_font = Font(name='Arial', bold=False, color='1A1A2E', size=9)
    val_font = Font(name='Arial', color='1A1A2E', size=9)
    center = Alignment(horizontal='center', vertical='center', wrap_text=True)
    left = Alignment(horizontal='left', vertical='center')

    ws.cell(row=1, column=1, value=f"{PRIMARY_BANK_FDIC_NAME} · since {REQUESTED_START_DATE_DISPLAY}").font = hdr_font
    ws.cell(row=1, column=1).fill = hdr_fill
    ws.cell(row=1, column=1).alignment = left
    col = 2
    for cat_name, cat_metrics in METRIC_CATEGORIES:
        present = [m for m in cat_metrics if m in metrics]
        if not present:
            continue
        start_col = col
        for m in present:
            ws.cell(row=1, column=col).fill = cat_fills.get(cat_name, white_fill)
            col += 1
        end_col = col - 1
        ws.merge_cells(start_row=1, start_column=start_col, end_row=1, end_column=end_col)
        merged = ws.cell(row=1, column=start_col)
        merged.value = cat_name
        merged.font = cat_font
        merged.alignment = center
        merged.fill = cat_fills.get(cat_name, white_fill)

    ws.cell(row=2, column=1, value="Report Date").font = metric_font
    ws.cell(row=2, column=1).fill = PatternFill('solid', fgColor='E8F0EB')
    ws.cell(row=2, column=1).alignment = left
    col = 2
    for m in metrics:
        cell = ws.cell(row=2, column=col, value=m)
        cell.font = metric_font
        cell.alignment = Alignment(horizontal='center', vertical='bottom', wrap_text=True)
        cell.fill = cat_fills.get(METRIC_TO_CATEGORY.get(m, ''), white_fill)
        col += 1

    for ri, dt in enumerate(dates):
        row_num = ri + 3
        row_data = ghb[ghb['Date'] == dt]
        if row_data.empty:
            continue
        row_data = row_data.iloc[0]
        fill = white_fill if ri % 2 == 0 else alt_fill
        dc = ws.cell(row=row_num, column=1, value=pd.Timestamp(dt).strftime('%m/%d/%Y'))
        dc.font = date_font
        dc.alignment = left
        dc.fill = fill
        dc.border = thin_border
        col = 2
        for m in metrics:
            val = row_data.get(m)
            cell = ws.cell(row=row_num, column=col)
            if pd.notna(val):
                cell.value = round(float(val), 4) if not is_dollar_metric(m) else round(float(val), 0)
                cell.number_format = '#,##0' if is_dollar_metric(m) else '0.00'
            else:
                cell.value = None
            cell.font = val_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.fill = fill
            cell.border = thin_border
            col += 1

    ws.column_dimensions['A'].width = 14
    for ci in range(2, len(metrics) + 2):
        ws.column_dimensions[get_column_letter(ci)].width = 14
    ws.row_dimensions[1].height = 22
    ws.row_dimensions[2].height = 56
    ws.freeze_panes = 'B3'
    ws.auto_filter.ref = f"A2:{get_column_letter(len(metrics) + 1)}{len(dates) + 2}"

    ws2 = wb.create_sheet("Metric Definitions")
    ws2.cell(row=1, column=1, value="Category").font = Font(name='Arial', bold=True, size=9)
    ws2.cell(row=1, column=2, value="Metric").font = Font(name='Arial', bold=True, size=9)
    ws2.cell(row=1, column=3, value="Definition").font = Font(name='Arial', bold=True, size=9)
    for c in range(1, 4):
        ws2.cell(row=1, column=c).fill = hdr_fill
        ws2.cell(row=1, column=c).font = hdr_font
    r = 2
    for m in metrics:
        cat = METRIC_TO_CATEGORY.get(m, '')
        defn = METRIC_DEFINITIONS.get(m, '')
        parts = defn.split(' \xb7 ', 1)
        txt = parts[1] if len(parts) == 2 else defn
        ws2.cell(row=r, column=1, value=cat).font = Font(name='Arial', size=9, color='005EB8', bold=True)
        ws2.cell(row=r, column=2, value=m).font = Font(name='Arial', size=9, bold=True)
        ws2.cell(row=r, column=3, value=txt).font = Font(name='Arial', size=9)
        ws2.cell(row=r, column=3).alignment = Alignment(wrap_text=True)
        r += 1
    ws2.column_dimensions['A'].width = 26
    ws2.column_dimensions['B'].width = 42
    ws2.column_dimensions['C'].width = 90
    ws2.freeze_panes = 'A2'

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


class DashboardBuilder:
    GHB = PRIMARY_BANK_DISPLAY_NAME
    SPARK_LOOKBACK = 12

    def __init__(self, df, missing_banks=None):
        self.df = df
        self.missing_banks = sorted(missing_banks or [])
        self.raw_dates = sorted(df['Date'].unique())
        self.loaded_banks = sorted(set(df['Bank'].unique()))

        # Historical-range design note:
        # The dashboard now hard-starts at 12/31/2008, which was the first
        # all-loaded-bank common reporting period in the prior longer-history
        # build. That avoids showing early JPMorgan-only history while peer
        # coverage is incomplete. Date selectors use JPMorgan's available dates
        # from this hard-start; peer stats still automatically exclude any bank
        # without real data for a selected period/window, which protects against
        # future mergers, missing filings, or temporary FDIC lag.
        primary_df = df[df['Bank'] == self.GHB].sort_values('Date').reset_index(drop=True)
        self._ghb_df = primary_df
        self.primary_dates = sorted(primary_df['Date'].unique())
        date_sets = [set(df.loc[df['Bank'] == bank, 'Date']) for bank in self.loaded_banks]
        self.common_dates = sorted(set.intersection(*date_sets)) if date_sets else []
        self.dates = self.primary_dates if self.primary_dates else self.raw_dates
        self.analysis_start_date = self.dates[0] if self.dates else None
        self.analysis_end_date = self.dates[-1] if self.dates else None
        self.raw_latest_date = self.raw_dates[-1] if self.raw_dates else None
        self.common_start_date = self.common_dates[0] if self.common_dates else None
        self.common_latest_date = self.common_dates[-1] if self.common_dates else None

        self.metrics = [m for m in METRIC_ORDER if m in df.columns]
        self.peers = sorted(set(df['Bank'].unique()) - {self.GHB})
        self._mo = [{'label': m, 'value': m} for m in self.metrics]
        self._do = [{'label': d.strftime('%m/%d/%Y'), 'value': d.strftime('%Y-%m-%d')}
                    for d in reversed(self.dates)]
        self._to = (
            [{'label': f'{y} Yr', 'value': y} for y in [1, 2, 3, 4, 5, 7, 10, 15, 20, 25]]
            + [{'label': 'Full History', 'value': 'FULL'}]
        )
        self._def_metric = 'Return on Assets' if 'Return on Assets' in self.metrics else self.metrics[0]
        self._def_r3_primary = 'Net Interest Margin' if 'Net Interest Margin' in self.metrics else self.metrics[0]
        self._def_r3_secondary = 'Cost of Funding Earning Assets' if 'Cost of Funding Earning Assets' in self.metrics else (self.metrics[1] if len(self.metrics) > 1 else self.metrics[0])
        if not self._ghb_df.empty:
            self._ghb_date_index = {pd.Timestamp(d): i for i, d in enumerate(self._ghb_df['Date'])}
        else:
            self._ghb_date_index = {}

    def _ghb_idx(self, date):
        return self._ghb_date_index.get(pd.Timestamp(date))

    def _ghb_value(self, metric, date):
        idx = self._ghb_idx(date)
        if idx is None or self._ghb_df.empty or metric not in self._ghb_df.columns:
            return None
        v = self._ghb_df.iloc[idx][metric]
        return None if pd.isna(v) else v

    def _ghb_qoq_yoy(self, metric, date):
        idx = self._ghb_idx(date)
        if idx is None or self._ghb_df.empty or metric not in self._ghb_df.columns:
            return (None, None)
        qoq_val = self._ghb_df.iloc[idx - 1][metric] if idx >= 1 else None
        target_yoy = pd.Timestamp(date) - pd.DateOffset(years=1)
        yoy_val = None
        best_j = None
        best_diff = None
        for j in range(idx):
            d = self._ghb_df.iloc[j]['Date']
            diff = abs((d - target_yoy).days)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_j = j
        if best_j is not None and best_diff <= PRIOR_PERIOD_TOLERANCE_DAYS:
            yoy_val = self._ghb_df.iloc[best_j][metric]
        if qoq_val is not None and pd.isna(qoq_val):
            qoq_val = None
        if yoy_val is not None and pd.isna(yoy_val):
            yoy_val = None
        return (qoq_val, yoy_val)

    def _ghb_spark(self, metric, end_date, lookback=None):
        if lookback is None:
            lookback = self.SPARK_LOOKBACK
        idx = self._ghb_idx(end_date)
        if idx is None or self._ghb_df.empty or metric not in self._ghb_df.columns:
            return []
        start = max(0, idx - lookback + 1)
        return self._ghb_df.iloc[start:idx + 1][metric].tolist()

    def create_dashboard(self):
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
        app.title = DASHBOARD_TITLE
        app.config.suppress_callback_exceptions = True
        app.index_string = self._css()
        app.layout = self._layout()
        self._cbs(app)
        return app

    def _mdd(self, id_, v=None, c="idd-m"):
        return dcc.Dropdown(id=id_, options=self._mo, value=v or self._def_metric,
                            clearable=False, optionHeight=48, className=c,
                            placeholder="Search metrics...")

    def _window_bounds(self, y, end=None, bank_filter=None):
        """Return (start, end) for trend/correlation windows.

        y may be an integer year lookback or 'FULL'. For full history, the
        lower bound is the earliest real reporting date for the selected banks;
        if no bank filter is supplied, it uses the JPMorgan analysis history.
        """
        if end is None:
            end = self.analysis_end_date if self.analysis_end_date is not None else self.df['Date'].max()
        end_ts = pd.Timestamp(end)
        if str(y).upper() == 'FULL':
            if bank_filter:
                subset = self.df[self.df['Bank'].isin(bank_filter)]
                if not subset.empty:
                    return pd.Timestamp(subset['Date'].min()), end_ts
            if self.analysis_start_date is not None:
                return pd.Timestamp(self.analysis_start_date), end_ts
            return pd.Timestamp(self.df['Date'].min()), end_ts
        try:
            years = int(y)
        except (TypeError, ValueError):
            years = 4
        return end_ts - pd.DateOffset(years=years), end_ts

    @staticmethod
    def _window_label(y, start, end):
        return f"{pd.Timestamp(start).strftime('%m/%Y')}–{pd.Timestamp(end).strftime('%m/%Y')}"

    @staticmethod
    def _axis_for_window(start, end):
        span_years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 0)
        if span_years <= 2:
            return 'M3', '%b %Y'
        if span_years <= 4:
            return 'M6', '%b %Y'
        if span_years <= 10:
            return 'M12', '%Y'
        if span_years <= 18:
            return 'M24', '%Y'
        return 'M36', '%Y'

    def _tdd(self, id_):
        return dcc.Dropdown(id=id_, options=self._to, value=4, clearable=False,
                            searchable=False, className="idd-t")

    def _dfoot(self, id_):
        return html.Div(id=id_, className="dfoot")

    def _exec_banner(self, selected_peers=None):
        latest_date = self.analysis_end_date
        if latest_date is None:
            return html.Div("No JPMorgan reporting date available", className="exec-banner")
        peer_selection = self.peers if selected_peers is None else selected_peers
        cohort = [self.GHB] + list(peer_selection)
        cohort_df = self.df[self.df['Bank'].isin(cohort)]
        cards = []
        for metric, label in EXECUTIVE_KPIS:
            if metric not in self.metrics:
                continue
            curr = self._ghb_value(metric, latest_date)
            qoq_prev, yoy_prev = self._ghb_qoq_yoy(metric, latest_date)
            qoq_text, qoq_color = fmt_delta(curr, qoq_prev, metric)
            spark_vals = self._ghb_spark(metric, latest_date, self.SPARK_LOOKBACK)
            rank, total, pctl = compute_peer_rank(cohort_df, latest_date, metric, self.GHB)
            has_peer_context = total > 1
            rank_color = CS['text3'] if has_peer_context and rank else CS['light']
            rank_text = f"#{rank}/{total}" if has_peer_context and rank else "\u2014"
            val_display = fmt_val(curr, metric, with_unit=True)
            card = html.Div([
                html.Div([
                    html.Span(label, className="exec-label"),
                    html.Span(rank_text, className="exec-rank",
                              style={'color': rank_color, 'borderColor': rank_color}),
                ], className="exec-hdr"),
                html.Div(val_display, className="exec-val"),
                html.Div(metric, className="exec-metric-name"),
                html.Div([make_sparkline_img(spark_vals, width=110, height=26)], className="exec-spark"),
                html.Div([
                    html.Span("QoQ", className="exec-delta-label"),
                    html.Span(qoq_text, className="exec-delta-val", style={'color': qoq_color}),
                ], className="exec-delta"),
            ], className="exec-card")
            cards.append(card)
        latest_peer_rows = cohort_df[(cohort_df['Date'] == pd.Timestamp(latest_date)) & (cohort_df['Bank'] != self.GHB)]
        available_peers = latest_peer_rows['Bank'].nunique()
        peer_note = f"JPM benchmark vs available selected SIB peers: {available_peers}/{len(peer_selection)}"
        return html.Div([
            html.Div([
                html.Div([
                    html.Span("JPMorgan Executive Snapshot", className="exec-banner-title"),
                    html.Span(f"Latest reported period \u00b7 {pd.Timestamp(latest_date).strftime('%b %d, %Y')} \u00b7 {peer_note}",
                              className="exec-banner-date"),
                ], className="exec-banner-hdr"),
                html.Div(cards, className="exec-grid"),
            ], className="exec-banner-inner"),
        ], className="exec-banner")

    def _missing_data_banner(self):
        messages = [
            f"Dashboard analysis starts with the {REQUESTED_START_DATE_DISPLAY} report period, "
            "which keeps the selected SIB peer universe on a cleaner common-history baseline after removing legacy Comerica. "
            "Peer averages, ranks, and percentiles automatically use only peers with real FDIC data for the selected date/window."
        ]
        if self.common_start_date is not None and self.analysis_start_date is not None:
            if pd.Timestamp(self.common_start_date) > pd.Timestamp(self.analysis_start_date):
                messages.append(
                    "The first all-loaded-bank common reporting period is "
                    f"{pd.Timestamp(self.common_start_date).strftime('%m/%d/%Y')}; before that, peer coverage varies by charter/history.")
        if self.missing_banks:
            missing = ', '.join(self.missing_banks)
            messages.append(
                f"Missing banks: {missing}. Peer averages, ranks, percentiles, "
                "and selected-peer counts exclude those banks until a complete "
                "FDIC fetch succeeds.")
        if (self.raw_latest_date is not None and self.analysis_end_date is not None
                and pd.Timestamp(self.raw_latest_date) > pd.Timestamp(self.analysis_end_date)):
            messages.append(
                "JPMorgan's latest available report period is "
                f"{pd.Timestamp(self.analysis_end_date).strftime('%m/%d/%Y')}; "
                f"the raw FDIC peer set includes a newer period of {pd.Timestamp(self.raw_latest_date).strftime('%m/%d/%Y')}.")
        return html.Div([html.Strong("FDIC data scope note. "), html.Span(" ".join(messages))],
                        className="warn-banner")

    def _layout(self):
        dv = self._do[0]['value'] if self._do else None
        chart_style = {'height': f'{PAIRED_GRAPH_HEIGHT}px'}
        return html.Div([
            html.Div([
                html.Div([
                    html.Div(PRIMARY_BANK_ABBR, className="hdr-mark"),
                    html.Div([
                        html.Span(PRIMARY_BANK_DISPLAY_NAME, className="hdr-title"),
                        html.Span(DASHBOARD_SHORT_TITLE + " · " + PEER_UNIVERSE_LABEL, className="hdr-sub")
                    ])
                ], className="hdr-brand"),
                html.Div([
                    html.Span(f"FDIC API · SIB peer set · since {REQUESTED_START_DATE_DISPLAY}", className="hdr-src"),
                    html.Span(f"\u00b7 {len(METRIC_ORDER)} metrics", className="hdr-cnt"),
                    html.Span(f"\u00b7 {len(BANK_INFO)} banks", className="hdr-cnt")
                ])
            ], className="hdr"),
            html.Div([
                html.Div(self._exec_banner(self.peers), id='exec-banner-wrap'),
                self._missing_data_banner(),
                html.Div([
                    html.Span("SIB Peers", className="tb-label"),
                    dcc.Dropdown(id='peer-sel',
                                 options=[{'label': p, 'value': p} for p in self.peers],
                                 value=self.peers, multi=True, className="tb-dd"),
                    html.Button("All", id="sel-all", className="tb-btn"),
                    html.Button("Clear", id="sel-clear", className="tb-btn tb-btn-secondary")
                ], className="toolbar"),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div([
                            html.H6("Peer Snapshot", className="ct"),
                            self._mdd('r1m'),
                            dcc.Dropdown(id='r1d', options=self._do, value=dv, clearable=False,
                                         searchable=False, className="idd-d")
                        ], className="ch ch-wrap"),
                        html.Div([
                            dcc.Loading(dcc.Graph(id='r1c', config={'displayModeBar': False},
                                                  style=chart_style, className="viz-graph"),
                                        type="dot", color=CS['primary'])
                        ], className="viz-shell"),
                        self._dfoot('r1f')
                    ], className="card pair-card pair-card-chart"), md=7, className="mb-3 pair-col"),
                    dbc.Col(html.Div([
                        html.Div([html.H6("Peer Snapshot Stats", className="ct")], className="ch"),
                        html.Div([
                            dcc.Loading(html.Div(id='r1o', className="insight-shell overview-shell"),
                                        type="dot", color=CS['primary'])
                        ], className="insight-load-shell")
                    ], className="card pair-card pair-card-side"), md=5, className="mb-3 pair-col")
                ], className="paired-row"),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div([
                            html.H6("Peer Trend", className="ct"),
                            self._mdd('r2m'), self._tdd('r2t'),
                            html.Span(id='r2r', className="rng")
                        ], className="ch ch-wrap"),
                        html.Div([
                            dcc.Loading(dcc.Graph(id='r2c', config={'displayModeBar': False},
                                                  style=chart_style, className="viz-graph"),
                                        type="dot", color=CS['primary'])
                        ], className="viz-shell"),
                        self._dfoot('r2f')
                    ], className="card pair-card pair-card-chart"), md=7, className="mb-3 pair-col"),
                    dbc.Col(html.Div([
                        html.Div([html.H6("Peer Trend Stats", className="ct")], className="ch"),
                        html.Div([
                            dcc.Loading(html.Div(id='r2a', className="insight-shell analysis-shell"),
                                        type="dot", color=CS['primary'])
                        ], className="insight-load-shell")
                    ], className="card pair-card pair-card-side"), md=5, className="mb-3 pair-col")
                ], className="paired-row"),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div([
                            html.H6("JPM Metric Correlation", className="ct"),
                            self._mdd('r3p', self._def_r3_primary, "idd-m idd-m2"),
                            html.Span("vs", className="vs"),
                            self._mdd('r3s', self._def_r3_secondary, "idd-m idd-m2"),
                            self._tdd('r3t')
                        ], className="ch ch-wrap"),
                        html.Div([
                            dcc.Loading(dcc.Graph(id='r3c', config={'displayModeBar': False},
                                                  style=chart_style, className="viz-graph"),
                                        type="dot", color=CS['primary'])
                        ], className="viz-shell"),
                        self._dfoot('r3f')
                    ], className="card pair-card pair-card-chart"), md=7, className="mb-3 pair-col"),
                    dbc.Col(html.Div([
                        html.Div([html.H6("JPM Metric Correlation Stats", className="ct")], className="ch"),
                        html.Div([
                            dcc.Loading(html.Div(id='r3x', className="insight-shell analysis-shell"),
                                        type="dot", color=CS['primary'])
                        ], className="insight-load-shell")
                    ], className="card pair-card pair-card-side"), md=5, className="mb-3 pair-col")
                ], className="paired-row"),
                html.Div([
                    html.Div([
                        html.Div([
                            html.H6("JPMorgan Chase — All Metrics", className="ct", style={"color": "#fff"}),
                            dcc.Dropdown(id='det-date', options=self._do, value=dv,
                                         clearable=False, searchable=False, className="idd-d-light"),
                            html.Div([
                                html.Span("\u25b8", className="legend-dot", style={'color': CS['good']}),
                                html.Span("Better", className="legend-txt"),
                                html.Span("\u25b8", className="legend-dot", style={'color': CS['bad']}),
                                html.Span("Worse", className="legend-txt"),
                            ], className="det-legend"),
                            html.Div(style={"flex": "1"}),
                            html.Button([
                                html.Span("\u21e9", style={"marginRight": "5px", "fontSize": "13px"}),
                                "Export All Periods"
                            ], id="export-btn", className="export-btn"),
                            dcc.Download(id="export-download")
                        ], className="ch det-hdr"),
                        dcc.Loading(html.Div(id='det'), type="dot", color=CS['primary'])
                    ], className="card det-card")
                ], className="mb-4"),
                self._reference_section(),
                html.Div([
                    html.Span(f"Data via FDIC BankFind API · requested history since {REQUESTED_START_DATE_DISPLAY} · {PEER_UNIVERSE_LABEL} · {len(METRIC_ORDER)} metrics "
                              f"\u00b7 {len(METRIC_CATEGORIES)} categories "
                              f"\u00b7 UBPR concept codes verified via ffiec.gov/data/ubpr/report-user-guide",
                              className="foot-txt")
                ], className="foot"),
            ], className="main")
        ])

    def _reference_section(self):
        sections = []
        for ci, (cat_name, cat_metrics) in enumerate(METRIC_CATEGORIES):
            rows = []
            for m in cat_metrics:
                d = METRIC_DEFINITIONS.get(m, '')
                parts = d.split(' \xb7 ', 1)
                txt = parts[1] if len(parts) == 2 else d
                rows.append(html.Div([
                    html.Div(m, className="ref-name"),
                    html.Div(txt, className="ref-desc")
                ], className="ref-row"))
            accent = CATEGORY_ACCENTS.get(cat_name, CS['primary'])
            sections.append(html.Div([
                html.Div([
                    html.Div(style={"width": "3px", "background": accent,
                                    "borderRadius": "2px", "flexShrink": "0"}),
                    html.Span(cat_name, className="ref-cat-label"),
                    html.Span(f"{len(cat_metrics)} metrics", className="ref-cat-count")
                ], className="ref-cat"),
                html.Div(rows, className="ref-body")
            ], className="ref-section"))
        return html.Div([
            html.Div([
                html.H6("Metric Reference Guide", className="ct"),
                html.Span(f"{len(METRIC_ORDER)} metrics across {len(METRIC_CATEGORIES)} categories "
                          f"\xb7 Verify UBPR concept codes at ffiec.gov/data/ubpr/report-user-guide",
                          className="rng")
            ], className="ch"),
            html.Div(sections, className="ref-wrap")
        ], className="card ref-card")

    def _cbs(self, app):
        @app.callback(Output('peer-sel', 'value'),
                      [Input('sel-all', 'n_clicks'), Input('sel-clear', 'n_clicks')],
                      State('peer-sel', 'options'))
        def sel_action(n_all, n_clear, options):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            trig = ctx.triggered[0]['prop_id'].split('.')[0]
            if trig == 'sel-all' and n_all:
                return [x['value'] for x in options]
            if trig == 'sel-clear' and n_clear:
                return []
            raise PreventUpdate

        @app.callback(Output('exec-banner-wrap', 'children'), Input('peer-sel', 'value'))
        def ue(p):
            return self._exec_banner(p or [])

        @app.callback(Output('r1f', 'children'), Input('r1m', 'value'))
        def d1(m):
            return self._rdef(m)

        @app.callback(Output('r2f', 'children'), Input('r2m', 'value'))
        def d2(m):
            return self._rdef(m)

        @app.callback(Output('r3f', 'children'), [Input('r3p', 'value'), Input('r3s', 'value')])
        def d3(a, b):
            return html.Div([self._rdef(a, "Primary"), self._rdef(b, "Secondary")])

        @app.callback([Output('r1c', 'figure'), Output('r1o', 'children')],
                      [Input('r1m', 'value'), Input('r1d', 'value'), Input('peer-sel', 'value')])
        def u1(m, ds, p):
            if not m or not ds:
                return self._ef(""), html.Div()
            dt = pd.to_datetime(ds)
            bk = [self.GHB] + (p or [])
            f = self.df[(self.df['Date'] == dt) & self.df['Bank'].isin(bk)]
            if f.empty:
                return self._ef("No data"), html.Div()
            return self._bar(f.sort_values(m, ascending=is_inverse_metric(m)), m, dt), self._ov(f, m, dt)

        @app.callback([Output('r2c', 'figure'), Output('r2a', 'children'), Output('r2r', 'children')],
                      [Input('r2m', 'value'), Input('peer-sel', 'value'), Input('r2t', 'value')])
        def u2(m, p, y):
            if not m:
                return self._ef(""), html.Div(), ""
            bk = [self.GHB] + (p or [])
            end = self.analysis_end_date if self.analysis_end_date is not None else self.df['Date'].max()
            start_ts, end_ts = self._window_bounds(y, end=end, bank_filter=bk)
            return self._trend(bk, m, y), self._ta(bk, m, y), self._window_label(y, start_ts, end_ts)

        @app.callback([Output('r3c', 'figure'), Output('r3x', 'children')],
                      [Input('r3p', 'value'), Input('r3s', 'value'), Input('r3t', 'value')])
        def u3(a, b, y):
            if not a or not b:
                return self._ef(""), html.Div()
            return self._dual(a, b, y), self._corr(a, b, y)

        @app.callback(Output('det', 'children'), Input('det-date', 'value'))
        def ud(ds):
            if not ds:
                return html.P("Select a date", className="emp")
            dt = pd.to_datetime(ds)
            bf = self.df[(self.df['Bank'] == self.GHB) & (self.df['Date'] == dt)]
            if bf.empty:
                return html.P("No data for this date", className="emp")
            return self._bd(bf.iloc[0], dt)

        @app.callback(Output('export-download', 'data'), Input('export-btn', 'n_clicks'),
                      prevent_initial_call=True)
        def export_all_periods(n_clicks):
            if not n_clicks:
                raise PreventUpdate
            xlsx_bytes = build_primary_bank_export(self.df)
            if xlsx_bytes is None:
                raise PreventUpdate
            return dcc.send_bytes(xlsx_bytes,
                                  f"JPMorgan_SIB_Metrics_All_Periods_{datetime.now().strftime('%Y%m%d')}.xlsx")

    def _rdef(self, m, label=None):
        d = METRIC_DEFINITIONS.get(m, '')
        if not d:
            return None
        parts = d.split(' \xb7 ', 1)
        cat = parts[0] if len(parts) == 2 else ""
        txt = parts[1] if len(parts) == 2 else d
        pre = f"{label}: " if label else ""
        return html.Div([
            html.Span(f"{pre}{cat}", className="df-cat") if cat else None,
            html.Span(f" {txt}", className="df-txt")
        ], className="df-line")

    @staticmethod
    def _fmt(v, m=None):
        return fmt_val(v, m)

    def _ef(self, msg):
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text=msg, xref="paper", yref="paper",
                                             showarrow=False,
                                             font=dict(size=13, color=CS['text2']))],
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=20, r=20, t=20, b=20))
        return fig

    def _bl(self, **kw):
        return dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="'Inter',sans-serif", color=CS['text'], size=11),
                    hoverlabel=dict(bgcolor="white", font_size=11, font_color=CS['text'],
                                    font_family="'Inter',sans-serif", bordercolor=CS['border']),
                    **kw)

    def _bar(self, df, m, dt):
        isdol = is_dollar_metric(m)
        ispct = is_percent_metric(m)
        inverse = is_inverse_metric(m)
        with_vals = df[['Bank', m]].dropna()
        if not with_vals.empty:
            rank_series = with_vals[m].rank(method='min', ascending=inverse).astype(int)
            rank_map = dict(zip(with_vals['Bank'], rank_series))
        else:
            rank_map = {}
        def bar_color(bank):
            return CS['ghb'] if bank == self.GHB else CS['peer']
        c = [bar_color(b) for b in df['Bank']]
        o = [1.0 if b == self.GHB else 0.72 for b in df['Bank']]
        hover_vals = [fmt_val(v, m, with_unit=True) for v in df[m]]
        ht = '<b>%{x}</b><br>%{customdata}<extra></extra>'
        rank_texts = [(f"#{rank_map[b]}" if b in rank_map else "") for b in df['Bank']]
        fig = go.Figure(go.Bar(x=df['Bank'], y=df[m], customdata=hover_vals,
                               marker_color=c, marker_opacity=o,
                               marker_line_width=0, hovertemplate=ht,
                               text=rank_texts, textposition='outside',
                               textfont=dict(size=10, color=CS['text2'], family="'Inter',sans-serif"),
                               cliponaxis=False))
        v = df[m].dropna()
        mn, mx = (v.min(), v.max()) if len(v) else (0, 1)
        pad = max((mx - mn) * 0.22, 0.01)
        peer_vals = df[df['Bank'] != self.GHB][m].dropna()
        if len(peer_vals) > 0:
            pavg = peer_vals.mean()
            ann_txt = f"Peer Avg: {fmt_val(pavg, m, with_unit=True)}"
            fig.add_hline(y=pavg, line_dash="dot", line_color=CS['text3'], line_width=1,
                          annotation_text=ann_txt, annotation_position="top right",
                          annotation_font_size=9, annotation_font_color=CS['text2'])
        if len(v):
            if mn >= 0:
                y_min = 0
            elif mx <= 0:
                y_min = mn - pad
            else:
                y_min = mn - pad
            y_max = mx + pad if mx != 0 else pad
        else:
            y_min, y_max = 0, 1
        tfmt = ',.0f' if isdol else '.2f'
        y_title = '$000s' if isdol else ('%' if ispct else None)
        fig.update_layout(**self._bl(
            margin=dict(l=48, r=12, t=24, b=68),
            xaxis=dict(tickangle=-35, tickfont=dict(size=9.5), showgrid=False, showline=False),
            yaxis=dict(title_text=y_title, title_font=dict(size=9, color=CS['text3']),
                       tickformat=tfmt, range=[y_min, y_max], showgrid=True,
                       gridcolor=CS['grid'], showline=False, tickfont=dict(size=9.5),
                       zeroline=True, zerolinecolor=CS['border_strong']),
            bargap=0.35))
        return fig

    def _ov(self, df, m, dt):
        isdol = is_dollar_metric(m)
        inverse = is_inverse_metric(m)
        gh = df[df['Bank'] == self.GHB]
        gv = gh[m].values[0] if not gh.empty else None
        if gv is not None and pd.isna(gv):
            gv = None
        peer_df = df[df['Bank'] != self.GHB].copy()
        peer_vals = peer_df[m].dropna()
        peer_count = len(peer_vals)
        f = lambda val: fmt_val(val, m, with_unit=True)
        rank, total, pctl = compute_peer_rank(df, dt, m, self.GHB)

        qoq_prev, yoy_prev = self._ghb_qoq_yoy(m, dt) if self._ghb_idx(dt) is not None else (None, None)
        if qoq_prev is None and yoy_prev is None:
            qoq_prev, yoy_prev = compute_period_deltas(self.df, self.GHB, m, dt)
        qoq_text, qoq_color = fmt_delta(gv, qoq_prev, m)
        yoy_text, yoy_color = fmt_delta(gv, yoy_prev, m)

        if peer_count == 0:
            pf, pc, pi = "No peer comparison", CS['text3'], "\u2022"
        elif gv is None:
            pf, pc, pi = "N/A", CS['text2'], ""
        elif peer_count < 4:
            pmed = float(peer_vals.median())
            if np.isclose(gv, pmed, equal_nan=False):
                pf, pc, pi = "At Peer Median", CS['peer_band_mid'], "\u2022"
            elif (gv < pmed and inverse) or (gv > pmed and not inverse):
                pf, pc, pi = "Better than Peer Median", CS['peer_band_top'], "\u25b4"
            else:
                pf, pc, pi = "Worse than Peer Median", CS['peer_band_low'], "\u25be"
        else:
            q1, q3 = np.percentile(peer_vals, [25, 75])
            # FIX #4: symmetric inclusive quartile boundaries for both branches.
            if inverse:
                if gv <= q1:
                    pf, pc, pi = "Top Quartile (best)", CS['peer_band_top'], "\u25b4"
                elif gv >= q3:
                    pf, pc, pi = "Bottom Quartile (worst)", CS['peer_band_low'], "\u25be"
                else:
                    pf, pc, pi = "Middle 50%", CS['peer_band_mid'], "\u2022"
            else:
                if gv >= q3:
                    pf, pc, pi = "Top Quartile (best)", CS['peer_band_top'], "\u25b4"
                elif gv <= q1:
                    pf, pc, pi = "Bottom Quartile (worst)", CS['peer_band_low'], "\u25be"
                else:
                    pf, pc, pi = "Middle 50%", CS['peer_band_mid'], "\u2022"

        def sr(l, val, h=False):
            return html.Div([html.Span(l, className="ol"), html.Span(str(val), className="ov")],
                            className="or" + (" oh" if h else ""))

        def sr_colored(l, val, color):
            return html.Div([html.Span(l, className="ol"),
                             html.Span(str(val), className="ov", style={'color': color, 'fontWeight': '600'})],
                            className="or")

        unit_note = " ($000s)" if isdol else ""
        rank_text = f"#{rank} of {total}" if rank and total > 1 else "\u2014"
        peer_avg = f(peer_vals.mean()) if peer_count else "N/A"
        peer_median = f(peer_vals.median()) if peer_count else "N/A"
        if peer_count:
            high_idx = peer_df[m].idxmax()
            low_idx = peer_df[m].idxmin()
            peer_high = f"{f(peer_df.loc[high_idx, m])} \u2014 {peer_df.loc[high_idx, 'Bank']}"
            peer_low = f"{f(peer_df.loc[low_idx, m])} \u2014 {peer_df.loc[low_idx, 'Bank']}"
        else:
            peer_high = "N/A"
            peer_low = "N/A"

        gauge_section = html.Div([
            html.Div([
                make_percentile_arc_img(pctl, size=OVERVIEW_GAUGE_SIZE),
                html.Div([
                    html.Div("Peer Position", className="pct-label"),
                    html.Div(rank_text, className="pct-rank"),
                    html.Div([
                        html.Span(pi, style={'color': pc, 'marginRight': '4px'}),
                        html.Span(pf, style={'color': pc, 'fontWeight': '600'})
                    ], className="pct-band"),
                ], className="pct-info"),
            ], className="pct-gauge-wrap"),
        ], className="pct-gauge-section")

        return html.Div([
            gauge_section,
            html.Div([
                html.Div(f"Peer Snapshot{unit_note}", className="ost"),
                sr("Peer Average", peer_avg),
                sr("Peer Median", peer_median),
                sr(self.GHB, f(gv) if gv is not None else "N/A", h=True),
                sr("Peer High", peer_high),
                sr("Peer Low", peer_low)
            ], className="os"),
            html.Div([
                html.Div("JPM Momentum", className="ost"),
                sr_colored("QoQ Change", qoq_text, qoq_color),
                sr_colored("YoY Change", yoy_text, yoy_color),
            ], className="os"),
        ], className="ow")

    def _trend(self, bk, m, y):
        isdol = is_dollar_metric(m)
        f = self.df[self.df['Bank'].isin(bk)]
        if f.empty:
            return self._ef("No data")
        start, end = self._window_bounds(y, bank_filter=bk)
        f = f[(f['Date'] <= end) & (f['Date'] >= start)]
        if f.empty:
            return self._ef("No data for selected historical window")
        pv = f.pivot(index='Date', columns='Bank', values=m)
        fig = go.Figure()
        peer_cols = [c for c in pv.columns if c != self.GHB]
        if len(peer_cols) >= 2:
            fig.add_trace(go.Scatter(x=pv.index, y=pv[peer_cols].max(axis=1), mode='lines',
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=pv.index, y=pv[peer_cols].min(axis=1), mode='lines',
                                     line=dict(width=0), fill='tonexty',
                                     fillcolor='rgba(148,163,184,0.10)',
                                     showlegend=False, hoverinfo='skip'))
        ispct = is_percent_metric(m)
        for b in pv.columns:
            ig = b == self.GHB
            hover_vals = [fmt_val(v, m, with_unit=True) for v in pv[b]]
            ht = '<b>' + b + '</b><br>%{x|%m/%d/%Y}<br>%{customdata}<extra></extra>'
            fig.add_trace(go.Scatter(x=pv.index, y=pv[b], customdata=hover_vals, mode='lines', name=b,
                                     line=dict(color=CS['ghb'] if ig else CS['peer'],
                                               width=2.5 if ig else 1.2, shape='spline'),
                                     opacity=1 if ig else CS['peer_op'], hovertemplate=ht))
        dt, tick_fmt = self._axis_for_window(start, end)
        tfmt = ',.0f' if isdol else '.2f'
        y_title = '$000s' if isdol else ('%' if ispct else None)
        fig.update_layout(**self._bl(
            showlegend=False, hovermode='x unified',
            margin=dict(l=48, r=12, t=6, b=40),
            xaxis=dict(showgrid=False, tickformat=tick_fmt,
                       dtick=dt, tickangle=-35, tickfont=dict(size=9)),
            yaxis=dict(title_text=y_title, title_font=dict(size=9, color=CS['text3']),
                       showgrid=True, gridcolor=CS['grid'], tickformat=tfmt,
                       tickfont=dict(size=9), zeroline=True, zerolinecolor=CS['border_strong'])))
        return fig

    def _ta(self, bk, m, y):
        isdol = is_dollar_metric(m)
        inverse = is_inverse_metric(m)
        f = self.df[self.df['Bank'].isin(bk)]
        if f.empty:
            return html.Div("No data", className="emp")
        start, end = self._window_bounds(y, bank_filter=bk)
        f = f[(f['Date'] <= end) & (f['Date'] >= start)]
        if f.empty:
            return html.Div("No data for selected historical window", className="emp")
        pv = f.pivot(index='Date', columns='Bank', values=m)
        if self.GHB not in pv.columns or pv[self.GHB].count() < 2:
            return html.Div("Insufficient data", className="emp")
        ghd = pv[self.GHB].dropna()
        stats_by_bank = {}
        for b in pv.columns:
            bd = pv[b].dropna()
            if len(bd) < 2:
                continue
            sv, ev = bd.iloc[0], bd.iloc[-1]
            chg = calc_trend_change(sv, ev, m)
            vol = bd.std()
            mean_val = bd.mean()
            cv = (vol / abs(mean_val)) * 100 if mean_val != 0 and not pd.isna(mean_val) else np.nan
            sl = np.polyfit(np.arange(len(bd)), bd.values, 1)[0]
            cr = np.nan
            if b != self.GHB:
                ov = pd.concat([ghd, bd], axis=1).dropna()
                if len(ov) >= 2 and ov.iloc[:, 0].nunique() >= 2 and ov.iloc[:, 1].nunique() >= 2:
                    cr = ov.iloc[:, 0].corr(ov.iloc[:, 1])
            stats_by_bank[b] = {'g': chg, 'v': vol, 'cv': cv, 'c': cr, 't': trend_direction_label(sl)}
        ghb_stats = stats_by_bank.get(self.GHB, {})
        peer_stats = {b: s for b, s in stats_by_bank.items() if b != self.GHB}
        peer_growth = {b: s for b, s in peer_stats.items() if not pd.isna(s['g'])}
        peer_corr = {b: s['c'] for b, s in peer_stats.items() if not pd.isna(s['c'])}
        most_similar = max(peer_corr.items(), key=lambda x: x[1]) if peer_corr else (None, np.nan)
        fg = lambda v: fmt_trend_change(v, m)
        fvol = lambda v: ("N/A" if v is None or pd.isna(v) else
                          (fmt_val(v, m, with_unit=True) if isdol else f"{v:.4f} pp"))
        fcv = lambda v: "N/A" if v is None or pd.isna(v) else f"{v:.1f}%"

        def sr(l, v, h=False):
            return html.Div([html.Span(l, className="ol"), html.Span(str(v), className="ov")],
                            className="or" + (" oh" if h else ""))

        if peer_growth:
            best_key = (lambda x: x[1]['g'])
            best_peer_name, best_peer_stats = (min(peer_growth.items(), key=best_key)
                                               if inverse else max(peer_growth.items(), key=best_key))
            avg_peer_growth = fg(np.nanmean([s['g'] for s in peer_growth.values()]))
            best_peer_text = f"{best_peer_name} ({fg(best_peer_stats['g'])})"
        else:
            avg_peer_growth = "N/A"
            best_peer_text = "N/A"
        change_label = "Growth" if isdol else "Change"
        trend_window_label = "Full-History Trend" if str(y).upper() == "FULL" else f"{y}Y Trend"
        return html.Div([
            html.Div([
                html.Div(trend_window_label, className="ost"),
                sr(f"JPM {change_label}", fg(ghb_stats.get('g')), h=True),
                sr("Direction", ghb_stats.get('t', 'N/A')),
                sr("Volatility (std)", fvol(ghb_stats.get('v'))),
                sr("Volatility (CV)", fcv(ghb_stats.get('cv')))
            ], className="os"),
            html.Div([
                html.Div("Peers", className="ost"),
                sr(f"Avg Peer {change_label}", avg_peer_growth),
                sr("Most Similar", f"{most_similar[0]} ({most_similar[1]:.2f})" if most_similar[0] else "N/A"),
                sr("Best Peer Improvement" if inverse else f"Best Peer {change_label}", best_peer_text)
            ], className="os")
        ], className="ow")

    def _dual(self, m1, m2, y):
        g = self.df[self.df['Bank'] == self.GHB].copy()
        if g.empty:
            return self._ef("No JPMorgan data")
        start, end = self._window_bounds(y, bank_filter=[self.GHB])
        g = g[(g['Date'] <= end) & (g['Date'] >= start)]
        if g.empty:
            return self._ef("No JPMorgan data for selected historical window")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        isdol1 = is_dollar_metric(m1)
        isdol2 = is_dollar_metric(m2)
        cd1 = [fmt_val(v, m1, with_unit=True) for v in g[m1]]
        cd2 = [fmt_val(v, m2, with_unit=True) for v in g[m2]]
        ht1 = '%{x|%m/%d/%Y}<br>' + m1[:40] + ': %{customdata}<extra></extra>'
        ht2 = '%{x|%m/%d/%Y}<br>' + m2[:40] + ': %{customdata}<extra></extra>'
        fig.add_trace(go.Scatter(x=g['Date'], y=g[m1], customdata=cd1, mode='lines', name=m1[:40],
                                 line=dict(color=CS['ghb'], width=2.5, shape='spline'),
                                 hovertemplate=ht1), secondary_y=False)
        fig.add_trace(go.Scatter(x=g['Date'], y=g[m2], customdata=cd2, mode='lines', name=m2[:40],
                                 line=dict(color=CS['ghb2'], width=2.5, dash='dot', shape='spline'),
                                 hovertemplate=ht2), secondary_y=True)
        dt, tick_fmt = self._axis_for_window(start, end)
        fig.update_layout(**self._bl(
            showlegend=True, hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left",
                        x=0, font=dict(size=9)),
            margin=dict(l=48, r=48, t=26, b=40),
            xaxis=dict(showgrid=False, tickformat=tick_fmt,
                       dtick=dt, tickangle=-35, tickfont=dict(size=9))))
        tfmt1 = ',.0f' if isdol1 else '.2f'
        tfmt2 = ',.0f' if isdol2 else '.2f'
        ytitle1 = '$000s' if isdol1 else ('%' if is_percent_metric(m1) else None)
        ytitle2 = '$000s' if isdol2 else ('%' if is_percent_metric(m2) else None)
        fig.update_yaxes(title_text=ytitle1, title_font=dict(size=9, color=CS['ghb']),
                         tickformat=tfmt1, showgrid=True, gridcolor=CS['grid'],
                         tickfont=dict(size=9, color=CS['ghb']),
                         zeroline=True, zerolinecolor=CS['border_strong'], secondary_y=False)
        fig.update_yaxes(title_text=ytitle2, title_font=dict(size=9, color=CS['ghb2']),
                         tickformat=tfmt2, showgrid=False,
                         tickfont=dict(size=9, color=CS['ghb2']), zeroline=True,
                         zerolinecolor=CS['border_strong'], secondary_y=True)
        return fig

    def _corr(self, m1, m2, y):
        g = self.df[self.df['Bank'] == self.GHB].copy()
        if g.empty:
            return html.Div("No data", className="emp")
        start, end = self._window_bounds(y, bank_filter=[self.GHB])
        g = g[(g['Date'] <= end) & (g['Date'] >= start)]
        if g.empty:
            return html.Div("No JPMorgan data for selected historical window", className="emp")
        cm = g[[m1, m2]].dropna()
        n = len(cm)

        def sr(l, v, h=False):
            return html.Div([html.Span(l, className="ol"), html.Span(str(v), className="ov")],
                            className="or" + (" oh" if h else ""))

        if n < 3:
            return html.Div([html.Div([
                html.Div("Insufficient Data", className="ost"),
                sr("Periods", str(n)), sr("Required", "3+")
            ], className="os")], className="ow")
        if cm[m1].nunique() < 2 or cm[m2].nunique() < 2:
            return html.Div([html.Div([
                html.Div("Correlation Unavailable", className="ost"),
                sr("Reason", "One metric is constant"),
                sr("Periods", str(n))
            ], className="os")], className="ow")
        r, pv = stats.pearsonr(cm[m1], cm[m2])
        if pd.isna(r) or pd.isna(pv):
            return html.Div([html.Div([
                html.Div("Correlation Unavailable", className="ost"),
                sr("Reason", "Correlation returned N/A"),
                sr("Periods", str(n))
            ], className="os")], className="ow")
        r2 = r ** 2
        st = "Strong" if abs(r) >= 0.7 else ("Moderate" if abs(r) >= 0.4 else "Weak")
        dr = "positive" if r > 0 else ("negative" if r < 0 else "flat")
        if abs(r) >= 0.7:
            rc = CS['primary']
        elif abs(r) >= 0.4:
            rc = CS['text2']
        else:
            rc = CS['text3']

        def ms(s, met):
            if len(s) < 2:
                return [sr("Data", "N/A")]
            sv, ev = s.iloc[0], s.iloc[-1]
            ch = calc_trend_change(sv, ev, met)
            sl = np.polyfit(np.arange(len(s)), s.values, 1)[0]
            mean_val = s.mean()
            std_val = s.std()
            cv_val = (std_val / abs(mean_val)) * 100 if mean_val != 0 else np.nan
            fcv = "N/A" if pd.isna(cv_val) else f"{cv_val:.1f}%"
            return [
                sr("Direction", trend_direction_label(sl)),
                sr("Change", fmt_trend_change(ch, met)),
                sr("Volatility (CV)", fcv)
            ]

        # FIX #6: each metric's per-metric trend stats use its OWN windowed
        # series, not the m1 intersection m2 frame. The correlation (r, p, n)
        # still uses the intersection cm.
        s1 = g[m1].dropna()
        s2 = g[m2].dropna()
        return html.Div([
            html.Div([
                html.Div("Relationship", className="ost"),
                sr("Correlation (r)", f"{r:.4f}", h=True),
                sr("R\u00b2", f"{r2:.4f}"),
                html.Div([
                    html.Span("Strength", className="ol"),
                    html.Span(f"{st} {dr}", style={"color": rc, "fontWeight": "600"}, className="ov")
                ], className="or"),
                sr("p-value", f"{pv:.4f}" if pv >= 0.0001 else "< 0.0001"),
                sr("Periods", str(n))
            ], className="os"),
            html.Div([html.Div("Primary", className="ost")] + ms(s1, m1), className="os"),
            html.Div([html.Div("Secondary", className="ost")] + ms(s2, m2), className="os")
        ], className="ow")

    def _bd(self, data, date):
        sections = []
        for cat_name, cat_metrics in METRIC_CATEGORIES:
            present = [m for m in cat_metrics if m in data.index]
            if not present:
                continue
            accent = CATEGORY_ACCENTS.get(cat_name, CS['primary'])
            bg = CATEGORY_BG.get(cat_name, '#f8fafc')
            mid = (len(present) + 1) // 2

            def make_rows(ms):
                rows = []
                for m in ms:
                    curr = data[m]
                    if pd.isna(curr):
                        curr = None
                    qoq_prev, yoy_prev = self._ghb_qoq_yoy(m, date)
                    qoq_text, qoq_color = fmt_delta(curr, qoq_prev, m)
                    yoy_text, yoy_color = fmt_delta(curr, yoy_prev, m)
                    spark_vals = self._ghb_spark(m, date, self.SPARK_LOOKBACK)
                    val_display = fmt_val(curr, m, with_unit=True)
                    rows.append(html.Div([
                        html.Div(m, className="dn"),
                        html.Div([
                            html.Div(val_display, className="dv"),
                            html.Div([make_sparkline_img(spark_vals, width=78, height=20)], className="dspark"),
                            html.Div([
                                html.Span("QoQ", className="ddelta-lbl"),
                                html.Span(qoq_text, className="ddelta-val", style={'color': qoq_color}),
                            ], className="ddelta"),
                            html.Div([
                                html.Span("YoY", className="ddelta-lbl"),
                                html.Span(yoy_text, className="ddelta-val", style={'color': yoy_color}),
                            ], className="ddelta"),
                        ], className="dright"),
                    ], className="dr"))
                return rows

            sections.append(html.Div([
                html.Div([
                    html.Div(style={"width": "3px", "background": accent, "borderRadius": "2px",
                                    "flexShrink": "0", "alignSelf": "stretch"}),
                    html.Span(cat_name, className="det-cat-label"),
                    html.Span(f"{len(present)}", className="det-cat-count")
                ], className="det-cat-hdr", style={"background": bg}),
                dbc.Row([
                    dbc.Col(html.Div(make_rows(present[:mid]), className="dc"), xs=12, md=6),
                    dbc.Col(html.Div(make_rows(present[mid:]), className="dc"), xs=12, md=6)
                ], className="det-cat-body")
            ], className="det-cat-section"))
        return html.Div(sections, className="dg")

    def _css(self):
        c = dict(CS)
        c['paired_card_min_height'] = PAIRED_CARD_MIN_HEIGHT
        css = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box }
html { scroll-behavior: smooth }
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: %(bg)s; color: %(text)s; margin: 0;
    -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
    font-feature-settings: 'cv11', 'ss01', 'ss03';
}
::-webkit-scrollbar { width: 8px; height: 8px }
::-webkit-scrollbar-track { background: transparent }
::-webkit-scrollbar-thumb { background: rgba(15,23,42,0.12); border-radius: 4px }
::-webkit-scrollbar-thumb:hover { background: rgba(15,23,42,0.22) }
.hdr {
    display: flex; align-items: center; justify-content: space-between;
    padding: 11px 28px; background: rgba(255,255,255,0.92);
    border-bottom: 1px solid rgba(15,23,42,0.06);
    position: sticky; top: 0; z-index: 20; backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}
.hdr-brand { display: flex; align-items: center; gap: 11px }
.hdr-mark {
    width: 36px; height: 36px; background: linear-gradient(135deg, %(primary)s 0%%, %(primary_light)s 100%%);
    color: #fff; border-radius: 9px; display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 10.5px; letter-spacing: 0.6px;
    box-shadow: 0 2px 6px rgba(0,94,184,0.25), inset 0 1px 0 rgba(255,255,255,0.15);
}
.hdr-title { font-size: 0.98rem; font-weight: 700; color: %(primary)s; line-height: 1.15; display: block; letter-spacing: -0.01em }
.hdr-sub { font-size: 0.68rem; color: %(text2)s; font-weight: 500; display: block; letter-spacing: 0.15px }
.hdr-src { font-size: 0.7rem; font-weight: 600; color: %(text2)s }
.hdr-cnt { font-size: 0.7rem; color: %(light)s; margin-left: 4px }
.toolbar {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; background: #fff;
    border: 1px solid rgba(15,23,42,0.06); border-radius: 12px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03), 0 3px 10px rgba(15,23,42,0.02);
    margin-bottom: 14px;
}
.tb-label { font-size: 0.64rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: %(text2)s; flex-shrink: 0 }
.tb-dd { flex: 1 }
.tb-dd .Select-control { min-height: 32px !important; border-radius: 8px !important }
.tb-dd .Select-value-label { font-size: 0.64rem !important }
.tb-dd .Select-multi-value-wrapper .Select-value {
    background: %(accent_light)s !important; border: none !important;
    border-radius: 5px !important; color: %(accent)s !important;
    font-size: 0.62rem !important; font-weight: 500 !important;
}
.tb-dd .Select-multi-value-wrapper .Select-value-icon { border-right: none !important }
.tb-btn {
    background: %(primary)s; color: #fff; border: none; padding: 6px 12px;
    border-radius: 8px; font-size: 0.64rem; font-weight: 600; cursor: pointer;
    transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 1px 2px rgba(14,62,27,0.15);
}
.tb-btn:hover { background: %(primary_light)s; transform: translateY(-1px); box-shadow: 0 2px 6px rgba(14,62,27,0.22) }
.tb-btn:active { transform: translateY(0); box-shadow: 0 1px 2px rgba(14,62,27,0.15) }
.tb-btn-secondary { background: %(neutral_light)s; color: %(text2)s; box-shadow: 0 1px 2px rgba(15,23,42,0.05) }
.tb-btn-secondary:hover { background: %(lighter)s; color: %(text)s }
.main { padding: 14px 28px 28px; max-width: 1600px; margin: 0 auto }
.warn-banner {
    margin: 0 0 14px; padding: 10px 14px; border-radius: 12px;
    border: 1px solid %(warn)s; background: %(warn_light)s; color: %(warn_dark)s;
    font-size: 0.74rem; line-height: 1.45; box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
.warn-banner strong { font-weight: 800 }
.paired-row { --paired-card-min-height: %(paired_card_min_height)spx; margin-bottom: 0 }
.pair-col { display: flex }
.pair-card { width: 100%%; min-height: var(--paired-card-min-height); display: flex; flex-direction: column }
.viz-shell, .insight-load-shell { flex: 1 1 auto; min-height: 0; display: flex }
.viz-shell > div, .insight-load-shell > div { width: 100%%; flex: 1 1 auto }
.viz-shell .dash-loading, .insight-load-shell .dash-loading { width: 100%%; flex: 1 1 auto; display: flex }
.viz-graph { height: 100%% !important }
.insight-shell { width: 100%%; height: 100%%; display: flex }
.insight-shell > .ow { width: 100%% }
.card {
    background: #fff; border-radius: 12px; border: 1px solid rgba(15,23,42,0.06);
    box-shadow: 0 1px 2px rgba(15,23,42,0.03), 0 3px 10px rgba(15,23,42,0.02);
    overflow: hidden; transition: box-shadow 0.22s ease, transform 0.22s ease;
}
.card:hover { box-shadow: 0 2px 4px rgba(15,23,42,0.04), 0 8px 24px rgba(15,23,42,0.05) }
.exec-banner {
    margin-bottom: 14px; background: linear-gradient(135deg, %(primary_dark)s 0%%, %(primary)s 55%%, %(primary_light)s 100%%);
    border-radius: 14px; padding: 1px;
    box-shadow: 0 4px 18px rgba(0,94,184,0.18), 0 1px 3px rgba(0,94,184,0.10);
    position: relative; overflow: hidden;
}
.exec-banner::before {
    content: ''; position: absolute; top: 0; right: 0; width: 280px; height: 100%%;
    background: radial-gradient(circle at 100%% 0%%, rgba(255,255,255,0.08) 0%%, transparent 60%%);
    pointer-events: none;
}
.exec-banner-inner { background: linear-gradient(180deg, rgba(0,59,115,0.98) 0%%, rgba(0,94,184,0.96) 55%%, rgba(11,79,138,0.98) 100%%); border-radius: 13px; padding: 16px 22px }
.exec-banner-hdr { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.08) }
.exec-banner-title { font-size: 0.78rem; font-weight: 700; color: #fff; text-transform: uppercase; letter-spacing: 1.2px }
.exec-banner-date { font-size: 0.7rem; color: rgba(255,255,255,0.6); font-weight: 500 }
.exec-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px }
.exec-card {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 11px 13px; display: flex; flex-direction: column; gap: 4px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden;
}
.exec-card:hover { background: rgba(255,255,255,0.10); border-color: rgba(255,255,255,0.18); transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,0.18) }
.exec-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    opacity: 0; transition: opacity 0.3s;
}
.exec-card:hover::before { opacity: 1 }
.exec-hdr { display: flex; align-items: center; justify-content: space-between; margin-bottom: 2px }
.exec-label { font-size: 0.58rem; font-weight: 700; color: rgba(255,255,255,0.65); text-transform: uppercase; letter-spacing: 0.9px }
.exec-rank { font-size: 0.6rem; font-weight: 700; padding: 1.5px 7px; border-radius: 10px; border: 1px solid; background: rgba(0,0,0,0.2); font-variant-numeric: tabular-nums }
.exec-val { font-size: 1.45rem; font-weight: 700; color: #fff; line-height: 1.1; letter-spacing: -0.015em; font-variant-numeric: tabular-nums; margin-top: 2px }
.exec-metric-name { font-size: 0.6rem; color: rgba(255,255,255,0.55); font-weight: 500; line-height: 1.3; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-bottom: 3px }
.exec-spark { background: rgba(255,255,255,0.04); border-radius: 5px; padding: 3px 5px; margin: 2px 0 }
.exec-spark img { filter: brightness(0) saturate(100%%) invert(100%%) !important; opacity: 0.85; max-width: 100%%; height: auto }
.exec-delta { display: flex; align-items: center; justify-content: space-between; padding-top: 4px; border-top: 1px solid rgba(255,255,255,0.06); margin-top: 2px }
.exec-delta-label { font-size: 0.58rem; font-weight: 600; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 0.7px }
.exec-delta-val { font-size: 0.7rem; font-weight: 700; font-variant-numeric: tabular-nums }
.ch { padding: 9px 15px; display: flex; align-items: center; gap: 9px; border-bottom: 1px solid rgba(15,23,42,0.04); flex-wrap: nowrap }
.ch-wrap { flex-wrap: wrap; gap: 7px }
.ct { font-size: 0.78rem; font-weight: 700; color: %(text)s; margin: 0; white-space: nowrap; flex-shrink: 0; letter-spacing: -0.005em }
.rng { font-size: 0.62rem; color: %(text2)s; white-space: nowrap; margin-left: auto }
.vs { font-size: 0.64rem; color: %(light)s; font-weight: 500; flex-shrink: 0 }
.idd-m { width: 380px !important; flex-shrink: 0 }
.idd-m .Select-control, .idd-m2 .Select-control {
    min-height: 26px !important; border-radius: 7px !important;
    background: %(hover_bg)s !important; border-color: rgba(15,23,42,0.07) !important; transition: all 0.15s ease;
}
.idd-m .Select-control:hover, .idd-m2 .Select-control:hover { border-color: rgba(14,62,27,0.3) !important; background: #fff !important }
.idd-m .Select-value, .idd-m2 .Select-value { line-height: 26px !important }
.idd-m .Select-value-label, .idd-m2 .Select-value-label { font-size: 0.64rem !important; font-weight: 500 !important }
.idd-m .Select-placeholder, .idd-m2 .Select-placeholder { font-size: 0.64rem !important }
.idd-m .Select-input > input, .idd-m2 .Select-input > input { font-size: 0.64rem !important }
.idd-m2 { width: 310px !important }
.idd-t { width: 128px !important; min-width: 128px !important; flex-shrink: 0 }
.idd-t .Select-control { min-height: 26px !important; border-radius: 7px !important; background: %(hover_bg)s !important; border-color: rgba(15,23,42,0.07) !important }
.idd-t .Select-value { line-height: 26px !important }
.idd-t .Select-value-label { font-size: 0.64rem !important; font-weight: 600 !important; color: %(primary)s !important; white-space: nowrap !important }
.idd-d { width: 165px !important; flex-shrink: 0 }
.idd-d .Select-control { min-height: 26px !important; border-radius: 7px !important; background: %(hover_bg)s !important; border-color: rgba(15,23,42,0.07) !important }
.idd-d .Select-value { line-height: 26px !important }
.idd-d .Select-value-label { font-size: 0.64rem !important; font-weight: 600 !important; color: %(primary)s !important }
.idd-d-light { width: 165px !important; flex-shrink: 0 }
.idd-d-light .Select-control { min-height: 28px !important; border-radius: 7px !important; background: rgba(255,255,255,0.15) !important; border-color: rgba(255,255,255,0.25) !important; transition: all 0.18s ease }
.idd-d-light .Select-control:hover { background: rgba(255,255,255,0.22) !important; border-color: rgba(255,255,255,0.4) !important }
.idd-d-light .Select-value { line-height: 28px !important }
.idd-d-light .Select-value-label { font-size: 0.64rem !important; font-weight: 600 !important; color: #fff !important }
.idd-d-light .Select-arrow { border-color: #fff transparent transparent !important }
.det-hdr { background: linear-gradient(135deg, %(ghb)s 0%%, %(primary_light)s 100%%) !important; border-radius: 12px 12px 0 0 !important; border-bottom: none !important; padding: 10px 16px !important }
.det-card { overflow: hidden }
.det-legend { display: flex; align-items: center; gap: 6px; margin-left: 12px }
.legend-dot { font-size: 0.8rem; line-height: 1 }
.legend-txt { color: #fff !important; font-size: 0.68rem; font-weight: 600; line-height: 1; letter-spacing: 0.2px }
.export-btn {
    background: rgba(255,255,255,0.18); color: #fff; border: 1px solid rgba(255,255,255,0.35);
    padding: 5px 14px; border-radius: 7px; font-size: 0.68rem; font-weight: 600;
    cursor: pointer; white-space: nowrap; display: flex; align-items: center;
    transition: all 0.15s ease; flex-shrink: 0; backdrop-filter: blur(4px);
}
.export-btn:hover { background: rgba(255,255,255,0.30); border-color: rgba(255,255,255,0.55); transform: translateY(-1px); box-shadow: 0 2px 6px rgba(0,0,0,0.15) }
.export-btn:active { background: rgba(255,255,255,0.10); transform: scale(0.97) }
.dfoot { padding: 5px 16px 8px; min-height: 20px; flex: 0 0 auto }
.df-line { font-size: 0.66rem; line-height: 1.35; margin-bottom: 1px }
.df-cat { color: %(primary)s; font-weight: 700; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.4px }
.df-txt { color: %(text2)s }
.ow { padding: 10px 16px; display: flex; flex-direction: column; justify-content: flex-start; gap: 10px; height: 100%% }
.os { margin-bottom: 0; padding: 11px 14px; background: %(hover_bg)s; border-radius: 9px; border: 1px solid rgba(15,23,42,0.04); transition: border-color 0.15s ease, box-shadow 0.15s ease }
.os:last-child { margin-bottom: 0 }
.os:hover { border-color: rgba(15,23,42,0.10); box-shadow: 0 1px 2px rgba(0,0,0,0.03) }
.ost { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: %(primary)s; margin-bottom: 7px }
.or { display: flex; justify-content: space-between; align-items: center; padding: 4px 0; border-bottom: 1px solid rgba(15,23,42,0.03) }
.or:last-child { border-bottom: none }
.oh { background: %(accent_light)s; margin: 3px -7px; padding: 5px 7px; border-radius: 6px; border-bottom: none }
.ol { font-size: 0.78rem; font-weight: 500; color: %(text2)s }
.ov { font-size: 0.78rem; font-weight: 600; color: %(text)s; text-align: right; max-width: 55%%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-variant-numeric: tabular-nums }
.pct-gauge-section { padding: 10px 12px 8px; background: linear-gradient(180deg, %(peer_tint)s 0%%, transparent 100%%); border-radius: 9px; margin-bottom: 0; border: 1px solid rgba(15,23,42,0.04) }
.pct-gauge-wrap { display: flex; align-items: center; gap: 12px; min-height: 84px }
.pct-gauge-wrap > img { flex-shrink: 0 }
.pct-info { min-width: 0 }
.pct-label { font-size: 0.58rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: %(text2)s; margin-bottom: 2px }
.pct-rank { font-size: 0.88rem; font-weight: 700; color: %(text)s; line-height: 1.15; font-variant-numeric: tabular-nums }
.pct-band { font-size: 0.72rem; font-weight: 600; margin-top: 2px }
@media (max-width: 991.98px) {
    .pair-col { display: block }
    .pair-card { min-height: auto }
    .viz-shell, .insight-load-shell, .insight-shell, .ow { height: auto }
}
.dg { padding: 6px 12px 14px }
.det-cat-section { margin-bottom: 3px }
.det-cat-section:last-child { margin-bottom: 0 }
.det-cat-hdr { display: flex; align-items: center; gap: 8px; padding: 7px 12px; border-radius: 7px; margin: 8px 4px 5px }
.det-cat-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.55px; color: %(text)s }
.det-cat-count { font-size: 0.6rem; font-weight: 600; color: %(text2)s; background: rgba(15,23,42,0.06); padding: 2px 7px; border-radius: 10px; margin-left: auto }
.det-cat-body { padding: 0 4px }
.dc { padding: 0 6px }
.dr { display: flex; justify-content: space-between; align-items: center; padding: 6px 8px; border-bottom: 1px solid rgba(15,23,42,0.03); gap: 10px; border-radius: 4px; transition: background 0.1s ease }
.dr:last-child { border-bottom: none }
.dr:hover { background: rgba(15,23,42,0.015) }
.dn { font-size: 0.68rem; font-weight: 500; color: %(text2)s; flex: 1 1 auto; line-height: 1.3; min-width: 0 }
.dright { display: flex; align-items: center; gap: 10px; flex-shrink: 0 }
.dv { font-size: 0.74rem; font-weight: 700; color: %(text)s; text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; min-width: 56px }
.dspark { flex-shrink: 0; opacity: 0.85; display: flex; align-items: center }
.ddelta { display: flex; flex-direction: column; align-items: flex-end; gap: 1px; font-variant-numeric: tabular-nums; min-width: 46px }
.ddelta-lbl { font-size: 0.52rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; color: %(light)s; line-height: 1 }
.ddelta-val { font-size: 0.66rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums }
.emp { font-size: 0.78rem; color: %(text2)s; padding: 20px; text-align: center }
.ref-card { margin-bottom: 28px }
.ref-wrap { padding: 14px 18px }
.ref-section { margin-bottom: 13px }
.ref-section:last-child { margin-bottom: 0 }
.ref-cat { display: flex; align-items: center; gap: 9px; padding: 8px 12px; background: %(hover_bg)s; border-radius: 7px; margin-bottom: 7px; border: 1px solid rgba(15,23,42,0.03) }
.ref-cat-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.55px; color: %(text)s }
.ref-cat-count { font-size: 0.58rem; font-weight: 600; color: %(text2)s; margin-left: auto; background: rgba(15,23,42,0.05); padding: 2px 8px; border-radius: 10px }
.ref-body { padding: 0 4px }
.ref-row { display: flex; gap: 18px; padding: 6px 8px; border-bottom: 1px solid rgba(15,23,42,0.025); align-items: baseline; border-radius: 4px; transition: background 0.1s ease }
.ref-row:last-child { border-bottom: none }
.ref-row:hover { background: %(hover_bg)s }
.ref-name { font-size: 0.72rem; font-weight: 600; color: %(text)s; min-width: 280px; flex-shrink: 0; line-height: 1.35 }
.ref-desc { font-size: 0.72rem; color: %(text2)s; line-height: 1.5 }
.foot { text-align: center; padding: 20px 28px 26px }
.foot-txt { font-size: 0.64rem; color: %(light)s; letter-spacing: 0.35px }
.js-plotly-plot .plotly .modebar { display: none !important }
.Select-control { border-radius: 7px !important; border-color: rgba(15,23,42,0.09) !important; min-height: 30px !important; transition: all 0.15s ease }
.Select-control:hover { border-color: %(primary)s !important }
.is-focused .Select-control { border-color: %(primary)s !important; box-shadow: 0 0 0 3px %(accent_light)s !important }
.Select-menu-outer { border-radius: 9px !important; box-shadow: 0 6px 24px rgba(15,23,42,0.12), 0 2px 6px rgba(15,23,42,0.06) !important; border: 1px solid rgba(15,23,42,0.06) !important; margin-top: 4px !important; overflow: hidden }
.Select-option { font-size: 0.64rem !important; padding: 7px 11px !important; transition: background 0.1s ease }
.Select-option.is-focused { background: %(accent_light)s !important }
.Select-option.is-selected { background: %(primary)s !important; color: #fff !important }
.dash-spinner { margin: 30px auto !important }
@media (max-width: 1320px) { .exec-grid { grid-template-columns: repeat(4, 1fr) } }
@media (max-width: 1100px) { .exec-grid { grid-template-columns: repeat(3, 1fr) } }
@media (max-width: 992px) {
    .main { padding: 10px 16px }
    .idd-m { width: 300px !important }
    .idd-m2 { width: 240px !important }
    .idd-t { width: 118px !important; min-width: 118px !important }
    .exec-grid { grid-template-columns: repeat(3, 1fr); gap: 8px }
    .exec-card { padding: 10px 11px }
    .dr { flex-wrap: wrap }
}
@media (max-width: 768px) {
    .main { padding: 8px 10px }
    .idd-m, .idd-m2 { width: 100%% !important }
    .toolbar { flex-wrap: wrap; gap: 6px }
    .ch { flex-wrap: wrap; gap: 6px }
    .ref-name { min-width: 140px }
    .ref-row { flex-direction: column; gap: 4px }
    .export-btn { font-size: 0.6rem; padding: 4px 10px }
    .hdr { padding: 8px 14px }
    .det-cat-hdr { flex-wrap: wrap }
    .exec-grid { grid-template-columns: repeat(2, 1fr); gap: 7px }
    .exec-banner-inner { padding: 10px 12px }
    .exec-val { font-size: 1.15rem }
    .dright { gap: 6px }
    .dspark { display: none }
    .ddelta { min-width: 40px }
}
""" % c
        return '<!DOCTYPE html>\n<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}<style>' + css + '</style></head>\n<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'


def build_error_dashboard(title, message, missing_banks=None):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
    app.title = "JPMorgan Chase — SIB Dashboard Data Unavailable"
    sub_blocks = []
    if missing_banks:
        sub_blocks.append(html.Div([
            html.Div("Banks affected:", style={
                'fontSize': '12px', 'fontWeight': '700', 'color': CS['text2'],
                'textTransform': 'uppercase', 'letterSpacing': '0.6px',
                'marginBottom': '8px', 'marginTop': '20px'}),
            html.Ul([html.Li(b, style={'color': CS['text'], 'marginBottom': '4px'})
                     for b in sorted(missing_banks)],
                    style={'paddingLeft': '20px', 'margin': '0'})
        ]))
    app.layout = html.Div(
        style={'fontFamily': "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
               'background': CS['bg'], 'minHeight': '100vh', 'padding': '60px 24px'},
        children=[html.Div(
            style={'maxWidth': '720px', 'margin': '0 auto', 'background': '#fff',
                   'borderRadius': '14px', 'padding': '36px 40px',
                   'boxShadow': '0 4px 16px rgba(15,23,42,0.06), 0 1px 3px rgba(15,23,42,0.04)',
                   'borderLeft': f'4px solid {CS["bad"]}'},
            children=[
                html.Div("\u26a0", style={'fontSize': '34px', 'color': CS['bad'], 'marginBottom': '8px', 'lineHeight': '1'}),
                html.H2(title, style={'color': CS['text'], 'fontWeight': '700', 'marginBottom': '14px', 'fontSize': '22px'}),
                html.P(message, style={'color': CS['text2'], 'lineHeight': '1.6', 'whiteSpace': 'pre-wrap', 'fontSize': '14px'}),
                *sub_blocks,
                html.Hr(style={'margin': '24px 0', 'borderColor': CS['border']}),
                html.P([
                    html.Strong("Real-data-only dashboard.", style={'color': CS['text']}),
                    " No synthetic, sample, or fallback data will be substituted if the FDIC API "
                    "is unavailable. Refresh after verifying connectivity to retry."
                ], style={'color': CS['text3'], 'fontSize': '13px', 'lineHeight': '1.55'}),
            ])])
    return app


def main():
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    try:
        df = BankDataService().get_metrics_data()
    except FDICDataUnavailableError as e:
        logger.error(f"FDIC data unavailable (pid={os.getpid()}): {e}")
        return build_error_dashboard("FDIC Data Unavailable",
                                     f"The dashboard could not retrieve data from the FDIC BankFind API.\n\n{e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching FDIC data (pid={os.getpid()}): {type(e).__name__}: {e}")
        return build_error_dashboard("Unexpected Error",
                                     f"An unexpected error occurred while fetching FDIC data.\n\n{type(e).__name__}: {e}")
    if df.empty:
        return build_error_dashboard("No Data Returned",
                                     "The FDIC BankFind API responded but returned no usable data for the "
                                     "configured peer set. Check the cache directory and bank configuration, then retry.")
    expected_banks = {b['display'] for b in BANK_INFO}
    actual_banks = set(df['Bank'].unique())
    missing_banks = expected_banks - actual_banks
    logger.info(f"Dashboard ready (pid={os.getpid()}): {len(actual_banks)} of {len(expected_banks)} banks loaded. "
                f"Active: {sorted(actual_banks)}" + (f". MISSING: {sorted(missing_banks)}" if missing_banks else ""))
    if missing_banks:
        logger.warning(f"FDIC data missing for: {sorted(missing_banks)}. Absent from dashboard until next "
                       f"successful fetch. (Real data only by design.)")
    return DashboardBuilder(df, missing_banks=missing_banks).create_dashboard()


app = main()
server = app.server

if __name__ == "__main__":
    app.run_server(debug=False)
