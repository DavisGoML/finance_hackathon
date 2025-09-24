# 🏗️ Agents Overview

In this system, we have used a **pandas and text-based mechanism as a fallback** for core data processing tasks like cleaning, transformation, and formatting.  
This ensures that even if an agent fails or produces incomplete results, the pipeline can still rely on deterministic pandas operations for stable handling of financial datasets.  

---

## 1. Generic Data Intelligence Agent
- **Role:** Universal Financial Data Intelligence Specialist  
- **Goal:** Detects schema of any dataset. Identifies key columns like:  
  - Dates, amounts, user IDs  
  - Status (success/failed), direction (debit/credit)  
  - Merchant/description fields  
  - Location, reference numbers, metadata  
- **Backstory:** 25+ years as a financial data archaeologist. Can interpret datasets from banks, wallets, crypto exchanges, UPI, etc.  

👉 **Output:** A structured JSON schema of the dataset.  

---

## 2. Schema Discovery
- **Uses:** The Generic Data Intelligence Agent.  
- **Process:** Examines sample rows, data types, null counts, unique values.  
- **Returns JSON with:**  
  - `date_col`, `amount_col`, `desc_col`, etc.  
  - Dataset type (single-user, multi-user, bank statement, credit card, etc.)  
  - Detected currency  
  - Data quality rating  

---

## 3. Preprocessing Agent
- **Process:**  
  - Applies discovered schema to clean and transform data.  
  - Maps raw columns → standardized ones (`Date`, `Amount`, `Description`, `UserID`, etc.).  
  - Cleans amounts, parses dates, removes invalid rows.  
  - Filters by:  
    - Transaction direction → only expenses/debits.  
    - Status → only successful/settled.  
  - Applies intelligent categorization (food, travel, shopping, etc.) using merchant keywords.  

👉 **Output:** A clean, standardized DataFrame ready for deeper analysis.  

---

## 4. Multi-Agent Analysis System
Created via `create_agents()`. It includes:  

### 🔹 Profile Builder Agent
- **Role:** Universal Financial Profile Builder  
- **Goal:** Build user’s financial profile → income class, life stage, spending personality.  
- **Backstory:** Expert in financial behavior across all payment systems.  

### 🔹 Trend Analyzer Agent
- **Role:** Universal Financial Trend Analyzer  
- **Goal:** Detect trends, seasonality, anomalies.  
- **Backstory:** Data scientist specializing in temporal financial patterns.  

### 🔹 Budgeting Expert Agent
- **Role:** Universal Budget & Financial Planning Expert  
- **Goal:** Suggest personalized budgeting strategies.  
- **Backstory:** Certified planner who adapts budgets to any financial system.  

### 🔹 Insight Generator Agent
- **Role:** Universal Financial Insights Synthesizer  
- **Goal:** Combine results from all agents into one actionable report.  
- **Backstory:** Senior financial advisor who translates data into recommendations.  
