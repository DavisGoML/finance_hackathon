import pandas as pd
import numpy as np
from datetime import datetime
import json
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from fpdf import FPDF

class BudgetAdvisor:
    def __init__(self, model="gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0.1, max_tokens=2000)

    def preprocess_upi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess UPI transaction data (same logic as crew.py)"""
        processed_df = df.copy()
        column_mapping = {
            'timestamp': 'Date',
            'amount': 'Amount',
            'merchant_name': 'Description',
            'category': 'Category',
            'direction': 'Direction',
            'user_id': 'UserID',
            'city': 'City',
            'transaction_type': 'TransactionType',
            'status': 'Status',
            'note': 'Note'
        }
        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        processed_df = processed_df.rename(columns=existing_mapping)
        if 'Date' in processed_df.columns:
            processed_df['Date'] = pd.to_datetime(processed_df['Date'], errors='coerce')
        if 'Amount' in processed_df.columns:
            processed_df['Amount'] = pd.to_numeric(processed_df['Amount'], errors='coerce')
            processed_df = processed_df[processed_df['Amount'] > 0]
        if 'Direction' in processed_df.columns:
            processed_df = processed_df[processed_df['Direction'].str.lower() == 'debit']
        if 'Status' in processed_df.columns:
            processed_df = processed_df[processed_df['Status'].str.lower() == 'success']
        if 'Category' not in processed_df.columns and 'Description' in processed_df.columns:
            processed_df['Category'] = processed_df['Description'].apply(self.smart_categorize_upi)
        return processed_df.dropna(subset=['Date', 'Amount'])

    def smart_categorize_upi(self, description):
        """Enhanced UPI transaction categorization (same as crew.py)"""
        desc = str(description).lower()
        categories = {
            'Food & Dining': [
                'zomato', 'swiggy', 'uber eats', 'food panda', 'dominos', 'pizza hut', 
                'mcdonalds', 'kfc', 'restaurant', 'cafe', 'food', 'dining', 'meal',
                'grocery', 'supermarket', 'big bazaar', 'dmart', 'reliance fresh'
            ],
            'Transportation': [
                'uber', 'ola', 'rapido', 'taxi', 'auto', 'bus', 'metro', 'irctc',
                'petrol', 'fuel', 'gas', 'parking', 'toll', 'transport'
            ],
            'Shopping': [
                'amazon', 'flipkart', 'myntra', 'ajio', 'nykaa', 'shopping',
                'mall', 'retail', 'store', 'purchase', 'buy', 'myntra', 'meesho'
            ],
            'Bills & Utilities': [
                'electricity', 'water', 'gas cylinder', 'internet', 'broadband',
                'mobile', 'phone', 'recharge', 'bill', 'utility', 'bsnl', 'airtel',
                'jio', 'vi', 'vodafone'
            ],
            'Entertainment': [
                'netflix', 'amazon prime', 'hotstar', 'spotify', 'youtube',
                'movie', 'cinema', 'pvr', 'inox', 'game', 'entertainment',
                'subscription', 'music', 'book my show'
            ],
            'Healthcare': [
                'hospital', 'doctor', 'medical', 'pharmacy', 'medicine',
                'health', 'clinic', 'apollo', '1mg', 'pharmeasy', 'netmeds'
            ],
            'Financial Services': [
                'bank', 'atm', 'loan', 'emi', 'insurance', 'investment',
                'mutual fund', 'sip', 'credit card', 'paytm', 'phonepe', 'gpay'
            ],
            'Education': [
                'school', 'college', 'university', 'course', 'training',
                'education', 'book', 'byju', 'unacademy', 'upgrad'
            ],
            'Travel': [
                'hotel', 'flight', 'train', 'bus booking', 'oyo', 'makemytrip',
                'goibibo', 'yatra', 'travel', 'booking', 'holiday'
            ],
            'Personal Care': [
                'salon', 'spa', 'beauty', 'grooming', 'urban company',
                'personal care', 'cosmetics'
            ]
        }
        for category, keywords in categories.items():
            if any(keyword in desc for keyword in keywords):
                return category
        return 'Other'

    def suggest_budget(
        self,
        file_path: str,
        monthly_income: float,
        requirements: list,
        duration_months: int,
        output_path: str = None
    ) -> str:
        """
        file_path: path to the user's UPI data file (.csv or .xlsx)
        requirements: list of dicts, e.g. [{"name": "iPhone", "price": 80000}, ...]
        duration_months: int, number of months to plan for
        output_path: optional, path to save the output as .txt or .pdf
        """
        # Load and preprocess data
        if file_path.lower().endswith('.csv'):
            upi_data = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.xls', '.xlsx')):
            upi_data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Please provide a .csv or .xlsx file.")

        upi_data = self.preprocess_upi_data(upi_data)
        # Prepare context for the agent
        upi_summary = {
            "total_spending": float(upi_data['Amount'].sum()),
            "average_transaction": float(upi_data['Amount'].mean()),
            "top_categories": upi_data.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(3).to_dict()
        }

        agent = Agent(
            role="Personal Budgeting Advisor",
            goal="""
            Given a user's UPI spending data, monthly income, and a list of financial goals (with prices), 
            create a personalized budget plan. Advise how much to save monthly, how to allocate spending, 
            and estimate how long it will take to reach each goal. Suggest practical saving tips and 
            spending optimizations. Output should be clear, actionable, and tailored to the user's data.
            """,
            backstory="""
            You are an expert personal finance advisor. You help users plan their budgets, save for goals, 
            and optimize their spending based on real transaction data and income.
            """,
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

        task = Task(
            description=f"""
            User's monthly income: Rs.{monthly_income:,.2f}
            Savings duration: {duration_months} months

            UPI SPENDING SUMMARY:
            {json.dumps(upi_summary, indent=2)}

            USER GOALS:
            {json.dumps(requirements, indent=2)}

            Please:
            1. Suggest a monthly budget allocation (spending, saving, etc.)
            2. Advise how much to save each month to reach each goal within the duration
            3. If some goals are not feasible, explain why and suggest alternatives
            4. Give 3-5 practical saving/spending tips based on the UPI data
            5. Output should be clear, actionable, and easy to follow
            """,
            agent=agent,
            expected_output="Personalized budget plan with savings advice and timeline for each goal"
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()
        output_text = result.raw if hasattr(result, 'raw') else str(result)

        # Save output as text or PDF if output_path is provided
        if output_path:
            if output_path.lower().endswith('.txt'):
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output_text)
            elif output_path.lower().endswith('.pdf'):
                try:
              
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.set_font("Arial", size=12)
                    for line in output_text.split('\n'):
                        pdf.multi_cell(0, 10, line)
                    pdf.output(output_path)
                except ImportError:
                    print("fpdf not installed. Please install it with 'pip install fpdf' to enable PDF export.")
            else:
                raise ValueError("Unsupported output file type. Use .txt or .pdf")
        return output_text

if __name__ == "__main__":

    file_path = "D:/ojje_ad/upi_transactions.xlsx" 
    income = 35000
    goals = [
        {"name": "New Laptop", "price": 60000},
        {"name": "Vacation", "price": 25000}
    ]
    duration = 6

    advisor = BudgetAdvisor()
    plan = advisor.suggest_budget(file_path, income, goals, duration, output_path="budget_plan.txt")
    print(plan)
    # plan = advisor.suggest_budget(file_path, income, goals, duration, output_path="budget_plan.pdf")

