import pandas as pd
import os
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter
import numpy as np
from dotenv import load_dotenv
import concurrent.futures
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

from utils import generate_pdf

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Period):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

class MultiUserFinancialAnalyzer:
    """
    Enhanced Multi-User Financial Analysis System
    Processes UPI transaction data for multiple users simultaneously
    """
    
    def __init__(self, openai_api_key: str = None):
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=4000)
        
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate UPI transaction data"""
        try:
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
                
            print(f"Loaded dataset with {len(df)} transactions")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Unique users: {df['user_id'].nunique() if 'user_id' in df.columns else 'No user_id column'}")
            
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def preprocess_upi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess UPI transaction data according to schema"""
        processed_df = df.copy()
        
        # Standardize column names
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
        
        # Apply mapping only for existing columns
        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        processed_df = processed_df.rename(columns=existing_mapping)
        
        # Data cleaning
        if 'Date' in processed_df.columns:
            processed_df['Date'] = pd.to_datetime(processed_df['Date'], errors='coerce')
        
        if 'Amount' in processed_df.columns:
            processed_df['Amount'] = pd.to_numeric(processed_df['Amount'], errors='coerce')
            processed_df = processed_df[processed_df['Amount'] > 0]
        
        # Filter only debit transactions (money spent)
        if 'Direction' in processed_df.columns:
            processed_df = processed_df[processed_df['Direction'].str.lower() == 'debit']
        
        # Filter only successful transactions
        if 'Status' in processed_df.columns:
            processed_df = processed_df[processed_df['Status'].str.lower() == 'success']
        
        # Clean category data
        if 'Category' not in processed_df.columns and 'Description' in processed_df.columns:
            processed_df['Category'] = processed_df['Description'].apply(self.smart_categorize_upi)
        
        return processed_df.dropna(subset=['Date', 'Amount', 'UserID'])
    
    def smart_categorize_upi(self, description):
        """Enhanced UPI transaction categorization"""
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
    
    def get_users_list(self, df: pd.DataFrame) -> List[str]:
        """Get list of unique users from the dataset"""
        return df['UserID'].unique().tolist()
    
    def filter_user_data(self, df: pd.DataFrame, user_id: str) -> pd.DataFrame:
        """Filter data for a specific user"""
        user_df = df[df['UserID'] == user_id].copy()
        print(f"User {user_id}: {len(user_df)} transactions")
        return user_df
    
    def create_agents(self):
        """Create the multi-agent system for financial analysis"""
        
        # Profile Builder Agent
        profile_builder = Agent(
            role='UPI Financial Profile Builder',
            goal="""
            Analyze UPI transaction patterns to infer user's income class, life stage, 
            spending personality, and financial behavior profile from digital payment habits.
            Focus on UPI-specific patterns like merchant preferences, digital adoption, and spending frequency.
            """,
            backstory="""
            You are a UPI and digital payments expert with deep knowledge of Indian fintech ecosystem.
            You understand spending patterns across different income segments, age groups, and cities in India.
            You can identify lifestyle indicators from UPI merchant data and transaction frequencies.
            """,
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        # Trend Analyzer Agent
        trend_analyzer = Agent(
            role='UPI Spending Trend Analyzer',
            goal="""
            Identify spending trends, seasonal patterns, merchant loyalty, day-by-day and 
            month-on-month changes in UPI transactions. Detect peak expense periods and 
            anomalous spending behaviors specific to digital payments.
            """,
            backstory="""
            You are a digital payments analytics expert specializing in UPI transaction patterns.
            You excel at identifying spending cycles, merchant preferences, and digital payment trends.
            You understand how Indians use UPI for different types of transactions throughout the month.
            """,
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        # Budgeting Expert Agent
        budgeting_expert = Agent(
            role='Digital Payment Budgeting Expert',
            goal="""
            Create personalized budget allocation proposals based on UPI spending patterns,
            considering Indian cost structures and digital payment habits. Provide specific
            recommendations for optimizing digital spending and building savings.
            """,
            backstory="""
            You are a personal finance expert specializing in digital payment optimization
            and budgeting for Indian consumers. You understand regional cost variations,
            UPI merchant categories, and evidence-based budgeting for digital-native users.
            """,
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        # Insight Generator Agent
        insight_generator = Agent(
            role='UPI Financial Insights Synthesizer',
            goal="""
            Synthesize all UPI analysis into one coherent, personalized, actionable 
            financial report. Focus on digital payment optimization, merchant loyalty 
            programs, and UPI-specific financial recommendations.
            """,
            backstory="""
            You are a senior fintech advisor who specializes in translating UPI transaction
            analysis into clear, actionable insights. You excel at creating personalized
            recommendations for digital payment optimization and financial health improvement.
            """,
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        return profile_builder, trend_analyzer, budgeting_expert, insight_generator
    
    def analyze_single_user(self, user_data: pd.DataFrame, user_id: str) -> Dict[str, Any]:
        """Analyze financial data for a single user"""
        
        if len(user_data) < 5:  # Skip users with very few transactions
            return {
                'user_id': user_id,
                'status': 'insufficient_data',
                'message': f'User {user_id} has only {len(user_data)} transactions - insufficient for analysis'
            }
        
        try:
            # Generate analysis context
            analysis_context = self.generate_user_analysis_context(user_data, user_id)
            
            # Create agents
            profile_builder, trend_analyzer, budgeting_expert, insight_generator = self.create_agents()
            
            # Create tasks
            tasks = self.create_analysis_tasks(analysis_context, user_id, 
                                            profile_builder, trend_analyzer, 
                                            budgeting_expert, insight_generator)
            
            # Execute analysis
            crew = Crew(
                agents=[profile_builder, trend_analyzer, budgeting_expert, insight_generator],
                tasks=tasks,
                verbose=True,
                process=Process.sequential
            )
            
            result = crew.kickoff()
            final_report = result.raw if hasattr(result, 'raw') else str(result)
            
            # Generate visualizations
            chart_path = self.create_user_visualizations(user_data, user_id)
            
            # Prepare result package
            result_package = {
                'user_id': user_id,
                'status': 'success',
                'executive_summary': self.extract_executive_summary(final_report),
                'comprehensive_narrative': final_report,  # Fixed: using 'comprehensive_narrative'
                'financial_health_score': float(self.extract_health_score(final_report)),
                'key_recommendations': self.extract_recommendations(final_report),
                'data_insights': convert_to_json_serializable(analysis_context),
                'chart_path': chart_path,
                'summary': {
                    'total_transactions': int(len(user_data)),
                    'total_spending': float(user_data['Amount'].sum()),
                    'date_range': f"{user_data['Date'].min().strftime('%Y-%m-%d')} to {user_data['Date'].max().strftime('%Y-%m-%d')}",
                    'top_category': str(user_data.groupby('Category')['Amount'].sum().idxmax()),
                    'average_transaction': float(user_data['Amount'].mean()),
                    'most_frequent_merchant': str(user_data['Description'].mode().iloc[0] if not user_data['Description'].mode().empty else 'N/A')
                }
            }
            
            # Generate PDF report
            pdf_path = self.generate_user_pdf_report(result_package, user_data)
            result_package['pdf_path'] = pdf_path
            
            return result_package
            
        except Exception as e:
            print(f"Error analyzing user {user_id}: {str(e)}")
            return {
                'user_id': user_id,
                'status': 'error',
                'message': f'Analysis failed for user {user_id}: {str(e)}'
            }
    
    def generate_user_analysis_context(self, user_data: pd.DataFrame, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive analysis context for a user"""
        
        total_spending = float(user_data['Amount'].sum())
        
        # Category analysis
        category_analysis = user_data.groupby('Category')['Amount'].agg(['sum', 'count', 'mean']).to_dict('index')
        category_analysis = {cat: {
            'sum': float(data['sum']), 
            'count': int(data['count']), 
            'mean': float(data['mean'])
        } for cat, data in category_analysis.items()}
        
        # Time series analysis
        monthly_spending = user_data.groupby(user_data['Date'].dt.to_period('M'))['Amount'].sum()
        monthly_analysis = {str(period): float(amount) for period, amount in monthly_spending.items()}
        
        # Merchant analysis
        top_merchants = user_data.groupby('Description')['Amount'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(10)
        merchant_analysis = {merchant: {
            'total_spent': float(data['sum']),
            'transaction_count': int(data['count'])
        } for merchant, data in top_merchants.to_dict('index').items()}
        
        # Spending patterns
        daily_patterns = user_data.groupby(user_data['Date'].dt.day_name())['Amount'].sum().to_dict()
        daily_patterns = {day: float(amount) for day, amount in daily_patterns.items()}
        
        # City-wise spending (if available)
        city_analysis = {}
        if 'City' in user_data.columns:
            city_spending = user_data.groupby('City')['Amount'].sum().to_dict()
            city_analysis = {city: float(amount) for city, amount in city_spending.items()}
        
        return {
            'user_id': user_id,
            'summary': {
                'total_transactions': int(len(user_data)),
                'total_spending': total_spending,
                'average_transaction': float(user_data['Amount'].mean()),
                'date_range': f"{user_data['Date'].min().strftime('%Y-%m-%d')} to {user_data['Date'].max().strftime('%Y-%m-%d')}",
                'unique_merchants': int(user_data['Description'].nunique()),
                'unique_categories': int(user_data['Category'].nunique())
            },
            'category_analysis': category_analysis,
            'monthly_analysis': monthly_analysis,
            'merchant_analysis': merchant_analysis,
            'daily_patterns': daily_patterns,
            'city_analysis': city_analysis,
            'spending_breakdown': {
                cat: float(amount/total_spending*100) 
                for cat, amount in user_data.groupby('Category')['Amount'].sum().items()
            }
        }
    
    def create_analysis_tasks(self, analysis_context, user_id, profile_builder, trend_analyzer, budgeting_expert, insight_generator):
        """Create analysis tasks for the user"""
        
        # Profile Building Task
        profile_task = Task(
            description=f"""
            Analyze UPI transaction patterns for User {user_id}:
            
            USER SUMMARY:
            {json.dumps(analysis_context['summary'], indent=2)}
            
            SPENDING PATTERNS:
            {json.dumps(analysis_context['category_analysis'], indent=2)}
            
            MERCHANT PREFERENCES:
            {json.dumps(analysis_context['merchant_analysis'], indent=2)}
            
            Determine:
            1. Income class estimation (based on spending volume and patterns)
            2. Life stage indicators (student, working professional, family person)
            3. Digital payment behavior (heavy user, moderate, occasional)
            4. Lifestyle indicators from merchant preferences
            5. Financial maturity level
            6. City/location influence on spending (if available)
            
            Provide detailed reasoning with specific evidence from UPI transaction data.
            """,
            agent=profile_builder,
            expected_output="Comprehensive user profile with income estimation and lifestyle analysis"
        )
        
        # Trend Analysis Task
        trend_task = Task(
            description=f"""
            Analyze spending trends for User {user_id}:
            
            MONTHLY TRENDS:
            {json.dumps(analysis_context['monthly_analysis'], indent=2)}
            
            DAILY PATTERNS:
            {json.dumps(analysis_context['daily_patterns'], indent=2)}
            
            CATEGORY BREAKDOWN:
            {json.dumps(analysis_context['spending_breakdown'], indent=2)}
            
            Analyze:
            1. Month-over-month spending changes
            2. Day-of-week spending patterns
            3. Peak spending periods
            4. Category-wise growth/decline trends
            5. Merchant loyalty patterns
            6. Seasonal variations in UPI usage
            7. Predictive insights for next month
            
            Identify top 3 most significant trends with quantified impacts.
            """,
            agent=trend_analyzer,
            expected_output="Detailed trend analysis with MoM changes and predictive insights"
        )
        
        # Budget Planning Task
        budget_task = Task(
            description=f"""
            Create personalized budget plan for User {user_id}:
            
            CURRENT SPENDING ANALYSIS:
            {json.dumps(analysis_context['spending_breakdown'], indent=2)}
            
            MONTHLY AVERAGE: Rs.{analysis_context['summary']['total_spending']/max(len(analysis_context['monthly_analysis']), 1):.0f}
            
            TOP MERCHANTS:
            {json.dumps(analysis_context['merchant_analysis'], indent=2)}
            
            Create budget recommendations:
            1. Suggested monthly budget by category
            2. Essential vs discretionary spending allocation
            3. Savings target recommendations
            4. UPI spending optimization tips
            5. Merchant-specific spending limits
            6. Emergency fund recommendations
            
            Provide specific rupee amounts and percentages for each category.
            """,
            agent=budgeting_expert,
            expected_output="Personalized budget plan with specific allocations and UPI optimization tips"
        )
        
        # Final Synthesis Task
        synthesis_task = Task(
            description=f"""
            Create comprehensive financial report for User {user_id}:
            
            Synthesize all previous analysis into ONE personalized UPI financial report:
            
            STRUCTURE THE REPORT AS:
            1. EXECUTIVE SUMMARY (2-3 sentences about user's financial profile)
            2. UPI SPENDING SNAPSHOT (key numbers and patterns)
            3. FINANCIAL PROFILE (income class, life stage, digital behavior)
            4. SPENDING TRENDS (top 3-5 insights with specific numbers)
            5. BUDGET RECOMMENDATIONS (specific monthly allocations)
            6. ACTIONABLE TIPS (3-5 specific UPI optimization recommendations)
            7. FINANCIAL HEALTH SCORE (1-10 with explanation)
            
            Requirements:
            - Use clear, jargon-free language
            - Include specific rupee amounts and percentages
            - Focus on UPI-specific insights and recommendations
            - Make recommendations immediately actionable
            - Maintain encouraging but realistic tone
            - Structure as single, coherent narrative
            
            This is the FINAL consolidated output for User {user_id}.
            """,
            agent=insight_generator,
            expected_output="Single comprehensive UPI financial report with all insights and recommendations"
        )
        
        return [profile_task, trend_task, budget_task, synthesis_task]
    
    def create_user_visualizations(self, user_data: pd.DataFrame, user_id: str) -> str:
        """Create visualization for individual user"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'UPI Spending Analysis - User {user_id}', fontsize=16, fontweight='bold')
            
            # Category spending pie chart
            category_spending = user_data.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(8)
            if not category_spending.empty:
                ax1.pie(category_spending.values, labels=category_spending.index, autopct='%1.1f%%')
                ax1.set_title('Spending by Category', fontweight='bold')
            
            # Monthly trend
            monthly_spending = user_data.groupby(user_data['Date'].dt.to_period('M'))['Amount'].sum()
            if len(monthly_spending) > 1:
                monthly_spending.plot(kind='line', ax=ax2, marker='o', color='green')
                ax2.set_title('Monthly Spending Trend', fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
            else:
                ax2.text(0.5, 0.5, 'Insufficient data\nfor monthly trend', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Monthly Spending Trend', fontweight='bold')
            
            # Daily pattern
            daily_spending = user_data.groupby(user_data['Date'].dt.day_name())['Amount'].sum()
            if not daily_spending.empty:
                daily_spending.plot(kind='bar', ax=ax3, color='orange')
                ax3.set_title('Spending by Day of Week', fontweight='bold')
                ax3.tick_params(axis='x', rotation=45)
            
            # Top merchants
            top_merchants = user_data.groupby('Description')['Amount'].sum().sort_values(ascending=False).head(6)
            if not top_merchants.empty:
                top_merchants.plot(kind='bar', ax=ax4, color='purple')
                ax4.set_title('Top Merchants by Spending', fontweight='bold')
                ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            chart_path = f"user_{user_id}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
        except Exception as e:
            print(f"Error creating visualization for user {user_id}: {str(e)}")
            return None
    
    def extract_executive_summary(self, report_text: str) -> str:
        """Extract executive summary from report"""
        lines = report_text.split('\n')
        for i, line in enumerate(lines):
            if 'executive summary' in line.lower() or 'summary' in line.lower():
                summary_lines = []
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip() and not lines[j].strip().startswith(('#', '##', '###')):
                        summary_lines.append(lines[j].strip())
                return ' '.join(summary_lines) if summary_lines else "UPI financial analysis completed successfully."
        
        paragraphs = [p.strip() for p in report_text.split('\n\n') if p.strip()]
        return paragraphs[0] if paragraphs else "UPI financial analysis completed successfully."
    
    def extract_health_score(self, report_text: str) -> float:
        """Extract financial health score from report"""
        score_match = re.search(r'(?:health score|score)[:\s]*([0-9](?:\.[0-9])?)\s*(?:/\s*10)?', report_text.lower())
        return float(score_match.group(1)) if score_match else 7.5
    
    def extract_recommendations(self, report_text: str) -> List[str]:
        """Extract key recommendations from report"""
        recommendations = []
        lines = report_text.split('\n')
        
        in_recommendations = False
        for line in lines:
            if any(word in line.lower() for word in ['recommendation', 'tip', 'action', 'suggest', 'advice']):
                in_recommendations = True
                continue
            
            if in_recommendations and line.strip():
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•', '*')) or line.strip().startswith('•'):
                    clean_rec = line.strip().lstrip('1234567890.-•* ')
                    if clean_rec:
                        recommendations.append(clean_rec)
                elif len(recommendations) >= 5:
                    break
        
        return recommendations[:5] if recommendations else [
            "Set up monthly UPI spending budgets for each category",
            "Monitor your top 3 spending merchants regularly",
            "Use UPI recurring payments for bills to avoid late fees",
            "Review and optimize your largest expense categories",
            "Build an emergency fund using automated UPI transfers"
        ]
    
    def generate_user_pdf_report(self, result_package: Dict[str, Any], user_data: pd.DataFrame) -> str:
        """Generate PDF report for individual user - Fixed function"""
        try:
            # Prepare data for PDF generation
            major_spends = user_data.nlargest(10, 'Amount')[['Date', 'Description', 'Amount', 'Category']]
            category_summary = user_data.groupby('Category')['Amount'].agg(['sum', 'count', 'mean']).round(2)
            category_summary.columns = ['Total', 'Count', 'Average']
            category_summary['Percentage'] = (category_summary['Total'] / category_summary['Total'].sum() * 100).round(1)
            
            monthly_summary = user_data.groupby(user_data['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
            monthly_summary.columns = ['Month', 'Amount']
            
            # Prepare summary dictionary with all required fields - FIXED
            summary_dict = {
                'Total Spending': f"Rs.{result_package['summary']['total_spending']:,.2f}",
                'Total Transactions': str(result_package['summary']['total_transactions']),
                'Average Transaction': f"Rs.{result_package['summary']['average_transaction']:,.2f}",
                'Top Category': result_package['summary']['top_category'],
                'Most Frequent Category': result_package['summary']['top_category'],  # Using same as top category
                'Date Range': result_package['summary']['date_range']
            }
            
            # Create result dict for PDF generation - FIXED
            pdf_result = {
                'summary': summary_dict,
                'comprehensive_narrative': result_package['comprehensive_narrative'],  # Fixed key name
                'chart_path': result_package.get('chart_path')
            }
            
            # Generate PDF using the utils function
            pdf_path = generate_pdf(
                pdf_result, 
                user_data.head(20), 
                major_spends, 
                category_summary, 
                monthly_summary
            )
            
            return pdf_path
            
        except Exception as e:
            print(f"Error generating PDF for user {result_package['user_id']}: {str(e)}")
            return None
    
    def process_multiple_users(self, file_path: str, max_users: int = None, parallel: bool = True) -> Dict[str, Any]:
        """Process multiple users from UPI dataset"""
        
        print("Loading and preprocessing data...")
        df = self.load_and_validate_data(file_path)
        processed_df = self.preprocess_upi_data(df)
        
        users = self.get_users_list(processed_df)
        if max_users:
            users = users[:max_users]
        
        print(f"Processing {len(users)} users...")
        
        results = {}
        
        if parallel and len(users) > 1:
            # Parallel processing for multiple users
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_user = {
                    executor.submit(self.analyze_single_user, 
                                  self.filter_user_data(processed_df, user_id), 
                                  user_id): user_id 
                    for user_id in users
                }
                
                for future in concurrent.futures.as_completed(future_to_user):
                    user_id = future_to_user[future]
                    try:
                        result = future.result()
                        results[user_id] = result
                        print(f"Completed analysis for user {user_id}")
                    except Exception as e:
                        print(f"Error processing user {user_id}: {str(e)}")
                        results[user_id] = {
                            'user_id': user_id,
                            'status': 'error',
                            'message': str(e)
                        }
        else:
            # Sequential processing
            for user_id in users:
                try:
                    user_data = self.filter_user_data(processed_df, user_id)
                    result = self.analyze_single_user(user_data, user_id)
                    results[user_id] = result
                    print(f"Completed analysis for user {user_id}")
                except Exception as e:
                    print(f"Error processing user {user_id}: {str(e)}")
                    results[user_id] = {
                        'user_id': user_id,
                        'status': 'error',
                        'message': str(e)
                    }
        
        # Generate summary report
        summary_report = self.generate_summary_report(results, processed_df)
        
        return {
            'overall_summary': summary_report,
            'user_results': results,
            'total_users_processed': len(users),
            'successful_analyses': len([r for r in results.values() if r.get('status') == 'success']),
            'failed_analyses': len([r for r in results.values() if r.get('status') == 'error']),
            'insufficient_data': len([r for r in results.values() if r.get('status') == 'insufficient_data'])
        }
    
    def generate_summary_report(self, results: Dict[str, Any], processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate overall summary report for all users"""
        
        successful_results = [r for r in results.values() if r.get('status') == 'success']
        
        if not successful_results:
            return {
                'message': 'No successful analyses to summarize',
                'total_users': len(results),
                'processed_successfully': 0
            }
        
        # Aggregate statistics
        total_transactions = sum([r['summary']['total_transactions'] for r in successful_results])
        total_spending = sum([r['summary']['total_spending'] for r in successful_results])
        avg_health_score = sum([r['financial_health_score'] for r in successful_results]) / len(successful_results)
        
        # Most common categories across all users
        all_categories = []
        for r in successful_results:
            for category, percentage in r['data_insights']['spending_breakdown'].items():
                all_categories.append(category)
        
        category_frequency = Counter(all_categories)
        top_categories = category_frequency.most_common(5)
        
        # Average spending by category
        category_totals = {}
        category_counts = {}
        for r in successful_results:
            for category, percentage in r['data_insights']['spending_breakdown'].items():
                user_spending = r['summary']['total_spending']
                category_amount = (percentage / 100) * user_spending
                if category not in category_totals:
                    category_totals[category] = 0
                    category_counts[category] = 0
                category_totals[category] += category_amount
                category_counts[category] += 1
        
        avg_category_spending = {
            cat: category_totals[cat] / category_counts[cat] 
            for cat in category_totals.keys()
        }
        
        return {
            'total_users_analyzed': len(successful_results),
            'total_transactions': total_transactions,
            'total_spending_across_users': total_spending,
            'average_spending_per_user': total_spending / len(successful_results),
            'average_financial_health_score': round(avg_health_score, 2),
            'most_common_categories': top_categories,
            'average_spending_by_category': avg_category_spending,
            'user_distribution': {
                'high_spenders': len([r for r in successful_results if r['summary']['total_spending'] > 50000]),
                'medium_spenders': len([r for r in successful_results if 20000 <= r['summary']['total_spending'] <= 50000]),
                'low_spenders': len([r for r in successful_results if r['summary']['total_spending'] < 20000])
            }
        }


def main():
    """Main function to demonstrate the multi-user financial analysis system"""
    
    # Initialize the analyzer
    analyzer = MultiUserFinancialAnalyzer()
    
    # Example usage
    file_path = "D:/ojje_ad/upi_transactions.xlsx"  # Replace with your file path
    
    try:
        # Process multiple users (limit to 5 for demo, remove limit for full processing)
        results = analyzer.process_multiple_users(file_path, max_users=5, parallel=True)
        
        print("\n" + "="*60)
        print("MULTI-USER FINANCIAL ANALYSIS COMPLETE")
        print("="*60)
        
        print(f"Total Users Processed: {results['total_users_processed']}")
        print(f"Successful Analyses: {results['successful_analyses']}")
        print(f"Failed Analyses: {results['failed_analyses']}")
        print(f"Insufficient Data: {results['insufficient_data']}")
        
        print("\nOVERALL SUMMARY:")
        summary = results['overall_summary']
        if 'total_users_analyzed' in summary:
            print(f"- Average spending per user: Rs.{summary['average_spending_per_user']:,.2f}")
            print(f"- Average financial health score: {summary['average_financial_health_score']}/10")
            print(f"- Most common spending categories: {[cat[0] for cat in summary['most_common_categories'][:3]]}")
        
        print("\nINDIVIDUAL USER RESULTS:")
        for user_id, result in results['user_results'].items():
            if result['status'] == 'success':
                print(f"\nUser {user_id}:")
                print(f"  - Total Spending: Rs.{result['summary']['total_spending']:,.2f}")
                print(f"  - Health Score: {result['financial_health_score']}/10")
                print(f"  - Top Category: {result['summary']['top_category']}")
                print(f"  - PDF Report: {result.get('pdf_path', 'Not generated')}")
            else:
                print(f"\nUser {user_id}: {result['status']} - {result.get('message', 'Unknown error')}")
        
        # Save results to JSON
        output_file = f"multi_user_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()