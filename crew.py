import pandas as pd
import os
import json
import re
import matplotlib
matplotlib.use('Agg', force=True)

import matplotlib.pyplot as plt
plt.ioff()
plt.close('all')
            
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter
import numpy as np
from dotenv import load_dotenv
import concurrent.futures
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew, Process, LLM
import boto3

from utils import generate_pdf

load_dotenv()

# AWS Bedrock Configuration
# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY"
# AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# BEDROCK_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Period):
        return str(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def create_fallback_column_mapping(column_names, sample_data):
    """Create fallback column mapping when JSON parsing fails"""
    mapping = {
        "date_col": None,
        "desc_col": None,
        "amount_col": None,
        "user_id_col": None,
        "existing_category_col": None,
        "direction_col": None,
        "status_col": None,
        "has_existing_categories": False,
        "has_user_ids": False,
        "data_quality_notes": "Fallback mapping used",
        "recommended_categorization": "create_new"
    }
    
    lower_cols = [col.lower() for col in column_names]
    
    date_keywords = ['date', 'time', 'timestamp', 'transaction_date', 'txn_date']
    for i, col in enumerate(lower_cols):
        if any(keyword in col for keyword in date_keywords):
            mapping["date_col"] = column_names[i]
            break
    
    amount_keywords = ['amount', 'debit', 'withdrawal', 'expense', 'spend', 'value', 'cost']
    for i, col in enumerate(lower_cols):
        if any(keyword in col for keyword in amount_keywords):
            mapping["amount_col"] = column_names[i]
            break
    
    desc_keywords = ['description', 'desc', 'narration', 'details', 'transaction', 'merchant', 'payee']
    for i, col in enumerate(lower_cols):
        if any(keyword in col for keyword in desc_keywords):
            mapping["desc_col"] = column_names[i]
            break
    
    user_keywords = ['user_id', 'user', 'customer_id', 'account', 'userid']
    for i, col in enumerate(lower_cols):
        if any(keyword in col for keyword in user_keywords):
            mapping["user_id_col"] = column_names[i]
            mapping["has_user_ids"] = True
            break
    
    cat_keywords = ['category', 'type', 'class', 'group', 'classification']
    for i, col in enumerate(lower_cols):
        if any(keyword in col for keyword in cat_keywords):
            mapping["existing_category_col"] = column_names[i]
            mapping["has_existing_categories"] = True
            mapping["recommended_categorization"] = "use_existing"
            break
    
    direction_keywords = ['direction', 'debit', 'credit', 'type', 'transaction_type']
    for i, col in enumerate(lower_cols):
        if any(keyword in col for keyword in direction_keywords):
            mapping["direction_col"] = column_names[i]
            break
    
    status_keywords = ['status', 'state', 'success', 'failed']
    for i, col in enumerate(lower_cols):
        if any(keyword in col for keyword in status_keywords):
            mapping["status_col"] = column_names[i]
            break
    
    if not mapping["date_col"] and len(column_names) > 0:
        mapping["date_col"] = column_names[0]
    if not mapping["desc_col"] and len(column_names) > 1:
        mapping["desc_col"] = column_names[1]
    if not mapping["amount_col"] and len(column_names) > 2:
        mapping["amount_col"] = column_names[2]
    
    return mapping

class MultiUserFinancialAnalyzer:
    """Enhanced Multi-User Financial Analysis System with AWS Bedrock"""
    
    def __init__(self, 
                 aws_access_key_id: str = None,
                 aws_secret_access_key: str = None,
                 aws_region: str = None):
        """Initialize with AWS Bedrock credentials"""
        
        self.aws_access_key_id = aws_access_key_id or AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = aws_secret_access_key or AWS_SECRET_ACCESS_KEY
        self.aws_region = aws_region or AWS_REGION
        
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("AWS credentials not provided")
        
        # Set AWS credentials in environment
        os.environ['AWS_ACCESS_KEY_ID'] = self.aws_access_key_id
        os.environ['AWS_SECRET_ACCESS_KEY'] = self.aws_secret_access_key
        os.environ['AWS_REGION_NAME'] = self.aws_region
        
        # Use CrewAI's LLM wrapper with bedrock/ prefix
        self.llm = LLM(
            model=f"bedrock/{BEDROCK_MODEL_ID}",
            temperature=0.1,
            max_tokens=4000,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_region_name=self.aws_region
        )
        
        self.data_schema = None
        print(f"✅ AWS Bedrock initialized with Claude Sonnet 3.7")
        print(f"   Region: {self.aws_region}")
        print(f"   Model: {BEDROCK_MODEL_ID}")
    
    def create_generic_data_intelligence_agent(self):
        """Create the generic data intelligence agent"""
        return Agent(
            role='Universal Financial Data Intelligence Specialist',
            goal="""
            Analyze ANY financial dataset structure and intelligently identify key columns:
            - Date/timestamp columns
            - Amount/transaction value columns
            - Description/merchant/payee columns
            - User/customer/account identifier columns
            - Existing category/classification columns
            - Transaction direction columns
            - Status columns
            """,
            backstory="""
            You are a financial data archaeologist with 25+ years of experience analyzing 
            financial datasets from banks, wallets, exchanges, and payment processors worldwide.
            """,
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
    
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate financial transaction data"""
        try:
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
                
            print(f"Loaded dataset with {len(df)} transactions")
            print(f"Columns: {df.columns.tolist()}")
            
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def intelligent_data_schema_discovery(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Use AI agent to discover data schema"""
        
        sample_size = min(15, len(df))
        data_sample = df.head(sample_size).to_string()
        column_names = df.columns.tolist()
        
        column_types = {col: str(df[col].dtype) for col in df.columns}
        null_counts = {col: int(df[col].isnull().sum()) for col in df.columns}
        unique_counts = {col: int(df[col].nunique()) for col in df.columns}
        
        sample_values = {}
        for col in df.columns:
            sample_vals = df[col].dropna().head(5).tolist()
            sample_values[col] = [str(val) for val in sample_vals]
        
        data_agent = self.create_generic_data_intelligence_agent()
        
        schema_task = Task(
            description=f"""
            Analyze this financial dataset and discover its structure:
            
            DATASET OVERVIEW:
            - Total rows: {len(df)}
            - Total columns: {len(df.columns)}
            - Column names: {column_names}
            - Data types: {column_types}
            - Null counts: {null_counts}
            - Unique value counts: {unique_counts}
            - Sample values: {sample_values}
            
            SAMPLE DATA ({sample_size} rows):
            {data_sample}
            
            CRITICAL: Respond with ONLY a valid JSON object:
            
            {{
                "date_col": "exact_column_name_or_null",
                "amount_col": "exact_column_name_or_null",
                "desc_col": "exact_column_name_or_null",
                "user_id_col": "exact_column_name_or_null",
                "existing_category_col": "exact_column_name_or_null",
                "direction_col": "exact_column_name_or_null",
                "status_col": "exact_column_name_or_null",
                "location_col": "exact_column_name_or_null",
                "reference_col": "exact_column_name_or_null",
                "has_existing_categories": false,
                "has_user_ids": false,
                "has_direction_info": false,
                "has_status_info": false,
                "dataset_type": "single_user|multi_user|bank_statement|credit_card|upi|other",
                "currency_detected": "USD|INR|EUR|GBP|other|unknown",
                "data_quality": "excellent|good|fair|poor",
                "recommended_categorization": "create_new|use_existing|hybrid",
                "processing_complexity": "simple|moderate|complex",
                "special_notes": "any_important_observations"
            }}
            
            Use exact column names. RESPOND ONLY WITH JSON - NO OTHER TEXT.
            """,
            agent=data_agent,
            expected_output="Valid JSON object with complete data schema analysis"
        )
        
        try:
            crew = Crew(
                agents=[data_agent],
                tasks=[schema_task],
                verbose=True,
                process=Process.sequential
            )
            
            result = crew.kickoff()
            response_text = result.raw if hasattr(result, 'raw') else str(result)
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                schema = json.loads(json_str)
                print(f"✅ AI-discovered schema: {schema}")
                self.data_schema = schema
                return schema
            else:
                raise ValueError("No valid JSON found")
                
        except Exception as e:
            print(f"⚠️ AI schema discovery failed: {e}")
            print("Using fallback detection...")
            fallback_schema = create_fallback_column_mapping(column_names, df.head())
            self.data_schema = fallback_schema
            return fallback_schema
    
    def preprocess_upi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess financial data using AI-discovered schema"""
        
        schema = self.intelligent_data_schema_discovery(df)
        processed_df = df.copy()
        
        column_mapping = {}
        
        if schema.get('date_col'):
            column_mapping[schema['date_col']] = 'Date'
        if schema.get('amount_col'):
            column_mapping[schema['amount_col']] = 'Amount'
        if schema.get('desc_col'):
            column_mapping[schema['desc_col']] = 'Description'
        if schema.get('user_id_col'):
            column_mapping[schema['user_id_col']] = 'UserID'
        if schema.get('existing_category_col'):
            column_mapping[schema['existing_category_col']] = 'Category'
        if schema.get('direction_col'):
            column_mapping[schema['direction_col']] = 'Direction'
        if schema.get('status_col'):
            column_mapping[schema['status_col']] = 'Status'
        if schema.get('location_col'):
            column_mapping[schema['location_col']] = 'City'
        
        processed_df = processed_df.rename(columns=column_mapping)
        
        if 'Date' in processed_df.columns:
            processed_df['Date'] = pd.to_datetime(processed_df['Date'], errors='coerce')
        
        if 'Amount' in processed_df.columns:
            processed_df['Amount'] = pd.to_numeric(processed_df['Amount'], errors='coerce')
            processed_df = processed_df[processed_df['Amount'] > 0]
        
        if 'Direction' in processed_df.columns and schema.get('has_direction_info'):
            debit_keywords = ['debit', 'out', 'expense', 'withdrawal', 'send', 'payment']
            processed_df = processed_df[
                processed_df['Direction'].str.lower().str.contains('|'.join(debit_keywords), na=False)
            ]
        
        if 'Status' in processed_df.columns and schema.get('has_status_info'):
            success_keywords = ['success', 'completed', 'posted', 'settled', 'approved']
            processed_df = processed_df[
                processed_df['Status'].str.lower().str.contains('|'.join(success_keywords), na=False)
            ]
        
        if not schema.get('has_existing_categories') and 'Description' in processed_df.columns:
            processed_df['Category'] = processed_df['Description'].apply(self.smart_categorize_upi)
        
        essential_cols = ['Date', 'Amount']
        if 'UserID' in processed_df.columns:
            essential_cols.append('UserID')
        
        processed_df = processed_df.dropna(subset=essential_cols)
        
        print(f"✅ Preprocessing completed: {len(processed_df)} valid transactions")
        print(f"   Dataset type: {schema.get('dataset_type', 'unknown')}")
        if schema.get('has_user_ids'):
            print(f"   Multi-user: {processed_df['UserID'].nunique()} users")
        
        return processed_df
    
    def smart_categorize_upi(self, description):
        """AI-powered transaction categorization"""
        if pd.isna(description):
            return 'Other'
            
        desc = str(description).lower()
        
        categories = {
            'Food & Dining': ['zomato', 'swiggy', 'uber eats', 'dominos', 'pizza', 'restaurant', 'cafe', 'food', 'grocery', 'dmart'],
            'Transportation': ['uber', 'ola', 'rapido', 'taxi', 'auto', 'bus', 'metro', 'irctc', 'petrol', 'fuel'],
            'Shopping & Retail': ['amazon', 'flipkart', 'myntra', 'ajio', 'nykaa', 'shopping', 'mall', 'retail'],
            'Bills & Utilities': ['electricity', 'water', 'internet', 'broadband', 'mobile', 'recharge', 'bill'],
            'Entertainment': ['netflix', 'prime', 'hotstar', 'spotify', 'movie', 'cinema', 'pvr', 'inox'],
            'Healthcare': ['hospital', 'doctor', 'medical', 'pharmacy', 'medicine', 'health', '1mg', 'pharmeasy'],
            'Financial Services': ['bank', 'atm', 'loan', 'emi', 'insurance', 'investment', 'paytm', 'phonepe', 'gpay'],
            'Education': ['school', 'college', 'course', 'training', 'education', 'book', 'byju', 'unacademy'],
            'Travel & Vacation': ['hotel', 'flight', 'train', 'oyo', 'makemytrip', 'goibibo', 'travel', 'booking'],
            'Personal Care': ['salon', 'spa', 'beauty', 'grooming', 'urban company'],
            'Housing & Rent': ['rent', 'housing', 'maintenance', 'society', 'apartment']
        }
        
        for category, keywords in categories.items():
            if any(keyword in desc for keyword in keywords):
                return category
        
        return 'Other'
    
    def get_users_list(self, df: pd.DataFrame) -> List[str]:
        """Get list of unique users"""
        if 'UserID' in df.columns:
            return df['UserID'].unique().tolist()
        else:
            return ['single_user']
    
    def filter_user_data(self, df: pd.DataFrame, user_id: str) -> pd.DataFrame:
        """Filter data for specific user"""
        if user_id == 'single_user':
            return df.copy()
        else:
            user_df = df[df['UserID'] == user_id].copy()
            print(f"User {user_id}: {len(user_df)} transactions")
            return user_df
    
    def create_agents(self):
        """Create multi-agent system"""
        
        profile_builder = Agent(
            role='Financial Profile Builder',
            goal='Analyze transaction patterns to infer financial profile',
            backstory='Financial behavior analyst with payment systems expertise',
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        trend_analyzer = Agent(
            role='Financial Trend Analyzer',
            goal='Identify spending trends and patterns',
            backstory='Financial data scientist specializing in trend analysis',
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        budgeting_expert = Agent(
            role='Budget & Planning Expert',
            goal='Create personalized budget recommendations',
            backstory='Certified financial planner with budgeting expertise',
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        insight_generator = Agent(
            role='Financial Insights Synthesizer',
            goal='Synthesize analysis into actionable reports',
            backstory='Senior financial advisor translating analysis to insights',
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        return profile_builder, trend_analyzer, budgeting_expert, insight_generator
    
    def analyze_single_user(self, user_data: pd.DataFrame, user_id: str) -> Dict[str, Any]:
        """Analyze financial data for single user"""
        
        if len(user_data) < 5:
            return {
                'user_id': user_id,
                'status': 'insufficient_data',
                'message': f'Only {len(user_data)} transactions - insufficient for analysis'
            }
        
        try:
            analysis_context = self.generate_user_analysis_context(user_data, user_id)
            profile_builder, trend_analyzer, budgeting_expert, insight_generator = self.create_agents()
            tasks = self.create_analysis_tasks(analysis_context, user_id, 
                                            profile_builder, trend_analyzer, 
                                            budgeting_expert, insight_generator)
            
            crew = Crew(
                agents=[profile_builder, trend_analyzer, budgeting_expert, insight_generator],
                tasks=tasks,
                verbose=True,
                process=Process.sequential
            )
            
            result = crew.kickoff()
            final_report = result.raw if hasattr(result, 'raw') else str(result)
            
            chart_path = self.create_user_visualizations(user_data, user_id)
            
            result_package = {
                'user_id': user_id,
                'status': 'success',
                'executive_summary': self.extract_executive_summary(final_report),
                'comprehensive_narrative': final_report,
                'financial_health_score': float(self.extract_health_score(final_report)),
                'key_recommendations': self.extract_recommendations(final_report),
                'data_insights': convert_to_json_serializable(analysis_context),
                'chart_path': chart_path,
                'data_schema': self.data_schema,
                'summary': {
                    'total_transactions': int(len(user_data)),
                    'total_spending': float(user_data['Amount'].sum()),
                    'date_range': f"{user_data['Date'].min().strftime('%Y-%m-%d')} to {user_data['Date'].max().strftime('%Y-%m-%d')}",
                    'top_category': str(user_data.groupby('Category')['Amount'].sum().idxmax()),
                    'average_transaction': float(user_data['Amount'].mean()),
                    'most_frequent_merchant': str(user_data['Description'].mode().iloc[0] if not user_data['Description'].mode().empty else 'N/A')
                }
            }
            
            pdf_path = self.generate_user_pdf_report(result_package, user_data)
            result_package['pdf_path'] = pdf_path
            
            return result_package
            
        except Exception as e:
            print(f"Error analyzing user {user_id}: {str(e)}")
            return {
                'user_id': user_id,
                'status': 'error',
                'message': f'Analysis failed: {str(e)}'
            }
    
    def generate_user_analysis_context(self, user_data: pd.DataFrame, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive analysis context"""
        
        total_spending = float(user_data['Amount'].sum())
        
        category_analysis = user_data.groupby('Category')['Amount'].agg(['sum', 'count', 'mean']).to_dict('index')
        category_analysis = {cat: {
            'sum': float(data['sum']), 
            'count': int(data['count']), 
            'mean': float(data['mean'])
        } for cat, data in category_analysis.items()}
        
        monthly_spending = user_data.groupby(user_data['Date'].dt.to_period('M'))['Amount'].sum()
        monthly_analysis = {str(period): float(amount) for period, amount in monthly_spending.items()}
        
        top_merchants = user_data.groupby('Description')['Amount'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(10)
        merchant_analysis = {merchant: {
            'total_spent': float(data['sum']),
            'transaction_count': int(data['count'])
        } for merchant, data in top_merchants.to_dict('index').items()}
        
        daily_patterns = user_data.groupby(user_data['Date'].dt.day_name())['Amount'].sum().to_dict()
        daily_patterns = {day: float(amount) for day, amount in daily_patterns.items()}
        
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
        """Create analysis tasks"""
        
        profile_task = Task(
            description=f"""
            Analyze transaction patterns for User {user_id}:
            
            USER SUMMARY: {json.dumps(analysis_context['summary'], indent=2)}
            SPENDING PATTERNS: {json.dumps(analysis_context['category_analysis'], indent=2)}
            MERCHANT PREFERENCES: {json.dumps(analysis_context['merchant_analysis'], indent=2)}
            
            Determine: income class, life stage, digital payment behavior, lifestyle indicators, 
            financial maturity level. Provide detailed reasoning with evidence.
            """,
            agent=profile_builder,
            expected_output="Comprehensive user profile with income estimation and lifestyle analysis"
        )
        
        trend_task = Task(
            description=f"""
            Analyze spending trends for User {user_id}:
            
            MONTHLY TRENDS: {json.dumps(analysis_context['monthly_analysis'], indent=2)}
            DAILY PATTERNS: {json.dumps(analysis_context['daily_patterns'], indent=2)}
            CATEGORY BREAKDOWN: {json.dumps(analysis_context['spending_breakdown'], indent=2)}
            
            Analyze: month-over-month changes, day-of-week patterns, peak periods, 
            category trends, merchant loyalty, seasonal variations, predictive insights.
            """,
            agent=trend_analyzer,
            expected_output="Detailed trend analysis with MoM changes and predictions"
        )
        
        budget_task = Task(
            description=f"""
            Create budget plan for User {user_id}:
            
            CURRENT SPENDING: {json.dumps(analysis_context['spending_breakdown'], indent=2)}
            MONTHLY AVERAGE: Rs.{analysis_context['summary']['total_spending']/max(len(analysis_context['monthly_analysis']), 1):.0f}
            TOP MERCHANTS: {json.dumps(analysis_context['merchant_analysis'], indent=2)}
            
            Create: suggested monthly budget by category, essential vs discretionary allocation, 
            savings targets, spending optimization tips, merchant-specific limits, emergency fund recommendations.
            """,
            agent=budgeting_expert,
            expected_output="Personalized budget plan with specific allocations"
        )
        
        synthesis_task = Task(
            description=f"""
            Create comprehensive report for User {user_id}:
            
            Structure as:
            1. EXECUTIVE SUMMARY (2-3 sentences)
            2. SPENDING SNAPSHOT (key numbers)
            3. FINANCIAL PROFILE (income class, life stage)
            4. SPENDING TRENDS (top 3-5 insights with numbers)
            5. BUDGET RECOMMENDATIONS (specific allocations)
            6. ACTIONABLE TIPS (3-5 specific recommendations)
            7. FINANCIAL HEALTH SCORE (1-10 with explanation)
            
            Use clear language, specific amounts, actionable recommendations.
            """,
            agent=insight_generator,
            expected_output="Single comprehensive financial report with all insights"
        )
        
        return [profile_task, trend_task, budget_task, synthesis_task]
    
    def create_user_visualizations(self, user_data: pd.DataFrame, user_id: str) -> str:
        """Create visualizations for user"""
        try:
            with plt.style.context('default'):
                fig = plt.figure(figsize=(16, 12), facecolor='white')
                fig.suptitle(f'Financial Analysis - User {user_id}', fontsize=16, fontweight='bold')
                
                gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3, left=0.08, right=0.95, top=0.92, bottom=0.08)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[1, 0])
                ax4 = fig.add_subplot(gs[1, 1])
                
                # Category spending pie
                category_spending = user_data.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(8)
                if not category_spending.empty:
                    ax1.pie(category_spending.values, labels=category_spending.index, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Spending by Category', fontweight='bold', pad=10)
                
                # Monthly trend
                monthly_spending = user_data.groupby(user_data['Date'].dt.to_period('M'))['Amount'].sum()
                if len(monthly_spending) > 1:
                    ax2.plot(monthly_spending.index.astype(str), monthly_spending.values, marker='o', color='green')
                    ax2.set_title('Monthly Spending Trend', fontweight='bold', pad=10)
                    ax2.tick_params(axis='x', rotation=45, labelsize=8)
                    ax2.grid(True, alpha=0.3)
                
                # Daily pattern
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_spending = user_data.groupby(user_data['Date'].dt.day_name())['Amount'].sum()
                daily_spending = daily_spending.reindex(day_order, fill_value=0)
                
                if not daily_spending.empty and daily_spending.sum() > 0:
                    ax3.bar(daily_spending.index, daily_spending.values, color='orange', alpha=0.7)
                    ax3.set_title('Spending by Day of Week', fontweight='bold', pad=10)
                    ax3.tick_params(axis='x', rotation=45, labelsize=8)
                    ax3.grid(True, alpha=0.3)
                
                # Top merchants
                top_merchants = user_data.groupby('Description')['Amount'].sum().sort_values(ascending=False).head(6)
                if not top_merchants.empty:
                    truncated_names = [name[:15] + '...' if len(name) > 15 else name for name in top_merchants.index]
                    ax4.bar(range(len(top_merchants)), top_merchants.values, color='purple', alpha=0.7)
                    ax4.set_xticks(range(len(top_merchants)))
                    ax4.set_xticklabels(truncated_names, rotation=45, ha='right', fontsize=8)
                    ax4.set_title('Top Merchants by Spending', fontweight='bold', pad=10)
                    ax4.grid(True, alpha=0.3)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                chart_path = os.path.join(os.getcwd(), f"user_{user_id}_analysis_{timestamp}.png")
                
                fig.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                plt.close('all')
                
                return chart_path
                
        except Exception as e:
            try:
                plt.close('all')
            except:
                pass
            print(f"Visualization error for user {user_id}: {str(e)}")
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
                return ' '.join(summary_lines) if summary_lines else "Financial analysis completed successfully."
        
        paragraphs = [p.strip() for p in report_text.split('\n\n') if p.strip()]
        return paragraphs[0] if paragraphs else "Financial analysis completed successfully."
    
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
            "Set up monthly spending budgets for each category",
            "Monitor your top 3 spending merchants regularly",
            "Use recurring payments for bills to avoid late fees",
            "Review and optimize your largest expense categories",
            "Build an emergency fund using automated transfers"
        ]
    
    def generate_user_pdf_report(self, result_package: Dict[str, Any], user_data: pd.DataFrame) -> str:
        """Generate PDF report for user"""
        try:
            major_spends = user_data.nlargest(10, 'Amount')[['Date', 'Description', 'Amount', 'Category']].copy()
            major_spends['Date'] = major_spends['Date'].dt.strftime('%Y-%m-%d')
            
            category_summary = user_data.groupby('Category')['Amount'].agg(['sum', 'count', 'mean']).round(2)
            category_summary.columns = ['Total', 'Count', 'Average']
            category_summary['Percentage'] = (category_summary['Total'] / category_summary['Total'].sum() * 100).round(1)
            category_summary = category_summary.reset_index()
            
            monthly_summary = user_data.groupby(user_data['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
            monthly_summary['Date'] = monthly_summary['Date'].astype(str)
            monthly_summary.columns = ['Month', 'Amount']
            
            summary_dict = {
                'Total Spending': f"Rs.{result_package['summary']['total_spending']:,.2f}",
                'Total Transactions': str(result_package['summary']['total_transactions']),
                'Average Transaction': f"Rs.{result_package['summary']['average_transaction']:,.2f}",
                'Top Category': result_package['summary']['top_category'],
                'Most Frequent Category': result_package['summary']['top_category'],
                'Date Range': result_package['summary']['date_range']
            }
            
            pdf_result = {
                'summary': summary_dict,
                'comprehensive_narrative': result_package.get('comprehensive_narrative', 
                                                            result_package.get('narrative', 'Analysis completed successfully.')),
                'chart_path': result_package.get('chart_path')
            }
            
            pdf_path = generate_pdf(
                pdf_result, 
                user_data.head(20), 
                major_spends, 
                category_summary, 
                monthly_summary
            )
            
            return pdf_path
            
        except Exception as e:
            print(f"PDF generation error for user {result_package['user_id']}: {str(e)}")
            return None
    
    def process_multiple_users(self, file_path: str, max_users: int = None, parallel: bool = True) -> Dict[str, Any]:
        """Process multiple users from dataset"""
        
        print("Loading and preprocessing data...")
        df = self.load_and_validate_data(file_path)
        processed_df = self.preprocess_upi_data(df)
        
        users = self.get_users_list(processed_df)
        if max_users:
            users = users[:max_users]
        
        print(f"Processing {len(users)} users...")
        
        results = {}
        
        if parallel and len(users) > 1:
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
                        print(f"✅ Completed analysis for user {user_id}")
                    except Exception as e:
                        print(f"❌ Error processing user {user_id}: {str(e)}")
                        results[user_id] = {
                            'user_id': user_id,
                            'status': 'error',
                            'message': str(e)
                        }
        else:
            for user_id in users:
                try:
                    user_data = self.filter_user_data(processed_df, user_id)
                    result = self.analyze_single_user(user_data, user_id)
                    results[user_id] = result
                    print(f"✅ Completed analysis for user {user_id}")
                except Exception as e:
                    print(f"❌ Error processing user {user_id}: {str(e)}")
                    results[user_id] = {
                        'user_id': user_id,
                        'status': 'error',
                        'message': str(e)
                    }
        
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
        """Generate overall summary report"""
        
        successful_results = [r for r in results.values() if r.get('status') == 'success']
        
        if not successful_results:
            return {
                'message': 'No successful analyses to summarize',
                'total_users': len(results),
                'processed_successfully': 0
            }
        
        total_transactions = sum([r['summary']['total_transactions'] for r in successful_results])
        total_spending = sum([r['summary']['total_spending'] for r in successful_results])
        avg_health_score = sum([r['financial_health_score'] for r in successful_results]) / len(successful_results)
        
        all_categories = []
        for r in successful_results:
            for category, percentage in r['data_insights']['spending_breakdown'].items():
                all_categories.append(category)
        
        category_frequency = Counter(all_categories)
        top_categories = category_frequency.most_common(5)
        
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
    """Main function to demonstrate AWS Bedrock multi-user financial analysis"""
    
    analyzer = MultiUserFinancialAnalyzer()
    
    file_path = "D:/ojje_ad/upi_transactions.xlsx"
    
    try:
        results = analyzer.process_multiple_users(file_path, max_users=2, parallel=False)
        
        print("\n" + "="*60)
        print("✅ MULTI-USER FINANCIAL ANALYSIS COMPLETE")
        print("="*60)
        
        print(f"\nTotal Users Processed: {results['total_users_processed']}")
        print(f"Successful Analyses: {results['successful_analyses']}")
        print(f"Failed Analyses: {results['failed_analyses']}")
        print(f"Insufficient Data: {results['insufficient_data']}")
        
        print("\n📊 OVERALL SUMMARY:")
        summary = results['overall_summary']
        if 'total_users_analyzed' in summary:
            print(f"- Average spending per user: Rs.{summary['average_spending_per_user']:,.2f}")
            print(f"- Average financial health score: {summary['average_financial_health_score']}/10")
            print(f"- Most common categories: {[cat[0] for cat in summary['most_common_categories'][:3]]}")
        
        print("\n👤 INDIVIDUAL USER RESULTS:")
        for user_id, result in results['user_results'].items():
            if result['status'] == 'success':
                print(f"\n✅ User {user_id}:")
                print(f"  - Total Spending: Rs.{result['summary']['total_spending']:,.2f}")
                print(f"  - Health Score: {result['financial_health_score']}/10")
                print(f"  - Top Category: {result['summary']['top_category']}")
                print(f"  - PDF Report: {result.get('pdf_path', 'Not generated')}")
            else:
                print(f"\n❌ User {user_id}: {result['status']} - {result.get('message', 'Unknown error')}")
        
        output_file = f"multi_user_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()