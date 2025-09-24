import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
from openai import OpenAI
from datetime import datetime, timedelta
import json
import re
from typing import Dict, List, Tuple
import time
import calendar

# Advanced Futuristic CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');
    
    .page-header {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        background: linear-gradient(45deg, #00f5ff, #0080ff, #8000ff, #ff0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 900;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px #00f5ff); }
        to { filter: drop-shadow(0 0 40px #0080ff); }
    }
    
    .oracle-card {
        background: linear-gradient(135deg, #0f0f23 0%, #1a0d40 50%, #2d1b69 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid #00f5ff;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.3);
        color: white;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .oracle-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(transparent, rgba(0, 245, 255, 0.1), transparent);
        animation: rotate 4s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .insights-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #00ff88;
        margin: 1.5rem 0;
        color: white;
        position: relative;
        z-index: 1;
    }
    
    .future-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #00f5ff;
        margin: 1.5rem 0;
        color: white;
        position: relative;
        z-index: 1;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #ff6b6b;
        margin: 1.5rem 0;
        color: white;
        z-index: 1;
    }
    
    .metric-oracle {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #00f5ff;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .metric-oracle::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 245, 255, 0.2), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .structured-section {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 245, 255, 0.3);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AIFinancialOracle:
    """Advanced AI Oracle for financial analysis"""
    
    def __init__(self):
        self.api_key = ""
        self.client = None
        self.client = OpenAI(api_key=self.api_key)
    
    def auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Automatically detect column mappings"""
        columns = df.columns.str.lower().str.strip()
        mapping = {}
        
        # Date detection
        date_keywords = ['date', 'time', 'timestamp', 'txn_date', 'transaction_date']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                mapping['date'] = col
                break
        
        # Amount/Withdrawal detection
        amount_keywords = ['amount', 'withdrawal', 'debit', 'spent', 'expense', 'value', 'sum']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in amount_keywords):
                mapping['amount'] = col
                break
        
        # Category detection
        category_keywords = ['category', 'type', 'merchant', 'description', 'purpose', 'narration']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in category_keywords):
                mapping['category'] = col
                break
        
        # Income/Deposit detection
        income_keywords = ['deposit', 'credit', 'income', 'received', 'salary', 'earning']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in income_keywords):
                mapping['income'] = col
                break
        
        # Fallback to positional
        if 'date' not in mapping and len(df.columns) > 0:
            mapping['date'] = df.columns[0]
        if 'amount' not in mapping and len(df.columns) > 1:
            mapping['amount'] = df.columns[1]
        if 'category' not in mapping and len(df.columns) > 2:
            mapping['category'] = df.columns[2]
        
        return mapping
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the data automatically"""
        mapping = self.auto_detect_columns(df)
        
        df_clean = df.copy()
        
        # Process Date
        if 'date' in mapping:
            df_clean['Date'] = pd.to_datetime(df[mapping['date']], errors='coerce')
        
        # Process Amount (withdrawals/spending)
        if 'amount' in mapping:
            df_clean['Amount'] = df[mapping['amount']].astype(str).str.replace('[₹$,]', '', regex=True)
            df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce').fillna(0).abs()
        
        # Process Income (if available)
        if 'income' in mapping:
            df_clean['Income'] = df[mapping['income']].astype(str).str.replace('[₹$,]', '', regex=True)
            df_clean['Income'] = pd.to_numeric(df_clean['Income'], errors='coerce').fillna(0).abs()
        else:
            df_clean['Income'] = 0
        
        # Process Category
        if 'category' in mapping:
            df_clean['Category'] = df[mapping['category']].fillna('Other').astype(str).str.title()
        
        # Clean data
        df_clean = df_clean.dropna(subset=['Date'])
        df_clean = df_clean[(df_clean['Amount'] > 0) | (df_clean['Income'] > 0)]
        
        # Add time-based columns
        df_clean['Year'] = df_clean['Date'].dt.year
        df_clean['Month'] = df_clean['Date'].dt.month
        df_clean['MonthName'] = df_clean['Date'].dt.strftime('%B %Y')
        
        return df_clean
    
    def get_available_periods(self, df: pd.DataFrame) -> Dict:
        """Get available time periods from data"""
        years = sorted(df['Year'].unique())
        months = []
        
        for year in years:
            year_data = df[df['Year'] == year]
            year_months = sorted(year_data['Month'].unique())
            for month in year_months:
                month_name = calendar.month_name[month]
                months.append(f"{month_name} {year}")
        
        return {
            'years': years,
            'months': months,
            'full_period': f"{df['Date'].min().strftime('%B %Y')} to {df['Date'].max().strftime('%B %Y')}"
        }
    
    def filter_data_by_period(self, df: pd.DataFrame, period_type: str, period_value: str) -> pd.DataFrame:
        """Filter data by selected period"""
        if period_type == "Full Period":
            return df
        elif period_type == "Yearly":
            year = int(period_value.split()[-1])
            return df[df['Year'] == year]
        elif period_type == "Monthly":
            month_name, year = period_value.split()
            month_num = list(calendar.month_name).index(month_name)
            year = int(year)
            return df[(df['Month'] == month_num) & (df['Year'] == year)]
        return df
    
    def analyze_period_data(self, df: pd.DataFrame, period_name: str) -> Dict:
        """Analyze data for the selected period"""
        total_spending = df['Amount'].sum()
        total_income = df['Income'].sum()
        net_savings = total_income - total_spending
        
        # Category analysis
        category_stats = df.groupby('Category').agg({
            'Amount': ['sum', 'mean', 'count']
        }).round(2)
        category_stats.columns = ['total', 'avg', 'count']
        category_stats['percentage'] = (category_stats['total'] / total_spending * 100).round(2) if total_spending > 0 else 0
        category_stats = category_stats.sort_values('total', ascending=False)
        
        # Time analysis
        days_in_period = (df['Date'].max() - df['Date'].min()).days + 1
        daily_avg_spending = total_spending / days_in_period if days_in_period > 0 else 0
        
        return {
            'period_name': period_name,
            'total_spending': total_spending,
            'total_income': total_income,
            'net_savings': net_savings,
            'savings_rate': (net_savings / total_income * 100) if total_income > 0 else 0,
            'daily_avg_spending': daily_avg_spending,
            'transaction_count': len(df),
            'category_stats': category_stats,
            'days_in_period': days_in_period,
            'top_category': category_stats.index[0] if len(category_stats) > 0 else 'Unknown'
        }
    
    def generate_structured_insights(self, analysis: Dict) -> str:
        """Generate structured insights and disadvantages"""
        if not self.client:
            return self._generate_fallback_insights(analysis)
        
        try:
            top_categories = analysis['category_stats'].head(3)
            
            prompt = f"""
            As an AI Financial Oracle, provide a STRUCTURED analysis of this {analysis['period_name']} financial data:
            
            FINANCIAL SUMMARY:
            📊 Total Income: ₹{analysis['total_income']:,.2f}
            💸 Total Spending: ₹{analysis['total_spending']:,.2f}
            💰 Net Savings: ₹{analysis['net_savings']:,.2f}
            📈 Savings Rate: {analysis['savings_rate']:.1f}%
            🎯 Top Category: {analysis['top_category']} ({analysis['category_stats'].iloc[0]['percentage'] if len(analysis['category_stats']) > 0 else 0:.1f}%)
            ⚡ Daily Average: ₹{analysis['daily_avg_spending']:,.2f}
            
            TOP SPENDING AREAS:
            {self._format_categories_for_prompt(top_categories)}
            
            Provide analysis in EXACTLY this structure:
            
            ## 💡 KEY INSIGHTS
            - [3-4 specific insights about spending patterns]
            
            ## ⚠️ DISADVANTAGES & CONCERNS
            - [3-4 specific problems or areas of concern]
            
            ## 📊 SPENDING EFFICIENCY
            - [2-3 observations about spending effectiveness]
            
            Use specific numbers from the data. Be direct and actionable.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a precise financial analyst. Provide structured, data-driven insights with specific numbers. Be direct and actionable."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"AI Error: {e}")
            return self._generate_fallback_insights(analysis)
    
    def generate_future_recommendations(self, analysis: Dict) -> str:
        """Generate structured future recommendations"""
        if not self.client:
            return self._generate_fallback_future(analysis)
        
        try:
            prompt = f"""
            Based on this {analysis['period_name']} financial data, provide STRUCTURED future recommendations:
            
            CURRENT STATE:
            - Savings Rate: {analysis['savings_rate']:.1f}%
            - Daily Spending: ₹{analysis['daily_avg_spending']:,.2f}
            - Top Risk Category: {analysis['top_category']}
            - Net Position: {"Positive" if analysis['net_savings'] > 0 else "Negative"}
            
            Provide recommendations in EXACTLY this structure:
            
            ## 🚀 WHAT TO DO (Action Items)
            - [4-5 specific actions with exact amounts/percentages]
            
            ## 🛑 WHAT NOT TO DO (Avoid These)
            - [3-4 specific things to avoid or reduce]
            
            ## 📈 30-DAY TARGETS
            - [3-4 specific targets for next month with numbers]
            
            ## 🎯 LONG-TERM STRATEGY
            - [3-4 strategic recommendations for 3-6 months]
            
            Use specific, actionable advice with numbers.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a strategic financial advisor. Provide specific, actionable recommendations with exact numbers and timeframes."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            print(response)
            return response.choices[0].message.content
            
        except Exception as e:
            return self._generate_fallback_future(analysis)
    
    def _format_categories_for_prompt(self, category_stats):
        """Format categories for AI prompt"""
        formatted = ""
        for i, (category, row) in enumerate(category_stats.iterrows()):
            formatted += f"- {category}: ₹{row['total']:,.2f} ({row['percentage']:.1f}%)\n"
        return formatted
    
    def _generate_fallback_insights(self, analysis: Dict) -> str:
        """Fallback insights when AI is unavailable"""
        return f"""
## 💡 KEY INSIGHTS
- Your {analysis['period_name']} shows {analysis['savings_rate']:.1f}% savings rate
- Daily spending average of ₹{analysis['daily_avg_spending']:,.2f} indicates {'controlled' if analysis['daily_avg_spending'] < 1000 else 'high'} spending
- {analysis['top_category']} dominates your spending at {analysis['category_stats'].iloc[0]['percentage'] if len(analysis['category_stats']) > 0 else 0:.1f}%
- Total of {analysis['transaction_count']} transactions shows {'frequent' if analysis['transaction_count'] > 100 else 'moderate'} financial activity

## ⚠️ DISADVANTAGES & CONCERNS
- {'Negative savings rate needs immediate attention' if analysis['savings_rate'] < 0 else 'Low savings rate below recommended 20%' if analysis['savings_rate'] < 20 else 'Savings rate could be optimized further'}
- High dependency on {analysis['top_category']} spending creates budget vulnerability
- {'Income sources need diversification' if analysis['total_income'] < 50000 else 'Income vs spending ratio needs balancing'}

## 📊 SPENDING EFFICIENCY
- Current spending pattern shows {'inefficient resource allocation' if analysis['savings_rate'] < 10 else 'moderate financial discipline'}
- Category diversification {'needs improvement' if len(analysis['category_stats']) < 5 else 'shows good variety'}
        """
    
    def _generate_fallback_future(self, analysis: Dict) -> str:
        """Fallback future recommendations"""
        return f"""
## 🚀 WHAT TO DO (Action Items)
- Set monthly spending limit to ₹{analysis['total_spending'] * 0.9:,.0f} (10% reduction)
- Increase savings rate to 25% of income (₹{analysis['total_income'] * 0.25:,.0f}/month)
- Track {analysis['top_category']} expenses daily using apps
- Build emergency fund of ₹{analysis['total_income'] * 3:,.0f}

## 🛑 WHAT NOT TO DO (Avoid These)
- Don't exceed ₹{analysis['daily_avg_spending'] * 1.2:,.0f}/day spending
- Avoid impulse purchases in {analysis['top_category']} category
- Don't ignore small recurring expenses
- Avoid taking on new debt or loans

## 📈 30-DAY TARGETS
- Reduce daily spending to ₹{analysis['daily_avg_spending'] * 0.85:,.0f}
- Save ₹{max(5000, analysis['total_income'] * 0.15):,.0f} this month
- Track every expense for 30 days
- Review and optimize {analysis['top_category']} spending

## 🎯 LONG-TERM STRATEGY
- Achieve 25% savings rate within 3 months
- Diversify spending across more categories
- Build 6-month emergency fund
- Start investing ₹{max(2000, analysis['net_savings'] * 0.5):,.0f}/month if savings are positive
        """

def create_period_visualizations(analysis: Dict):
    """Create visualizations for the selected period"""
    
    # Spending Distribution
    fig_pie = go.Figure(data=[go.Pie(
        labels=analysis['category_stats'].index,
        values=analysis['category_stats']['total'],
        hole=0.3,
        marker=dict(
            colors=px.colors.sequential.Plasma,
            line=dict(color='#FFFFFF', width=2)
        ),
        textfont=dict(size=12, color='white'),
        textposition='inside',
        textinfo='percent+label'
    )])
    
    fig_pie.update_layout(
        title=dict(
            text=f"🌌 {analysis['period_name']} Spending Distribution",
            font=dict(size=18, color='white', family='Orbitron')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False,
        height=400
    )
    
    # Financial Health Gauge
    savings_rate = analysis['savings_rate']
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = savings_rate,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Savings Rate %", 'font': {'color': 'white', 'family': 'Orbitron'}},
        delta = {'reference': 20, 'increasing': {'color': "#00ff88"}, 'decreasing': {'color': "#ff416c"}},
        gauge = {
            'axis': {'range': [None, 50], 'tickcolor': 'white'},
            'bar': {'color': "#00f5ff"},
            'steps': [
                {'range': [0, 10], 'color': "#ff416c"},
                {'range': [10, 20], 'color': "#ffa726"},
                {'range': [20, 50], 'color': "#00ff88"}],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 4},
                'thickness': 0.75,
                'value': 20}}
    ))
    
    fig_gauge.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400
    )
    
    return fig_pie, fig_gauge

def ai_spending_oracle_page():
    """Main AI Spending Oracle Page"""
    
    # Page Header
    st.markdown('<div class="page-header">🤖 AI FINANCIAL ORACLE</div>', unsafe_allow_html=True)
    
    # Get API key from session state if available
    api_key = st.session_state.get('openai_api_key', '')
    
    # File upload in main area
    st.markdown("### 📊 Upload Your Financial Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with your transaction data",
        type=['csv'],
        help="The Oracle will automatically detect your data format"
    )
    
    if uploaded_file is not None:
        try:
            # Process data
            with st.spinner("🔮 Oracle is reading your financial patterns..."):
                df = pd.read_csv(uploaded_file)
                oracle = AIFinancialOracle()
                df_processed = oracle.process_data(df)
                periods = oracle.get_available_periods(df_processed)
                time.sleep(1)
            
            # Period Selection
            st.markdown("### 📅 Select Analysis Period")
            
            col1, col2 = st.columns(2)
            
            with col1:
                period_type = st.selectbox(
                    "Analysis Scope:",
                    ["Full Period", "Yearly", "Monthly"]
                )
            
            with col2:
                if period_type == "Full Period":
                    period_value = periods['full_period']
                    st.info(f"Analyzing: {period_value}")
                elif period_type == "Yearly":
                    period_value = st.selectbox("Select Year:", periods['years'])
                else:  # Monthly
                    period_value = st.selectbox("Select Month:", periods['months'])
            
            # Filter and analyze data
            if st.button("🚀 CONSULT THE AI ORACLE", type="primary"):
                with st.spinner("🧠 Oracle is channeling financial wisdom..."):
                    # Filter data based on selection
                    filtered_df = oracle.filter_data_by_period(df_processed, period_type, str(period_value))
                    
                    # Analyze the period
                    period_name = period_value if period_type == "Full Period" else f"{period_type}: {period_value}"
                    analysis = oracle.analyze_period_data(filtered_df, period_name)
                    
                    time.sleep(2)
                
                # Display Period Summary
                st.markdown(f"## 📊 {analysis['period_name']} Financial Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-oracle">
                    <h3>💰 Total Income</h3>
                    <h2>₹{analysis['total_income']:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-oracle">
                    <h3>💸 Total Spending</h3>
                    <h2>₹{analysis['total_spending']:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-oracle">
                    <h3>💎 Net Savings</h3>
                    <h2>₹{analysis['net_savings']:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-oracle">
                    <h3>📈 Savings Rate</h3>
                    <h2>{analysis['savings_rate']:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Two Column Layout for Insights
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("## 🧠 CURRENT PERIOD INSIGHTS")
                    
                    with st.spinner("🤖 Analyzing patterns and disadvantages..."):
                        insights = oracle.generate_structured_insights(analysis)
                        time.sleep(1)
                    
                    st.markdown(f"""
                    <div class="insights-card">
                    {insights}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("## 🚀 FUTURE STRATEGY & ACTIONS")
                    
                    with st.spinner("🔮 Generating future recommendations..."):
                        recommendations = oracle.generate_future_recommendations(analysis)
                        time.sleep(1)
                    
                    st.markdown(f"""
                    <div class="future-card">
                    {recommendations}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualizations
                st.markdown("## 📊 Oracle's Visual Insights")
                
                fig_pie, fig_gauge = create_period_visualizations(analysis)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Success message
                st.success("🔮 The Oracle has revealed your financial destiny! Use these insights wisely.")
                st.balloons()
        
        except Exception as e:
            st.error(f"💀 Oracle's vision is clouded: {e}")
            st.info("Ensure your CSV contains transaction data with dates, amounts, and categories")
    
    else:
        # Landing interface
        st.markdown("""
        <div class="oracle-card">
        <h2>🌟 THE AI ORACLE AWAITS YOUR DATA</h2>
        <p>Upload your financial data and witness the Oracle's power to:</p>
        
        <div class="structured-section">
        <h3>🔮 AUTOMATIC INTELLIGENCE:</h3>
        <ul>
        <li>📊 <strong>Smart Detection:</strong> Automatically reads any CSV format</li>
        <li>📅 <strong>Period Analysis:</strong> Analyze full year, specific months, or custom periods</li>
        <li>💰 <strong>Income Tracking:</strong> Calculates total income and savings rates</li>
        <li>🎯 <strong>Category Insights:</strong> Deep analysis of spending patterns</li>
        </ul>
        </div>
        
        <div class="structured-section">
        <h3>🧠 STRUCTURED INSIGHTS:</h3>
        <ul>
        <li>💡 <strong>Current Period Analysis:</strong> Key insights and disadvantages</li>
        <li>🚀 <strong>Future Recommendations:</strong> What to do and what to avoid</li>
        <li>📈 <strong>Action Plans:</strong> 30-day targets and long-term strategy</li>
        <li>🔮 <strong>AI-Powered Advice:</strong> Personalized financial guidance</li>
        </ul>
        </div>
        </div>
        """, unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    ai_spending_oracle_page()