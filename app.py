import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import base64
from datetime import datetime
import tempfile

# Import your existing classes
try:
    from crew import MultiUserFinancialAnalyzer
    from budget import BudgetAdvisor
except ImportError:
    st.error("Please ensure crew.py and budget.py are in the same directory as this script")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="UPI Financial Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Financial Insights", "Budget Planning"])

# Helper functions
@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and preprocess the uploaded file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx' if uploaded_file.name.endswith('.xlsx') else '.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        analyzer = MultiUserFinancialAnalyzer()
        df = analyzer.load_and_validate_data(tmp_file_path)
        processed_df = analyzer.preprocess_upi_data(df)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return processed_df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def create_visualizations(df, user_id=None):
    """Create interactive visualizations using Plotly"""
    if user_id:
        df = df[df['UserID'] == user_id]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Spending by Category', 'Monthly Trend', 'Daily Patterns', 'Top Merchants'),
        specs=[[{"type": "pie"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Category spending pie chart
    category_spending = df.groupby('Category')['Amount'].sum().sort_values(ascending=False).head(8)
    fig.add_trace(
        go.Pie(labels=category_spending.index, values=category_spending.values, name="Categories"),
        row=1, col=1
    )
    
    # Monthly trend
    if len(df) > 1:
        monthly_spending = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum()
        fig.add_trace(
            go.Scatter(x=monthly_spending.index.astype(str), y=monthly_spending.values, 
                      mode='lines+markers', name="Monthly Trend", line=dict(color='green')),
            row=1, col=2
        )
    
    # Daily patterns
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_spending = df.groupby(df['Date'].dt.day_name())['Amount'].sum()
    daily_spending = daily_spending.reindex(day_order, fill_value=0)
    
    fig.add_trace(
        go.Bar(x=daily_spending.index, y=daily_spending.values, name="Daily Spending", marker_color='orange'),
        row=2, col=1
    )
    
    # Top merchants
    top_merchants = df.groupby('Description')['Amount'].sum().sort_values(ascending=False).head(6)
    fig.add_trace(
        go.Bar(x=top_merchants.index, y=top_merchants.values, name="Top Merchants", marker_color='purple'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="UPI Spending Analysis Dashboard")
    fig.update_xaxes(tickangle=45, row=2, col=2)
    
    return fig

def display_user_metrics(result):
    """Display user metrics in an attractive format"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Spending",
            value=f"₹{result['summary']['total_spending']:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Total Transactions",
            value=result['summary']['total_transactions'],
            delta=None
        )
    
    with col3:
        st.metric(
            label="Average Transaction",
            value=f"₹{result['summary']['average_transaction']:,.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Financial Health Score",
            value=f"{result['financial_health_score']}/10",
            delta=None
        )

def download_link(object_to_download, download_filename):
    """Create a download link for files"""
    if isinstance(object_to_download, bytes):
        b64 = base64.b64encode(object_to_download).decode()
    else:
        with open(object_to_download, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
    
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{download_filename}">Download {download_filename}</a>'

# PAGE 1: FINANCIAL INSIGHTS
if page == "Financial Insights":
    st.markdown('<h1 class="main-header">💰 UPI Financial Insights</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📁 Upload Your UPI Transaction Data")
    uploaded_file = st.file_uploader(
        "Choose your UPI data file",
        type=['csv', 'xlsx'],
        help="Upload your UPI transaction data in CSV or Excel format"
    )
    
    if uploaded_file is not None:
        # Process the uploaded file
        with st.spinner("Processing your data..."):
            processed_df = load_and_process_data(uploaded_file)
        
        if processed_df is not None:
            st.session_state.processed_data = processed_df
            
            st.markdown('<div class="success-box">✅ Data loaded successfully!</div>', unsafe_allow_html=True)
            
            # Display basic data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(processed_df))
            with col2:
                st.metric("Unique Users", processed_df['UserID'].nunique())
            with col3:
                st.metric("Date Range", f"{(processed_df['Date'].max() - processed_df['Date'].min()).days} days")
            
            # User selection
            users = processed_df['UserID'].unique().tolist()
            
            if len(users) == 1:
                selected_user = users[0]
                st.info(f"Analyzing data for User: {selected_user}")
            else:
                selected_user = st.selectbox("Select a user for analysis:", users)
            
            # Analysis options
            col1, col2 = st.columns(2)
            with col1:
                max_users = st.number_input("Maximum users to analyze", min_value=1, max_value=len(users), value=min(3, len(users)))
            with col2:
                parallel_processing = st.checkbox("Enable parallel processing", value=True)
            
            # Run analysis button
            if st.button("🔍 Run Financial Analysis", type="primary"):
                try:
                    with st.spinner("Running comprehensive financial analysis... This may take a few minutes."):
                        analyzer = MultiUserFinancialAnalyzer()
                        
                        # Save processed data temporarily for analysis
                        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx').name
                        processed_df.to_excel(temp_path, index=False)
                        
                        # Run analysis
                        results = analyzer.process_multiple_users(
                            temp_path, 
                            max_users=max_users, 
                            parallel=parallel_processing
                        )
                        
                        st.session_state.analysis_results = results
                        
                        # Clean up temp file
                        os.unlink(temp_path)
                        
                        st.success("Analysis completed successfully!")
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    # Display results if analysis is complete
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        
        st.markdown("## 📊 Analysis Results")
        
        # Overall summary
        st.markdown("### 📈 Overall Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Users Processed", results['total_users_processed'])
        with col2:
            st.metric("Successful Analyses", results['successful_analyses'])
        with col3:
            st.metric("Failed Analyses", results['failed_analyses'])
        with col4:
            st.metric("Insufficient Data", results['insufficient_data'])
        
        if results['successful_analyses'] > 0:
            summary = results['overall_summary']
            
            # Display aggregate metrics
            st.markdown("### 💡 Key Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Average Spending per User",
                    f"₹{summary.get('average_spending_per_user', 0):,.2f}"
                )
                st.metric(
                    "Average Financial Health Score",
                    f"{summary.get('average_financial_health_score', 0)}/10"
                )
            
            with col2:
                if 'most_common_categories' in summary:
                    st.write("**Most Common Categories:**")
                    for cat, count in summary['most_common_categories'][:3]:
                        st.write(f"• {cat}: {count} users")
        
        # Individual user results
        st.markdown("### 👤 Individual User Results")
        
        successful_users = [user_id for user_id, result in results['user_results'].items() 
                          if result.get('status') == 'success']
        
        if successful_users:
            selected_user_result = st.selectbox("Select user to view detailed results:", successful_users)
            
            if selected_user_result:
                user_result = results['user_results'][selected_user_result]
                
                # Display user metrics
                st.markdown(f"#### 📋 Results for User {selected_user_result}")
                display_user_metrics(user_result)
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "📝 Full Report", "🎯 Recommendations", "📄 Downloads"])
                
                with tab1:
                    if st.session_state.processed_data is not None:
                        fig = create_visualizations(st.session_state.processed_data, selected_user_result)
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.markdown("#### 📖 Comprehensive Analysis")
                    st.markdown(user_result.get('comprehensive_narrative', 'Report not available'))
                
                with tab3:
                    st.markdown("#### 💡 Key Recommendations")
                    recommendations = user_result.get('key_recommendations', [])
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
                
                with tab4:
                    st.markdown("#### 📥 Download Reports")
                    
                    # JSON download
                    json_str = json.dumps(user_result, indent=2, default=str)
                    st.download_button(
                        label="Download JSON Report",
                        data=json_str,
                        file_name=f"user_{selected_user_result}_analysis.json",
                        mime="application/json"
                    )
                    
                    # PDF download link (if available)
                    if user_result.get('pdf_path') and os.path.exists(user_result['pdf_path']):
                        with open(user_result['pdf_path'], 'rb') as pdf_file:
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_file.read(),
                                file_name=f"user_{selected_user_result}_report.pdf",
                                mime="application/pdf"
                            )

# PAGE 2: BUDGET PLANNING
elif page == "Budget Planning":
    st.markdown('<h1 class="main-header">💡 Smart Budget Planning</h1>', unsafe_allow_html=True)
    
    # File upload for budget planning
    st.markdown("### 📁 Upload Your UPI Transaction Data")
    budget_file = st.file_uploader(
        "Choose your UPI data file for budget planning",
        type=['csv', 'xlsx'],
        key="budget_file",
        help="Upload your UPI transaction data to get personalized budget advice"
    )
    
    # Input fields for budget planning
    st.markdown("### 💰 Your Financial Information")
    
    col1, col2 = st.columns(2)
    with col1:
        monthly_income = st.number_input(
            "Monthly Income (₹)",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            help="Enter your monthly income in rupees"
        )
    
    with col2:
        duration_months = st.number_input(
            "Planning Duration (months)",
            min_value=1,
            max_value=60,
            value=6,
            step=1,
            help="How many months do you want to plan for?"
        )
    
    # Financial goals section
    st.markdown("### 🎯 Your Financial Goals")
    
    # Initialize goals in session state
    if 'goals' not in st.session_state:
        st.session_state.goals = [{"name": "", "price": 0.0}]
    
    # Function to add/remove goals
    def add_goal():
        st.session_state.goals.append({"name": "", "price": 0.0})
    
    def remove_goal(index):
        if len(st.session_state.goals) > 1:
            st.session_state.goals.pop(index)
    
    # Display goal inputs
    for i, goal in enumerate(st.session_state.goals):
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            goal_name = st.text_input(
                f"Goal {i+1} Name",
                value=goal["name"],
                key=f"goal_name_{i}",
                placeholder="e.g., New Laptop, Vacation, Emergency Fund"
            )
            st.session_state.goals[i]["name"] = goal_name
        
        with col2:
            goal_price = st.number_input(
                f"Target Amount (₹)",
                min_value=0.0,
                value=goal["price"],
                step=1000.0,
                key=f"goal_price_{i}"
            )
            st.session_state.goals[i]["price"] = goal_price
        
        with col3:
            if st.button("❌", key=f"remove_{i}", help="Remove this goal"):
                remove_goal(i)
                st.rerun()
    
    # Add goal button
    if st.button("➕ Add Another Goal"):
        add_goal()
        st.rerun()
    
    # Budget analysis
    if budget_file is not None and monthly_income > 0:
        # Filter out empty goals
        valid_goals = [goal for goal in st.session_state.goals if goal["name"].strip() and goal["price"] > 0]
        
        if valid_goals and st.button("📊 Generate Budget Plan", type="primary"):
            try:
                with st.spinner("Creating your personalized budget plan..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx' if budget_file.name.endswith('.xlsx') else '.csv') as tmp_file:
                        tmp_file.write(budget_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Create budget advisor and generate plan
                    advisor = BudgetAdvisor()
                    budget_plan = advisor.suggest_budget(
                        file_path=tmp_file_path,
                        monthly_income=monthly_income,
                        requirements=valid_goals,
                        duration_months=duration_months
                    )
                    
                    # Clean up temp file
                    os.unlink(tmp_file_path)
                    
                    # Display results
                    st.markdown("## 📋 Your Personalized Budget Plan")
                    
                    # Create tabs for different sections
                    tab1, tab2, tab3 = st.tabs(["📊 Budget Overview", "📝 Detailed Plan", "📥 Download"])
                    
                    with tab1:
                        # Display goals summary
                        st.markdown("### 🎯 Your Financial Goals")
                        total_goals_cost = sum([goal["price"] for goal in valid_goals])
                        monthly_savings_needed = total_goals_cost / duration_months
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Goals Value", f"₹{total_goals_cost:,.2f}")
                        with col2:
                            st.metric("Monthly Savings Needed", f"₹{monthly_savings_needed:,.2f}")
                        with col3:
                            savings_percentage = (monthly_savings_needed / monthly_income) * 100
                            st.metric("Savings Rate Required", f"{savings_percentage:.1f}%")
                        
                        # Goals breakdown
                        st.markdown("#### 📈 Goals Breakdown")
                        goals_df = pd.DataFrame(valid_goals)
                        goals_df['Monthly Savings Needed'] = goals_df['price'] / duration_months
                        
                        fig = px.bar(
                            goals_df, 
                            x='name', 
                            y='price',
                            title="Financial Goals Overview",
                            labels={'price': 'Target Amount (₹)', 'name': 'Goal'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        st.markdown("### 📝 Detailed Budget Recommendations")
                        st.markdown(budget_plan)
                    
                    with tab3:
                        st.markdown("### 📥 Download Your Budget Plan")
                        
                        # Text download
                        st.download_button(
                            label="Download Budget Plan (Text)",
                            data=budget_plan,
                            file_name=f"budget_plan_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
                        
                        # Create summary for JSON download
                        budget_summary = {
                            "monthly_income": monthly_income,
                            "planning_duration_months": duration_months,
                            "goals": valid_goals,
                            "total_goals_cost": total_goals_cost,
                            "monthly_savings_needed": monthly_savings_needed,
                            "savings_rate_required": savings_percentage,
                            "detailed_plan": budget_plan,
                            "generated_date": datetime.now().isoformat()
                        }
                        
                        st.download_button(
                            label="Download Budget Plan (JSON)",
                            data=json.dumps(budget_summary, indent=2),
                            file_name=f"budget_plan_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json"
                        )
                        
            except Exception as e:
                st.error(f"Error generating budget plan: {str(e)}")
        
        elif not valid_goals:
            st.warning("Please add at least one valid financial goal with a name and target amount.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        💰 UPI Financial Analyzer | Built with Streamlit & AI-powered insights
    </div>
    """, 
    unsafe_allow_html=True
)