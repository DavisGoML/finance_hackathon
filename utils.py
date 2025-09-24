from fpdf import FPDF
import pandas as pd
import os
from datetime import datetime

class EnhancedPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(25, 25, 112)
        self.cell(0, 12, 'Comprehensive Financial Analysis Report', 0, 1, 'C')
        self.set_text_color(0, 0, 0)
        self.ln(3)
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 8, f'Generated on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 0, 1, 'C')
        self.set_text_color(0, 0, 0)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} | AI-Powered Financial Insights', 0, 0, 'C')
        self.set_text_color(0, 0, 0)
    
    def section_header(self, title):
        self.ln(8)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(25, 25, 112)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(2)
    
    def subsection_header(self, title):
        self.ln(5)
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(34, 139, 34)
        self.cell(0, 8, title, 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(2)

def generate_pdf(result: dict, sample_df: pd.DataFrame = None, major_spends_df: pd.DataFrame = None, category_summary_df: pd.DataFrame = None, monthly_summary_df: pd.DataFrame = None) -> str:
    """
    Generate an enhanced PDF with comprehensive financial analysis.

    This function is optimized for robustness and readability, handling potential
    KeyErrors and streamlining table generation.
    """
    pdf = EnhancedPDF()
    pdf.add_page()
    
    # Use .get() to safely retrieve data, preventing KeyErrors.
    summary_data = result.get('summary', {})
    narrative_data = result.get('comprehensive_narrative', result.get('narrative', 'No insights available.'))
    chart_path = result.get('chart_path')

    # Helper function for generating tables
    def create_table(header_data, rows_data, column_widths, header_color, row_colors):
        # Set table header
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(*header_color)
        pdf.set_text_color(255, 255, 255)
        for i, header in enumerate(header_data):
            pdf.cell(column_widths[i], 10, header, 1, 0, 'C', True)
        pdf.ln()

        # Set table rows
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(0, 0, 0)
        for i, row in enumerate(rows_data):
            pdf.set_fill_color(row_colors[i % 2], row_colors[i % 2], row_colors[i % 2])
            for j, cell in enumerate(row):
                pdf.cell(column_widths[j], 8, cell, 1, 0, 'C', True)
            pdf.ln()

    # Section: Executive Summary
    pdf.section_header("Executive Summary")
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(240, 248, 255)
    
    summary_items = [
        ("Total Spending", summary_data.get('Total Spending', 'N/A')),
        ("Total Transactions", summary_data.get('Total Transactions', 'N/A')),
        ("Average Transaction", summary_data.get('Average Transaction', 'N/A')),
        ("Top Spending Category", summary_data.get('Top Category', 'N/A')),
        ("Most Frequent Category", summary_data.get('Most Frequent Category', 'N/A'))
    ]
    
    for key, value in summary_items:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(70, 8, f"{key}:", 1, 0, 'L', True)
        pdf.set_font('Helvetica', '', 10)
        sanitized_value = str(value).replace('₹', 'Rs.')
        pdf.cell(120, 8, sanitized_value, 1, 1, 'L')
    pdf.ln(8)
    
    # Section: Financial Narrative
    pdf.section_header("Comprehensive Financial Insights")
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    
    narrative = narrative_data.replace('₹', 'Rs.')
    paragraphs = narrative.split('\n\n')
    for paragraph in paragraphs:
        if paragraph.strip():
            pdf.multi_cell(0, 6, paragraph.strip())
            pdf.ln(3)
    pdf.set_text_color(0, 0, 0)

    # Section: Top 10 Major Transactions
    if major_spends_df is not None and not major_spends_df.empty:
        if pdf.get_y() > 200:
            pdf.add_page()
        pdf.section_header("Top 10 Major Transactions")
        
        headers = ['Date', 'Description', 'Amount', 'Category']
        widths = [25, 85, 25, 35]
        
        # Prepare rows for the table
        rows = major_spends_df.head(10).apply(
            lambda row: [
                str(row.get('Date', 'N/A'))[:10],
                (str(row.get('Description', ''))[:35] + '...'),
                str(row.get('Amount', 'N/A')).replace('₹', 'Rs.'),
                str(row.get('Category', 'N/A'))[:15]
            ], axis=1
        ).tolist()
        
        create_table(headers, rows, widths, (220, 20, 60), [255, 248])
    
    # Section: Category-wise Spending Analysis
    if category_summary_df is not None and not category_summary_df.empty:
        if pdf.get_y() > 220:
            pdf.add_page()
        pdf.section_header("Category-wise Spending Analysis")
        
        headers = ['Category', 'Total Amount', 'Count', 'Average', 'Percentage']
        widths = [45, 30, 25, 30, 25]
        
        # Prepare rows for the table
        rows = category_summary_df.apply(
            lambda row: [
                str(row.name)[:20],
                str(row.get('Total', 'N/A')).replace('₹', 'Rs.'),
                str(row.get('Count', 'N/A')),
                str(row.get('Average', 'N/A')).replace('₹', 'Rs.'),
                str(row.get('Percentage', 'N/A'))
            ], axis=1
        ).tolist()
        
        create_table(headers, rows, widths, (34, 139, 34), [248, 255])

    # Section: Monthly Spending Trends
    if monthly_summary_df is not None and not monthly_summary_df.empty:
        if pdf.get_y() > 240:
            pdf.add_page()
        pdf.section_header("Monthly Spending Trends")
        
        headers = ['Month', 'Total Spending']
        widths = [60, 60]
        
        # Prepare rows for the table
        rows = monthly_summary_df.apply(
            lambda row: [
                str(row.get('Month', 'N/A')),
                str(row.get('Amount', 'N/A')).replace('₹', 'Rs.')
            ], axis=1
        ).tolist()
        
        create_table(headers, rows, widths, (255, 140, 0), [248, 255])
    
    # Section: Spending Visualization
    if chart_path and os.path.exists(chart_path):
        if pdf.get_y() > 150:
            pdf.add_page()
        pdf.section_header("Spending Visualization")
        try:
            chart_width = 180
            x = (210 - chart_width) / 2
            pdf.image(chart_path, x=x, y=pdf.get_y(), w=chart_width)
            pdf.ln(pdf.get_y() + 10)
        except Exception as e:
            pdf.set_font('Helvetica', 'I', 10)
            pdf.cell(0, 8, f"Chart could not be displayed: {str(e)}", 0, 1, 'C')

    # Section: Recent Transactions Sample
    if sample_df is not None and not sample_df.empty:
        if pdf.get_y() > 200:
            pdf.add_page()
        pdf.section_header("Sample Recent Transactions")
        
        headers = ['Date', 'Description', 'Amount', 'Category']
        widths = [25, 85, 25, 35]

        # Prepare rows for the table
        rows = sample_df.head(15).apply(
            lambda row: [
                str(row.get('Date', 'N/A').date()) if hasattr(row.get('Date'), 'date') else str(row.get('Date', 'N/A'))[:10],
                str(row.get('Description', ''))[:40] + '...' if len(str(row.get('Description', ''))) > 40 else str(row.get('Description', 'N/A')),
                f"Rs.{row.get('Amount', 0):.2f}",
                str(row.get('Category', 'N/A'))[:15]
            ], axis=1
        ).tolist()

        create_table(headers, rows, widths, (75, 0, 130), [248, 255])
    
    # Section: Key Financial Recommendations
    pdf.add_page()
    pdf.section_header("Key Financial Recommendations")
    pdf.set_font('Helvetica', '', 10)
    recommendations = [
        "📊 Review your spending patterns regularly to maintain financial awareness",
        "💰 Focus on reducing expenses in your highest spending categories",
        "🎯 Set monthly budgets for each spending category",
        "📈 Track monthly spending trends to identify seasonal patterns",
        "⚠️ Monitor large transactions and verify their necessity",
        "💡 Consider automated savings based on your spending patterns",
        "📱 Use expense tracking apps to maintain real-time awareness"
    ]
    for rec in recommendations:
        rec_clean = rec.replace('•', '-').replace('📊', '-').replace('💰', '-').replace('🎯', '-').replace('📈', '-').replace('⚠️', '-').replace('💡', '-').replace('📱', '-')
        pdf.multi_cell(0, 7, rec_clean)
        pdf.ln(2)
    
    # Output the PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"Enhanced_Financial_Analysis_Report_{timestamp}.pdf"
    
    try:
        pdf.output(output_path)
        return output_path
    except Exception as e:
        fallback_path = "Enhanced_Financial_Analysis_Report.pdf"
        pdf.output(fallback_path)
        return fallback_path