import pandas as pd
import streamlit as st
import re
import requests
import json
import yfinance as yf
from datetime import datetime, timedelta
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import shutil # Import shutil for cleaning up temporary directory

# --- Your existing functions (unmodified) ---

def find_isin_column(df):
    isin_pattern = r'^[A-Z]{2}[A-Z0-9]{10}$'
    
    for col in df.columns:
        # Drop NA and convert to string
        sample_values = df[col].dropna().astype(str)
        
        # Check if at least 5 values match the ISIN pattern
        matches = sample_values.apply(lambda x: bool(re.match(isin_pattern, x)))
        if matches.sum() >= 5:
            return col  
    
    return None  

def format_currency(value):
    if abs(value) >= 10000000:
        return f'{value / 10000000:.2f} Crs'
    elif abs(value) >= 100000:
        return f"{value / 100000:.2f} L"
    elif abs(value) >= 1000:
        return f"{value / 1000:.2f} K"
    else:
        return f"{value:.2f}"

def analyze_portfolio_data(equity_df):
    """Analyze portfolio data and return key metrics"""
    analysis = {}
    
    # Basic portfolio stats
    analysis['total_stocks'] = len(equity_df)
    analysis['sectors'] = equity_df['Industry Name'].nunique()
    
    # Market caps - handle potential non-numeric conversion if needed, but your current code should be okay
    # Ensure MarketCap is properly categorized (e.g., Small Cap, Mid Cap, Large Cap)
    # For now, let's just count unique values if they are already categorized strings
    analysis['market_caps'] = equity_df['MarketCap'].value_counts().to_dict()
    
    # Performance analysis
    valid_1y_returns = equity_df['1Y Return (%)'].dropna()
    analysis['avg_1y_return'] = valid_1y_returns.mean() if len(valid_1y_returns) > 0 else 0
    analysis['best_performer'] = equity_df.loc[equity_df['1Y Return (%)'].idxmax()] if len(valid_1y_returns) > 0 else None
    analysis['worst_performer'] = equity_df.loc[equity_df['1Y Return (%)'].idxmin()] if len(valid_1y_returns) > 0 else None
    
    # Risk analysis
    valid_beta = equity_df['Beta 3Year'].dropna()
    analysis['avg_beta'] = valid_beta.mean() if len(valid_beta) > 0 else 0
    analysis['high_risk_stocks'] = len(equity_df[equity_df['Beta 3Year'] > 1.2])
    analysis['low_risk_stocks'] = len(equity_df[equity_df['Beta 3Year'] < 0.8])
    
    # Valuation analysis
    valid_pe = equity_df['PE TTM Price to Earnings'].dropna()
    analysis['avg_pe'] = valid_pe.mean() if len(valid_pe) > 0 else 0
    analysis['overvalued_stocks'] = len(equity_df[equity_df['PE TTM Price to Earnings'] > 30])
    analysis['undervalued_stocks'] = len(equity_df[equity_df['PE TTM Price to Earnings'] < 15])
    
    # Financial health
    valid_roe = equity_df['ROE Annual %'].dropna()
    analysis['avg_roe'] = valid_roe.mean() if len(valid_roe) > 0 else 0
    analysis['high_roe_stocks'] = len(equity_df[equity_df['ROE Annual %'] > 20])
    
    # Sector analysis
    sector_analysis = equity_df.groupby('Industry Name').agg({
        'Stock Name': 'count',
        '1Y Return (%)': 'mean',
        'PE TTM Price to Earnings': 'mean',
        'ROE Annual %': 'mean'
    }).round(2)
    analysis['sector_breakdown'] = sector_analysis.to_dict('index')
    
    return analysis

def create_portfolio_charts(equity_df, temp_dir , portfolio_df,nifty100_data,niftymidcap_data,smallcap_df):
    """Create various charts for portfolio analysis"""
    chart_paths = {}
    
    # 1. Sector Distribution Pie Chart
    plt.figure(figsize=(5, 6))
    sector_counts = equity_df['Industry Name'].value_counts()
    top10 = sector_counts[:10]
    others = sector_counts[10:].sum()
    if others > 0:
            top10['Others'] = others

    colors_list = plt.cm.tab20(np.linspace(0, 1, len(top10)))

    plt.pie( top10.values, labels=top10.index, autopct='%1.1f%%', colors=colors_list,startangle=90)
    plt.title('Sector Distribution', fontsize=14, fontweight='bold')
    plt.axis('equal')
    chart_paths['sector_pie'] = os.path.join(temp_dir, "sector_distribution.png")
    plt.savefig(chart_paths['sector_pie'], bbox_inches='tight', dpi=200)
    plt.close()
    
    # 2. Performance vs Risk Scatter Plot
    plt.figure(figsize=(5, 3))
    Portfolio_1y_Return = (portfolio_df['1Y Return (%)'].astype(float) * portfolio_df['Weightage'].astype(float) / 100).sum().round(2)
    Portfolio_SD = (portfolio_df['Annualized Standard Deviation (%)'].astype(float) * portfolio_df['Weightage'].astype(float) / 100).sum().round(2)
    Index_Equiweighted_Return = (
    portfolio_df[['Nifty100_1y_Return (%)',
                  'Nifty Midcap150_1y_Return (%)',
                  'Nifty_Smallcap_1y_Return(%)']]
    .iloc[0].astype(float).mean().round(2))

    nifty100_std_dev = nifty100_data['Close'].pct_change().std() * np.sqrt(252) * 100
    niftymidcap_std_dev = niftymidcap_data['Close'].pct_change().std() * np.sqrt(252) * 100
    niftysmallcap_std_dev = smallcap_df['nav'].pct_change().std() * np.sqrt(252) * 100
    equiweight_index_std_dev = ((nifty100_std_dev + niftymidcap_std_dev + niftysmallcap_std_dev) / 3).round(2)

    plt.scatter(Portfolio_SD, Portfolio_1y_Return, color='steelblue', s=60, alpha=0.8, label='Portfolio')
    plt.scatter(equiweight_index_std_dev, Index_Equiweighted_Return, color='orange', s=60, alpha=0.8, label='Equiweighted Index')

    plt.xlabel('Risk (Standard Deviation %)')
    plt.ylabel('1-Year Return (%)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    plt.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.15),  
    ncol=2,    frameon=True)
    plt.subplots_adjust(top=0.8)

    chart_paths['risk_return'] = os.path.join(temp_dir, "risk_return_scatter.png")
    plt.savefig(chart_paths['risk_return'], bbox_inches='tight', dpi=200)
    plt.close()
    
    # 3. Market Cap Distribution
    plt.figure(figsize=(5, 3))

    market_cap_counts = equity_df['MarketCap'].value_counts()
    total = market_cap_counts.sum()  # total number of stocks

    bars = plt.bar(
    market_cap_counts.index,
    market_cap_counts.values,
    color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
    width=0.4)

    plt.ylabel('Number of Stocks')
    plt.xlabel('Market Cap Category')

    for bar, count in zip(bars, market_cap_counts.values):
        height = bar.get_height()
        percentage = (count / total) * 100
        plt.text(
        bar.get_x() + bar.get_width() / 2.,
        height + 0.1,
        f'{percentage:.1f}%',  # percentage with 1 decimal place
        ha='center',
        va='bottom',
        fontsize=8)

    chart_paths['market_cap'] = os.path.join(temp_dir, "market_cap_distribution.png")
    plt.savefig(chart_paths['market_cap'], bbox_inches='tight', pad_inches=0.5, dpi=200)
    plt.close()

    
    # 4. ROE Distribution Histogram
    plt.figure(figsize=(5, 3))
    valid_roe = equity_df['ROE Annual %'].dropna()
    if len(valid_roe) > 0:
        counts, bins, patches = plt.hist(valid_roe, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        for count, patch in zip(counts, patches):
           height = patch.get_height()
           if height > 0:
             plt.text(patch.get_x() + patch.get_width() / 2, height, 
                     f'{int(count)}', ha='center', va='bottom', fontsize=8)
        plt.axvline(valid_roe.mean(), color='red', linestyle='--', 
                    label=f'Average ROE: {valid_roe.mean():.1f}%')
        plt.xlabel('ROE (%)')
        plt.ylabel('Number of Stocks')
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15),   ncol=2,    frameon=True)
        plt.subplots_adjust(top=0.8)
        plt.grid(True, alpha=0.3)
    chart_paths['roe_dist'] = os.path.join(temp_dir, "roe_distribution.png")
    plt.savefig(chart_paths['roe_dist'], bbox_inches='tight', dpi=200)
    plt.close()
    
    return chart_paths
def draw_footer(canvas, doc):
    """Draw footer with disclaimer text at the bottom of each page."""
    canvas.saveState()
    footer_text = (
        "*Disclaimer: Investments in equity shares, are not obligations of or guaranteed by us, "
        "and are subject to investment risks. The data and analysis shared in this document are offered "
        "as part of our enhanced service offerings. This is not an explicit recommendation for any direct "
        "stock transactions. Our suggestions are not intended to serve as a substitute for professional "
        "investment advice."
    )
    footer_style = ParagraphStyle(
        name='FooterStyle',
        fontSize=7,
        textColor=colors.grey,
        alignment=TA_CENTER,
        leading=9 )

    # Create a paragraph and wrap it to fit page width
    p = Paragraph(footer_text, footer_style)
    w, h = p.wrap(doc.width, doc.bottomMargin)
    p.drawOn(canvas, doc.leftMargin, h + 5)  # 5 points above bottom
    canvas.restoreState()

def create_enhanced_investment_report(equity_df):
    """Create enhanced investment report with portfolio analysis, improved formatting."""

    def draw_border(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(1.5) # Slightly thinner border
        margin = 25 # Increased margin for better look
        canvas.rect(margin, margin, doc.width + doc.leftMargin + doc.rightMargin - 2 * margin,
                    doc.height + doc.topMargin + doc.bottomMargin - 2 * margin)
        canvas.restoreState()

    try:
        # Create temporary directory for charts and PDF
        temp_dir = tempfile.mkdtemp()
        filename = f"Enhanced_Investment_Report.pdf"
        output_path = os.path.join(temp_dir, filename)
        
        doc = SimpleDocTemplate(
            output_path, 
            pagesize=letter, 
            topMargin=0.75*inch, 
            bottomMargin=0.75*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles with improved spacing and fonts
        title_style = ParagraphStyle(
            'Title',
            parent=styles['h1'],
            fontSize=24,
            leading=28, # Line spacing
            alignment=TA_CENTER,
            spaceAfter=24, # More space after title
            textColor=colors.darkblue,)
            
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['h2'],
            fontSize=16,
            leading=18,
            spaceBefore=18, 
            spaceAfter=10, 
            textColor=colors.darkgreen,
        )
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=10,
            leading=14, 
            spaceAfter=6,
        )
        
        # Specific style for portfolio analysis section heading
        portfolio_section_style = ParagraphStyle(
            'PortfolioSection',
            parent=styles['h1'],
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceBefore=30, # More space before this major section
            spaceAfter=18,
            textColor=colors.darkred,
        )
        
        key_metric_style = ParagraphStyle(
            'KeyMetric',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.darkgreen,
            leftIndent=30, # Increased indent for bullet points
            spaceAfter=4,
            bulletText='• ' # Add a proper bullet
        )

        # --- Report Content ---

        # Company Logo
        logo_path = "Armstrong_logo.png"
        if os.path.exists(logo_path):
            logo = Image(logo_path, width=7.5 * inch, height=1 * inch)
            logo.hAlign = 'CENTER' # Center the logo
            elements.append(logo)
            elements.append(Spacer(1, 24)) # More space after logo
        else:
            elements.append(Paragraph("Company Logo Missing (Place Armstrong_logo.png in the same directory)", normal_style))
            elements.append(Spacer(1, 12))

        # Report Header
        report_title = f"Investment Portfolio Analysis"
        elements.append(Paragraph(report_title, title_style))
        elements.append(Spacer(1, 12))

        # Name & Date in same row
        name_text = f"Client Name: {client_name}"

        date_text = datetime.now().strftime("%B %d, %Y")

        data = [[name_text, date_text]]
        table = Table(data, colWidths=[4*inch, 2.5*inch])  # Adjust column widths
        table.setStyle(TableStyle([
    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # --- PORTFOLIO ANALYSIS SECTION ---
        
        # Analyze portfolio data
        portfolio_analysis = analyze_portfolio_data(equity_df)
        
        # Key Portfolio Metrics Summary
        elements.append(Paragraph("Executive Summary", subtitle_style))
        
        metrics = [ ("Total Stocks", str(portfolio_analysis['total_stocks']), "Diversification base"),
     ("Sectors Covered", str(portfolio_analysis['sectors']), "Sector diversification"),
     ("Average 1Y Return", f"{portfolio_analysis['avg_1y_return']:.2f}%", "Portfolio performance"),
     ("Average PE Ratio", f"{portfolio_analysis['avg_pe']:.2f}", "Valuation level"),
     #("Average ROE", f"{portfolio_analysis['avg_roe']:.2f}%", "Profitability measure"),
     ("Average Beta", f"{portfolio_analysis['avg_beta']:.2f}", "Market risk exposure"),
     ("High ROE Stocks (>20%)", str(portfolio_analysis['high_roe_stocks']), "Quality companies")]
        
        metric_style = ParagraphStyle(name='Metric', fontSize=8, textColor=colors.grey, leading=10)
        value_style = ParagraphStyle(name='Value', fontSize=14, textColor=colors.darkgreen, leading=16)
        insight_style = ParagraphStyle(name='Insight', fontSize=7, textColor=colors.black, leading=9)

        # Build card-style tables 
        cards_per_row = 3
        card_width = 170
        card_rows = []
        card_row = []

        for i, (metric, value, insight) in enumerate(metrics):
           card = Table([  [Paragraph(metric, metric_style)], [Paragraph(value, value_style)],  [Paragraph(insight, insight_style)] ], colWidths=card_width)

           card.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F9FAFB')),
           ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#D1D5DB')),      
           ('LEFTPADDING', (0, 0), (-1, -1), 10),
           ('RIGHTPADDING', (0, 0), (-1, -1), 10),
           ('TOPPADDING', (0, 0), (-1, -1), 8),
           ('BOTTOMPADDING', (0, 0), (-1, -1), 8)]))

           card_row.append(card)

           if len(card_row) == cards_per_row:
                card_rows.append(card_row)
                card_row = []

        # Add leftover row if any
        if card_row:
          card_rows.append(card_row)
 
        # Add grid of cards to elements
        for row in card_rows:
             elements.append(Table([row], colWidths=[card_width]*len(row), hAlign='LEFT'))
             elements.append(Spacer(1, 10))
        elements.append(Spacer(1, 20)) # Increased spacing

        #Comparision with Nifty Indices
        nifty100_6m = equity_df.loc[0, 'Nifty100_6m_Return(%)']
        niftymid_6m = equity_df.loc[0, 'Nifty_midcap150_6m_Return(%)']
        nifty100_1y = equity_df.loc[0, 'Nifty100_1y_Return (%)']
        niftymid_1y = equity_df.loc[0, 'Nifty Midcap150_1y_Return (%)'] 
        nifty100_3m = equity_df.loc[0, 'Nifty100_3m_Return(%)']
        niftymid_3m = equity_df.loc[0, 'Nifty_midcap150_3m_Return(%)']
        nifty100_2y = equity_df.loc[0, 'Nifty100_2y_Return(%)']
        niftymid_2y = equity_df.loc[0, 'Nifty_midcap150_2y_Return(%)']
        nifty100_3y = equity_df.loc[0, 'Nifty100_3y_Return(%)']
        niftymid_3y = equity_df.loc[0, 'Nifty_midcap150_3y_Return(%)']
        niftysmallcap_6m = equity_df.loc[0, 'Nifty_Smallcap_6m_Return(%)']
        niftysmallcap_3m = equity_df.loc[0, 'Nifty_Smallcap_3m_Return(%)']
        niftysmallcap_1y = equity_df.loc[0, 'Nifty_Smallcap_1y_Return(%)']
        niftysmallcap_2y = equity_df.loc[0, 'Nifty_Smallcap_2y_Return(%)']
        niftysmallcap_3y = equity_df.loc[0, 'Nifty_Smallcap_3y_Return(%)']

        Portfolio_1y_Return = (portfolio_df['1Y Return (%)'] * portfolio_df['Weightage'].astype(float)/100).sum().round(2)
        Portfolio_3m_Return = (portfolio_df['3M Return (%)'] * portfolio_df['Weightage'].astype(float)/100).sum().round(2)
        Portfolio_6m_Return = (portfolio_df['6M Return (%)'] * portfolio_df['Weightage'].astype(float) /100).sum().round(2)
        Portfolio_2y_Annulized_Return = (portfolio_df['2Y Annualized Return (%)'] * portfolio_df['Weightage'].astype(float) / 100).sum().round(2)
        Portfolio_3y_Annualized_Return= (portfolio_df['3Y Annualized Return (%)'] * portfolio_df['Weightage'].astype(float) / 100).sum().round(2)
        benchmark_data = [ ["Index","3M","6M","1Y","2Y"],
         ["Nifty 100",nifty100_3m,nifty100_6m,nifty100_1y, nifty100_2y],
         ["Nifty Midcap 150",niftymid_3m,niftymid_6m, niftymid_1y, niftymid_2y],
         ["Nifty Smallcap 250",niftysmallcap_3m,niftysmallcap_6m,niftysmallcap_1y,niftysmallcap_2y],
         ['Portfolio',Portfolio_3m_Return,Portfolio_6m_Return,Portfolio_1y_Return,Portfolio_2y_Annulized_Return]]

        benchmark_table = Table(benchmark_data, colWidths=[2.2*inch, 1*inch])
        benchmark_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F618D')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),]))

        elements.append(Paragraph("Performance Highlights", subtitle_style))
        elements.append(Spacer(1, 10))
        elements.append(benchmark_table)
        elements.append(Spacer(1, 20))

        # Best and Worst Performers
        if portfolio_analysis['best_performer'] is not None:
            title_style = ParagraphStyle(name='Title', fontSize=10, textColor=colors.black, spaceAfter=4)
            value_style = ParagraphStyle(name='Value', fontSize=10, textColor=colors.black)
            highlight_best = ParagraphStyle(name='Best', fontSize=10, textColor=colors.HexColor('#145A32'))
            highlight_worst = ParagraphStyle(name='Worst', fontSize=10, textColor=colors.HexColor('#922B21'))

            # Best Performer Data
            best = portfolio_analysis['best_performer']
            best_row = [ Paragraph("🏆 <b>Best Performing Stock</b>", highlight_best),Paragraph(best['Stock Name'], value_style),Paragraph(best['Industry Name'], value_style),
            Paragraph(f"1Y Return: {best['1Y Return (%)']:.2f}%", highlight_best),Paragraph(f"PE: {best['PE TTM Price to Earnings']:.2f}", value_style)]
            best_table = Table([best_row], colWidths=[1.8*inch, 1.2*inch, 1.5*inch, 1.4*inch, 1*inch]) 
            best_table.setStyle(TableStyle([ ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#E8F8F5')),
              ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#2ECC71')),
              ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
              ('LEFTPADDING', (0, 0), (-1, -1), 6),
              ('RIGHTPADDING', (0, 0), (-1, -1), 6), ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),]))
            elements.append(best_table)
            elements.append(Spacer(1, 10))

            # Worst Performer Data
            worst = portfolio_analysis['worst_performer']
            worst_row = [
    Paragraph("📉 <b>Worst Performing Stock</b>", highlight_worst),
    Paragraph(worst['Stock Name'], value_style),
    Paragraph(worst['Industry Name'], value_style),
    Paragraph(f"1Y Return: {worst['1Y Return (%)']:.2f}%", highlight_worst),
    Paragraph(f"PE: {worst['PE TTM Price to Earnings']:.2f}", value_style)]
            worst_table = Table([worst_row], colWidths=[1.8*inch, 1.2*inch, 1.5*inch, 1.4*inch, 1*inch])
            worst_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FDEDEC')),
    ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#E74C3C')),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5)],))
            elements.append(worst_table)
            elements.append(Spacer(1, 20))

        # Risk Analysis
        elements.append(Paragraph("Risk Profile Analysis", subtitle_style))
        elements.append(Spacer(1, 6))
        risk_table_data = [ [
        Paragraph("<b>Metric</b>", normal_style),
        Paragraph("<b>Value</b>", normal_style),
        Paragraph("<b>Insight</b>", normal_style),  ],
        [
        "High Risk Stocks (β > 1.2)",
        f"{portfolio_analysis['high_risk_stocks']} stocks",
        "Market-sensitive picks",  ],
        [
        "Low Risk Stocks (β < 0.8)",
        f"{portfolio_analysis['low_risk_stocks']} stocks",
        "Defensive positioning", ],
        [
        "Average Portfolio Beta",
        f"{portfolio_analysis['avg_beta']:.2f}",
        "Overall risk level",  ]]

        risk_table = Table(risk_table_data, colWidths=[2.3*inch, 1.5*inch, 2.2*inch])
        risk_table.setStyle(TableStyle([
          ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
          ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
          ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
          ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
          ('FONTSIZE', (0, 0), (-1, 0), 10),
          ('FONTSIZE', (0, 1), (-1, -1), 9),
          ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F4F6F7')]),
          ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D5D8DC')),
          ('LEFTPADDING', (0, 0), (-1, -1), 6),
          ('RIGHTPADDING', (0, 0), (-1, -1), 6),
          ('TOPPADDING', (0, 0), (-1, -1), 5),
          ('BOTTOMPADDING', (0, 0), (-1, -1), 5),]))

        elements.append(risk_table)
        elements.append(Spacer(1, 24))

        # Create and add portfolio charts
        try:
            chart_paths = create_portfolio_charts(equity_df, temp_dir,portfolio_df,nifty100_data,niftymidcap_data,smallcap_df)
            
            # Sector Distribution Chart
            elements.append(Paragraph("Portfolio Composition - Sector Distribution", subtitle_style))
            if 'sector_pie' in chart_paths:
                elements.append(Image(chart_paths['sector_pie'], width=6.5*inch, height=4.5*inch, hAlign='CENTER'))
                elements.append(Spacer(1, 18))
            
            # Page break before more charts
            elements.append(PageBreak())
            
            # Risk-Return Analysis
            elements.append(Paragraph("Performance vs. Risk Analysis", subtitle_style))
            if 'risk_return' in chart_paths:
                elements.append(Image(chart_paths['risk_return'], width=6.5*inch, height=4*inch, hAlign='CENTER'))
                elements.append(Spacer(1, 18))
            
            # Market Cap Distribution
            elements.append(Paragraph("Market Capitalization Distribution", subtitle_style))
            if 'market_cap' in chart_paths:
                elements.append(Image(chart_paths['market_cap'], width=6.5*inch, height=4*inch, hAlign='CENTER'))
                elements.append(Spacer(1, 18))

            # ROE Distribution
            elements.append(Paragraph("Return on Equity (ROE) Distribution", subtitle_style))
            if 'roe_dist' in chart_paths:
                elements.append(Image(chart_paths['roe_dist'], width=6.5*inch, height=4*inch, hAlign='CENTER'))
                elements.append(Spacer(1, 18))


            # High Retuns Low Weightage Stocks & Low Returns High Weightage Stocks
            elements.append(Paragraph("High Returns, Low Weightage Stocks", subtitle_style))
            high_return_thresh = portfolio_df['1Y Return (%)'].astype(float).median()
            low_return_thresh = portfolio_df['1Y Return (%)'].astype(float).median()
            high_weight_thresh = portfolio_df['Weightage'].astype(float).median()
            low_weight_thresh = portfolio_df['Weightage'].astype(float).median()

            high_return_low_weight = portfolio_df[(portfolio_df['1Y Return (%)'].astype(float) > high_return_thresh) &
            (portfolio_df['Weightage'].astype(float) < low_weight_thresh) & (portfolio_df['1Y Return (%)'].astype(float) >= 1)   # filter out returns < 1%
            ][['Stock Name', '1Y Return (%)', 'Weightage']]

            low_return_high_weight = portfolio_df[(portfolio_df['1Y Return (%)'].astype(float) < low_return_thresh) &
            (portfolio_df['Weightage'].astype(float) > high_weight_thresh)][['Stock Name', '1Y Return (%)', 'Weightage']]

            if not high_return_low_weight.empty:
                hr_lw_data = [["Stock Name", "1Y Return (%)", "Weightage"]] + high_return_low_weight.values.tolist()
                hr_lw_table = Table(hr_lw_data, colWidths=[3.5*inch, 1.5*inch, 1.5*inch])
                hr_lw_table.setStyle(TableStyle([ ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                               ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('FONTSIZE', (0, 0), (-1, 0), 10),
                               ('FONTSIZE', (0, 1), (-1, -1), 9),
                               ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F4F6F7')]),
                               ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D5D8DC')), ]))
                elements.append(hr_lw_table)
                elements.append(Spacer(1, 18))

                # Low Returns, High Weightage table
                elements.append(Paragraph("Low Returns, High Weightage Stocks", subtitle_style))
                if not low_return_high_weight.empty:
                      lr_hw_data = [["Stock Name", "1Y Return (%)", "Weightage"]] + low_return_high_weight.values.tolist()
                      lr_hw_table = Table(lr_hw_data, colWidths=[3.5*inch, 1.5*inch, 1.5*inch])
                      lr_hw_table.setStyle(TableStyle([ ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#C0392B')),
                      ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                      ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                      ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                      ('FONTSIZE', (0, 0), (-1, 0), 10),
                      ('FONTSIZE', (0, 1), (-1, -1), 9),
                      ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F4F6F7')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D5D8DC')),]))
                elements.append(lr_hw_table)
                elements.append(Spacer(1, 18))

         
            
        except Exception as e:
            elements.append(Paragraph(f"Chart Generation Error: {e}", normal_style))
            elements.append(Spacer(1, 12))
        
        # Sector-wise Analysis Table
        elements.append(Paragraph("Detailed Portfolio Analysis", subtitle_style))
        equity_df_short = equity_df[['Stock Name', '1Y Return (%)', 'Weightage', 'Score','Revised Score']].copy()
        equity_df_short['Score'] = equity_df_short['Score'].round(2)
        equity_df_short['Revised Score'] = equity_df_short['Revised Score'].round(2)
        equity_df_short = equity_df_short.sort_values(by='Revised Score', ascending=False)
        header = ['Stock Name', '1Y Return (%)', 'Weightage', 'Score', 'Revised Score']
        data_rows = equity_df_short.astype(str).values.tolist()
        table_data = [header] + data_rows
        portfolio_table = Table(table_data, colWidths=[2.8 * inch, 1.2 * inch, 1 * inch, 1 * inch])
        portfolio_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')), # Dark blue header
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ECF0F1'), colors.white]), # Light grey/white rows
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4) ]))
        elements.append(portfolio_table)
        elements.append(Spacer(1, 24))

        # Add a note about the scoring system
        scoring_note = """
        <b>Note on Scoring Framework</b><br/><br/>
         The stock scores presented in this report are derived from a structured multi-factor framework that evaluates companies across four key dimensions: <b>Risk, Returns, Valuation, and Profitability & Growth</b>. 
         The objective is to provide a balanced measure of financial strength and consistency.<br/><br/>

        <b>Scoring Criteria</b><br/>
         - <b>Risk</b>: Beta ≤ 1, Standard Deviation ≤ 20%, Debt-to-Capital ≤ 40%<br/>
         - <b>Returns</b>: ROE ≥ 12%, ROA ≥ 6%, ROCE > 12%<br/>
         - <b>Valuation</b>: PEG ≤ Sector PEG, PE ≤ Sector PE<br/>
         - <b>Profitability & Growth</b>: Net Profit CAGR ≥ 12% (3 years), Net Profit Margin ≥ 10%<br/><br/>

        <b>Interpretation</b><br/>
         - A <b>higher score</b> indicates stronger fundamentals, with the company demonstrating consistent performance across multiple dimensions.<br/>
         - A <b>lower score</b> highlights areas of relative weakness.<br/>"""

        note_paragraph = Paragraph(scoring_note, normal_style)

        note_table = Table([[note_paragraph]], colWidths=[450])  # adjust width as per your page

        note_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),   ('BOX', (0, 0), (-1, -1), 1, colors.grey), ('INNERPADDING', (0, 0), (-1, -1), 8),   ]))
        elements.append(note_table)

        # Build PDF with page border
        doc.build( elements,
       onFirstPage=lambda c, d: (draw_border(c, d), draw_footer(c, d)),
       onLaterPages=lambda c, d: (draw_border(c, d), draw_footer(c, d)))
        
        # Read the generated PDF into bytes
        with open(output_path, "rb") as f:
            pdf_bytes = f.read()

        # Clean up the temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_e:
            print(f"Error cleaning up temporary directory: {cleanup_e}")
        
        return pdf_bytes
    except Exception as e:
        st.error(f"Error generating PDF report: {e}")
        return None


def fetch_mf_data_to_dataframe(url: str) -> pd.DataFrame:
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'data' in data and isinstance(data['data'], list):
            df = pd.DataFrame(data['data'])
            print("Data successfully fetched and stored in DataFrame.")
            return df
        else:
            print(f"Error: 'data' key not found or not a list in the JSON response from {url}")
            return pd.DataFrame() # Return an empty DataFrame
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return pd.DataFrame()
    
# --- Streamlit Application (remains the same) ---

st.title("📂 Upload CSV or Excel File")
client_name = st.text_input("Enter Client Name:", value="Client")
# File uploader widget
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx","xls"])
valid_isins = []

if uploaded_file is not None:
    try:
        # Read Excel or CSV depending on file type
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")
        st.write("Here's a preview of your data:")
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        isin_col = find_isin_column(df)
        if isin_col:
            isin_pattern = r'^[A-Z]{2}[A-Z0-9]{10}$'
            isin_values = df[isin_col].dropna().unique().astype(str)
            valid_isins = [isin for isin in isin_values if re.match(isin_pattern, isin.strip().upper())]       

    except Exception as e:
        st.error(f"Error reading file: {e}")

    # Load Trendlyne and AMFI data
    try:
        trendlyne_data = pd.read_excel("combined_trendlyne_data.xlsx", engine='openpyxl', skiprows=0)
        AMFI_data = pd.read_excel("AMFI-MARKETCAP-DATA-30Jun2025.xlsx", engine='openpyxl', skiprows=1)
    except FileNotFoundError:
        st.error("Missing data files. Please ensure 'combined_trendlyne_data.xlsx' and 'AMFI-MARKETCAP-DATA-30Jun2025.xlsx' are in the same directory as your Streamlit app.")
        st.stop()

    portfolio_df = trendlyne_data[trendlyne_data['ISIN'].isin(valid_isins)].copy()
    
    # Merge Market Cap data
    AMFI_data_filtered = AMFI_data[AMFI_data['ISIN'].isin(valid_isins)]
    portfolio_df = pd.merge(portfolio_df, AMFI_data_filtered[['ISIN', 'Market Cap']], on='ISIN', how='left')
    portfolio_df.rename(columns={'Market Cap': 'MarketCap'}, inplace=True)


    portfolio_df = portfolio_df.fillna(0)
    portfolio_df['Stock Code'] = portfolio_df['Stock Code'].astype(str) + '.NS'
    portfolio_df.loc[9, 'Stock Code'] = portfolio_df.loc[9, 'Stock Code'].replace('.NS', '.BO')
    df_quantity = df[[isin_col, 'Quantity']].copy()
    df_quantity.rename(columns={isin_col: 'ISIN'}, inplace=True)
    portfolio_df = pd.merge(portfolio_df, df_quantity, on='ISIN', how='left')
    three_months_returns=[]
    six_month_returns = []
    one_year_returns = []
    two_years_annualized_returns = []
    three_year_annualized_returns = []
    std_devs = []
    # Add a progress bar for fetching stock data
    progress_text = "Fetching stock data. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, row in portfolio_df.iterrows():
        stocks = row['Stock Code']
        stk_qty = row['Quantity']
        ticker = yf.Ticker(stocks)
        try:
            # Fetch 1 year of data for returns and standard deviation
            # ** MODIFIED: Fetched history for 1 year to calculate returns and volatility **
            stock_data = ticker.history(period="max") 
            stock_data.reset_index(inplace=True)
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            today = stock_data['Date'].max()
            price_today = stock_data.loc[stock_data['Date'] == today, 'Close'].values[0]
           
                        
            def get_closest_price(target_date):
                subset = stock_data[stock_data['Date'] <= target_date]
                if not subset.empty:
                    return subset.iloc[-1]['Close']
                return None
            
            three_months_ago = today - timedelta(days=90)
            six_months_ago = today - timedelta(days=182)
            one_year_ago = today - timedelta(days=365)
            three_years_ago = today - timedelta(days=3 * 365)

            price_3m = get_closest_price(three_months_ago)
            ret_3m = (price_today / price_3m - 1) if price_3m else None

            price_6m = get_closest_price(six_months_ago)
            ret_6m = (price_today / price_6m - 1) if price_6m else None

            price_1y = get_closest_price(one_year_ago)
            ret_1y = (price_today / price_1y - 1) if price_1y else None
            
            # ** MODIFIED: Added fetch for 3 years to calculate annualized returns **
            stock_data_3y = ticker.history(period="max")
            price_3y = stock_data_3y.loc[stock_data_3y.index < three_years_ago, 'Close'].iloc[-1] if not stock_data_3y.loc[stock_data_3y.index < three_years_ago, 'Close'].empty else None
            ret_3y_ann = ((price_today / price_3y) ** (1 / 3) - 1) if price_3y else None

            price_2y = stock_data_3y.loc[stock_data_3y.index < (today - timedelta(days=730)), 'Close'].iloc[-1] if not stock_data_3y.loc[stock_data_3y.index < (today - timedelta(days=730)), 'Close'].empty else None
            ret_2y_ann = ((price_today / price_2y) ** (1 / 2) - 1) if price_2y else None

            # ** NEW: Calculate daily returns for the last year for standard deviation **
            daily_returns = stock_data['Close'].pct_change().dropna()
            
            # ** NEW: Calculate annualized standard deviation (volatility) **
            annualized_std_dev = daily_returns.std() * np.sqrt(252) * 100
            
            three_months_returns.append(round(ret_3m * 100, 2) if ret_3m is not None else 0)
            six_month_returns.append(round(ret_6m * 100, 2) if ret_6m is not None else 0)
            one_year_returns.append(round(ret_1y * 100, 2) if ret_1y is not None else 0)
            three_year_annualized_returns.append(round(ret_3y_ann * 100, 2) if ret_3y_ann is not None else 0)
            two_years_annualized_returns.append(round(ret_2y_ann * 100, 2) if ret_2y_ann is not None else 0)
  
            std_devs.append(round(annualized_std_dev, 2))

        except Exception as e:
            st.warning(f"Could not fetch data for {stocks}: {e}. Skipping returns and standard deviation calculation.")
            three_months_returns.append(0)
            six_month_returns.append(0)
            one_year_returns.append(0)
            two_years_annualized_returns.append(0)
            three_year_annualized_returns.append(0)
            std_devs.append(0)
        
        my_bar.progress((i + 1) / len(portfolio_df), text=progress_text)
    
    portfolio_df['3M Return (%)'] = three_months_returns
    portfolio_df['6M Return (%)'] = six_month_returns
    portfolio_df['1Y Return (%)'] = one_year_returns
    portfolio_df['2Y Annualized Return (%)'] = two_years_annualized_returns
    portfolio_df['3Y Annualized Return (%)'] = three_year_annualized_returns
    portfolio_df['Annualized Standard Deviation (%)'] = std_devs
    portfolio_df['Last Price'] = portfolio_df['Stock Code'].apply(lambda code: yf.Ticker(code).history(period="1d")['Close'].iloc[-1])
    portfolio_df['Weight_2'] = portfolio_df['Quantity'] * portfolio_df['Last Price']
    portfolio_df['Weightage'] = ((portfolio_df['Weight_2'] / portfolio_df['Weight_2'].sum()) * 100).round(2).astype(str)

    # Add Nifty100  and smallcap 150 for comaprison
    nifty100 = yf.Ticker("^CNX100")
    nitymidcap = yf.Ticker('NIFTYMIDCAP150.NS')
    niftymidcap_data = nitymidcap.history(period="max").iloc[:, 0:4]
    nifty100_data = nifty100.history(period="max").iloc[:, 0:4]
    api_url = "https://api.mfapi.in/mf/147623"
    smallcap_df = fetch_mf_data_to_dataframe(api_url).reset_index(drop=True)
    smallcap_df['date'] = pd.to_datetime(smallcap_df['date'], format='%d-%m-%Y')
    smallcap_df=smallcap_df.sort_values(by='date', ascending=True)
    smallcap_df['nav'] = smallcap_df['nav'].astype(float)
    nifty100_1y_return = ((nifty100_data['Close'].iloc[-1] / nifty100_data['Close'].iloc[-252] - 1) * 100).round(2)
    niftymidcap_1y_return = ((niftymidcap_data['Close'].iloc[-1] / niftymidcap_data['Close'].iloc[-252] - 1) * 100).round(2)
    nifty100_6m_return = ((nifty100_data['Close'].iloc[-1] / nifty100_data['Close'].iloc[-126] - 1) * 100).round(2)
    niftymidcap_6m_return = ((niftymidcap_data['Close'].iloc[-1] / niftymidcap_data['Close'].iloc[-126] - 1) * 100).round(2)
    nifty100_3m_returns = ((nifty100_data['Close'].iloc[-1] / nifty100_data['Close'].iloc[-63] - 1) * 100).round(2)
    niftymidcap_3m_return = ((niftymidcap_data['Close'].iloc[-1] / niftymidcap_data['Close'].iloc[-63] - 1) * 100).round(2)
    nifty100_2y_return = ((nifty100_data['Close'].iloc[-1] / nifty100_data['Close'].iloc[-504] - 1) * 100).round(2)
    niftymidcap_2y_return = ((niftymidcap_data['Close'].iloc[-1] / niftymidcap_data['Close'].iloc[-504] - 1) * 100).round(2)
    nifty100_2y_annualized_return = (((1 + nifty100_2y_return / 100) ** (1 / 2) - 1) * 100).round(2)
    niftymidcap_2y_annualized_return = (((1 + niftymidcap_2y_return / 100) ** (1 / 2) - 1) * 100).round(2)
    nifty100_3y_return = ((nifty100_data['Close'].iloc[-1] / nifty100_data['Close'].iloc[0] - 1) * 100).round(2)
    niftymidcap_3y_return = ((niftymidcap_data['Close'].iloc[-1] / niftymidcap_data['Close'].iloc[0] - 1) * 100).round(2)
    nifty100_3y_annualized_return = (((1 + nifty100_3y_return / 100) ** (1 / 3) - 1) * 100).round(2)
    niftymidcap_3y_annualized_return = (((1 + niftymidcap_3y_return / 100) ** (1 / 3) - 1) * 100).round(2)
    niftysmallcap_1y_return = ((smallcap_df['nav'].iloc[-1] / smallcap_df['nav'].iloc[-252] - 1) * 100).round(2)
    niftysmallcap_6m_return = ((smallcap_df['nav'].iloc[-1] / smallcap_df['nav'].iloc[-126] - 1) * 100).round(2)
    niftysmallcap_3m_return = ((smallcap_df['nav'].iloc[-1] / smallcap_df['nav'].iloc[-63] - 1) * 100).round(2)
    niftysmallcap_2y_return = ((smallcap_df['nav'].iloc[-1] / smallcap_df['nav'].iloc[-504] - 1) * 100).round(2)
    niftysmallcap_2y_annualized_return = (((1 + niftysmallcap_2y_return / 100) ** (1 / 2) - 1) * 100).round(2)
    niftysmallcap_3y_return = ((smallcap_df['nav'].iloc[-1] / smallcap_df['nav'].iloc[0] - 1) * 100).round(2)
    niftysmallcap_3y_annualized_return = (((1 + niftysmallcap_3y_return / 100) ** (1 / 3) - 1) * 100).round(2)

   
    portfolio_df['Nifty100_1y_Return (%)'] = nifty100_1y_return.astype(str)
    portfolio_df['Nifty Midcap150_1y_Return (%)'] = niftymidcap_1y_return.astype(str)
    portfolio_df['Nifty100_6m_Return(%)'] = nifty100_6m_return.astype(str)
    portfolio_df['Nifty_midcap150_6m_Return(%)'] = niftymidcap_6m_return.astype(str)
    portfolio_df['Nifty100_3m_Return(%)'] = nifty100_3m_returns.astype(str)
    portfolio_df['Nifty_midcap150_3m_Return(%)'] = niftymidcap_3m_return.astype(str)
    portfolio_df['Nifty100_2y_Return(%)'] = nifty100_2y_annualized_return.astype(str)
    portfolio_df['Nifty_midcap150_2y_Return(%)'] = niftymidcap_2y_annualized_return.astype(str)
    portfolio_df['Nifty100_3y_Return(%)'] = nifty100_3y_annualized_return.astype(str)
    portfolio_df['Nifty_midcap150_3y_Return(%)'] = niftymidcap_3y_annualized_return.astype(str)
    portfolio_df['Nifty_Smallcap_1y_Return(%)'] = niftysmallcap_1y_return.astype(str)
    portfolio_df['Nifty_Smallcap_6m_Return(%)'] = niftysmallcap_6m_return.astype(str)
    portfolio_df['Nifty_Smallcap_3m_Return(%)'] = niftysmallcap_3m_return.astype(str)
    portfolio_df['Nifty_Smallcap_2y_Return(%)'] = niftysmallcap_2y_annualized_return.astype(str)
    portfolio_df['Nifty_Smallcap_3y_Return(%)'] = niftysmallcap_3y_annualized_return.astype(str)
    
    #portfolio_df['Weight'] = ((portfolio_df['Quantity']  / portfolio_df['Quantity'].sum()).round(2) * 100).astype(str)
    portfolio_df['Portfolio_SD'] = (portfolio_df['Annualized Standard Deviation (%)'] * portfolio_df['Weightage'].astype(float) / 100).sum().round(2)
    portfolio_df['Portfolio_SD'] = portfolio_df['Portfolio_SD'].astype(str)

   
    # Define the scoring function
    def compute_score(row):
        score = 0
        weight_score = 0
        # 1. Balanced Risk
        beta = row['Beta 3Year']
        if beta < 1:
            weight_score += 1
            
        score += 100 / beta if beta != 0 else 0


   
        # 2. Low Debt: approximate Total Debt to Total Capital = D/E / (1 + D/E)
        try:
            de = row['Total Debt to Total Equity Annual']
            debt_to_cap = de / (1 + de) if (1 + de) != 0 else 1
            if debt_to_cap < 0.4:
                weight_score += 1    

            score +=  0.4 / debt_to_cap * 100 if debt_to_cap != 0 else 0    
        except:
            pass # Handle cases where D/E might be missing or non-numeric

        # 3. Profitability Strength
        profit_score = 0
        if row.get('ROE Annual %', 0) > 0 and row.get('RoA Annual %', 0) > 0 and row.get('ROCE Annual %', 0) > 0:
            if row.get('ROE Annual %', 0) > 12:
                weight_score += 1
                profit_score += row.get('ROE Annual %', 0) / 12  * 100
            if row.get('RoA Annual %', 0) > 6:
                weight_score += 1
                profit_score += row.get('RoA Annual %', 0) / 6 * 100
            if row.get('ROCE Annual %', 0) > 12:
                weight_score += 1
                profit_score += row.get('ROCE Annual %', 0) / 12 * 100
        score += profit_score   

        # 4. Undervaluation & Growth
        undervaluation_growth_score  = 0
        # Use a large number instead of float('inf') for comparison to avoid potential issues
        company_peg = row.get('PEG TTM PE to Growth', 1e10)
        sector_peg = row.get('Sector PEG TTM', 1e10)
        if (company_peg < sector_peg) and (company_peg > 0) and (sector_peg > 0):
               weight_score += 1
               undervaluation_growth_score += (sector_peg / company_peg) * 100

        company_pe = row.get('PE TTM Price to Earnings', 1e10)
        sector_pe = row.get('Sector PE TTM', 1e10)       
        if (company_pe < sector_pe) and (company_pe > 0) and (sector_pe > 0):
            weight_score += 1
            undervaluation_growth_score += (sector_pe / company_pe) * 100
        score += undervaluation_growth_score

        # 5. Net Profit 3Y CAGR > 40%
        if row.get('Net Profit 3Yr Growth %', 0) > 40 :
            weight_score += 1
        score += row.get('Net Profit 3Yr Growth %', 0) / 40 * 100

        # 6. Revenue & Profit Growth (Quarterly & Annual YoY > Sector)
        revenue_profit_score=0
        if row.get('Net Profit Margin TTM %', -1e10) > 0:
                weight_score += 1
                revenue_profit_score += row.get('Net Profit Margin TTM %', -1e10) / 10 * 100
        score += revenue_profit_score
        # Standard Deviation
        std_dev = row['Annualized Standard Deviation (%)']
        if std_dev > 0 and std_dev < 20:
            weight_score += 1
        score += 20 / std_dev * 100
        weight_score = round(weight_score / 10, 2)
        return score,weight_score

    # Apply the function to the dataframe
    portfolio_df[['Score', 'Weight_Score']] = portfolio_df.apply(lambda row: pd.Series(compute_score(row)), axis=1)
    portfolio_df['Revised Score'] = portfolio_df['Score'] * portfolio_df['Weight_Score'].astype(float).round(2)

    st.subheader("Analyzed Portfolio Data:")
    st.dataframe(portfolio_df)

    # Generate and allow download of the PDF report
    if st.button("Generate Investment Report"):
        with st.spinner("Generating PDF report... This may take a moment."):
            pdf_bytes = create_enhanced_investment_report(equity_df=portfolio_df)
            if pdf_bytes:
                st.download_button(
                    label="Download Investment Report (PDF)",
                    data=pdf_bytes,
                    file_name="Enhanced_Investment_Report.pdf",
                    mime="application/pdf"
                )
                st.success("Report generated! Click the button above to download.")
            else:

                st.error("Failed to generate PDF report. Check logs for details.") 




















