import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Personal Finance Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("💰 Personal Finance Analytics Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    data_source = st.radio(
        "Choose Data Source:",
        ["Generate Sample Data", "Upload CSV File"]
    )
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload your transactions CSV",
            type=['csv'],
            help="CSV should have columns: date, category, amount, description"
        )
    
    st.markdown("---")
    st.subheader("📊 Filters")

# Function to generate sample data
@st.cache_data
def generate_sample_data(n_transactions=500):
    np.random.seed(42)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_transactions)
    
    categories = {
        'Groceries': (50, 200),
        'Rent': (800, 1200),
        'Utilities': (50, 150),
        'Transportation': (20, 100),
        'Entertainment': (30, 150),
        'Restaurants': (15, 80),
        'Healthcare': (30, 200),
        'Shopping': (40, 300),
        'Subscriptions': (10, 50),
        'Income': (2000, 5000)
    }
    
    transactions = []
    for date in dates:
        category = np.random.choice(list(categories.keys()))
        min_amt, max_amt = categories[category]
        
        if category == 'Income':
            amount = np.random.uniform(min_amt, max_amt)
        else:
            amount = -np.random.uniform(min_amt, max_amt)
        
        transactions.append({
            'date': date,
            'category': category,
            'amount': round(amount, 2),
            'description': f"{category} transaction"
        })
    
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Load data
if data_source == "Generate Sample Data":
    df = generate_sample_data()
    st.sidebar.success(f"✅ Loaded {len(df)} sample transactions")
else:
    if 'uploaded_file' in locals() and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['date'] = pd.to_datetime(df['date'])
        st.sidebar.success(f"✅ Loaded {len(df)} transactions")
    else:
        st.info("👈 Please upload a CSV file from the sidebar")
        st.stop()

# Date range filter
df['date'] = pd.to_datetime(df['date'])
min_date = df['date'].min().date()
max_date = df['date'].max().date()

with st.sidebar:
    date_range = st.date_input(
        "Select Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    
    # Category filter
    all_categories = df['category'].unique().tolist()
    selected_categories = st.multiselect(
        "Select Categories:",
        options=all_categories,
        default=all_categories
    )
    
    df = df[df['category'].isin(selected_categories)]

# Calculate metrics
total_income = df[df['amount'] > 0]['amount'].sum()
total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
net_savings = total_income - total_expenses
avg_daily_expense = total_expenses / max((df['date'].max() - df['date'].min()).days, 1)

# Display key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="💵 Total Income",
        value=f"${total_income:,.2f}",
        delta=f"{len(df[df['amount'] > 0])} transactions"
    )

with col2:
    st.metric(
        label="💸 Total Expenses",
        value=f"${total_expenses:,.2f}",
        delta=f"{len(df[df['amount'] < 0])} transactions",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="💰 Net Savings",
        value=f"${net_savings:,.2f}",
        delta="Positive" if net_savings > 0 else "Negative",
        delta_color="normal" if net_savings > 0 else "inverse"
    )

with col4:
    st.metric(
        label="📊 Avg Daily Expense",
        value=f"${avg_daily_expense:,.2f}",
        delta=f"{len(df)} total transactions"
    )

st.markdown("---")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "📈 Trends", 
    "🎯 Categories", 
    "📋 Transactions",
    "⚠️ Insights"
])

with tab1:
    st.subheader("Financial Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Spending by category - Pie chart
        expenses = df[df['amount'] < 0].copy()
        expenses['amount'] = abs(expenses['amount'])
        category_spending = expenses.groupby('category')['amount'].sum().reset_index()
        category_spending = category_spending.sort_values('amount', ascending=False)
        
        fig_pie = px.pie(
            category_spending,
            values='amount',
            names='category',
            title='Spending Distribution by Category',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Monthly income vs expenses
        df['month'] = df['date'].dt.to_period('M').astype(str)
        monthly_summary = df.groupby('month').apply(
            lambda x: pd.Series({
                'Income': x[x['amount'] > 0]['amount'].sum(),
                'Expenses': abs(x[x['amount'] < 0]['amount'].sum())
            })
        ).reset_index()
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=monthly_summary['month'],
            y=monthly_summary['Income'],
            name='Income',
            marker_color='green'
        ))
        fig_bar.add_trace(go.Bar(
            x=monthly_summary['month'],
            y=monthly_summary['Expenses'],
            name='Expenses',
            marker_color='red'
        ))
        fig_bar.update_layout(
            title='Monthly Income vs Expenses',
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            barmode='group',
            hovermode='x unified'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("Spending Trends Over Time")
    
    # Daily spending trend
    expenses_daily = df[df['amount'] < 0].copy()
    expenses_daily['amount'] = abs(expenses_daily['amount'])
    daily_spending = expenses_daily.groupby('date')['amount'].sum().reset_index()
    
    fig_trend = px.line(
        daily_spending,
        x='date',
        y='amount',
        title='Daily Spending Trend',
        labels={'amount': 'Amount ($)', 'date': 'Date'}
    )
    fig_trend.update_traces(line_color='coral', line_width=2)
    fig_trend.update_layout(hovermode='x unified')
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Cumulative savings
    df_sorted = df.sort_values('date')
    df_sorted['cumulative_savings'] = df_sorted['amount'].cumsum()
    
    fig_cumulative = px.line(
        df_sorted,
        x='date',
        y='cumulative_savings',
        title='Cumulative Savings Over Time',
        labels={'cumulative_savings': 'Cumulative Savings ($)', 'date': 'Date'}
    )
    fig_cumulative.update_traces(line_color='darkgreen', line_width=2)
    fig_cumulative.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    st.plotly_chart(fig_cumulative, use_container_width=True)

with tab3:
    st.subheader("Category Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top spending categories
        expenses = df[df['amount'] < 0].copy()
        expenses['amount'] = abs(expenses['amount'])
        category_totals = expenses.groupby('category')['amount'].sum().sort_values(ascending=True).tail(10)
        
        fig_cat = px.bar(
            x=category_totals.values,
            y=category_totals.index,
            orientation='h',
            title='Top 10 Spending Categories',
            labels={'x': 'Total Spent ($)', 'y': 'Category'},
            color=category_totals.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        # Transaction frequency
        transaction_counts = expenses.groupby('category').size().sort_values(ascending=True).tail(10)
        
        fig_freq = px.bar(
            x=transaction_counts.values,
            y=transaction_counts.index,
            orientation='h',
            title='Transaction Frequency by Category',
            labels={'x': 'Number of Transactions', 'y': 'Category'},
            color=transaction_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    
    # Detailed category table
    st.subheader("📊 Category Statistics")
    category_stats = expenses.groupby('category')['amount'].agg([
        ('Total', 'sum'),
        ('Average', 'mean'),
        ('Count', 'count'),
        ('Max', 'max'),
        ('Min', 'min')
    ]).round(2).sort_values('Total', ascending=False)
    
    st.dataframe(category_stats, use_container_width=True)

with tab4:
    st.subheader("Transaction History")
    
    # Search and filter
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("🔍 Search transactions:", "")
    with col2:
        sort_by = st.selectbox("Sort by:", ["Date (Newest)", "Date (Oldest)", "Amount (High)", "Amount (Low)"])
    
    # Apply search
    display_df = df.copy()
    if search_term:
        display_df = display_df[
            display_df['description'].str.contains(search_term, case=False) |
            display_df['category'].str.contains(search_term, case=False)
        ]
    
    # Apply sorting
    if sort_by == "Date (Newest)":
        display_df = display_df.sort_values('date', ascending=False)
    elif sort_by == "Date (Oldest)":
        display_df = display_df.sort_values('date', ascending=True)
    elif sort_by == "Amount (High)":
        display_df = display_df.sort_values('amount', ascending=False)
    else:
        display_df = display_df.sort_values('amount', ascending=True)
    
    # Display transactions
    st.dataframe(
        display_df[['date', 'category', 'amount', 'description']].style.format({
            'amount': '${:,.2f}',
            'date': lambda x: x.strftime('%Y-%m-%d')
        }),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with tab5:
    st.subheader("💡 Financial Insights & Anomalies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Key Insights")
        
        # Calculate insights
        expenses_only = df[df['amount'] < 0].copy()
        expenses_only['amount'] = abs(expenses_only['amount'])
        
        top_category = expenses_only.groupby('category')['amount'].sum().idxmax()
        top_category_amt = expenses_only.groupby('category')['amount'].sum().max()
        
        avg_transaction = expenses_only['amount'].mean()
        largest_expense = expenses_only.nlargest(1, 'amount').iloc[0]
        
        savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
        
        st.info(f"🎯 **Top Spending Category**: {top_category} (${top_category_amt:,.2f})")
        st.info(f"💳 **Average Transaction**: ${avg_transaction:.2f}")
        st.info(f"📈 **Savings Rate**: {savings_rate:.1f}%")
        st.info(f"🔴 **Largest Expense**: ${largest_expense['amount']:.2f} - {largest_expense['category']}")
    
    with col2:
        st.markdown("### ⚠️ Unusual Transactions")
        
        # Detect anomalies
        anomalies = []
        for category in expenses_only['category'].unique():
            cat_data = expenses_only[expenses_only['category'] == category]['amount']
            if len(cat_data) > 3:
                mean = cat_data.mean()
                std = cat_data.std()
                
                if std > 0:
                    z_scores = np.abs((cat_data - mean) / std)
                    anomaly_mask = z_scores > 2.5
                    
                    if anomaly_mask.any():
                        anomaly_df = expenses_only[expenses_only['category'] == category][anomaly_mask]
                        anomalies.append(anomaly_df)
        
        if anomalies:
            all_anomalies = pd.concat(anomalies).sort_values('amount', ascending=False).head(10)
            
            for idx, row in all_anomalies.iterrows():
                st.warning(f"⚠️ {row['date'].strftime('%Y-%m-%d')} | **{row['category']}**: ${row['amount']:.2f}")
        else:
            st.success("✅ No unusual spending detected!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit 🎈 | Personal Finance Analytics Dashboard"
    "</div>",
    unsafe_allow_html=True
)