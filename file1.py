import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FinanceAnalyzer:
    def __init__(self):
        self.df = None
        
    def generate_sample_data(self, n_transactions=500):
        """Generate realistic sample transaction data"""
        np.random.seed(42)
        
        # Date range: last 12 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, periods=n_transactions)
        
        # Categories with typical spending patterns
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
            
            # Income is positive, expenses are negative
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
        
        self.df = pd.DataFrame(transactions)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['month'] = self.df['date'].dt.to_period('M')
        self.df['year'] = self.df['date'].dt.year
        
        print(f"✓ Generated {len(self.df)} sample transactions")
        return self.df
    
    def load_data(self, filepath):
        """Load data from CSV file"""
        self.df = pd.read_csv(filepath)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['month'] = self.df['date'].dt.to_period('M')
        print(f"✓ Loaded {len(self.df)} transactions from {filepath}")
        return self.df
    
    def get_summary_stats(self):
        """Calculate summary statistics"""
        expenses = self.df[self.df['amount'] < 0]['amount'].sum()
        income = self.df[self.df['amount'] > 0]['amount'].sum()
        net = income + expenses
        
        stats = {
            'Total Income': f"${income:,.2f}",
            'Total Expenses': f"${abs(expenses):,.2f}",
            'Net Savings': f"${net:,.2f}",
            'Avg Monthly Expense': f"${abs(expenses)/12:,.2f}",
            'Transactions': len(self.df)
        }
        
        print("\n" + "="*50)
        print("FINANCIAL SUMMARY")
        print("="*50)
        for key, value in stats.items():
            print(f"{key:.<30} {value:>15}")
        print("="*50 + "\n")
        
        return stats
    
    def spending_by_category(self):
        """Analyze spending by category"""
        expenses = self.df[self.df['amount'] < 0].copy()
        expenses['amount'] = abs(expenses['amount'])
        
        category_spending = expenses.groupby('category')['amount'].agg(['sum', 'mean', 'count'])
        category_spending.columns = ['Total', 'Average', 'Count']
        category_spending = category_spending.sort_values('Total', ascending=False)
        
        print("SPENDING BY CATEGORY")
        print(category_spending.to_string())
        print()
        
        return category_spending
    
    def monthly_trend(self):
        """Analyze monthly spending trends"""
        monthly = self.df.groupby('month')['amount'].sum()
        
        print("MONTHLY NET FLOW")
        for month, amount in monthly.items():
            status = "📈" if amount > 0 else "📉"
            print(f"{month}: ${amount:>10,.2f} {status}")
        print()
        
        return monthly
    
    def visualize_all(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Spending by Category (Pie Chart)
        ax1 = plt.subplot(2, 3, 1)
        expenses = self.df[self.df['amount'] < 0].copy()
        expenses['amount'] = abs(expenses['amount'])
        category_totals = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        colors = sns.color_palette("Set3", len(category_totals))
        ax1.pie(category_totals, labels=category_totals.index, autopct='%1.1f%%',
                startangle=90, colors=colors)
        ax1.set_title('Spending Distribution by Category', fontsize=12, fontweight='bold')
        
        # 2. Monthly Income vs Expenses
        ax2 = plt.subplot(2, 3, 2)
        monthly_summary = self.df.groupby('month').apply(
            lambda x: pd.Series({
                'income': x[x['amount'] > 0]['amount'].sum(),
                'expenses': abs(x[x['amount'] < 0]['amount'].sum())
            })
        )
        
        x = range(len(monthly_summary))
        width = 0.35
        ax2.bar([i - width/2 for i in x], monthly_summary['income'], width, 
                label='Income', color='green', alpha=0.7)
        ax2.bar([i + width/2 for i in x], monthly_summary['expenses'], width,
                label='Expenses', color='red', alpha=0.7)
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Amount ($)')
        ax2.set_title('Monthly Income vs Expenses', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Spending Trend Over Time
        ax3 = plt.subplot(2, 3, 3)
        expenses_only = self.df[self.df['amount'] < 0].copy()
        expenses_only['amount'] = abs(expenses_only['amount'])
        daily_expenses = expenses_only.groupby('date')['amount'].sum()
        
        ax3.plot(daily_expenses.index, daily_expenses.values, color='coral', linewidth=2)
        ax3.fill_between(daily_expenses.index, daily_expenses.values, alpha=0.3, color='coral')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Daily Expenses ($)')
        ax3.set_title('Daily Spending Trend', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Category Comparison (Bar Chart)
        ax4 = plt.subplot(2, 3, 4)
        top_categories = category_totals.head(8)
        bars = ax4.barh(top_categories.index, top_categories.values, color=sns.color_palette("viridis", len(top_categories)))
        ax4.set_xlabel('Total Spent ($)')
        ax4.set_title('Top Spending Categories', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2, 
                    f'${width:,.0f}', ha='left', va='center', fontsize=9)
        
        # 5. Net Savings Over Time
        ax5 = plt.subplot(2, 3, 5)
        monthly_net = self.df.groupby('month')['amount'].sum()
        cumulative_savings = monthly_net.cumsum()
        
        ax5.plot(range(len(cumulative_savings)), cumulative_savings.values, 
                marker='o', linewidth=2, markersize=6, color='darkgreen')
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Cumulative Savings ($)')
        ax5.set_title('Cumulative Savings Trend', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Transaction Count by Category
        ax6 = plt.subplot(2, 3, 6)
        transaction_counts = expenses.groupby('category').size().sort_values(ascending=True)
        ax6.barh(transaction_counts.index, transaction_counts.values, 
                color=sns.color_palette("muted", len(transaction_counts)))
        ax6.set_xlabel('Number of Transactions')
        ax6.set_title('Transaction Frequency by Category', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('finance_analytics_dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Dashboard saved as 'finance_analytics_dashboard.png'")
        plt.show()
    
    def detect_anomalies(self, threshold=2.5):
        """Detect unusual spending patterns"""
        expenses = self.df[self.df['amount'] < 0].copy()
        expenses['amount'] = abs(expenses['amount'])
        
        # Calculate z-score for each category
        anomalies = []
        for category in expenses['category'].unique():
            cat_data = expenses[expenses['category'] == category]['amount']
            mean = cat_data.mean()
            std = cat_data.std()
            
            if std > 0:
                z_scores = np.abs((cat_data - mean) / std)
                anomaly_mask = z_scores > threshold
                
                if anomaly_mask.any():
                    anomaly_transactions = expenses[expenses['category'] == category][anomaly_mask]
                    anomalies.extend(anomaly_transactions.to_dict('records'))
        
        if anomalies:
            print(f"\n⚠️  Detected {len(anomalies)} unusual transactions:")
            for a in anomalies[:5]:  # Show first 5
                print(f"   {a['date'].strftime('%Y-%m-%d')} | {a['category']}: ${a['amount']:.2f}")
        else:
            print("\n✓ No unusual spending detected")
        
        return anomalies


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("PERSONAL FINANCE ANALYTICS SYSTEM")
    print("="*60)
    
    # Initialize analyzer
    analyzer = FinanceAnalyzer()
    
    # Generate sample data (or use: analyzer.load_data('your_file.csv'))
    df = analyzer.generate_sample_data(n_transactions=500)
    
    # Display first few transactions
    print("\nSample Transactions:")
    print(df.head(10).to_string(index=False))
    print()
    
    # Run analyses
    analyzer.get_summary_stats()
    analyzer.spending_by_category()
    analyzer.monthly_trend()
    analyzer.detect_anomalies()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.visualize_all()
    
    print("\n" + "="*60)
    print("Analysis complete! Check the generated dashboard image.")
    print("="*60)