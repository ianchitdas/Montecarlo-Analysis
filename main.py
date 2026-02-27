"""
Monte Carlo Simulation using Geometric Brownian Motion (GBM)
for Indian Stocks (NSE/BSE)

GBM Formula:
S(t+dt) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)

Where:
- S(t) = Stock price at time t
- μ = Expected return (drift)
- σ = Volatility
- dt = Time step
- Z = Standard normal random variable
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try importing yfinance for real data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not installed. Install via: pip install yfinance")
    print("Using manual parameters instead.\n")


class MonteCarloGBM:
    """
    Monte Carlo Simulation using Geometric Brownian Motion
    for Indian Stock Price Prediction.
    """

    def __init__(self, ticker=None, stock_name=None):
        self.ticker = ticker
        self.stock_name = stock_name or ticker
        self.historical_data = None
        self.S0 = None          # Current stock price
        self.mu = None          # Expected annual return (drift)
        self.sigma = None       # Annual volatility
        self.simulated_paths = None
        self.final_prices = None

    # ------------------------------------------------------------------ #
    #                       DATA FETCHING                                 #
    # ------------------------------------------------------------------ #
    def fetch_data(self, period="2y"):
        """Fetch historical data from Yahoo Finance."""
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is required. pip install yfinance")

        print(f"📥 Fetching data for {self.ticker}...")
        data = yf.download(self.ticker, period=period, auto_adjust=True)

        if data.empty:
            raise ValueError(f"No data found for ticker: {self.ticker}")

        self.historical_data = data
        self.S0 = float(data['Close'].iloc[-1])

        # Daily log returns
        log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()

        # Annualise (252 trading days)
        self.mu = float(log_returns.mean()) * 252
        self.sigma = float(log_returns.std()) * np.sqrt(252)

        print(f"✅ Data fetched successfully!")
        print(f"   📊 Period           : {data.index[0].date()} → {data.index[-1].date()}")
        print(f"   💰 Current Price     : ₹{self.S0:,.2f}")
        print(f"   📈 Annual Return (μ) : {self.mu*100:.2f}%")
        print(f"   📉 Annual Vol    (σ) : {self.sigma*100:.2f}%")
        print(f"   📋 Data Points       : {len(data)}")
        return self

    def set_manual_params(self, S0, mu, sigma, stock_name="Custom Stock"):
        """Set parameters manually if yfinance is unavailable."""
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.stock_name = stock_name
        print(f"✅ Manual parameters set for {stock_name}")
        print(f"   💰 Current Price     : ₹{S0:,.2f}")
        print(f"   📈 Annual Return (μ) : {mu*100:.2f}%")
        print(f"   📉 Annual Vol    (σ) : {sigma*100:.2f}%")
        return self

    # ------------------------------------------------------------------ #
    #                        SIMULATION                                   #
    # ------------------------------------------------------------------ #
    def simulate(self, T=1.0, num_simulations=10000, num_steps=252, seed=42):
        """
        Run Monte Carlo simulation.

        Parameters
        ----------
        T               : float – time horizon in years
        num_simulations : int   – number of paths
        num_steps       : int   – number of time steps (trading days)
        seed            : int   – random seed for reproducibility
        """
        if self.S0 is None:
            raise ValueError("Set parameters first via fetch_data() or set_manual_params().")

        np.random.seed(seed)
        dt = T / num_steps

        print(f"\n🚀 Running Monte Carlo Simulation...")
        print(f"   ⏳ Time Horizon      : {T} year(s) ({num_steps} trading days)")
        print(f"   🔄 Simulations       : {num_simulations:,}")

        # Pre-allocate price matrix: (num_steps + 1) x num_simulations
        prices = np.zeros((num_steps + 1, num_simulations))
        prices[0] = self.S0

        # Vectorised GBM path generation
        for t in range(1, num_steps + 1):
            Z = np.random.standard_normal(num_simulations)
            prices[t] = prices[t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * dt
                + self.sigma * np.sqrt(dt) * Z
            )

        self.simulated_paths = prices
        self.final_prices = prices[-1]
        self.T = T
        self.num_simulations = num_simulations
        self.num_steps = num_steps

        print(f"   ✅ Simulation Complete!\n")
        return self

    # ------------------------------------------------------------------ #
    #                        ANALYSIS                                     #
    # ------------------------------------------------------------------ #
    def analyse(self, confidence_level=0.95):
        """Print comprehensive statistical analysis."""
        if self.final_prices is None:
            raise ValueError("Run simulate() first.")

        fp = self.final_prices
        alpha = 1 - confidence_level

        mean_price = np.mean(fp)
        median_price = np.median(fp)
        std_price = np.std(fp)
        min_price = np.min(fp)
        max_price = np.max(fp)

        ci_lower = np.percentile(fp, alpha / 2 * 100)
        ci_upper = np.percentile(fp, (1 - alpha / 2) * 100)

        var_5 = np.percentile(fp, 5)         # 5th percentile (Value at Risk)
        percentile_25 = np.percentile(fp, 25)
        percentile_75 = np.percentile(fp, 75)
        percentile_95 = np.percentile(fp, 95)

        expected_return = (mean_price - self.S0) / self.S0 * 100
        prob_profit = np.mean(fp > self.S0) * 100
        prob_loss = np.mean(fp < self.S0) * 100
        prob_10up = np.mean(fp > self.S0 * 1.10) * 100
        prob_20up = np.mean(fp > self.S0 * 1.20) * 100
        prob_10down = np.mean(fp < self.S0 * 0.90) * 100
        prob_20down = np.mean(fp < self.S0 * 0.80) * 100
        prob_50up = np.mean(fp > self.S0 * 1.50) * 100

        print("=" * 65)
        print(f"{'MONTE CARLO SIMULATION RESULTS':^65}")
        print(f"{'(' + self.stock_name + ')':^65}")
        print("=" * 65)

        print(f"\n📌 CURRENT PRICE: ₹{self.S0:,.2f}")
        print(f"📅 FORECAST HORIZON: {self.T} year(s) | "
              f"SIMULATIONS: {self.num_simulations:,}\n")

        print("─" * 65)
        print("  PRICE STATISTICS")
        print("─" * 65)
        print(f"  {'Mean Price':<30}: ₹{mean_price:>12,.2f}")
        print(f"  {'Median Price':<30}: ₹{median_price:>12,.2f}")
        print(f"  {'Std Deviation':<30}: ₹{std_price:>12,.2f}")
        print(f"  {'Minimum Price':<30}: ₹{min_price:>12,.2f}")
        print(f"  {'Maximum Price':<30}: ₹{max_price:>12,.2f}")

        print(f"\n  PERCENTILES")
        print("─" * 65)
        print(f"  {'5th  Percentile (VaR 95%)':<30}: ₹{var_5:>12,.2f}")
        print(f"  {'25th Percentile':<30}: ₹{percentile_25:>12,.2f}")
        print(f"  {'50th Percentile (Median)':<30}: ₹{median_price:>12,.2f}")
        print(f"  {'75th Percentile':<30}: ₹{percentile_75:>12,.2f}")
        print(f"  {'95th Percentile':<30}: ₹{percentile_95:>12,.2f}")

        print(f"\n  CONFIDENCE INTERVAL ({confidence_level*100:.0f}%)")
        print("─" * 65)
        print(f"  Lower Bound : ₹{ci_lower:>12,.2f}")
        print(f"  Upper Bound : ₹{ci_upper:>12,.2f}")

        print(f"\n  RETURN & PROBABILITY ANALYSIS")
        print("─" * 65)
        print(f"  {'Expected Return':<30}: {expected_return:>10.2f}%")
        print(f"  {'Probability of Profit':<30}: {prob_profit:>10.2f}%")
        print(f"  {'Probability of Loss':<30}: {prob_loss:>10.2f}%")
        print(f"  {'Prob. of >10% Gain':<30}: {prob_10up:>10.2f}%")
        print(f"  {'Prob. of >20% Gain':<30}: {prob_20up:>10.2f}%")
        print(f"  {'Prob. of >50% Gain':<30}: {prob_50up:>10.2f}%")
        print(f"  {'Prob. of >10% Loss':<30}: {prob_10down:>10.2f}%")
        print(f"  {'Prob. of >20% Loss':<30}: {prob_20down:>10.2f}%")

        print("=" * 65)

        # Store for plotting
        self.stats = {
            'mean': mean_price, 'median': median_price,
            'std': std_price, 'ci_lower': ci_lower,
            'ci_upper': ci_upper, 'var_5': var_5,
            'percentile_25': percentile_25, 'percentile_75': percentile_75,
            'percentile_95': percentile_95,
            'expected_return': expected_return, 'prob_profit': prob_profit,
        }
        return self

    # ------------------------------------------------------------------ #
    #                        PLOTTING                                     #
    # ------------------------------------------------------------------ #
    def plot_all(self, num_paths_to_show=200):
        """Generate all plots in a single figure."""
        if self.simulated_paths is None:
            raise ValueError("Run simulate() first.")

        fig, axes = plt.subplots(2, 2, figsize=(18, 13))
        fig.suptitle(
            f'Monte Carlo Simulation (GBM) — {self.stock_name}\n'
            f'{self.num_simulations:,} simulations | {self.T} year horizon',
            fontsize=16, fontweight='bold', y=1.02
        )

        # ---- 1. Simulated Price Paths ---- #
        ax1 = axes[0, 0]
        time_axis = np.linspace(0, self.T, self.num_steps + 1)
        paths_idx = np.random.choice(
            self.num_simulations,
            min(num_paths_to_show, self.num_simulations),
            replace=False
        )
        for i in paths_idx:
            ax1.plot(time_axis, self.simulated_paths[:, i],
                     alpha=0.15, linewidth=0.5, color='steelblue')

        # Mean path
        mean_path = np.mean(self.simulated_paths, axis=1)
        ax1.plot(time_axis, mean_path, color='red',
                 linewidth=2.5, label='Mean Path')

        # Percentile bands
        p5 = np.percentile(self.simulated_paths, 5, axis=1)
        p95 = np.percentile(self.simulated_paths, 95, axis=1)
        ax1.fill_between(time_axis, p5, p95,
                         alpha=0.15, color='orange', label='5th–95th Percentile')

        ax1.axhline(y=self.S0, color='green', linestyle='--',
                     linewidth=1.5, label=f'Current: ₹{self.S0:,.0f}')
        ax1.set_title('Simulated Price Paths', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Time (Years)')
        ax1.set_ylabel('Stock Price (₹)')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ---- 2. Distribution of Final Prices ---- #
        ax2 = axes[0, 1]
        ax2.hist(self.final_prices, bins=100, density=True,
                 alpha=0.7, color='steelblue', edgecolor='white', linewidth=0.3)

        ax2.axvline(self.stats['mean'], color='red', linewidth=2,
                     linestyle='-', label=f"Mean: ₹{self.stats['mean']:,.0f}")
        ax2.axvline(self.stats['median'], color='orange', linewidth=2,
                     linestyle='--', label=f"Median: ₹{self.stats['median']:,.0f}")
        ax2.axvline(self.S0, color='green', linewidth=2,
                     linestyle=':', label=f"Current: ₹{self.S0:,.0f}")
        ax2.axvline(self.stats['ci_lower'], color='purple', linewidth=1.5,
                     linestyle='-.', label=f"95% CI Low: ₹{self.stats['ci_lower']:,.0f}")
        ax2.axvline(self.stats['ci_upper'], color='purple', linewidth=1.5,
                     linestyle='-.')

        ax2.set_title('Distribution of Final Prices', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Stock Price (₹)')
        ax2.set_ylabel('Density')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ---- 3. Distribution of Returns ---- #
        ax3 = axes[1, 0]
        returns = (self.final_prices - self.S0) / self.S0 * 100
        ax3.hist(returns, bins=100, density=True,
                 alpha=0.7, color='coral', edgecolor='white', linewidth=0.3)
        ax3.axvline(0, color='black', linewidth=2, linestyle='-', label='Break-even')
        ax3.axvline(np.mean(returns), color='red', linewidth=2,
                     linestyle='--', label=f"Mean Return: {np.mean(returns):.1f}%")

        ax3.set_title('Distribution of Returns (%)', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Density')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # ---- 4. Cone of Uncertainty ---- #
        ax4 = axes[1, 1]
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        colors_cone = ['#d73027', '#fc8d59', '#fee08b', '#1a9850',
                        '#fee08b', '#fc8d59', '#d73027']

        perc_values = {}
        for p in percentiles:
            perc_values[p] = np.percentile(self.simulated_paths, p, axis=1)
            ax4.plot(time_axis, perc_values[p],
                     linewidth=1.5, label=f'{p}th Percentile')

        ax4.fill_between(time_axis, perc_values[5], perc_values[95],
                         alpha=0.1, color='blue')
        ax4.fill_between(time_axis, perc_values[25], perc_values[75],
                         alpha=0.2, color='green')

        ax4.axhline(y=self.S0, color='black', linestyle='--',
                     linewidth=1, alpha=0.5)
        ax4.set_title('Cone of Uncertainty', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Time (Years)')
        ax4.set_ylabel('Stock Price (₹)')
        ax4.legend(fontsize=8, loc='upper left')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('monte_carlo_simulation.png', dpi=150, bbox_inches='tight')
        print("\n📊 Plot saved as 'monte_carlo_simulation.png'")
        plt.show()
        return self

    # ------------------------------------------------------------------ #
    #                     HISTORICAL VS SIMULATED                         #
    # ------------------------------------------------------------------ #
    def plot_historical_and_forecast(self, num_paths=100):
        """Plot historical prices followed by simulated forecast."""
        if self.historical_data is None:
            print("⚠️  No historical data available (manual mode). Skipping.")
            return self

        fig, ax = plt.subplots(figsize=(16, 7))

        # Historical
        hist = self.historical_data['Close']
        ax.plot(hist.index, hist.values, color='black',
                linewidth=1.5, label='Historical Price')

        # Forecast dates
        last_date = hist.index[-1]
        forecast_dates = pd.bdate_range(
            start=last_date + timedelta(days=1),
            periods=self.num_steps
        )
        forecast_dates = np.insert(forecast_dates, 0, last_date)

        paths_idx = np.random.choice(
            self.num_simulations, min(num_paths, self.num_simulations), replace=False
        )
        for i in paths_idx:
            ax.plot(forecast_dates, self.simulated_paths[:, i],
                    alpha=0.12, linewidth=0.5, color='steelblue')

        mean_path = np.mean(self.simulated_paths, axis=1)
        ax.plot(forecast_dates, mean_path, color='red',
                linewidth=2.5, label='Mean Forecast')

        p5 = np.percentile(self.simulated_paths, 5, axis=1)
        p95 = np.percentile(self.simulated_paths, 95, axis=1)
        ax.fill_between(forecast_dates, p5, p95,
                        alpha=0.15, color='orange', label='90% Confidence Band')

        ax.axvline(x=last_date, color='gray', linestyle='--',
                   linewidth=1, alpha=0.7, label='Forecast Start')

        ax.set_title(
            f'{self.stock_name} — Historical + Monte Carlo Forecast',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (₹)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('historical_and_forecast.png', dpi=150, bbox_inches='tight')
        print("📊 Plot saved as 'historical_and_forecast.png'")
        plt.show()
        return self


# ===================================================================== #
#                            MAIN EXECUTION                              #
# ===================================================================== #
def main():
    print("=" * 65)
    print("  MONTE CARLO SIMULATION (GBM) FOR INDIAN STOCKS")
    print("=" * 65)

    # ─── Popular Indian Stock Tickers (NSE via Yahoo Finance) ──────── #
    INDIAN_STOCKS = {
        '1': ('RELIANCE.NS',   'Reliance Industries'),
        '2': ('TCS.NS',        'Tata Consultancy Services'),
        '3': ('INFY.NS',       'Infosys'),
        '4': ('HDFCBANK.NS',   'HDFC Bank'),
        '5': ('ICICIBANK.NS',  'ICICI Bank'),
        '6': ('HINDUNILVR.NS', 'Hindustan Unilever'),
        '7': ('SBIN.NS',       'State Bank of India'),
        '8': ('BHARTIARTL.NS', 'Bharti Airtel'),
        '9': ('ITC.NS',        'ITC Limited'),
        '10': ('WIPRO.NS',     'Wipro'),
        '11': ('TATAMOTORS.NS','Tata Motors'),
        '12': ('ADANIENT.NS',  'Adani Enterprises'),
        '13': ('BAJFINANCE.NS','Bajaj Finance'),
        '14': ('LT.NS',        'Larsen & Toubro'),
        '15': ('MARUTI.NS',    'Maruti Suzuki'),
    }

    print("\n📋 Select a stock or enter a custom NSE ticker:\n")
    for key, (ticker, name) in INDIAN_STOCKS.items():
        print(f"   {key:>3}. {name:<30} ({ticker})")
    print(f"   {'C':>3}. Custom ticker")
    print(f"   {'M':>3}. Manual parameters (no internet needed)")

    choice = input("\n👉 Enter your choice: ").strip()

    mc = MonteCarloGBM()

    if choice.upper() == 'M':
        # Manual mode
        print("\n📝 Enter parameters manually:")
        name = input("   Stock Name: ").strip() or "Custom Stock"
        S0 = float(input("   Current Price (₹): "))
        mu = float(input("   Annual Expected Return (e.g. 0.15 for 15%): "))
        sigma = float(input("   Annual Volatility (e.g. 0.30 for 30%): "))
        mc.set_manual_params(S0, mu, sigma, name)

    elif choice.upper() == 'C':
        custom = input("   Enter NSE ticker (e.g. TATASTEEL.NS): ").strip()
        mc.ticker = custom
        mc.stock_name = custom.replace('.NS', '').replace('.BO', '')
        mc.fetch_data(period="2y")

    elif choice in INDIAN_STOCKS:
        ticker, name = INDIAN_STOCKS[choice]
        mc.ticker = ticker
        mc.stock_name = name
        mc.fetch_data(period="2y")

    else:
        print("Invalid choice. Using Reliance Industries as default.")
        mc.ticker = 'RELIANCE.NS'
        mc.stock_name = 'Reliance Industries'
        mc.fetch_data(period="2y")

    # Simulation parameters
    print("\n⚙️  Simulation Parameters (press Enter for defaults):")
    T_input = input("   Time Horizon in years [1.0]: ").strip()
    T = float(T_input) if T_input else 1.0

    n_input = input("   Number of simulations [10000]: ").strip()
    n_sims = int(n_input) if n_input else 10000

    steps_input = input("   Number of time steps [252]: ").strip()
    n_steps = int(steps_input) if steps_input else 252

    # Run simulation
    mc.simulate(T=T, num_simulations=n_sims, num_steps=n_steps)

    # Analysis
    mc.analyse(confidence_level=0.95)

    # Plots
    mc.plot_all(num_paths_to_show=300)
    mc.plot_historical_and_forecast(num_paths=150)

    print("\n🎯 Simulation Complete! Check the saved plots.")
    print("⚠️  Disclaimer: This is for educational purposes only.")
    print("   GBM assumes log-normal returns and constant volatility.")
    print("   Real markets exhibit fat tails, jumps, and regime changes.")
    print("   Do NOT use this as sole basis for investment decisions.\n")


if __name__ == "__main__":
    main()