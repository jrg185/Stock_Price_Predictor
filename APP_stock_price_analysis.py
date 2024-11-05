import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')

class StockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
    
    def fetch_data(self):
        """Fetch and prepare stock data"""
        print(f"\nFetching data for {self.symbol}...\n")
        df = yf.download(self.symbol, start='2020-01-01', progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(level=1, axis=1)
        
        if 'Close' not in df.columns or df['Close'].isnull().all():
            raise ValueError("Error: 'Close' column is missing or contains no data.")
        
        # Add technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ATR Calculation
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()

        # Volume ratio
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=50).mean()

        return df

    def predict_future_prices_with_linear_regression(self, df, days=90):
        """Predict future prices using linear regression with robust handling for volatile stocks"""
        df = df.dropna(subset=['Close'])
        
        # Calculate rolling stats to determine trend
        df['returns'] = df['Close'].pct_change()
        df['rolling_std'] = df['returns'].rolling(window=20).std()
        df['rolling_mean'] = df['Close'].rolling(window=20).mean()
        
        # Use recent data for volatile stocks (last 3 months)
        df = df.tail(63)  # ~3 months of trading days
        
        df['Day'] = np.arange(len(df))
        X = df[['Day']]
        y = df['Close']  # Use raw prices for less volatile prediction
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Model statistics
        r_squared = model.score(X, y)
        coefficients = model.coef_[0]
        intercept = model.intercept_
        
        # Calculate metrics
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        
        # Predict future values
        future_days = np.array([[len(df) + i] for i in range(1, days + 1)])
        future_prices = model.predict(future_days)
        
        # Add mean reversion component for extreme predictions
        current_price = df['Close'].iloc[-1]
        vol = df['returns'].std() * np.sqrt(252)  # Annualized volatility
        
        # Apply stronger dampening for high volatility stocks
        dampening = np.exp(-vol * np.arange(len(future_prices)) / 252)
        future_prices = current_price + (future_prices - current_price) * dampening
        
        model_summary = {
            'r_squared': r_squared,
            'coefficient': coefficients,
            'intercept': intercept,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

        return future_prices, model_summary

    def predict_future_prices_with_arima(self, df, days=90):
        """Predict future prices using ARIMA with mean reversion component"""
        try:
            returns = df['Returns'].dropna()
            
            # Calculate historical volatility
            vol = returns.std() * np.sqrt(252)
            
            # Adjust ARIMA order based on RSI
            rsi = df['RSI'].iloc[-1]
            if rsi < 30 or rsi > 70:
                # Increase mean reversion component when oversold/overbought
                order = (3, 0, 2)
            else:
                order = (2, 0, 1)
            
            model = ARIMA(returns, order=order)
            model_fit = model.fit()
            
            # Forecast returns
            returns_forecast = model_fit.forecast(steps=days)
            
            # Add mean reversion component based on RSI
            reversion_strength = 0.1 if 30 <= rsi <= 70 else 0.2
            mean_return = returns.mean()
            returns_forecast = returns_forecast + (mean_return - returns_forecast) * reversion_strength
            
            # Convert returns to prices
            last_price = df['Close'].iloc[-1]
            prices = [last_price]
            for ret in returns_forecast:
                next_price = prices[-1] * (1 + ret)
                prices.append(next_price)
            
            prices = np.array(prices[1:])
            
            model_summary = {
                'aic': model_fit.aic,
                'bic': model_fit.bic
            }

            return prices, model_summary
        except Exception as e:
            print(f"ARIMA model error: {e}")
            return [np.nan] * days, {'model': 'ARIMA', 'aic': None, 'bic': None}

    def analyze_position(self, df):
        """Analyze current and future market position"""
        latest = df.iloc[-1]
        price = latest['Close']
        rsi = latest['RSI']
        volume_ratio = latest['Volume_Ratio']
        sma20 = latest['SMA20']
        sma50 = latest['SMA50']
        atr = latest['ATR']

        # Calculate additional metrics
        volatility = df['Returns'].std() * np.sqrt(252)
        annual_return = df['Returns'].mean() * 252
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        signal = 'HOLD'
        if price > sma20 and price > sma50 and rsi < 40 and volume_ratio > 1.2:
            signal = 'BUY'
        elif price < sma20 and price < sma50 and rsi > 60:
            signal = 'SELL'

        future_lr, summary_lr = self.predict_future_prices_with_linear_regression(df)
        future_arima, summary_arima = self.predict_future_prices_with_arima(df)

        # Create visualizations
        self.create_analysis_plots(df, future_lr, future_arima)

        return {
            'current_metrics': {
                'price': price,
                'signal': signal,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'sma20': sma20,
                'sma50': sma50,
                'atr': atr,
                'volatility': volatility,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio
            },
            'predictions': {
                'linear_regression': {
                    'prices': future_lr,
                    'summary': summary_lr
                },
                'arima': {
                    'prices': future_arima,
                    'summary': summary_arima
                }
            }
        }

    def create_analysis_plots(self, df, future_lr, future_arima):
        """Create comprehensive analysis plots using plotly"""
        # Create subplots
        fig = make_subplots(rows=2, cols=2, 
                            specs=[[{"colspan": 2}, None],
                                [{}, {}]],
                            vertical_spacing=0.1,
                            horizontal_spacing=0.1,
                            subplot_titles=(f'{self.symbol} Price History and Predictions', 
                                        'RSI Indicator', 
                                        'Volume Analysis'))

        # Plot 1: Price History and Predictions
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'],
                                name='Historical Price',
                                line=dict(color='blue')),
                    row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'],
                                name='20-day SMA',
                                line=dict(color='orange', dash='dash')),
                    row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'],
                                name='50-day SMA',
                                line=dict(color='red', dash='dash')),
                    row=1, col=1)
        
        # Add predictions
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(future_lr) + 1)[1:]
        
        fig.add_trace(go.Scatter(x=future_dates, y=future_lr,
                                name='Linear Regression',
                                line=dict(color='green', dash='dot')),
                    row=1, col=1)
        
        fig.add_trace(go.Scatter(x=future_dates, y=future_arima,
                                name='ARIMA',
                                line=dict(color='purple', dash='dot')),
                    row=1, col=1)
        
        # Plot 2: RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                                name='RSI',
                                line=dict(color='blue')),
                    row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Plot 3: Volume and Volume Ratio
        volume_norm = df['Volume'] / df['Volume'].max()
        volume_ratio_norm = df['Volume_Ratio'] / df['Volume_Ratio'].max()
        
        fig.add_trace(go.Bar(x=df.index, y=volume_norm,
                            name='Volume',
                            marker_color='lightblue',
                            opacity=0.3),
                    row=2, col=2)
        
        fig.add_trace(go.Scatter(x=df.index, y=volume_ratio_norm,
                                name='Volume Ratio',
                                line=dict(color='red')),
                    row=2, col=2)
        
        # Update layout
        fig.update_layout(height=800,
                        showlegend=True,
                        title_text="Stock Analysis Dashboard")
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="Normalized Values", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
pass

def create_candlestick_plot(df, future_lr, future_arima):
    """Create interactive candlestick chart with predictions"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='OHLC'),
                  row=1, col=1)

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'],
                            name='SMA20', line=dict(color='orange')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'],
                            name='SMA50', line=dict(color='blue')),
                  row=1, col=1)

    # Add predictions
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(future_lr) + 1)[1:]
    
    fig.add_trace(go.Scatter(x=future_dates, y=future_lr,
                            name='LR Prediction',
                            line=dict(color='green', dash='dash')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=future_dates, y=future_arima,
                            name='ARIMA Prediction',
                            line=dict(color='red', dash='dash')),
                  row=1, col=1)

    # Volume bar chart
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
                        name='Volume'),
                  row=2, col=1)

    # Update layout
    fig.update_layout(
        title='Stock Price Analysis with Predictions',
        yaxis_title='Stock Price ($)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=800
    )

    return fig
        
def format_prediction_summary(prices, days=90):
    """Format prediction summary for output"""
    intervals = [5, 10, 15, 30, 60, 90]
    return {f"{i}d": prices[i-1] for i in intervals if i <= days}

def main():
    st.set_page_config(layout="wide")
    st.title("Stock Analysis Dashboard")

    # Sidebar inputs
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL").upper()
    prediction_days = st.sidebar.slider("Prediction Days:", 5, 90, 30)
    
    # Main analysis
    if st.sidebar.button("Analyze"):
        try:
            analyzer = StockAnalyzer(symbol)
            df = analyzer.fetch_data()
            
            if df is not None:
                analysis = analyzer.analyze_position(df)
                
                # Create two columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Price Chart and Predictions")
                    # Create and display interactive plot
                    fig = create_candlestick_plot(
                        df, 
                        analysis['predictions']['linear_regression']['prices'],
                        analysis['predictions']['arima']['prices']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Current Metrics
                    st.subheader("Current Metrics")
                    metrics = analysis['current_metrics']
                    metrics_df = pd.DataFrame({
                        'Metric': [
                            'Current Price',
                            'Trading Signal',
                            'RSI',
                            'Volume Ratio',
                            'ATR',
                            'Volatility (Annual)',
                            'Annual Return',
                            'Sharpe Ratio'
                        ],
                        'Value': [
                            f"${metrics['price']:.2f}",
                            metrics['signal'],
                            f"{metrics['rsi']:.2f}",
                            f"{metrics['volume_ratio']:.2f}",
                            f"{metrics['atr']:.2f}",
                            f"{metrics['volatility']*100:.2f}%",
                            f"{metrics['annual_return']*100:.2f}%",
                            f"{metrics['sharpe_ratio']:.2f}"
                        ]
                    })
                    st.dataframe(metrics_df, hide_index=True)
                
                # Predictions
                st.subheader("Price Predictions")
                lr_pred = analysis['predictions']['linear_regression']['prices']
                arima_pred = analysis['predictions']['arima']['prices']
                current_price = metrics['price']
                
                intervals = [5, 10, 15, 30, 60, 90]
                pred_data = []
                
                for i in intervals:
                    if i <= prediction_days:
                        idx = i - 1
                        lr_price = lr_pred[idx]
                        arima_price = arima_pred[idx]
                        lr_pct = ((lr_price - current_price) / current_price) * 100
                        arima_pct = ((arima_price - current_price) / current_price) * 100
                        
                        pred_data.append({
                            'Horizon': f"{i}d",
                            'Linear Regression': f"${lr_price:.2f} ({lr_pct:+.1f}%)",
                            'ARIMA': f"${arima_price:.2f} ({arima_pct:+.1f}%)"
                        })
                
                pred_df = pd.DataFrame(pred_data)
                st.dataframe(pred_df, hide_index=True)
                
                # Model Performance Metrics
                st.subheader("Model Performance")
                lr_metrics = analysis['predictions']['linear_regression']['summary']
                arima_metrics = analysis['predictions']['arima']['summary']
                
                perf_data = {
                    'Metric': ['R-squared', 'RMSE', 'MAE', 'AIC', 'BIC'],
                    'Linear Regression': [
                        f"{lr_metrics['r_squared']:.4f}",
                        f"{lr_metrics['rmse']:.2f}",
                        f"{lr_metrics['mae']:.2f}",
                        'N/A',
                        'N/A'
                    ],
                    'ARIMA': [
                        'N/A',
                        'N/A',
                        'N/A',
                        f"{arima_metrics['aic']:.2f}",
                        f"{arima_metrics['bic']:.2f}"
                    ]
                }
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, hide_index=True)
                
            else:
                st.error(f"Unable to fetch data for {symbol}")
                
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {str(e)}")

if __name__ == "__main__":
    main()
    
    
