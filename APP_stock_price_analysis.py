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
        """Fetch and prepare stock data with retries and better error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"\nFetching data for {self.symbol}... (Attempt {attempt + 1}/{max_retries})")
                
                # Try different date ranges if needed
                if attempt == 0:
                    df = yf.download(self.symbol, start='2020-01-01', progress=False)
                elif attempt == 1:
                    # Try a more recent start date
                    df = yf.download(self.symbol, start='2022-01-01', progress=False)
                else:
                    # Try with period instead of start date
                    df = yf.download(self.symbol, period="2y", progress=False)
                
                if df.empty:
                    print(f"No data found for {self.symbol} on attempt {attempt + 1}")
                    continue
                    
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.droplevel(level=1, axis=1)
                
                if 'Close' not in df.columns:
                    print(f"Missing 'Close' column for {self.symbol} on attempt {attempt + 1}")
                    continue
                    
                if df['Close'].isnull().all():
                    print(f"No valid closing prices for {self.symbol} on attempt {attempt + 1}")
                    continue
                
                # If we got here, we have valid data
                print(f"Successfully fetched data for {self.symbol}")
                
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
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to fetch data for {self.symbol} after {max_retries} attempts: {str(e)}")
                continue

    def predict_future_prices_with_linear_regression(self, df, days=90):
        """Predict future prices using linear regression with robust handling for volatile stocks and standard deviation bands"""
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
        
        # Calculate standard deviation bands
        std_daily = np.sqrt(mse)  # Daily standard deviation of prediction error
        time_factor = np.sqrt(np.arange(1, days + 1))  # Uncertainty increases with time
        
        std1_upper = future_prices + std_daily * time_factor
        std1_lower = future_prices - std_daily * time_factor
        std2_upper = future_prices + 2 * std_daily * time_factor
        std2_lower = future_prices - 2 * std_daily * time_factor
        
        confidence_intervals = {
            'std1_upper': std1_upper,
            'std1_lower': std1_lower,
            'std2_upper': std2_upper,
            'std2_lower': std2_lower
        }
        
        model_summary = {
            'r_squared': r_squared,
            'coefficient': coefficients,
            'intercept': intercept,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

        return future_prices, confidence_intervals, model_summary

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
        volatility = df['Returns'].std() * np.sqrt(252)  # Annualized volatility
        annual_return = df['Returns'].mean() * 252
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # Enhanced trading signal logic
        signal = 'HOLD'
        
        # Buy conditions:
        # 1. Extremely oversold (RSI < 30)
        # 2. OR Price below both SMAs with improving RSI
        # 3. OR Strong volume with price above SMAs
        if (rsi < 30) or \
        (price < sma20 and price < sma50 and rsi > df['RSI'].shift(1).iloc[-1]) or \
        (price > sma20 and price > sma50 and volume_ratio > 1.2):
            signal = 'BUY'
        
        # Sell conditions:
        # 1. Extremely overbought (RSI > 70)
        # 2. OR Price below both SMAs with declining volume
        elif (rsi > 70) or \
            (price < sma20 and price < sma50 and volume_ratio < 0.8):
            signal = 'SELL'

        # Get predictions with confidence intervals
        future_lr, lr_confidence, summary_lr = self.predict_future_prices_with_linear_regression(df)
        future_arima, summary_arima = self.predict_future_prices_with_arima(df)

        # Create visualizations
        self.create_analysis_plots(df, future_lr, future_arima, lr_confidence)

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
                    'confidence_intervals': lr_confidence,
                    'summary': summary_lr
                },
                'arima': {
                    'prices': future_arima,
                    'summary': summary_arima
                }
            }
        }

    def create_analysis_plots(self, df, future_lr, future_arima, lr_confidence):
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
        
        # Add predictions and confidence intervals
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(future_lr) + 1)[1:]
        
        # Add confidence intervals (shaded areas)
        fig.add_trace(go.Scatter(
            x=future_dates.tolist() + future_dates.tolist()[::-1],
            y=lr_confidence['std2_upper'].tolist() + lr_confidence['std2_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,255,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence',
            showlegend=True
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=future_dates.tolist() + future_dates.tolist()[::-1],
            y=lr_confidence['std1_upper'].tolist() + lr_confidence['std1_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,255,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='68% Confidence',
            showlegend=True
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=future_dates, y=future_lr,
                                name='Linear Regression',
                                line=dict(color='green', dash='dash')),
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

def create_candlestick_plot(df, future_lr, future_arima, lr_confidence, display_days):
    """Create interactive candlestick chart with predictions and confidence intervals"""
    # Filter dataframe to show only the selected number of days
    df_filtered = df.last(f'{display_days}D')
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df_filtered.index,
                                open=df_filtered['Open'],
                                high=df_filtered['High'],
                                low=df_filtered['Low'],
                                close=df_filtered['Close'],
                                name='OHLC'),
                  row=1, col=1)

    # Add Moving Averages
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['SMA20'],
                            name='SMA20', line=dict(color='orange')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['SMA50'],
                            name='SMA50', line=dict(color='blue')),
                  row=1, col=1)

    # Add predictions and confidence intervals
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(future_lr) + 1)[1:]
    
    # Select points for labels (30, 60, and 90 days out if available)
    label_indices = [min(i, len(future_lr)-1) for i in [29, 59, 89] if i < len(future_lr)]
    
    # Add annotations for confidence intervals at selected points
    for idx in label_indices:
        # 95% CI annotation
        fig.add_annotation(
            x=future_dates[idx],
            y=lr_confidence['std2_upper'][idx],
            text=f"${lr_confidence['std2_upper'][idx]:.2f}",
            showarrow=False,
            yshift=10,
            font=dict(size=10)
        )
        fig.add_annotation(
            x=future_dates[idx],
            y=lr_confidence['std2_lower'][idx],
            text=f"${lr_confidence['std2_lower'][idx]:.2f}",
            showarrow=False,
            yshift=-10,
            font=dict(size=10)
        )
        
        # 68% CI annotation
        fig.add_annotation(
            x=future_dates[idx],
            y=lr_confidence['std1_upper'][idx],
            text=f"${lr_confidence['std1_upper'][idx]:.2f}",
            showarrow=False,
            yshift=10,
            font=dict(size=10)
        )
        fig.add_annotation(
            x=future_dates[idx],
            y=lr_confidence['std1_lower'][idx],
            text=f"${lr_confidence['std1_lower'][idx]:.2f}",
            showarrow=False,
            yshift=-10,
            font=dict(size=10)
        )
        
        # Add predicted price at this point
        fig.add_annotation(
            x=future_dates[idx],
            y=future_lr[idx],
            text=f"${future_lr[idx]:.2f}",
            showarrow=False,
            font=dict(size=10, color='green')
        )
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=future_dates.tolist() + future_dates.tolist()[::-1],
        y=lr_confidence['std2_upper'].tolist() + lr_confidence['std2_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,255,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence',
        showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=future_dates.tolist() + future_dates.tolist()[::-1],
        y=lr_confidence['std1_upper'].tolist() + lr_confidence['std1_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,255,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='68% Confidence',
        showlegend=True
    ), row=1, col=1)

    # Add prediction lines
    fig.add_trace(go.Scatter(x=future_dates, y=future_lr,
                            name='LR Prediction',
                            line=dict(color='green', dash='dash')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=future_dates, y=future_arima,
                            name='ARIMA Prediction',
                            line=dict(color='red', dash='dash')),
                  row=1, col=1)

    # Volume bar chart
    fig.add_trace(go.Bar(x=df_filtered.index, y=df_filtered['Volume'],
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

def main():
    st.set_page_config(layout="wide")
    st.title("Stock Analysis Dashboard")

    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None

    # Fixed header for inputs
    with st.container():
        st.markdown("### Settings")
        col_input1, col_input2, col_input3 = st.columns([2, 1, 1])
        with col_input1:
            symbol = st.text_input("Enter Stock Symbol:", value="AAPL").upper()
        with col_input2:
            prediction_days = st.slider("Prediction Days:", 5, 90, 30)
        with col_input3:
            analyze_button = st.button("Analyze")

    # Main analysis
    if analyze_button:
        try:
            analyzer = StockAnalyzer(symbol)
            with st.spinner(f'Fetching data for {symbol}...'):
                df = analyzer.fetch_data()
                st.session_state.current_df = df
                analysis = analyzer.analyze_position(df)
                st.session_state.current_analysis = analysis
                st.session_state.analysis_complete = True

        except Exception as e:
            st.error(f"Error analyzing {symbol}: {str(e)}")
            st.write("If this error persists, please:")
            st.write("1. Verify the stock symbol")
            st.write("2. Check if the stock is actively traded")
            st.write("3. Try again in a few minutes")
            return

    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.current_df is not None:
        df = st.session_state.current_df
        analysis = st.session_state.current_analysis

        # Your existing metrics tables code here...
        
        st.markdown("---")  # Separator

        # Add the candlestick chart
        st.subheader("Price Chart (Candlestick)")

        # Calculate total days in the dataset
        total_days = (df.index.max() - df.index.min()).days

        # Create a slider for selecting the number of days to display
        display_days = st.slider("Select time window (days)", 
                       min_value=5,    # Minimum 5 days
                       max_value=60,   # Maximum 60 days
                       value=30,       # Default to 30 days
                       step=1)         # Single day steps

        # In your main function, modify the candlestick plot creation:
        fig = create_candlestick_plot(
            df, 
            analysis['predictions']['linear_regression']['prices'],
            analysis['predictions']['arima']['prices'],
            analysis['predictions']['linear_regression']['confidence_intervals'],
            display_days
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()