import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# --- 0. 配置與數據獲取 ---

# 映射資產到 Yahoo Finance Ticker
TICKER_MAP = {
    'BTC/USDT': 'BTC-USD',
    'ETH/USDT': 'ETH-USD',
    'SOL/USDT': 'SOL-USD',
    'BNB/USDT': 'BNB-USD',
}

@st.cache_data
def get_historical_prices(asset_name, period_days=365):
    """從 Yahoo Finance 獲取指定資產的歷史收盤價格 (1 年)。"""
    ticker_symbol = TICKER_MAP.get(asset_name, 'BTC-USD')
    
    end_date = date.today()
    start_date = end_date - timedelta(days=period_days)
    
    st.info(f"🔄 正在從 Yahoo Finance 獲取 {ticker_symbol} 過去 {period_days} 天的歷史數據...")
    
    try:
        # 使用進度條來模擬數據加載
        data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.error(f"❌ 未能獲取 {ticker_symbol} 的數據。")
            return None
        
        # 使用 Close 價格進行回測，並將價格轉換為 DataFrame 以便繪圖
        prices = data['Close'].dropna()
        return prices.rename('Price')
        
    except Exception as e:
        st.error(f"獲取數據時發生錯誤: {e}。請檢查資產名稱或網絡連線。")
        return None

# --- 1. 價格數據模擬 (作為真實數據獲取失敗時的備用) ---
def generate_mock_prices(initial_price=60000, num_steps=1000):
    """模擬一個價格路徑 (帶有輕微向上趨勢和波動)"""
    np.random.seed(42) 
    trend = np.linspace(0, 0.05 * initial_price, num_steps)
    volatility = np.random.randn(num_steps) * (initial_price / 3000)
    prices = initial_price + trend + volatility
    prices = np.maximum(prices, initial_price * 0.95) 
    return pd.Series(prices, name='Price')

# --- 2. 網格計算邏輯 ---

def calculate_grids(lower_limit, upper_limit, num_grids, grid_type):
    """根據選擇的類型生成網格價格。"""
    if num_grids < 1:
        st.error("網格數必須大於0。")
        return []
    
    if grid_type == '等差網格 (Arithmetic)':
        grids = np.linspace(lower_limit, upper_limit, num_grids + 1)
    
    elif grid_type == '等比網格 (Geometric)':
        grids = np.geomspace(lower_limit, upper_limit, num_grids + 1)
    
    grids.sort()
    return [round(float(p), 2) for p in grids]

# --- 3. 回測模擬器 ---

def run_backtest(price_data, grids, trade_size=0.01, fee_rate=0.001):
    """執行網格回測模擬 (使用真實或模擬數據)。"""
    
    num_levels = len(grids)
    if num_levels < 2: return 0, 0, 0, []

    total_profit = 0
    completed_cycles = 0
    current_position = 0
    last_buy_price = 0
    
    # 根據起始價格確定初始網格位置
    initial_price = price_data.iloc[0]
    last_grid_index = next((i for i, p in enumerate(grids) if p >= initial_price), num_levels - 1)
    
    trade_log = []

    for i in range(1, len(price_data)):
        current_price = price_data.iloc[i]
        
        # 價格下跌觸發買入
        if current_price < grids[last_grid_index] and last_grid_index > 0:
            triggered_index = -1
            for j in range(last_grid_index - 1, -1, -1):
                if current_price < grids[j]:
                    triggered_index = j
                else:
                    break
            
            if triggered_index != -1:
                buy_price = grids[triggered_index]
                current_position += trade_size
                last_buy_price = buy_price
                
                trade_log.append({
                    'Time_Index': price_data.index[i], 'Price': current_price,
                    'Action': 'BUY (買入)', 'Amount': trade_size, 
                    'Grid_Price': buy_price, 'Profit': 0,
                    'Note': f"價格下穿網格線 {triggered_index}"
                })
                last_grid_index = triggered_index
                
        # 價格上漲觸發賣出
        elif current_price > grids[last_grid_index] and last_grid_index < num_levels - 1:
            triggered_index = -1
            for j in range(last_grid_index + 1, num_levels):
                if current_price > grids[j]:
                    triggered_index = j
                else:
                    break
            
            if triggered_index != -1:
                sell_price = grids[triggered_index]
                revenue = sell_price * trade_size * (1 - fee_rate)
                
                if current_position >= trade_size:
                    profit = revenue - (last_buy_price * trade_size * (1 + fee_rate))
                    total_profit += profit
                    current_position -= trade_size
                    completed_cycles += 1
                    
                    trade_log.append({
                        'Time_Index': price_data.index[i], 'Price': current_price,
                        'Action': 'SELL (賣出)', 'Amount': trade_size, 
                        'Grid_Price': sell_price, 'Profit': profit,
                        'Note': f"價格上穿網格線 {triggered_index}，完成循環"
                    })
                else:
                    trade_log.append({
                        'Time_Index': price_data.index[i], 'Price': current_price,
                        'Action': 'SELL (賣出)', 'Amount': trade_size, 
                        'Grid_Price': sell_price, 'Profit': 0,
                        'Note': f"價格上穿網格線 {triggered_index}，無對應買入倉位"
                    })
                
                last_grid_index = triggered_index
    
    average_grid_profit = total_profit / completed_cycles if completed_cycles > 0 else 0
    
    return total_profit, completed_cycles, average_grid_profit, trade_log

# --- 4. Streamlit 應用程式界面 ---

st.set_page_config(layout="wide", page_title="現貨網格機器人模擬推演")

st.title("💰 現貨網格機器人模擬推演 (Pionex Style)")
st.caption("作者：Google Gemini | **數據來源: Yahoo Finance 過去一年歷史收盤價**")

# --- 側邊欄輸入設定 ---
st.sidebar.header("📈 策略與參數設定 (派網風格)")

asset = st.sidebar.selectbox(
    "選擇標的資產 (Asset)",
    ('BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'),
    index=0
)

# 根據選擇的資產動態設定網格上限
num_grids_max = 1000 if 'BTC' in asset else 500
num_grids_default = min(100, num_grids_max)

st.sidebar.subheader("網格區間設定")

# 獲取一年的歷史價格
with st.spinner(f"正在加載 {asset} 過去 1 年的數據..."):
    # 預先加載數據
    price_data_real = get_historical_prices(asset)

# 設定價格區間預設值
if price_data_real is not None and len(price_data_real) > 0:
    real_min = price_data_real.min()
    real_max = price_data_real.max()
    mid_price = (real_min + real_max) / 2
    
    st.sidebar.info(f"實際價格區間: {real_min:,.2f} ~ {real_max:,.2f}")
    
    # 預設網格範圍為實際價格範圍的 80%
    price_range = real_max - real_min
    default_lower = max(1.0, real_min + price_range * 0.1)
    default_upper = real_max - price_range * 0.1
else:
    # 數據加載失敗，使用模擬價格的預設值
    st.warning("⚠️ 無法獲取真實數據，使用模擬價格預設值。")
    if 'BTC' in asset: mock_start_price = 60000.0 
    elif 'ETH' in asset: mock_start_price = 3000.0
    else: mock_start_price = 150.0
    
    default_lower = mock_start_price * 0.9
    default_upper = mock_start_price * 1.1

col_lower, col_upper = st.sidebar.columns(2)
lower_limit = col_lower.number_input("下限價格 (Lower Limit)", min_value=1.0, value=default_lower, step=10.0, format="%.2f")
upper_limit = col_upper.number_input("上限價格 (Upper Limit)", min_value=1.0, value=default_upper, step=10.0, format="%.2f")

# 調整後的網格數量限制
num_grids = st.sidebar.slider("網格數量 (Grid Count)", 
                              min_value=5, 
                              max_value=num_grids_max, 
                              value=num_grids_default, 
                              step=5,
                              help=f"BTC 最大 1000 格，其他最大 500 格。")
                              
grid_type = st.sidebar.radio(
    "網格類型 (Grid Type)",
    ('等差網格 (Arithmetic)', '等比網格 (Geometric)'),
    horizontal=True
)

st.sidebar.subheader("交易參數")
trade_size = st.sidebar.number_input("單筆交易量 (Trade Size, 基礎資產)", min_value=0.0001, value=0.01, step=0.0001, format="%.4f", help="每次買入/賣出的基礎資產數量 (例如 0.01 BTC)")
fee_rate = st.sidebar.number_input("單邊手續費率 (Fee Rate, 例如 0.1%)", min_value=0.0, max_value=0.01, value=0.001, step=0.0001, format="%.4f", help="每筆交易的費率 (例如 0.001 代表 0.1%)")

# --- 主要內容區塊 ---

if lower_limit >= upper_limit:
    st.error("❌ 錯誤：上限價格必須大於下限價格。請調整側邊欄的設定。")
else:
    # 1. 確定價格數據源
    if price_data_real is not None and len(price_data_real) > 0:
        price_data = price_data_real
    else:
        # 使用模擬數據作為最終備用
        st.warning("⚠️ 由於無法獲取真實數據，將使用模擬價格進行回測。")
        price_data = generate_mock_prices(initial_price=mid_price, num_steps=1000)

    # 2. 計算網格價格
    grids = calculate_grids(lower_limit, upper_limit, num_grids, grid_type)
    
    # 計算網格利潤率
    grid_profit_rates = [
        (grids[i+1] / grids[i] - 1) * 100 
        for i in range(len(grids) - 1)
    ]
    
    min_profit_rate = min(grid_profit_rates) if grid_profit_rates else 0
    avg_profit_rate = sum(grid_profit_rates) / len(grid_profit_rates) if grid_profit_rates else 0

    # 估算所需資金
    estimated_min_capital = num_grids * trade_size * lower_limit
    
    
    # 3. 執行回測
    total_profit, completed_cycles, average_grid_profit, trade_log = run_backtest(
        price_data, grids, trade_size, fee_rate
    )
    
    # 網格利潤甜蜜點指標 (效率指標)
    grid_profitability = (total_profit / estimated_min_capital) * 100 if estimated_min_capital > 0 else 0
    
    
    # --- 指標卡片顯示 (Pionex Style) ---
    st.header("🎯 策略回測表現 (過去 1 年)")
    st.markdown(f"**回測期間**: {price_data.index.min().strftime('%Y-%m-%d')} 至 {price_data.index.max().strftime('%Y-%m-%d')}")

    # 第一行：主要成果
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        label="🟢 總網格利潤 (USDT)", 
        value=f"{total_profit:,.2f}",
        delta="已實現套利"
    )
    col2.metric(
        label="🔄 完整循環次數", 
        value=f"{completed_cycles}",
        delta="越多代表震盪越頻繁"
    )
    col3.metric(
        label="📊 網格套利效率 (%)", 
        value=f"{grid_profitability:,.2f}%",
        delta="基於最低資金的 ROI (簡化)"
    )
    col4.metric(
        label="💸 估計最低資金 (USDT)", 
        value=f"約 {estimated_min_capital:,.2f}",
        help="簡化估算：網格數 × 單筆交易量 × 下限價格"
    )

    # 第二行：網格參數細節
    st.subheader("⚙️ 網格參數細節")
    col5, col6, col7, col8 = st.columns(4)
    
    col5.metric(
        label="⬆️ 價格上限", 
        value=f"{upper_limit:,.2f}"
    )
    col6.metric(
        label="⬇️ 價格下限", 
        value=f"{lower_limit:,.2f}"
    )
    col7.metric(
        label="📉 最小網格利潤率", 
        value=f"{min_profit_rate:,.2f}%",
        help="單格未扣手續費的最小利潤百分比"
    )
    col8.metric(
        label="💰 平均單格利潤 (USDT)", 
        value=f"{average_grid_profit:,.4f}"
    )

    
    # --- 網格線價格與分佈圖表 ---

    st.subheader("價格路徑與網格分佈")
    
    # 顯示網格細節表格
    grid_df = pd.DataFrame({
        'Level': range(num_grids),
        'Buy_Grid_Price': grids[:-1],
        'Sell_Grid_Price': grids[1:],
        'Grid_Profit_Rate (%)': grid_profit_rates,
    }) 
    st.dataframe(grid_df, use_container_width=True, hide_index=True)


    # 繪製價格曲線和網格線
    st.subheader("價格路徑與網格分佈圖")
    
    # 使用日期作為 X 軸
    chart_df = price_data.to_frame().reset_index()
    chart_df.columns = ['Date', 'Price']
    
    chart_data = [{'price': p, 'type': 'Grid Level'} for p in grids]
    chart_data.append({'price': lower_limit, 'type': 'Lower Limit'})
    chart_data.append({'price': upper_limit, 'type': 'Upper Limit'})
    
    import altair as alt
    
    line_chart = alt.Chart(chart_df).mark_line(color='#10B981', size=1).encode(
        x=alt.X('Date', title='日期 (Date)'),
        y=alt.Y('Price', title=f'{asset} 價格 (Price)'),
        tooltip=[alt.Tooltip('Date', format='%Y-%m-%d'), alt.Tooltip('Price', format=',.2f')]
    ).properties(
        title=f'{asset} 過去一年價格路徑與網格分佈'
    )
    
    grid_lines = alt.Chart(pd.DataFrame(chart_data)).mark_rule().encode(
        y='price',
        color=alt.Color('type', scale=alt.Scale(domain=['Lower Limit', 'Upper Limit', 'Grid Level'], range=['#EF4444', '#3B82F6', '#9CA3AF'])),
        tooltip=[alt.Tooltip('price', format=',.2f'), 'type']
    )
    
    st.altair_chart(line_chart + grid_lines, use_container_width=True)

    # 5. 交易記錄
    st.subheader("交易記錄 (Trade Log)")
    if trade_log:
        log_df = pd.DataFrame(trade_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ 在當前網格設定下，價格路徑未觸發任何完整的套利循環交易。請調整您的上下限區間。")

    st.header("💡 甜蜜點尋找策略")
    st.markdown("""
    網格利潤的**甜蜜點**是風險、回報與投入資金之間的最佳平衡點。參考派網的實戰經驗，您可以專注於以下調整：

    1.  **最小網格利潤率 (Min Grid Profit Rate)**：
        * **原則**：此值**必須**高於雙邊手續費率的總和（例如 $2 \times 0.1\% = 0.2\%$）。如果您的最小利潤率低於總手續費，您每完成一個網格循環就會虧損。
        * **調整方式**：增加網格區間或減少網格數量。
    2.  **網格套利效率 (%)**：
        * **原則**：這是衡量您的資金效率的關鍵指標。您希望在有限的資金投入下（估計最低資金），獲得最大的總網格利潤。
        * **調整方式**：
            * **高波動性資產** (如 SOL, ETH)：適合使用**等比網格**，並將區間設置得更寬鬆一些。
            * **低波動性資產** (如 BTC)：適合使用**等差網格**，並將區間設置得更緊密。
    3.  **交易量 (Trade Size)**：如果資金充裕，增加單筆交易量會直接增加總利潤，但也會增加所需資金。

    **最佳化目標：在確保 `最小網格利潤率 > 2 * 手續費率` 的前提下，最大化 `網格套利效率 (%)`。**
    """)
