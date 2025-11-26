import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import altair as alt

# --- 0. 配置與數據獲取 ---

# 映射資產到 Yahoo Finance Ticker
TICKER_MAP = {
    'BTC/USDT': 'BTC-USD',
    'ETH/USDT': 'ETH-USD',
    'SOL/USDT': 'SOL-USD',
    'BNB/USDT': 'BNB-USD',
}

# 派網現貨網格標準單邊手續費率 (0.05%)
DEFAULT_FEE_RATE = 0.0005 

@st.cache_data
def get_historical_prices(asset_name, period_days=365):
    """從 Yahoo Finance 獲取指定資產的歷史收盤價格 (1 年)。"""
    ticker_symbol = TICKER_MAP.get(asset_name, 'BTC-USD')
    
    end_date = date.today()
    start_date = end_date - timedelta(days=period_days)
    
    st.info(f"🔄 正在從 Yahoo Finance 獲取 {ticker_symbol} 過去 {period_days} 天的歷史數據...")
    
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.error(f"❌ 未能獲取 {ticker_symbol} 的數據。")
            return None
        
        prices = data['Close'].dropna()
        return prices.rename('Price')
        
    except Exception as e:
        st.error(f"獲取數據時發生錯誤: {e}。請檢查資產名稱或網絡連線。")
        return None

# --- 1. 網格計算邏輯 ---

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

# --- 2. 回測模擬器 ---

def run_backtest(price_data, grids, trade_size, fee_rate):
    """執行網格回測模擬 (使用真實或模擬數據)。"""
    
    num_levels = len(grids)
    if num_levels < 2: return 0, 0, 0, []

    total_profit = 0
    completed_cycles = 0
    current_position = 0 # 追蹤基礎資產持倉
    last_buy_price = 0
    
    # 根據起始價格確定初始網格位置
    initial_price = price_data.iloc[0]
    last_grid_index = next((i for i, p in enumerate(grids) if p >= initial_price), num_levels - 1)
    
    trade_log = []

    # 使用 .iteritems() 迭代包含時間索引的價格數據
    for i, (time_index, current_price) in enumerate(price_data.items()):
        
        # 價格下跌觸發買入
        if current_price < grids[last_grid_index] and last_grid_index > 0:
            triggered_index = -1
            # 向下搜尋觸發的網格線
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
                    'Time_Index': time_index, 'Price': current_price,
                    'Action': 'BUY (買入)', 'Amount': trade_size, 
                    'Grid_Price': buy_price, 'Profit': 0,
                    'Note': f"價格下穿網格線 {triggered_index}"
                })
                last_grid_index = triggered_index
                
        # 價格上漲觸發賣出
        elif current_price > grids[last_grid_index] and last_grid_index < num_levels - 1:
            triggered_index = -1
            # 向上搜尋觸發的網格線
            for j in range(last_grid_index + 1, num_levels):
                if current_price > grids[j]:
                    triggered_index = j
                else:
                    break
            
            if triggered_index != -1:
                sell_price = grids[triggered_index]
                
                if current_position >= trade_size:
                    # 計算淨利潤: 賣出收入 - 買入成本 (含雙邊手續費)
                    # 買入成本 = last_buy_price * trade_size * (1 + fee_rate)
                    # 賣出收入 = sell_price * trade_size * (1 - fee_rate)
                    profit_on_trade = (sell_price * trade_size * (1 - fee_rate)) - (last_buy_price * trade_size * (1 + fee_rate))
                    
                    total_profit += profit_on_trade
                    current_position -= trade_size
                    completed_cycles += 1
                    
                    trade_log.append({
                        'Time_Index': time_index, 'Price': current_price,
                        'Action': 'SELL (賣出)', 'Amount': trade_size, 
                        'Grid_Price': sell_price, 'Profit': profit_on_trade,
                        'Note': f"價格上穿網格線 {triggered_index}，完成循環"
                    })
                else:
                    trade_log.append({
                        'Time_Index': time_index, 'Price': current_price,
                        'Action': 'SELL (賣出)', 'Amount': trade_size, 
                        'Grid_Price': sell_price, 'Profit': 0,
                        'Note': f"價格上穿網格線 {triggered_index}，無對應買入倉位"
                    })
                
                last_grid_index = triggered_index
    
    average_grid_profit = total_profit / completed_cycles if completed_cycles > 0 else 0
    
    return total_profit, completed_cycles, average_grid_profit, trade_log

# --- 3. Streamlit 應用程式界面 ---

st.set_page_config(layout="wide", page_title="現貨網格機器人模擬推演")

st.title("💰 現貨網格機器人回測與淨利潤推算")
st.caption("作者：Google Gemini | **數據來源: Yahoo Finance 過去一年歷史收盤價**")

# --- 側邊欄輸入設定 ---
st.sidebar.header("📈 策略與參數設定")

asset = st.sidebar.selectbox(
    "選擇標的資產 (Asset)",
    ('BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'),
    index=0
)

# 根據選擇的資產動態設定網格上限
num_grids_max = 1000 if 'BTC' in asset else 500
num_grids_default = min(500, num_grids_max) # 預設使用 500 格

st.sidebar.subheader("網格區間設定")

# 預先加載數據
price_data_real = get_historical_prices(asset)

# 設定價格區間預設值
if price_data_real is not None and len(price_data_real) > 0:
    real_min = price_data_real.min()
    real_max = price_data_real.max()
    
    st.sidebar.info(f"實際價格區間: {real_min:,.2f} ~ {real_max:,.2f}")
    
    # 預設網格範圍為實際價格範圍的 80% (或使用特定建議區間)
    if 'BTC' in asset: 
        default_lower, default_upper = 40000.0, 140000.0
    elif 'ETH' in asset:
        default_lower, default_upper = 1500.0, 5500.0
    elif 'SOL' in asset:
        default_lower, default_upper = 50.0, 300.0
    else:
        # 一般預設
        price_range = real_max - real_min
        default_lower = max(1.0, real_min * 0.9)
        default_upper = real_max * 1.1

    default_lower = max(1.0, min(default_lower, real_min))
    default_upper = max(real_max, default_upper)
else:
    st.error("⚠️ 無法獲取真實數據，請手動輸入區間。")
    default_lower = 30000.0
    default_upper = 70000.0


col_lower, col_upper = st.sidebar.columns(2)
lower_limit = col_lower.number_input("下限價格 (Lower Limit)", min_value=1.0, value=default_lower, step=10.0, format="%.2f")
upper_limit = col_upper.number_input("上限價格 (Upper Limit)", min_value=1.0, value=default_upper, step=10.0, format="%.2f")

# 調整後的網格數量限制
num_grids = st.sidebar.slider("網格數量 (Grid Count)", 
                              min_value=5, 
                              max_value=num_grids_max, 
                              value=num_grids_default, 
                              step=5,
                              help=f"BTC 最大 {num_grids_max} 格，其他最大 {num_grids_max} 格。")
                              
grid_type = st.sidebar.radio(
    "網格類型 (Grid Type)",
    ('等比網格 (Geometric)', '等差網格 (Arithmetic)'), # 預設等比
    horizontal=True
)

st.sidebar.subheader("交易與利潤目標")
trade_size = st.sidebar.number_input("單筆交易量 (Trade Size, 基礎資產)", min_value=0.0001, value=0.01, step=0.0001, format="%.4f", help="每次買入/賣出的基礎資產數量 (例如 0.01 BTC)")

# 根據研究結果，手續費率預設為 0.05%
fee_rate = st.sidebar.number_input("單邊手續費率 (Fee Rate, 0.05% = 0.0005)", min_value=0.0, max_value=0.01, value=DEFAULT_FEE_RATE, step=0.0001, format="%.4f", help="派網標準為 0.0005 (0.05%)")

# 淨利潤目標
target_net_profit_rate = st.sidebar.number_input("目標淨網格利潤 (%)", min_value=0.01, max_value=5.0, value=0.15, step=0.01, format="%.2f", help="您希望每個網格完成一買一賣後，扣除手續費的淨利潤百分比。")

# 執行回測按鈕
run_button = st.sidebar.button("🚀 執行回測 (使用歷史數據)", type="primary")

# --- 主要內容區塊 ---

if run_button and lower_limit < upper_limit:
    
    # 1. 確定價格數據源
    if price_data_real is not None and len(price_data_real) > 0:
        price_data = price_data_real
    else:
        st.error("⚠️ 無法取得歷史數據，請確認網絡連線或稍後重試。")
        st.stop()
    
    # 2. 計算網格價格
    grids = calculate_grids(lower_limit, upper_limit, num_grids, grid_type)
    
    # 計算網格利潤率 (毛利潤)
    grid_profit_rates = [
        (grids[i+1] / grids[i] - 1) * 100 
        for i in range(len(grids) - 1)
    ]
    
    min_profit_rate_gross = min(grid_profit_rates) if grid_profit_rates else 0
    avg_profit_rate_gross = sum(grid_profit_rates) / len(grid_profit_rates) if grid_profit_rates else 0

    # 估算所需資金
    estimated_min_capital = num_grids * trade_size * lower_limit
    
    # 3. 執行回測
    total_profit, completed_cycles, average_grid_profit, trade_log = run_backtest(
        price_data, grids, trade_size, fee_rate
    )
    
    # 網格利潤甜蜜點指標 (效率指標)
    grid_profitability = (total_profit / estimated_min_capital) * 100 if estimated_min_capital > 0 else 0
    
    # 4. 淨利潤要求計算
    # 總手續費率 = 單邊手續費率 * 2 (一買一賣)
    total_fee_rate_percent = fee_rate * 2 * 100 
    
    # 達成目標淨利潤所需的最小毛利潤率
    required_gross_rate = target_net_profit_rate + total_fee_rate_percent
    
    # --- 指標卡片顯示 (Pionex Style) ---
    st.header("🎯 策略回測表現 (過去 1 年)")
    st.markdown(f"**回測期間**: {price_data.index.min().strftime('%Y-%m-%d')} 至 {price_data.index.max().strftime('%Y-%m-%d')} | **數據點**: {len(price_data)} 點")

    # 第一行：主要成果
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        label="🟢 總網格淨利潤 (USDT)", 
        value=f"{total_profit:,.2f}",
        delta="已實現套利 (扣除手續費)"
    )
    col2.metric(
        label="🔄 完整循環次數", 
        value=f"{completed_cycles}",
        delta="總交易網格對數"
    )
    col3.metric(
        label="📊 網格套利效率 (%)", 
        value=f"{grid_profitability:,.2f}%",
        delta="資金總回報率 (年化需乘上倍數)"
    )
    col4.metric(
        label="💸 估計最低資金 (USDT)", 
        value=f"約 {estimated_min_capital:,.2f}",
        help="簡化估算：網格數 × 單筆交易量 × 下限價格"
    )

    # 第二行：網格參數與利潤要求細節
    st.subheader("⚙️ 網格利潤要求檢測")
    col5, col6, col7, col8 = st.columns(4)
    
    col5.metric(
        label="💰 單格雙邊總手續費率", 
        value=f"{total_fee_rate_percent:,.2f}%",
        help=f"單邊 {fee_rate*100:,.2f}%"
    )
    col6.metric(
        label="🎯 目標淨利潤率", 
        value=f"{target_net_profit_rate:,.2f}%"
    )
    col7.metric(
        label="⚠️ 最小毛利潤率要求", 
        value=f"{required_gross_rate:,.2f}%",
        help="網格間距毛利潤必須大於此值才能達標"
    )
    col8.metric(
        label="📈 當前最小網格毛利潤率", 
        value=f"{min_profit_rate_gross:,.2f}%"
    )
    
    # 網格利潤檢查
    if min_profit_rate_gross < required_gross_rate:
        st.error(f"❌ 警告：您的最小網格毛利潤率 ({min_profit_rate_gross:,.2f}%) **低於**目標要求 ({required_gross_rate:,.2f}%)！請減少網格數或擴大價格區間。")
    elif min_profit_rate_gross < total_fee_rate_percent:
        st.warning(f"⚠️ 注意：您的最小網格毛利潤率 ({min_profit_rate_gross:,.2f}%) **低於**總手續費 ({total_fee_rate_percent:,.2f}%)！網格循環將會虧損。")
    else:
        st.success("✅ 網格利潤率合格！已覆蓋手續費並達到目標淨利潤要求。")

    
    # --- 網格線價格與分佈圖表 ---

    st.subheader("價格路徑與網格分佈圖")
    
    # 顯示網格細節表格
    grid_df = pd.DataFrame({
        'Level': range(num_grids),
        'Buy_Grid_Price': grids[:-1],
        'Sell_Grid_Price': grids[1:],
        'Grid_Profit_Rate (Gross %)': grid_profit_rates,
    }) 
    
    # 繪製價格曲線和網格線
    
    # 使用日期作為 X 軸
    chart_df = price_data.to_frame().reset_index()
    chart_df.columns = ['Date', 'Price']
    
    chart_data = [{'price': p, 'type': 'Grid Level'} for p in grids]
    chart_data.append({'price': lower_limit, 'type': 'Lower Limit'})
    chart_data.append({'price': upper_limit, 'type': 'Upper Limit'})
    
    # 避免繪製過多網格線，僅顯示 50 條 (約每 N/50 條顯示一條)
    if len(grids) > 50:
        step = len(grids) // 50
        filtered_grid_data = [{'price': p, 'type': 'Grid Level'} for i, p in enumerate(grids) if i % step == 0]
        chart_data = filtered_grid_data
        chart_data.append({'price': lower_limit, 'type': 'Lower Limit'})
        chart_data.append({'price': upper_limit, 'type': 'Upper Limit'})


    line_chart = alt.Chart(chart_df).mark_line(color='#10B981', size=1).encode(
        x=alt.X('Date', title='日期 (Date)'),
        y=alt.Y('Price', title=f'{asset} 價格 (Price)'),
        tooltip=[alt.Tooltip('Date', format='%Y-%m-%d'), alt.Tooltip('Price', format=',.2f')]
    ).properties(
        title=f'{asset} 歷史價格路徑與網格分佈'
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
        st.caption("僅顯示前 100 筆交易")
        log_df = pd.DataFrame(trade_log)
        st.dataframe(log_df.head(100), use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ 在當前網格設定下，歷史價格路徑未觸發任何完整的套利循環交易。請調整您的上下限區間，確保價格在其範圍內波動。")

    st.header("💡 網格優化總結")
    st.markdown(f"""
    **手續費總結：** 派網現貨網格單邊手續費為 {DEFAULT_FEE_RATE * 100}%，一買一賣總手續費為 **{total_fee_rate_percent:,.2f}%**。
    
    **關鍵優化目標：**
    1. **利潤率安全線：** 您的網格最小毛利潤率必須 $\mathbf{\ge {required_gross_rate:,.2f}\%}$ 才能達到 $\mathbf{{target_net_profit_rate:,.2f}\%}$ 的淨利潤目標。
    2. **資金效率：** 觀察「網格套利效率 (%)」。這個值越高，代表在過去一年的市場條件下，您的網格設定用最少的資金捕捉到最多的套利機會。
    3. **網格類型：** 由於您主要採用**等比網格**，當價格上漲時，網格間距會擴大，**最小毛利潤率** 通常會在**最低價**區間，這是您最需要關注的瓶頸。
    """)

elif lower_limit >= upper_limit:
    st.error("❌ 錯誤：上限價格必須嚴格大於下限價格。請調整側邊欄的設定。")
else:
    st.info("👈 請在左側設定您的網格參數，並點擊 **🚀 執行回測** 開始分析。")
