import streamlit as st
import numpy as np
import pandas as pd

# --- 1. 價格數據模擬 ---
# 由於無法存取外部API，我們模擬一個有趨勢和噪音的價格數據
def generate_mock_prices(initial_price=30000, num_steps=500):
    """模擬一個價格路徑 (帶有輕微向上趨勢和波動)"""
    np.random.seed(42) # 確保每次運行結果一致
    
    # 建立一個基礎趨勢 (例如，緩慢上漲)
    trend = np.linspace(0, 0.05 * initial_price, num_steps)
    # 建立隨機波動
    volatility = np.random.randn(num_steps) * (initial_price / 3000)
    
    prices = initial_price + trend + volatility
    # 確保價格不會低於某個合理值
    prices = np.maximum(prices, initial_price * 0.95) 
    
    # 確保是整數索引 (模擬時間序列)
    return pd.Series(prices, name='Price')

# --- 2. 網格計算邏輯 ---

def calculate_grids(lower_limit, upper_limit, num_grids, grid_type):
    """根據選擇的類型生成網格價格。"""
    if num_grids < 1:
        st.error("網格數必須大於0。")
        return []
    
    if grid_type == '等差網格 (Arithmetic)':
        # 等差網格: 價格間距相等
        grids = np.linspace(lower_limit, upper_limit, num_grids + 1)
    
    elif grid_type == '等比網格 (Geometric)':
        # 等比網格: 價格比例相等 (log空間均分)
        # 使用 np.geomspace 但需要處理極端情況
        grids = np.geomspace(lower_limit, upper_limit, num_grids + 1)
    
    # 確保網格點是有序的 (從低到高)
    grids.sort()
    # 轉換為 Python list 並四捨五入到小數點後兩位
    return [round(float(p), 2) for p in grids]

# --- 3. 回測模擬器 ---

def run_backtest(price_data, grids, trade_size=0.01, fee_rate=0.001):
    """
    執行網格回測模擬。
    假設:
    1. 機器人從市場中性 (零倉位) 開始。
    2. 每筆交易量 (trade_size) 固定。
    3. 每個網格線都是一個觸發點。當價格穿越網格線時，嘗試進行交易。
    4. 採用市場中性策略: 價格下跌至網格線時買入 (建立倉位)，價格上漲至更高網格線時賣出 (平倉並獲利)。
    5. 網格線是 Buy/Sell 的觸發點，但為了簡化，我們只追蹤 Buy Low / Sell High 的循環。
    """
    
    # 網格線數量
    num_levels = len(grids)
    if num_levels < 2:
        return 0, 0, 0, []

    # 交易追蹤
    total_profit = 0  # 總利潤 (以 USDT 計價)
    completed_cycles = 0  # 完整 Buy -> Sell 循環次數
    current_position = 0  # 當前持倉量 (以基礎資產計價，例如 BTC)
    last_buy_price = 0  # 上次買入價格
    
    # 追蹤當前價格所處的網格區間 (索引從 0 到 num_levels - 1)
    # last_grid_index 儲存上次觸發交易的網格線索引
    # 初始化為中間網格區間
    last_grid_index = next((i for i, p in enumerate(grids) if p >= price_data.iloc[0]), num_levels - 1)
    
    # 用於儲存每次交易的詳細資訊
    trade_log = []

    # 模擬價格變動
    for i in range(1, len(price_data)):
        current_price = price_data.iloc[i]
        
        # 尋找當前價格位於哪個網格區間
        # 價格下跌觸發買入
        if current_price < grids[last_grid_index] and last_grid_index > 0:
            
            # 找到價格下穿的網格線
            triggered_index = -1
            for j in range(last_grid_index - 1, -1, -1):
                if current_price < grids[j]:
                    triggered_index = j
                else:
                    break
            
            if triggered_index != -1:
                buy_price = grids[triggered_index]
                cost = buy_price * trade_size * (1 + fee_rate)
                
                current_position += trade_size
                last_buy_price = buy_price
                
                trade_log.append({
                    'Time_Index': i, 
                    'Price': current_price,
                    'Action': 'BUY (買入)', 
                    'Amount': trade_size, 
                    'Grid_Price': buy_price,
                    'Note': f"價格下穿網格線 {triggered_index}"
                })
                
                # 更新當前所在網格區間
                last_grid_index = triggered_index
                
        
        # 價格上漲觸發賣出
        elif current_price > grids[last_grid_index] and last_grid_index < num_levels - 1:
            
            # 找到價格上穿的網格線
            triggered_index = -1
            for j in range(last_grid_index + 1, num_levels):
                if current_price > grids[j]:
                    triggered_index = j
                else:
                    break
            
            if triggered_index != -1:
                sell_price = grids[triggered_index]
                revenue = sell_price * trade_size * (1 - fee_rate)
                
                # 如果有足夠的倉位可以賣出 (確保是 Buy-Sell 循環)
                if current_position >= trade_size:
                    profit = revenue - (last_buy_price * trade_size * (1 + fee_rate))
                    total_profit += profit
                    current_position -= trade_size
                    completed_cycles += 1
                    
                    trade_log.append({
                        'Time_Index': i, 
                        'Price': current_price,
                        'Action': 'SELL (賣出)', 
                        'Amount': trade_size, 
                        'Grid_Price': sell_price,
                        'Profit': profit,
                        'Note': f"價格上穿網格線 {triggered_index}，完成循環"
                    })
                else:
                     # 僅記錄賣出行為，但不計為完整循環利潤 (因為是初始平倉或超出網格範圍的交易)
                    trade_log.append({
                        'Time_Index': i, 
                        'Price': current_price,
                        'Action': 'SELL (賣出)', 
                        'Amount': trade_size, 
                        'Grid_Price': sell_price,
                        'Profit': 0,
                        'Note': f"價格上穿網格線 {triggered_index}，無對應買入倉位"
                    })
                
                # 更新當前所在網格區間
                last_grid_index = triggered_index
    
    # 網格利潤計算: 這是網格策略的核心利潤，來自於 Buy Low / Sell High 的完成循環。
    average_grid_profit = total_profit / completed_cycles if completed_cycles > 0 else 0
    
    # 浮動盈虧 (最終倉位價值 - 最終成本)
    # 這裡的模擬過於簡化，為了實用性，我們只關注已實現的網格利潤 (total_profit)
    
    return total_profit, completed_cycles, average_grid_profit, trade_log

# --- 4. Streamlit 應用程式界面 ---

st.set_page_config(layout="wide", page_title="現貨網格機器人模擬推演")

st.title("💰 現貨網格機器人模擬推演 (Streamlit)")
st.caption("作者：Google Gemini | **注意: 本應用使用模擬價格數據進行回測**")

# --- 側邊欄輸入設定 ---
st.sidebar.header("📈 策略與參數設定")

# 選擇標的 (對模擬結果無影響，僅供展示)
asset = st.sidebar.selectbox(
    "選擇標的資產 (Asset)",
    ('BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'),
    index=0
)

st.sidebar.subheader("網格區間設定")
initial_price = 30000.0 if asset == 'BTC/USDT' else 2000.0

col_lower, col_upper = st.sidebar.columns(2)
lower_limit = col_lower.number_input("下限價格 (Lower Limit)", min_value=1.0, value=initial_price * 0.95, step=100.0)
upper_limit = col_upper.number_input("上限價格 (Upper Limit)", min_value=1.0, value=initial_price * 1.05, step=100.0)

num_grids = st.sidebar.slider("網格數量 (Grid Count)", min_value=5, max_value=50, value=20, step=1)
grid_type = st.sidebar.radio(
    "網格類型 (Grid Type)",
    ('等差網格 (Arithmetic)', '等比網格 (Geometric)'),
    horizontal=True
)

st.sidebar.subheader("交易參數")
trade_size = st.sidebar.number_input("單筆交易量 (Trade Size, 基礎資產)", min_value=0.001, value=0.01, step=0.001, format="%.3f", help="每次買入/賣出的基礎資產數量 (例如 0.01 BTC)")
fee_rate = st.sidebar.number_input("單邊手續費率 (Fee Rate, 例如 0.1%)", min_value=0.0, max_value=0.01, value=0.001, step=0.0001, format="%.4f", help="每筆交易的費率 (例如 0.001 代表 0.1%)")
num_steps = st.sidebar.slider("模擬價格點數 (Simulation Steps)", min_value=100, max_value=1000, value=500, step=50, help="模擬回測的價格數據點數量")

# --- 主要內容區塊 ---

if lower_limit >= upper_limit:
    st.error("❌ 錯誤：上限價格必須大於下限價格。請調整側邊欄的設定。")
else:
    # 1. 計算網格價格
    grids = calculate_grids(lower_limit, upper_limit, num_grids, grid_type)
    
    # 2. 準備價格數據
    # 根據用戶選擇的資產和區間，調整初始價格來生成模擬數據
    mock_initial_price = (lower_limit + upper_limit) / 2
    price_data = generate_mock_prices(initial_price=mock_initial_price, num_steps=num_steps)
    
    st.header("參數一覽")
    st.markdown(f"""
    - **標的資產**: `{asset}`
    - **價格區間**: {lower_limit} - {upper_limit}
    - **網格類型**: `{grid_type}`
    - **網格總數**: {num_grids}
    - **單筆交易量**: {trade_size}
    - **單邊手續費**: {fee_rate * 100:.2f}%
    """)
    
    st.subheader("網格線價格 (Grid Prices)")
    grid_df = pd.DataFrame({
        'Level': range(num_grids + 1),
        'Price': grids,
        'Range_Profit_Rate': [
            (grids[i+1] / grids[i] - 1) * 100 if i < num_grids else 0
            for i in range(num_grids + 1)
        ]
    }).iloc[:-1] # 最後一個點是上限，不需要計算網格利潤率
    
    st.dataframe(grid_df, use_container_width=True, hide_index=True)
    
    # 3. 執行回測
    st.header("🚀 回測結果分析")
    
    total_profit, completed_cycles, average_grid_profit, trade_log = run_backtest(
        price_data, grids, trade_size, fee_rate
    )
    
    # 網格利潤甜蜜點指標
    # 總網格利潤 / 網格完成次數 / 總模擬步數 / 區間資金佔用 (簡化)
    # 這裡的 Grid Profitability 是為了量化效率
    grid_profitability = (total_profit * 100) / (upper_limit * trade_size * num_grids) if upper_limit > 0 and trade_size > 0 and num_grids > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("總網格利潤 (USDT)", f"{total_profit:.2f}")
    col2.metric("完整套利循環次數", completed_cycles)
    col3.metric("平均每格利潤 (USDT)", f"{average_grid_profit:.4f}")
    col4.metric("網格套利效率指標 (%)", f"{grid_profitability:.4f}", help="總利潤相對於區間所需資金的百分比 (簡化指標)")

    # 4. 價格與網格圖表
    st.subheader("價格路徑與網格分佈")
    
    # 繪製價格曲線和網格線
    chart_df = pd.DataFrame({'Price': price_data.values})
    
    # 加入網格線作為參考線
    chart_data = [{'price': p, 'type': 'Grid Level'} for p in grids]
    chart_data.append({'price': lower_limit, 'type': 'Lower Limit'})
    chart_data.append({'price': upper_limit, 'type': 'Upper Limit'})
    
    import altair as alt
    
    # 價格線圖
    line_chart = alt.Chart(chart_df.reset_index()).mark_line(color='#10B981').encode(
        x=alt.X('index', title='時間步 (Time Step)'),
        y=alt.Y('Price', title=f'{asset} 價格 (Price)'),
        tooltip=['index', alt.Tooltip('Price', format='.2f')]
    ).properties(
        title=f'{asset} 模擬價格路徑與網格分佈'
    )
    
    # 網格線 (Reference Lines)
    grid_lines = alt.Chart(pd.DataFrame(chart_data)).mark_rule().encode(
        y='price',
        color=alt.Color('type', scale=alt.Scale(domain=['Lower Limit', 'Upper Limit', 'Grid Level'], range=['#EF4444', '#3B82F6', '#9CA3AF'])),
        tooltip=[alt.Tooltip('price', format='.2f'), 'type']
    )
    
    st.altair_chart(line_chart + grid_lines, use_container_width=True)

    # 5. 交易記錄
    st.subheader("交易記錄 (Trade Log)")
    if trade_log:
        log_df = pd.DataFrame(trade_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ 在當前價格路徑和網格設定下，沒有發生完整的套利循環交易。請檢查價格區間是否包含價格波動。")

    st.header("🎯 尋找甜蜜點：回測建議")
    st.markdown("""
    網格利潤的**甜蜜點**通常位於以下因素的平衡點：

    1.  **網格密度 (網格數)**：
        * **網格數越多 (密度高)**：單格利潤率低，但套利次數多，總利潤可能高 (適合震盪頻繁的市場)。
        * **網格數越少 (密度低)**：單格利潤率高，但套利次數少，容易錯過機會，且價格可能快速出區間。
    2.  **區間大小 (上下限)**：
        * **區間大**：持續時間長，但單格利潤率低。
        * **區間小**：單格利潤率高，但很容易被突破 (出區間)。
    3.  **網格類型 (等差/等比)**：
        * **等差網格**：適用於價格區間相對較小的幣種，或當您認為價格波動的絕對幅度相對穩定時。
        * **等比網格**：適用於高波動性資產，它確保價格越高，網格間距越大 (保持固定百分比的利潤率)，更適合長期持有和應對指數級增長的價格。

    **您的目標是找到能最大化「網格套利效率指標」的參數組合。**
    """)
