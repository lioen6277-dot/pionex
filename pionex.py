import streamlit as st
import numpy as np
import pandas as pd

# --- 1. 價格數據模擬 ---
# 由於無法存取外部API，我們模擬一個有趨勢和噪音的價格數據
def generate_mock_prices(initial_price=30000, num_steps=500):
    """模擬一個價格路徑 (帶有輕微向上趨勢和波動)"""
    # 確保每次運行結果一致，但使用不同的種子來確保資產間的價格略有不同
    # 這裡只使用一個固定種子確保單次回測穩定性
    np.random.seed(42) 
    
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
    
    # 網格數 N 實際上產生 N+1 個價格點
    if grid_type == '等差網格 (Arithmetic)':
        # 等差網格: 價格間距相等
        grids = np.linspace(lower_limit, upper_limit, num_grids + 1)
    
    elif grid_type == '等比網格 (Geometric)':
        # 等比網格: 價格比例相等 (log空間均分)
        grids = np.geomspace(lower_limit, upper_limit, num_grids + 1)
    
    # 確保網格點是有序的 (從低到高)
    grids.sort()
    # 轉換為 Python list 並四捨五入到小數點後兩位
    return [round(float(p), 2) for p in grids]

# --- 3. 回測模擬器 ---

def run_backtest(price_data, grids, trade_size=0.01, fee_rate=0.001):
    """
    執行網格回測模擬。
    (邏輯與前一版本相同，核心是追蹤 Buy Low / Sell High 的循環)
    """
    
    num_levels = len(grids)
    if num_levels < 2:
        return 0, 0, 0, []

    total_profit = 0
    completed_cycles = 0
    current_position = 0
    last_buy_price = 0
    
    # 追蹤當前價格所處的網格區間 (索引從 0 到 num_levels - 1)
    # last_grid_index 儲存上次觸發交易的網格線索引
    last_grid_index = next((i for i, p in enumerate(grids) if p >= price_data.iloc[0]), num_levels - 1)
    
    trade_log = []

    # 模擬價格變動
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
                # cost = buy_price * trade_size * (1 + fee_rate) # 實際成本計算 (用於追蹤)
                
                current_position += trade_size
                last_buy_price = buy_price
                
                trade_log.append({
                    'Time_Index': i, 'Price': current_price,
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
                    # 計算利潤: 賣出收入 - 買入成本 (含手續費)
                    profit = revenue - (last_buy_price * trade_size * (1 + fee_rate))
                    total_profit += profit
                    current_position -= trade_size
                    completed_cycles += 1
                    
                    trade_log.append({
                        'Time_Index': i, 'Price': current_price,
                        'Action': 'SELL (賣出)', 'Amount': trade_size, 
                        'Grid_Price': sell_price, 'Profit': profit,
                        'Note': f"價格上穿網格線 {triggered_index}，完成循環"
                    })
                else:
                    trade_log.append({
                        'Time_Index': i, 'Price': current_price,
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
st.caption("作者：Google Gemini | **注意: 本應用使用模擬價格數據進行回測**")

# --- 側邊欄輸入設定 ---
st.sidebar.header("📈 策略與參數設定 (派網風格)")

# 選擇標的 (對模擬結果無影響，僅供展示)
asset = st.sidebar.selectbox(
    "選擇標的資產 (Asset)",
    ('BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'),
    index=0
)

st.sidebar.subheader("網格區間設定")
# 根據選定的資產給定合理的起始價格作為預設值
if 'BTC' in asset:
    mock_start_price = 60000.0 
elif 'ETH' in asset:
    mock_start_price = 3000.0
else:
    mock_start_price = 150.0

col_lower, col_upper = st.sidebar.columns(2)
lower_limit = col_lower.number_input("下限價格 (Lower Limit)", min_value=1.0, value=mock_start_price * 0.9, step=10.0)
upper_limit = col_upper.number_input("上限價格 (Upper Limit)", min_value=1.0, value=mock_start_price * 1.1, step=10.0)

num_grids = st.sidebar.slider("網格數量 (Grid Count)", min_value=5, max_value=100, value=50, step=1)
grid_type = st.sidebar.radio(
    "網格類型 (Grid Type)",
    ('等差網格 (Arithmetic)', '等比網格 (Geometric)'),
    horizontal=True
)

st.sidebar.subheader("交易參數")
trade_size = st.sidebar.number_input("單筆交易量 (Trade Size, 基礎資產)", min_value=0.0001, value=0.01, step=0.0001, format="%.4f", help="每次買入/賣出的基礎資產數量 (例如 0.01 BTC)")
fee_rate = st.sidebar.number_input("單邊手續費率 (Fee Rate, 例如 0.1%)", min_value=0.0, max_value=0.01, value=0.001, step=0.0001, format="%.4f", help="每筆交易的費率 (例如 0.001 代表 0.1%)")
num_steps = st.sidebar.slider("模擬價格點數 (Simulation Steps)", min_value=100, max_value=2000, value=1000, step=100, help="模擬回測的價格數據點數量")


# --- 主要內容區塊 ---

if lower_limit >= upper_limit:
    st.error("❌ 錯誤：上限價格必須大於下限價格。請調整側邊欄的設定。")
else:
    # 1. 計算網格價格
    grids = calculate_grids(lower_limit, upper_limit, num_grids, grid_type)
    
    # 計算網格利潤率 (單邊，未扣除手續費)
    grid_profit_rates = [
        (grids[i+1] / grids[i] - 1) * 100 
        for i in range(len(grids) - 1)
    ]
    
    min_profit_rate = min(grid_profit_rates) if grid_profit_rates else 0
    avg_profit_rate = sum(grid_profit_rates) / len(grid_profit_rates) if grid_profit_rates else 0

    # 估算所需資金 (簡化計算，以中間價位和網格總數為基礎)
    mid_price = (lower_limit + upper_limit) / 2
    # 假設初始倉位中性，需要一半的網格作為 USDT 儲備，一半的網格作為基礎資產儲備
    # 為了保守，我們估算全部網格的成本 (這是一個高估值，但安全)
    # 最低資金 = 網格數 * 交易量 * 最低價 (極度保守)
    estimated_min_capital = num_grids * trade_size * lower_limit
    
    
    # 2. 準備價格數據
    # 以區間中點作為模擬價格的起點
    mock_initial_price = mid_price
    price_data = generate_mock_prices(initial_price=mock_initial_price, num_steps=num_steps)
    
    # 3. 執行回測
    total_profit, completed_cycles, average_grid_profit, trade_log = run_backtest(
        price_data, grids, trade_size, fee_rate
    )
    
    # 網格利潤甜蜜點指標 (效率指標)
    # Grid Profitability = 總網格利潤 / 估算所需最低資金 (類似 ROI)
    grid_profitability = (total_profit / estimated_min_capital) * 100 if estimated_min_capital > 0 else 0
    
    
    # --- 指標卡片顯示 (Pionex Style) ---
    st.header("🎯 策略回測表現")

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

    st.subheader("網格線價格與分佈")
    
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
    
    chart_df = pd.DataFrame({'Price': price_data.values})
    
    chart_data = [{'price': p, 'type': 'Grid Level'} for p in grids]
    chart_data.append({'price': lower_limit, 'type': 'Lower Limit'})
    chart_data.append({'price': upper_limit, 'type': 'Upper Limit'})
    
    import altair as alt
    
    line_chart = alt.Chart(chart_df.reset_index()).mark_line(color='#10B981', size=1).encode(
        x=alt.X('index', title='時間步 (Time Step)'),
        y=alt.Y('Price', title=f'{asset} 價格 (Price)'),
        tooltip=['index', alt.Tooltip('Price', format=',.2f')]
    ).properties(
        title=f'{asset} 模擬價格路徑與網格分佈'
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
        st.info("ℹ️ 在當前價格路徑和網格設定下，沒有發生完整的套利循環交易。請檢查價格區間是否包含價格波動。")

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
