"""
strategy.py

Chiến lược giao dịch long-only được tách từ notebook `quant-entry-test(2).ipynb`.

Ý tưởng chính:
- Chỉ mua khi cổ phiếu đang ở regime tăng giá.
- Entry gồm 2 nhóm: breakout và pullback.
- Exit gồm: mất trend, thủng Donchian low, RSI yếu, ATR stop/trailing stop, time stop.
- `position` luôn là 0/1 để biểu diễn trạng thái nắm giữ.
- `signal` luôn là -1/0/1 để biểu diễn tín hiệu bán/không làm gì/mua.

Quy ước output quan trọng:
- position = 1: đang nắm giữ vị thế long.
- position = 0: đứng ngoài thị trường.
- signal = 1: tín hiệu mua.
- signal = -1: tín hiệu bán/thoát lệnh.
- signal = 0: không có hành động.
- shares_position: số cổ phiếu nắm giữ nếu bật dynamic position sizing.
"""

import numpy as np
import pandas as pd

from getpass import getpass

from quantvn import client
from quantvn.vn.data import get_stock_hist

# ============================================================
# 1. Indicator functions
# ============================================================


def calculate_rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Tính RSI theo phương pháp Wilder.

    RSI dùng để đo momentum:
    - RSI cao: lực mua mạnh hơn.
    - RSI thấp: lực bán mạnh hơn.
    - Trong strategy này, RSI được dùng để xác nhận pullback đã hồi lại
      và để exit khi momentum quá yếu.
    """
    # Tính thay đổi giá đóng cửa từng phiên.
    delta = close.diff()

    # Phần tăng giá: nếu delta âm thì đưa về 0.
    gain = delta.clip(lower=0)

    # Phần giảm giá: nếu delta dương thì đưa về 0, sau đó đổi dấu để thành số dương.
    loss = -delta.clip(upper=0)

    # Trung bình tăng theo Wilder, dùng EWM với alpha = 1 / period.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Trung bình giảm theo Wilder.
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Relative Strength = trung bình tăng / trung bình giảm.
    rs = avg_gain / avg_loss.replace(0, np.nan)

    # Công thức RSI chuẩn.
    rsi = 100 - (100 / (1 + rs))

    # Các phiên đầu chưa đủ dữ liệu được xem là trung tính RSI = 50.
    return rsi.fillna(50)


def calculate_atr(
    df: pd.DataFrame,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    period: int = 14,
) -> pd.Series:
    """
    Tính ATR - Average True Range.

    ATR đo độ biến động tuyệt đối của giá.
    Trong strategy này ATR được dùng cho:
    - volatility filter: tránh mua khi biến động quá cao.
    - initial stop-loss: stop ban đầu.
    - trailing stop: kéo stop lên khi trade đi đúng hướng.
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    # Close của phiên trước, dùng để tính gap qua đêm/ngày.
    prev_close = close.shift(1)

    # Biên độ trong ngày.
    range_high_low = high - low

    # Biên độ giữa high hôm nay và close hôm qua.
    range_high_prev_close = (high - prev_close).abs()

    # Biên độ giữa low hôm nay và close hôm qua.
    range_low_prev_close = (low - prev_close).abs()

    # True Range là giá trị lớn nhất trong 3 loại biên độ.
    true_range = pd.concat(
        [range_high_low, range_high_prev_close, range_low_prev_close],
        axis=1,
    ).max(axis=1)

    # ATR là trung bình động hàm mũ của True Range theo Wilder.
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return atr


def calculate_adx(
    df: pd.DataFrame,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    period: int = 14,
) -> pd.DataFrame:
    """
    Tính ADX, +DI và -DI.

    ADX dùng để đo sức mạnh xu hướng, không đo hướng tăng/giảm.
    +DI và -DI dùng để xác nhận hướng xu hướng:
    - +DI > -DI: lực tăng chiếm ưu thế.
    - -DI > +DI: lực giảm chiếm ưu thế.

    Strategy này yêu cầu:
    - ADX >= adx_min.
    - +DI > -DI.
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    # Mức tăng của high so với phiên trước.
    up_move = high.diff()

    # Mức giảm của low so với phiên trước, đổi dấu để thành số dương khi low giảm.
    down_move = -low.diff()

    # +DM chỉ được ghi nhận khi up_move lớn hơn down_move và dương.
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)

    # -DM chỉ được ghi nhận khi down_move lớn hơn up_move và dương.
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Tính lại True Range để làm mẫu số cho +DI/-DI.
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # ATR dùng trong công thức DI.
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # +DI: mức directional movement tăng chia cho ATR.
    plus_di = 100 * (
        pd.Series(plus_dm, index=df.index)
        .ewm(alpha=1 / period, min_periods=period, adjust=False)
        .mean()
        / atr.replace(0, np.nan)
    )

    # -DI: mức directional movement giảm chia cho ATR.
    minus_di = 100 * (
        pd.Series(minus_dm, index=df.index)
        .ewm(alpha=1 / period, min_periods=period, adjust=False)
        .mean()
        / atr.replace(0, np.nan)
    )

    # DX đo độ chênh lệch tương đối giữa +DI và -DI.
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    # ADX là trung bình động của DX.
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    result = pd.DataFrame(index=df.index)
    result["plus_di"] = plus_di
    result["minus_di"] = minus_di
    result["adx"] = adx

    return result


# ============================================================
# 2. Main strategy function
# ============================================================


def gen_position(
    df: pd.DataFrame,
    close_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    volume_col: str = "volume",
    use_shift: bool = True,
    # EMA regime parameters.
    ema_fast: int = 20,
    ema_mid: int = 50,
    ema_slow: int = 200,
    ema_slow_slope_lookback: int = 20,
    # Momentum parameters.
    rsi_period: int = 14,
    rsi_pullback_level: int = 45,
    rsi_exit: int = 35,
    adx_period: int = 14,
    adx_min: float = 15.0,
    # Donchian parameters.
    breakout_window: int = 35,
    exit_low_window: int = 20,
    # Volume parameters.
    volume_window: int = 20,
    breakout_volume_multiplier: float = 1.00,
    pullback_volume_multiplier: float = 0.60,
    # Volatility filter parameters.
    atr_period: int = 14,
    atr_quantile_window: int = 252,
    max_atr_quantile: float = 0.85,
    max_atr_pct_absolute: float = 0.085,
    # Avoid chasing parameters.
    max_extension_from_ema_fast: float = 0.15,
    max_extension_from_ema_mid: float = 0.25,
    # Pullback parameters.
    pullback_lookback: int = 7,
    pullback_tolerance: float = 0.025,
    # Momentum confirmation parameters.
    ret_20d_min_for_breakout: float = 0.00,
    ret_60d_min_for_regime: float = 0.00,
    # Stop-loss / trailing stop parameters.
    initial_atr_stop_mult: float = 2.5,
    trailing_atr_stop_mult: float = 3.0,
    hard_stop_pct: float = 0.07,
    # Time stop parameters.
    time_stop_days: int = 60,
    min_profit_after_time_stop: float = 0.00,
    # Optional position sizing parameters.
    capital: float = 100_000_000,
    risk_per_trade: float = 0.01,
    max_allocation_pct: float = 0.95,
    lot_size: int = 100,
    shares: int = 100,
    use_dynamic_position_sizing: bool = True,
    # Trade control.
    cooldown_days: int = 3,
) -> pd.DataFrame:
    """
    Tạo position và signal cho chiến lược long-only.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đầu vào. Cần có các cột OHLCV, mặc định:
        Date, Open, High, Low, Close, volume/Volume.

    use_shift : bool
        Nếu True, shift position/signal 1 phiên để tránh look-ahead bias.
        Tức là tín hiệu được tạo ở cuối phiên t sẽ được thực thi từ phiên t+1.

    Returns
    -------
    pd.DataFrame
        DataFrame gốc kèm thêm indicator, filter, position và signal.

    Output quan trọng:
    - position: 0/1, dùng cho return-based backtest.
    - signal: -1/0/1, dùng để biết ngày nào mua/bán.
    - signal_text: BUY/SELL/HOLD, dễ đọc hơn signal số.
    - shares_position: số cổ phiếu nếu dùng dynamic sizing.
    - entry_type: breakout/pullback/none.
    - exit_reason: lý do thoát lệnh.
    """
    # Copy dữ liệu để không làm thay đổi DataFrame gốc bên ngoài hàm.
    df = df.copy()

    # Gán các series chính để code phía dưới ngắn và dễ đọc hơn.
    close = df[close_col]
    high = df[high_col]
    low = df[low_col]
    volume = df[volume_col]

    # ========================================================
    # 2.1. Tính các indicator cần thiết
    # ========================================================

    # EMA nhanh dùng để đo xu hướng ngắn hạn.
    df["ema_fast"] = close.ewm(span=ema_fast, min_periods=ema_fast, adjust=False).mean()

    # EMA trung bình dùng để đo xu hướng trung hạn.
    df["ema_mid"] = close.ewm(span=ema_mid, min_periods=ema_mid, adjust=False).mean()

    # EMA chậm dùng để đo xu hướng dài hạn.
    df["ema_slow"] = close.ewm(span=ema_slow, min_periods=ema_slow, adjust=False).mean()

    # Alias giữ lại để tương thích với notebook cũ.
    df["ma20"] = df["ema_fast"]
    df["ma50"] = df["ema_mid"]
    df["ma200"] = df["ema_slow"]

    # Độ dốc EMA dài hạn: EMA200 hiện tại trừ EMA200 của 20 phiên trước.
    df["ema_slow_slope"] = df["ema_slow"].diff(ema_slow_slope_lookback)

    # RSI đo momentum.
    df["rsi"] = calculate_rsi_wilder(close, period=rsi_period)

    # ATR đo volatility và dùng cho stop-loss.
    df["atr"] = calculate_atr(
        df,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        period=atr_period,
    )

    # ATR theo phần trăm giá, giúp so sánh volatility theo tương đối.
    df["atr_pct"] = df["atr"] / close

    # Ngưỡng volatility động: quantile của ATR% trong 252 phiên gần nhất.
    df["atr_pct_threshold"] = (
        df["atr_pct"]
        .rolling(atr_quantile_window, min_periods=60)
        .quantile(max_atr_quantile)
    )

    # Volume trung bình 20 phiên, dùng để xác nhận breakout/pullback.
    df["volume_ma"] = volume.rolling(volume_window, min_periods=volume_window).mean()

    # Alias giữ lại để tương thích với notebook cũ.
    df["volume_ma20"] = df["volume_ma"]

    # Tính ADX, +DI, -DI.
    adx_df = calculate_adx(
        df,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        period=adx_period,
    )
    df["adx"] = adx_df["adx"]
    df["plus_di"] = adx_df["plus_di"]
    df["minus_di"] = adx_df["minus_di"]

    # Donchian high: đỉnh cao nhất của các phiên trước đó, dùng để nhận diện breakout.
    df["donchian_high"] = high.shift(1).rolling(
        breakout_window,
        min_periods=breakout_window,
    ).max()

    # Donchian low: đáy thấp nhất của các phiên trước đó, dùng làm một điều kiện exit.
    df["donchian_low"] = low.shift(1).rolling(
        exit_low_window,
        min_periods=exit_low_window,
    ).min()

    # Return 20 phiên, dùng xác nhận breakout không quá yếu.
    df["ret_20d"] = close / close.shift(20) - 1

    # Return 60 phiên, dùng xác nhận regime trung hạn đang không âm.
    df["ret_60d"] = close / close.shift(60) - 1

    # ========================================================
    # 2.2. Entry filters: chỉ mua trong môi trường thuận lợi
    # ========================================================

    # Regime tăng: giá trên EMA200, EMA50 trên EMA200, EMA200 dốc lên, return 60D dương.
    stock_regime_ok = (
        (close > df["ema_slow"])
        & (df["ema_mid"] > df["ema_slow"])
        & (df["ema_slow_slope"] > 0)
        & (df["ret_60d"] > ret_60d_min_for_regime)
    )

    # Xu hướng đủ mạnh: ADX vượt ngưỡng và +DI > -DI.
    trend_strength_ok = (df["adx"] >= adx_min) & (df["plus_di"] > df["minus_di"])

    # Nếu chưa đủ dữ liệu rolling quantile, dùng ngưỡng tuyệt đối max_atr_pct_absolute.
    atr_threshold = df["atr_pct_threshold"].fillna(max_atr_pct_absolute)

    # Tránh mua khi volatility quá cao.
    volatility_ok = (df["atr_pct"] <= atr_threshold) & (
        df["atr_pct"] <= max_atr_pct_absolute
    )

    # Tránh mua khi giá đã đi quá xa so với EMA20/EMA50.
    not_too_extended = (
        ((close / df["ema_fast"] - 1) <= max_extension_from_ema_fast)
        & ((close / df["ema_mid"] - 1) <= max_extension_from_ema_mid)
    )

    # Bộ lọc nền trước khi xét breakout hoặc pullback.
    base_entry_filter = stock_regime_ok & trend_strength_ok & volatility_ok & not_too_extended

    # ========================================================
    # 2.3. Entry setup 1: breakout
    # ========================================================

    # Breakout cần volume lớn hơn volume trung bình.
    breakout_volume_ok = volume > df["volume_ma"] * breakout_volume_multiplier

    # Breakout: close vượt Donchian high và momentum 20D không quá yếu.
    breakout_entry = (
        (close > df["donchian_high"])
        & breakout_volume_ok
        & (df["ret_20d"] > ret_20d_min_for_breakout)
    )

    # ========================================================
    # 2.4. Entry setup 2: pullback
    # ========================================================

    # Giá thấp nhất chạm gần EMA20.
    touched_ema_fast = low <= df["ema_fast"] * (1 + pullback_tolerance)

    # Giá thấp nhất chạm gần EMA50.
    touched_ema_mid = low <= df["ema_mid"] * (1 + pullback_tolerance)

    # Trong một số phiên gần đây có chạm vùng EMA20/EMA50.
    touched_pullback_zone_recently = (
        (touched_ema_fast | touched_ema_mid)
        .rolling(pullback_lookback, min_periods=1)
        .max()
        .astype(bool)
    )

    # RSI hồi phục: RSI trên ngưỡng pullback và tăng so với phiên trước.
    rsi_recovery = (df["rsi"] > rsi_pullback_level) & (df["rsi"] > df["rsi"].shift(1))

    # Giá đóng cửa hồi lại trên EMA20.
    recovered_above_ema_fast = close > df["ema_fast"]

    # Pullback cần volume không quá yếu.
    pullback_volume_ok = volume > df["volume_ma"] * pullback_volume_multiplier

    # Điều kiện pullback hoàn chỉnh.
    pullback_entry = (
        touched_pullback_zone_recently
        & recovered_above_ema_fast
        & rsi_recovery
        & pullback_volume_ok
    )

    # Tín hiệu entry candidate trước khi xét trạng thái vị thế/cooldown.
    df["entry_signal"] = base_entry_filter & (breakout_entry | pullback_entry)

    # Lưu loại entry candidate để debug.
    candidate_entry_type = pd.Series("none", index=df.index, dtype="object")
    candidate_entry_type.loc[base_entry_filter & breakout_entry] = "breakout"
    candidate_entry_type.loc[base_entry_filter & pullback_entry] = "pullback"
    candidate_entry_type.loc[base_entry_filter & breakout_entry & pullback_entry] = "breakout"
    df["candidate_entry_type"] = candidate_entry_type

    # ========================================================
    # 2.5. Exit signals: các điều kiện thoát lệnh cơ bản
    # ========================================================

    # Giá đóng cửa dưới EMA50 trong 2 phiên liên tiếp.
    close_below_ema_mid_2d = (close < df["ema_mid"]) & (
        close.shift(1) < df["ema_mid"].shift(1)
    )

    # Giá thủng EMA200, tức trend dài hạn suy yếu.
    slow_trend_exit = close < df["ema_slow"]

    # Giá thủng đáy Donchian low.
    donchian_exit = close < df["donchian_low"]

    # RSI quá yếu.
    momentum_exit = df["rsi"] < rsi_exit

    # Tín hiệu exit cơ bản, chưa gồm stop-loss/trailing/time stop.
    df["base_exit_signal"] = (
        close_below_ema_mid_2d | slow_trend_exit | donchian_exit | momentum_exit
    )

    # ========================================================
    # 2.6. Stateful position management
    # ========================================================

    # raw_position là trạng thái vị thế trước khi shift: 0 = đứng ngoài, 1 = đang long.
    raw_position = np.zeros(len(df), dtype=int)

    # raw_shares lưu số cổ phiếu nếu dùng dynamic sizing.
    raw_shares = np.zeros(len(df), dtype=int)

    # Các mảng debug để biết trade vào giá nào, stop ở đâu, giữ bao lâu.
    entry_price_arr = np.full(len(df), np.nan)
    initial_stop_arr = np.full(len(df), np.nan)
    trailing_stop_arr = np.full(len(df), np.nan)
    active_stop_arr = np.full(len(df), np.nan)
    bars_in_trade_arr = np.full(len(df), np.nan)
    trade_id_arr = np.full(len(df), np.nan)

    # Mảng ghi lại entry type và exit reason thực tế sau khi xét position/cooldown.
    raw_entry_type_arr = np.array(["none"] * len(df), dtype=object)
    raw_exit_reason_arr = np.array(["none"] * len(df), dtype=object)

    # Trạng thái nội bộ của vòng lặp.
    in_position = False
    shares_held = 0
    entry_price = np.nan
    initial_stop = np.nan
    highest_close = np.nan
    bars_in_trade = 0
    cooldown = 0
    trade_id = 0

    # Các dòng chưa đủ indicator thì không được trade.
    indicator_not_ready = (
        df["ema_fast"].isna()
        | df["ema_mid"].isna()
        | df["ema_slow"].isna()
        | df["ema_slow_slope"].isna()
        | df["rsi"].isna()
        | df["atr"].isna()
        | df["atr_pct"].isna()
        | df["volume_ma"].isna()
        | df["adx"].isna()
        | df["plus_di"].isna()
        | df["minus_di"].isna()
        | df["donchian_high"].isna()
        | df["donchian_low"].isna()
        | df["ret_20d"].isna()
        | df["ret_60d"].isna()
    )

    # Duyệt từng phiên để mô phỏng trạng thái nắm giữ theo thời gian.
    for i in range(len(df)):
        # Giá đóng cửa hiện tại.
        price = close.iloc[i]

        # ATR hiện tại.
        atr_value = df["atr"].iloc[i]

        # Nếu indicator chưa đủ dữ liệu thì đứng ngoài.
        if indicator_not_ready.iloc[i]:
            raw_position[i] = 0
            raw_shares[i] = 0
            continue

        # Giảm cooldown sau mỗi phiên.
        if cooldown > 0:
            cooldown -= 1

        # ----------------------------------------------------
        # Trường hợp 1: hiện chưa có vị thế
        # ----------------------------------------------------
        if not in_position:
            # Chỉ được vào lệnh khi không còn cooldown và có entry signal.
            if cooldown == 0 and bool(df["entry_signal"].iloc[i]):
                # Stop theo ATR.
                atr_stop = price - initial_atr_stop_mult * atr_value

                # Hard stop theo phần trăm tối đa từ entry.
                hard_stop = price * (1 - hard_stop_pct)

                # Chọn stop cao hơn để không đặt stop quá xa.
                initial_stop = max(atr_stop, hard_stop)

                # Khoảng cách từ entry đến stop.
                stop_distance = price - initial_stop

                # Nếu stop_distance không hợp lệ thì bỏ qua entry.
                if stop_distance <= 0 or np.isnan(stop_distance):
                    raw_position[i] = 0
                    raw_shares[i] = 0
                    continue

                # Nếu bật dynamic sizing, tính số cổ phiếu theo risk/capital.
                if use_dynamic_position_sizing:
                    # Số tiền rủi ro tối đa cho mỗi trade.
                    risk_amount = capital * risk_per_trade

                    # Số cổ phiếu tối đa theo khoảng cách stop-loss.
                    shares_by_risk = risk_amount / stop_distance

                    # Số cổ phiếu tối đa theo vốn được phép phân bổ.
                    shares_by_capital = capital * max_allocation_pct / price

                    # Lấy số nhỏ hơn để không vượt risk hoặc vốn.
                    target_shares = min(shares_by_risk, shares_by_capital)

                    # Làm tròn xuống theo lot size.
                    shares_held = int(target_shares // lot_size * lot_size)
                else:
                    # Nếu không dùng dynamic sizing, dùng số cổ phiếu cố định.
                    shares_held = int(shares)

                # Nếu số cổ phiếu <= 0 thì không vào lệnh.
                if shares_held <= 0:
                    raw_position[i] = 0
                    raw_shares[i] = 0
                    continue

                # Cập nhật trạng thái sang đang nắm giữ.
                in_position = True
                trade_id += 1
                entry_price = price
                highest_close = price
                bars_in_trade = 0

                # Stop kéo theo ban đầu.
                trailing_stop = price - trailing_atr_stop_mult * atr_value

                # Active stop là stop cao hơn giữa initial stop và trailing stop.
                active_stop = max(initial_stop, trailing_stop)

                # Ghi nhận position và shares.
                raw_position[i] = 1
                raw_shares[i] = shares_held

                # Ghi nhận entry type thực tế.
                raw_entry_type_arr[i] = str(candidate_entry_type.iloc[i])

                # Lưu thông tin debug.
                entry_price_arr[i] = entry_price
                initial_stop_arr[i] = initial_stop
                trailing_stop_arr[i] = trailing_stop
                active_stop_arr[i] = active_stop
                bars_in_trade_arr[i] = bars_in_trade
                trade_id_arr[i] = trade_id
            else:
                # Không có vị thế và không có entry signal thì đứng ngoài.
                raw_position[i] = 0
                raw_shares[i] = 0

            continue

        # ----------------------------------------------------
        # Trường hợp 2: đang có vị thế
        # ----------------------------------------------------

        # Tăng số phiên đã giữ lệnh.
        bars_in_trade += 1

        # Cập nhật close cao nhất kể từ khi vào lệnh.
        highest_close = max(highest_close, price)

        # Tính trailing stop mới.
        trailing_stop = highest_close - trailing_atr_stop_mult * atr_value

        # Stop chỉ được giữ nguyên hoặc kéo lên, không kéo xuống.
        active_stop = max(initial_stop, trailing_stop)

        # Kiểm tra chạm stop.
        stop_hit = price <= active_stop

        # Time stop: nếu giữ quá lâu mà chưa đạt lợi nhuận tối thiểu thì thoát.
        time_stop = (bars_in_trade >= time_stop_days) and (
            price < entry_price * (1 + min_profit_after_time_stop)
        )

        # Xác định lý do exit để debug rõ ràng hơn.
        exit_reason = "none"
        if stop_hit:
            exit_reason = "stop_hit"
        elif time_stop:
            exit_reason = "time_stop"
        elif bool(close_below_ema_mid_2d.iloc[i]):
            exit_reason = "close_below_ema_mid_2d"
        elif bool(slow_trend_exit.iloc[i]):
            exit_reason = "close_below_ema_slow"
        elif bool(donchian_exit.iloc[i]):
            exit_reason = "donchian_low_break"
        elif bool(momentum_exit.iloc[i]):
            exit_reason = "rsi_exit"

        # Điều kiện exit tổng hợp.
        exit_now = bool(df["base_exit_signal"].iloc[i]) or stop_hit or time_stop

        if exit_now:
            # Thoát lệnh: raw_position về 0.
            raw_position[i] = 0
            raw_shares[i] = 0
            raw_exit_reason_arr[i] = exit_reason

            # Lưu thông tin debug tại ngày thoát.
            entry_price_arr[i] = entry_price
            initial_stop_arr[i] = initial_stop
            trailing_stop_arr[i] = trailing_stop
            active_stop_arr[i] = active_stop
            bars_in_trade_arr[i] = bars_in_trade
            trade_id_arr[i] = trade_id

            # Reset trạng thái nội bộ.
            in_position = False
            shares_held = 0
            entry_price = np.nan
            initial_stop = np.nan
            highest_close = np.nan
            bars_in_trade = 0

            # Sau khi thoát, nghỉ một số phiên trước khi được vào lại.
            cooldown = cooldown_days
        else:
            # Tiếp tục giữ vị thế.
            raw_position[i] = 1
            raw_shares[i] = shares_held

            # Lưu thông tin debug khi đang giữ lệnh.
            entry_price_arr[i] = entry_price
            initial_stop_arr[i] = initial_stop
            trailing_stop_arr[i] = trailing_stop
            active_stop_arr[i] = active_stop
            bars_in_trade_arr[i] = bars_in_trade
            trade_id_arr[i] = trade_id

    # ========================================================
    # 2.7. Tạo output position/signal rõ ràng
    # ========================================================

    # Lưu raw_position trước khi shift để debug.
    df["raw_position"] = raw_position
    df["raw_shares"] = raw_shares
    df["raw_entry_type"] = raw_entry_type_arr
    df["raw_exit_reason"] = raw_exit_reason_arr

    # Nếu use_shift=True, tín hiệu được thực thi ở phiên kế tiếp để tránh look-ahead bias.
    if use_shift:
        df["position"] = df["raw_position"].shift(1).fillna(0).astype(int)
        df["shares_position"] = df["raw_shares"].shift(1).fillna(0).astype(int)
        df["entry_type"] = pd.Series(raw_entry_type_arr, index=df.index).shift(1).fillna("none")
        df["exit_reason"] = pd.Series(raw_exit_reason_arr, index=df.index).shift(1).fillna("none")
    else:
        df["position"] = df["raw_position"].astype(int)
        df["shares_position"] = df["raw_shares"].astype(int)
        df["entry_type"] = raw_entry_type_arr
        df["exit_reason"] = raw_exit_reason_arr

    # position_change giúp nhận diện ngày position thay đổi.
    df["position_change"] = df["position"].diff().fillna(0)

    # signal số: 1 = mua, -1 = bán, 0 = giữ nguyên/không làm gì.
    df["signal"] = 0
    df.loc[df["position_change"] > 0, "signal"] = 1
    df.loc[df["position_change"] < 0, "signal"] = -1
    df["signal"] = df["signal"].astype(int)

    # signal_text dễ đọc hơn khi kiểm tra bảng kết quả.
    df["signal_text"] = "HOLD"
    df.loc[df["signal"] == 1, "signal_text"] = "BUY"
    df.loc[df["signal"] == -1, "signal_text"] = "SELL"

    # ========================================================
    # 2.8. Lưu các cột debug/filter
    # ========================================================

    # Các cột giá trị stop và trạng thái trade.
    df["entry_price"] = entry_price_arr
    df["initial_stop"] = initial_stop_arr
    df["trailing_stop"] = trailing_stop_arr
    df["active_stop"] = active_stop_arr
    df["bars_in_trade"] = bars_in_trade_arr
    df["trade_id"] = trade_id_arr

    # Các filter chính để debug vì sao có/không có entry.
    df["stock_regime_ok"] = stock_regime_ok
    df["trend_strength_ok"] = trend_strength_ok
    df["volatility_ok"] = volatility_ok
    df["not_too_extended"] = not_too_extended
    df["breakout_entry"] = breakout_entry
    df["pullback_entry"] = pullback_entry

    # Đảm bảo cột position/signal đúng format kỳ vọng.
    if not set(df["position"].dropna().unique()).issubset({0, 1}):
        raise ValueError("position phải chỉ gồm 0 và 1.")

    if not set(df["signal"].dropna().unique()).issubset({-1, 0, 1}):
        raise ValueError("signal phải chỉ gồm -1, 0 và 1.")

    return df

def main():
    # Nhập API key khi chạy file, tránh hard-code API key trực tiếp trong code
    api_key = getpass("Enter QuantVN API key: ")

    # Kết nối QuantVN client
    client(apikey=api_key)

    # Chọn mã cổ phiếu muốn chạy chiến lược
    symbol = "FPT"

    # Lấy dữ liệu lịch sử ngày của cổ phiếu FPT
    df = get_stock_hist(symbol, resolution="1D")

    # Sắp xếp dữ liệu theo thời gian tăng dần để indicator và position tính đúng
    df = df.sort_values("Date").reset_index(drop=True)

    # Chạy chiến lược để tạo position và signal
    df_strategy = gen_position(
        df,
        close_col="Close",
        high_col="High",
        low_col="Low",
        volume_col="volume",
        use_shift=True,
        use_dynamic_position_sizing=False,
        shares=100,
    )

if __name__ == "__main__":
    main()