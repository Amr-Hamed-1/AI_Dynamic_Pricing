import os
import sys

import plotly.express as px
import streamlit as st

IMPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if IMPORT_DIR not in sys.path:
    sys.path.insert(0, IMPORT_DIR)

from optimization import PriceOptimizer

st.set_page_config(page_title="Dynamic Pricing AI", layout="wide")

st.title("🤖 Dynamic Pricing & Demand Forecasting")
st.markdown("### نظام ذكي لتحديد السعر المثالي بناءً على مرونة الطلب وأسعار المنافسين")


@st.cache_resource
def load_optimizer():
    root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root, "models", "best_model.pkl")
    data_path = os.path.join(root, "data", "processed", "processed_retail_price.csv")
    if os.path.exists(model_path) and os.path.exists(data_path):
        return PriceOptimizer(model_path, data_path)
    return None


optimizer = load_optimizer()

if optimizer is None:
    st.error("❌ ملفات الموديل أو الداتا غير موجودة! شغّل `generate_data.py` ثم `train.py` أولاً.")
    st.stop()

st.sidebar.header("⚙️ إعدادات المحاكاة")

product_id = st.sidebar.selectbox(
    "المنتج (Product)",
    options=optimizer.products,
    index=0,
)

default_comp = float(
    optimizer.df[optimizer.df["product_id"].astype(str) == product_id]["comp_1"].median()
)

competitor_price = st.sidebar.number_input(
    "سعر المنافس (Comp Price)",
    min_value=10.0,
    max_value=500.0,
    value=round(default_comp, 2),
    step=1.0,
)

cost = st.sidebar.number_input(
    "تكلفة المنتج (Product Cost)",
    min_value=0.0,
    max_value=500.0,
    value=round(default_comp * 0.4, 2),
    step=1.0,
)

month = st.sidebar.slider(
    "الشهر (Month / Seasonality)",
    min_value=1,
    max_value=12,
    value=6,
)

if st.sidebar.button("🚀 احسب السعر المثالي"):
    summary, all_results = optimizer.get_optimal_price(
        competitor_price, product_cost=cost, product_id=product_id, month=month
    )

    is_profit = summary["optimal_by"] == "profit"
    lift_key = f"{summary['optimal_by']}_pct"
    metric_label = "الربح" if is_profit else "الإيراد"
    best_metric_val = summary["best_profit"] if is_profit else summary["best_revenue"]
    baseline_metric_val = (
        summary["baseline_profit"] if is_profit else summary["baseline_revenue"]
    )

    # -----------------------------------------------------
    # 1. KPIs
    # -----------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 السعر المقترح (Best Price)",
            value=f"{summary['best_price']:.2f} EGP",
            delta=f"{summary['best_price'] - competitor_price:+.2f} vs Competitor",
        )

    with col2:
        st.metric(
            label="📦 المبيعات المتوقعة (Units)",
            value=f"{summary['best_qty']:.1f}",
            delta=f"{summary['lift']['qty_units']:+.1f} vs baseline",
            delta_color="normal",
        )

    with col3:
        st.metric(
            label=f"💵 {metric_label} المتوقع",
            value=f"{best_metric_val:.2f} EGP",
            delta=f"{summary['lift'][lift_key]:+.1f}% vs baseline",
            delta_color="normal",
        )

    with col4:
        st.metric(
            label="📊 السعر الحالي (Baseline)",
            value=f"{summary['current_price']:.2f} EGP",
            delta=f"{metric_label} حالي: {baseline_metric_val:.2f} EGP",
        )

    # -----------------------------------------------------
    # 2. Profit/Revenue Curve & Demand Elasticity Curve
    # -----------------------------------------------------
    st.divider()
    chart_col1, chart_col2 = st.columns(2)

    y_col = "profit" if is_profit else "revenue"
    with chart_col1:
        st.subheader(f"منحنى {metric_label} مقابل السعر")
        fig_profit = px.line(
            all_results,
            x="price",
            y=y_col,
            title=f"تغيّر {metric_label} المتوقع مع السعر",
            labels={"price": "السعر (Price)", y_col: metric_label},
        )
        # Correctly plot optimal profit/revenue on the matching y-axis scale
        fig_profit.add_scatter(
            x=[summary["best_price"]],
            y=[best_metric_val],
            mode="markers",
            marker=dict(color="red", size=14),
            name="السعر المثالي (Optimal)",
        )
        fig_profit.add_scatter(
            x=[summary["current_price"]],
            y=[baseline_metric_val],
            mode="markers",
            marker=dict(color="orange", size=11, symbol="diamond"),
            name="السعر الحالي (Baseline)",
        )
        st.plotly_chart(fig_profit, use_container_width=True)

    with chart_col2:
        st.subheader("منحنى مرونة الطلب (Demand Curve)")
        fig_demand = px.line(
            all_results,
            x="price",
            y="predicted_qty",
            title="الكمية المتوقعة عند كل نقطة سعرية (Monotonic Elasticity)",
            labels={"price": "السعر (Price)", "predicted_qty": "الكمية المتوقعة (Qty)"},
        )
        fig_demand.add_scatter(
            x=[summary["best_price"]],
            y=[summary["best_qty"]],
            mode="markers",
            marker=dict(color="red", size=14),
            name="عند السعر المثالي",
        )
        st.plotly_chart(fig_demand, use_container_width=True)

    # -----------------------------------------------------
    # 3. Strategy explanation
    # -----------------------------------------------------
    if summary["best_price"] < competitor_price:
        strategy = "Undercut — بيع أرخص من المنافس لزيادة الحصة السوقية والحجم"
        delta_sign = "أرخص"
    else:
        strategy = "Premium — بيع أغلى من المنافس لتعظيم هامش الربح للوحدة"
        delta_sign = "أغلى"

    st.info(
        f"**الاستراتيجية:** {strategy} — السعر المثالي {delta_sign} من المنافس "
        f"بمقدار {abs(summary['best_price'] - competitor_price):.2f} EGP."
    )

    with st.expander("عرض جدول المحاكاة الكامل (60 نقطة سعرية)"):
        st.dataframe(all_results, use_container_width=True)
else:
    st.info("👈 اختر منتجًا واضغط على الزر في القائمة الجانبية لبدء التحليل.")
