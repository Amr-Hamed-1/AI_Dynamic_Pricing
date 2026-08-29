import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

IMPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if IMPORT_DIR not in sys.path:
    sys.path.insert(0, IMPORT_DIR)

from optimization import PriceOptimizer

st.set_page_config(page_title="Dynamic Pricing AI", layout="wide")

st.title("🤖 Dynamic Pricing & Demand Forecasting")
st.markdown("### نظام ذكي لتحديد السعر المثالي بناءً على المنافسين")


@st.cache_resource
def load_optimizer():
    root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root, "models", "best_model.pkl")
    data_path = os.path.join(root, "data", "processed",
                             "processed_retail_price.csv")
    if os.path.exists(model_path) and os.path.exists(data_path):
        return PriceOptimizer(model_path, data_path)
    return None


optimizer = load_optimizer()

if optimizer is None:
    st.error("❌ ملفات الموديل أو الداتا مش موجودة! تأكد من المسارات.")
    st.stop()

st.sidebar.header("⚙️ إعدادات المحاكاة")

# Pick the product we optimise for
product_id = st.sidebar.selectbox(
    "المنتج (Product)",
    options=optimizer.products,
    index=0,
)

# Default competitor price = that product's typical comp_1
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
    value=50.0,
    step=1.0,
)

if st.sidebar.button("🚀 احسب السعر المثالي"):

    summary, all_results = optimizer.get_optimal_price(
        competitor_price, product_cost=cost, product_id=product_id
    )

    lift_key = f"{summary['optimal_by']}_pct"
    metric_label = "الربح" if summary["optimal_by"] == "profit" else "الإيراد"

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
            value=f"{summary['best_revenue'] if summary['optimal_by'] == 'revenue' else summary['best_profit']:.2f} EGP",
            delta=f"{summary['lift'][lift_key]:+.1f}% vs baseline",
            delta_color="normal",
        )

    with col4:
        st.metric(
            label="📊 السعر الحالي (Baseline)",
            value=f"{summary['current_price']:.2f} EGP",
            delta=f"إيراد {summary['baseline_revenue']:.2f} EGP",
        )

    # -----------------------------------------------------
    # 2. Revenue / Profit curve
    # -----------------------------------------------------
    st.divider()
    st.subheader("تحليل منحنى الإيرادات (Revenue Curve)")

    y_col = "profit" if summary["optimal_by"] == "profit" else "revenue"
    fig = px.line(
        all_results,
        x="price",
        y=y_col,
        title=f"تغيير {metric_label} مع تغيير السعر",
    )
    fig.add_scatter(
        x=[summary["best_price"]],
        y=[summary["best_price"] * summary["best_qty"]],
        mode="markers",
        marker=dict(color="red", size=15),
        name="السعر المثالي",
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # 3. Strategy explanation
    # -----------------------------------------------------
    if summary["best_price"] < competitor_price:
        strategy = "Undercut — بيع أرخص من المنافس لزيادة الحصة السوقية والحجم"
        delta_sign = "أرخص"
    else:
        strategy = "Premium — بيع أغلى من المنافس بناءً على الجودة والتميز"
        delta_sign = "أغلى"

    st.info(
        f"**الاستراتيجية:** {strategy} — السعر المثالي {delta_sign} من المنافس "
        f"بمقدار {abs(summary['best_price'] - competitor_price):.2f} EGP."
    )

    with st.expander("عرض تفاصيل الأرقام (Data Table)"):
        st.dataframe(all_results)

else:
    st.info("👈 اختر منتجًا واضغط على الزر في القائمة الجانبية لبدء التحليل.")
