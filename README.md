# 🤖 Dynamic Pricing & Demand Forecasting

نظام **end-to-end** لتحديد السعر الأمثل للمنتج تلقائيًا بناءً على سعر المنافسين،
يستخدم **توقّع الطلب (Demand Forecasting)** بالتعلم الآلي ثم محرّك قرار بسيط
يختار السعر الذي يحقق **أعلى ربح** (أو إيراد) مع قياس **الأثر (Lift)** مقابل
السعر الحالي.

---

## 📌 الفكرة (القيمة التجارية)

في التجارة الإلكترونية، السعر مش ثابت — السعر "الصحيح" يعتمد على:
- كم سعر المنافسين.
- حجم الطلب المتوقع عند كل سعر (مرونة السعر / Price Elasticity).

المشروع يحل ده في مرحلتين:
1. **Model**: يتنبأ بعدد الوحدات (`qty`) المتوقع بيعها لوضع سعر معيّن في سياق معيّن.
2. **Optimizer**: يمسح مجموعة أسعار حول سعر المنافس، يتنبأ بالطلب لكل سعر،
   يحسب `revenue = qty × price` و `profit = (price − cost) × qty`، ويختار
   الأفضل، ويقارنه بـ baseline (الحفاظ على السعر الحالي) لإظهار الـ **Lift**.

---

## 🗂️ بنية المشروع

```
Dynamic_Pricing_Project/
├── app.py                  # Streamlit dashboard (اختيار المنتج + المحاكاة)
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                # retail_price.csv (خام)
│   └── processed/          # processed_retail_price.csv (جاهز للتدريب)
├── src/
│   ├── generate_data.py    # توليد داتا تركيبية أكبر بعلاقة سعر-طلب صريحة
│   ├── preprocessing.py    # تنظيف + feature engineering
│   ├── train.py            # تدريب + تقييم + اختيار نماذج + feature importance
│   ├── optimization.py     # PriceOptimizer (محرّك السعر + الـ Lift)
│   └── data_loader.py
├── models/
│   ├── best_model.pkl      # أفضل موديل (XGBoost)
│   └── training_report.json # مقاييس + feature importance
└── notebooks/
    └── Cleaning&Eda.ipynb
```

> `data/` و `models/` مستثناه من git (بيانات/موديلات كبيرة). تُنشأ محليًا عبر
> تشغيل `generate_data.py` ثم `train.py`.

---

## 🚀 التشغيل

```bash
# 1. تثبيت المتطلبات
pip install -r requirements.txt

# 2. توليد الداتا (وإنشاء data/processed)
python src/generate_data.py

# 3. التدريب والتقييم
python src/train.py

# 4. تشغيل المحرّك من سطر الأوامر (يعرض Lift)
python src/optimization.py

# 5. لوحة التحكم التفاعلية
streamlit run app.py
```

---

## 🧠 المنهجية (Modeling)

- **التقسيم**: تقسيم **زمني** (أول 80% من الأشهر = train، آخر 20% = test) —
  يحاكي الاستخدام الحقيقي (توقّع المستقبل) بدلًا من التقسيم العشوائي.
- **النماذج**: `RandomForestRegressor` و `XGBoost` مع **GridSearchCV** (5-fold)
  لضبط الـ hyperparameters.
- **المقياس**: RMSE (لأنه يعاقب الأخطاء الكبيرة) + MAE + R².
- **المتغيرات المهمة**: `volume`, `customers`, و `price_diff_*` (الفرق بين سعرنا
  والمنافسين) هي أهم العوامل — وهذا منطقي لأنه جوهر التسعير الديناميكي.

### 📊 النتائج (على داتا تركيبية 4000 صف)

| Model      | MAE    | RMSE   | R²    |
|------------|--------|--------|-------|
| RandomForest | 5.16  | 7.42   | 0.50  |
| **XGBoost**  | **4.79** | **6.85** | **0.58** |

الموديل الأفضل: **XGBoost** → يُحفظ في `models/best_model.pkl`.

---

## 💡 مثال على الإخراج

```
Product       : prod_0000
Competitor    : 145.94$
Product cost  : 50.00$

✅ Recommended Price : 118.48$
📦 Expected Sales    : 26.43 units
💵 Expected Profit   : 1810.13$

📊 Baseline (keep current price 131.67$):
   qty=20.57, revenue=2708.77$
🚀 Profit lift vs baseline : +7.7%
   Quantity change         : +5.9 units
💡 Strategy: Undercut (أرخص من المنافس)
```

بمعنى: لو خفّضنا السعر من 131.67 إلى 118.48 (أرخص قليلًا من المنافس)، الطلب هيرتفع
لدرجة إن الربح الإجمالي يزيد **+7.7%** مقارنة بترك السعر كما هو.

---

## 🖥️ لوحة التحكم (Streamlit)

- اختيار **المنتج** من القائمة.
- تعديل **سعر المنافس** و **تكلفة المنتج**.
- عرض: السعر المقترح، الوحدات المتوقعة، الربح المتوقع، السعر الحالي (baseline).
- **منحنى الإيراد/الربح** مقابل السعر، وشرح الاستراتيجية (Undercut / Premium).

---

## 🧾 التقارير

عند كل تدريب، `train.py` يكتب `models/training_report.json` يحوي:
- مقاييس كل نموذج (MAE, RMSE, R²).
- أهم 20 متغير حسب `feature_importance`.

---

## ⚠️ ملاحظات صادقة

- البيانات **تركيبية (synthetic)** حاليًا — أُعيد توليدها ليصبح المشروع قابلًا
  للتكرار ومشروطًا بعلاقة سعر-طلب واضحة. في نسخة الإنتاج نستبدلها بداتا حقيقية
  من نفس المخطط.
- حجم البيانات 4000 صف — كافٍ للـ demo لكن ليس للإنتاج.
- الهدف `qty` مستمر → استخدمنا **Regressors** (اضبطنا أن أول نسخة كانت تستخدم
  `RandomForestClassifier` بالخطأ وتم تصحيحها).
```

---

Built with Python, scikit-learn, XGBoost, Streamlit, Plotly & pandas.
