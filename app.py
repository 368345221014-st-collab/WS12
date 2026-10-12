"""
Women's Clothing Review — Recommendation Predictor
เว็บแอปทำนายว่ารีวิวสินค้าแฟชั่นผู้หญิงจะถูก "แนะนำ" หรือไม่ โดยใช้โมเดล Pipeline SVM (womens_clothing_svm.pkl)

วิธีรัน:
 streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Women's Clothing Recommendation Predictor",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "womens_clothing_svm.pkl"

# ---------------------------------------------------------------------------
# Custom CSS — soft fashion-tech palette
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #FBF7F5; }

    .hero {
        background: linear-gradient(135deg, #C97B84 0%, #E8B4B8 55%, #F4D9DA 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(201,123,132,0.25);
    }
    .hero h1 { margin: 0; font-size: 2rem; font-weight: 800; }
    .hero p { margin: 0.4rem 0 0 0; font-size: 1.02rem; opacity: 0.95; }

    .card {
        background: white;
        border-radius: 16px;
        padding: 1.6rem 1.7rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        border: 1px solid #F0E4E3;
        margin-bottom: 1.2rem;
    }

    .result-good {
        background: linear-gradient(135deg, #DFF3E7 0%, #C9EBD7 100%);
        border: 1px solid #8FD6AE;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-bad {
        background: linear-gradient(135deg, #FBE3E4 0%, #F6C8CA 100%);
        border: 1px solid #E79EA1;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
    }
    .result-title { font-size: 1.6rem; font-weight: 800; margin: 0.3rem 0; }
    .result-sub { font-size: 0.95rem; color: #555; }

    .footer-box {
        margin-top: 2.5rem;
        padding: 1.2rem 1.5rem;
        background: #FFFFFF;
        border-top: 3px solid #C97B84;
        border-radius: 12px;
        font-size: 0.88rem;
        color: #555;
    }
    .footer-box b { color: #C97B84; }

    div[data-testid="stMetricValue"] { color: #C97B84; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>👗 Women's Clothing Recommendation Predictor</h1>
        <p>ระบบทำนายแนวโน้มการ "แนะนำสินค้า" จากพฤติกรรมและคะแนนรีวิวของผู้รีวิว
        โดยใช้แบบจำลอง Support Vector Machine (SVM)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — model loading + about
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ โมเดล")

    model = None
    model_source = None

    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            model_source = f"โหลดจากไฟล์ `{MODEL_PATH}` สำเร็จ"
        except Exception as e:
            st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")

    uploaded = st.file_uploader("หรืออัปโหลดไฟล์โมเดล (.pkl)", type=["pkl"])
    if uploaded is not None:
        try:
            model = joblib.load(uploaded)
            model_source = "โหลดจากไฟล์ที่อัปโหลดสำเร็จ"
        except Exception as e:
            st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")

    if model is not None:
        st.success(model_source)
        st.caption(f"ประเภทโมเดล: `{type(model).__name__}`")
        
        # ดึงโมเดลย่อยกรณีที่ถูกเซฟมาเป็น Pipeline
        classifier = model
        if hasattr(model, "named_steps"):
            classifier = model.steps[-1][1]
            st.caption("โครงสร้าง: `Pipeline (StandardScaler + Estimator)`")

        if hasattr(classifier, "kernel"):
            st.caption(f"Kernel: `{classifier.kernel}`")
        if hasattr(classifier, "n_features_in_"):
            st.caption(f"จำนวน Feature ที่โมเดลต้องการ: `{classifier.n_features_in_}`")
    else:
        st.warning("ยังไม่พบไฟล์โมเดล กรุณาวาง `womens_clothing_svm.pkl` ไว้โฟลเดอร์เดียวกับ app.py "
                   "หรืออัปโหลดไฟล์ด้านบน")

    st.divider()
    st.header("ℹ️ เกี่ยวกับแอปนี้")
    st.caption(
        "แอปนี้พัฒนาขึ้นเพื่อสาธิตการนำแบบจำลอง Machine Learning (SVM) "
        "มาทำนายว่าผู้รีวิวจะแนะนำสินค้าเสื้อผ้าแฟชั่นผู้หญิงหรือไม่ "
        "โดยอิงจาก 3 ตัวแปรนำเข้า ได้แก่ อายุผู้รีวิว คะแนนรีวิว "
        "และจำนวนผู้ที่เห็นว่ารีวิวมีประโยชน์"
    )

# ---------------------------------------------------------------------------
# Main input form
# ---------------------------------------------------------------------------
left, right = st.columns([1.1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 กรอกข้อมูลรีวิวสินค้า")

    age = st.slider("อายุของผู้รีวิว (Age)", min_value=13, max_value=99, value=35, step=1)

    rating = st.select_slider(
        "คะแนนรีวิว (Rating)",
        options=[1, 2, 3, 4, 5],
        value=4,
        format_func=lambda x: "⭐" * x,
    )

    feedback_count = st.number_input(
        "จำนวนผู้ที่เห็นว่ารีวิวมีประโยชน์ (Positive Feedback Count)",
        min_value=0, max_value=500, value=3, step=1,
    )

    st.caption("ℹ️ ลำดับ feature ที่ส่งเข้าโมเดล: **[Age, Rating, Positive Feedback Count]** "
               "ตามชุดข้อมูล Women's Clothing E-Commerce Reviews")

    predict_clicked = st.button("🔮 ทำนายผล", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 ข้อมูลที่จะส่งเข้าโมเดล")
    input_df = pd.DataFrame(
        {
            "Feature": ["Age", "Rating", "Positive Feedback Count"],
            "ค่าที่กรอก": [age, rating, feedback_count],
        }
    )
    st.write(input_df.to_html(index=False), unsafe_allow_html=True)

    st.markdown("##### แนวโน้มคร่าว ๆ จากข้อมูลที่กรอก")
    c1, c2, c3 = st.columns(3)
    c1.metric("อายุ", age)
    c2.metric("คะแนน", f"{rating} / 5")
    c3.metric("Feedback", feedback_count)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_clicked:
    if model is None:
        st.error("⚠️ กรุณาโหลดไฟล์โมเดล womens_clothing_svm.pkl ก่อนทำการทำนาย")
    else:
        # ส่งค่าอินพุตเข้าโมเดล Pipeline (Pipeline จะทำการ StandardScaler แล้วส่งให้ SVC ทำนายอัตโนมัติ)
        X = np.array([[age, rating, feedback_count]])
        try:
            pred = model.predict(X)[0]

            confidence_pct = None
            if hasattr(model, "decision_function"):
                score = model.decision_function(X)[0]
                confidence_pct = 1 / (1 + np.exp(-score)) * 100

            st.markdown("---")
            if pred == 1:
                st.markdown(
                    f"""
                    <div class="result-good">
                        <div style="font-size:2.5rem;">✅</div>
                        <div class="result-title" style="color:#2E7D4F;">แนะนำสินค้า (Recommended)</div>
                        <div class="result-sub">โมเดลคาดว่าลูกค้าน่าจะแนะนำสินค้าชิ้นนี้</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-bad">
                        <div style="font-size:2.5rem;">❌</div>
                        <div class="result-title" style="color:#B23A3E;">ไม่แนะนำสินค้า (Not Recommended)</div>
                        <div class="result-sub">โมเดลคาดว่าลูกค้าน่าจะไม่แนะนำสินค้าชิ้นนี้</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if confidence_pct is not None:
                st.markdown("##### ระดับความมั่นใจโดยประมาณ (จาก decision function)")
                st.progress(min(max(confidence_pct / 100, 0.0), 1.0))
                st.caption(f"≈ {confidence_pct:.1f}% — เป็นค่าประมาณจากระยะห่างของจุดข้อมูลต่อเส้นแบ่งของ SVM "
                           "ไม่ใช่ค่าความน่าจะเป็นที่แท้จริง เนื่องจากโมเดลนี้ฝึกโดยไม่ได้เปิดใช้ probability=True")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างทำนายผล: {e}")

# ---------------------------------------------------------------------------
# Footer — team credit
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-box">
        <b>👥 จัดทำโดยสมาชิก</b><br>
        1. ณภัทร คล้ายวรรณ &nbsp; (014)<br>
        2. ธวัลรัตน์ พันพิลา &nbsp; (019)<br>
        3. บัณฑิตา ภักดี &nbsp; (023)
    </div>
    """,
    unsafe_allow_html=True,
)