import streamlit as st
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ✅ โหลดโมเดล Titanic (ใช้ Scikit-learn)
titanic_model = joblib.load("models/titanic_model.pkl")

# ✅ โหลดโมเดล MNIST (ใช้ PyTorch CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cpu")
mnist_model = CNN()
mnist_model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
mnist_model.eval()

# ✅ ตั้งค่า UI
st.set_page_config(page_title="AI Prediction Web App", page_icon="🤖", layout="wide")

st.sidebar.title("🔍 AI Prediction Web App")
page = st.sidebar.radio(
    "เลือกหน้า",
    ["🏠 หน้าแรก", "📘 About Machine Learning", "📘 About Neural Network", "📊 ML Demo (Titanic)", "🤖 NN Demo (MNIST)"]
)

# -------------- หน้าแรก --------------
if page == "🏠 หน้าแรก":
    st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🤖 AI Prediction Web App</h1>", unsafe_allow_html=True)
    st.write("เลือกฟังก์ชันที่ต้องการจาก Sidebar ด้านซ้าย")
    
    
    
    
    st.subheader("📘 About Machine Learning")
    st.write("ศึกษาข้อมูลเกี่ยวกับ Machine Learning และ Titanic Dataset เรียนรู้กระบวนการพัฒนาโมเดลตั้งแต่การเตรียมข้อมูล, ทฤษฎีอัลกอริทึม, และการสร้างโมเดลคาดการณ์การรอดชีวิตของผู้โดยสารไททานิค")
        
    st.subheader("📊 Machine Learning Demo")
    st.write("ทดลองใช้โมเดล Machine Learning ที่สร้างขึ้นจาก Random Forest, Logistic Regression, และ SVM ป้อนข้อมูลผู้โดยสารแล้วทำนายโอกาสรอดชีวิตจากอุบัติเหตุไททานิค")
        
    
    st.subheader("📘 About Neural Network")
    st.write("เรียนรู้เกี่ยวกับ Neural Networks และ Convolutional Neural Networks (CNN) ศึกษาการใช้ CNN ในการจำแนกภาพ เช่น การจดจำลายมือ หรือรูปแบบข้อมูลเชิงซ้อน")
        

    st.subheader("🤖 Neural Network Demo")
    st.write("ทดลองใช้งานโมเดล CNN ที่ผ่านการฝึกมาเพื่อจดจำตัวเลขจากชุดข้อมูล MNIST อัปโหลดภาพหรือลองป้อนข้อมูลเพื่อดูผลลัพธ์")

# -------------- About Machine Learning --------------
elif page == "📘 About Machine Learning":
    st.header("📘 About Machine Learning & Titanic Dataset")
    
    st.subheader("🔧 แนวทางการพัฒนา Machine Learning")
    st.write("""
    การพัฒนาโมเดล Machine Learning สำหรับ Titanic Dataset เป็นไปตามขั้นตอนหลักดังนี้:
    """)

    st.markdown("📊 การเตรียมข้อมูล")
    st.write("""
    - ลบค่าข้อมูลที่หายไป (Missing Values)
    - แปลงค่าข้อมูลประเภทหมวดหมู่เป็นตัวเลข (Categorical Encoding)
    - ทำการปรับขนาดค่าข้อมูล (Feature Scaling) เพื่อให้โมเดลเรียนรู้ได้ดีขึ้น
    """)

    st.markdown("🧠 ทฤษฎีของอัลกอริทึม")
    st.write("""
    โมเดลที่ถูกนำมาใช้ในการเรียนรู้:
    - *Random Forest*: ใช้การรวมผลจาก Decision Trees
    - *Logistic Regression*: ใช้สำหรับปัญหาจำแนกประเภท (Classification)
    - *Support Vector Machine (SVM)*: ใช้เส้น Hyperplane ในการแยกข้อมูล
    """)

    st.markdown("🏗️ ขั้นตอนการพัฒนาโมเดล")
    st.write("""
    1. โหลดและทำความสะอาดข้อมูล Titanic Dataset
    2. ทำการวิเคราะห์ข้อมูล และเลือก Feature ที่สำคัญ
    3. แบ่งชุดข้อมูลเป็น Train/Test Split
    4. เลือกอัลกอริทึมและฝึก Train โมเดล
    5. วิเคราะห์โมเดลโดยใช้ค่าต่างๆ เช่น Accuracy, Precision, Recall, และ F1-score
    """)
    
    st.markdown("### 📂 Dataset: Titanic Survival")
    st.write("ชุดข้อมูลนี้ใช้ข้อมูลจากผู้โดยสาร Titanic และพยากรณ์ว่าผู้โดยสารรอดชีวิตหรือไม่")

    st.markdown("### 📌 Feature ของ Titanic Dataset")
    data = {
        "Feature Name": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
        "คำอธิบาย": [
            "ระดับชั้นโดยสาร (1st, 2nd, 3rd class)",
            "เพศของผู้โดยสาร (Male/Female)",
            "อายุของผู้โดยสาร",
            "จำนวนพี่น้อง/คู่สมรสที่เดินทางด้วยกัน",
            "จำนวนพ่อแม่/ลูกที่เดินทางด้วยกัน",
            "ค่าตั๋วโดยสาร",
            "ท่าเรือขึ้นเรือ (C = Cherbourg, Q = Queenstown, S = Southampton)"
        ]
    }
    df = pd.DataFrame(data)
    st.table(df)

    st.markdown("### 📄 ตัวอย่างข้อมูลจากไฟล์ CSV")
    st.code("Pclass, Sex, Age, SibSp, Parch, Fare, Embarked\n3, Male, 22, 1, 0, 7.25, S\n1, Female, 38, 1, 0, 71.28, C", language="text")

    st.markdown("### 📚 แหล่งอ้างอิง")
    st.markdown("""
    - 📌 Titanic - Machine Learning from Disaster 🔗 [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
    - 📌 Dataset Titanic - Machine Learning from Disaster 🔗(https://www.kaggle.com/c/titanic/data)
    - 📌 Scikit-learn Machine Learning in Python 🔗 [Scikit-learn Machine Learning in Python](https://scikit-learn.org/stable/)
    - 📌 TensorFlow Guide 🔗 [TensorFlow Guide](https://www.tensorflow.org/)
    """)

# -------------- About Neural Network --------------
elif page == "📘 About Neural Network":
    st.markdown("<h1 style='color: #ff9900;'>📘 About Neural Network & MNIST Dataset</h1>", unsafe_allow_html=True)

    st.subheader("📌 แนวทางการพัฒนา Neural Network")
    st.write("""
    การพัฒนาโมเดล Neural Network สำหรับ MNIST Dataset เริ่มต้นจากกระบวนการสำคัญ 3 ขั้นตอน
    """)

    st.subheader("🔍 การเตรียมข้อมูล")
    st.write("""
    - ชุดข้อมูล MNIST มีรูปภาพขนาด 28x28 พิกเซล และถูกแปลงเป็น grayscale  
    - ค่า pixel อยู่ในช่วง 0-255 และถูก normalize ให้มีค่าอยู่ในช่วง 0-1  
    - ทำให้โมเดลสามารถเรียนรู้ได้มีประสิทธิภาพมากขึ้น  
    """)

    st.subheader("📌 ทฤษฎีของอัลกอริทึม")
    st.write("""
    โมเดลที่ใช้คือ *Convolutional Neural Network (CNN)* ซึ่งเหมาะสำหรับการประมวลผลภาพ โดยมีชั้นสำคัญ เช่น:
    - *Conv2D:* ใช้ฟิลเตอร์ดึงลักษณะเด่นของภาพ  
    - *MaxPooling:* ลดขนาดภาพเพื่อลดพารามิเตอร์  
    - *Flatten:* แปลงข้อมูลภาพเป็นเวกเตอร์  
    - *Dense:* ใช้ Fully Connected Layer เชื่อมข้อมูลลักษณะ  
    """)

    st.subheader("⚙️ ขั้นตอนการพัฒนาโมเดล")
    st.write("""
    การพัฒนาโมเดล Neural Network มีขั้นตอนดังนี้:
    1. โหลดและเตรียมข้อมูล MNIST  
    2. สร้างโมเดล CNN ตามโครงสร้างที่กำหนด  
    3. กำหนดฟังก์ชัน Activation และ Optimizer  
    4. ทำการ Train โมเดลด้วยชุดข้อมูล  
    5. ประเมินผลลัพธ์ของโมเดล  
    """)

    st.subheader("📊 โครงสร้างของ Dataset")
    st.write("ข้อมูลมี 60,000 ตัวอย่างสำหรับฝึก และ 10,000 ตัวอย่างสำหรับทดสอบ")

    # แสดงตาราง Feature ของ MNIST Dataset
    mnist_table = pd.DataFrame({
        "Feature Name": ["label", "pixel0 - pixel783"],
        "คำอธิบาย": ["ค่าจริงของตัวเลข (0-9)", "ค่าความเข้มของพิกเซล (0-255)"]
    })
    st.table(mnist_table)

    st.subheader("📂 ตัวอย่างข้อมูลจากไฟล์ CSV")
    st.code("""
    label, pixel0, pixel1, ..., pixel782, pixel783
    5, 0, 0, ..., 159, 253, 255
    """)

    st.subheader("🖼️ ตัวอย่างการใช้งานไฟล์รูปภาพ 28x28")
    st.code("""
    mnist_png
    ├── training/testing
    │   ├── label
    │   │   ├── id.png
    """)

    st.subheader("📚 แหล่งอ้างอิง")
    st.markdown("""
    - 📌 LeCun, Y., Cortes, C., & Burges, C. (1998). The MNIST database of handwritten digits.
    - 📌 Kaggle MNIST Dataset: 🔗 [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download)
    - 📌 TensorFlow Guide for CNN: 🔗 [TensorFlow Guide for CNN](https://www.tensorflow.org/tutorials/images/cnn)
    - 📌 Keras Documentation for CNN: 🔗 [Keras Conv2D Layer](https://keras.io/api/layers/convolution_layers/convolution2d/)
    - 📌 GitHub MNIST PNG Dataset: 🔗 [GitHub MNIST PNG Dataset](https://github.com/myleott/mnist_png)
    """)

# -------------- Machine Learning Demo (Titanic) --------------
elif page == "📊 ML Demo (Titanic)":
    st.title("🚢 Titanic Survival Prediction")

    pclass = st.selectbox("Pclass (ชั้นโดยสาร)", [1, 2, 3])
    sex = st.selectbox("Sex (เพศ)", ["ชาย", "หญิง"])
    age = st.slider("Age (อายุ)", 0, 100, 30)
    sibsp = st.slider("SibSp (จำนวนพี่น้อง/คู่สมรส)", 0, 10, 0)
    parch = st.slider("Parch (จำนวนพ่อแม่/ลูก)", 0, 10, 0)
    fare = st.number_input("Fare (ค่าโดยสาร)", 0.0, 500.0, 50.0)
    embarked = st.selectbox("Embarked (ท่าเรือขึ้นเรือ)", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])

    sex = 0 if sex == "ชาย" else 1
    embarked = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}[embarked]

    if st.button("🔍 ทำนายผล"):
        input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        prediction = titanic_model.predict(input_data)[0]
        result = "✅ รอดชีวิต" if prediction == 1 else "❌ ไม่รอดชีวิต"
        st.success(f"ผลลัพธ์: {result}")

# -------------- Neural Network Demo (MNIST) --------------
elif page == "🤖 NN Demo (MNIST)":
    st.title("🖼️ MNIST Handwritten Digit Recognition")

    uploaded_file = st.file_uploader("📤 อัปโหลดภาพตัวเลข (0-9)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        st.image(img, caption="📸 ภาพที่อัปโหลด", width=150)

        if st.button("🔍 ทำนายตัวเลข"):
            with torch.no_grad():
                prediction = mnist_model(img_tensor)
                predicted_label = torch.argmax(prediction).item()
            st.success(f"โมเดลทำนายว่าเป็นเลข: {predicted_label}")