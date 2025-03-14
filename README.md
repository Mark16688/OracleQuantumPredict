# OracleQuantumPredict

AI Prediction Web App & Demo เป็นแอปพลิเคชันที่ใช้ Machine Learning (ML) และ Neural Network (NN) ในการวิเคราะห์และทำนายข้อมูลสองประเภทหลัก:
	1.	Titanic Survival Prediction – ทำนายโอกาสรอดชีวิตของผู้โดยสารเรือ Titanic โดยใช้อัลกอริทึม Machine Learning
	2.	MNIST Handwritten Digit Recognition – จำแนกตัวเลขจากลายมือด้วย Convolutional Neural Network (CNN)

 Tech Stack ที่ใช้ในโปรเจค
	•	Frontend & UI: Streamlit
	•	Machine Learning Model: Scikit-learn, Joblib
	•	Neural Network Model: TensorFlow, Keras
	•	Data Processing: Pandas, NumPy
	•	Deployment: GitHub, Streamlit Cloud (หรือแพลตฟอร์มอื่น ๆ)

⸻

 คุณสมบัติหลักของโปรเจค

 1. Machine Learning: Titanic Survival Prediction
	•	ใช้ Titanic Dataset ในการทำนายว่าผู้โดยสารรอดชีวิตหรือไม่
	•	อัลกอริทึมที่ใช้: Random Forest, Logistic Regression, Support Vector Machine (SVM)
	•	รับอินพุตจากผู้ใช้ เช่น ชั้นโดยสาร เพศ อายุ ค่าตั๋ว ฯลฯ
	•	แสดงผลลัพธ์ว่า รอดชีวิต หรือ ไม่รอดชีวิต

 2. Neural Network: MNIST Handwritten Digit Recognition
	•	ใช้ CNN Model เพื่อทำนายตัวเลขจากภาพ
	•	อัปโหลดรูปภาพ (ไฟล์ .png, .jpg, .jpeg) แล้วให้โมเดลทำนายค่าที่เป็นตัวเลข (0-9) (ขนาด28*28 พิกเซล)
	•	ใช้โครงสร้างโมเดล CNN: Conv2D, MaxPooling, Flatten, Dense layers

ตัวอย่างการใช้งาน

หน้าแรก(Home)
 • About Machine Learning ให้อ่านเนื้อหาคร่าวๆของ Machine Learining นี้
   • ถ้ากดไปที่ปุ่มของ About Machine Learning จะแสดงเนื้อหาทั้งหมด เช่น อัลกอริทึมที่ใช้ , แหล่งอ้างอิง เป็นต้น
 • About Neural Network ให้อ่านเนื้อหาคร่าวๆของ Neural Network นี้
   • ถ้ากดไปที่ปุ่มของ About Neural Network จะแสดงเนื้อหาทั้งหมด เช่น อัลกอริทึมที่ใช้ , แหล่งอ้างอิง เป็นต้น
 • Machine Learning Demo จะอธิบายการทำงานคร่าวๆของ Machine Learning
   • จะให้เราลอง Demo ดู ในการใส่ค่าเช่น อยู่คลาสชั้นไหนของเรือ, อายุ , พี่น้องหรือพ่อแม่ 
 • Neural Network Demo จะอธิบายการทำงานคร่าวๆของ Neural Network
   • จะให้เราลอง Demo ดู โดยอัพโหลดภาพของตัวเลข (0-9) (28*28 pixel)
