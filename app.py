
import pandas as pd
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

df = pd.read_csv('/content/realistic_diabetes_dataset_1000 (1).csv')

X = df[['Age', 'BMI', 'BloodSugar_mg_dL', 'BloodPressure']]
y = df['DiabetesRisk']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Perceptron(
    max_iter=1000,
    eta0=0.01,
    random_state=42
)

model.fit(X_train, y_train)

def predict_diabetes(age, weight, height, sugar, pressure):

    bmi = weight / ((height / 100) ** 2)

    input_data = [[age, bmi, sugar, pressure]]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        result = "Có nguy cơ mắc tiểu đường"
    else:
        result = "Nguy cơ thấp"

    return f"""
BMI của bạn: {bmi:.2f}

Kết quả chẩn đoán:
{result}
"""

interface = gr.Interface(
    fn=predict_diabetes,

    inputs=[
        gr.Number(label="Tuổi"),
        gr.Number(label="Cân nặng (kg)"),
        gr.Number(label="Chiều cao (cm)"),
        gr.Number(label="Đường huyết (mg/dL)"),
        gr.Number(label="Huyết áp")
    ],

    outputs=gr.Textbox(
        label="Kết quả",
        lines=8
    ),

    title="Hệ thống dự đoán tiểu đường",
    description="Nhập thông tin sức khỏe để kiểm tra nguy cơ tiểu đường",

    flagging_mode="never"
)

interface.launch()
