# 必要なライブラリをインポートする
import streamlit as st
import torch
from PIL import Image
from pathlib import Path

# YOLOv5モデルの読み込み
model_path = torch.save('best.pt') # 学習済みモデルのパス
model = torch.hub.load('ultralytics/yolov5', 'custom', model=model_path)

# ユーザーインターフェイスの作成
st.title('Object Detection App')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# 物体検出の実行
if uploaded_file is not None:
    # 画像を読み込む
    image = Image.open(uploaded_file)

    # モデルに画像を渡して物体検出を実行する
    results = model(image)

    # 結果を表示する
    st.image(image, caption='Original Image', use_column_width=True)
    st.write('Detected Objects:')
    for result in results.xyxy[0]:
        label = result[-1]
        score = result[-2]
        st.write(f'{label} with a confidence of {score}')
