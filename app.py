import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

st.title("💎 Diamond Price Analysis & Prediction")

# Load data
df = pd.read_csv("diamonds.csv")

# Preprocessing
df.dropna(inplace=True)
df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]

le_cut = LabelEncoder()
le_color = LabelEncoder()
le_clarity = LabelEncoder()
df['cut'] = le_cut.fit_transform(df['cut'])
df['color'] = le_color.fit_transform(df['color'])
df['clarity'] = le_clarity.fit_transform(df['clarity'])

# Sidebar option
option = st.sidebar.selectbox("Choose Action", ["Dataset Overview", "Visualizations", "Model & Predict"])

if option == "Dataset Overview":
    st.write(df.head())
    st.write(df.describe())

elif option == "Visualizations":
    st.subheader("Histogram of Price")
    fig1 = plt.figure()
    sns.histplot(df["price"], kde=True)
    st.pyplot(fig1)

    st.subheader("Carat vs Price")
    fig2 = plt.figure()
    sns.scatterplot(x="carat", y="price", data=df)
    st.pyplot(fig2)

    st.subheader("Boxplot by Cut")
    fig3 = plt.figure()
    sns.boxplot(x="cut", y="price", data=df)
    st.pyplot(fig3)

    st.subheader("Heatmap Correlation")
    fig4 = plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig4)

elif option == "Model & Predict":
    X = df[['carat','depth','table','x','y','z','cut','color','clarity']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"📉 Mean Squared Error: {mse}")
    st.write(f"📈 R-squared Score: {r2}")

