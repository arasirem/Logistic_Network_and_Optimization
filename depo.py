import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title('Optimization of Warehouse Layouts')

uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader('Sample of Data')
    st.write(df.head())

    df_clean = df[['city_lat', 'city_lng']].dropna().drop_duplicates()
    
    # Elbow Yöntemi ile Küme Sayısını Belirlemek
    wcss = []
    for i in range(2, 11):  # Küme sayısını 2 ile 10 arasında deniyoruz
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(df_clean[['city_lat', 'city_lng']])
        wcss.append(kmeans.inertia_)

   
    # Küme Sayısını Kullanıcıya Seçtirme
    n_clusters = st.slider("Choice number of Depot", min_value=2, max_value=10, value=4)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_clean['cluster'] = kmeans.fit_predict(df_clean[['city_lat', 'city_lng']])
    df_clean['warehouse'] = df_clean['cluster'].apply(lambda x: f"DEPO {x+1}")
    centers = kmeans.cluster_centers_

    st.subheader("Optimum Depot Locations")
    for i, (lat, lng) in enumerate(centers):
        st.write(f"DEPOT {i+1} Location: Ltd = {lat:.5f}, Lng = {lng:.5f}")

    st.subheader("City and Depot Locations")

    # Plotly Harita Verileri
    city_scatter = go.Scattermapbox(
        lon = df_clean['city_lng'],
        lat = df_clean['city_lat'],
        text = df_clean['cluster'].apply(lambda x: f'Küme {x}'),
        mode = 'markers',
        marker = dict(
            size = 8,
            color = df_clean['cluster'],
            colorscale = 'Viridis',
            showscale = True,
            colorbar=dict(title="Küme")
        ),
        name = 'Cities'
    )

    depot_scatter = go.Scattermapbox(
    lon = centers[:, 1],
    lat = centers[:, 0],
    mode = 'markers+text',
    marker = dict(size = 20, color = 'black'),
    text = [f"📦 Depo {i+1}" for i in range(len(centers))],
    textfont = dict(size=14, color='black'),
    textposition = "top center",
    name = 'Depot Locations'
)



    fig = go.Figure(data=[city_scatter, depot_scatter])

    # 🔧 Harita Ayarları (Mapbox)
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=37.8, lon=-96.9),  # ABD merkezli, Türkiye için lat=39.0, lon=35.0 yapabilirsin
            zoom=3
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=700,
        title="LOGISTIC NETWORK: Factory -> Customer Points"
    )

    st.plotly_chart(fig, use_container_width=True)
    