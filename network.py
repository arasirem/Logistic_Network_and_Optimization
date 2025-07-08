import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
from streamlit_folium import folium_static
from folium import plugins
from geopy.distance import geodesic

st.set_page_config(layout="wide")
st.title('Optimization of Warehouse Layouts and Network Visualization')

uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader('Sample of Data')
    st.write(df.head())

    # Temel doğrulama: Gerekli kolonlar
    required_cols = ['city', 'city_lat', 'city_lng', 'Factory', 'Factory Lat', 'Factory Lng']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV dosyanızda aşağıdaki kolonlar olmalı: {required_cols}")
        st.stop()

    # Müşteri konumları için clustering
    df_clean = df[['city_lat', 'city_lng']].dropna().drop_duplicates()

    # Küme sayısı seçimi
    n_clusters = st.slider("Choice number of Depots", min_value=2, max_value=10, value=4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_clean['cluster'] = kmeans.fit_predict(df_clean[['city_lat', 'city_lng']])
    centers = kmeans.cluster_centers_

    # Depo merkezleri dataframe
    depots = pd.DataFrame(centers, columns=["Depot Lat", "Depot Lng"])
    depots['Depot'] = ["DEPOT " + str(i + 1) for i in range(len(depots))]

    st.subheader("Optimum Depot Locations")
    st.write(depots)

    # 1. Başlangıç: Boş eşleşme listesi ve kontrol için set
    factory_coords = df[["Factory", "Factory Lat", "Factory Lng"]].drop_duplicates()
    assigned_factories = set()
    assigned_depots = set()
    factory_depot_map = []

    # 2. İlk olarak her depo için en yakın fabrikayı bul 
    for i, depot_row in depots.iterrows():
        depot_coord = (depot_row["Depot Lat"], depot_row["Depot Lng"])
        min_distance = float("inf")
        nearest_factory = None

        for _, factory in factory_coords.iterrows():
            if factory["Factory"] in assigned_factories:
                continue  # Zaten atandıysa geç

            factory_coord = (factory["Factory Lat"], factory["Factory Lng"])
            distance = geodesic(factory_coord, depot_coord).km
            if distance < min_distance:
                min_distance = distance
                nearest_factory = factory

        if nearest_factory is not None:
            assigned_factories.add(nearest_factory["Factory"])
            assigned_depots.add(depot_row["Depot"])
            factory_depot_map.append({
                "Factory": nearest_factory["Factory"],
                "Factory Lat": nearest_factory["Factory Lat"],
                "Factory Lng": nearest_factory["Factory Lng"],
                "Depot": depot_row["Depot"],
                "Depot Lat": depot_row["Depot Lat"],
                "Depot Lng": depot_row["Depot Lng"]
            })

    # 3. Geriye kalan fabrikaları en yakın depoya ata
    for _, factory in factory_coords.iterrows():
        if factory["Factory"] in assigned_factories:
            continue

        factory_coord = (factory["Factory Lat"], factory["Factory Lng"])
        distances = [geodesic(factory_coord, (lat, lng)).km for lat, lng in zip(depots["Depot Lat"], depots["Depot Lng"])]
        nearest_depot_idx = np.argmin(distances)

        factory_depot_map.append({
            "Factory": factory["Factory"],
            "Factory Lat": factory["Factory Lat"],
            "Factory Lng": factory["Factory Lng"],
            "Depot": depots.iloc[nearest_depot_idx]["Depot"],
            "Depot Lat": depots.iloc[nearest_depot_idx]["Depot Lat"],
            "Depot Lng": depots.iloc[nearest_depot_idx]["Depot Lng"]
        })

    factory_depot_df = pd.DataFrame(factory_depot_map)


    # Müşterilere en yakın depoyu ata
    def get_nearest_depot(row):
        customer_location = (row["city_lat"], row["city_lng"])
        distances = [geodesic(customer_location, (lat, lng)).km for lat, lng in zip(depots["Depot Lat"], depots["Depot Lng"])]
        nearest_depot_idx = np.argmin(distances)
        return depots.iloc[nearest_depot_idx][["Depot", "Depot Lat", "Depot Lng"]]

    nearest_depots = df.apply(get_nearest_depot, axis=1)
    df_with_depot = pd.concat([df, nearest_depots], axis=1)

    # Maksimum gösterilecek rota sayısı
    max_rows = st.slider("How many maximum routes should be displayed?", min_value=10, max_value=10000, value=300)
    filtered_df = df_with_depot.head(max_rows)

    # Harita oluşturma
    st.subheader("Logistics Network Map")
    center_of_map = [df["city_lat"].mean(), df["city_lng"].mean()]
    map_logistics = folium.Map(location=center_of_map, zoom_start=6)
    plugins.Fullscreen().add_to(map_logistics)

    # Fabrika marker (kırmızı)
    for _, row in factory_depot_df.iterrows():
        folium.Marker(
            location=(row["Factory Lat"], row["Factory Lng"]),
            icon=folium.Icon(color='red', icon="industry", prefix="fa"),
            tooltip=f"Fabrika: {row['Factory']}"
        ).add_to(map_logistics)

    # Müşteri marker (yeşil)
    customer_coords = filtered_df[["city", "city_lat", "city_lng"]].drop_duplicates()
    for _, row in customer_coords.iterrows():
        folium.CircleMarker(
            location=(row["city_lat"], row["city_lng"]),
            radius=4,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.7,
            tooltip=row["city"]
        ).add_to(map_logistics)

    # Depo marker (mavi)
    for _, row in depots.iterrows():
        folium.Marker(
            location=(row["Depot Lat"], row["Depot Lng"]),
            icon=folium.Icon(color='blue', icon="warehouse", prefix="fa"),
            tooltip=row["Depot"]
        ).add_to(map_logistics)

    # Rota çizgileri:

    # 1) Tüm fabrikalar için Fabrika → Depo çizgileri (her fabrika kendi nearest depoya bağlanacak)
    for _, factory_row in factory_depot_df.iterrows():
        factory_coord = (factory_row["Factory Lat"], factory_row["Factory Lng"])
        depot_coord = (factory_row["Depot Lat"], factory_row["Depot Lng"])
        folium.PolyLine([factory_coord, depot_coord], color="orange", weight=2.5).add_to(map_logistics)

    # 2) Müşteriler için Depo → Müşteri çizgileri
    for _, row in filtered_df.iterrows():
        depot_coord = (row["Depot Lat"], row["Depot Lng"])
        customer_coord = (row["city_lat"], row["city_lng"])
        folium.PolyLine([depot_coord, customer_coord], color="purple", weight=2.5).add_to(map_logistics)

    folium_static(map_logistics, width=1200, height=700)

else:
    st.warning("Lütfen bir CSV dosyası yükleyin.")
