# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import altair as alt
from math import radians, cos, sin, sqrt, atan2
import pydeck as pdk
import gdown

# Load saved models/data
df = pd.read_pickle('hotel_data_malaysia.pkl', compression='gzip')
# @st.cache_data
# def load_hotel_data():
#     file_id = "12cYKf_a8t2_sHsScXSXj_u5YOo_CBqEm"
#     url = f"https://drive.google.com/uc?id={file_id}"
#     output = "hotel_data.pkl"
#     gdown.download(url, output, quiet=False)
#     return pd.read_pickle(output)

# df = load_hotel_data()
vectorizer = joblib.load('tfidf_vectorizer_malaysia.pkl')

# Streamlit UI
st.title("🏨 Hotel Recommendation System")

# User select input country and city
selected_country = st.selectbox("Select Country (required)", sorted(df['countyName'].dropna().unique()))
city_list = df[df['countyName'] == selected_country]['cityName'].dropna().unique()
selected_city = st.selectbox("Select City (required)", sorted(city_list))

# Filter df to selected country and city, display respective info
filter_hotels = df[(df['countyName'] == selected_country) & (df['cityName'] == selected_city)]
num_hotels = filter_hotels.shape[0]
st.metric(label=f"🧾 Total Hotels in {selected_city}", value=num_hotels)
st.dataframe(filter_hotels[['HotelName', 'HotelRating', 'Address', 'PhoneNumber', 'HotelWebsiteUrl', 'HotelFacilities']])
st.map(filter_hotels[['latitude', 'longitude']])

# Check and clean up your HotelRating column
# Ensure HotelRating is treated as a string for grouping
filter_hotels['HotelRating'] = filter_hotels['HotelRating'].fillna('Unknown').astype(str)

# Group and count ratings
rating_counts = (
    filter_hotels.groupby('HotelRating')
    .size()
    .reset_index(name='Count')
)

# Bar chart
chart = alt.Chart(rating_counts).mark_bar().encode(
    x=alt.X('HotelRating:N', title='Hotel Rating', sort='-y'),
    y=alt.Y('Count:Q', title='Number of Hotels'),
    color='HotelRating:N'
).properties(
    title=f'Hotel Ratings Distribution in {selected_city}',
    width=600,
    height=400
)

st.altair_chart(chart, use_container_width=True)

# Optional, user able to select rating and facility
rating_list = df[df['cityName'] == selected_city]['HotelRating'].dropna().unique()
selected_ratings = st.multiselect("Select Star Ratings (optional)", sorted(rating_list))
selected_facility = st.text_input("Enter a facility (e.g., pool, wifi)")

# Filter df and return hotel list that fit user-defined requirement
if selected_ratings:
    rating_hotels= filter_hotels[(filter_hotels ['HotelRating'].isin(selected_ratings))]
    rating_hotels_list= rating_hotels['HotelName'].dropna().unique()
    selected_hotel = st.selectbox(f"Hotels in {selected_city} with stars & facilities choosen", rating_hotels_list)
else:
    hotel_list = filter_hotels['HotelName'].dropna().unique()
    selected_hotel = st.selectbox(f"Hotels in {selected_city} with facilities choosen", hotel_list)

top_n = 5

# Function for content base recommendation
def recommend_similar_hotels_priority(hotel_name, city_name, top_n=5):
    selected_hotel = df[df['HotelName'] == hotel_name].iloc[0]
    selected_rating = selected_hotel['HotelRating']
    selected_vector = vectorizer.transform([selected_hotel['combined_features']])
    
    city_hotels = df[df['cityName'].str.lower() == city_name.lower()]
    same_rating_hotels = city_hotels[city_hotels['HotelRating'] == selected_rating]
    
    if len(same_rating_hotels) < top_n:
        other_hotels = city_hotels[city_hotels['HotelRating'] != selected_rating]
        other_hotels = other_hotels.sort_values(by='RatingValue', ascending=False)
        fallback_hotels = pd.concat([same_rating_hotels, other_hotels]).head(top_n + 1)
    else:
        fallback_hotels = same_rating_hotels

    tfidf_features = vectorizer.transform(fallback_hotels['combined_features'])
    cosine_sim = cosine_similarity(selected_vector, tfidf_features).flatten()

    fallback_hotels = fallback_hotels.copy()
    fallback_hotels['Similarity'] = cosine_sim
    fallback_hotels = fallback_hotels[fallback_hotels['HotelName'] != hotel_name]
    results = fallback_hotels.sort_values(by=['RatingValue', 'Similarity'], ascending=[False, False])

    return results.head(top_n)

# Button Action
if st.button("Find Similar Hotels"):
    result_df = recommend_similar_hotels_priority(
        hotel_name=selected_hotel,
        city_name=selected_city,
        top_n=top_n
    )

    st.subheader("Recommended Hotels Similar to " + selected_hotel)

    # Get selected hotel coordinates
    selected_coords = df[df['HotelName'] == selected_hotel][['latitude', 'longitude']].iloc[0]
    selected_lat, selected_lon = selected_coords['latitude'], selected_coords['longitude']

    # Compute geospatial distances
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    result_df = result_df.copy()
    result_df['Distance_km'] = result_df.apply(
        lambda row: haversine(selected_lat, selected_lon, row['latitude'], row['longitude']), axis=1
    )

    # Display updated result with distance
    st.dataframe(result_df[['HotelName', 'HotelRating', 'Similarity', 'Distance_km',
                            'Address', 'PhoneNumber', 'HotelWebsiteUrl', 'HotelFacilities']])

    # Create a label with hotel name and distance
    result_df['Label'] = result_df.apply(
        lambda row: f"{row['HotelName']} ({row['Distance_km']:.1f} km)", axis=1
    )

    # Define Scatterplot Layer 
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=result_df,
        get_position='[longitude, latitude]',
        get_color='[0, 128, 255, 160]',
        get_radius=150,
        pickable=True
    )

    # View state centered around selected hotel
    view_state = pdk.ViewState(
        latitude=selected_lat,
        longitude=selected_lon,
        zoom=12,
        pitch=0
    )

    # Display map with only scatter points and tooltip
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{Label}"}
    ))
