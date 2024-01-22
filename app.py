from cmath import sin, cos, sqrt, atan
import numpy as np
import streamlit as st
import folium
import json
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from math import sqrt
from matplotlib import pyplot as plt
from docplex.mp.model import Model
import networkx as nx
import csv
# Fonction pour convertir un fichier CSV en JSON

# def saveCSVToJson(csv_file_path, json_file_path):
#     data = []
#     # Read CSV file and convert to JSON
#     with open(csv_file_path, 'r') as csv_file:
#         csv_reader = csv.DictReader(csv_file)
#         for row in csv_reader:
#             data.append(row)

#     # Write JSON file
#     with open(json_file_path, 'w') as json_file:
#         json.dump(data, json_file, indent=2)


# cities_data = {
#     'New York': (40.7128, -74.0060),
#     'Los Angeles': (34.0522, -118.2437),
#     'Chicago': (41.8781, -87.6298),
#     'Houston': (29.7604, -95.3698),
#     'Miami': (25.7617, -80.1918)
# }

def euclidean_distance(lat1, lon1, lat2, lon2):
    lat_diff = (lat1 - lat2) * 111000  
    lon_diff = (lon1 - lon2) * 111000 
    distance = sqrt(pow(lat_diff, 2) + pow(lon_diff, 2))
    return distance

# Lit le fichier JSON avec toutes les villes
with open('data.json', 'r') as json_file:
    cities_data = json.load(json_file)

selected_cities_data = {}

for city_info in cities_data:
    city_name = city_info["city"] + city_info["iso2"]
    latitude = float(city_info["lat"])  # Convert to float
    longitude = float(city_info["lng"])  # Convert to float

    selected_cities_data[city_name] = (latitude, longitude)

#
#
#
#
#
# Function to calculate haversine distances for a list of cities
def calculate_haversine_distances(city_coordinates):
    
    # Calculate haversine distances
    distances = haversine_distances(city_coordinates)

    return distances * (6371) # resutat en km


# 
# 
# 
# 
# 
# 
# 

# Streamlit app
st.title('Choix des villes')
# Widget to select multiple cities
selected_cities = st.multiselect('Select Cities:', list(selected_cities_data.keys()), default=['TokyoJP'])

if(len(selected_cities) > 30):
    selected_cities = selected_cities[:30]
    
if selected_cities:
    st.write(str(len(selected_cities)) + ' villes sélectionnées')
    st.warning('La première ville sélectionnée sera la ville de départ') 
    city_coordinates = [tuple(np.radians(selected_cities_data[city])) for city in selected_cities]

    # Calculate haversine distances
    distances = calculate_haversine_distances(city_coordinates)
    
    # st.title('Distance entre les villes')
    # distances_df = pd.DataFrame(distances, index=selected_cities, columns=selected_cities)
    # st.dataframe(distances_df)
    
    st.title('Distance entre les villes (en km)')
    distances_df = pd.DataFrame(distances, index=selected_cities, columns=selected_cities)
    st.dataframe(distances_df)

    st.title('Visualisation des villes sélectionnées sur la map')
    map_center = selected_cities_data[selected_cities[0]]
    mymap = folium.Map(location=map_center, zoom_start=5)

    # Highlight selected cities on the map
    for city in selected_cities:
        folium.Marker(location=selected_cities_data[city], popup=city, icon=folium.Icon(color='red')).add_to(mymap)

    # Save the map as an HTML file
    map_html = mymap._repr_html_()

    # Display the Folium map in Streamlit
    st.components.v1.html(map_html, width=800, height=600)
    
    
    #Définition du model
    
    button = st.button('Calculer le plus court chemin')
    if button:
        villes=range(len(selected_cities))

        model=Model('TSP')
        ## Variable xij
        x=model.binary_var_matrix(keys1=villes,keys2=villes,name='x')
        ## Varible ui
        u=model.integer_var_list(keys=villes, lb=1, ub=len(selected_cities),name='u')

        model.minimize(model.sum(distances[i,j] * x[i,j] for i in villes for j in villes))

        for i in villes:
            model.add_constraint(model.sum(x[i , j] for j in villes)==1)
            model.add_constraint(model.sum(x[j , i] for j in villes)==1)

        for i in villes:
            for j in villes:
                if j != 0:
                    model.add_constraint(u[i]-u[j]+len(selected_cities) * x[i,j]<=len(selected_cities)-1)

        solution = model.solve(log_output=True)
        # Get the variable name with a non-zero value
        variables_with_value_1 = [var_name for var_name, var_value in solution.iter_var_values() if var_value == 1]

        l=[]

        for var in variables_with_value_1:
          # Split the string based on the underscore character
          if str(var).startswith("x"):
            split_numbers = str(var).split('_')

            # Extract the numerical values and convert them to integers
            number1 = int(split_numbers[1])
            number2 = int(split_numbers[2])

           # Create a tuple
            result_tuple = (number1, number2)
            l.append(result_tuple)

        G = nx.Graph()

        # Example list of tuples representing edges
        edges_list = l

        # Create a non directed graph
        G = nx.Graph()

        # Add edges from the list
        G.add_edges_from(edges_list)
        
        node_mapping = {}
            
        for index, city in enumerate(selected_cities):
            if index == 0:
                city = city + " (ville de départ)"
            node_mapping [index] = city
        
        G = nx.relabel_nodes(G, node_mapping)
        # Draw the graph
        pos = nx.spring_layout(G)  # You can use different layout algorithms


        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=5.5)
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict, font_size=5)
        st.pyplot(fig)
        # st.balloons()
        
        st.subheader('Solution de la fonction objective (distance minimale totale)')
        st.write(solution.get_objective_value())
    