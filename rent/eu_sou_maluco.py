from pathlib import Path
import pandas as pd
import requests
from math import radians, sin, cos, sqrt, atan2

# Caminho e leitura dos dados
DATA_BASE_PATH = Path(__file__).parent / "data"
rent_path = DATA_BASE_PATH / "apartments_for_rent_classified_10K.csv"
rent = pd.read_csv(rent_path, sep=";", encoding="latin1")
df_lenght = 1

# Função para calcular distância entre dois pontos (Haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Raio da Terra em metros
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Função para montar a query da Overpass com node, way e relation
def build_overpass_query(locations, radius=1000):
    query_parts = []
    for lat, lon in locations:
        query_parts.append(f"""
          node["amenity"~"restaurant|cafe|bar"](around:{radius},{lat},{lon});
          way["amenity"~"restaurant|cafe|bar"](around:{radius},{lat},{lon});
          relation["amenity"~"restaurant|cafe|bar"](around:{radius},{lat},{lon});
        """)
    
    full_query = f"""
    [out:json];
    (
    {''.join(query_parts)}
    );
    out body;
    """

    return full_query

# Pega os 100 primeiros apartamentos e extrai coordenadas
sample_rent = rent.iloc[198:199]
coordinates = list(zip(sample_rent['latitude'], sample_rent['longitude']))

# Monta e envia a query
query = build_overpass_query(coordinates)
url = "https://overpass-api.de/api/interpreter"
response = requests.post(url, data={"data": query})

# Inicializa o contador para cada tipo de amenity por ponto
poi_counts_by_amenity = [dict() for _ in range(df_lenght)]  # Cada apartamento vai ter seu dicionário de contagem

# Tipos de amenidades que estamos buscando
amenities_list = ["restaurant", "cafe", "bar"]

# Realiza a requisição e processa a resposta
if response.status_code == 200:
    data = response.json()
    for element in data["elements"]:
        el_lat = element.get("lat") or element.get("center", {}).get("lat")
        el_lon = element.get("lon") or element.get("center", {}).get("lon")
        if el_lat is None or el_lon is None:
            continue
        # Associa o POI ao ponto mais próximo
        min_index = None
        min_dist = float("inf")
        for i, (apt_lat, apt_lon) in enumerate(coordinates):
            dist = haversine(el_lat, el_lon, apt_lat, apt_lon)
            if dist < min_dist:
                min_dist = dist
                min_index = i
        if min_dist <= 1000:
            amenity = element["tags"].get("amenity")
            if amenity in amenities_list:
                # Atualiza a contagem para aquele amenity específico
                poi_counts_by_amenity[min_index][amenity] = poi_counts_by_amenity[min_index].get(amenity, 0) + 1
else:
    print("Erro na requisição:", response.status_code)

# Cria as colunas no DataFrame para cada tipo de amenity
for amenity in amenities_list:
    rent[amenity] = 0  # Inicializa as colunas com zero

# Preenche as colunas com as contagens dos amenities
# Cria um novo DataFrame com latitude e longitude
amenities_df = pd.DataFrame({
    'latitude': sample_rent['latitude'].values,
    'longitude': sample_rent['longitude'].values,
})

# Adiciona as colunas de amenidades ao novo DataFrame
for amenity in amenities_list:
    amenities_df[amenity] = [poi_counts_by_amenity[i].get(amenity, 0) for i in range(df_lenght)]

# Exibe o novo DataFrame
print(amenities_df)
amenities_df.to_csv('result2.csv')