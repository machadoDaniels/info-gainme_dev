import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

base = "https://raw.githubusercontent.com/dr5hn/countries-states-cities-database/master"

regions   = pd.read_json(f"{base}/json/regions.json")
subregions = pd.read_json(f"{base}/json/subregions.json")
countries = pd.read_json(f"{base}/json/countries.json")
states    = pd.read_json(f"{base}/json/states.json")
cities    = pd.read_json(f"{base}/json/cities.json")


# Copiar e normalizar tipos numéricos de IDs
countries2 = countries.copy()
for col in ["id", "region_id", "subregion_id"]:
    countries2[col] = pd.to_numeric(countries2[col], errors="coerce").astype("Int64")

states2 = states.copy()
for col in ["id", "country_id", "parent_id", "level"]:
    states2[col] = pd.to_numeric(states2[col], errors="coerce").astype("Int64")

cities2 = cities.copy()
for col in ["id", "state_id", "country_id"]:
    cities2[col] = pd.to_numeric(cities2[col], errors="coerce").astype("Int64")

regions2 = regions.copy()
regions2["id"] = pd.to_numeric(regions2["id"], errors="coerce").astype("Int64")

subregions2 = subregions.copy()
for col in ["id", "region_id"]:
    subregions2[col] = pd.to_numeric(subregions2[col], errors="coerce").astype("Int64")

# Selecionar e renomear colunas para evitar colisões
cities_sel = cities2.rename(columns={
    "id": "city_id",
    "name": "city_name",
    "latitude": "city_latitude",
    "longitude": "city_longitude",
    "timezone": "city_timezone",
    "wikiDataId": "city_wikiDataId",
}).drop(columns=["state_name", "country_name"], errors="ignore")

states_sel = states2.rename(columns={
    "id": "state_id_ref",
    "name": "state_name",
    "iso2": "state_iso2",
    "iso3166_2": "state_iso3166_2",
    "fips_code": "state_fips_code",
    "type": "state_type",
    "level": "state_level",
    "parent_id": "state_parent_id",
    "latitude": "state_latitude",
    "longitude": "state_longitude",
    "timezone": "state_timezone",
    # "country_id": "country_id_ref",  # não carregar para evitar duplicação
})[[
    "state_id_ref",
    "state_name",
    "state_iso2",
    "state_iso3166_2",
    "state_fips_code",
    "state_type",
    "state_level",
    "state_parent_id",
    "state_latitude",
    "state_longitude",
    "state_timezone",
]]

countries_sel = countries2.rename(columns={
    "id": "country_id_ref",
    "name": "country_name",
    "iso2": "country_iso2",
    "iso3": "country_iso3",
    "phonecode": "country_phonecode",
    "capital": "country_capital",
    "latitude": "country_latitude",
    "longitude": "country_longitude",
})[[
    "country_id_ref",
    "country_name",
    "country_iso2",
    "country_iso3",
    "numeric_code",
    "country_phonecode",
    "country_capital",
    "currency",
    "currency_name",
    "currency_symbol",
    "tld",
    "native",
    "nationality",
    "region",
    "region_id",
    "subregion",
    "subregion_id",
    "timezones",
    "translations",
    "country_latitude",
    "country_longitude",
    "emoji",
    "emojiU",
]]

regions_sel = regions2.rename(columns={"id": "region_id_ref", "name": "region_name"})[[
    "region_id_ref",
    "region_name",
]]

subregions_sel = subregions2.rename(columns={"id": "subregion_id_ref", "name": "subregion_name"})[[
    "subregion_id_ref",
    "subregion_name",
    # não carregar region_id para evitar duplicação
]]

# Merges (cidade -> estado -> país -> região -> sub-região)
world_flat = cities_sel.merge(states_sel, left_on="state_id", right_on="state_id_ref", how="left")
world_flat = world_flat.merge(countries_sel, left_on="country_id", right_on="country_id_ref", how="left")
world_flat = world_flat.merge(regions_sel, left_on="region_id", right_on="region_id_ref", how="left")
world_flat = world_flat.merge(subregions_sel, left_on="subregion_id", right_on="subregion_id_ref", how="left")

# Limpeza de chaves auxiliares
for c in ["state_id_ref", "country_id_ref", "region_id_ref", "subregion_id_ref"]:
    if c in world_flat.columns:
        world_flat.drop(columns=[c], inplace=True)

# Remover quaisquer sufixos _x/_y restantes consolidando os valores
suffix_bases = {}
for col in list(world_flat.columns):
    if col.endswith("_x") or col.endswith("_y"):
        base = col[:-2]
        suffix_bases.setdefault(base, set()).add(col)

for base, cols in suffix_bases.items():
    x = f"{base}_x"
    y = f"{base}_y"
    if x in world_flat.columns and y in world_flat.columns:
        world_flat[x] = world_flat[x].combine_first(world_flat[y])
        world_flat.drop(columns=[y], inplace=True)
        world_flat.rename(columns={x: base}, inplace=True)
    else:
        only = list(cols)[0]
        world_flat.rename(columns={only: base}, inplace=True)

# Selecionar apenas as colunas necessárias
world_flat_clean = world_flat[[
    "city_id",
    "city_name",
    "state_id",
    "state_name",
    "country_id",
    "country_name",
    "region_id",
    "region_name",
    "subregion_id",
    "subregion_name",
]]

# Filtering top pop cities
cities_pops = pd.read_csv("/Users/daniels/Documents/AKCIT-RL/clary_quest/data/cities_pops.csv")

# Merge cities_pops with world_flat_clean on world_flat_clean.city_name, world_flat_clean.country_name and cities_pops.city_name, cities_pops.country_name
top_40_pop_cities = world_flat_clean.merge(
    cities_pops[["city_name", "country_name", "population_2025"]], 
    on=["city_name", "country_name"], 
    how="inner"
    ).sort_values(by="population_2025", ascending=False)

# for i, row in cities_pops.iterrows():
#     if row["city_name"] not in top_40_pop_cities["city_name"].values:
#         print("City:", row["city_name"], "| Country:", row["country_name"])

top_20_pop_cities = top_40_pop_cities.head(20)

top_10_pop_cities = top_40_pop_cities.head(10)

# Check the number of cities
assert len(top_40_pop_cities) == 40
assert len(top_20_pop_cities) == 20
assert len(top_10_pop_cities) == 10

# Salva os dados em formato csv
out_parquet = "/Users/daniels/Documents/AKCIT-RL/clary_quest/data/world_flat.csv"
world_flat.to_csv(out_parquet, index=False)
print(f"Dados salvos em: {out_parquet}")

out_parquet = "/Users/daniels/Documents/AKCIT-RL/clary_quest/data/world_flat_clean.csv"
world_flat_clean.to_csv(out_parquet, index=False)
print(f"Dados salvos em: {out_parquet}")


out_parquet = "/Users/daniels/Documents/AKCIT-RL/clary_quest/data/top_40_pop_cities.csv"
top_40_pop_cities.to_csv(out_parquet, index=False)
print(f"Dados salvos em: {out_parquet}")

out_parquet = "/Users/daniels/Documents/AKCIT-RL/clary_quest/data/top_20_pop_cities.csv"
top_20_pop_cities.to_csv(out_parquet, index=False)
print(f"Dados salvos em: {out_parquet}")

out_parquet = "/Users/daniels/Documents/AKCIT-RL/clary_quest/data/top_10_pop_cities.csv"
top_10_pop_cities.to_csv(out_parquet, index=False)
print(f"Dados salvos em: {out_parquet}")