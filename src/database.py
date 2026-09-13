import duckdb
import pandas as pd

df = pd.read_csv("research/data.csv")

conn = duckdb.connect("my_data.db")
conn.register("data", df)

RESULTS = {}