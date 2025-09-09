import pandas as pd
from sqlalchemy import create_engine,  text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import Integer, Float, String
from rapidfuzz import process, fuzz
import streamlit as st
from sqlalchemy import create_engine,text
import os
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_HOST = st.secrets["DB_HOST"]
DB_NAME = st.secrets["DBBASE"]
DB_PORT = st.secrets.get("DB_PORT", 5432)

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require")


def get_best_fuzzy_match(input_value, choices):

    match, score, _ = process.extractOne(input_value, choices, scorer=fuzz.token_set_ratio)
    return match, score


def get_values(table_name: str, column_name: str, engine) -> list:
    """
    Fetch distinct non-null values from a given table column.
    Handles errors gracefully and rolls back any failed transactions.
    """
    import sqlalchemy

    # SQL query to get distinct values
    query = f"SELECT DISTINCT {column_name} FROM {table_name}"

    try:
        # Use a transaction block for safety
        with engine.begin() as conn:
            df = pd.read_sql(query, con=conn)

        # Convert to list if you want raw values
        unique_values = df[column_name].dropna().tolist()
        return unique_values

    except sqlalchemy.exc.SQLAlchemyError as e:
        # Rollback is automatic in `engine.begin()` on exceptions
        print(f"Error fetching values from {table_name}.{column_name}: {e}")
        return []

def call_match(val):
    final = []
    for lst in val[1:]:
        table = lst[0]
        column = lst[1]
        str_lst = [i.strip() for i in lst[2].split(',')]

        unq_col_val = get_values(table, column,engine)
        unq_col_val = [str(i) for i in unq_col_val]

        for subval in str_lst:
            best_match, score = get_best_fuzzy_match(subval, unq_col_val)
            final.append(["table name:"+table, "column_name:"+column, "filter_value:"+best_match])

    return final
