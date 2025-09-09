# pipeline.py
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add
import pickle
import os
import pandas as pd
from sqlalchemy import create_engine
from router_agent import agent_2
from agent import graph_final
from agent_helper import chain_filter_extractor, chain_query_extractor, chain_query_validator
from fuzzy_match import call_match


# ------------------ Data Store ------------------
d_store = {
    "dim": [
        "brand_master",
        "cost_center_hierarchy",
        "cost_element_hierarchy",
        "functional_area_hierarchy",
        "functional_area_metric_map",
        "key_figure_metric_map",
        "profit_center_hierarchy",
    ],
    "sales": ["sales_data"],
    "expense": ["income_expense_reporting"],
}

with open("kb.pkl", "rb") as f:
    loaded_dict = pickle.load(f)

# ------------------ DB Config ------------------
# pipeline.py
import streamlit as st
from sqlalchemy import create_engine
import pickle
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# ------------------ Load Secrets ------------------
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_HOST = st.secrets["DB_HOST"]
DB_NAME = st.secrets["DBBASE"]
DB_PORT = st.secrets.get("DB_PORT", 5432)

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
)



# ------------------ Helpers ------------------
def remove_duplicates(f):
    """Flatten and deduplicate extracted columns across agents."""
    s = set()
    final = []
    for k, v in f.items():
        if k in ("dim_out", "sales_out", "expense_out"):
            for item in v["column_extract"]:
                key = tuple(item)
                if key not in s:
                    final.append(item)
                    s.add(key)
    return final


# ------------------ State ------------------
class FinalState(TypedDict):
    user_query: str
    router_out: list[str]
    dim_out: str
    sales_out: str
    expense_out: str
    filtered_col: str
    filter_extractor: list[str]
    fuzz_match: list[str]
    sql_query: str
    final_query: str


# ------------------ Nodes ------------------
def router(state: FinalState):
    q = state["user_query"]
    o = agent_2(q)
    return {"router_out": o}


def route_request(state: FinalState):
    routes = state["router_out"]
    print("➡️ Routed request to " + str(routes) + " agents")
    return routes


def dim(state: FinalState):
    q = state["user_query"]
    print("📊 Extracting relevant tables and columns from dim agent...")
    sub = graph_final.invoke({"user_query": q, "table_lst": d_store["dim"]})
    return {"dim_out": sub}


def sales(state: FinalState):
    q = state["user_query"]
    print("💰 Extracting relevant tables and columns from sales agent...")
    sub = graph_final.invoke({"user_query": q, "table_lst": d_store["sales"]})
    return {"sales_out": sub}


def expense(state: FinalState):
    q = state["user_query"]
    print("📉 Extracting relevant tables and columns from expense agent...")
    sub = graph_final.invoke({"user_query": q, "table_lst": d_store["expense"]})
    return {"expense_out": sub}


def filter_check(state: FinalState):
    q = state["user_query"]
    f = {}
    for key in ["dim_out", "sales_out", "expense_out"]:
        if key in state:
            f[key] = state.get(key)
    col_details = remove_duplicates(f)
    print("🔍 Checking the need for filter...")
    response = chain_filter_extractor.invoke({"columns": str(col_details), "query": q}).replace("```", "")
    return {"filter_extractor": eval(response), "filtered_col": str(col_details)}


def fuzz_match_node(state: FinalState):
    val = state["filter_extractor"]
    print("🧩 Matching filters with fuzzy logic...")
    lst = call_match(val)
    return {"fuzz_match": lst}


def filter_condition(state: FinalState):
    return "no" if len(state["filter_extractor"]) == 1 else "yes"


def query_generation(state: FinalState):
    q = state["user_query"]
    tab_cols = state["filtered_col"]
    filters = state.get("fuzz_match", "")
    print("🛠️ Generating SQL query...")
    final_query = chain_query_extractor.invoke({"columns": tab_cols, "query": q, "filters": filters})
    return {"sql_query": final_query}


def query_validation(state: FinalState):
    print("✅ Validating and finalizing SQL query...")
    o = chain_query_validator.invoke(
        {
            "columns": state["filtered_col"],
            "query": state["user_query"],
            "filters": state.get("fuzz_match"),
            "sql_query": state["sql_query"],
        }
    )
    return {"final_query": o}


# ------------------ Graph Builder ------------------
def build_graph():
    builder = StateGraph(FinalState)
    builder.add_node("router", router)
    builder.add_node("dim", dim)
    builder.add_node("sales", sales)
    builder.add_node("expense", expense)
    builder.add_node("filter_check", filter_check)
    builder.add_node("fuzz_filter", fuzz_match_node)
    builder.add_node("query_generator", query_generation)
    builder.add_node("query_validation", query_validation)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", route_request, ["dim", "sales", "expense"])
    builder.add_edge("dim", "filter_check")
    builder.add_edge("sales", "filter_check")
    builder.add_edge("expense", "filter_check")
    builder.add_conditional_edges("filter_check", filter_condition, {"no": "query_generator", "yes": "fuzz_filter"})
    builder.add_edge("fuzz_filter", "query_generator")
    builder.add_edge("query_generator", "query_validation")
    builder.add_edge("query_validation", END)

    return builder.compile()


graph_main = build_graph()
