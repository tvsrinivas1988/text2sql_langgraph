from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
import pickle
import re

from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, TypedDict, Annotated
from operator import add

from agent_helper import *
from IPython.display import Image

from dotenv import load_dotenv

import re
import ast  # safer than eval for simple Python literals


load_dotenv()

with open('kb.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

d_store = {
    "dim" : ['brand_master', 'cost_center_hierarchy','cost_element_hierarchy','functional_area_hierarchy','functional_area_metric_map','key_figure_metric_map','profit_center_hierarchy'],
    "sales" : ['sales_data'],
    "expense": ["income_expense_reporting"]
}

class overallstate(TypedDict):
    user_query: str
    table_lst: list[str]
    table_extract : Annotated[list[str], add]
    column_extract : Annotated[list[str], add]



def agent_subquestion(q, v):
    """
    Invoke the LLM chain for a subquestion, extract the resulting list of lists, and return as Python object.
    """
    response = chain_subquestion.invoke({"tables": v, "user_query": q}).replace('```', '')
    
    # Regex to find the list-of-lists structure
    match = re.search(r"\[\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\]", response, re.DOTALL)
    
    if match:
        matched_str = match.group(0)
        try:
            # Convert the string to a Python list safely
            result_list = ast.literal_eval(matched_str)
            return result_list
        except Exception as e:
            print("Error parsing LLM output:", e)
            return []
    else:
        print("No match found in LLM output")
        return []


def solve_subquestion(q, lst):
    """
    For a given question `q` and list of table names `lst`,
    return the LLM-extracted subquestions as a Python list.
    """
    final = []
    for tab in lst:
        desc = loaded_dict[tab]['table_description']   # get table description
        final.append([tab, desc])

    # Create dictionary mapping table -> description
    result_dict = {item[0]: item[1] for item in final}

    # Call agent_subquestion and get list directly
    subquestion_list = agent_subquestion(q, str(result_dict))
    
    return subquestion_list




def agent_column_selection(main_q, q, c):
    """
    Call the column extraction LLM chain, and safely return list-of-lists.
    """
    response = chain_column_extractor.invoke({
        "columns": c,
        "query": q,
        "main_question": main_q
    }).replace('```', '')

    # Regex to extract list-of-lists pattern
    match = re.search(r"\[\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\]", response, re.DOTALL)
    if match:
        matched_str = match.group(0)
        try:
            result_list = ast.literal_eval(matched_str)  # safe parsing
            return result_list
        except Exception as e:
            print("Error parsing LLM output:", e)
            return [[]]   # fallback empty list
    else:
        return [[]]  # fallback empty list



def solve_column_selection(main_q, list_sub):
    final_col = []
    inter = []
    for tab in list_sub:
        if len(tab)==0:
            continue
        table_name = tab[1]
        question = tab[0]
        columns = loaded_dict[table_name]["columns"]
        out_column = agent_column_selection(main_q, question, str(columns))
        trans_col = (out_column)

        for col_selec in trans_col:
            new_col = ["name of table:" + table_name] + col_selec
            inter.append(new_col)
        final_col.extend(inter)
    return final_col


def sq_node(state: overallstate):
    q = state['user_query']
    lst = state['table_lst']
    o = solve_subquestion(q, lst)
    
    return {"table_extract": (o)}

def column_node(state: overallstate):
    subq = state['table_extract']
    mq = state['user_query']
    
    o = solve_column_selection(mq, subq)
    return {"column_extract": o}


builder_final = StateGraph(overallstate)
builder_final.add_node("subquestion", sq_node)
builder_final.add_node("column_e", column_node)

builder_final.add_edge(START, "subquestion")
builder_final.add_edge("subquestion", "column_e")

builder_final.add_edge("column_e", END)
graph_final = builder_final.compile()

