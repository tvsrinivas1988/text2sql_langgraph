import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pipeline import graph_main, engine  # reuse pipeline + engine
import io
# ---------------- Utility ----------------
import re

def extract_sql_from_output(output: str) -> str:
    """
    Extracts only the SQL query from an LLM response.
    Handles cases with ```sql fences or extra text.
    """
    if not output:
        return ""

    # Look for SQL inside ```sql ... ```
    match = re.search(r"```sql\s+(.*?)```", output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Look for SQL inside generic ``` ... ```
    match = re.search(r"```(.*?)```", output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no fences, return the raw output (assume it's already SQL)
    return output.strip()

def run_sql(query: str):
    """
    Run SQL query safely with proper error handling.
    Ensures rollback and connection closure in case of errors.
    Returns either a DataFrame or an error message.
    """
    conn = None
    try:
        conn = engine.connect()
        trans = conn.begin()  # start transaction explicitly
        df = pd.read_sql(text(query), conn)
        trans.commit()
        return df
    except SQLAlchemyError as e:
        if conn:
            try:
                trans.rollback()
            except Exception:
                pass  # ignore rollback failure
        return f"❌ SQLAlchemy error: {str(e.__cause__ or e)}"
    except Exception as e:
        if conn:
            try:
                trans.rollback()
            except Exception:
                pass
        return f"❌ Unexpected error: {str(e)}"
    finally:
        if conn:
            conn.close()

st.set_page_config(page_title="LangGraph Text2SQL", layout="wide")
st.title("🧠 LangGraph + OpenAI based Text2SQL Agent")

user_q = st.text_input("💬 Enter your question:")

if st.button("Run Query") and user_q:
    with st.spinner("🔎 Processing..."):
        try:
            result = graph_main.invoke({"user_query": user_q})

            st.write("### 🔍 Router Output")
            st.json(result.get("router_out", {}))

            st.write("### 📝  SQL Query")
            sql_query1 = result.get("sql_query", "No query generated")
            st.code(sql_query1, language="sql")

            st.write("### 📝 Final SQL Query")
            sql_query = result.get("final_query", "No query generated")
            st.code(sql_query, language="sql")

            if sql_query and "SELECT" in sql_query.upper():
                sql_query_new = extract_sql_from_output(sql_query)
                df = run_sql(sql_query_new)

                if isinstance(df, pd.DataFrame):
                    st.success("✅ Query executed successfully")
                    st.dataframe(df)

                    # --- Download functionality ---
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="query_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(df)
            else:
                st.warning("⚠️ No valid SQL query was generated.")

        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")
