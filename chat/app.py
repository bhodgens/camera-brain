#!/usr/bin/env python3
"""
Camera Brain Chat - Minimalist web interface for querying observations.
Uses LFM2.5-1.2B-Instruct for natural language to SQL translation.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify

try:
    import psycopg2
    DB_TYPE = "postgresql"
except ImportError:
    import sqlite3
    DB_TYPE = "sqlite"

app = Flask(__name__)

# Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "camera_brain")
DB_USER = os.getenv("DB_USER", "camera_brain")
DB_PASSWORD = os.getenv("DB_PASSWORD", "camera_brain")
DB_PATH = os.getenv("DB_PATH", "/home/camera-brain/camera-brain.db")  # SQLite fallback
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8889")
MODEL_PATH = os.getenv("MODEL_PATH", "/home/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf")

# Schema awareness for the LLM
SCHEMA_CONTEXT = """
Database schema:
- observations(id, camera_id, detected_at, type, class_name, confidence, bbox, crop_path)
- cameras(id, name, rtsp_url, location)

The 'bbox' column is JSON: [x1, y1, x2, y2]
'detected_at' is a timestamp
'class_name' includes: person, car, truck, bus, bicycle, etc.
"""

SYSTEM_PROMPT = f"""You are a SQL query generator for a camera surveillance database.
{SCHEMA_CONTEXT}

Convert the user's natural language question into a PostgreSQL SELECT query.
- Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP)
- Use DATE() for date filtering: DATE(detected_at) = '2026-07-08'
- Use LIMIT to restrict results (max 100)
- Return ONLY the SQL query, no explanation

Examples:
Q: "Show me all people detected today"
A: SELECT * FROM observations WHERE class_name='person' AND DATE(detected_at)='2026-07-08' LIMIT 50

Q: "How many cars were detected yesterday?"
A: SELECT count(*) FROM observations WHERE class_name IN ('car', 'truck', 'bus') AND DATE(detected_at)='2026-07-07'

Q: "What detections occurred on camera cam4?"
A: SELECT * FROM observations WHERE camera_id='cam4' ORDER BY detected_at DESC LIMIT 50

Q: "Show me high confidence detections above 0.8"
A: SELECT * FROM observations WHERE confidence > 0.8 ORDER BY detected_at DESC LIMIT 50
"""


def get_db_connection():
    """Connect to database (PostgreSQL or SQLite)."""
    if DB_TYPE == "postgresql":
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.autocommit = True
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def generate_sql_query(question: str) -> str:
    """Use LFM 1.2B to convert natural language to SQL."""
    try:
        response = requests.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ],
                "max_tokens": 256,
                "temperature": 0.1,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        sql = result["choices"][0]["message"]["content"].strip()

        # Clean up the response - extract just the SQL
        if "SELECT" in sql.upper():
            sql = sql.upper().split("SELECT", 1)[1]
            sql = "SELECT " + sql.split(";")[0].strip()
        return sql
    except requests.RequestException as e:
        app.logger.error(f"LLM request failed: {e}")
        return None


def validate_sql(sql: str) -> tuple[bool, str]:
    """Validate that the SQL is safe and SELECT-only."""
    if not sql:
        return False, "Empty query"

    sql_upper = sql.upper().strip()

    # Only allow SELECT queries
    if not sql_upper.startswith("SELECT"):
        return False, "Only SELECT queries allowed"

    # Block dangerous operations
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "ATTACH", "DETACH"]
    for keyword in dangerous:
        if keyword in sql_upper:
            return False, f"Dangerous keyword: {keyword}"

    return True, ""


def execute_query(sql: str) -> tuple[list, list]:
    """Execute SQL and return (columns, rows)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()

        # Convert timestamps to strings
        for row in rows:
            for key, val in row.items():
                if isinstance(val, bytes):
                    row[key] = val.decode('utf-8')
                elif isinstance(val, datetime):
                    row[key] = val.isoformat()

        return columns, rows
    except Exception as e:
        app.logger.error(f"SQL error: {e}")
        raise


def generate_answer(question: str, results: list) -> str:
    """Use LLM to generate a natural language answer from query results."""
    if not results:
        return "No matching observations found."

    try:
        # Summarize results for the LLM
        summary = json.dumps(results[:10], indent=2)  # Limit context
        prompt = f"""Based on these database results, answer the user's question concisely.

Question: {question}

Results (showing first {len(results[:10])} of {len(results)}):
{summary}

Provide a clear, concise answer summarizing what was found. If there are many results, mention the count."""

        response = requests.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 512,
                "temperature": 0.3,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        app.logger.error(f"Answer generation failed: {e}")
        return f"Found {len(results)} results. (Answer generation failed: {e})"


@app.route("/")
def index():
    """Serve the chat interface."""
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat query."""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    app.logger.info(f"Question: {question}")

    # Step 1: Generate SQL from natural language
    sql = generate_sql_query(question)
    if not sql:
        return jsonify({
            "error": "Failed to generate SQL query",
            "question": question
        })

    app.logger.info(f"Generated SQL: {sql}")

    # Step 2: Validate SQL
    valid, error = validate_sql(sql)
    if not valid:
        return jsonify({"error": f"Invalid query: {error}"}), 400

    # Step 3: Execute query
    try:
        columns, rows = execute_query(sql)
    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}", "sql": sql}), 500

    # Step 4: Generate natural language answer
    answer = generate_answer(question, rows)

    return jsonify({
        "success": True,
        "question": question,
        "sql": sql,
        "answer": answer,
        "result_count": len(rows),
        "results": rows[:50]  # Limit returned results
    })


@app.route("/health")
def health():
    """Health check endpoint."""
    status = {"status": "ok", "llama_server": "unknown", "db": "unknown"}

    # Check Llama server
    try:
        resp = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=5)
        status["llama_server"] = "connected" if resp.status_code == 200 else "error"
    except:
        status["llama_server"] = "disconnected"

    # Check database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM observations")
        count = cursor.fetchone()[0]
        conn.close()
        status["db"] = f"connected ({count} observations)"
    except Exception as e:
        status["db"] = f"error ({str(e)})"

    return jsonify(status)


if __name__ == "__main__":
    # Port 80 requires CAP_NET_BIND_SERVICE capability (set by deploy script)
    port = int(os.getenv("PORT", 80))
    app.run(host="0.0.0.0", port=port, debug=False)
