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
DB_PASSWORD = os.getenv("DB_PASSWORD", "camera_brain_password_change_me")
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
'class_name' includes: person, car, truck, bus, bicycle, bird, cat, dog, etc. (80 COCO classes)
"""

SYSTEM_PROMPT = f"""You are a SQL query generator for a camera surveillance database.
{SCHEMA_CONTEXT}

Convert the user's natural language question into a PostgreSQL SELECT query.
CRITICAL: Return ONLY the raw SQL query - no explanations, no markdown, no text before or after.

Rules:
- Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP)
- Use DATE(detected_at) for date filtering
- Use LIMIT to restrict results (max 100)
- Start with SELECT keyword

Examples:
Q: "Show me all people detected today"
A: SELECT * FROM observations WHERE class_name='person' AND DATE(detected_at)=CURRENT_DATE LIMIT 50

Q: "How many cars were detected yesterday?"
A: SELECT count(*) FROM observations WHERE class_name IN ('car', 'truck', 'bus') AND DATE(detected_at)=CURRENT_DATE - INTERVAL '1 day'

Q: "What detections occurred on camera cam4?"
A: SELECT * FROM observations WHERE camera_id='cam4' ORDER BY detected_at DESC LIMIT 50
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
                    {"role": "user", "content": "Question: " + question}
                ],
                "max_tokens": 200,
                "temperature": 0.01,  # Very low for deterministic SQL output
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        raw_content = result["choices"][0]["message"]["content"].strip()

        app.logger.info(f"LLM raw response: {raw_content[:200]}")

        # Extract SQL from response - handle various formats
        # Remove markdown code blocks if present
        sql = raw_content.replace("```sql", "").replace("```", "").strip()

        # Remove common prefixes that LLMs add
        prefixes_to_remove = [
            "Here's the query:",
            "Here is the query:",
            "Query:",
            "The query:",
            "SQL:",
            "The SQL query:"
        ]
        for prefix in prefixes_to_remove:
            if sql.upper().startswith(prefix.upper()):
                sql = sql[len(prefix):].strip()

        # Find SELECT keyword and extract from there
        select_idx = sql.upper().find("SELECT")
        if select_idx >= 0:
            sql = sql[select_idx:]
            # Remove anything after the statement (explanations, etc.)
            if ";" in sql:
                sql = sql.split(";")[0].strip()
            # Remove trailing explanations
            if "\n" in sql:
                sql = sql.split("\n")[0].strip()
            app.logger.info(f"Extracted SQL: {sql}")
            return sql

        # If no SELECT found, log and return None
        app.logger.warning(f"No SELECT found in LLM response: {raw_content}")
        return None
    except requests.RequestException as e:
        app.logger.error(f"LLM request failed: {e}")
        return None


def validate_sql(sql: str) -> tuple[bool, str]:
    """Validate that the SQL is safe and SELECT-only."""
    if not sql:
        return False, "Empty query"

    sql_upper = sql.upper().strip()

    # Only allow SELECT queries - must start with SELECT followed by space or *
    if not (sql_upper.startswith("SELECT ") or sql_upper.startswith("SELECT*")):
        # Check if it looks like English text instead of SQL
        if "FOR YOU" in sql_upper or "TO RETRIEVE" in sql_upper or "QUERY" in sql_upper:
            return False, "LLM generated explanation instead of SQL - try rephrasing your question"
        return False, f"Invalid SQL format (must start with SELECT): {sql[:50]}..."

    # Block dangerous operations
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "ATTACH", "DETACH"]
    for keyword in dangerous:
        if keyword in sql_upper:
            return False, f"Dangerous keyword: {keyword}"

    # Basic syntax validation - must have FROM clause for non-COUNT queries
    if "FROM" not in sql_upper and "COUNT" not in sql_upper:
        return False, "Missing FROM clause in query"

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


def generate_answer(question: str, results: list, sql: str = None) -> str:
    """Use LLM to generate a natural language answer from query results.

    Provides accurate, specific answers based on actual data - not vague summaries.
    """
    if not results:
        return "No matching observations found for your query."

    # For COUNT queries, provide a direct answer
    if sql and 'count(*)' in sql.lower():
        count = results[0].get('count', len(results)) if results else 0
        return f"Found {count} observations matching your query."

    # For aggregation queries (GROUP BY), summarize the distribution
    if sql and 'group by' in sql.lower():
        summary_parts = []
        for row in results[:5]:  # Top 5 results
            class_name = row.get('class_name', row.get('class', 'unknown'))
            cnt = row.get('count', row.get('total', 0))
            if cnt:
                summary_parts.append(f"{cnt} {class_name}")

        if summary_parts:
            return f"Detected: {', '.join(summary_parts)}."
        return f"Found {len(results)} categories of observations."

    # For detail queries, provide a concise summary
    try:
        # Extract key patterns from results
        class_counts = {}
        time_range = None
        cameras = set()

        for row in results:
            cn = row.get('class_name', 'unknown')
            class_counts[cn] = class_counts.get(cn, 0) + 1
            if 'detected_at' in row and row['detected_at']:
                cameras.add(row.get('camera_id', 'unknown'))

        # Build specific answer
        answer_parts = []
        total = len(results)

        if class_counts:
            top_classes = sorted(class_counts.items(), key=lambda x: -x[1])[:3]
            class_summary = ', '.join(f"{cnt} {name}" for name, cnt in top_classes)
            answer_parts.append(f"Found {total} observations: {class_summary}")

        if len(class_counts) > 3:
            answer_parts.append(f"across {len(class_counts)} different classes")

        if cameras:
            answer_parts.append(f"on {len(cameras)} camera(s)")

        return '. '.join(answer_parts) + '.'

    except Exception as e:
        app.logger.error(f"Answer generation error: {e}")
        return f"Query returned {len(results)} results."


def generate_answer_v2(question: str, results: list) -> str:
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

Provide a clear, concise answer summarizing what was found. If there are many results, mention the count. Be specific about numbers and categories."""

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

    app.logger.info(f"=== Chat Request ===")
    app.logger.info(f"Question: {question}")

    # Step 1: Generate SQL from natural language
    sql = generate_sql_query(question)
    if not sql:
        app.logger.warning(f"LLM failed to generate SQL for question: {question}")
        return jsonify({
            "error": "Failed to generate SQL query - the AI couldn't understand the question. Try rephrasing.",
            "question": question,
            "debug": {
                "llm_response": "No SELECT statement found in LLM response"
            }
        })

    app.logger.info(f"Generated SQL: {sql}")

    # Step 2: Validate SQL
    valid, error = validate_sql(sql)
    if not valid:
        app.logger.warning(f"SQL validation failed: {error}")
        app.logger.warning(f"Invalid SQL: {sql}")
        return jsonify({
            "error": f"Invalid query: {error}",
            "sql": sql,
            "question": question
        }), 400

    # Step 3: Execute query
    try:
        columns, rows = execute_query(sql)
        app.logger.info(f"Query returned {len(rows)} rows")
    except Exception as e:
        app.logger.error(f"=== Query Execution Failed ===")
        app.logger.error(f"Question: {question}")
        app.logger.error(f"Generated SQL: {sql}")
        app.logger.error(f"Error: {e}")
        return jsonify({
            "error": f"Query failed: {str(e)}",
            "sql": sql,
            "question": question,
            "debug": {
                "error_type": type(e).__name__,
                "error_detail": str(e)
            }
        }), 500

    # Step 4: Generate natural language answer
    answer = generate_answer(question, rows, sql)
    app.logger.info(f"Answer: {answer}")

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
