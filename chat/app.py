#!/usr/bin/env python3
"""
Camera Brain Chat - Minimalist web interface for querying observations.
Uses LFM2.5-1.2B-Instruct for natural language to SQL translation.
"""

import os
import json
import re
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
DB_PATH = os.getenv("DB_PATH", "/home/camera-brain/camera-brain.db")
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8889")
MODEL_PATH = os.getenv("MODEL_PATH", "/home/camera-brain/models/LFM2.5-1.2B-Instruct.Q4_K_M.gguf")

# Schema awareness for the LLM
SCHEMA_CONTEXT = """
Database schema:
- observations(id, camera_id, detected_at, type, class_name, confidence, bbox, crop_path, color, vehicle_type, gender, gender_conf, age)
- cameras(id, name, rtsp_url, location, active, created_at)

Available fields: id, camera_id, detected_at, type, class_name, confidence, bbox, crop_path, color, vehicle_type, gender, gender_conf, age, name, rtsp_url, location, active, created_at
Available class_name values: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush (80 COCO classes)
Available vehicle_type values: sedan, SUV, truck, van, pickup, bus, motorcycle, bicycle
Available gender values: man, woman
Available color values: red, blue, green, yellow, white, black, silver, gray, orange, brown

NOT available: make, model (brand-specific), license_plate, speed, direction, emotion, clothing details, person identity
"""

SYSTEM_PROMPT = f"""You are a SQL query generator for a camera surveillance database.
{SCHEMA_CONTEXT}

Your task: Convert the user's natural language question into a PostgreSQL SELECT query.

IMPORTANT BEHAVIOR:
1. If the question asks about fields that DON'T exist (make, model, brand, license_plate, speed, emotion, clothing, etc.):
   - Do NOT generate SQL
   - Respond with: "I can search for observations by class (car, person, truck, etc.), camera location, time, and confidence level. However, I cannot identify [unavailable field] from the available data. Would you like me to search for [suggested alternative] instead?"

2. If the question is ambiguous or vague:
   - Do NOT generate SQL
   - Ask for clarification: "Could you clarify what you're looking for? I can search by class name (car, person, etc.), camera location, time period, or confidence level."

3. If the question CAN be translated to SQL:
   - Return ONLY the raw SQL query - no explanations, no markdown, no text before or after
   - Start with SELECT keyword
   - Use DATE(detected_at) for date filtering
   - Use LIMIT to restrict results (max 100)

Examples:
Q: "Show me all women detected today"
A: SELECT * FROM observations WHERE gender='woman' AND DATE(detected_at)=CURRENT_DATE LIMIT 50

Q: "What vehicle types were detected?"
A: SELECT vehicle_type, COUNT(*) FROM observations WHERE vehicle_type IS NOT NULL GROUP BY vehicle_type

Q: "Show me red cars"
A: SELECT * FROM observations WHERE class_name='car' AND color='red' LIMIT 50

Q: "How many SUVs were detected yesterday?"
A: SELECT COUNT(*) FROM observations WHERE vehicle_type='SUV' AND DATE(detected_at)=CURRENT_DATE - INTERVAL '1 day'

Q: "Show me all people detected today"
A: SELECT * FROM observations WHERE class_name='person' AND DATE(detected_at)=CURRENT_DATE LIMIT 50

Q: "What color cars were detected?"
A: I can search for observations by car class, but the system doesn't capture color information. Available data includes detection time, camera location, and confidence score. Would you like me to show you all car detections instead?

Q: "Tell me about the red trucks"
A: I can find truck detections, but color information isn't available in the system. Would you like to see all truck detections, or trucks from a specific camera or time?

Q: "How many cars were detected yesterday?"
A: SELECT count(*) FROM observations WHERE class_name IN ('car', 'truck', 'bus') AND DATE(detected_at)=CURRENT_DATE - INTERVAL '1 day'

Q: "What detections occurred on camera cam4?"
A: SELECT * FROM observations WHERE camera_id='cam4' ORDER BY detected_at DESC LIMIT 50

Q: "Show me fast moving objects"
A: I can search by class and time, but speed information isn't captured. Would you like to see recent detections across all classes, or filter by a specific class like car or person?
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


# Fields that are NOT available in the database
UNAVAILABLE_FIELDS = [
    # Vehicle specifics
    "make", "model", "brand", "manufacturer",
    "license_plate", "license", "plate", "registration",
    # Movement
    "speed", "velocity", "fast", "slow",
    "direction", "heading", "bearing",
    # Personal attributes (not captured)
    "emotion", "mood", "expression", "face",
    "clothing", "clothes", "shirt", "pants", "dress", "hat", "shoes",
    "race", "ethnicity",
    # Weapons
    "weapon", "knife", "gun", "sword",
    # Other
    "bag_type", "purse", "backpack_type"
]

# Phrases that indicate the LLM returned clarification instead of SQL
CLARIFICATION_PATTERNS = [
    r"i can", r"i cannot", r"can't", r"couldn't",
    r"would you like", r"would you like me to",
    r"available data", r"not available", r"isn't available", r"doesn't capture",
    r"could you clarify", r"please clarify",
    r"i can search", r"i can find",
    r"unavailable", r"not captured", r"not recorded"
]


def is_clarification_response(text: str) -> bool:
    """Check if the LLM response is a clarification message rather than SQL."""
    text_lower = text.lower()

    # Check for unavailable field mentions
    for field in UNAVAILABLE_FIELDS:
        if field in text_lower:
            return True

    # Check for clarification patterns
    for pattern in CLARIFICATION_PATTERNS:
        if re.search(pattern, text_lower):
            return True

    return False


def extract_unavailable_field(question: str) -> str:
    """Extract the unavailable field mentioned in the question."""
    question_lower = question.lower()
    for field in UNAVAILABLE_FIELDS:
        if field in question_lower:
            return field
    return "that attribute"


def get_suggested_alternative(field: str) -> str:
    """Suggest an alternative query based on the unavailable field."""
    field = field.lower()

    if field in ["color", "colors", "colour", "colours"]:
        return "all detections of that class"
    elif field in ["make", "model", "brand", "manufacturer"]:
        return "all vehicles of that class"
    elif field in ["speed", "velocity", "fast", "slow"]:
        return "recent detections sorted by time"
    elif field in ["direction", "heading", "bearing"]:
        return "detections from a specific camera"
    elif field in ["emotion", "mood", "expression", "face"]:
        return "person detections"
    elif field in ["clothing", "clothes", "shirt", "pants"]:
        return "person detections with high confidence"

    return "related observations"


def generate_sql_query(question: str) -> str:
    """Use LFM 1.2B to convert natural language to SQL."""
    # First check if the question asks about unavailable fields
    question_lower = question.lower()
    unavailable_found = None
    for field in UNAVAILABLE_FIELDS:
        if field in question_lower:
            unavailable_found = field
            break

    if unavailable_found:
        alternative = get_suggested_alternative(unavailable_found)
        return f"CLARIFICATION: I can search by class (car, person, etc.), camera, time, or confidence, but {unavailable_found} information isn't available. Would you like me to show {alternative} instead?"

    try:
        response = requests.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Question: " + question}
                ],
                "max_tokens": 200,
                "temperature": 0.01,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        raw_content = result["choices"][0]["message"]["content"].strip()

        app.logger.info(f"LLM raw response: {raw_content[:200]}")

        # Check if LLM returned a clarification response
        if is_clarification_response(raw_content):
            app.logger.info(f"LLM returned clarification for: {question}")
            return f"CLARIFICATION: {raw_content}"

        # Extract SQL from response
        sql = raw_content.replace("```sql", "").replace("```", "").strip()

        # Remove common prefixes
        prefixes = ["Here's the query:", "Here is the query:", "Query:", "The query:", "SQL:", "The SQL query:"]
        for prefix in prefixes:
            if sql.upper().startswith(prefix.upper()):
                sql = sql[len(prefix):].strip()

        # Find SELECT keyword
        select_idx = sql.upper().find("SELECT")
        if select_idx >= 0:
            sql = sql[select_idx:]
            if ";" in sql:
                sql = sql.split(";")[0].strip()
            if "\n" in sql:
                sql = sql.split("\n")[0].strip()
            app.logger.info(f"Extracted SQL: {sql}")
            return sql

        app.logger.warning(f"No SELECT found in LLM response: {raw_content}")
        return None

    except requests.RequestException as e:
        app.logger.error(f"LLM request failed: {e}")
        return None


def validate_sql(sql: str) -> tuple[bool, str]:
    """Validate that the SQL is safe and SELECT-only."""
    if not sql:
        return False, "Empty query"

    if sql.startswith("CLARIFICATION:"):
        return True, ""  # clarification messages are valid

    sql_upper = sql.upper().strip()

    if not (sql_upper.startswith("SELECT ") or sql_upper.startswith("SELECT*")):
        if "FOR YOU" in sql_upper or "TO RETRIEVE" in sql_upper or "QUERY" in sql_upper:
            return False, "LLM generated explanation instead of SQL - try rephrasing your question"
        return False, f"Invalid SQL format: {sql[:50]}..."

    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "ATTACH", "DETACH"]
    for keyword in dangerous:
        if keyword in sql_upper:
            return False, f"Dangerous keyword: {keyword}"

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
    """Generate a natural language answer from query results."""
    if not results:
        return "No matching observations found for your query."

    if sql and 'count(*)' in sql.lower():
        count = results[0].get('count', len(results)) if results else 0
        return f"Found {count} observations matching your query."

    if sql and 'group by' in sql.lower():
        summary_parts = []
        for row in results[:5]:
            class_name = row.get('class_name', row.get('class', 'unknown'))
            cnt = row.get('count', row.get('total', 0))
            if cnt:
                summary_parts.append(f"{cnt} {class_name}")
        if summary_parts:
            return f"Detected: {', '.join(summary_parts)}."
        return f"Found {len(results)} categories of observations."

    try:
        class_counts = {}
        cameras = set()

        for row in results:
            cn = row.get('class_name', 'unknown')
            class_counts[cn] = class_counts.get(cn, 0) + 1
            if 'camera_id' in row and row['camera_id']:
                cameras.add(row.get('camera_id', 'unknown'))

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

    sql = generate_sql_query(question)

    # Handle clarification responses
    if sql and sql.startswith("CLARIFICATION:"):
        clarification_text = sql[16:]  # Remove "CLARIFICATION: " prefix
        app.logger.info(f"Returning clarification: {clarification_text}")
        return jsonify({
            "success": True,
            "question": question,
            "answer": clarification_text,
            "is_clarification": True,
            "sql": None,
            "result_count": 0,
            "results": []
        })

    if not sql:
        app.logger.warning(f"LLM failed to generate SQL for: {question}")
        return jsonify({
            "error": "Failed to generate SQL query - try rephrasing your question",
            "question": question
        })

    app.logger.info(f"Generated SQL: {sql}")

    valid, error = validate_sql(sql)
    if not valid:
        app.logger.warning(f"SQL validation failed: {error}")
        return jsonify({
            "error": f"Invalid query: {error}",
            "sql": sql,
            "question": question
        }), 400

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

    answer = generate_answer(question, rows, sql)
    app.logger.info(f"Answer: {answer}")

    return jsonify({
        "success": True,
        "question": question,
        "sql": sql,
        "answer": answer,
        "result_count": len(rows),
        "results": rows[:50]
    })


@app.route("/health")
def health():
    """Health check endpoint."""
    status = {"status": "ok", "llama_server": "unknown", "db": "unknown"}

    try:
        resp = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=5)
        status["llama_server"] = "connected" if resp.status_code == 200 else "error"
    except:
        status["llama_server"] = "disconnected"

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
    port = int(os.getenv("PORT", 80))
    app.run(host="0.0.0.0", port=port, debug=False)
