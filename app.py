from src.AI_processing import ask_agent, load_api_key
from flask import Flask, render_template, request, session, jsonify,send_file
import pandas as pd
import io
import os
import markdown
from src.database import conn

from src.database import load_dataframe


app = Flask(__name__)

app.secret_key = "FLASK_SECRET_KEY"


# CLEAR OLD CHARTS

def clear_old_charts():

    chart_folder = "static/charts"

    if not os.path.exists(chart_folder):
        os.makedirs(chart_folder)

    for filename in os.listdir(chart_folder):

        if filename.endswith(".html"):

            file_path = os.path.join(chart_folder, filename)

            try:
                os.remove(file_path)
                print(f"Deleted old chart: {filename}")

            except OSError as e:
                print(f"Could not delete {filename}: {e}")



# HOME PAGE

@app.route("/")
def home():

    if "messages" not in session:
        session["messages"] = []

    if "charts" not in session:
        session["charts"] = []

    return render_template(
        "index.html",
        messages=session["messages"],
        charts=session["charts"]
    )



# UPLOAD CSV

@app.route("/upload", methods=["POST"])
def upload():

    if "csv_file" not in request.files:

        return jsonify({
            "success": False,
            "message": "No CSV file selected."
        })


    file = request.files["csv_file"]


    if file.filename == "":

        return jsonify({
            "success": False,
            "message": "Please select a CSV file."
        })


    try:

        stream = io.StringIO(
            file.stream.read().decode("UTF8"),
            newline=None
        )

        df = pd.read_csv(stream)


        # Delete plots from previous dataset
        clear_old_charts()


        # Load new dataframe
        load_dataframe(df)


        # New dataset = new charts + new conversation
        session["charts"] = []
        session["messages"] = []


        return jsonify({
            "success": True,
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns)
        })


    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })



# API KEY

@app.route("/api-key", methods=["POST"])
def set_api_key():

    key = request.form.get("api_key", "").strip()


    try:

        if key:
            client = load_api_key(key)
        else:
            client = load_api_key()


        return jsonify({
            "success": True,
            "message": "API key configured."
        })


    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })



# ASK AI

@app.route("/ask", methods=["POST"])
def ask():

    question = request.form.get("question", "").strip()


    if not question:

        return jsonify({
            "success": False,
            "message": "Please enter a question."
        })


    
    # Save USER message
    
    messages = session.get("messages", [])

    messages.append({
        "role": "user",
        "content": question
    })

    session["messages"] = messages


    try:

        
        # Load API client

        client = load_api_key()


        
        # Ask agent

        output = ask_agent(
            question,
            client
        )


        
        # Save AI response
        
        messages = session.get("messages", [])

        answer_html = markdown.markdown(
            output["answer"],
            extensions=["fenced_code", "tables"]
        )

        messages.append({
            "role": "assistant",
            "content": answer_html
        })
        session["messages"] = messages


        
        # Save charts
        
        charts = session.get("charts", [])

        charts.extend(output["charts"])

        session["charts"] = charts


        return jsonify({
            "success": True,
            "answer": answer_html,
            "charts": output["charts"]
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })

# DOWNLOAD CSV

@app.route("/download-csv")
def download_csv():

    df = conn.execute("SELECT * FROM data").fetchdf()

    csv_data = io.BytesIO()
    df.to_csv(csv_data, index=False)
    csv_data.seek(0)

    return send_file(
        csv_data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="data.csv"
    )

# RUN

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )