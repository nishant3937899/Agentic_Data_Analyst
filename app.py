from src.AI_processing import ask_agent, load_api_key
from flask import Flask, render_template, request, session
import pandas as pd
import io
from src.database import load_dataframe
from flask import session

app = Flask(__name__)



# Needed for Flask session
app.secret_key = "change-this-later"


@app.route("/", methods=["GET", "POST"])
def home():
    session.clear()
    if "messages" not in session:
        session["messages"] = []

    charts = []

    if request.method == "POST":

        # -------------------------
        # CSV
        # -------------------------
        if "csv_file" in request.files:
            file = request.files["csv_file"]

            if file.filename != "":
                stream = io.StringIO(
                    file.stream.read().decode("UTF8"),
                    newline=None
                )

                df = pd.read_csv(stream)

                load_dataframe(df)

        # -------------------------
        # API KEY
        # -------------------------
        key = request.form.get("api_key", "").strip()

        if key:
            client = load_api_key(key)
        else:
            client = load_api_key()

        # -------------------------
        # USER MESSAGE
        # -------------------------
        question = request.form.get("question", "").strip()

        if question:

            # Save user message
            messages = session["messages"]

            messages.append({
                "role": "user",
                "content": question
            })

            # Ask agent
            output = ask_agent(question, client)

            # Save AI response
            messages.append({
                "role": "assistant",
                "content": output
            })

            session["messages"] = messages

            # Find generated charts
            charts = [
                "chart1.png",
                "chart2.png",
                "chart3.png"
            ]

    return render_template(
        "index.html",
        messages=session["messages"],
        charts=charts
    )


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)