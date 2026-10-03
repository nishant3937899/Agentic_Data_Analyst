# Agentic Data Analyst

Agentic Data Analyst is an AI-powered web application that allows users to upload a CSV dataset and analyze it using natural language. Powered by Google Gemini, the application acts as an intelligent data analyst that can inspect data, generate SQL queries, perform analysis, create visualizations, check data quality, and perform data cleaning and feature engineering when needed.

The project uses Flask for the web interface, DuckDB for fast data analysis, and Pandas, Plotly, and Matplotlib for data processing and visualization.

Users can simply upload their CSV file and ask questions about their data. The AI agent decides which tools to use and returns understandable analysis with charts when appropriate. After performing data cleaning or feature engineering, users can also download the updated dataset as a CSV file.

🔗 **[Live Project](https://agentic-data-analyst-3v3f.onrender.com/)**

##  Demo

<!-- Add your project GIF here -->

![Demo](path/to/demo.gif)

##  Features

-  **AI-powered analysis** using Google Gemini
-  **Automatic data visualization**
-  **Data cleaning & quality inspection**
-  **Feature engineering**
-  **SQL analysis with DuckDB**
-  **Natural-language questions**
-  **CSV file upload**
## Tech Stack

- **Backend & Web Framework:** Python, Flask, Gunicorn
- **AI & LLM:** Google Gemini (`google-genai`)
- **Data Analysis & Querying:** DuckDB, Pandas, NumPy
- **Visualization:** Plotly, Matplotlib
- **Utilities:** Markdown

## How It Works
```
User Question
 ↓
Flask Frontend
 ↓
Question + Available Tools
 ↓
Gemini
 ↓
Gemini decides which tool to use
 ↓
Tool name + Parameters
 ↓
Tool Execution
 ↓
Tool Result
 ↓
Gemini
 ↓
Does the question require more analysis?
 ├── Yes → Select another tool
 │          ↓
 │       Execute Tool
 │          ↓
 │       Send Result to Gemini
 │          ↓
 │       Repeat
 │
 └── No
      ↓
   Final Answer
      ↓
Flask
 ↓
Display Answer + Charts
```
## Project Structure
```
Agentic_Data_Analyst-main/
│
├── research/                 # Research notebooks and experiments
│   ├── test.ipynb
│   └── test2.0.ipynb
│
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── AI_processing.py      # Core AI logic and processing
│   ├── ai_tools.py           # Tools info and executing function
│   ├── database.py           # Database connection 
│   └── tools.py              # Main tools
│
├── static/                   # Static assets
│   └── style.css             # Stylesheet for frontend styling
│
├── templates/                # HTML templates
│   └── index.html            # Main web interface view
│
├── .gitignore                # Git ignore file
├── README.md                 # Project documentation
├── __init__.py               
├── app.py                    # Main Flask/Web application entry point
└── requirement.txt           # Project dependencies
```
## How to Run 

```
git pull origin main
    ↓
pip install -r requirement.txt
    ↓
run app.py
```
## Author

Nishant Chandra Verma


