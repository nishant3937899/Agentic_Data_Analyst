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

