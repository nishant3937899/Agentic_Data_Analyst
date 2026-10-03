import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from google import genai
from google.genai import types
from .tools import lookup_schema,run_sql,make_chart,feature_engineering,inspect_data_quality

#gemini execute tools
def execute_tool(name, args):

    if name == "lookup_schema":
        return lookup_schema()

    elif name == "run_sql":
        return run_sql(args["query"])

    elif name == "make_chart":
        return make_chart(
            result_id=args["result_id"],
            chart_type=args["chart_type"],
            x=args.get("x"),
            y=args.get("y"),
            hue=args.get("hue"),
            title=args.get("title"),
            color=args.get("color"),
            colors=args.get("colors"),
            palette=args.get("palette"),
            alpha=args.get("alpha", 0.8),
            height=args.get("height", 500),
            width=args.get("width", 900),
            rotation=args.get("rotation", 0),
            marker=args.get("marker", "circle"),
            linewidth=args.get("linewidth", 2),
            annotate=args.get("annotate", False),
            colormap=args.get("colormap", "Viridis"),
            bins=args.get("bins", 20),
            rolling_window=args.get("rolling_window", 7),
            bubble_scale=args.get("bubble_scale", 20),
            size=args.get("size")
    )

  
    elif name == 'inspect_data_quality':
        return inspect_data_quality()
        
    elif name == 'feature_engineering':
        return feature_engineering(
            result_id=args.get('result_id'),
            operation=args.get('operation'),
            new_column=args.get('new_column'),
            column=args.get('column'),
            column2=args.get('column2'),
            value=args.get('value')
        )
    else:
        raise ValueError(
            f"Unknown tool: {name}"
        )

    
#gemini tool here
gemini_tools = [

    types.Tool(
        function_declarations=[

            # <----->
            # TOOL 1 — LOOKUP SCHEMA
            # <----->

            types.FunctionDeclaration(
                name="lookup_schema",
                description="""
                Inspect the DuckDB database schema.

                Use this when you need to know:
                - available tables
                - column names
                - column data types

                Call this before writing SQL when the schema
                is not already known.
                """,
                parameters={
                    "type": "object",
                    "properties": {}
                }
            ),


            # <----->
            # TOOL 2 — RUN SQL
            # <----->

            types.FunctionDeclaration(
                name="run_sql",
                description="""
                
                    Execute a single SQL statement against the uploaded dataset.

                    This tool can be used for both data analysis and data modification.

                    The uploaded dataset is normally available as the DuckDB table 'data'.

                    Allowed SQL operations:
                    - SELECT
                    - WITH
                    - UPDATE
                    - DELETE
                    - ALTER

                    Use SELECT or WITH when you need to:
                    - inspect the dataset
                    - filter rows
                    - select columns
                    - group and aggregate data
                    - calculate statistics
                    - compare values
                    - investigate data-quality problems
                    - answer analytical questions

                    Use UPDATE when you need to:
                    - clean incorrect values
                    - replace missing or blank values
                    - standardize values
                    - modify existing column values

                    Use DELETE when you need to:
                    - remove unwanted rows
                    - remove rows based on a specific condition
                    - remove duplicate or invalid records when appropriate

                    IMPORTANT:
                    - Only ONE SQL statement is allowed per tool call.
                    - Do not use DROP, CREATE, TRUNCATE, ATTACH, or DETACH.
                    - Do not invent table names. The uploaded dataset is normally 'data'.
                    - Inspect the data before modifying it when necessary.
                    - Use WHERE conditions carefully with UPDATE and DELETE to avoid
                    modifying unintended rows.
                    - Do not modify data unnecessarily.
                    - After SELECT or WITH, the tool returns the query result.
                    - After UPDATE or DELETE, the tool returns the updated dataset.
                    - The result is limited to the first 100 rows when returned to the model.

                    Examples:

                    Analysis:
                    SELECT * FROM data LIMIT 10

                    Analysis:
                    SELECT gender, AVG(math_score)
                    FROM data
                    GROUP BY gender

                    Cleaning:
                    UPDATE data
                    SET TotalCharges = '0'
                    WHERE TotalCharges = ' '

                    Cleaning:
                    UPDATE data
                    SET column_name = NULL
                    WHERE column_name = ''

                    Removing rows:
                    DELETE FROM data
                    WHERE column_name IS NULL
                    """,
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": """
                                A single SQL statement to analyze, clean, or modify
                                the uploaded dataset.

                                Use the table name 'data'.

                                The statement must be one of:
                                SELECT, WITH, UPDATE, or DELETE.
                                """
                            }
                        },
                        "required": ["query"]
                    }
            ),

            # <----->
            # TOOL 6 — featur engineering
            # <----->
            types.FunctionDeclaration(
                name="feature_engineering",

                description="""
                Create a new derived column from an existing query result to improve
                analysis.

                Use this tool when an existing dataset does not contain a useful metric
                directly, but the metric can be derived from existing columns.
                
                MOST IMPORTANT: 
                -to use this function you must first run run_sql to get result_id. you should then pass this result id to function
                -you should not pass result_id = data

                Examples:

                - profit = revenue - cost
                - sales_value = quantity * price
                - profit_margin = profit / revenue * 100
                - year/month/quarter from a date
                - day of week from a date
                - log transformation of a numeric column
                - absolute value
                - text length
                - uppercase/lowercase transformation
                - categorical bins such as Low/Medium/High

                Choose meaningful feature names such as:
                profit, profit_margin, sales_value, year, month,
                quarter, delivery_days, customer_age.

                Do not create unnecessary features.
                Prefer business-useful derived metrics that help answer the user's question.
                Use the result_id returned by run_sql.
                this directly modify the current data in the database.
                After creating a feature, you should pull data using run_sql function.
                
                """,

                    parameters={
                        "type": "object",

                        "properties": {

                            "result_id": {
                                "type": "string",
                                "description":
                                    "ID of the existing query result from run_sql."
                            },

                            "operation": {
                                "type": "string",
                                "enum": [
                                    "add",
                                    "subtract",
                                    "multiply",
                                    "divide",
                                    "percentage",
                                    "absolute",
                                    "log",
                                    "year",
                                    "month",
                                    "quarter",
                                    "day_of_week",
                                    "length",
                                    "uppercase",
                                    "lowercase",
                                    "bin"
                                ],
                                "description":
                                    "Operation used to create the new feature."
                            },

                            "new_column": {
                                "type": "string",
                                "description":
                                    "Name of the new derived column."
                            },

                            "column": {
                                "type": "string",
                                "description":
                                    "Primary source column."
                            },

                            "column2": {
                                "type": "string",
                                "description":
                                    "Second source column for arithmetic operations."
                            },

                            "value": {
                                "type": "object",
                                "description":
                                    "Optional configuration such as bins and labels."
                            }
                        },

                        "required": [
                            "result_id",
                            "operation",
                            "new_column"
                        ]
                    }
                ),
                # <----->
                # TOOL 5 — inspect data quality
                # <----->
                types.FunctionDeclaration(
                    name="inspect_data_quality",

                    description="""
                        Inspect the dataset for data quality issues before performing analysis.

                        Use this tool when the user asks whether the data is clean, whether there are
                        missing values, duplicates, invalid values, inconsistent data, or other
                        data-quality problems.

                        The tool checks:
                        - Total number of rows and columns
                        - Missing/null values
                        - Missing-value percentages
                        - Duplicate rows
                        - Number of unique values in each column
                        - Data types
                        - Basic information about each column

                        Use this tool instead of running many separate SQL queries for basic
                        data-quality inspection.

                        IMPORTANT:
                        - This tool ONLY INSPECTS the data.
                        - It does NOT modify, delete, or clean any data.
                        - If problems are found, explain them clearly.
                        - If the user asks to actually clean/fix the data, use the cleaning tool
                        after inspecting the data.
                        """,

                            parameters={
                                "type": "object",
                                "properties": {}
                            }
                ),


            # <----->
            # TOOL 3 — MAKE CHARTS
            # <----->

            types.FunctionDeclaration(
                name="make_chart",

                description="""
            Create an interactive Plotly chart from the result of a previous
            run_sql or feature_engineering tool call.

            IMPORTANT:
            - Use the result_id returned by run_sql or feature_engineering.
            - Do NOT provide a chart number.
            - Do NOT provide a filename.
            - The application automatically generates a unique chart ID and filename.
            - Every call creates a new chart, even if the same chart is requested again.
            - Multiple charts can be created from the same result_id.

            WORKFLOW:
            1. Use run_sql to obtain the data needed for the chart.
            2. Use the returned result_id with make_chart.
            3. If feature engineering was used, use the NEW result_id returned by
            feature_engineering.
            4. Create a chart only when visualization helps answer the user's question.

            CHART TYPES:

            bar:
                Use for comparing a categorical column with one numeric metric.
                Example: average math score by gender.

            horizontal_bar:
                Use when category labels are long or there are many categories.

            line:
                Use for trends or ordered/time-based data.
                y can contain multiple numeric columns.

            area:
                Use for trends where the magnitude or cumulative pattern is important.

            scatter:
                Use to examine the relationship between two numeric variables.
                hue can optionally divide points into categories.

            histogram:
                Use to show the distribution of a numeric variable.
                bins controls the number of bins.

            box:
                Use to compare distributions, spread, median, and outliers.
                x can be a categorical grouping column.

            violin:
                Use to compare distributions and their shapes.
                x can be a categorical grouping column.

            pie:
                Use for part-to-whole relationships with a small number of categories.

            donut:
                Same purpose as pie, with a hole in the center.

            grouped_bar:
                Use to compare multiple groups side by side.
                Two valid patterns:
                1. x + one y + hue
                2. x + multiple y columns
                NEVER set hue equal to x.
                If there is only one categorical grouping column, leave hue empty.

            stacked_bar:
                Use to show how multiple metrics contribute to a total across categories.

            heatmap:
                Use for matrix-style relationships or categorical/numeric summaries.
                If x/y/hue are not suitable, the tool can create a numeric correlation-style
                heatmap.

            correlation_heatmap:
                Use specifically to show correlations between numeric columns.

            density:
                Use to examine the shape of a numeric distribution.

            ecdf:
                Use to compare cumulative distributions.

            pareto:
                Use when categories should be ordered by contribution and cumulative
                percentage is useful.

            rolling_line:
                Use for ordered/time-based data when a moving average is useful.
                rolling_window controls the window size.

            hexbin:
                Use for large datasets when a scatter plot would contain too many points.
                It shows point density.

            bubble:
                Use for three-variable relationships:
                x, y, and a size variable.
                Use the size argument to specify the numeric size column.

            waterfall:
                Use to show sequential positive and negative contributions to a total.

            COLUMN RULES:
            - x must be an existing column.
            - y must be an existing column or a list of existing columns.
            - hue is optional.
            - size is optional and must be a numeric column.
            - Never invent column names.
            - Use lookup_schema or run_sql results to determine valid column names.
            - Never use hue equal to x.
            - Choose chart_type based on the analytical question, not randomly.

            STYLING:
            You may provide:
            - title
            - color
            - colors
            - palette
            - alpha
            - height
            - width
            - rotation
            - marker
            - linewidth
            - annotate
            - colormap
            - bins
            - rolling_window
            - bubble_scale
            - size

            Keep styling appropriate to the data.
            Do not provide unnecessary styling arguments.

            CHART IDs:
            The application generates the chart ID automatically.
            NEVER send:
            - number
            - filename
            - chart_id

            The chart returned by this tool is interactive Plotly HTML and can be
            displayed directly by the web application.
            """,

                parameters={
                    "type": "object",

                    "properties": {

                        "result_id": {
                            "type": "string",
                            "description": (
                                "The result_id returned by run_sql or "
                                "feature_engineering."
                            )
                        },

                        "chart_type": {
                            "type": "string",
                            "enum": [
                                "bar",
                                "horizontal_bar",
                                "line",
                                "area",
                                "scatter",
                                "histogram",
                                "box",
                                "violin",
                                "pie",
                                "donut",
                                "grouped_bar",
                                "stacked_bar",
                                "heatmap",
                                "correlation_heatmap",
                                "density",
                                "ecdf",
                                "pareto",
                                "rolling_line",
                                "hexbin",
                                "bubble",
                                "waterfall"
                            ],
                            "description": "Type of Plotly chart to create."
                        },

                        "x": {
                            "type": "string",
                            "description": "Column used for the x-axis or category."
                        },

                        "y": {
                            "description": (
                                "Column or list of columns used for the y-axis. "
                                "Use a list for multiple metrics."
                            ),
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            ]
                        },

                        "hue": {
                            "type": "string",
                            "description": (
                                "Optional categorical grouping column. "
                                "NEVER use the same column as x."
                            )
                        },

                        "title": {
                            "type": "string",
                            "description": "Chart title."
                        },

                        "color": {
                            "type": "string",
                            "description": "Optional single color."
                        },

                        "colors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of colors."
                        },

                        "palette": {
                            "type": "string",
                            "description": "Optional Plotly color palette."
                        },

                        "alpha": {
                            "type": "number",
                            "description": "Opacity from 0 to 1."
                        },

                        "height": {
                            "type": "integer",
                            "description": "Chart height in pixels."
                        },

                        "width": {
                            "type": "integer",
                            "description": "Chart width in pixels."
                        },

                        "rotation": {
                            "type": "number",
                            "description": "Rotation angle for x-axis labels."
                        },

                        "marker": {
                            "type": "string",
                            "description": "Plotly marker style."
                        },

                        "linewidth": {
                            "type": "number",
                            "description": "Line width."
                        },

                        "annotate": {
                            "type": "boolean",
                            "description": "Whether to display values/annotations where supported."
                        },

                        "colormap": {
                            "type": "string",
                            "description": "Plotly continuous color scale."
                        },

                        "bins": {
                            "type": "integer",
                            "description": "Number of histogram bins."
                        },

                        "rolling_window": {
                            "type": "integer",
                            "description": "Window size for rolling_line."
                        },

                        "bubble_scale": {
                            "type": "number",
                            "description": "Maximum bubble size."
                        },

                        "size": {
                            "type": "string",
                            "description": "Numeric column controlling bubble size."
                        }
                    },

                    "required": [
                        "result_id",
                        "chart_type"
                    ]
                }
            )


        ]
    )
]




