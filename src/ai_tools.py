import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from google import genai
from google.genai import types
from .tools import lookup_schema,run_sql,clean_data,make_chart,feature_engineering,inspect_data_quality

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
            title=args.get("title"),
            hue=args.get("hue"),
            bins=args.get("bins", 20),
            number=args.get("number")
        )

    elif name == "clean_data":
        return clean_data(
            table_name=args.get("table_name", "data"),
            operations=args["operations"]
        )
    elif name == 'inspect_data_quality':
        return inspect_data_quality()
    else:
        raise ValueError(
            f"Unknown tool: {name}"
        )

    
#gemini tool here
gemini_tools = [

    types.Tool(
        function_declarations=[

            # =====================================================
            # TOOL 1 — LOOKUP SCHEMA
            # =====================================================

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


            # =====================================================
            # TOOL 2 — RUN SQL
            # =====================================================

            types.FunctionDeclaration(
                name="run_sql",
                description="""
                Execute a read-only SQL query against DuckDB.

                Use this to retrieve, aggregate, filter,
                compare, or analyze data.

                Only SELECT and WITH queries are allowed.

                The result is stored internally and returns
                a result_id that can later be used by make_chart.

                Always use the actual table and column names
                discovered from lookup_schema.

                IMPORTANT:
                - Only ONE SQL statement may be provided per call.
                - Do NOT send multiple SELECT statements separated by semicolons.
                - If you need information about multiple columns, combine the analysis
                into one SQL query whenever possible.
                - Use SELECT or WITH queries only.
                - Never modify the database.
                - Results are automatically limited to 100 rows.

                For example, DO NOT do:

                SELECT DISTINCT gender FROM data;
                SELECT DISTINCT lunch FROM data;
                SELECT DISTINCT race_ethnicity FROM data;

                Instead, make separate tool calls when necessary, or use a single
                query that summarizes the required information.
                """,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A read-only SQL query."
                        }
                    },
                    "required": ["query"]
                }
            ),

            # =====================================================
            # TOOL 3 — MAKE CHARTS
            # =====================================================
            types.FunctionDeclaration(
                name="make_chart",

                description="""
                Create a chart from an existing query result.

                IMPORTANT:
                - You MUST use a result_id returned by run_sql or feature_engineering.
                - Only use column names that exist in that result.
                - Do NOT invent column names.
                - Choose the simplest chart that clearly communicates the insight.
                - The chart is created by Python; NEVER generate plotting code yourself.

                SUPPORTED CHART TYPES:
                1. bar
                2. horizontal_bar
                3. line
                4. area
                5. scatter
                6. histogram
                7. box
                8. violin
                9. pie
                10. donut
                11. grouped_bar
                12. stacked_bar
                13. heatmap
                14. correlation_heatmap
                15. density
                16. ecdf
                17. pareto
                18. rolling_line
                19. hexbin
                20. bubble
                21. waterfall


                CHART SELECTION RULES:

                BAR:
                Use when comparing one numeric metric across categories.

                Example:
                x = "test_preparation_course"
                y = "avg_math"
                chart_type = "bar"


                HORIZONTAL_BAR:
                Use when category names are long or when a horizontal comparison is clearer.


                LINE:
                Use for trends over time or ordered numerical values.

                Example:
                x = "month"
                y = "avg_sales"


                AREA:
                Use for trends where the magnitude or cumulative contribution is important.


                SCATTER:
                Use to investigate the relationship between two numerical variables.

                Example:
                x = "math_score"
                y = "reading_score"


                HISTOGRAM:
                Use to show the distribution of a numerical variable.

                Example:
                y = "math_score"


                BOX:
                Use to show distribution, spread, median and outliers.


                VIOLIN:
                Use to compare distributions across categories.

                Example:
                x = "gender"
                y = "math_score"


                PIE:
                Use only when showing the composition of a small number of categories.


                DONUT:
                Same use case as pie, but with a donut-style visualization.


                GROUPED_BAR:
                Use to compare multiple metrics or subcategories.

                IMPORTANT RULES:
                - hue is OPTIONAL.
                - NEVER set hue equal to x.
                - If there is only one categorical column and one metric,
                leave hue empty.
                - If x and hue are the same column, do NOT provide hue.
                - If multiple numeric metrics are available, y may contain
                multiple columns.

                Example 1:
                x = "test_preparation_course"
                y = "avg_math"
                hue = null

                Example 2:
                x = "test_preparation_course"
                y = ["avg_math", "avg_reading", "avg_writing"]
                hue = null

                Example 3:
                x = "gender"
                y = "avg_math"
                hue = "test_preparation_course"

                Do NOT do this:
                x = "test_preparation_course"
                hue = "test_preparation_course"


                STACKED_BAR:
                Use when categories should be stacked to show composition.

                HEATMAP:
                Use to show values across two categorical dimensions or a matrix.

                CORRELATION_HEATMAP:
                Use when the goal is to understand relationships between multiple
                numerical columns.

                DENSITY:
                Use to show the distribution of a numerical variable.

                ECDF:
                Use to compare cumulative distributions.

                PARETO:
                Use when categories should be ranked and cumulative contribution
                should be shown.

                ROLLING_LINE:
                Use for noisy ordered/time-series data where a rolling average
                helps reveal the trend.

                HEXBIN:
                Use for very large numerical datasets where a scatter plot would
                contain too many overlapping points.

                BUBBLE:
                Use when a third numerical variable can meaningfully represent
                magnitude through bubble size.

                WATERFALL:
                Use when showing how individual positive/negative contributions
                lead to a final total.


                IMPORTANT Y RULE:
                - y can be either one column name or a list of column names.
                - For charts that require a single numerical variable, use one y column.
                - For grouped_bar and stacked_bar, multiple y columns are allowed.
                - Never pass an empty y value when the selected chart requires y.


                COLOR AND STYLE:
                You may optionally specify:
                - color
                - colors
                - palette
                - alpha
                - figsize
                - grid
                - grid_axis
                - legend
                - rotation
                - font_size
                - title_size
                - label_size
                - marker
                - linewidth
                - annotate
                - colormap
                - bins
                - rolling_window
                - bubble_scale

                Do not invent unusual styling parameters.
                Use default styling unless the user requests a specific style.

                For heatmap, correlation_heatmap and hexbin, use colormap.
                For categorical charts, color/colors/palette may be used.

                If the user does not request a specific color or palette,
                use the chart tool's default.


                GENERAL RULES:
                0.You should give number to the plot you are makin for example first plot will be number one, the number should aways start form 1 and go so on.
                1. First obtain or inspect the relevant data using run_sql.
                2. Use the returned result_id.
                3. Choose a chart that directly supports the analytical conclusion.
                4. Do not create a chart just for decoration.
                5. If the result has only one category and one metric,
                use a simple bar chart.
                6. If comparing several metrics across categories,
                use grouped_bar.
                7. If analyzing a relationship between two numerical variables,
                use scatter.
                8. If analyzing distributions, use histogram, box or violin.
                9. If analyzing correlation among numerical variables,
                use correlation_heatmap.
                10. If analyzing time trends, use line or rolling_line.
                11. After creating the chart, continue reasoning if another tool
                    is required.
                12. Do not stop merely because a chart was successfully created.
                """,

                parameters={
                    "type": "object",

                    "properties": {

                        "result_id": {
                            "type": "string",
                            "description": (
                                "ID of the existing query result returned by "
                                "run_sql or feature_engineering."
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

                            "description": (
                                "Type of chart to create. "
                                "Choose the simplest chart appropriate "
                                "for the analytical question."
                            )
                        },

                        "x": {
                            "type": "string",
                            "description": (
                                "Column used for the x-axis or category. "
                                "Must exist in the result_id dataset."
                            )
                        },

                        "y": {
                            "type": "string",
                            "description": (
                                "Numeric column to visualize. "
                                "For grouped_bar and stacked_bar, multiple "
                                "numeric columns may be supplied when supported."
                            )
                        },
                        "number":{
                            "type":"integer",
                            "description":(
                                "here you should put the plot number for example 1,2,3,4 and so on always sent this."
                                "the number should start form 1 and go so on."
                            )
                        },
                        "hue": {
                            "type": "string",
                            "description": (
                                "Optional second categorical grouping column. "
                                "IMPORTANT: never set hue equal to x. "
                                "Leave empty when there is only one categorical "
                                "grouping variable."
                            )
                        },

                        "title": {
                            "type": "string",
                            "description": (
                                "Clear descriptive title explaining what "
                                "the chart shows."
                            )
                        },

                        "color": {
                            "type": "string",
                            "description": (
                                "Optional single color for the chart."
                            )
                        },

                        "colors": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": (
                                "Optional list of colors for categories."
                            )
                        },

                        "palette": {
                            "type": "string",
                            "description": (
                                "Optional color palette name supported by "
                                "the chart implementation."
                            )
                        },

                        "alpha": {
                            "type": "number",
                            "description": (
                                "Transparency of chart elements. "
                                "Usually between 0 and 1."
                            )
                        },

                        "figsize": {
                            "type": "array",
                            "items": {
                                "type": "number"
                            },
                            "description": (
                                "Chart size as [width, height]."
                            )
                        },

                        "grid": {
                            "type": "boolean",
                            "description": (
                                "Whether to display the chart grid."
                            )
                        },

                        "grid_axis": {
                            "type": "string",
                            "enum": [
                                "both",
                                "x",
                                "y"
                            ],
                            "description": (
                                "Axis on which the grid should appear."
                            )
                        },

                        "legend": {
                            "type": "boolean",
                            "description": (
                                "Whether to display the legend."
                            )
                        },

                        "rotation": {
                            "type": "number",
                            "description": (
                                "Rotation angle for x-axis labels."
                            )
                        },

                        "font_size": {
                            "type": "number",
                            "description": (
                                "Font size for axis tick labels."
                            )
                        },

                        "title_size": {
                            "type": "number",
                            "description": (
                                "Font size for the chart title."
                            )
                        },

                        "label_size": {
                            "type": "number",
                            "description": (
                                "Font size for axis labels."
                            )
                        },

                        "marker": {
                            "type": "string",
                            "description": (
                                "Marker style for line charts."
                            )
                        },

                        "linewidth": {
                            "type": "number",
                            "description": (
                                "Line width for line-based charts."
                            )
                        },

                        "annotate": {
                            "type": "boolean",
                            "description": (
                                "Whether to display values directly on "
                                "the chart where supported."
                            )
                        },

                        "colormap": {
                            "type": "string",
                            "description": (
                                "Colormap used by heatmap, correlation_heatmap "
                                "and hexbin charts."
                            )
                        },

                        "bins": {
                            "type": "integer",
                            "description": (
                                "Number of bins for histogram."
                            )
                        },

                        "rolling_window": {
                            "type": "integer",
                            "description": (
                                "Window size for rolling_line."
                            )
                        },

                        "bubble_scale": {
                            "type": "number",
                            "description": (
                                "Scaling factor for bubble sizes."
                            )
                            }
                        },
                    

                    "required": [
                        "result_id",
                        "chart_type"
                    ]
                }
            ),

            # =====================================================
            # TOOL 4 — CLEAN DATA
            # =====================================================

            types.FunctionDeclaration(
                name="clean_data",
                description="""
                Clean the dataset when data-quality problems could
                affect the analysis.

                IMPORTANT:
                - Inspect the data before deciding that cleaning
                  is necessary.
                - Do not clean data unnecessarily.
                - Prefer the least destructive operation that
                  solves the problem.
                - Cleaning changes the current in-memory DuckDB table.
                - Do not invent cleaning operations.
                - Only use the supported operations listed below.

                Supported operations:

                remove_duplicates:
                    Remove completely duplicated rows.

                drop_null_rows:
                    Remove rows containing one or more NULL values.
                    Use cautiously because this can remove many rows.

                fill_numeric_nulls_mean:
                    Replace missing numerical values with the column mean.
                    Use when mean imputation is appropriate.

                fill_numeric_nulls_median:
                    Replace missing numerical values with the column median.
                    Prefer this when numerical data may contain outliers.

                fill_numeric_nulls_zero:
                    Replace missing numerical values with zero.
                    Only use when zero has a meaningful interpretation.

                fill_text_nulls:
                    Replace missing text values with 'Unknown'.

                strip_whitespace:
                    Remove leading and trailing whitespace from text columns.
                    Useful for inconsistent categorical values.

                lowercase_text:
                    Convert text columns to lowercase when capitalization
                    should not represent different categories.

                uppercase_text:
                    Convert text columns to uppercase when standardized
                    uppercase text is appropriate.

                Multiple operations may be provided when several
                data-quality problems are present.

                Return a summary of the cleaning performed,
                including rows removed and remaining NULL values.
                """,
                parameters={
                    "type": "object",
                    "properties": {

                        "table_name": {
                            "type": "string",
                            "description": """
                            Name of the DuckDB table to clean.
                            Usually 'data'.
                            """
                        },

                        "operations": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "remove_duplicates",
                                    "drop_null_rows",
                                    "fill_numeric_nulls_mean",
                                    "fill_numeric_nulls_median",
                                    "fill_numeric_nulls_zero",
                                    "fill_text_nulls",
                                    "strip_whitespace",
                                    "lowercase_text",
                                    "uppercase_text"
                                ]
                            },
                            "description": """
                            List of cleaning operations to perform.
                            Use only the supported operations.
                            """
                        }
                    },

                    "required": [
                        "table_name",
                        "operations"
                    ]
                }
            ),
            # =====================================================
            # TOOL 6 — featur engineering
            # =====================================================
            types.FunctionDeclaration(
                name="feature_engineering",

                description="""
                Create a new derived column from an existing query result to improve
                analysis.

                Use this tool when an existing dataset does not contain a useful metric
                directly, but the metric can be derived from existing columns.

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
                After creating a feature, use the new result_id for further analysis.
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
                # =====================================================
                # TOOL 5 — inspect data quality
                # =====================================================
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
                )
        ]
    )
]




