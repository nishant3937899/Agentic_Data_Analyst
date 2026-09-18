import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import uuid
from .database import conn, RESULTS
import os
import plotly.express as px
import plotly.graph_objects as go

#Tool 1 
def lookup_schema():
    """
    Return the schema of all tables in the DuckDB database.
    """

    tables = conn.sql("SHOW TABLES").fetchdf()

    schema = {}

    for table_name in tables["name"]:

        columns = conn.sql(
            f"DESCRIBE {table_name}"
        ).fetchdf()

        schema[table_name] = columns[
            ["column_name", "column_type"]
        ].to_dict("records")

    return schema

#Tool 2 
FORBIDDEN_SQL = [
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "ATTACH",
    "DETACH"
]
RESULTS = {}

def run_sql(query):

   
    query_clean = query.strip()


    statements = [
        statement.strip()
        for statement in query_clean.split(";")
        if statement.strip()
    ]

    if len(statements) > 1:
        raise ValueError(
            "Only one SQL statement is allowed per tool call."
        )

    if not statements:
        raise ValueError("Empty SQL query.")

    query_clean = statements[0]

 

    query_upper = query_clean.upper()

    if not (
        query_upper.startswith("SELECT")
        or query_upper.startswith("WITH")
    ):
        raise ValueError(
            "Only SELECT and WITH queries are allowed."
        )


    for keyword in FORBIDDEN_SQL:

        if keyword in query_upper:
            raise ValueError(
                f"Forbidden SQL keyword detected: {keyword}"
            )
    result = conn.execute(query_clean).fetchdf()



    result = result.head(100)


    result_id = str(uuid.uuid4())[:8]

    RESULTS[result_id] = result

 

    return {
        "result_id": result_id,
        "row_count": len(result),
        "columns": list(result.columns),
        "data": result.to_dict("records")
    }


SUPPORTED_CHART_TYPES = [
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
]


# ============================================================
# DEFAULT COLOR PALETTE
# ============================================================

DEFAULT_PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC"
]

#tool3
def make_chart(
    result_id,
    chart_type,
    x=None,
    y=None,
    hue=None,
    title=None,
    color=None,
    colors=None,
    palette=None,
    alpha=0.8,
    height=500,
    width=900,
    rotation=0,
    marker="circle",
    linewidth=2,
    annotate=False,
    colormap="Viridis",
    bins=20,
    rolling_window=7,
    bubble_scale=20,
    size=None,
    **kwargs
):

    

    # ============================================================
    # 1. VALIDATE RESULT
    # ============================================================

    if result_id not in RESULTS:
        raise ValueError(f"Unknown result_id: {result_id}")

    df = RESULTS[result_id].copy()

    if df.empty:
        raise ValueError("The result contains no data to plot.")

    # ============================================================
    # 2. NORMALIZE INPUTS
    # ============================================================

    chart_type = str(chart_type).lower().strip()

    if isinstance(y, str):
        y_columns = [y]
    elif isinstance(y, (list, tuple)):
        y_columns = list(y)
    else:
        y_columns = []

    # ============================================================
    # 3. VALIDATE COLUMNS
    # ============================================================

    if x and x not in df.columns:
        raise ValueError(
            f"x column '{x}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )

    if hue and hue not in df.columns:
        raise ValueError(
            f"hue column '{hue}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )

    if size and size not in df.columns:
        raise ValueError(
            f"size column '{size}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )

    for col in y_columns:
        if col not in df.columns:
            raise ValueError(
                f"y column '{col}' does not exist. "
                f"Available columns: {list(df.columns)}"
            )

    # Gemini sometimes sends x == hue.
    if hue == x:
        hue = None

    # ============================================================
    # 4. CREATE CHART DIRECTORY
    # ============================================================

    os.makedirs("charts", exist_ok=True)

    # ============================================================
    # 5. UNIQUE CHART ID
    # ============================================================

    chart_id = str(uuid.uuid4())

    chart_filename = f"{chart_id}.html"
    chart_path = os.path.join("charts", chart_filename)

    # ============================================================
    # 6. COLOR SETTINGS
    # ============================================================

    color_sequence = None

    if colors:
        if isinstance(colors, str):
            color_sequence = [colors]
        elif isinstance(colors, (list, tuple)):
            color_sequence = list(colors)

    elif palette:
        color_sequence = px.colors.qualitative.Plotly

    # ============================================================
    # 7. CREATE FIGURE
    # ============================================================

    fig = None

    # ------------------------------------------------------------
    # BAR
    # ------------------------------------------------------------

    if chart_type == "bar":

        if not x or not y_columns:
            raise ValueError("bar requires x and y.")

        grouped = (
            df.groupby(x, as_index=False)[y_columns[0]]
            .mean()
        )

        fig = px.bar(
            grouped,
            x=x,
            y=y_columns[0],
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # HORIZONTAL BAR
    # ------------------------------------------------------------

    elif chart_type == "horizontal_bar":

        if not x or not y_columns:
            raise ValueError("horizontal_bar requires x and y.")

        grouped = (
            df.groupby(x, as_index=False)[y_columns[0]]
            .mean()
        )

        fig = px.bar(
            grouped,
            x=y_columns[0],
            y=x,
            orientation="h",
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # LINE
    # ------------------------------------------------------------

    elif chart_type == "line":

        if not x or not y_columns:
            raise ValueError("line requires x and y.")

        grouped = (
            df.groupby(x, as_index=False)[y_columns]
            .mean()
        )

        fig = px.line(
            grouped,
            x=x,
            y=y_columns,
            title=title,
            markers=True,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # AREA
    # ------------------------------------------------------------

    elif chart_type == "area":

        if not x or not y_columns:
            raise ValueError("area requires x and y.")

        grouped = (
            df.groupby(x, as_index=False)[y_columns]
            .mean()
        )

        fig = px.area(
            grouped,
            x=x,
            y=y_columns,
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # SCATTER
    # ------------------------------------------------------------

    elif chart_type == "scatter":

        if not x or not y_columns:
            raise ValueError("scatter requires x and y.")

        fig = px.scatter(
            df,
            x=x,
            y=y_columns[0],
            color=hue,
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # HISTOGRAM
    # ------------------------------------------------------------

    elif chart_type == "histogram":

        if not y_columns:
            raise ValueError("histogram requires y.")

        fig = px.histogram(
            df,
            x=y_columns[0],
            color=hue,
            nbins=bins,
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # BOX
    # ------------------------------------------------------------

    elif chart_type == "box":

        if not y_columns:
            raise ValueError("box requires y.")

        fig = px.box(
            df,
            x=x,
            y=y_columns[0],
            color=hue,
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # VIOLIN
    # ------------------------------------------------------------

    elif chart_type == "violin":

        if not y_columns:
            raise ValueError("violin requires y.")

        fig = px.violin(
            df,
            x=x,
            y=y_columns[0],
            color=hue,
            box=True,
            points=False,
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # PIE
    # ------------------------------------------------------------

    elif chart_type == "pie":

        if not x or not y_columns:
            raise ValueError("pie requires x and y.")

        grouped = (
            df.groupby(x, as_index=False)[y_columns[0]]
            .sum()
        )

        fig = px.pie(
            grouped,
            names=x,
            values=y_columns[0],
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # DONUT
    # ------------------------------------------------------------

    elif chart_type == "donut":

        if not x or not y_columns:
            raise ValueError("donut requires x and y.")

        grouped = (
            df.groupby(x, as_index=False)[y_columns[0]]
            .sum()
        )

        fig = px.pie(
            grouped,
            names=x,
            values=y_columns[0],
            hole=0.45,
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # GROUPED BAR
    # ------------------------------------------------------------

    elif chart_type == "grouped_bar":

        if not x or not y_columns:
            raise ValueError("grouped_bar requires x and y.")

        # x + one y + hue
        if hue:

            grouped = (
                df.groupby([x, hue], as_index=False)[y_columns[0]]
                .mean()
            )

            fig = px.bar(
                grouped,
                x=x,
                y=y_columns[0],
                color=hue,
                barmode="group",
                title=title,
                color_discrete_sequence=color_sequence
            )

        # x + multiple y
        elif len(y_columns) > 1:

            grouped = (
                df.groupby(x, as_index=False)[y_columns]
                .mean()
            )

            fig = px.bar(
                grouped,
                x=x,
                y=y_columns,
                barmode="group",
                title=title,
                color_discrete_sequence=color_sequence
            )

        # x + one y
        else:

            grouped = (
                df.groupby(x, as_index=False)[y_columns[0]]
                .mean()
            )

            fig = px.bar(
                grouped,
                x=x,
                y=y_columns[0],
                title=title,
                color_discrete_sequence=color_sequence
            )

    # ------------------------------------------------------------
    # STACKED BAR
    # ------------------------------------------------------------

    elif chart_type == "stacked_bar":

        if not x or not y_columns:
            raise ValueError("stacked_bar requires x and y.")

        grouped = (
            df.groupby(x, as_index=False)[y_columns]
            .mean()
        )

        fig = px.bar(
            grouped,
            x=x,
            y=y_columns,
            barmode="stack",
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # HEATMAP
    # ------------------------------------------------------------

    elif chart_type == "heatmap":

        if x and y_columns:

            pivot = pd.pivot_table(
                df,
                index=x,
                columns=hue if hue else None,
                values=y_columns[0],
                aggfunc="mean"
            )

            if isinstance(pivot, pd.Series):
                pivot = pivot.to_frame()

            fig = px.imshow(
                pivot,
                text_auto=annotate,
                aspect="auto",
                color_continuous_scale=colormap,
                title=title
            )

        else:

            numeric_df = df.select_dtypes(include=np.number)

            if numeric_df.empty:
                raise ValueError(
                    "heatmap requires numeric data."
                )

            fig = px.imshow(
                numeric_df.corr(),
                text_auto=annotate,
                aspect="auto",
                color_continuous_scale=colormap,
                title=title
            )

    # ------------------------------------------------------------
    # CORRELATION HEATMAP
    # ------------------------------------------------------------

    elif chart_type == "correlation_heatmap":

        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] < 2:
            raise ValueError(
                "correlation_heatmap requires at least "
                "two numeric columns."
            )

        correlation = numeric_df.corr()

        fig = px.imshow(
            correlation,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=colormap,
            zmin=-1,
            zmax=1,
            title=title
        )

    # ------------------------------------------------------------
    # DENSITY
    # ------------------------------------------------------------

    elif chart_type == "density":

        if not y_columns:
            raise ValueError("density requires y.")

        fig = px.density_contour(
            df,
            x=y_columns[0],
            color=hue,
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # ECDF
    # ------------------------------------------------------------

    elif chart_type == "ecdf":

        if not y_columns:
            raise ValueError("ecdf requires y.")

        fig = px.ecdf(
            df,
            x=y_columns[0],
            color=hue,
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # PARETO
    # ------------------------------------------------------------

    elif chart_type == "pareto":

        if not x or not y_columns:
            raise ValueError("pareto requires x and y.")

        grouped = (
            df.groupby(x)[y_columns[0]]
            .sum()
            .sort_values(ascending=False)
        )

        cumulative = (
            grouped.cumsum() / grouped.sum() * 100
        )

        fig = go.Figure()

        fig.add_bar(
            x=grouped.index.astype(str),
            y=grouped.values,
            name=y_columns[0]
        )

        fig.add_scatter(
            x=grouped.index.astype(str),
            y=cumulative.values,
            name="Cumulative %",
            yaxis="y2",
            mode="lines+markers"
        )

        fig.update_layout(
            title=title,
            yaxis2=dict(
                title="Cumulative Percentage (%)",
                overlaying="y",
                side="right",
                range=[0, 100]
            )
        )

    # ------------------------------------------------------------
    # ROLLING LINE
    # ------------------------------------------------------------

    elif chart_type == "rolling_line":

        if not x or not y_columns:
            raise ValueError(
                "rolling_line requires x and y."
            )

        grouped = (
            df.groupby(x)[y_columns[0]]
            .mean()
            .sort_index()
        )

        rolling = grouped.rolling(
            window=rolling_window,
            min_periods=1
        ).mean()

        fig = go.Figure()

        fig.add_scatter(
            x=grouped.index,
            y=grouped.values,
            mode="lines",
            name="Original"
        )

        fig.add_scatter(
            x=rolling.index,
            y=rolling.values,
            mode="lines",
            name=f"Rolling {rolling_window}"
        )

        fig.update_layout(title=title)

    # ------------------------------------------------------------
    # HEXBIN
    # ------------------------------------------------------------

    elif chart_type == "hexbin":

        if not x or not y_columns:
            raise ValueError("hexbin requires x and y.")

        fig = px.density_heatmap(
            df,
            x=x,
            y=y_columns[0],
            nbinsx=30,
            nbinsy=30,
            title=title,
            color_continuous_scale=colormap
        )

    # ------------------------------------------------------------
    # BUBBLE
    # ------------------------------------------------------------

    elif chart_type == "bubble":

        if not x or not y_columns:
            raise ValueError("bubble requires x and y.")

        fig = px.scatter(
            df,
            x=x,
            y=y_columns[0],
            size=size,
            color=hue,
            size_max=bubble_scale,
            title=title,
            color_discrete_sequence=color_sequence
        )

    # ------------------------------------------------------------
    # WATERFALL
    # ------------------------------------------------------------

    elif chart_type == "waterfall":

        if not x or not y_columns:
            raise ValueError("waterfall requires x and y.")

        grouped = (
            df.groupby(x)[y_columns[0]]
            .sum()
        )

        fig = go.Figure(
            go.Waterfall(
                x=grouped.index.astype(str),
                y=grouped.values
            )
        )

        fig.update_layout(title=title)

    # ============================================================
    # 8. UNKNOWN CHART
    # ============================================================

    else:

        supported_charts = [
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
        ]

        raise ValueError(
            f"Unsupported chart_type '{chart_type}'. "
            f"Supported charts: {supported_charts}"
        )

    # ============================================================
    # 9. COMMON PLOTLY STYLING
    # ============================================================

    fig.update_layout(
        height=height,
        width=width,
        title_font_size=16,
        hovermode="closest"
    )

    if rotation:
        fig.update_xaxes(tickangle=rotation)

    # ============================================================
    # 10. SAVE INTERACTIVE HTML
    # ============================================================

    chart_id = uuid.uuid4().hex

    os.makedirs("static/charts", exist_ok=True)

    chart_filename = f"{chart_id}.html"
    chart_path = os.path.join("static", "charts", chart_filename)

    fig.update_layout(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(
        color="#E5E7EB"
    )
    )
    fig.write_html(
        chart_path,
        include_plotlyjs="cdn",
        full_html=True
    )

    with open(chart_path, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace(
        "<body>",
        """
        <body style="
            margin: 0;
            padding: 0;
            background: #111827;
            overflow: hidden;
        ">
        """
    )

    with open(chart_path, "w", encoding="utf-8") as f:
        f.write(html)
    # ============================================================
    # 11. RETURN INFORMATION
    # ============================================================

    return {
        "status": "success",
        "message": "Interactive Plotly chart created successfully.",
        "chart_id": chart_id,
        "chart_type": chart_type,
        "result_id": result_id,
        "chart_path": chart_path,
        "chart_path": f"charts/{chart_filename}"
    }

#tool 4
def clean_data(
    
    table_name="data",
    operations=None
):
    """
    Clean a dataset using a predefined set of safe operations.

    Supported operations:
        - remove_duplicates
        - drop_null_rows
        - fill_numeric_nulls_mean
        - fill_numeric_nulls_median
        - fill_numeric_nulls_zero
        - fill_text_nulls
        - strip_whitespace
        - lowercase_text
        - uppercase_text

    Parameters
    ----------
    table_name : str
        Name of the DuckDB table to clean.

    operations : list
        List of cleaning operations to perform.

    Returns
    -------
    dict
        Summary of the cleaning operations performed.
    """

    if operations is None:
        operations = []

    # ---------------------------------------------------------
    # Validate table
    # ---------------------------------------------------------

    tables = conn.sql("SHOW TABLES").fetchdf()

    if table_name not in tables["name"].tolist():
        raise ValueError(
            f"Table '{table_name}' does not exist."
        )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    df = conn.sql(
        f'SELECT * FROM "{table_name}"'
    ).fetchdf()

    original_rows = len(df)
    original_columns = len(df.columns)

    changes = []

    # ---------------------------------------------------------
    # Perform cleaning operations
    # ---------------------------------------------------------

    for operation in operations:

        # ---------------------------------------------
        # Remove duplicate rows
        # ---------------------------------------------

        if operation == "remove_duplicates":

            before = len(df)

            df = df.drop_duplicates()

            removed = before - len(df)

            changes.append({
                "operation": operation,
                "rows_removed": removed
            })

        # ---------------------------------------------
        # Drop rows containing NULL values
        # ---------------------------------------------

        elif operation == "drop_null_rows":

            before = len(df)

            df = df.dropna()

            removed = before - len(df)

            changes.append({
                "operation": operation,
                "rows_removed": removed
            })

        # ---------------------------------------------
        # Fill numeric NULLs with mean
        # ---------------------------------------------

        elif operation == "fill_numeric_nulls_mean":

            numeric_columns = df.select_dtypes(
                include="number"
            ).columns

            filled = {}

            for column in numeric_columns:

                count = int(df[column].isna().sum())

                if count > 0:

                    mean_value = df[column].mean()

                    df[column] = df[column].fillna(
                        mean_value
                    )

                    filled[column] = count

            changes.append({
                "operation": operation,
                "columns": filled
            })

        # ---------------------------------------------
        # Fill numeric NULLs with median
        # ---------------------------------------------

        elif operation == "fill_numeric_nulls_median":

            numeric_columns = df.select_dtypes(
                include="number"
            ).columns

            filled = {}

            for column in numeric_columns:

                count = int(df[column].isna().sum())

                if count > 0:

                    median_value = df[column].median()

                    df[column] = df[column].fillna(
                        median_value
                    )

                    filled[column] = count

            changes.append({
                "operation": operation,
                "columns": filled
            })

        # ---------------------------------------------
        # Fill numeric NULLs with zero
        # ---------------------------------------------

        elif operation == "fill_numeric_nulls_zero":

            numeric_columns = df.select_dtypes(
                include="number"
            ).columns

            filled = {}

            for column in numeric_columns:

                count = int(df[column].isna().sum())

                if count > 0:

                    df[column] = df[column].fillna(0)

                    filled[column] = count

            changes.append({
                "operation": operation,
                "columns": filled
            })

        # ---------------------------------------------
        # Fill text NULLs
        # ---------------------------------------------

        elif operation == "fill_text_nulls":

            text_columns = df.select_dtypes(
                include=["object", "string"]
            ).columns

            filled = {}

            for column in text_columns:

                count = int(df[column].isna().sum())

                if count > 0:

                    df[column] = df[column].fillna(
                        "Unknown"
                    )

                    filled[column] = count

            changes.append({
                "operation": operation,
                "columns": filled
            })

        # ---------------------------------------------
        # Strip whitespace from text
        # ---------------------------------------------

        elif operation == "strip_whitespace":

            text_columns = df.select_dtypes(
                include=["object", "string"]
            ).columns

            affected = []

            for column in text_columns:

                df[column] = df[column].apply(
                    lambda value:
                    value.strip()
                    if isinstance(value, str)
                    else value
                )

                affected.append(column)

            changes.append({
                "operation": operation,
                "columns": affected
            })

        # ---------------------------------------------
        # Lowercase text
        # ---------------------------------------------

        elif operation == "lowercase_text":

            text_columns = df.select_dtypes(
                include=["object", "string"]
            ).columns

            for column in text_columns:

                df[column] = df[column].apply(
                    lambda value:
                    value.lower()
                    if isinstance(value, str)
                    else value
                )

            changes.append({
                "operation": operation,
                "columns": list(text_columns)
            })

        # ---------------------------------------------
        # Uppercase text
        # ---------------------------------------------

        elif operation == "uppercase_text":

            text_columns = df.select_dtypes(
                include=["object", "string"]
            ).columns

            for column in text_columns:

                df[column] = df[column].apply(
                    lambda value:
                    value.upper()
                    if isinstance(value, str)
                    else value
                )

            changes.append({
                "operation": operation,
                "columns": list(text_columns)
            })

        # ---------------------------------------------
        # Unknown operation
        # ---------------------------------------------

        else:

            raise ValueError(
                f"Unsupported cleaning operation: {operation}"
            )

    # ---------------------------------------------------------
    # Replace DuckDB table
    # ---------------------------------------------------------

    conn.unregister(table_name)

    conn.register(table_name, df)

    # ---------------------------------------------------------
    # Return summary
    # ---------------------------------------------------------

    return {
        "status": "success",
        "table": table_name,
        "original_rows": original_rows,
        "final_rows": len(df),
        "rows_removed": original_rows - len(df),
        "columns": original_columns,
        "null_values_remaining": int(
            df.isna().sum().sum()
        ),
        "operations_performed": changes
    }

#tool 5
def inspect_data_quality():

    df = conn.execute("SELECT * FROM data").fetchdf()

    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": {},
        "duplicate_rows": int(df.duplicated().sum())
    }

    for column in df.columns:

        report["columns"][column] = {
            "dtype": str(df[column].dtype),
            "missing": int(df[column].isna().sum()),
            "missing_percentage": round(
                df[column].isna().mean() * 100, 2
            ),
            "unique_values": int(df[column].nunique()),
        }

    return report

#tool 6
def feature_engineering(
    result_id,
    operation,
    new_column,
    column=None,
    column2=None,
    value=None
):
    """
    Create a derived column from an existing query result.

    Supported operations:
    - add
    - subtract
    - multiply
    - divide
    - percentage
    - log
    - absolute
    - year
    - month
    - quarter
    - day_of_week
    - length
    - uppercase
    - lowercase
    - bin
    """

    if result_id not in RESULTS:
        raise ValueError(f"Unknown result_id: {result_id}")

    df = RESULTS[result_id].copy()

    if new_column in df.columns:
        raise ValueError(
            f"Column '{new_column}' already exists."
        )

    if column and column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if column2 and column2 not in df.columns:
        raise ValueError(
            f"Column '{column2}' does not exist."
        )

    # -------------------------
    # Arithmetic features
    # -------------------------

    if operation == "add":
        df[new_column] = df[column] + df[column2]

    elif operation == "subtract":
        df[new_column] = df[column] - df[column2]

    elif operation == "multiply":
        df[new_column] = df[column] * df[column2]

    elif operation == "divide":
        df[new_column] = df[column] / df[column2].replace(0, pd.NA)

    # -------------------------
    # Percentage
    # -------------------------

    elif operation == "percentage":
        df[new_column] = (
            df[column] / df[column2].replace(0, pd.NA)
        ) * 100

    # -------------------------
    # Mathematical
    # -------------------------

    elif operation == "absolute":
        df[new_column] = df[column].abs()

    elif operation == "log":
        import numpy as np
        df[new_column] = np.log1p(df[column])

    # -------------------------
    # Date features
    # -------------------------

    elif operation == "year":
        dates = pd.to_datetime(df[column], errors="coerce")
        df[new_column] = dates.dt.year

    elif operation == "month":
        dates = pd.to_datetime(df[column], errors="coerce")
        df[new_column] = dates.dt.month

    elif operation == "quarter":
        dates = pd.to_datetime(df[column], errors="coerce")
        df[new_column] = dates.dt.quarter

    elif operation == "day_of_week":
        dates = pd.to_datetime(df[column], errors="coerce")
        df[new_column] = dates.dt.day_name()

    # -------------------------
    # Text features
    # -------------------------

    elif operation == "length":
        df[new_column] = df[column].astype(str).str.len()

    elif operation == "uppercase":
        df[new_column] = df[column].astype(str).str.upper()

    elif operation == "lowercase":
        df[new_column] = df[column].astype(str).str.lower()

    # -------------------------
    # Binning
    # -------------------------

    elif operation == "bin":
        if not value:
            raise ValueError(
                "For bin operation, provide bin boundaries in 'value'."
            )

        bins = value["bins"]
        labels = value["labels"]

        df[new_column] = pd.cut(
            df[column],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

    else:
        raise ValueError(
            f"Unsupported feature engineering operation: {operation}"
        )

    # Store updated result
    new_result_id = str(uuid.uuid4())[:8]

    RESULTS[new_result_id] = df

    return {
        "status": "success",
        "message": f"Feature '{new_column}' created successfully.",
        "result_id": new_result_id,
        "new_column": new_column,
        "operation": operation,
        "row_count": len(df),
        "columns": list(df.columns),
        "data": df.head(100).to_dict("records")
    }





