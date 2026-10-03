import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import uuid
from .database import conn, RESULTS
import os
import plotly.express as px
import plotly.graph_objects as go
import re





RESULTS = {}
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


# 
# DEFAULT COLOR PALETTE
# 

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

#TOOL-1 
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

#TOOL-2
def run_sql(query):

    query_clean = query.strip()

    # Remove empty statements
    statements = [
        statement.strip()
        for statement in query_clean.split(";")
        if statement.strip()
    ]

    if not statements:
        raise ValueError("Empty SQL query.")

    if len(statements) > 1:
        raise ValueError(
            "Only one SQL statement is allowed per tool call."
        )

    query_clean = statements[0]
    query_upper = query_clean.upper()

    # Block dangerous database operations
    forbidden = [
        "DROP",
        "CREATE",
        "TRUNCATE",
        "ATTACH",
        "DETACH"
    ]

    for keyword in forbidden:
        if re.search(rf"\b{keyword}\b", query_upper):
            raise ValueError(
                f"SQL operation '{keyword}' is not allowed."
            )

    # Execute SQL
    result = conn.execute(query_clean)

    # SELECT / WITH → return query result
    if query_upper.startswith(("SELECT", "WITH")):

        df = result.fetchdf()
        df = df.head(50)

        result_id = str(uuid.uuid4())[:8]
        RESULTS[result_id] = df

        return {
            "status": "success",
            "operation": "query",
            "result_id": result_id,
            "row_count": len(df),
            "columns": list(df.columns),
            "data": df.to_dict("records")
        }

    # UPDATE / DELETE → return updated dataset
    elif query_upper.startswith(("UPDATE", "DELETE","ALTER")):

        df = conn.execute(
            "SELECT * FROM data"
        ).fetchdf()

        df_preview = df.head(50)

        result_id = str(uuid.uuid4())[:8]
        RESULTS[result_id] = df

        return {
            "status": "success",
            "operation": "data_modified",
            "message": "SQL operation executed successfully.",
            "result_id": result_id,
            "row_count": len(df),
            "columns": list(df.columns),
            "data": df_preview.to_dict("records")
        }

    else:
        raise ValueError(
            "Only SELECT, WITH, UPDATE, and DELETE statements "
            "are allowed."
        )
#TOOL-3
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

    

    # 
    # 1. VALIDATE RESULT
    # 

    if result_id not in RESULTS:
        raise ValueError(f"Unknown result_id: {result_id}")

    df = RESULTS[result_id].copy()

    if df.empty:
        raise ValueError("The result contains no data to plot.")

    # 
    # 2. NORMALIZE INPUTS
    # 

    chart_type = str(chart_type).lower().strip()

    if isinstance(y, str):
        y_columns = [y]
    elif isinstance(y, (list, tuple)):
        y_columns = list(y)
    else:
        y_columns = []

    # 
    # 3. VALIDATE COLUMNS
    # 

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

    # 
    # 4. CREATE CHART DIRECTORY
    # 

    os.makedirs("charts", exist_ok=True)

    # 
    # 5. UNIQUE CHART ID
    # 

    chart_id = str(uuid.uuid4())

    chart_filename = f"{chart_id}.html"
    chart_path = os.path.join("charts", chart_filename)

    # 
    # 6. COLOR SETTINGS
    # 

    color_sequence = None

    if colors:
        if isinstance(colors, str):
            color_sequence = [colors]
        elif isinstance(colors, (list, tuple)):
            color_sequence = list(colors)

    elif palette:
        color_sequence = px.colors.qualitative.Plotly

    # 
    # 7. CREATE FIGURE
    # 

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

        if not x:
            raise ValueError("histogram requires x.")

        fig = px.histogram(
            df,
            x=x,
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

    # 
    # 8. UNKNOWN CHART
    # 

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

    # 
    # 9. COMMON PLOTLY STYLING
    # 

    fig.update_layout(
        height=height,
        width=width,
        title_font_size=16,
        hovermode="closest"
    )

    if rotation:
        fig.update_xaxes(tickangle=rotation)

    # 
    # 10. SAVE INTERACTIVE HTML
    # 

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
    # 
    # 11. RETURN INFORMATION
    # 

    return {
        "status": "success",
        "message": "Interactive Plotly chart created successfully.",
        "chart_id": chart_id,
        "chart_type": chart_type,
        "result_id": result_id,
        "chart_path": chart_path,
        "chart_path": f"charts/{chart_filename}"
    }


#TOOL-4
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

#TOOL-5
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

    df = RESULTS[result_id]

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

    
    conn.unregister("data")
    conn.register("data", df)

    return {
        "status": "success",
        "message": f"Feature '{new_column}' created successfully.",
        "result_id": result_id,
        "new_column": new_column,
        "operation": operation,
        "row_count": len(df),
        "columns": list(df.columns),
        "data": df.head(100).to_dict("records")
    }


