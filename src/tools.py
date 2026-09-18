import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import uuid
from .database import conn, RESULTS


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

#Tool 3
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
    figsize=(10, 6),
    grid=True,
    grid_axis="both",
    legend=True,
    rotation=0,
    font_size=10,
    title_size=14,
    label_size=11,
    marker="o",
    linewidth=2,
    annotate=False,
    colormap="viridis",
    bins=20,
    rolling_window=7,
    bubble_scale=100,
    number=0,
    **kwargs
):

    import uuid
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

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

    # Allow y="avg_math" OR y=["avg_math", "avg_reading"]
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

    for col in y_columns:
        if col not in df.columns:
            raise ValueError(
                f"y column '{col}' does not exist. "
                f"Available columns: {list(df.columns)}"
            )

    # ============================================================
    # 4. FIX GEMINI'S COMMON HUE ERROR
    # ============================================================

    # x == hue creates a meaningless grouping and can break pivot().
    # Simply ignore hue in this situation.
    if hue == x:
        hue = None

    # ============================================================
    # 5. COLOR / PALETTE HELPER
    # ============================================================

    def get_colors(n=10):

        # Explicit colors take priority
        if colors:
            if isinstance(colors, str):
                return [colors] * n

            if isinstance(colors, (list, tuple)):
                if len(colors) >= n:
                    return list(colors)

                # Repeat colors if fewer were supplied
                repeated = []
                for i in range(n):
                    repeated.append(colors[i % len(colors)])
                return repeated

        # Single explicit color
        if color:
            return [color] * n

        # User-selected palette
        if palette:
            try:
                return sns.color_palette(palette, n).as_hex()
            except Exception:
                try:
                    cmap = plt.get_cmap(palette)
                    return [
                        cmap(i / max(n - 1, 1))
                        for i in range(n)
                    ]
                except Exception:
                    # Never crash because of a palette name
                    return sns.color_palette("deep", n).as_hex()

        # Default palette
        return sns.color_palette("deep", n).as_hex()

    # ============================================================
    # 6. CREATE FIGURE
    # ============================================================

    fig, ax = plt.subplots(figsize=figsize)

    # ============================================================
    # 7. BAR
    # ============================================================

    if chart_type == "bar":

        if not x or not y_columns:
            raise ValueError("bar requires x and y.")

        grouped = df.groupby(x)[y_columns[0]].mean()

        bars = ax.bar(
            grouped.index.astype(str),
            grouped.values,
            color=get_colors(len(grouped)),
            alpha=alpha
        )

        if annotate:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(
                        bar.get_x() + bar.get_width() / 2,
                        height
                    ),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=font_size
                )

    # ============================================================
    # 8. HORIZONTAL BAR
    # ============================================================

    elif chart_type == "horizontal_bar":

        if not x or not y_columns:
            raise ValueError("horizontal_bar requires x and y.")

        grouped = df.groupby(x)[y_columns[0]].mean()

        ax.barh(
            grouped.index.astype(str),
            grouped.values,
            color=get_colors(len(grouped)),
            alpha=alpha
        )

    # ============================================================
    # 9. LINE
    # ============================================================

    elif chart_type == "line":

        if not x or not y_columns:
            raise ValueError("line requires x and y.")

        for i, col in enumerate(y_columns):

            grouped = df.groupby(x)[col].mean()

            ax.plot(
                grouped.index,
                grouped.values,
                marker=marker,
                linewidth=linewidth,
                alpha=alpha,
                label=col
            )

        if len(y_columns) > 1 and legend:
            ax.legend()

    # ============================================================
    # 10. AREA
    # ============================================================

    elif chart_type == "area":

        if not x or not y_columns:
            raise ValueError("area requires x and y.")

        grouped = df.groupby(x)[y_columns].mean()

        ax.stackplot(
            grouped.index,
            *[
                grouped[col].values
                for col in y_columns
            ],
            labels=y_columns,
            alpha=alpha
        )

        if legend:
            ax.legend()

    # ============================================================
    # 11. SCATTER
    # ============================================================

    elif chart_type == "scatter":

        if not x or not y_columns:
            raise ValueError("scatter requires x and y.")

        if hue:

            categories = df[hue].dropna().unique()
            palette_colors = get_colors(len(categories))

            for category, c in zip(categories, palette_colors):

                subset = df[df[hue] == category]

                ax.scatter(
                    subset[x],
                    subset[y_columns[0]],
                    label=str(category),
                    color=c,
                    alpha=alpha
                )

            if legend:
                ax.legend()

        else:

            ax.scatter(
                df[x],
                df[y_columns[0]],
                color=get_colors(1)[0],
                alpha=alpha
            )

    # ============================================================
    # 12. HISTOGRAM
    # ============================================================

    elif chart_type == "histogram":

        if not y_columns:
            raise ValueError("histogram requires y.")

        ax.hist(
            df[y_columns[0]].dropna(),
            bins=bins,
            color=get_colors(1)[0],
            alpha=alpha
        )

    # ============================================================
    # 13. BOX
    # ============================================================

    elif chart_type == "box":

        if not y_columns:
            raise ValueError("box requires y.")

        sns.boxplot(
            data=df,
            y=y_columns[0],
            color=get_colors(1)[0],
            ax=ax
        )

    # ============================================================
    # 14. VIOLIN
    # ============================================================

    elif chart_type == "violin":

        if not y_columns:
            raise ValueError("violin requires y.")

        if x:

            sns.violinplot(
                data=df,
                x=x,
                y=y_columns[0],
                palette=get_colors(
                    df[x].nunique()
                ),
                ax=ax
            )

        else:

            sns.violinplot(
                data=df,
                y=y_columns[0],
                color=get_colors(1)[0],
                ax=ax
            )

    # ============================================================
    # 15. PIE
    # ============================================================

    elif chart_type == "pie":

        if not x or not y_columns:
            raise ValueError("pie requires x and y.")

        grouped = df.groupby(x)[y_columns[0]].sum()

        ax.pie(
            grouped.values,
            labels=grouped.index.astype(str),
            colors=get_colors(len(grouped)),
            autopct="%1.1f%%",
            startangle=90
        )

    # ============================================================
    # 16. DONUT
    # ============================================================

    elif chart_type == "donut":

        if not x or not y_columns:
            raise ValueError("donut requires x and y.")

        grouped = df.groupby(x)[y_columns[0]].sum()

        ax.pie(
            grouped.values,
            labels=grouped.index.astype(str),
            colors=get_colors(len(grouped)),
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": 0.4}
        )

    # ============================================================
    # 17. GROUPED BAR
    # ============================================================

    elif chart_type == "grouped_bar":

        if not x or not y_columns:
            raise ValueError("grouped_bar requires x and y.")

        # --------------------------------------------------------
        # CASE 1:
        # x = category
        # y = one metric
        # hue = None
        #
        # Example:
        # test_preparation_course | avg_math
        # --------------------------------------------------------

        if len(y_columns) == 1 and not hue:

            grouped = (
                df.groupby(x)[y_columns[0]]
                .mean()
            )

            bars = ax.bar(
                grouped.index.astype(str),
                grouped.values,
                color=get_colors(len(grouped)),
                alpha=alpha
            )

            if annotate:
                for bar in bars:

                    height = bar.get_height()

                    ax.annotate(
                        f"{height:.2f}",
                        xy=(
                            bar.get_x() + bar.get_width() / 2,
                            height
                        ),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        fontsize=font_size
                    )

        # --------------------------------------------------------
        # CASE 2:
        # multiple y metrics
        #
        # x = course
        # y = [math, reading, writing]
        # --------------------------------------------------------

        elif len(y_columns) > 1 and not hue:

            grouped = (
                df.groupby(x)[y_columns]
                .mean()
            )

            grouped.plot(
                kind="bar",
                ax=ax,
                alpha=alpha
            )

            if legend:
                ax.legend()

        # --------------------------------------------------------
        # CASE 3:
        # x + hue
        #
        # Example:
        # x = gender
        # hue = test_preparation_course
        # y = avg_math
        # --------------------------------------------------------

        else:

            grouped = (
                df.groupby([x, hue])[y_columns[0]]
                .mean()
                .unstack(fill_value=0)
            )

            grouped.plot(
                kind="bar",
                ax=ax,
                alpha=alpha
            )

            if legend:
                ax.legend(title=hue)

    # ============================================================
    # 18. STACKED BAR
    # ============================================================

    elif chart_type == "stacked_bar":

        if not x or not y_columns:
            raise ValueError("stacked_bar requires x and y.")

        grouped = (
            df.groupby(x)[y_columns]
            .mean()
        )

        grouped.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            alpha=alpha
        )

        if legend:
            ax.legend()

    # ============================================================
    # 19. HEATMAP
    # ============================================================

    elif chart_type == "heatmap":

        if x and y_columns and len(y_columns) >= 1:

            pivot = pd.pivot_table(
                df,
                index=x,
                columns=hue if hue else None,
                values=y_columns[0],
                aggfunc="mean"
            )

            if isinstance(pivot, pd.Series):
                pivot = pivot.to_frame()

            sns.heatmap(
                pivot,
                annot=annotate,
                cmap=colormap,
                ax=ax
            )

        else:

            numeric_df = df.select_dtypes(
                include=np.number
            )

            if numeric_df.empty:
                raise ValueError(
                    "heatmap requires numeric data."
                )

            sns.heatmap(
                numeric_df.corr(),
                annot=annotate,
                cmap=colormap,
                ax=ax
            )

    # ============================================================
    # 20. CORRELATION HEATMAP
    # ============================================================

    elif chart_type == "correlation_heatmap":

        numeric_df = df.select_dtypes(
            include=np.number
        )

        if numeric_df.shape[1] < 2:
            raise ValueError(
                "correlation_heatmap requires at least "
                "two numeric columns."
            )

        correlation = numeric_df.corr()

        sns.heatmap(
            correlation,
            annot=True,
            cmap=colormap,
            center=0,
            ax=ax
        )

    # ============================================================
    # 21. DENSITY
    # ============================================================

    elif chart_type == "density":

        if not y_columns:
            raise ValueError("density requires y.")

        sns.kdeplot(
            data=df,
            x=y_columns[0],
            fill=True,
            alpha=alpha,
            color=get_colors(1)[0],
            ax=ax
        )

    # ============================================================
    # 22. ECDF
    # ============================================================

    elif chart_type == "ecdf":

        if not y_columns:
            raise ValueError("ecdf requires y.")

        sns.ecdfplot(
            data=df,
            x=y_columns[0],
            color=get_colors(1)[0],
            ax=ax
        )

    # ============================================================
    # 23. PARETO
    # ============================================================

    elif chart_type == "pareto":

        if not x or not y_columns:
            raise ValueError("pareto requires x and y.")

        grouped = (
            df.groupby(x)[y_columns[0]]
            .sum()
            .sort_values(ascending=False)
        )

        cumulative = (
            grouped.cumsum()
            / grouped.sum()
            * 100
        )

        ax.bar(
            grouped.index.astype(str),
            grouped.values,
            color=get_colors(len(grouped)),
            alpha=alpha
        )

        ax2 = ax.twinx()

        ax2.plot(
            range(len(cumulative)),
            cumulative.values,
            marker=marker,
            linewidth=linewidth
        )

        ax2.set_ylabel("Cumulative Percentage (%)")

        ax2.axhline(
            80,
            linestyle="--",
            linewidth=1
        )

    # ============================================================
    # 24. ROLLING LINE
    # ============================================================

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

        ax.plot(
            grouped.index,
            grouped.values,
            alpha=0.35,
            label="Original"
        )

        ax.plot(
            rolling.index,
            rolling.values,
            linewidth=linewidth,
            label=f"Rolling {rolling_window}"
        )

        if legend:
            ax.legend()

    # ============================================================
    # 25. HEXBIN
    # ============================================================

    elif chart_type == "hexbin":

        if not x or not y_columns:
            raise ValueError("hexbin requires x and y.")

        hb = ax.hexbin(
            df[x],
            df[y_columns[0]],
            gridsize=30,
            cmap=colormap,
            mincnt=1
        )

        fig.colorbar(hb, ax=ax)

    # ============================================================
    # 26. BUBBLE
    # ============================================================

    elif chart_type == "bubble":

        if not x or not y_columns:
            raise ValueError("bubble requires x and y.")

        # Use first numeric column available as bubble size
        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        size_column = kwargs.get("size")

        if size_column and size_column in df.columns:
            sizes = df[size_column].abs()
        elif len(numeric_columns) >= 3:
            sizes = df[numeric_columns[2]].abs()
        else:
            sizes = pd.Series(
                1,
                index=df.index
            )

        ax.scatter(
            df[x],
            df[y_columns[0]],
            s=sizes * bubble_scale,
            alpha=alpha,
            color=get_colors(1)[0]
        )

    # ============================================================
    # 27. WATERFALL
    # ============================================================

    elif chart_type == "waterfall":

        if not x or not y_columns:
            raise ValueError("waterfall requires x and y.")

        grouped = (
            df.groupby(x)[y_columns[0]]
            .sum()
        )

        values = grouped.values
        cumulative = np.cumsum(
            np.insert(values, 0, 0)
        )

        starts = cumulative[:-1]

        ax.bar(
            range(len(values)),
            values,
            bottom=starts,
            alpha=alpha,
            color=get_colors(len(values))
        )

        ax.set_xticks(
            range(len(values))
        )

        ax.set_xticklabels(
            grouped.index.astype(str)
        )

    # ============================================================
    # 28. UNKNOWN CHART
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
    # 29. COMMON STYLING
    # ============================================================

    if title:
        ax.set_title(
            title,
            fontsize=title_size
        )

    if x:
        ax.set_xlabel(
            x,
            fontsize=label_size
        )

    if y_columns:
        ax.set_ylabel(
            ", ".join(y_columns),
            fontsize=label_size
        )

    ax.tick_params(
        axis="both",
        labelsize=font_size
    )

    if rotation:
        plt.xticks(
            rotation=rotation
        )

    # Grid
    if grid:
        ax.grid(
            True,
            axis=grid_axis,
            alpha=0.3
        )
    else:
        ax.grid(False)

    plt.tight_layout()

    # ============================================================
    # 30. SAVE CHART
    # ============================================================

    chart_path = f"charts/chart{number}.png"

    plt.savefig(
        chart_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return {
        "status": "success",
        "message": "Chart created successfully.",
        "chart_type": chart_type,
        "result_id": result_id,
        "chart_path": chart_path
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





