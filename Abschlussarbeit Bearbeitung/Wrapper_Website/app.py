from pathlib import Path
import re

from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import core_processing

# Core Processing Python import
from core_processing import (
    finalize_global_schema,
    TARGET_COLUMNS,
)

# ---------------------------------------

# All target colums except Database_number & Database_name
MAPPING_COLUMNS = TARGET_COLUMNS[:-2]

# Standard-Python-Code
DEFAULT_PYTHON_CODE = (
    "# Edit this code to modify the DataFrame `df` before mapping.\n"
    "# Example code\n"
    "import pandas as pd\n"

)

# ------------------------------------

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def normalize_colname(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    download_file = None

    if request.method == "POST":
        
        # Read File
        file = request.files.get("excel_file")
        sheet_name_raw = request.form.get("sheet_name", "").strip()

        database_name = request.form.get("database_name", "").strip()
        database_number = request.form.get("database_number", "").strip()
        python_code = request.form.get("python_code") or DEFAULT_PYTHON_CODE

        if not file or file.filename == "":
            error = "Select your Excel file"
            return render_template(
                "index.html",
                mapping_columns=MAPPING_COLUMNS,
                error=error,
                download_file=None,
                python_code=python_code,
                database_name=database_name,
                database_number=database_number,
            )

        upload_path = UPLOAD_DIR / file.filename
        file.save(upload_path)

        # --------------------- Excel and Sheet ---------------------

        # Try open excel
        try:
            xls = pd.ExcelFile(upload_path)
        except Exception as e:
            error = f"Error on reacding excel-file: {e}"
            return render_template(
                "index.html",
                mapping_columns=MAPPING_COLUMNS,
                error=error,
                download_file=None,
                python_code=python_code,
                database_name=database_name,
                database_number=database_number,
            )

        # Find Sheet
        if sheet_name_raw:
            sheet_name = sheet_name_raw
        else:
            sheet_name = xls.sheet_names[0]

        # ------------------- Mapping ---------------------

        # Mapping from HTML-Table
        mapping = {}
        for i, col in enumerate(MAPPING_COLUMNS):
            src_name = request.form.get(f"src_{i}", "").strip()
            if src_name:
                mapping[normalize_colname(src_name)] = col

        if not mapping:
            error = "No Mapping written down"
            return render_template(
                "index.html",
                mapping_columns=MAPPING_COLUMNS,
                error=error,
                download_file=None,
                python_code=python_code,
                database_name=database_name,
                database_number=database_number,
            )

        # ------------------------ Dataframe loading and customized code

        try:
            df = pd.read_excel(upload_path, sheet_name=sheet_name)

            exec_locals = {"df": df, "pd": pd, "core_processing": core_processing}
            try:
                exec(python_code, {}, exec_locals)
                df = exec_locals.get("df", df)
            except Exception as ue:
                error = f"Error in your custom Python code: {ue}"
                return render_template(
                    "index.html",
                    mapping_columns=MAPPING_COLUMNS,
                    error=error,
                    download_file=None,
                    python_code=python_code,
                    database_name=database_name,
                    database_number=database_number,
                )

            df = core_processing._normalize_columns(df)
            df = core_processing._apply_special_source_fixes(df)

            if (
                "dissolved_oxygen_mg_per_l" in df.columns
                and "dissolved_oxigen_mg_per_l" not in df.columns
            ):
                df = df.rename(
                    columns={"dissolved_oxygen_mg_per_l": "dissolved_oxigen_mg_per_l"}
                )

            ren = core_processing._available_mapping(df, mapping)
            if not ren:
                error = "None of the mapped columns were found in the Excel sheet."
                return render_template(
                    "index.html",
                    mapping_columns=MAPPING_COLUMNS,
                    error=error,
                    download_file=None,
                    python_code=python_code,
                    database_name=database_name,
                    database_number=database_number,
                )

            df = df.rename(columns=ren)
            df = df[list(ren.values())]

            # automatic conversion and TDS
            df = finalize_global_schema(df)

            # ------------- Database info in each row -------------

            if database_name:
                df["Database_name"] = database_name
            if database_number:
                df["Database_number"] = database_number

            # Columns in standardized order
            df = df[TARGET_COLUMNS]

        except Exception as e:
            error = f"Error on transformation: {e}"
            return render_template(
                "index.html",
                mapping_columns=MAPPING_COLUMNS,
                error=error,
                download_file=None,
                python_code=python_code,
                database_name=database_name,
                database_number=database_number,
            )

        # ------------- Safe csv with same name as excel file ----------

        output_name = Path(file.filename).with_suffix(".csv").name
        output_path = OUTPUT_DIR / output_name
        df.to_csv(output_path, index=False)
        download_file = output_name

        return render_template(
            "index.html",
            mapping_columns=MAPPING_COLUMNS,
            error=None,
            download_file=download_file,
            python_code=python_code,
            database_name=database_name,
            database_number=database_number,
        )

    # GET: Site initil
    return render_template(
        "index.html",
        mapping_columns=MAPPING_COLUMNS,
        error=None,
        download_file=None,
        python_code=DEFAULT_PYTHON_CODE,
        database_name="",
        database_number="",
    )


@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
