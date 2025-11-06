import mysql.connector
import pandas as pd
import base64

# חיבור למסד הנתונים
connection =mysql.connector.connect(
        host="localhost",
        user="root",
        password="your password",
        database="your database"
    )
cur = connection.cursor(dictionary=True)

# קלט מהמשתמש או קוד


def get_project_dataset_by_name(conn, Adress, city):
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT * FROM SolarProject WHERE Adress=%s AND City = %s LIMIT 1", (Adress, city))
    project = cur.fetchone()
    if not project:
        return None

    pid = project["ProjectID"]

    cur.execute("SELECT YearIndex, IncomeNIS FROM SolarAnnualIncome WHERE ProjectID=%s ORDER BY YearIndex", (pid,))
    annual = cur.fetchall()

    cur.execute("SELECT MonthIndex, ProfitNIS FROM SolarMonthlyProfit WHERE ProjectID=%s ORDER BY MonthIndex", (pid,))
    monthly = cur.fetchall()

    cur.execute("""
    SELECT FileName, MimeType, FileSizeBytes, ImageData
    FROM SolarProjectImage_initial
    WHERE ProjectID = %s
""", (pid,))
    initial_img = cur.fetchone()

    cur.execute("""
    SELECT FileName, MimeType, FileSizeBytes, ImageData
    FROM SolarProjectImage_mask
    WHERE ProjectID = %s
""", (pid,))
    mask_img = cur.fetchone()

    return {
        "project": project,
        "annual_income": annual,
        "monthly_profit": monthly,
        "initial_image": initial_img,
        "mask_image": mask_img,
    }
    
def pack_image(img_row):
        if not img_row:
            return None
        b64 = base64.b64encode(img_row["ImageData"]).decode("utf-8") if img_row["ImageData"] is not None else None
        return {
            "src": f"data:{img_row['MimeType']};base64,{b64}",  # מספיק להצגה
            "alt": img_row["FileName"], 
            "sizeBytes": img_row["FileSizeBytes"] 
        }

def output_dataset_as_arrays(Adress, city, connection = connection):
        dataset = get_project_dataset_by_name(connection, Adress, city)
        if not dataset:
            return None
        annual_income_df = pd.DataFrame(dataset["annual_income"])
        monthly_profit_df = pd.DataFrame(dataset["monthly_profit"])
        Project_Data = pd.DataFrame([dataset["project"]])

        array_annual = annual_income_df.values.tolist()
        array_monthly = monthly_profit_df.values.tolist()

        return {
            "array_annual": array_annual,
            "array_monthly": array_monthly,
            "initial_image": pack_image(dataset["initial_image"]),
            "mask_image": pack_image(dataset["mask_image"]),
            "Cost": float(Project_Data["InstallCostNIS"].values[0]),
            "PaybackYears": float(Project_Data["PaybackYears"].values[0]),
            "YearIncome": float(Project_Data["YearlyProfitNIS"].values[0]),
            "PanelsCount": int(Project_Data["PanelsCount"].values[0])
            }