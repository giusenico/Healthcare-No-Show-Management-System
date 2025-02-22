# streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import os
import json
import re
import logging
from datetime import datetime, time, timedelta
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== CONFIGURATION ==========
openai_api_key = os.getenv(
    "OPENAI_API_KEY",
    ""
)
GPT_MODEL_NAME = "gpt-4o-mini"
API_URL = "http://localhost:8000/predict"
BASE_OVERBOOKING_PROBABILITY_SUM_THRESHOLD = 2.5  # Base threshold for overbooking suggestions
OVERBOOKING_THRESHOLD_INCREMENT_FACTOR = 0.75  # Increment factor per extra patient beyond 3
CACHE_FILE = "predictions_cache.json"
DATA_UPDATED = "data/updated_dataset.csv"
ORIGINAL_DATA = "data/Mock_Healthcare_Appointments.csv"

openai.api_key = openai_api_key

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)
logger.info("Application started")

# ========== PAGE CONFIGURATION & CSS ==========
st.set_page_config(layout="wide", page_title="Healthcare Appointments")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
        background-color: #f8f9fa;
    }
    .stDataFrame table {
        width: 100% !important;
        table-layout: fixed;
        border-collapse: collapse;
    }
    .stDataFrame th {
        background-color: #e9ecef;
        color: #495057;
        padding: 0.75rem;
        border: 1px solid #dee2e6;
        text-align: left;
    }
    .stDataFrame td {
        padding: 0.75rem;
        border: 1px solid #dee2e6;
    }
    .stDataFrame tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .stTextInput > div > div > input {
        border: 2px solid #ced4da;
        border-radius: 0.25rem;
        padding: 0.5rem 0.75rem;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 0.25rem;
        padding: 0.6rem 1rem;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .stSelectbox > div > div > div {
        border: 2px solid #ced4da;
        border-radius: 0.25rem;
        overflow: hidden;
    }
    .stDateInput > div > div > input {
        border: 2px solid #ced4da;
        border-radius: 0.25rem;
        padding: 0.5rem 0.75rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #343a40;
    }
    .stSuccess {
        color: #28a745;
    }
    .stWarning, .stInfo {
        color: #ffc107;
    }
    .stError {
        color: #dc3545;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== FILE & CACHE FUNCTIONS ==========
def save_cache(data: dict) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f)
        logger.info("Cache saved successfully.")
    except Exception as e:
        st.error(f"Error saving cache to file: {e}")
        logger.error(f"Error saving cache: {e}")

def load_cache() -> dict:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
                logger.info("Cache loaded successfully.")
                return data
    except Exception as e:
        st.error(f"Error loading cache from file: {e}")
        logger.error(f"Error loading cache: {e}")
    return {}

def get_file_mod_time(filepath: str) -> float:
    return os.path.getmtime(filepath) if os.path.exists(filepath) else 0

# Define a hash function that uses file modification time for caching
def _hash_file_path(filepath):
    return os.path.getmtime(filepath) if filepath else None

@st.cache_data(show_spinner=False, hash_funcs={str: _hash_file_path})
def load_patients_data_cached(file_to_load: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading patient data from '{file_to_load}'...")
        df = pd.read_csv(file_to_load)
        df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="coerce")
        df.dropna(subset=["AppointmentDay"], inplace=True)
        if "AppointmentTime" in df.columns:
            df["AppointmentTime"] = pd.to_datetime(
                df["AppointmentTime"], errors="coerce", format="%H:%M"
            ).dt.strftime("%H:%M").fillna("-")
        else:
            df["AppointmentTime"] = "-"
        # Ensure the dataset contains a "Phone" column for patient contact information
        if "Phone" not in df.columns:
            df["Phone"] = "N/A"
        logger.info(f"Patient data loaded successfully from '{file_to_load}'.")
        return df
    except FileNotFoundError:
        logger.error(f"Data file '{file_to_load}' not found.")
        st.error(f"Error: Data file '{file_to_load}' not found. Please check the file path.")
        st.stop()
    except pd.errors.ParserError:
        logger.error(f"Could not parse CSV file '{file_to_load}'.")
        st.error(f"Error: Could not parse CSV file '{file_to_load}'. Please ensure it is a valid CSV format.")
        st.stop()
    except Exception as e:
        logger.exception("An unexpected error occurred while loading data:")
        st.error(f"An unexpected error occurred while loading data: {e}")
        st.stop()

def load_patients_data() -> pd.DataFrame:
    file_to_load = DATA_UPDATED if os.path.exists(DATA_UPDATED) else ORIGINAL_DATA
    with st.spinner(f"Loading patient data from '{file_to_load}'..."):
        return load_patients_data_cached(file_to_load)

# ========== PREDICTION FUNCTIONS ==========
weekday_map = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday",
    3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
}

# Create a global session to reuse HTTP connections
session = requests.Session()

def predict_one(index: int, row: pd.Series) -> tuple:
    gender = "Male" if row.get("Gender", "M") == "M" else "Female"
    payload = {
        "gender": gender,
        "age": int(row["Age"]),
        "days_until_appointment": int(row["days_from_ref"]),
        "distance_km": float(row["Distance_Hospital_km"]),
        "wealth_level": row["Wealth_Level"],
        "scholarship": bool(row["Scholarship"]),
        "hypertension": bool(row["Hipertension"]),
        "diabetes": bool(row["Diabetes"]),
        "alcoholism": bool(row["Alcoholism"]),
        "handicap": bool(row["Handcap"]),
        "sms_received": bool(row["SMS_received"] if "SMS_received" in row else False),
        "weekday": weekday_map.get(int(row["Weekday"]) if not pd.isnull(row["Weekday"]) else 0, "Monday")
    }
    try:
        r = session.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        js = r.json()
        return index, js.get("prediction", "Error"), js.get("probability_no_show", None)
    except requests.exceptions.Timeout as e:
        st.error(f"Prediction request timed out for index {index}: {e}")
        logger.warning(f"Prediction request timed out for index {index}: {e}")
        return index, "Error", None
    except requests.exceptions.ConnectionError as e:
        st.error(f"Prediction request connection error for index {index}: {e}")
        logger.warning(f"Prediction request connection error for index {index}: {e}")
        return index, "Error", None
    except requests.exceptions.RequestException as e:
        st.error(f"Error during prediction request for index {index}: {e}")
        logger.error(f"Request exception during prediction for index {index}: {e}")
        return index, "Error", None
    except Exception as e:
        st.error(f"Error during prediction for index {index}: {e}")
        logger.exception(f"Unexpected error during prediction for index {index}:")
        return index, "Error", None

def compute_predictions_parallel(df: pd.DataFrame, max_workers: int = 5) -> pd.DataFrame:
    """
    Compute predictions in parallel.
    """
    df_copy = df.copy()
    total = len(df_copy)
    pb = st.progress(0, text="Calculating No-Show Predictions...")
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, row in df_copy.iterrows():
            futures.append(executor.submit(predict_one, i, row))
        done_cnt = 0
        for fut in as_completed(futures):
            idx, pred, prob = fut.result()
            df_copy.at[idx, "Prediction"] = pred
            df_copy.at[idx, "Probability_NoShow"] = prob
            done_cnt += 1
            pb.progress(int(done_cnt / total * 100),
                        text=f"Calculating No-Show Predictions... ({done_cnt}/{total} completed)")
    pb.empty()
    st.success("No-Show Predictions calculated successfully!")
    return df_copy

# ========== GPT SUGGESTIONS FOR OVERBOOKING ==========
def get_gpt4_suggestions_for_overbooking(dataset_summary: str, dynamic_threshold: float) -> list:
    """
    Send a prompt to the GPT model including the dynamic threshold.
    """
    prompt = (
        "You are an expert scheduling assistant specialized in overbooking for healthcare appointments. "
        "Analyze the following summary of appointment slots and patient data. For each slot, note the 'Probability Sum' "
        "which indicates the chance of no-shows, and also check the SMS_received status. "
        "If there are appointments with high no-show probability (Probability_NoShow >= 0.6) and SMS_received is False, "
        "include a recommendation to send an SMS reminder to that patient (specify the PatientId). "
        "Also, suggest specific date and time slots suitable for overbooking if a slot shows a high 'Probability Sum'. "
        "Return suggestions in JSON format, where each suggestion has the following fields: 'type' (either 'overbooking' "
        "or 'send_reminder'), 'date', 'from_hour', 'to_hour' (if applicable), 'capacity' (if applicable), 'patient_id' (if applicable), and 'reason'. "
        f"For overbooking, use capacity 1 if Probability Sum >= {dynamic_threshold:.2f} and capacity 2 if Probability Sum >= {dynamic_threshold + 1:.2f}. "
        "Return an empty JSON array if no suggestions. Ensure valid JSON output.\n\n"
        f"Data summary:\n{dataset_summary}\n\n"
        "Example JSON Response:\n"
        "[\n  {\"type\": \"overbooking\", \"date\": \"2024-07-15\", \"from_hour\": 8, \"to_hour\": 9, \"capacity\": 1, \"reason\": \"Probability Sum of 2.8 suggests capacity for 1 extra patient.\"},\n"
        "  {\"type\": \"send_reminder\", \"date\": \"2024-07-15\", \"patient_id\": \"123\", \"reason\": \"Patient with high no-show probability has not received an SMS reminder.\"}\n]"
    )
    suggestions = []
    try:
        logger.info("GPT Prompt sent:\n" + prompt)
        with st.spinner("Getting suggestions from AI..."):
            response = openai.ChatCompletion.create(
                model=GPT_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert scheduling assistant specialized in overbooking."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=4000,
                n=1,
                timeout=20
            )
            content = response['choices'][0]['message']['content']
            logger.info("GPT Raw Response:\n" + content)

            def parse_json_response(text: str) -> list:
                try:
                    result = json.loads(text)
                    return result if isinstance(result, list) else []
                except json.JSONDecodeError as e:
                    logger.warning(f"JSONDecodeError parsing GPT response: {e}. Raw response: {text}")
                    match = re.search(r'\[.*\]', text, re.DOTALL)
                    if match:
                        logger.info(f"Regex matched for JSON: {match.group()}")
                        try:
                            result = json.loads(match.group())
                            return result if isinstance(result, list) else []
                        except Exception as e_nested:
                            logger.error(f"Nested JSON parse error after regex: {e_nested}. Regex matched: {match.group()}")
                            return []
                    logger.error("No JSON array found in GPT response even with regex.")
                    return []
            suggestions = parse_json_response(content)
        if suggestions:
            logger.info(f"GPT suggestions obtained: {suggestions}")
        else:
            st.info("AI did not find any suggestions based on the current data.")
            logger.info("GPT returned no suggestions.")
    except openai.APIError as e:
        st.error(f"Error communicating with OpenAI API: {e}")
        logger.error(f"OpenAI APIError: {e}")
    except Exception as e:
        st.error(f"Unexpected error while getting AI suggestions: {e}")
        logger.exception("Unexpected error during GPT suggestion request:")
    return suggestions

# ========== APPOINTMENT UPDATE ==========
def update_appointment(df: pd.DataFrame, patient_id: str, new_date: str, new_hour: int = None, new_time_str: str = None) -> pd.DataFrame:
    """
    Update an appointment's date and time, then recalculate its prediction.
    """
    df2 = df.copy()
    idxs = df2.index[df2["PatientId"].astype(str) == patient_id].tolist()
    if not idxs:
        st.error("Patient not found!")
        return df2
    i = idxs[0]
    if new_date:
        df2.at[i, "AppointmentDay"] = new_date
    if new_time_str:
        df2.at[i, "AppointmentTime"] = new_time_str
    elif new_hour is not None:
        df2.at[i, "AppointmentTime"] = f"{new_hour:02d}:00"
    refd = pd.Timestamp("today").normalize()
    try:
        new_day = pd.to_datetime(df2.at[i, "AppointmentDay"])
        dfr = (new_day - refd).days
    except Exception:
        dfr = 0
    df2.at[i, "days_from_ref"] = dfr
    gender = "Male" if df2.at[i, "Gender"] == "M" else "Female"
    payload = {
        "gender": gender,
        "age": int(df2.at[i, "Age"]),
        "days_until_appointment": int(dfr),
        "distance_km": float(df2.at[i, "Distance_Hospital_km"]),
        "wealth_level": df2.at[i, "Wealth_Level"],
        "scholarship": bool(df2.at[i, "Scholarship"]),
        "hypertension": bool(df2.at[i, "Hipertension"]),
        "diabetes": bool(df2.at[i, "Diabetes"]),
        "alcoholism": bool(df2.at[i, "Alcoholism"]),
        "handicap": bool(df2.at[i, "Handcap"]),
        "sms_received": bool(df2.at[i, "SMS_received"] if "SMS_received" in df2.columns else False),
        "weekday": weekday_map.get(int(df2.at[i, "Weekday"]) if not pd.isnull(df2.at[i, "Weekday"]) else 0, "Monday")
    }
    try:
        r = session.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        out = r.json()
        df2.at[i, "Prediction"] = out.get("prediction", "N/A")
        df2.at[i, "Probability_NoShow"] = out.get("probability_no_show", None)
        logger.info(f"Prediction updated for Patient ID {patient_id}.")
    except requests.exceptions.Timeout as e:
        st.error(f"Error updating prediction for Patient ID {patient_id}: Network issue - {e}")
        logger.warning(f"Prediction update timed out for Patient ID {patient_id}: {e}")
        df2.at[i, "Prediction"] = "Error"
        df2.at[i, "Probability_NoShow"] = None
    except requests.exceptions.ConnectionError as e:
        st.error(f"Error updating prediction for Patient ID {patient_id}: Network issue - {e}")
        logger.warning(f"Prediction update connection error for Patient ID {patient_id}: {e}")
        df2.at[i, "Prediction"] = "Error"
        df2.at[i, "Probability_NoShow"] = None
    except requests.exceptions.RequestException as e:
        st.error(f"Error updating prediction for Patient ID {patient_id}: {e}")
        logger.error(f"Request exception during prediction update for Patient ID {patient_id}: {e}")
        df2.at[i, "Prediction"] = "Error"
        df2.at[i, "Probability_NoShow"] = None
    except Exception as e:
        st.error(f"Error updating prediction for Patient ID {patient_id}: {e}")
        logger.exception(f"Unexpected error during prediction update for Patient ID {patient_id}:")
        df2.at[i, "Prediction"] = "Error"
        df2.at[i, "Probability_NoShow"] = None

    try:
        df2.to_csv(DATA_UPDATED, index=False)
        logger.info(f"Dataset updated and saved to '{DATA_UPDATED}'.")
    except Exception as e:
        st.error(f"Error saving updated dataset to file: {e}")
        logger.error(f"Error saving updated dataset: {e}")

    return df2

# ========== NEW FUNCTIONALITY: SEND REMINDER ==========
def send_reminder(patient_id: str, phone: str, appointment_date: str, appointment_time: str) -> None:
    """
    Simulate sending an SMS reminder and update the dataset by setting SMS_received to True.
    """
    try:
        message = (f"Reminder sent to Patient ID {patient_id} on phone {phone} "
                   f"for appointment on {appointment_date} at {appointment_time}.")
        logger.info(message)
        st.success(message)
        # Update the dataset in session_state and persistent CSV
        if "predictions_data" in st.session_state:
            df = pd.read_json(st.session_state["predictions_data"], orient="split")
            df.loc[df["PatientId"].astype(str) == patient_id, "SMS_received"] = True
            st.session_state["predictions_data"] = df.to_json(date_format="iso", orient="split")
            df.to_csv(DATA_UPDATED, index=False)
            logger.info(f"Dataset updated: SMS_received set to True for Patient ID {patient_id}.")
    except Exception as e:
        st.error(f"Error sending reminder: {e}")
        logger.error(f"Error sending reminder for Patient ID {patient_id}: {e}")

# ==================== MAIN APPLICATION ====================
st.title("Healthcare Appointments")

# ----- Data Loading & Preparation -----
df_pat = load_patients_data()
refdate = pd.Timestamp("today").normalize()
df_pat["days_from_ref"] = (df_pat["AppointmentDay"] - refdate).dt.days

view_option = st.radio("View appointments for:", ("One Week", "One Month"))
if view_option == "One Week":
    df_f = df_pat.query("0 <= days_from_ref <= 7").copy()
else:
    df_f = df_pat.query("0 <= days_from_ref <= 30").copy()

if df_f.empty:
    st.warning("No appointments available for the selected period.")
    st.stop()

df_f.sort_values(by=["AppointmentDay", "AppointmentTime"], inplace=True)
if "Prediction" not in df_f.columns:
    df_f["Prediction"] = ""
if "Probability_NoShow" not in df_f.columns:
    df_f["Probability_NoShow"] = None

cur_mod_t = get_file_mod_time(DATA_UPDATED)
cache_d = load_cache()

if view_option == "One Week":
    if cache_d and cache_d.get("mod_time") == cur_mod_t:
        st.session_state["predictions_data"] = cache_d["predictions_data"]
    if st.button("Calculate No-Show Predictions (One Week)"):
        df_p = compute_predictions_parallel(df_f, max_workers=5)
        p_j = df_p.to_json(date_format="iso", orient="split")
        st.session_state["predictions_data"] = p_j
        save_cache({"predictions_data": p_j, "mod_time": cur_mod_t})
        st.rerun()
else:
    st.session_state["predictions_data"] = df_f.to_json(date_format="iso", orient="split")

if "predictions_data" not in st.session_state:
    st.session_state["predictions_data"] = df_f.to_json(date_format="iso", orient="split")

df_view = pd.read_json(st.session_state["predictions_data"], orient="split")

# ---------- Appointments Display with Filters ----------
st.subheader("Appointments List")

# --- Filters ---
col1, col2 = st.columns(2)
with col1:
    neighborhood_filter = st.selectbox(
        "Filter by Neighborhood (All for none):",
        ["All"] + sorted(df_view["Neighbourhood"].unique().tolist())
    )
with col2:
    date_filter = st.date_input("Filter by Date (leave empty for none):", value=None)

# Main display block (inside __main__ to encapsulate code)
if __name__ == '__main__':
    df_filtered = df_view.copy()

    if neighborhood_filter != "All":
        df_filtered = df_filtered[df_filtered["Neighbourhood"] == neighborhood_filter]

    if date_filter:
        df_filtered["AppointmentDay"] = pd.to_datetime(df_filtered["AppointmentDay"]).dt.date
        df_filtered = df_filtered[df_filtered["AppointmentDay"] == date_filter]
        df_filtered["AppointmentDay"] = pd.to_datetime(df_filtered["AppointmentDay"])

    if df_filtered.empty:
        st.info("No appointments match the current filters.")
    else:
        page_size = st.selectbox("Patients per page", [10, 20, 50, 100], index=0)
        total = len(df_filtered)
        total_pages = (total - 1) // page_size + 1
        page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start_index = (page_number - 1) * page_size
        end_index = start_index + page_size
        df_page_display = df_filtered.iloc[start_index:end_index].copy()

        include_cols_patient_table = [
            "Patient ID", "Date", "Time", "Gender", "Neighborhood",
            "Scholarship", "Hypertension", "Diabetes", "Alcoholism", "Handicap",
            "SMS Received", "Prediction", "No-Show Probability"
        ]

        exclude_cols = ["AppointmentID", "Showed_up", "days_from_ref", "Distance_Hospital_km", "Weekday", "Age", "Wealth_Level"]
        df_page = df_page_display[[col for col in df_page_display.columns if col not in exclude_cols]]

        rename_map = {
            "PatientId": "Patient ID",
            "Gender": "Gender",
            "AppointmentDay": "Date",
            "AppointmentTime": "Time",
            "Neighbourhood": "Neighborhood",
            "Scholarship": "Scholarship",
            "Hipertension": "Hypertension",
            "Diabetes": "Diabetes",
            "Alcoholism": "Alcoholism",
            "Handcap": "Handicap",
            "SMS_received": "SMS Received",
            "Prediction": "Prediction",
            "Probability_NoShow": "No-Show Probability",
            "Wealth_Level": "Wealth Level"
        }
        df_page.rename(columns=rename_map, inplace=True)

        df_page_display = df_page[include_cols_patient_table]

        if "Date" in df_page_display.columns:
            df_page_display["Date"] = pd.to_datetime(df_page_display["Date"]).dt.strftime("%Y-%m-%d")
        if "No-Show Probability" in df_page_display.columns:
            df_page_display["No-Show Probability"] = df_page_display["No-Show Probability"].apply(
                lambda x: f"{x:.1%}" if pd.notnull(x) else "-"
            )

        def highlight(row):
            pred_text = str(row.get("Prediction", "")).lower().strip()
            if pred_text in ["no-show", "no show"]:
                return ["background-color: #FFEEEE"] * len(row)
            return ["" for _ in row]

        st.dataframe(
            df_page_display.style.apply(highlight, axis=1).format(na_rep="-"),
            use_container_width=True
        )
        st.write(f"Page {page_number} of {total_pages} - Total: {total}")

    # ---------- Staff Actions ----------
    st.subheader("Staff Actions")

    # --- Send Reminder ---
    st.markdown("#### Send Appointment Reminder")
    reminder_patients = df_view.copy()
    if not reminder_patients.empty:
        reminder_options = reminder_patients.apply(
            lambda r: f"{r['PatientId']} - {r['AppointmentDay'][:10]} {r.get('AppointmentTime', '-')}, Phone: {r.get('Phone', 'N/A')}",
            axis=1
        ).tolist()
        selected_reminder = st.selectbox("Select an appointment to send reminder:", reminder_options)
        selected_reminder_id = selected_reminder.split(" - ")[0]
        patient_reminder = reminder_patients[reminder_patients["PatientId"].astype(str) == selected_reminder_id].iloc[0]
        if st.button("Send Reminder"):
            send_reminder(
                patient_id=selected_reminder_id,
                phone=patient_reminder.get("Phone", "N/A"),
                appointment_date=str(patient_reminder["AppointmentDay"])[:10],
                appointment_time=patient_reminder.get("AppointmentTime", "-")
            )

    # ---------- Manual Overbooking Management ----------
    st.subheader("Manual Overbooking Management")
    patient_options = df_view.apply(
        lambda r: f"{r['PatientId']} - {str(r['AppointmentDay'])[:10]} - {r.get('AppointmentTime', '-')}",
        axis=1
    ).tolist()
    if patient_options:
        selected = st.selectbox("Select patient to reschedule:", patient_options)
        selected_id = selected.split(" - ")[0]
        available_times = []
        start_dt = datetime.combine(datetime.today(), time(8, 0))
        end_dt = datetime.combine(datetime.today(), time(18, 0))
        while start_dt <= end_dt:
            available_times.append(start_dt.strftime("%H:%M"))
            start_dt += timedelta(minutes=20)
        new_date = st.date_input("New date for appointment:")
        new_time = st.selectbox("New time for appointment (20-min intervals):", available_times)
        if st.button("Update Appointment"):
            with st.spinner("Updating appointment..."):
                df_view = update_appointment(df_view, selected_id, new_date.strftime("%Y-%m-%d"), new_time_str=new_time)
                pj = df_view.to_json(date_format="iso", orient="split")
                st.session_state["predictions_data"] = pj
                save_cache({"predictions_data": pj, "mod_time": get_file_mod_time(DATA_UPDATED)})
            st.success("Appointment updated successfully!")
            st.rerun()

    # ---------- Overbooking Suggestions (GPT) ----------
    if view_option == "One Month":
        st.warning("GPT suggestions not available for the 'One Month' interval.")
    else:
        st.subheader("Overbooking Suggestions (GPT)")
        # Prepare data for grouping
        df_g = pd.read_json(st.session_state["predictions_data"], orient="split").copy()
        df_g["DateOnly"] = pd.to_datetime(df_g["AppointmentDay"]).dt.date
        try:
            df_g["Hour"] = pd.to_datetime(df_g["AppointmentTime"], format="%H:%M", errors="coerce").dt.hour
        except Exception:
            df_g["Hour"] = None

        def group_data(g):
            total_appts = len(g)
            dynamic_threshold = BASE_OVERBOOKING_PROBABILITY_SUM_THRESHOLD + max(0, total_appts - 3) * OVERBOOKING_THRESHOLD_INCREMENT_FACTOR
            probability_sum = g["Probability_NoShow"].sum()
            potential_overbooking = probability_sum > dynamic_threshold
            overbook_capacity = 0
            if probability_sum >= dynamic_threshold + 1:
                overbook_capacity = 2
            elif probability_sum >= dynamic_threshold:
                overbook_capacity = 1
            patient_details = []
            for _, row in g.iterrows():
                patient_details.append({
                    "PatientId": str(row["PatientId"]),
                    "Probability_NoShow": row["Probability_NoShow"],
                    "SMS_received": row["SMS_received"]
                })
            return pd.Series({
                "Probability Sum": probability_sum,
                "Potential Overbooking": potential_overbooking,
                "Overbook Capacity": overbook_capacity,
                "Patient Details": patient_details,
                "Dynamic Threshold": dynamic_threshold
            })

        gp = df_g.groupby(["DateOnly", "Hour"], dropna=False).apply(group_data).reset_index()
        logger.info("Grouped Data (gp):\n" + gp.to_string())
        logger.info("Data Types of df_g after grouping:\n" + str(df_g.dtypes))

        if "gpt_suggestions" in st.session_state:
            gpt_suggestions = st.session_state["gpt_suggestions"]
        else:
            gpt_suggestions = None

        if st.button("Get Overbooking Suggestions from AI"):
            summary_text_lines = []
            for _, row in gp.iterrows():
                date_str = str(row['DateOnly'])
                hour_str = str(row['Hour']) if pd.notnull(row['Hour']) else 'N/A'
                summary_text_lines.append(f"Date: {date_str}, Hour: {hour_str}")
                for patient_detail in row['Patient Details']:
                    summary_text_lines.append(
                        f"  - Patient ID: {patient_detail['PatientId']}, Probability: {patient_detail['Probability_NoShow']:.2f}, SMS_received: {patient_detail['SMS_received']}"
                    )
                summary_text_lines.append(
                    f"  - Probability Sum: {row['Probability Sum']:.2f}, Potential Overbooking: {row['Potential Overbooking']}, Overbook Capacity: {row['Overbook Capacity']}, Dynamic Threshold: {row['Dynamic Threshold']:.2f}"
                )
            summary_text = "\n".join(summary_text_lines)
            logger.info("Summary Text sent to GPT:\n" + summary_text)
            prompt = (
                "You are an expert scheduling assistant specialized in overbooking for healthcare appointments. "
                "Analyze the following summary of appointment slots and patient data. For each slot, note the 'Probability Sum' "
                "which indicates the chance of no-shows, and also check the SMS_received status. "
                "If there are appointments with high no-show probability (Probability_NoShow >= 0.6) and SMS_received is False, "
                "include a recommendation to send an SMS reminder to that patient (specify the PatientId). "
                "Also, suggest specific date and time slots suitable for overbooking if a slot shows a high 'Probability Sum'. "
                "Return suggestions in JSON format, where each suggestion has the following fields: 'type' (either 'overbooking' "
                "or 'send_reminder'), 'date', 'from_hour', 'to_hour' (if applicable), 'capacity' (if applicable), 'patient_id' (if applicable), and 'reason'. "
                f"For overbooking, use capacity 1 if Probability Sum >= {gp['Dynamic Threshold'].iloc[0]:.2f} and capacity 2 if Probability Sum >= {gp['Dynamic Threshold'].iloc[0] + 1:.2f}. "
                "Return an empty JSON array if no suggestions. Ensure valid JSON output.\n\n"
                f"Data summary:\n{summary_text}\n\n"
                "Example JSON Response:\n"
                "[\n  {\"type\": \"overbooking\", \"date\": \"2024-07-15\", \"from_hour\": 8, \"to_hour\": 9, \"capacity\": 1, \"reason\": \"Probability Sum of 2.8 suggests capacity for 1 extra patient.\"},\n"
                "  {\"type\": \"send_reminder\", \"date\": \"2024-07-15\", \"patient_id\": \"123\", \"reason\": \"Patient with high no-show probability has not received an SMS reminder.\"}\n]"
            )

            gpt_suggestions = get_gpt4_suggestions_for_overbooking(summary_text, gp['Dynamic Threshold'].iloc[0] if not gp.empty else BASE_OVERBOOKING_PROBABILITY_SUM_THRESHOLD)
            st.session_state["gpt_suggestions"] = gpt_suggestions
            st.rerun()

        if gpt_suggestions:
            st.success("Suggestions obtained from AI.")
            overbooking_suggestions = []
            reminder_suggestions = []

            for suggestion in gpt_suggestions:
                suggestion_type = suggestion.get("type")
                if suggestion_type == "overbooking":
                    date_suggestion = suggestion.get("date")
                    from_hour = suggestion.get("from_hour")
                    to_hour = suggestion.get("to_hour")
                    if to_hour is None:
                        to_hour = from_hour + 1
                    capacity = suggestion.get("capacity")
                    reason = suggestion.get("reason", "High probability of no-shows in this time slot.")
                    if date_suggestion and from_hour is not None and to_hour is not None and capacity is not None:
                        suggestion_message = (
                            f"On **{date_suggestion}**, from **{from_hour:02d}:00 to {to_hour:02d}:00**, "
                            f"it is suggested that **{capacity}** additional patient(s) be scheduled. "
                            f"Reason: {reason}"
                        )
                        overbooking_suggestions.append(suggestion_message)
                    else:
                        overbooking_suggestions.append(f"Suggestion format error in overbooking: Incomplete data - {suggestion}")
                elif suggestion_type == "send_reminder":
                    date_suggestion = suggestion.get("date")
                    patient_id_suggestion = suggestion.get("patient_id")
                    reason = suggestion.get("reason", "Patient with high no-show probability has not received an SMS reminder.")
                    if date_suggestion and patient_id_suggestion:
                        suggestion_message = (
                            f"Send SMS reminder to Patient ID **{patient_id_suggestion}** on **{date_suggestion}**. "
                            f"Reason: {reason}"
                        )
                        reminder_suggestions.append(suggestion_message)
                    else:
                        reminder_suggestions.append(f"Suggestion format error in send_reminder: Incomplete data - {suggestion}")
                else:
                    # Log unknown suggestion type
                    overbooking_suggestions.append(f"Unknown suggestion type: {suggestion_type}")

            if overbooking_suggestions:
                st.subheader("Overbooking Actions")
                for message in overbooking_suggestions:
                    st.markdown(f"- {message}")
            else:
                st.info("No overbooking suggestions found for the current criteria.")

            if reminder_suggestions:
                st.subheader("Reminder Actions")
                for message in reminder_suggestions:
                    st.markdown(f"- {message}")
            else:
                st.info("No reminder suggestions found for the current criteria.")
        elif "gpt_suggestions" in st.session_state and gpt_suggestions is None:
            st.warning("No suggestions were obtained from the AI model yet. Click 'Get Overbooking Suggestions from AI' to generate.")
        else:
            st.info("Click 'Get Overbooking Suggestions from AI' to get suggestions.")
