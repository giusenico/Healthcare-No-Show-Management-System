import streamlit as st
import pandas as pd
import plotly.express as px

# =============================================================================
# DATA LOADING FUNCTION WITH ERROR HANDLING
# =============================================================================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file, handles errors, and converts date columns.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame or an empty DataFrame if errors occur.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"File not found at: {path}. Please check the file path.")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV file: {e}. Please ensure the file is properly formatted.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the file: {e}")
        return pd.DataFrame()

    # Convert date columns to datetime
    for col in ["ScheduledDay", "AppointmentDay"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if df[col].isnull().sum() > 0:
                st.warning(f"Invalid dates found and coerced to NaT in column '{col}'.")
    return df

# =============================================================================
# OUTLIER REMOVAL FUNCTION (IQR Method)
# =============================================================================
def remove_outliers_iqr(df: pd.DataFrame, column: str, groupby_col: str = None) -> pd.DataFrame:
    """
    Removes outliers from a specified column using the IQR method.
    Optionally, removal is performed within groups defined by groupby_col.

    Args:
        df (pd.DataFrame): DataFrame to process.
        column (str): Column from which to remove outliers.
        groupby_col (str, optional): Column to group by before outlier removal.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if groupby_col:
        groups = df.groupby(groupby_col)
        filtered_list = []
        for name, group in groups:
            Q1 = group[column].quantile(0.25)
            Q3 = group[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            filtered_list.append(group[(group[column] >= lower_bound) & (group[column] <= upper_bound)])
        df_filtered = pd.concat(filtered_list)
    else:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# =============================================================================
# STREAMLIT APP INITIAL CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")
st.title("Healthcare Appointment No-Show Analysis")

# -----------------------------------------------------------------------------
# LOAD DATASET
# -----------------------------------------------------------------------------
DATA_PATH = "data/healthcare_noshows_appt_pre_encoded.csv"  # Consider making this configurable
df = load_data(DATA_PATH)
if df.empty:
    st.error("Failed to load the dataset. Please check the CSV file and path.")
    st.stop()

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================
st.sidebar.header("Data Filters")

# Gender Filter
gender_options = sorted(df["Gender"].unique().tolist())
selected_gender = st.sidebar.multiselect("Gender", options=gender_options, default=gender_options)

# Age Range Filter
min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
selected_age = st.sidebar.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

# Wealth Level Filter (if available)
if "Wealth_Level" in df.columns:
    wealth_options = sorted(df["Wealth_Level"].dropna().unique().tolist())
    selected_wealth = st.sidebar.multiselect("Wealth Level", options=wealth_options, default=wealth_options)
else:
    selected_wealth = []

# Distance Category Filter (if available)
if "Distance_Category" in df.columns:
    distance_options = sorted(df["Distance_Category"].dropna().unique().tolist())
    selected_distance = st.sidebar.multiselect("Distance Category", options=distance_options, default=distance_options)
else:
    selected_distance = []

# Medical Conditions (Inclusive Filters)
st.sidebar.subheader("Medical Conditions (Inclusive)")
filter_conditions = {}
if "Hipertension" in df.columns:
    filter_conditions["Hipertension"] = st.sidebar.checkbox("Hypertension", value=False)
if "Diabetes" in df.columns:
    filter_conditions["Diabetes"] = st.sidebar.checkbox("Diabetes", value=False)
if "Alcoholism" in df.columns:
    filter_conditions["Alcoholism"] = st.sidebar.checkbox("Alcoholism", value=False)
if "Handcap" in df.columns:
    filter_conditions["Handcap"] = st.sidebar.checkbox("Handicap", value=False)

# SMS Received Filter (if available)
st.sidebar.subheader("Communication")
filter_sms_received = False
if "SMS_received" in df.columns:
    filter_sms_received = st.sidebar.checkbox("SMS Received", value=False)

# =============================================================================
# APPLY FILTERS TO DATAFRAME
# =============================================================================
df_filtered = df.copy()
df_filtered = df_filtered[df_filtered["Gender"].isin(selected_gender)]
df_filtered = df_filtered[(df_filtered["Age"] >= selected_age[0]) & (df_filtered["Age"] <= selected_age[1])]

if selected_wealth and "Wealth_Level" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Wealth_Level"].isin(selected_wealth)]
if selected_distance and "Distance_Category" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Distance_Category"].isin(selected_distance)]

for condition, active in filter_conditions.items():
    if active:
        df_filtered = df_filtered[df_filtered[condition] == True]

if filter_sms_received and "SMS_received" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["SMS_received"] == True]

if df_filtered.empty:
    st.warning("No data available after applying the filters. Please adjust your selections.")
    st.stop()

# =============================================================================
# SUMMARY METRICS
# =============================================================================
st.subheader("Appointment Overview (Filtered)")
total_appointments = df_filtered.shape[0]
show_count = df_filtered[df_filtered["Showed_up"] == True].shape[0] if "Showed_up" in df_filtered.columns else 0
no_show_count = df_filtered[df_filtered["Showed_up"] == False].shape[0] if "Showed_up" in df_filtered.columns else 0
no_show_rate = (no_show_count / total_appointments * 100) if total_appointments > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Appointments", f"{total_appointments}")
col2.metric("Showed Up", f"{show_count}")
col3.metric("No-Show", f"{no_show_count}")
col4.metric("No-Show Rate", f"{no_show_rate:.1f}%")

st.write("---")

# -----------------------------------------------------------------------------
# COMMON LAYOUT SETTINGS FOR PLOTLY CHARTS
# -----------------------------------------------------------------------------
common_layout = dict(
    template="plotly_white",
    margin=dict(l=50, r=50, t=100, b=50),
    height=450,
    title_x=0.5
)

# =============================================================================
# CHART 1: DONUT CHART - SHOW VS NO-SHOW
# =============================================================================
st.subheader("Appointment Outcomes: Show vs No-Show")
if "Showed_up" in df_filtered.columns and total_appointments > 0:
    status_df = pd.DataFrame({
        "Status": ["Show", "No-Show"],
        "Count": [show_count, no_show_count]
    })
    fig_pie = px.pie(
        status_df,
        names="Status",
        values="Count",
        hole=0.4,
        color="Status",
        color_discrete_map={"Show": "#00AA5A", "No-Show": "#FF4B4B"},
        title="Distribution of Appointment Outcomes"
    )
    fig_pie.update_layout(**common_layout, legend_title_text="Outcome")
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown(f"**Insight:** The no-show rate is **{no_show_rate:.2f}%** based on the current filters. Demographic and medical condition filters significantly influence this rate.")
else:
    st.info("Insufficient data to display the Show vs No-Show chart.")

st.write("---")

# =============================================================================
# CHART 2: NO-SHOW RATE BY AGE GROUP (BAR CHART)
# =============================================================================
st.subheader("No-Show Rate by Age Group")
if "Age_Group" in df_filtered.columns:
    group_field = "Age_Group"
    group_rate = df_filtered.groupby(group_field)["Showed_up"].value_counts(normalize=True).mul(100).unstack(fill_value=0)
    if False not in group_rate.columns:
        st.info("No no-show data available for the selected filters to calculate age group no-show rates.")
    else:
        group_rate = group_rate[False].reset_index(name='% No-Show')
        fig_age = px.bar(
            group_rate,
            x=group_field,
            y='% No-Show',
            color='% No-Show',
            color_continuous_scale=px.colors.sequential.Reds,
            labels={group_field: 'Age Group', '% No-Show': 'No-Show Percentage'},
            title="No-Show Percentage Across Age Groups"
        )
        fig_age.update_layout(**common_layout, coloraxis_showscale=False)
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown("**Insight:** This chart shows how the no-show percentage varies across different age groups. Certain age groups may exhibit a higher tendency to miss appointments.")
else:
    st.info("Age group data is not available in the dataset to display this chart.")

st.write("---")

# =============================================================================
# CHART 3: MEDICAL CONDITIONS VS. SHOW-UP (GROUPED BAR CHART)
# =============================================================================
st.subheader("Medical Conditions vs. Show-Up Rate")
medical_cols = ["Hipertension", "Diabetes", "Alcoholism", "Handcap"]
available_med_cols = [col for col in medical_cols if col in df_filtered.columns]
if available_med_cols:
    chosen_med = st.selectbox("Select a medical condition to analyze:", available_med_cols)
    med_df = df_filtered.groupby([chosen_med, "Showed_up"]).size().reset_index(name="Count")
    fig_med = px.bar(
        med_df,
        x=chosen_med,
        y="Count",
        color="Showed_up",
        barmode="group",
        color_discrete_map={False: "#FF4B4B", True: "#00AA5A"},
        labels={"Showed_up": "Showed Up", chosen_med: chosen_med, "Count": "Number of Appointments"},
        title=f"Show-Up Distribution by {chosen_med}"
    )
    fig_med.update_layout(**common_layout, legend_title_text="Show-Up")
    st.plotly_chart(fig_med, use_container_width=True)
    st.markdown(f"**Insight:** This chart compares show-up rates for patients with and without **{chosen_med}** (0 = Absent, 1 = Present). Observe how the presence of this condition influences appointment attendance.")
else:
    st.info("No medical condition data available to display this chart.")

st.write("---")

# =============================================================================
# CHART 4: APPOINTMENT DISTRIBUTION BY WEEKDAY (BAR CHART)
# =============================================================================
if "Weekday" in df_filtered.columns:
    st.subheader("Appointment Distribution by Day of the Week")
    weekday_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    df_filtered["Weekday_Name"] = df_filtered["Weekday"].apply(lambda x: weekday_map.get(x, x))
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_filtered["Weekday_Name"] = pd.Categorical(df_filtered["Weekday_Name"], categories=weekday_order, ordered=True)
    weekday_df = df_filtered.groupby(["Weekday_Name", "Showed_up"]).size().reset_index(name="Count")
    fig_weekday = px.bar(
        weekday_df,
        x="Weekday_Name",
        y="Count",
        color="Showed_up",
        barmode="group",
        color_discrete_map={False: "#FF4B4B", True: "#00AA5A"},
        labels={"Weekday_Name": "Weekday", "Showed_up": "Showed Up", "Count": "Number of Appointments"},
        title="Appointments and Show-Up Rates by Weekday"
    )
    fig_weekday.update_layout(**common_layout, legend_title_text="Show-Up")
    st.plotly_chart(fig_weekday, use_container_width=True)
    st.markdown("**Insight:** This chart shows appointment scheduling and show-up behavior variations across different days of the week, starting from Monday.")
else:
    st.info("Weekday data is not available to display this chart.")

st.write("---")

# =============================================================================
# CHART 5: AGE VS. DISTANCE TO HOSPITAL SCATTER PLOT (OUTLIERS REMOVED)
# =============================================================================
st.subheader("Patient Demographics: Age vs. Distance to Hospital (Outliers Removed)")
if all(col in df_filtered.columns for col in ["Age", "Distance_Hospital_km", "Showed_up"]):
    df_no_outliers = remove_outliers_iqr(df_filtered, "Distance_Hospital_km", groupby_col="Showed_up")
    try:
        fig_scatter = px.scatter(
            df_no_outliers,
            x="Age",
            y="Distance_Hospital_km",
            color="Showed_up",
            color_discrete_map={False: "#FF4B4B", True: "#00AA5A"},
            labels={"Age": "Patient Age", "Distance_Hospital_km": "Distance to Hospital (km)"},
            hover_data=["PatientId", "AppointmentID", "Neighbourhood"],
            title="Relationship Between Age, Distance, and Show-Up (Outliers Removed)"
        )
        fig_scatter.update_layout(**common_layout, legend_title_text="Show-Up")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("**Insight:** This scatter plot explores how patient age and distance to the hospital may influence appointment attendance. Outliers in 'Distance to Hospital' have been removed using the IQR method for a clearer view.")
    except Exception as e:
        st.error(f"Error creating scatter plot: {e}")
else:
    st.info("Required data (Age, Distance to Hospital, Showed Up) is not available for this chart.")

st.write("---")

# =============================================================================
# CHART 6: DAILY APPOINTMENT TRENDS (LINE CHART)
# =============================================================================
st.subheader("Daily Appointment Trends Over Time")
if "AppointmentDay" in df_filtered.columns:
    time_data = df_filtered.dropna(subset=["AppointmentDay"])
    time_counts = time_data.groupby(time_data["AppointmentDay"].dt.date).size().reset_index(name="Count")
    time_counts.rename(columns={"AppointmentDay": "Date"}, inplace=True)
    fig_line = px.line(
        time_counts,
        x="Date",
        y="Count",
        markers=True,
        title="Daily Appointment Volume Trend",
        labels={"Date": "Appointment Date", "Count": "Number of Appointments"}
    )
    fig_line.update_layout(**common_layout)
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("**Insight:** Analyze the daily trend of appointments over time. Peaks or dips may indicate specific events or seasonal patterns.")
else:
    st.info("Appointment day data is not available to display this chart.")

st.write("---")

# =============================================================================
# CHART 7: NO-SHOW RATE BY DISTANCE CATEGORY (BAR CHART)
# =============================================================================
if "Distance_Category" in df_filtered.columns:
    st.subheader("No-Show Rate by Distance Category")
    distance_rate = df_filtered.groupby("Distance_Category")["Showed_up"].value_counts(normalize=True).mul(100).unstack(fill_value=0)
    if False not in distance_rate.columns:
        st.info("No no-show data available for the selected filters to calculate distance category no-show rates.")
    else:
        distance_rate = distance_rate[False].reset_index(name='% No-Show')
        fig_distance = px.bar(
            distance_rate,
            x="Distance_Category",
            y='% No-Show',
            color='% No-Show',
            color_continuous_scale=px.colors.sequential.Plasma_r,
            labels={"Distance_Category": "Distance Category", '% No-Show': 'No-Show Percentage'},
            title="No-Show Percentage by Distance Category"
        )
        fig_distance.update_layout(**common_layout, coloraxis_showscale=False)
        st.plotly_chart(fig_distance, use_container_width=True)
        st.markdown("**Insight:** This chart shows how no-show rates vary by distance category. Patients in certain distance categories may be more likely to miss appointments.")
    st.write("---")

# =============================================================================
# CHART 8: NO-SHOW RATE BY WEALTH LEVEL (BAR CHART)
# =============================================================================
if "Wealth_Level" in df_filtered.columns:
    st.subheader("No-Show Rate by Wealth Level")
    wealth_rate = df_filtered.groupby("Wealth_Level")["Showed_up"].value_counts(normalize=True).mul(100).unstack(fill_value=0)
    if False not in wealth_rate.columns:
        st.info("No no-show data available for the selected filters to calculate wealth level no-show rates.")
    else:
        wealth_rate = wealth_rate[False].reset_index(name='% No-Show')
        fig_wealth = px.bar(
            wealth_rate,
            x="Wealth_Level",
            y='% No-Show',
            color='% No-Show',
            color_continuous_scale=px.colors.sequential.Viridis_r,
            labels={"Wealth_Level": "Wealth Level", '% No-Show': 'No-Show Percentage'},
            title="No-Show Percentage by Wealth Level"
        )
        fig_wealth.update_layout(**common_layout, coloraxis_showscale=False)
        st.plotly_chart(fig_wealth, use_container_width=True)
        st.markdown("**Insight:** This chart examines the relationship between wealth level and no-show rates, highlighting if patients from certain wealth levels are more likely to miss appointments.")
    st.write("---")

st.info("This dashboard provides an interactive overview of appointment no-show patterns. Use the filters in the sidebar to explore specific patient segments and factors influencing appointment attendance.")
