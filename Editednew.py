from pathlib import Path
# pythonfile.py — Interactive Waste Management Dashboard (new dataset + city boundaries)
import streamlit as st
import pandas as pd
import numpy as np
import re
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from datetime import date
from pathlib import Path
import os
import streamlit as st, folium, streamlit_folium, pandas as pd
st.write("Libs OK:",
         {"streamlit": st.__version__,
          "folium": folium.__version__,
          "streamlit_folium": streamlit_folium.__version__})

st.write("App dir:", Path(__file__).parent.resolve())
st.write("CSV exists:", (Path(__file__).parent / "New_dataset2.csv").exists())
st.write("Dir listing:", os.listdir(Path(__file__).parent))

st.set_page_config(page_title="Interactive Waste Management Dashboard", layout="wide")


# ---------- CONFIG ----------
CSV_PATH = str(Path(__file__).parent / "New_dataset2.csv")

META_COLS = [
    "City", "Community", "Pincode", "Active Registrations", "Lat", "Lon"
]

def fmt_num(x, digits=1):
    try:
        x = float(x)
        if np.isnan(x) or not np.isfinite(x):
            return "N/A"
        return f"{x:.{digits}f}"
    except Exception:
        return "N/A"

# ---- boundary helper (convex hull via monotone chain) ----
def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 2:
        return [(p[1], p[0]) for p in pts]
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return [(lat, lon) for (lon, lat) in hull]

# ---------- helpers for parsing date labels ----------
def parse_date_label(label: str):
    s = str(label).strip()
    s = re.sub(r"\.\d+$", "", s)
    s = re.sub(r"\s+", " ", s)
    for fmt in ("%m-%Y", "%B %Y", "%b %Y"):
        try:
            return pd.to_datetime(s, format=fmt, errors="raise")
        except Exception:
            continue
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

# ---------- LOAD & PREP ----------
@st.cache_data(show_spinner=True)
def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, na=False)]

    meta_cols = [c for c in META_COLS if c in df.columns]
    base_df = df[meta_cols].copy()

    ts_cols = [c for c in df.columns if c not in meta_cols]
    buckets, order = {}, []

    for c in ts_cols:
        dt = parse_date_label(c)
        if pd.isna(dt):
            continue
        key = dt.strftime("%Y-%m")
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(c)

    blocks = []
    for key in order:
        cols = buckets[key]
        kgs_col = cols[0] if len(cols) >= 1 else None
        part_col = cols[1] if len(cols) >= 2 else None

        block = base_df.copy()
        block["Date"] = pd.to_datetime(key)

        block["Kgs"] = pd.to_numeric(df[kgs_col], errors="coerce") if kgs_col else np.nan
        if part_col:
            p = pd.to_numeric(df[part_col], errors="coerce")
            if pd.notna(p.dropna().median()) and 0 <= p.dropna().median() <= 1:
                p = p * 100.0
            block["Participation_Percent"] = p
        else:
            block["Participation_Percent"] = np.nan
        blocks.append(block)

    data = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=meta_cols + ["Date", "Kgs", "Participation_Percent"])

    if "Lat" in data.columns: data["Lat"] = pd.to_numeric(data["Lat"], errors="coerce")
    if "Lon" in data.columns: data["Lon"] = pd.to_numeric(data["Lon"], errors="coerce")
    if "Active Registrations" in data.columns: data["Active Registrations"] = pd.to_numeric(data["Active Registrations"], errors="coerce")

    if pd.notna(data["Participation_Percent"].dropna().median()) and 0 <= data["Participation_Percent"].dropna().median() <= 1:
        data["Participation_Percent"] *= 100.0

    return data

data = load_and_prepare(CSV_PATH)

# ---------- SIDEBAR ----------
st.sidebar.title("Filters")

min_d = pd.to_datetime(data["Date"].min()).date() if not data.empty else date(2024, 1, 1)
max_d = pd.to_datetime(data["Date"].max()).date() if not data.empty else date.today()
dr = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
start_d, end_d = dr if isinstance(dr, tuple) and len(dr) == 2 else (min_d, max_d)

cities = sorted([c for c in data["City"].dropna().unique()]) if "City" in data.columns else []
selected_cities = st.sidebar.multiselect("City", options=cities, default=cities) if cities else []

metric = st.sidebar.selectbox("Color/Size by", ["Participation %", "Waste (Kgs)"], index=0)
agg_level = st.sidebar.selectbox("Aggregate by", ["Community", "Pincode"], index=0)

# ---------- FILTER & AGG ----------
mask = (data["Date"].dt.date >= start_d) & (data["Date"].dt.date <= end_d)
if selected_cities:
    mask &= data["City"].isin(selected_cities)
filtered = data[mask].copy()

group_cols = []
if "City" in filtered.columns: group_cols.append("City")
if agg_level == "Community": group_cols.append("Community")
if agg_level == "Pincode": group_cols.append("Pincode")
if "Lat" in filtered.columns: group_cols.append("Lat")
if "Lon" in filtered.columns: group_cols.append("Lon")
if "Active Registrations" in filtered.columns: group_cols.append("Active Registrations")

agg = filtered.groupby(group_cols, dropna=True).agg({"Kgs": "sum", "Participation_Percent": "mean"}).reset_index()

if pd.notna(agg["Participation_Percent"].dropna().median()) and 0 <= agg["Participation_Percent"].dropna().median() <= 1:
    agg["Participation_Percent"] *= 100

# ---------- UI ----------
st.title("Interactive Waste Management Dashboard")


# ===== MAP =====
st.subheader("Map")
if agg.empty:
    st.info("No data in selected date range / city.")
else:
    center_lat = agg["Lat"].mean() if "Lat" in agg.columns else 20.5937
    center_lon = agg["Lon"].mean() if "Lon" in agg.columns else 78.9629

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    mc = MarkerCluster().add_to(m)

    values = agg["Participation_Percent"] if metric == "Participation %" else agg["Kgs"]
    vmin = float(np.nanmin(values)) if not values.isna().all() else 0.0
    vmax = float(np.nanmax(values)) if not values.isna().all() else 1.0
    if vmax == vmin:
        vmax = vmin + 1.0

    def color_scale(v):
        if pd.isna(v): return "#808080"
        x = (v - vmin) / (vmax - vmin)
        if x >= 2/3: return "#2ECC71"
        if x >= 1/3: return "#F1C40F"
        return "#E74C3C"

    def radius(v):
        if pd.isna(v): return 6
        return 6 + 18 * ((v - vmin) / (vmax - vmin))

    for _, r in agg.iterrows():
        title = r.get("Community", "") if agg_level == "Community" else str(r.get("Pincode", ""))
        p, k = float(r.get("Participation_Percent", np.nan)), float(r.get("Kgs", np.nan))
        act = r.get("Active Registrations", pd.NA)

        col_value = p if metric == "Participation %" else k
        col, rad = color_scale(col_value), radius(k if metric == "Waste (Kgs)" else col_value)

        popup_html = f"""
        <div style="font-size:14px;">
          <b>{agg_level}:</b> {title}<br>
          <b>Active Registrations:</b> {act if pd.notna(act) else '—'}<br>
          <b>Participation %:</b> {fmt_num(p, 1)}<br>
          <b>Total Kgs:</b> {fmt_num(k, 1)}
        </div>
        """
        tooltip_txt = f"{title} — {('Part %' if metric=='Participation %' else 'Kgs')}: {fmt_num(col_value, 1)}"

        folium.CircleMarker(
            location=[r["Lat"], r["Lon"]],
            radius=rad,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip_txt
        ).add_to(mc)

    # City boundary overlay
    if "City" in filtered.columns:
        for city in sorted(set(filtered["City"].dropna().unique())):
            city_pts = filtered.loc[filtered["City"] == city, ["Lat", "Lon"]].dropna().drop_duplicates()
            if len(city_pts) >= 3:
                lonlat = [(float(row["Lon"]), float(row["Lat"])) for _, row in city_pts.iterrows()]
                hull_latlon = convex_hull(lonlat)
                folium.Polygon(
                    locations=hull_latlon,
                    color="#3498DB",
                    weight=3,
                    fill=False,
                    tooltip=f"{city} boundary"
                ).add_to(m)

    st_folium(m, width=None, height=560)

from matplotlib.ticker import FixedLocator, FixedFormatter

# ===== BOTTOM CHARTS — FORCE CUSTOM X LABELS =====
st.subheader("Summary charts")
if filtered.empty:
    st.info("No data to chart for selected range / city.")
else:
    # Aggregate by month (already one point per month in your data)
    ts_part = (
        filtered.groupby("Date")["Participation_Percent"]
        .mean()
        .reset_index()
        .sort_values("Date")
    )
    ts_kgs = (
        filtered.groupby("Date")["Kgs"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    # If fractions (0–1), convert to %
    if not ts_part["Participation_Percent"].dropna().empty:
        med_ts = ts_part["Participation_Percent"].dropna().median()
        if 0 <= med_ts <= 1:
            ts_part["Participation_Percent"] *= 100

    # Build the exact labels you want: e.g., "March-2024"
    # (Use %B for full month name; change to %b for "Mar-2024".)
    labels = [d.strftime("%B-%Y") for d in ts_part["Date"]]

    # Integer x positions and hard-coded locators/formatters
    x_idx = np.arange(len(labels))

    # --- LINE: Avg Participation % ---
    c1, c2 = st.columns(2)
        # --- LINE: Avg Participation % ---
    with c1:
        fig1, ax1 = plt.subplots()
        ax1.plot(x_idx, ts_part["Participation_Percent"].to_numpy(), marker="o")
        ax1.set_title("Avg Participation % over time")
        ax1.set_ylabel("Participation (%)")
        ax1.grid(True)

        # Dynamically reduce number of ticks to avoid crowding
        max_ticks = 6  # Show at most 6 labels
        step = max(1, len(x_idx) // max_ticks)
        show_ticks = x_idx[::step]
        show_labels = [labels[i] for i in show_ticks]

        ax1.set_xticks(show_ticks)
        ax1.set_xticklabels(show_labels, rotation=30, ha="right")
        ax1.set_xlim(-0.5, len(x_idx) - 0.5)

        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)

    
    # --- BAR: Total Kgs ---
    with c2:
        fig2, ax2 = plt.subplots()
        ax2.bar(x_idx, ts_kgs["Kgs"].to_numpy())
        ax2.set_title("Total Kgs over time")
        ax2.set_ylabel("Kgs")
        ax2.grid(axis="y")

        step2 = max(1, int(np.ceil(len(x_idx) / 10)))
        show_ticks2 = x_idx[::step2]
        show_labels2 = [labels[i] for i in show_ticks2]

        ax2.xaxis.set_major_locator(FixedLocator(show_ticks2))
        ax2.xaxis.set_major_formatter(FixedFormatter(show_labels2))
        ax2.set_xlim(-0.5, len(x_idx) - 0.5)
        for label in ax2.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)



# ===== OPTIONAL DATA PREVIEW =====
with st.expander("Data sample (filtered)"):
    st.dataframe(filtered, use_container_width=True)



