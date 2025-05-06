import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(layout="wide")
st.title("🎮 Game Update Impact Dashboard")

# Load your merged dataset
@st.cache_data
def load_data():
    merged = pd.read_csv("filtered_steam_reviews_final_with_updates_and_players.csv", parse_dates=['month_dt'])
    return merged

merged = load_data()

# ------------------------
# SECTION 1: Game Explorer
# ------------------------
st.header("📈 Game Explorer")
selected_games = st.multiselect("Select one or more games:", merged['app_name'].unique(), default=merged['app_name'].unique()[0])

# Define marker styles for update types (all black)
update_markers = {
    'Low': ('o', 'black'),
    'Medium': ('s', 'black'),
    'High': ('D', 'black')
}

if selected_games:
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    update_legend_handles = {}
    game_handles = []

    for game in selected_games:
        game_data = merged[merged['app_name'] == game]
        line, = ax1.plot(game_data['month_dt'], game_data['peak_players'], marker='o', label=game)
        game_handles.append(line)

        updates = game_data[game_data['update_significance'] != 'No Update']
        for update_type, (marker, color) in update_markers.items():
            subset = updates[updates['update_significance'] == update_type]
            if not subset.empty:
                ax1.plot(subset['month_dt'], subset['peak_players'], linestyle='None', marker=marker, color=color)
                if update_type not in update_legend_handles:
                    update_legend_handles[update_type] = plt.Line2D([], [], color=color, marker=marker, linestyle='None', label=f'{update_type} Update')

    ax1.set_title("Peak Players Over Time")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Peak Players")

    # Add dynamic game legend on the right
    game_legend = ax1.legend(handles=game_handles, title="Games", loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax1.add_artist(game_legend)

    # Add static update type legend below
    ax1.legend(handles=list(update_legend_handles.values()), title="Update Impact", loc='center left', bbox_to_anchor=(1.25, 0.5))

    ax1.grid(True)
    st.pyplot(fig1)

    # ----------------------------------
    # Repeat same for positivity chart
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    update_legend_handles = {}
    game_handles = []

    for game in selected_games:
        game_data = merged[merged['app_name'] == game]
        line, = ax2.plot(game_data['month_dt'], game_data['positivity_rate'], marker='o', label=game)
        game_handles.append(line)

        updates = game_data[game_data['update_significance'] != 'No Update']
        for update_type, (marker, color) in update_markers.items():
            subset = updates[updates['update_significance'] == update_type]
            if not subset.empty:
                ax2.plot(subset['month_dt'], subset['positivity_rate'], linestyle='None', marker=marker, color=color)
                if update_type not in update_legend_handles:
                    update_legend_handles[update_type] = plt.Line2D([], [], color=color, marker=marker, linestyle='None', label=f'{update_type} Update')

    ax2.set_title("Positivity Rate Over Time")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Positivity Rate (%)")

    game_legend = ax2.legend(handles=game_handles, title="Games", loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax2.add_artist(game_legend)

    ax2.legend(handles=list(update_legend_handles.values()), title="Update Impact", loc='center left', bbox_to_anchor=(1.25, 0.5))

    ax2.grid(True)
    st.pyplot(fig2)


# -------------------------------------------
# SECTION 1.5: Total Updates vs Peak Player Metrics
# -------------------------------------------
st.header("📊 Total Updates vs Peak Player Metrics")

# Compute summary metrics
summary_data = merged.groupby('app_name').agg(
    total_updates=('updates_in_month', 'sum'),
    avg_peak_players=('peak_players', 'mean'),
    max_peak_players=('peak_players', 'max')
).reset_index()

# Let user choose metric to plot
metric_option = st.radio(
    "Select Y-axis metric:",
    ["Average Peak Players", "Maximum Peak Players"]
)

# Set axis labels and column names accordingly
if metric_option == "Average Peak Players":
    y_col = 'avg_peak_players'
    y_label = 'Average Peak Player Count'
else:
    y_col = 'max_peak_players'
    y_label = 'Maximum Peak Player Count'

# Create the scatter plot with annotations
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=summary_data, x='total_updates', y=y_col, s=100, color='royalblue', ax=ax)

# Annotate each point with its game name
for _, row in summary_data.iterrows():
    ax.text(row['total_updates'] + 0.2, row[y_col], row['app_name'], fontsize=9, alpha=0.8)

ax.set_title(f"Total Updates vs {y_label}")
ax.set_xlabel("Total Updates (2020–2021)")
ax.set_ylabel(y_label)
ax.grid(True)
st.pyplot(fig)



# -------------------------------------------
# SECTION 2: Update Impact - Before/After Boxplots
# -------------------------------------------
st.header("📦 Update Impact (Before vs After)")
change_type = st.selectbox("Select outcome to visualize:", ["Positivity Change", "Peak Player Change"])

if change_type == "Positivity Change":
    df = pd.read_csv("positivity_change_all_updates.csv")
    metric = "positivity_change"
    ylabel = "Change in Positivity Rate (%)"
    ylimits = (-10, 10)  # reasonable range for % change
else:
    df = pd.read_csv("peak_player_change_all_updates.csv")
    metric = "peak_change"
    ylabel = "Change in Peak Player Count"
    ylimits = (-50000, 100000)  # filter out massive outliers visually

fig3, ax3 = plt.subplots(figsize=(10, 4))
sns.boxplot(
    data=df,
    x='update_type',
    y=metric,
    hue='update_type',
    palette='Set2',
    legend=False,
    ax=ax3,
    order=["No Update", "Low", "Medium", "High"],
    showfliers=False
)
ax3.axhline(0, color='black')
ax3.set_title(f"{change_type} by Update Type")
ax3.set_ylabel(ylabel)
ax3.set_xlabel("Update Impact Type")
ax3.set_ylim(*ylimits)
ax3.grid(True)
st.pyplot(fig3)

# -------------------------------------------
# SECTION 2.5: Positivity Rate vs Peak Player Count (Interactive)
# -------------------------------------------
import plotly.express as px

st.header("🔍 Positivity Rate vs Peak Player Count")

# Create scatter data from merged dataframe
scatter_data = merged[['app_name', 'positivity_rate', 'peak_players']].dropna()

# Aggregate by app_name and month to avoid overlapping identical values
scatter_summary = scatter_data.groupby(['app_name', 'positivity_rate', 'peak_players']).size().reset_index(name='count')

# Plot using Plotly for interactivity
fig_plotly = px.scatter(
    scatter_summary,
    x='positivity_rate',
    y='peak_players',
    color='app_name',
    size='count',
    hover_data={
        'app_name': True,
        'positivity_rate': ':.2f',
        'peak_players': True,
        'count': False
    },
    labels={
        'positivity_rate': 'Positivity Rate (%)',
        'peak_players': 'Peak Player Count',
        'app_name': 'Game'
    },
    title="Positivity Rate vs Peak Player Count (Interactive)"
)

fig_plotly.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
fig_plotly.update_layout(height=500)
st.plotly_chart(fig_plotly, use_container_width=True)


# -------------------------------------------
# SECTION 3: ML Model Comparison
# -------------------------------------------
st.header("🧠 ML Model Comparison")
model_data = merged.groupby('app_name').agg(
    total_updates=('updates_in_month', 'sum'),
    avg_peak_players=('peak_players', 'mean')
).reset_index()

X = model_data[['total_updates']]
y = model_data['avg_peak_players']

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Support Vector Regression': SVR(),
    'Neural Network (MLPRegressor)': MLPRegressor(random_state=42, max_iter=1000)
}

selected_model_name = st.selectbox("Select a model to display:", list(models.keys()))
model = models[selected_model_name]
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.scatter(X, y, color='black', label='Actual')
ax4.plot(X, y_pred, color='red', label=f'{selected_model_name}\nR²={r2:.2f}, MAE={mae:.0f}')
ax4.set_title(f"Model: {selected_model_name}")
ax4.set_xlabel("Total Updates (2020–2021)")
ax4.set_ylabel("Avg Peak Player Count")
ax4.legend()
ax4.grid(True)
st.pyplot(fig4)

# Show R² and MAE separately
st.markdown(f"**R² Score:** {r2:.4f}  ")
st.markdown(f"**Mean Absolute Error (MAE):** {mae:.0f} players")
