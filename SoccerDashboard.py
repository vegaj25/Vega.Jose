import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import requests
import plotly.express as px
import sqlite3
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(layout="wide")

database_path = 'Soccer.db'
connection = sqlite3.connect(database_path)
cursor = connection.cursor()

query = """SELECT * 
        FROM Soccer 
        ORDER BY "Rank" """

cursor.execute(query)

results = cursor.fetchall()

columns = [desc[0] for desc in cursor.description]

soccer = pd.DataFrame(results, columns=columns)

connection.close()

# Start of Streamlit App

#Sidebar
st.sidebar.image('FIFA_Logo.png', use_container_width=True)
st.sidebar.markdown("# **Ballon d'Or Soccer Player Analysis Dashboard**")
st.sidebar.header("Page Navigation")

#Page Navigation
options = ["Data/Player Overview", "Player Comparison", "Player Analysis", "Player Position Analysis", "Position Analysis", "Performance by Jersey Number", "Player Performance Prediction"]
page = st.sidebar.pills("Select a Page", options, default=options[0])

if page == "Data/Player Overview":
    st.title("Data/Player Overview")
    
    st.divider()

    with st.container():
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Player Count", soccer['Player_Name'].nunique() + 1)
        col2.metric("Nationalities", soccer['Nation_Code'].nunique() + 1)
        col3.metric("Leagues", soccer['Competition'].nunique() + 1)
        col4.metric("Clubs", soccer['Club'].nunique() + 1)

    st.divider()

    # DATA TABLE 
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(soccer, use_container_width=True)

    st.divider()

    st.subheader("Player Information Lookup")
    player_options = ["Show All"] + list(soccer["Player_Name"].unique())
    player = st.selectbox("Choose a player:", player_options)

    selected_columns = [
        "Player_Name","Age","Jersey_Number", "Nation_Code", "Position",
        "Club","Competition"
    ]
    if player == "Show All":
        st.dataframe(soccer[selected_columns], use_container_width=True)
    else:
        player_row = soccer.loc[soccer["Player_Name"] == player, selected_columns]
        st.dataframe(player_row, use_container_width=True)
    
    st.divider()

    #  MAP SECTION 
    st.subheader("Player Nationalities Map")

    nation_map = {
        "FRA": "France", "ESP": "Spain", "POR": "Portugal", "EGY": "Egypt",
        "BRA": "Brazil", "MAR": "Morocco", "ENG": "United Kingdom",
        "GEO": "Georgia", "ITA": "Italy", "SWE": "Sweden", "POL": "Poland",
        "SCO": "United Kingdom", "ARG": "Argentina", "GUI": "Guinea",
        "NED": "Netherlands", "NOR": "Norway", "GER": "Germany",
    }

    player_options = ["Show All"] + list(soccer["Player_Name"].unique())
    player = st.selectbox("Choose a player:", player_options, key="player_select")

    if player == "Show All":
        unique_nations = soccer["Nation_Code"].unique()
        selected_countries = [nation_map[code] for code in unique_nations if code in nation_map]
    else:
        player_nation_code = soccer.loc[soccer["Player_Name"] == player, "Nation_Code"].values[0]
        selected_countries = [nation_map.get(player_nation_code)]

    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    geojson = requests.get(url).json()
    geojson["features"] = [f for f in geojson["features"] if f["properties"]["name"] in selected_countries]

    #  Helper: compute centroid of selected country 
    def get_centroid(feature):
        coords = feature["geometry"]["coordinates"]
        # Handle MultiPolygon vs Polygon
        if feature["geometry"]["type"] == "Polygon":
            flat_coords = np.array(coords[0])
        elif feature["geometry"]["type"] == "MultiPolygon":
            flat_coords = np.array(coords[0][0])
        else:
            return (20, 0)  # fallback
        lon, lat = flat_coords[:,0], flat_coords[:,1]
        return float(lat.mean()), float(lon.mean())

    if player != "Show All" and geojson["features"]:
        lat, lon = get_centroid(geojson["features"][0])
        zoom_level = 4  # closer zoom for single country
    else:
        lat, lon = 20, 0
        zoom_level = 1  # global view

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        opacity=0.7,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="[0, 128, 255, 160]",
        get_line_color="[255, 255, 255]",
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom_level)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

#______________________________________________________________________________________________________

elif page == "Player Comparison":
    st.title("Player Comparison")
    st.divider()

    # RADAR CHARTS SECTION 
    st.subheader("Player Comparison Radar Charts")

    scoring_metrics = ["Goals","NonPK_Goals","Goals_per90","NonPK_Goals_per90","Expected_Goals","NonPK_xG","xG_per90","NonPK_xG_per90"]
    playmaking_metrics = ["Assists","Expected_Assists","Assists_per90","xAG_per90","Goals_Assists"]
    progression_metrics = ["Prog_Carries","Prog_Passes","Prog_Received","NonPK_xG_xAG","NonPK_xG_xAG_per90"]
    impact_metrics = ["Goals_Assists","GA_per90","NonPK_GA_per90","xG_xAG_per90"]

    def radar_chart(soccer, metrics, title, color_map):
        radar_df = soccer[["Player_Name"] + metrics].melt(id_vars="Player_Name", var_name="Metric", value_name="Value")
        fig = px.line_polar(
            radar_df,
            r="Value",
            theta="Metric",
            color="Player_Name",
            line_close=True,
            template="plotly_dark",
            color_discrete_map=color_map
        )
        fig.update_traces(fill="toself")
        fig.update_layout(title=title, showlegend=True)
        return fig

    selected_players = st.multiselect("Select players to compare:", soccer["Player_Name"].unique())

    if selected_players:
        filtered_df = soccer[soccer["Player_Name"].isin(selected_players)]
        st.markdown("Customize Player Colors")
        color_map, used_colors = {}, set()
        for player in selected_players:
            default_color = "#%06x" % (hash(player) % 0xFFFFFF)
            chosen_color = st.color_picker(f"Color for {player}", default_color)
            if chosen_color in used_colors:
                st.warning(f"{player} has the same color as another player. Please pick a different one.")
            else:
                color_map[player] = chosen_color
                used_colors.add(chosen_color)

        colA, colB = st.columns(2)
        with colA:
            st.plotly_chart(radar_chart(filtered_df, scoring_metrics, "Scoring Efficiency", color_map), use_container_width=True)
            st.plotly_chart(radar_chart(filtered_df, progression_metrics, "Progression & Ball Carrying", color_map), use_container_width=True)
        with colB:
            st.plotly_chart(radar_chart(filtered_df, playmaking_metrics, "Playmaking & Creativity", color_map), use_container_width=True)
            st.plotly_chart(radar_chart(filtered_df, impact_metrics, "Overall Impact", color_map), use_container_width=True)
    else:
        st.info("Please select at least one player to generate radar charts.")

    # Question 9 – Who is the best overall player?
    st.divider()
    st.header("Who is the best overall player?")

    st.write(
        "To identify the best overall player, we combine several key performance "
        "metrics into a single composite score. Each metric is normalized so they "
        "can be compared fairly across different scales."
    )

    # Metrics to define overall player quality
    overall_metrics = [
        "Goals_per90",
        "Assists_per90",
        "xG_xAG_per90",
        "Prog_Carries",
        "Prog_Passes",
        "Prog_Received"
    ]

    # Build dataframe for ranking
    overall_df = soccer[["Player_Name", "Position"] + overall_metrics].dropna()

    # Normalize each (0–1)
    for m in overall_metrics:
        min_val = overall_df[m].min()
        max_val = overall_df[m].max()
        if max_val > min_val:
            overall_df[m + "_norm"] = (overall_df[m] - min_val) / (max_val - min_val)
        else:
            overall_df[m + "_norm"] = 0.0

    # Composite score = average of normalized metrics
    norm_cols = [m + "_norm" for m in overall_metrics]
    overall_df["Overall_Score"] = overall_df[norm_cols].mean(axis=1)

    # Sort high → low
    overall_df_sorted = overall_df.sort_values("Overall_Score", ascending=False)

    # Add ranking number: 1 = best
    overall_df_sorted["Statistical_Rank"] = range(1, len(overall_df_sorted) + 1)

    # Let user choose how many to display
    top_n = st.slider("How many top players to display?", min_value=5, max_value=30, value=10, step=1)
    top_players = overall_df_sorted.head(top_n)

    # Ranked bar chart
    st.subheader("Top Players by Overall Score (Ranked)")

    fig_q9 = px.bar(
        top_players,
        x="Statistical_Rank",                         # rank order: 1,2,3...
        y="Overall_Score",
        color="Position",
        hover_name="Player_Name",
        text="Player_Name",               # display names on bars
        labels={
            "Statistical_Rank": "Statistical_Rank (1 = Best)",
            "Overall_Score": "Overall Score"
        },
        title="Top Players Based on Composite Overall Score"
    )

    fig_q9.update_traces(textposition="outside")
    fig_q9.update_layout(xaxis=dict(dtick=1), showlegend=True)
    st.plotly_chart(fig_q9, use_container_width=True)

    # Table view showing details
    st.subheader("Detailed Ranking Table")
    st.dataframe(
        top_players[["Statistical_Rank", "Player_Name", "Position", "Overall_Score"] + overall_metrics]
        .reset_index(drop=True),
        use_container_width=True
    )

    # Highlight the single #1 player
    best_player = overall_df_sorted.iloc[0]
    st.success(
        f"Top-ranked player: {best_player['Player_Name']} "
        f"({best_player['Position']}) with an overall composite score of "
        f"{best_player['Overall_Score']:.3f}."
    )

    st.write(
        "Note: Rankings will change if you adjust the included metrics or their weights. "
        "In the presentation, explain why these particular metrics define overall quality."
    )
#______________________________________________________________________________________________________

elif page == "Player Analysis":
    st.title("Player Analysis")
    st.divider()

    # AGE VS MATCHES PLAYED SCATTER PLOT
    st.subheader("Age vs Matches Played")
    fig_age = px.scatter(
        soccer,
        x="Age",
        y="Matches_Played",
        size="Minutes_Played",
        color="Position",
        hover_name="Player_Name",
        title="Age vs Matches Played (Bubble size = Minutes Played, Color = Position)",
        template="plotly_white"
    )

    # Add trendline using statsmodels
    X = soccer["Age"]
    y = soccer["Matches_Played"]
    X = sm.add_constant(X)  # add intercept
    model = sm.OLS(y, X).fit()
    soccer["Trendline"] = model.predict(X)

    fig_age.add_traces(px.line(
        soccer,
        x="Age",
        y="Trendline"
    ).data)

    st.plotly_chart(fig_age, use_container_width=True)

    st.divider()


    #RANK VS GOALS + ASSISTS SCATTER PLOT
    st.subheader("Rank vs Goals + Assists")

    # Define composite metric (Goals + Assists)
    soccer["GA_Total"] = soccer["Goals_Assists"]

    # Scatter plot
    fig_rank = px.scatter(
        soccer,
        x="Rank",
        y="GA_Total",
        color="Position",
        size="Minutes_Played",
        hover_name="Player_Name",
        title="Rank vs Goals + Assists",
        template="plotly_white"
    )

    # Highlight underrated players: low rank but high stats
    threshold_rank = soccer["Rank"].quantile(0.75)   # bottom 25% ranks
    threshold_stats = soccer["GA_Total"].quantile(0.75)  # top 25% stats
    underrated = soccer[(soccer["Rank"] > threshold_rank) & (soccer["GA_Total"] > threshold_stats)]

    for _, row in underrated.iterrows():
        fig_rank.add_annotation(
            x=row["Rank"],
            y=row["GA_Total"],
            text=row["Player_Name"],
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-20,
            bgcolor="yellow"
        )

    st.plotly_chart(fig_rank, use_container_width=True) 

    ####question 6 :Do players who play more games perform better?
    st.divider()
    st.header("Do players who play more games perform better?")

    st.write(
        "This section compares how many games a player has appeared in "
        "against different performance metrics to see if heavy usage is "
        "associated with better output."
    )

    # Choose a performance metric to compare against Matches_Played
    performance_metric = st.selectbox(
        "Select a performance metric",
        [
            "Goals",
            "Assists",
            "Goals_Assists",
            "Goals_per90",
            "Assists_per90",
            "GA_per90",
            "xG_per90",
            "xAG_per90",
            "xG_xAG_per90"
        ]
    )

    # Drop rows with missing values in the selected columns
    q6_df = soccer[["Player_Name", "Position", "Matches_Played", performance_metric]].dropna()

    # Scatter plot: Matches Played vs chosen performance metric
    fig_q6 = px.scatter(
        q6_df,
        x="Matches_Played",
        y=performance_metric,
        color="Position",
        hover_name="Player_Name",
        labels={
            "Matches_Played": "Matches Played",
            performance_metric: performance_metric.replace("_", " ")
        },
        title=f"Matches Played vs {performance_metric.replace('_', ' ')}"
    )

    st.plotly_chart(fig_q6, use_container_width=True)

    st.write("Analysis:")
    st.write(
        "- Points toward the right show players who appear in more matches, while higher points on the vertical axis "
        "indicate stronger performance on the selected metric.\n"
        "- If the trendline slopes upward, it suggests that players who play more games tend to perform better on this "
        "metric. A flat or downward slope suggests that volume of games does not strongly drive this outcome."
    )

#______________________________________________________________________________________________________

elif page == "Player Position Analysis":
    st.title("Position Analysis")
    st.divider()

    st.header("Do Forwards Score More Efficiently Than Midfielders?")
    # Choose which metric to treat as a 'shot accuracy' proxy
    accuracy_metric = st.selectbox(
        "Select a scoring efficiency metric:",
        ["Goals_per90", "NonPK_Goals_per90", "xG_per90", "NonPK_xG_per90"],
        index=0
    )

    # Build position groups (any player whose Position contains FW or MF)
    fw_df = soccer[soccer["Position"].str.contains("FW", na=False)]
    mf_df = soccer[soccer["Position"].str.contains("MF", na=False)]

    fw_mean = fw_df[accuracy_metric].mean()
    mf_mean = mf_df[accuracy_metric].mean()

    summary_q7 = pd.DataFrame({
        "Position": ["Forward (FW)", "Midfielder (MF)"],
        "Average_" + accuracy_metric: [fw_mean, mf_mean]
    })

    st.subheader("Average scoring efficiency by position")
    st.table(summary_q7.style.format({ "Average_" + accuracy_metric: "{:.3f}" }))

    fig_q7 = px.bar(
        summary_q7,
        x="Position",
        y="Average_" + accuracy_metric,
        text_auto=".3f",
        title=f"{accuracy_metric} by Position (FW vs MF)"
    )
    st.plotly_chart(fig_q7, use_container_width=True)

    # Optional: simple interpretation
    if fw_mean > mf_mean:
        st.success(
            f"On average, forwards have higher {accuracy_metric} than midfielders in this dataset."
        )
    else:
        st.info(
            f"In this dataset, forwards do not have higher {accuracy_metric} than midfielders on average."
        )

    st.divider()
    st.header("What Metrics Define a Good Player by Position?")

    # Simplify positions to main role tokens
    base_position = st.selectbox(
        "Choose a primary position to analyze:",
        ["FW", "MF", "DF", "GK"]
    )

    pos_mask = soccer["Position"].str.contains(base_position, na=False)
    pos_df = soccer[pos_mask].copy()

    st.subheader(f"Players primarily listed as {base_position}")
    st.dataframe(pos_df[["Player_Name", "Position", "Club", "Goals", "Assists",
                        "Goals_per90", "Assists_per90", "Prog_Passes", "Prog_Carries"]])

    # Suggested key metrics by position
    position_metric_suggestions = {
        "FW": ["Goals_per90", "NonPK_Goals_per90", "xG_per90"],
        "MF": ["Assists_per90", "Prog_Passes", "Prog_Carries"],
        "DF": ["GA_per90", "Prog_Carries", "Prog_Passes"],
        "GK": ["GA_per90"]  # minimal in this dataset
    }

    st.subheader("Key metrics for this position")

    metrics_for_pos = position_metric_suggestions.get(base_position, ["Goals_per90"])
    metric_choice = st.selectbox(
        "Select a metric to rank players in this position:",
        metrics_for_pos
    )

    top_pos = (
        pos_df[["Player_Name", "Club", metric_choice]]
        .sort_values(metric_choice, ascending=False)
        .head(10)
    )

    st.markdown(f"Top players at {base_position} by **{metric_choice}**")
    st.dataframe(top_pos.style.format({metric_choice: "{:.3f}"}))


#______________________________________________________________________________________________________

elif page == "Position Analysis":
    st.title("Position Analysis")
    st.divider()

    st.header("Metrics Associated With Player Positions")

    # Choose metrics to analyze by position
    numeric_cols = soccer.select_dtypes(include=['float64', 'int64']).columns.tolist()
    pos_metrics = st.multiselect(
        "Select metrics related to position-specific performance:",
        numeric_cols,
        default=["Goals", "Assists", "Goals_Assists", "Expected_Goals", "Expected_Assists"]
    )

    if pos_metrics:
        # Aggregated averages
        pos_summary = soccer.groupby("Position")[pos_metrics].mean().round(2)

        st.subheader("Average Metrics by Position")
        st.dataframe(pos_summary)

        # Normalize each column individually (min-max scaling per column)
        pos_summary_norm = pos_summary.copy()
        for col in pos_summary_norm.columns:
            col_min = pos_summary_norm[col].min()
            col_max = pos_summary_norm[col].max()
            if col_max > col_min:  # avoid divide by zero
                pos_summary_norm[col] = (pos_summary_norm[col] - col_min) / (col_max - col_min)

        # Heatmap of position profiles (column-wise scaling)
        st.subheader("Position-Metric Heatmap")
        fig_pos_heat = px.imshow(
            pos_summary_norm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="How Metrics Vary by Position (Column-wise Normalized)"
        )
        fig_pos_heat.update_xaxes(side="top")  # metrics on top for readability
        st.plotly_chart(fig_pos_heat, use_container_width=True)

        st.subheader("Metric Distribution by Position")
        metric_box_choice = st.selectbox("Choose a metric to visualize distribution:", pos_metrics)

        fig_box = px.box(
            soccer,
            x="Position",
            y=metric_box_choice,
            color="Position",
            title=f"{metric_box_choice} Distribution by Position"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.info("Select metrics to explore how performance differs by position.")

#______________________________________________________________________________________________________
elif page == "Performance by Jersey Number":
    st.title("Performance by Jersey Number")
    st.divider()

    st.subheader("Jersey Number vs Metric")
    numeric_cols = soccer.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["Jersey_Number"]]  # exclude jersey num itself

    if numeric_cols:
        metric_choice = st.selectbox("Choose a metric to visualize:", numeric_cols)

        fig_scatter = px.scatter(
            soccer,
            x="Jersey_Number",
            y=metric_choice,
            color="Position",
            hover_name="Player_Name",
            title=f"Jersey Number vs {metric_choice}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    else:
        st.info("Select at least one metric to explore Jersey Number correlation.")

#______________________________________________________________________________________________________
elif page == "Player Performance Prediction":
    st.title("Player Performance Prediction")
    st.divider()
    
    st.header("Predictive Model: Predict Performance Score")

    # Create a composite performance score if not already present
    if "Performance_Score" not in soccer.columns:
        soccer["Performance_Score"] = (
            soccer["Goals"] + soccer["Assists"] +
            soccer["Expected_Goals"] + soccer["Expected_Assists"]
        )

    # Define target and feature candidates
    target = "Performance_Score"
    feature_options = [
        "Goals", "Assists", "Expected_Goals", "Expected_Assists",
        "Prog_Passes", "Prog_Carries", "Prog_Received", "Goals_per90",
        "Assists_per90", "xG_per90", "xAG_per90"
    ]

    # Let user select features for prediction
    selected_features = st.multiselect(
        "Select features to use for prediction:",
        feature_options,
        default=feature_options
    )

    if selected_features:
        X = soccer[selected_features]
        y = soccer[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Train RandomForest
        model = RandomForestRegressor(n_estimators=500, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        colA, colB = st.columns(2)
        colA.metric("MAE", round(mae, 4))
        colB.metric("R² Score", round(r2, 4))

        # Feature importance
        importances = pd.DataFrame({
            "Feature": selected_features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig_imp = px.bar(
            importances,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance in Predicting Performance Score"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # Predict and show top 5 players based on predicted performance
        soccer["Predicted_Score"] = model.predict(soccer[selected_features])
        top5_pred = soccer.sort_values(by="Predicted_Score", ascending=False).head(5)
        st.subheader("Top 5 Predicted Players")
        st.dataframe(top5_pred[["Player_Name", "Club", "Predicted_Score"] + selected_features])

    else:
        st.info("Please select at least one feature to run the predictive model.")
