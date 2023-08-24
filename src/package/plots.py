# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ This File Contains Plotting Functions ---------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
from package.helpers import *
    
# Dependencies and Setup__________________________________________________________________________________________________
import plotly.graph_objects as go

# Line Chart _____________________________________________________________________________________________________________
def line_chart(df, col, fil_section):
    
    average_value, lower_bound, upper_bound = avg_calc(df, col)
    
    # Convert the Spark dataframe to a Pandas DataFrame 
    pandas_df = df.toPandas()

    # Create the line trace
    line_trace = go.Scatter(x=pandas_df[col],
                            y=pandas_df[fil_section],
                            mode='lines+markers',
                            name='Data')

    # Create the average line trace
    average_trace = go.Scatter(x=[pandas_df[col].iloc[0], pandas_df[col].iloc[-1]],
                               y=[average_value, average_value],
                               mode='lines',
                               line=dict(color='red', dash='solid'),
                               name='Average')

    # Create the filled area trace for bounds
    bounds_trace = go.Scatter(x=np.concatenate([pandas_df["registration_year"], pandas_df["registration_year"][::-1]]),
                              y=np.concatenate([np.full_like(pandas_df["registration_year"], lower_bound),
                                                np.full_like(pandas_df["registration_year"], upper_bound)[::-1]]),
                              fill='toself', fillcolor='rgba(0, 0, 255, 0.05)',
                              name='Bounds', line=dict(color='rgba(0,0,0,0)'))

    # Combine traces into a data list
    data = [line_trace, average_trace, bounds_trace]

    # Create layout
    layout = go.Layout(
        width=500,
        height=500,
        title='Line Chart with Average and Bounds',
        xaxis=dict(title='Registration Year'),
        yaxis=dict(title='Churn Percentage'),
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff")

    # Create figure
    fig = go.Figure(data=data, layout=layout)

    # Show the figure
    fig.show()
# Bar Chart ______________________________________________________________________________________________________________
def bar_chart(df, col):
    # Convert the selected column of the DataFrame to a Pandas DataFrame
    pandas_df = df.select(col).toPandas()

    # Calculate the frequency of each unique value in the column
    value_counts = pandas_df[col].value_counts().reset_index()
    value_counts.columns = [col, "FREQUENCY"]

    # Sort the DataFrame by the column values for a more meaningful bar chart
    value_counts = value_counts.sort_values(by=col)

    # Define a color scale between two colors
    COLORSET = [[255, 0, 0], [0, 0, 255]]
    color_scale = np.linspace(COLORSET[0], COLORSET[1], len(value_counts)).tolist()
        
    # Convert RGB values to Plotly color format
    plotly_colors = ["rgb({}, {}, {})".format(*tuple(color)) for color in color_scale]

    # Create a Figure object and add a Bar chart trace
    fig = go.Figure(data=[go.Bar(
        x=value_counts[col],
        y=value_counts["FREQUENCY"],
        marker=dict(color=plotly_colors),
        hovertemplate=col.upper() + ": <b>%{x}</b><br>" + "FREQUENCY: <b>%{y}</b>" + "<extra></extra>",
        opacity=1,
        showlegend=False
    )])

    # Customize the layout of the plot
    fig.update_layout(
        xaxis_title=col.upper(),
        yaxis_title="FREQUENCY",
        width=600,
        height=600,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )

    # Display the bar chart
    fig.show()
    
def bar_chart_2(df, col):
    churn_df = (df.groupBy(col)
                           .agg(
                               F.sum(F.when(F.col("is_churn") == 1, 1).otherwise(0)).alias("churn_count"),
                               F.count("*").alias("total_count"))
                           .withColumn("churn_percentage", F.round((F.col("churn_count") / F.col("total_count")) * 100, 2))
                           .orderBy(col))

    # Convert the selected column of the DataFrame to a Pandas DataFrame
    pandas_df = churn_df.toPandas()
    
    # Convert the column to numeric (float or int)
    pandas_df[col] = pandas_df[col].astype(float)  # or int, depending on your data

    # Sort the DataFrame by the column values for a more meaningful bar chart
    pandas_df = pandas_df.sort_values(by=col)
    
    # Define a color scale between two colors
    COLORSET = [[255, 0, 0], [0, 0, 255]]
    color_scale = np.linspace(COLORSET[0], COLORSET[1], len(pandas_df)).tolist()
        
    # Convert RGB values to Plotly color format
    plotly_colors = ["rgb({}, {}, {})".format(*tuple(color)) for color in color_scale]

    # Create a Figure object and add a Bar chart trace
    fig = go.Figure(data=[go.Bar(
        x=pandas_df[col],
        y=pandas_df["total_count"],
        marker=dict(color=plotly_colors),
        hovertemplate=col.upper() + ": <b>%{x}</b><br>" + "FREQUENCY: <b>%{y}</b>" + "<extra></extra>",
        opacity=1,
        showlegend=False
    )])

    # Customize the layout of the plot
    fig.update_layout(
        xaxis_title=col.replace("_", " ").upper(),
        yaxis_title="FREQUENCY",
        width=600,
        height=600,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )
    
    # Display the bar chart
    fig.show()
    return pandas_df
    
# Staxked Bar Chart ________________________________________________________________________________________________________
def stacked_bar_chart(df, col):
    # Convert the selected column of the DataFrame to a Pandas DataFrame
    pandas_df = df.select("is_churn", col).toPandas()

    # Calculate the frequency of each unique value in the specified column for each "is_churn" category
    value_counts = pandas_df.groupby(['is_churn', col]).size().reset_index()
    value_counts.columns = ['is_churn', col, 'FREQUENCY']

    # Create a list to store the traces for each "is_churn" category
    data = []

    # Get the unique values of the "is_churn" column
    churn_values = pandas_df['is_churn'].unique()

    # Create a stacked bar chart trace for each "is_churn" category
    for churn_value in churn_values:
        data.append(go.Bar(
            x=value_counts[value_counts['is_churn'] == churn_value][col],
            y=value_counts[value_counts['is_churn'] == churn_value]['FREQUENCY'],
            name='Churn' if churn_value else 'Not Churn',
            hovertemplate=col.upper() + ': <b>%{x}</b><br>' + 'FREQUENCY: <b>%{y}</b><br>Churn: <b>%{customdata}</b><extra></extra>',
            customdata=['Yes' if churn_value else 'No'] * len(value_counts[col]),
            opacity=0.8
        ))

    # Create the Figure object with the stacked bar chart traces
    fig = go.Figure(data)

    # Customize the layout of the plot
    fig.update_layout(
        xaxis_title=col.upper(),
        yaxis_title='FREQUENCY',
        width=800,
        height=600,
        barmode='stack',
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )

    # Display the stacked bar chart
    fig.show()
    
# bar_scatter_chart _________________________________________________________________________________________________________________
def bar_scatter_chart(df, col):
    churn_df = (df.groupBy(col)
                           .agg(
                               F.sum(F.when(F.col("is_churn") == 1, 1).otherwise(0)).alias("churn_count"),
                               F.count("*").alias("total_count"))
                           .withColumn("churn_percentage", F.round((F.col("churn_count") / F.col("total_count")) * 100, 2))
                           .orderBy(col))
    def avg_cal(df, col): 
        # Calculate the average (mean) of the 'bd' column
        average_value = df[col].mean()

        # Calculate the standard deviation of the 'bd' column
        stddev_value  = df[col].std()

        # Define the sample size
        sample_size = len(df)

        # Define the desired confidence level (e.g., 95%)
        confidence_level = 0.95

        # Calculate the Z-score for the given confidence level
        z_score = 1.96  # For a 95% confidence level

        # Calculate the margin of error
        margin_of_error = z_score * (stddev_value / math.sqrt(sample_size))

        # Calculate the lower and upper bounds
        lower_bound = average_value - margin_of_error
        upper_bound = average_value + margin_of_error
        
        return average_value, lower_bound, upper_bound 

    # Convert the selected column of the DataFrame to a Pandas DataFrame
    pandas_df = churn_df.toPandas()
    
    # Convert the column to numeric (float or int)
    pandas_df[col] = pandas_df[col].astype(float)  # or int, depending on your data

    # Sort the DataFrame by the column values for a more meaningful bar chart
    pandas_df = pandas_df.sort_values(by=col)
    
    average_value, lower_bound, upper_bound=avg_cal(pandas_df, "churn_percentage")
    
    # Define your pandas_df, average_value, lower_bound, and upper_bound

    # Bar Chart Subplot
    color_scale = np.linspace([255, 0, 0], [0, 0, 255], len(pandas_df))
    plotly_colors = ["rgb({}, {}, {})".format(*tuple(color)) for color in color_scale]

    bar_trace = go.Bar(
        x=pandas_df[col],
        y=pandas_df["total_count"],
        marker=dict(color=plotly_colors),
        hovertemplate=col.upper() + ": <b>%{x}</b><br>" + "FREQUENCY: <b>%{y}</b>" + "<extra></extra>",
        opacity=1,
        showlegend=False
    )

    # Line Chart Subplot
    marker_sizes = pandas_df["churn_percentage"] * 1.3  # Adjust the multiplier as needed
    marker_colors = ['red' if y > upper_bound else 'black' for y in pandas_df["churn_percentage"]]
    line_trace = go.Scatter(
        x=pandas_df[col],
        y=pandas_df["churn_percentage"],
        mode='markers',
        marker_color=marker_colors,
        marker_size=marker_sizes,
        name='Data'
    )

    average_trace = go.Scatter(
        x=[pandas_df[col].iloc[0], pandas_df[col].iloc[-1]],
        y=[average_value, average_value],
        mode='lines',
        line=dict(color='red', dash='solid'),
        name='Average'
    )

    bounds_trace = go.Scatter(
        x=np.concatenate([pandas_df[col], pandas_df[col][::-1]]),
        y=np.concatenate([np.full_like(pandas_df[col], lower_bound), np.full_like(pandas_df[col], upper_bound)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.05)',
        name='Bounds',
        line=dict(color='rgba(0,0,0,0)')
    )

    line_data = [bounds_trace, average_trace, line_trace]

    # Create figure with subplots
    fig = make_subplots(
        rows=1,
        cols=2,
    )
    
    # Add traces to subplots
    fig.add_trace(bar_trace, row=1, col=1)
    for item in line_data:
        fig.add_trace(item, row=1, col=2)   
    # Update layout for both subplots
    fig.update_layout(
        width=1200,
        height=600,
        # title='Subplots: Bar Chart and Line Chart',
        # xaxis=dict(title='city'),
        # yaxis=dict(title='Churn Percentage', row=2, col=1),
        showlegend=False,
        # # plot_bgcolor="#ffffff",
        # paper_bgcolor="#ffffff"
    )

    # Display the figure with subplots
    fig.show()
    
# Histogram_______________________________________________________________________________________________________________
# This function creates a histogram plot for the specified column in the given DataFrame
def histogram(df, col):
    # Convert the selected column of the DataFrame to a Pandas DataFrame
    pandas_df = df.select(col).toPandas()

    # Create a Figure object and add a Histogram trace
    fig = go.Figure(data=[go.Histogram(
        x=pandas_df[col],
        marker_color="red",
        hovertemplate=col.upper() + ": <b>%{x}</b><br>" + "FREQUENCY: <b>%{y}</b>" + "<extra></extra>",
        opacity=1,
        showlegend=False
    )])

    # Customize the layout of the plot
    fig.update_layout(
        xaxis_title=col.upper(),
        yaxis_title='FREQUENCY',
        width=600,
        height=600,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )

    # Display the histogram plot
    fig.show()
    
# subplots Line Chart ____________________________________________________________________________________________________
def sub_line(df, col=[], dates=[]):
    # Calculate churn percentage
    month_df = (last_4_df.groupBy(col[0],col[1])
                .agg(
                    F.sum(F.when(F.col("is_churn") == 1, 1).otherwise(0)).alias("churn_count"),
                    F.count("*").alias("total_count"))
                .withColumn("churn_percentage", F.round((F.col("churn_count") / F.col("total_count")) * 100, 2))
                .orderBy(col[0],col[1]))

    fig = make_subplots(
            rows=1,
            cols=len(dates), shared_yaxes=True
        )
    for i,item in enumerate(dates):
        
        month_df_filter=month_df.filter(F.col(col[0]) == item)
        
        def avg_cal(df, col): 
            # Calculate the average (mean) of the 'bd' column
            average_value = df[col].mean()

            # Calculate the standard deviation of the 'bd' column
            stddev_value  = df[col].std()

            # Define the sample size
            sample_size = len(df)

            # Define the desired confidence level (e.g., 95%)
            confidence_level = 0.95

            # Calculate the Z-score for the given confidence level
            z_score = 1.96  # For a 95% confidence level

            # Calculate the margin of error
            margin_of_error = z_score * (stddev_value / math.sqrt(sample_size))

            # Calculate the lower and upper bounds
            lower_bound = average_value - margin_of_error
            upper_bound = average_value + margin_of_error

            return average_value, lower_bound, upper_bound 
        
        pandas_df = month_df_filter.toPandas()

        average_value, lower_bound, upper_bound = avg_cal(pandas_df, "churn_percentage")
        
        # Create the line trace
        line_trace = go.Scatter(x=pandas_df[col[1]], y=pandas_df["churn_percentage"], mode='lines+markers', marker_color='black', name='Data')

        # Create the average line trace
        average_trace = go.Scatter(x=[pandas_df[col[1]].iloc[0], pandas_df[col[1]].iloc[-1]], y=[average_value, average_value],
                                   mode='lines', line=dict(color='red', dash='solid'),
                                   name='Average')

        # Create the filled area trace for bounds
        bounds_trace = go.Scatter(x=np.concatenate([pandas_df[col[1]], pandas_df[col[1]][::-1]]),
                                  y=np.concatenate([np.full_like(pandas_df[col[1]], lower_bound), np.full_like(pandas_df[col[1]], upper_bound)[::-1]]),
                                  fill='toself', fillcolor='rgba(0, 0, 255, 0.05)',
                                  name='Bounds', line=dict(color='rgba(0,0,0,0)'))

        # Combine traces into a data list
        data = [line_trace, average_trace, bounds_trace]
        
        # Create figure with subplots

        for item in data:
            fig.add_trace(item, row=1, col=i+1)  

    fig.update_layout(
        width=2000,
        height=600,
        # title='Subplots: Bar Chart and Line Chart',
        # xaxis=dict(title='city'),
        # yaxis=dict(title='Churn Percentage', row=2, col=1),
        showlegend=False,
        # # plot_bgcolor="#ffffff",
        # paper_bgcolor="#ffffff"
    )

        # Show the figure
    fig.show()
    
# ------------------------------------------------------------------------------------------------------------------------    
print("The plots file is imported ☑")
