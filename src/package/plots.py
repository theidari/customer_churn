# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ This File Contains Plotting Functions ---------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
from package.config import *
from package.func import avg_cal_pandas

class PltAssets:
    def __init__(self, plt_dataframe):
        """
        A class for handling various plot-related assets and functionalities.

        This class provides methods for deriving variable names from local namespaces,
        generating color scales for Plotly, and potentially more plotting utilities in the future.
        
        Args:
            plt_dataframe (pd.DataFrame): A Pandas DataFrame containing data to be plotted.
        """
        self.dataframe = plt_dataframe

    def get_variable_name(self):
        """
        Try to find the name of self.dataframe in the local namespace of the caller.
        
        Returns:
            str or None: The name of self.dataframe or None if not found.
        """
        caller_locals = inspect.currentframe().f_back.f_back.f_locals  # Note the double .f_back
        for name, value in caller_locals.items():
            if value is self.dataframe:
                return name
        return None

    def generate_plotly_color_scale(self, value_counts, COLORSET=[[255, 0, 0], [0, 0, 255]]):
        """
        Generate a color scale for Plotly based on the number of provided value counts.

        Args:
            value_counts (list-like or int): Iterable containing values whose count determines the color scale length or an integer.
            COLORSET (list of lists): A list of two RGB colors to interpolate between. Defaults to red and blue.

        Returns:
            list: A list of colors in Plotly's 'rgb(r, g, b)' format.
        """
        # Define a color scale between two colors
        if isinstance(value_counts, pd.DataFrame):
            num_colors = len(value_counts)
        elif isinstance(value_counts, int):
            num_colors = int(value_counts)
        else:
            raise TypeError("value_counts must be either a DataFrame or an integer.")

        color_scale = np.linspace(COLORSET[0], COLORSET[1], num_colors).tolist()

        # Convert RGB values to Plotly color format
        plotly_colors = ["rgb({}, {}, {})".format(*tuple(color)) for color in color_scale]

        return plotly_colors

# Bar Chart ______________________________________________________________________________________________________________
def bar_chart(dfs, param):
    
    chart_title_data = []
    subplots_data = []

    for df in dfs:
        # Fetch the DataFrame name from the caller's local namespace
        df_name = PltAssets(df).get_variable_name()
        chart_title = df_name if df_name else "Not Defined"
        chart_title_data.append(chart_title)
        
        # Convert the selected column of the DataFrame to a Pandas DataFrame
        pandas_df = df.select(param).toPandas()

        # Convert 0 and 1 in the col to "No" and "Yes"
        pandas_df[param] = pandas_df[param].map({0: "No", 1: "Yes"})

        # Calculate the frequency and percentage of each unique value in the column
        value_counts = pandas_df[param].value_counts().reset_index()
        value_counts.columns = [param, "FREQUENCY"]
        total = value_counts["FREQUENCY"].sum()
        value_counts["PERCENTAGE"] = ((value_counts["FREQUENCY"] / total) * 100).round(2)

        # Sort the DataFrame by the column values
        value_counts.sort_values(by=param, inplace=True)

        # Define a color scale and convert RGB values to Plotly color format
        plotly_colors = PltAssets(df).generate_plotly_color_scale(value_counts, COLORSET=[[255, 0, 0], [0, 0, 255]])

        """ Plotting"""
        subplot = go.Bar(
            x=value_counts[param],
            y=value_counts["FREQUENCY"],
            text=value_counts["PERCENTAGE"].astype(str) + "%",
            textposition='auto',
            marker=dict(color=plotly_colors),
            hovertemplate=param.upper() + ": <b>%{x}</b><br>" +
                          "FREQUENCY: <b>%{y}</b>" + "<extra></extra>"
        )
        subplots_data.append(subplot)
        
    # Define layout and subplot titles
    subplot_titles = [
        f"<span style='font-size: 15px; color:black;'>{title.replace('_', ' ').title()}</span>" for title in chart_title_data
    ]

    layout = go.Layout(
        width=1200,
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='#f7f7f7', font=dict(color='black')),
        xaxis=dict(title=param.replace('_', ' ').upper(), color='black', showline=True, linewidth=1, linecolor='black', mirror=True),
        yaxis=dict(title="FREQUENCY", color='black', showline=True, linewidth=1, linecolor='black', mirror=True),
        plot_bgcolor='#ffffff',
        paper_bgcolor="#ffffff"
    )

    # Create the figure with subplots
    fig = make_subplots(rows=1, cols=len(dfs), subplot_titles=subplot_titles, shared_yaxes=True, horizontal_spacing=0.1)

    for idx, subplot in enumerate(subplots_data):
        fig.add_trace(subplot, row=1, col=idx+1)
        fig.update_xaxes(dict(title=param.replace('_', ' ').upper(), color='black', showline=True, linewidth=1, linecolor='black', mirror=True), 
                         showline=True, linewidth=0.5, linecolor="black", mirror=True, row=1, col=idx+1)
        fig.update_yaxes(showline=True, linewidth=0.5, linecolor="black", mirror=True, row=1, col=idx+1)

    fig.update_layout(layout, showlegend=False)

    return fig.show()
    
# bar_scatter_chart _________________________________________________________________________________________________________________
def bar_scatter_chart(df, param, chart_title=None, MARKER_SIZE=1.3):
    
    df_name = PltAssets(df).get_variable_name()
    if chart_title is None:
        chart_title = df_name if df_name else "Not Defined"
    
    churn_df = (df.groupBy(param)
                           .agg(
                               sum(when(col("is_churn") == 1, 1).otherwise(0)).alias("churn_count"),
                               count("*").alias("total_count"))
                           .withColumn("churn_percentage", round((col("churn_count") / col("total_count")) * 100, 2))
                           .orderBy(param))
    
    # Convert the selected column of the DataFrame to a Pandas DataFrame
    pandas_df = churn_df.toPandas()
    
    # Convert the column to numeric (float or int)
    pandas_df[param] = pandas_df[param].astype(float)  # or int, depending on your data

    # Sort the DataFrame by the column values for a more meaningful bar chart
    pandas_df = pandas_df.sort_values(by=param)
        
    # Define your pandas_df, average_value, lower_bound, and upper_bound
    average_value, lower_bound, upper_bound = avg_cal_pandas(pandas_df, "churn_percentage")
   
    # Bar Chart Subplot
    plotly_colors = PltAssets(df).generate_plotly_color_scale(pandas_df, COLORSET=[[255, 0, 0], [0, 0, 255]])

    """ Plotting"""
    bar_trace = go.Bar(
        x=pandas_df[param],
        y=pandas_df["total_count"],
        marker=dict(color=plotly_colors),
        hovertemplate=param.upper() + ": <b>%{x}</b><br>" + "FREQUENCY: <b>%{y}</b>" + "<extra></extra>",
        opacity=1,
        showlegend=False
    )

    # Line Chart Subplot
    marker_sizes = pandas_df["churn_percentage"] * MARKER_SIZE  # Adjust the multiplier as needed
    marker_colors = ['red' if y > upper_bound else 'black' for y in pandas_df["churn_percentage"]]
    line_trace = go.Scatter(
        x=pandas_df[param],
        y=pandas_df["churn_percentage"],
        mode='markers',
        hovertemplate=param.upper() + ": <b>%{x}</b><br>" + "CHURN PERCENTAGE: <b>%{y}</b>" + "<extra></extra>",
        marker_color=marker_colors,
        marker_size=marker_sizes,
        name='Data'
    )

    average_trace = go.Scatter(
        x=[pandas_df[param].iloc[0], pandas_df[param].iloc[-1]],
        y=[average_value, average_value],
        hovertemplate="Average:  <b>%{y}</b>" + "<extra></extra>",
        mode='lines',
        line=dict(color='red', dash='solid'),
        name='Average'
    )

    bounds_trace = go.Scatter(
        x=np.concatenate([pandas_df[param], pandas_df[param][::-1]]),
        y=np.concatenate([np.full_like(pandas_df[param], lower_bound), np.full_like(pandas_df[param], upper_bound)[::-1]]),
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
        title=dict(text=param.replace('_', ' ').title()+" In "+chart_title.replace('_', ' ').title(),
                   font=dict(color= 'black'),
                   x=0.5,
                   y=0.9),
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )
    tags=["Frequency", "churn_percentage"]
    for i in range(2):
        fig.update_xaxes(dict(title=param.replace('_', ' ').upper(), color='black', showline=True, linewidth=1, linecolor='black', mirror=True), 
                         showline=True, linewidth=0.5, linecolor="black", mirror=True, row=1, col=i+1)
        fig.update_yaxes(dict(title=tags[i].replace('_', ' ').upper(), color='black', showline=True, linewidth=1, linecolor='black', mirror=True),
                         showline=True, linewidth=0.5, linecolor="black", mirror=True, row=1, col=i+1)

    # Display the figure with subplots
    fig.show()
    
# subplots Line Chart ____________________________________________________________________________________________________
def sub_line(df, param=[], dates=[]):
    
    df_name = PltAssets(df).get_variable_name()
    chart_title = df_name if df_name else "Not Defined"
    
    # Calculate churn percentage
    month_df = (df.groupBy(param[0],param[1])
                .agg(
                    sum(when(col("is_churn") == 1, 1).otherwise(0)).alias("churn_count"),
                    count("*").alias("total_count"))
                .withColumn("churn_percentage", round((col("churn_count") / col("total_count")) * 100, 2))
                .orderBy(param[0],param[1]))

    # Define layout and subplot titles
   
    # Create the figure with subplots
    fig = make_subplots(rows=1, cols=len(dates), shared_yaxes=True, horizontal_spacing=0.02)
    for i,item in enumerate(dates):
        
        month_df_filter=month_df.filter(col(param[0]) == item)
        
        pandas_df = month_df_filter.toPandas()

        average_value, lower_bound, upper_bound = avg_cal_pandas(pandas_df, "churn_percentage")
                
        # Create the line trace
        line_trace = go.Scatter(x=pandas_df[param[1]], y=pandas_df["churn_percentage"], mode='lines+markers',
                                hovertemplate=param[1].upper() + ": <b>%{x}</b><br>" + "CHURN PERCENTAGE: <b>%{y}</b>" + "<extra></extra>",
                                marker_color='black', name='Data')

        # Create the average line trace
        average_trace = go.Scatter(x=[pandas_df[param[1]].iloc[0], pandas_df[param[1]].iloc[-1]], y=[average_value, average_value],
                                   hovertemplate="Average:  <b>%{y}</b>" + "<extra></extra>",
                                   mode='lines', line=dict(color='red', dash='solid'),
                                   name='Average')

        # Create the filled area trace for bounds
        bounds_trace = go.Scatter(x=np.concatenate([pandas_df[param[1]], pandas_df[param[1]][::-1]]),
                                  y=np.concatenate([np.full_like(pandas_df[param[1]], lower_bound), np.full_like(pandas_df[param[1]], upper_bound)[::-1]]),
                                  fill='toself', fillcolor='rgba(0, 0, 255, 0.05)',
                                  name='Bounds', line=dict(color='rgba(0,0,0,0)'))

        # Combine traces into a data list
        data = [line_trace, average_trace, bounds_trace]
        for item in data:
            fig.add_trace(item, row=1, col=i+1)
  

    # Update layout for both subplots
    fig.update_layout(
        width=300 * len(dates),
        height=450,
        title=dict(text=param[1].replace('_', ' ').title(),
                   font=dict(color= 'black'),
                   x=0.5,
                   y=0.85),
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )
    
    for i, date_name in enumerate(dates):
        fig.update_xaxes(dict(title=date_name, color='black', showline=True, linewidth=1, linecolor='black', mirror=True), 
                         showline=True, linewidth=0.5, linecolor="black", mirror=True, row=1, col=i+1)
        fig.update_yaxes(showline=True, linewidth=0.5, linecolor="black", mirror=True, row=1, col=i+1)
        # Show the figure
    fig.show()
    
# MultiBar Chart _________________________________________________________________________________________________________
def multiline(model_df, params=[], chart_title=""):
    """
    Plots a grouped bar chart for a given PySpark DataFrame based on specified parameters.
    
    Parameters:
    - model_df : PySpark DataFrame
        The dataframe to be plotted.
    - params : list
        A list of column names to be used for grouping and plotting.
    - chart_title : str
        The title of the chart.
    """
    chart_title_df = PltAssets(model_df).get_variable_name()
    
    # Groupby and count using PySpark's DataFrame
    df = (model_df.groupby(params)
          .agg(count('*').alias('counts'))
          .toPandas())  # Convert to Pandas DataFrame for easier handling thereafter
    
    # Calculate total count
    total_count = df['counts'].sum()
    
    # Create a grouped bar chart
    fig = go.Figure()
    # Define a color scale and convert RGB values to Plotly color format
    color_code=int(2**(len(params))/2)
    plotly_colors = PltAssets(df).generate_plotly_color_scale(color_code, COLORSET=[[255, 0, 0], [0, 0, 255]])
    
    combination_colors = {}
    counter = 0
    for auto_renew in [0, 1]:
        for churn in [0, 1]:
            combination_colors[(auto_renew, churn)] = plotly_colors[counter]
            counter += 1
            subset = df[(df[params[1]] == auto_renew) & (df[params[2]] == churn)].copy()
            
            # Calculate percentage for the subset
            subset['percentage'] = (subset['counts'] / total_count) * 100
            
            fig.add_trace(go.Bar(
                x=['No' if x == 0 else 'Yes' for x in subset[params[0]]],
                y=subset['counts'],
                marker=dict(color=combination_colors[(auto_renew, churn)]),
                hovertemplate=(
                    f"{params[0].replace('_', ' ').title()}: <b>%{{x}}</b><br>"
                    f"{params[1].replace('_', ' ').title()}: <b>{'No' if auto_renew == 0 else 'Yes'}</b><br>"
                    f"{params[2].replace('_', ' ').title()}: <b>{'No' if churn == 0 else 'Yes'}</b><br>"
                    "Frequency: <b>%{y}</b><br>"
                    "Percentage: <b>%{customdata[0]:.2f}%</b><extra></extra>"
                ),
                customdata=subset[['percentage']].values,
                showlegend=False
            ))

    fig.update_layout(
        width=1200,
        height=800,
        barmode='group',
        title=dict(text=chart_title.title()+" In "+chart_title_df.replace('_', ' ').title(),
                   font=dict(color='black'),
                   x=0.5,
                   y=0.9),
        xaxis=dict(title=params[0].replace('_', ' ').upper()+" STATUS", color='black', showline=True, linewidth=1, linecolor='black', mirror=True),
        yaxis=dict(title="FREQUENCY", color='black', showline=True, linewidth=1, linecolor='black', mirror=True),
        plot_bgcolor='#ffffff',
        paper_bgcolor="#ffffff"
    )

    fig.show()
# ------------------------------------------------------------------------------------------------------------------------
