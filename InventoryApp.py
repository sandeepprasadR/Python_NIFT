# Import necessary libraries
import base64
import pulp
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from dash import callback, State
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from dash.exceptions import PreventUpdate
import flask
from flask import Flask, request, redirect, url_for, render_template
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# Specify the file paths to your CSV files
products_csv_path = 'C:/Users/GN898ZK/OneDrive - EY/5. My Learnings/33. Python/NIFT Project/NIFT_Project/Python_NIFT/products.csv'
customer_demographics_csv_path = 'C:/Users/GN898ZK/OneDrive - EY/5. My Learnings/33. Python/NIFT Project/NIFT_Project/Python_NIFT/customer_demographics.csv'
inventory_csv_path = 'C:/Users/GN898ZK/OneDrive - EY/5. My Learnings/33. Python/NIFT Project/NIFT_Project/Python_NIFT/inventory.csv'
feedback_csv_path = 'C:/Users/GN898ZK/OneDrive - EY/5. My Learnings/33. Python/NIFT Project/NIFT_Project/Python_NIFT/feedback.csv'
customer_metrics_csv_path = 'C:/Users/GN898ZK/OneDrive - EY/5. My Learnings/33. Python/NIFT Project/NIFT_Project/Python_NIFT/customer_metrics.csv'
sales_csv_path = 'C:/Users/GN898ZK/OneDrive - EY/5. My Learnings/33. Python/NIFT Project/NIFT_Project/Python_NIFT/sales.csv'

# Read the CSV files using the specified file paths
products_data = pd.read_csv(products_csv_path, encoding='utf-8')
customer_data = pd.read_csv(customer_demographics_csv_path, sep=',')
inventory_data = pd.read_csv(inventory_csv_path)
feedback_data = pd.read_csv(feedback_csv_path)
customer_metrics_data = pd.read_csv(customer_metrics_csv_path)
df_sales = pd.read_csv(sales_csv_path)

# Define your Dash app layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.config['suppress_callback_exceptions'] = True

# Define your Dash app layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.config['suppress_callback_exceptions'] = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=[
        html.H1("Sales Data Dashboard"),  # Title
        html.P("Select a product to view its sales trend:"),  # Description

        # Use the product dropdown generated based on df_sales
        dcc.Dropdown(
            id='product-select',
            options=[{'label': product, 'value': product} for product in df_sales["Product_ID"].unique()],
            value=df_sales["Product_ID"].iloc[0]  # Set an initial value if needed
        ),

        # Line chart to display sales trend
        dcc.Graph(id='sales-trend-plot'),

        # DataTable with id 'datatable-interactivity'
        dash_table.DataTable(
            id='datatable-interactivity',
            columns=[{'name': col, 'id': col} for col in df_sales.columns],
            data=df_sales.to_dict('records'),
        ),
        
        # Placeholder for customer info
        html.Div(id='customer-info'),
        
        # Placeholder for gender distribution plot
        dcc.Graph(id='gender-distribution-plot'),
        
        # Placeholder for inventory table
        html.Div(id='inventory-table'),
        
        # Placeholder for customer metrics plot
        dcc.Graph(id='customer-metrics-plot'),
        
        # Placeholder for feedback ratings plot
        dcc.Graph(id='feedback-ratings-plot'),
        
        # Placeholder for feedback text
        dcc.Textarea(id='selected-feedback-text'),
        
        # Placeholder for filter options
        dcc.Dropdown(id='filter-brand'),
        dcc.Dropdown(id='filter-supplier'),
        dcc.Dropdown(id='filter-material'),
        dcc.Dropdown(id='filter-season'),
        dcc.Dropdown(id='filter-size'),
        dcc.Dropdown(id='filter-store-location'),
        
        # Placeholder for sentiment analysis plot
        dcc.Graph(id='sentiment-analysis-plot'),
        
        # Placeholder for LP optimization result
        html.Div(id='lp-optimization-result'),
    ]),
    html.H1("Linear Programming Optimization"),
    html.Label("Coefficient for x1:"),
    dcc.Input(id="coef_x1", type="number", value=10),
    html.Label("Coefficient for x2:"),
    dcc.Input(id="coef_x2", type="number", value=15),
    html.Label("Constraint 1 coefficient for x1:"),
    dcc.Input(id="constraint1_coef_x1", type="number", value=2),
    html.Label("Constraint 1 coefficient for x2:"),
    dcc.Input(id="constraint1_coef_x2", type="number", value=3),
    html.Label("Constraint 1 RHS value:"),
    dcc.Input(id="constraint1_rhs", type="number", value=100),
    html.Label("Constraint 2 coefficient for x1:"),
    dcc.Input(id="constraint2_coef_x1", type="number", value=1),
    html.Label("Constraint 2 coefficient for x2:"),
    dcc.Input(id="constraint2_coef_x2", type="number", value=2),
    html.Label("Constraint 2 RHS value:"),
    dcc.Input(id="constraint2_rhs", type="number", value=60),
    html.Button("Optimize", id="optimize_button"),
    html.Div(id="optimization_result"),
])


@app.callback(
    [Output("customer-info", "children"), Output("gender-distribution-plot", "figure")],
    [Input("product-select", "value")]
)
def update_customer_info_and_gender_distribution(selected_product):
    # Filter data based on the selected product
    product_data = df_sales[df_sales["Product_ID"] == selected_product]

    # Generate customer info content
    customer_info_content = html.Div([
        html.H3("Customer Information for Product: " + selected_product),
        dash_table.DataTable(
            id='customer-info-table',
            columns=[{'name': col, 'id': col} for col in customer_data.columns],
            data=product_data.merge(customer_data, on='Customer_ID', how='inner').to_dict('records'),
        )
    ])

    # Calculate gender distribution
    gender_counts = customer_data["Gender"].value_counts()
    gender_distribution_figure = go.Figure(data=[go.Pie(labels=gender_counts.index, values=gender_counts.values)])

    return customer_info_content, gender_distribution_figure

# Assuming you have already defined your Dash app and loaded data into df_sales

@app.callback(
    Output("productSelect", "options"),  # Output to 'productSelect' dropdown
    [
        Input("salesInput", "contents"),
        Input("feedbackInput", "contents"),
        Input("inventoryInput", "contents"),
        Input("enableUpload", "value"),
    ]
)
def update_product_dropdown_options(contents_sales, contents_feedback, contents_inventory, enable_upload):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return []  # No triggers, return an empty list

    if "enabled" in enable_upload and contents_sales:
        # Extract unique product IDs from the already loaded sales data
        product_options = [{'label': product, 'value': product} for product in df_sales["Product_ID"].unique()]

        return product_options  # Return the product options for the dropdown

    # If no data or upload is not enabled, return an empty list
    return []

# Assuming you have already defined your Dash app and loaded data into df_sales

@app.callback(
    Output("page-content", "children"),  # Output to 'page-content' div
    [
        Input('salesInput', 'contents'),
        Input('filterSeason', 'value'),
        Input('filterBrand', 'value'),
        Input('filterSupplier', 'value'),
        Input('filterMaterial', 'value'),
        Input('filterStoreLocation', 'value'),
    ]
)
def update_filtered_data_table_callback(contents_sales, season, brand, supplier, material, store_location):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return []  # No triggers, return an empty list

    if not contents_sales:
        return []  # No data available, return an empty list

    df_sales = pd.read_csv(contents_sales[0])  # Load data directly from contents_sales

    # Apply filters based on user selections
    if season:
        df_sales = df_sales[df_sales['Season'] == season]
    if brand:
        df_sales = df_sales[df_sales['Brand'] == brand]
    if supplier:
        df_sales = df_sales[df_sales['Supplier'] == supplier]
    if material:
        df_sales = df_sales[df_sales['Material'] == material]
    if store_location:
        df_sales = df_sales[df_sales['Store_Location'] == store_location]

    # Create the filtered data table component within this callback
    filtered_data_table = create_data_table(df_sales)

    return [filtered_data_table]  # Return the filtered data table as part of 'page-content'

# Create data table function (no change needed here)
def create_data_table(filtered_data):
    # Create a Dash DataTable based on the filtered data
    if not filtered_data.empty:
        return dash_table.DataTable(
            id='filtered-data-table',
            columns=[{'name': col, 'id': col} for col in filtered_data.columns],
            data=filtered_data.to_dict('records'),
            style_table={'overflowX': 'auto'},
        )
    else:
        return "No data to display."
    
# Assuming you have already defined your Dash app and loaded data into df_sales

@app.callback(
    Output("page-content", "children"),  # Output to 'page-content' div
    [Input("salesInput", "contents")]
)
def run_optimization(contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return []  # No triggers, return an empty list

    if contents_sales:
        optimization_results = "Optimization results:\n - Optimal order quantity: 100\n - Maximized profit: $10,000"
        optimization_results_component = html.P(optimization_results)
        return optimization_results_component  # Return the results component as part of 'page-content'
    else:
        message_component = html.P("Please upload sales data to perform optimization.")
        return message_component  # Return the message component as part of 'page-content'

# Assuming you have already defined your Dash app and loaded data into df_sales

@app.callback(
    Output("decompPlot", "figure"),
    [Input("salesInput", "contents")]
)
def update_decomposition_plot(contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update to prevent callback from running

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))

    result = seasonal_decompose(df_sales['Total_Revenue'], model='additive', freq=12)

    fig_decomposition = go.Figure()
    fig_decomposition.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend'))
    fig_decomposition.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal'))
    fig_decomposition.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residual'))
    fig_decomposition.update_layout(title="Decomposition Plot")

    return fig_decomposition

# Assuming you have already defined your Dash app and loaded data into df_sales

@app.callback(
    Output("salesTrendPlot", "figure"),
    [Input("salesInput", "contents")]
)
def update_sales_trend_plot(contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update to prevent callback from running

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    fig = px.line(df_sales, x="Date", y="Total_Revenue", title="Sales Trend")
    return fig

@app.callback(
    Output("seasonalDecompPlot", "figure"),
    [Input("salesInput", "contents")]
)
def update_seasonal_decomp_plot(contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update to prevent callback from running

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    
    df_sales['Date'] = pd.to_datetime(df_sales['Date'])
    df_sales.set_index('Date', inplace=True)
    decomposition = sm.tsa.seasonal_decompose(df_sales['Total_Revenue'], model='additive', period=7)
    
    seasonal_fig = go.Figure()
    seasonal_fig.add_trace(go.Scatter(x=df_sales.index, y=decomposition.trend, mode='lines', name='Trend'))
    seasonal_fig.add_trace(go.Scatter(x=df_sales.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
    seasonal_fig.add_trace(go.Scatter(x=df_sales.index, y=decomposition.resid, mode='lines', name='Residual'))
    seasonal_fig.update_layout(title='Seasonal Decomposition Plot', xaxis_title='Date', yaxis_title='Value')
    
    return seasonal_fig

@app.callback(
    Output("priceSensitivityPlot", "figure"),
    [Input("salesInput", "contents")]
)
def update_price_sensitivity_plot(contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update to prevent callback from running

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    
    fig = px.scatter(df_sales, x="Total_Revenue", y="Units_Sold", title="Price Sensitivity")
    return fig

@app.callback(
    Output("demandForecastPlot", "figure"),
    [Input("salesInput", "contents")]
)
def update_demand_forecast_plot(contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update to prevent callback from running

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    
    demand_fig = px.line(df_sales, x="Date", y="Units_Sold", title="Demand Forecast")
    return demand_fig

@app.callback(
    Output("dateRangeSelect", "start_date"),
    [Input("salesInput", "contents")]
)
def update_start_date(contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update to prevent callback from running

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    
    start_date = df_sales["Date"].min()
    return start_date


@app.callback(
    Output("dateRangeSelect", "end_date"),
    [Input("salesInput", "contents")]
)
def update_end_date(contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update to prevent callback from running

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    
    end_date = df_sales["Date"].max()
    return end_date

@app.callback(
    Output("ratingEffectPlot", "figure"),
    [Input("salesInput", "contents")]
)
def update_rating_effect_plot(contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update to prevent callback from running

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    
    import statsmodels.api as sm
    X = df_sales['Rating']
    X = sm.add_constant(X)
    model = sm.OLS(df_sales['Total_Revenue'], X).fit()
    fig = px.scatter(df_sales, x="Rating", y=model.predict(X), title="Rating Effect")
    return fig


@app.callback(
    [Output("customerInfo", "children"), Output("genderDistribution", "figure")],
    [Input("customerSelect", "value"), Input("salesInput", "contents")]
)
def update_customer_info(selected_customer, contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update, dash.no_update  # No triggers, return dash.no_update for both outputs

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    
    # Use the correct file path for 'customer_demographics.csv'
    customer_file_path = r'C:\Users\GN898ZK\OneDrive - EY\5. My Learnings\33. Python\NIFT Project\NIFT Project\customer_demographics.csv'
    df_customer = pd.read_csv(customer_file_path)
    customer_data = df_sales[df_sales["Customer_ID"] == selected_customer]
    customer_table = dbc.Table.from_dataframe(customer_data, striped=True, bordered=True, hover=True)
    gender_counts = df_customer["Gender"].value_counts()
    fig = px.pie(gender_counts, names=gender_counts.index, values=gender_counts.values,
                 title="Gender Distribution of Customers")
    return customer_table, fig

@app.callback(
    Output("customerSelect", "options"),
    [Input("salesInput", "contents")]
)
def update_customer_dropdown(contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update
    
    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    customer_options = [{"label": customer_id, "value": customer_id} for customer_id in df_sales["Customer_ID"].unique()]
    return customer_options

@app.callback(
    Output("inventoryTable", "children"),
    [Input("productSelect", "value"), Input("inventoryInput", "contents")]
)
def update_inventory_table(selected_product, contents_inventory):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update

    content_type, content_string = contents_inventory[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_inventory = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    product_inventory = df_inventory[df_inventory["Product_ID"] == selected_product]
    inventory_table = dbc.Table.from_dataframe(product_inventory, striped=True, bordered=True, hover=True)
    return inventory_table

@app.callback(
    Output("customerMetricsPlot", "figure"),
    [Input("productSelect", "value"), Input("salesInput", "contents"), Input("customerMetricsDropdown", "value")]
)
def update_customer_metrics_plot(selected_product, contents_sales, selected_metric):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    product_sales = df_sales[df_sales["Product_ID"] == selected_product]
    if selected_metric == "Total_Units_Sold":
        fig = px.bar(product_sales, x="Customer_ID", y="Units_Sold", title="Total Units Sold by Customer")
    elif selected_metric == "Total_Revenue":
        fig = px.bar(product_sales, x="Customer_ID", y="Total_Revenue", title="Total Revenue by Customer")
    else:
        fig = go.Figure()
    return fig

@app.callback(
    [Output("feedbackRatingsPlot", "figure"), Output("selectedFeedbackText", "value")],
    [Input("feedbackProductSelect", "value"), Input("feedbackInput", "contents")]
)
def update_feedback_ratings_plot(selected_product, contents_feedback):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update, dash.no_update  # No triggers, return dash.no_update

    content_type, content_string = contents_feedback[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_feedback = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    product_feedback = df_feedback[df_feedback["Product_ID"] == selected_product]

    fig = px.bar(product_feedback, x="Customer_ID", y="Rating", title="Customer Ratings for Selected Product")

    selected_feedback = product_feedback["Review"].iloc[0]
    return fig, selected_feedback

@app.callback(
    [Output("filterBrand", "options"),
     Output("filterSupplier", "options"),
     Output("filterMaterial", "options")],
    [Input("productSelect", "value"),
     Input("salesInput", "contents")] 
)
def update_filter_options(selected_product, contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update  # No triggers, return dash.no_update

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))

    brands_for_product = df_sales[df_sales["Product_ID"] == selected_product]["Brand"].unique()
    suppliers_for_product = df_sales[df_sales["Product_ID"] == selected_product]["Supplier"].unique()
    materials_for_product = df_sales[df_sales["Product_ID"] == selected_product]["Material"].unique()

    brand_options = [{"label": brand, "value": brand} for brand in brands_for_product]
    supplier_options = [{"label": supplier, "value": supplier} for supplier in suppliers_for_product]
    material_options = [{"label": material, "value": material} for material in materials_for_product]

    return brand_options, supplier_options, material_options

    
@app.callback(
    Output("filterSeason", "options"),
    [Input("productSelect", "value"),
     Input("salesInput", "contents")] 
)
def update_season_options(selected_product, contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))

    seasons_for_product = df_sales[df_sales["Product_ID"] == selected_product]["Season"].unique()

    season_options = [{"label": season, "value": season} for season in seasons_for_product]

    return season_options

    
@app.callback(
    Output("filterSize", "options"),
    [Input("productSelect", "value"),
     Input("salesInput", "contents")] 
)
def update_size_options(selected_product, contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))

    sizes_for_product = df_sales[df_sales["Product_ID"] == selected_product]["Size"].unique()

    size_options = [{"label": size, "value": size} for size in sizes_for_product]

    return size_options


@app.callback(
    Output("filterStoreLocation", "options"),
    [Input("productSelect", "value"),
     Input("salesInput", "contents")]
)
def update_store_location_options(selected_product, contents_sales):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update

    content_type, content_string = contents_sales[0].split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))

    locations_for_product = df_sales[df_sales["Product_ID"] == selected_product]["StoreLocation"].unique()

    store_location_options = [{"label": location, "value": location} for location in locations_for_product]

    return store_location_options


@app.callback(
    Output("feedbackSentimentAnalysis", "figure"),
    [Input("selectedFeedbackText", "value")]
)
def update_sentiment_analysis_plot(selected_feedback_text):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update

    if selected_feedback_text:
        sentiment_scores = {"Positive": 0.6, "Neutral": 0.3, "Negative": 0.1}

        fig = px.pie(
            names=list(sentiment_scores.keys()),
            values=list(sentiment_scores.values()),
            title="Sentiment Analysis of Selected Feedback"
        )
        return fig
    else:
        return {}

@app.callback(
    Output("optimization_result", "children"),
    [Input("optimize_button", "n_clicks"),
     Input("coef_x1", "value"),
     Input("coef_x2", "value"),
     Input("constraint1_coef_x1", "value"),
     Input("constraint1_coef_x2", "value"),
     Input("constraint1_rhs", "value"),
     Input("constraint2_coef_x1", "value"),
     Input("constraint2_coef_x2", "value"),
     Input("constraint2_rhs", "value")]
)
def optimize_lp(n_clicks, coef_x1, coef_x2, constraint1_coef_x1, constraint1_coef_x2, constraint1_rhs, constraint2_coef_x1, constraint2_coef_x2, constraint2_rhs):
    ctx = dash.callback_context  # Get the callback context

    if not ctx.triggered:
        return dash.no_update  # No triggers, return dash.no_update

    if n_clicks is None:
        return dash.no_update  # Button not clicked yet

    lp_problem = pulp.LpProblem("Inventory_Optimization", pulp.LpMaximize)

    x1 = pulp.LpVariable("Product1_Order", lowBound=0, cat=pulp.LpInteger)
    x2 = pulp.LpVariable("Product2_Order", lowBound=0, cat=pulp.LpInteger)

    objective_function = coef_x1 * x1 + coef_x2 * x2

    lp_problem += objective_function

    constraint1 = constraint1_coef_x1 * x1 + constraint1_coef_x2 * x2 <= constraint1_rhs
    constraint2 = constraint2_coef_x1 * x1 + constraint2_coef_x2 * x2 <= constraint2_rhs

    lp_problem += constraint1
    lp_problem += constraint2

    lp_problem.solve()

    if pulp.LpStatus[lp_problem.status] == "Optimal":
        optimal_x1 = x1.varValue
        optimal_x2 = x2.varValue

        return html.Div([
            html.H3("Optimal Solution:"),
            html.P(f"Product 1 Order Quantity: {optimal_x1}"),
            html.P(f"Product 2 Order Quantity: {optimal_x2}"),
            html.P(f"Optimal Objective Value: {pulp.value(lp_problem.objective)}"),
        ])
    else:
        return html.Div("LP problem has no optimal solution or is infeasible.")

if __name__ == "__main__":
    app.run_server(debug=True, port=8060)

