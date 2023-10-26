# Import necessary libraries
import base64
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

# Specify the absolute file path to the 'products.csv' file
file_path = r'C:\Users\GN898ZK\OneDrive - EY\5. My Learnings\33. Python\NIFT Project\NIFT Project\products.csv'

products_data = pd.read_csv(file_path, encoding='utf-8')

df_products = pd.DataFrame({
    'Product_ID': products_data['Product_ID'],
    'Product_Name': products_data['Product_Name'],
    'Category': products_data['Category'],
    'Season': products_data['Season'],
    'Cost_Price': products_data['Cost_Price'],
    'Selling_Price': products_data['Selling_Price'],
    'Brand': products_data['Brand'],
    'Supplier': products_data['Supplier'],
    'Material': products_data['Material'],
    'Store_Location': products_data['Store_Location'],
    'Size': products_data['Size'],
    'Discount': products_data['Discount'],
})

customer_data = pd.read_csv(r'C:\Users\GN898ZK\OneDrive - EY\5. My Learnings\33. Python\NIFT Project\NIFT Project\customer_demographics.csv', sep=',')  

df_customer = pd.DataFrame({
    'Customer_ID': customer_data['Customer_ID'],
    'Customer_Name': customer_data['Customer_Name'],
    'Age': customer_data['Age'],
    'Gender': customer_data['Gender'],
    'Location': customer_data['Location'],
    'Email': customer_data['Email'],
    'Phone_Number': customer_data['Phone_Number'],
})


def read_csv_file(file_name):
    try:
        file_path = r'C:\\Users\\GN898ZK\\OneDrive - EY\\5. My Learnings\\33. Python\\NIFT Project\\NIFT Project\\' + file_name
        data = pd.read_csv(file_path)
        if not data.empty:
            print(f"{file_name} read successfully!")
            if file_name == "sales.csv":
                data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
                data['Date'] = data['Date'].dt.strftime('%m-%d-%Y')
            return data
        else:
            print(f"{file_name} is empty. Please provide a non-empty file.")
    except FileNotFoundError as e:
        print(f"{file_name} not found. Please check the file name and path: {e}")
    except Exception as e:
        print(f"An error occurred while reading {file_name}: {e}")
    return None


inventory_data = read_csv_file("inventory.csv")


feedback_data = read_csv_file("feedback.csv")


customer_metrics_data = read_csv_file("customer_metrics.csv")


df_sales = pd.DataFrame()


def update_filtered_data_table(contents_sales, season, brand, supplier, material, store_location):
    global df_sales 
    if contents_sales:
        df_sales = pd.read_csv(contents_sales[0])

        filtered_df = df_sales
        if season:
            filtered_df = filtered_df[filtered_df["Season"] == season]
        if brand:
            filtered_df = filtered_df[filtered_df["Brand"] == brand]
        if supplier:
            filtered_df = filtered_df[filtered_df["Supplier"] == supplier]
        if material:
            filtered_df = filtered_df[filtered_df["Material"] == material]
        if store_location:
            filtered_df = filtered_df[filtered_df["StoreLocation"] == store_location]

        table = dash_table.DataTable(
            id='datatable-interactivity',
            columns=[
                {"name": col, "id": col} for col in filtered_df.columns
            ],
            data=filtered_df.to_dict('records'),
            style_table={'overflowX': 'scroll'},
            style_data={'whiteSpace': 'normal'},
        )
        return [table]
    else:
        return []
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.config['suppress_callback_exceptions'] = True


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(
    Output("productSelect", "options"),
    [
        Input("salesInput", "contents"),
        Input("feedbackInput", "contents"),
        Input("inventoryInput", "contents"),
        Input("enableUpload", "value"),
    ]
)
def update_product_dropdown_options(contents_sales, contents_feedback, contents_inventory, enable_upload):
    if "enabled" in enable_upload and contents_sales:
        df_sales = pd.read_csv(contents_sales[0])
        product_options = [{"label": product, "value": product} for product in df_sales["Product"].unique()]
        return product_options
    return []


@app.callback(
    Output('datatable-interactivity', 'children'), 
    Input('salesInput', 'contents'),
    Input('filterSeason', 'value'),
    Input('filterBrand', 'value'),
    Input('filterSupplier', 'value'),
    Input('filterMaterial', 'value'),
    Input('filterStoreLocation', 'value'),
)
def update_filtered_data_table_callback(contents_sales, season, brand, supplier, material, store_location):
    return update_filtered_data_table(contents_sales, season, brand, supplier, material, store_location)


@app.callback(
    Output("optimizationOutput", "children"),
    [Input("salesInput", "contents")]
)
def run_optimization(contents_sales):
    if contents_sales:
        
        
        optimization_results = "Optimization results:\n - Optimal order quantity: 100\n - Maximized profit: $10,000"
        return html.P(optimization_results)
    else:
        return html.P("Please upload sales data to perform optimization.")


@app.callback(
    [Output("decompPlot", "figure")],
    [Input("salesInput", "contents")],
    prevent_initial_call=True,
    allow_duplicate=True
)
def update_figures(contents_sales):
    ctx = dash.callback_context 
    
    if not ctx.triggered:
        
        return {}
    
    
    content_type, content_string = contents_sales.split(',')
    decoded_content = base64.b64decode(content_string)
    df_sales = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    
    
    print(df_sales.head())
    
    
    result = seasonal_decompose(df_sales['Total_Revenue'], model='additive', freq=12)

    
    import plotly.graph_objects as go
    fig_decomposition = go.Figure()
    fig_decomposition.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend'))
    fig_decomposition.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal'))
    fig_decomposition.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residual'))
    fig_decomposition.update_layout(title="Decomposition Plot")

    return fig_decomposition


@app.callback(
    Output("salesTrendPlot", "figure"),
    [Input("salesInput", "contents")]
)
def update_sales_trend_plot(contents_sales):
    if contents_sales:
        df_sales = pd.read_csv(contents_sales[0])
        fig = px.line(df_sales, x="Date", y="Total_Revenue", title="Sales Trend")
        return fig
    else:
        return {}


df_sales = None


@app.callback(
    Output("seasonalDecompPlot", "figure"),
    [Input("salesInput", "contents")]
)
def update_seasonal_decomp_plot(contents_sales):
    global df_sales 
    if contents_sales:
        df_sales = pd.read_csv(contents_sales[0])
        
        
        df_sales['Date'] = pd.to_datetime(df_sales['Date'])
        df_sales.set_index('Date', inplace=True)
        decomposition = sm.tsa.seasonal_decompose(df_sales['Total_Revenue'], model='additive', period=7)
        
        
        seasonal_fig = go.Figure()
        seasonal_fig.add_trace(go.Scatter(x=df_sales.index, y=decomposition.trend, mode='lines', name='Trend'))
        seasonal_fig.add_trace(go.Scatter(x=df_sales.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
        seasonal_fig.add_trace(go.Scatter(x=df_sales.index, y=decomposition.resid, mode='lines', name='Residual'))
        seasonal_fig.update_layout(title='Seasonal Decomposition Plot', xaxis_title='Date', yaxis_title='Value')
        
        return seasonal_fig
    else:
        return {}
@app.callback(
    Output("priceSensitivityPlot", "figure"),
    [Input("salesInput", "contents")]
)
def update_price_sensitivity_plot(contents_sales):
    if contents_sales:
        df_sales = pd.read_csv(contents_sales[0])
        fig = px.scatter(df_sales, x="Total_Revenue", y="Units_Sold", title="Price Sensitivity")
        return fig
    else:
        return {}


@app.callback(
    Output("demandForecastPlot", "figure"),
    [Input("salesInput", "contents")]
)
def update_demand_forecast_plot(contents_sales):
    if contents_sales:
        df_sales = pd.read_csv(contents_sales[0])
        
        
        demand_fig = px.line(df_sales, x="Date", y="Units_Sold", title="Demand Forecast")
        return demand_fig
    else:
        return {}


@app.callback(
    Output("dateRangeSelect", "start_date"),
    [Input("salesInput", "contents")]
)
def update_start_date(contents_sales):
    if contents_sales:
        df_sales = pd.read_csv(contents_sales[0])
        start_date = df_sales["Date"].min()
        return start_date
    else:
        return None


@app.callback(
    Output("dateRangeSelect", "end_date"),
    [Input("salesInput", "contents")]
)
def update_end_date(contents_sales):
    if contents_sales:
        df_sales = pd.read_csv(contents_sales[0])
        end_date = df_sales["Date"].max()
        return end_date
    else:
        return None

@app.callback(
    Output("ratingEffectPlot", "figure"),
    [Input("salesInput", "contents")]
)
def update_rating_effect_plot(contents_sales):
    if contents_sales:
        df_sales = pd.read_csv(contents_sales[0])
        import statsmodels.api as sm
        X = df_sales['Rating']
        X = sm.add_constant(X)
        model = sm.OLS(df_sales['Total_Revenue'], X).fit()
        fig = px.scatter(df_sales, x="Rating", y=model.predict(X), title="Rating Effect")
        return fig
    else:
        return {}

@app.callback(
    [Output("customerInfo", "children"), Output("genderDistribution", "figure")],
    [Input("customerSelect", "value"), Input("salesInput", "contents")]
)
def update_customer_info(selected_customer, contents_sales):
    if contents_sales and selected_customer:
        df_sales = pd.read_csv(contents_sales[0])
        # Use the correct file path for 'customer_demographics.csv'
        customer_file_path = r'C:\Users\GN898ZK\OneDrive - EY\5. My Learnings\33. Python\NIFT Project\NIFT Project\customer_demographics.csv'
        df_customer = pd.read_csv(customer_file_path)
        customer_data = df_sales[df_sales["Customer_ID"] == selected_customer]
        customer_table = dbc.Table.from_dataframe(customer_data, striped=True, bordered=True, hover=True)
        gender_counts = df_customer["Gender"].value_counts()
        fig = px.pie(gender_counts, names=gender_counts.index, values=gender_counts.values,
                     title="Gender Distribution of Customers")
        return customer_table, fig
    else:
        return "Select a customer to view their information.", {}


@app.callback(
    Output("customerSelect", "options"),
    [Input("salesInput", "contents")]
)
def update_customer_dropdown(contents_sales):
    if contents_sales:
        df_sales = pd.read_csv(contents_sales[0])
        customer_options = [{"label": customer_id, "value": customer_id} for customer_id in df_sales["Customer_ID"].unique()]
        return customer_options
    else:
        return []

@app.callback(
    Output("inventoryTable", "children"),
    [Input("productSelect", "value"), Input("inventoryInput", "contents")]
)
def update_inventory_table(selected_product, contents_inventory):
    if contents_inventory and selected_product:
        df_inventory = pd.read_csv(contents_inventory[0])
        product_inventory = df_inventory[df_inventory["Product_ID"] == selected_product]
        inventory_table = dbc.Table.from_dataframe(product_inventory, striped=True, bordered=True, hover=True)
        return inventory_table
    else:
        return "Select a product to view its inventory information."
@app.callback(
    Output("customerMetricsPlot", "figure"),
    [Input("productSelect", "value"), Input("salesInput", "contents"), Input("customerMetricsDropdown", "value")]
)
def update_customer_metrics_plot(selected_product, contents_sales, selected_metric):
    if contents_sales and selected_product and selected_metric:
        df_sales = pd.read_csv(contents_sales[0])
        product_sales = df_sales[df_sales["Product_ID"] == selected_product]
        if selected_metric == "Total_Units_Sold":
            fig = px.bar(product_sales, x="Customer_ID", y="Units_Sold", title="Total Units Sold by Customer")
        elif selected_metric == "Total_Revenue":
            fig = px.bar(product_sales, x="Customer_ID", y="Total_Revenue", title="Total Revenue by Customer")
        else: 
            fig = go.Figure()
        return fig
    else:
        return {}

@app.callback(
    Output("feedbackRatingsPlot", "figure"),
    Output("selectedFeedbackText", "value"),
    [Input("feedbackProductSelect", "value"),
     Input("feedbackInput", "contents")] 
)
def update_feedback_ratings_plot(selected_product, contents_feedback):
    if contents_feedback and selected_product:
        df_feedback = pd.read_csv(contents_feedback[0])
        product_feedback = df_feedback[df_feedback["Product_ID"] == selected_product]
        
       
        fig = px.bar(product_feedback, x="Customer_ID", y="Rating", title="Customer Ratings for Selected Product")
        
       
        selected_feedback = product_feedback["Review"].iloc[0] 
        return fig, selected_feedback
    else:
        return {}, ""

@app.callback(
    [Output("filterBrand", "options"),
     Output("filterSupplier", "options"),
     Output("filterMaterial", "options")],
    [Input("productSelect", "value"),
     Input("salesInput", "contents")] 
)
def update_filter_options(selected_product, contents_sales):
    if selected_product and contents_sales:
        df_sales = pd.read_csv(contents_sales[0])

       
        brands_for_product = df_sales[df_sales["Product_ID"] == selected_product]["Brand"].unique()
        suppliers_for_product = df_sales[df_sales["Product_ID"] == selected_product]["Supplier"].unique()
        materials_for_product = df_sales[df_sales["Product_ID"] == selected_product]["Material"].unique()

       
        brand_options = [{"label": brand, "value": brand} for brand in brands_for_product]
        supplier_options = [{"label": supplier, "value": supplier} for supplier in suppliers_for_product]
        material_options = [{"label": material, "value": material} for material in materials_for_product]

        return brand_options, supplier_options, material_options
    else:
        return [], [], []
    
@app.callback(
    Output("filterSeason", "options"),
    [Input("productSelect", "value"),
     Input("salesInput", "contents")] 
)
def update_season_options(selected_product, contents_sales):
    if selected_product and contents_sales:
        df_sales = pd.read_csv(contents_sales[0])

       
        seasons_for_product = df_sales[df_sales["Product_ID"] == selected_product]["Season"].unique()

       
        season_options = [{"label": season, "value": season} for season in seasons_for_product]

        return season_options
    else:
        return []
    
@app.callback(
    Output("filterSize", "options"),
    [Input("productSelect", "value"),
     Input("salesInput", "contents")] 
)
def update_size_options(selected_product, contents_sales):
    if selected_product and contents_sales:
        df_sales = pd.read_csv(contents_sales[0])

       
        sizes_for_product = df_sales[df_sales["Product_ID"] == selected_product]["Size"].unique()

       
        size_options = [{"label": size, "value": size} for size in sizes_for_product]

        return size_options
    else:
        return []
@app.callback(
    Output("filterStoreLocation", "options"),
    [Input("productSelect", "value"),
     Input("salesInput", "contents")]
)
def update_store_location_options(selected_product, contents_sales):
    if selected_product and contents_sales:
        df_sales = pd.read_csv(contents_sales[0])

        locations_for_product = df_sales[df_sales["Product_ID"] == selected_product]["StoreLocation"].unique()

        store_location_options = [{"label": location, "value": location} for location in locations_for_product]

        return store_location_options
    else:
        return []


@app.callback(
    Output("feedbackSentimentAnalysis", "figure"),
    [Input("selectedFeedbackText", "value")]
)
def update_sentiment_analysis_plot(selected_feedback_text):
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
import pulp

lp_problem = pulp.LpProblem("Inventory_Optimization", pulp.LpMaximize)

x1 = pulp.LpVariable("Product1_Order", lowBound=0, cat=pulp.LpInteger) 
x2 = pulp.LpVariable("Product2_Order", lowBound=0, cat=pulp.LpInteger) 

objective_function = 10 * x1 + 15 * x2 

lp_problem += objective_function 

constraint1 = 2 * x1 + 3 * x2 <= 100 
constraint2 = x1 + 2 * x2 <= 60 

lp_problem += constraint1 
lp_problem += constraint2 

lp_problem.solve()

if pulp.LpStatus[lp_problem.status] == "Optimal":
   
    optimal_x1 = x1.varValue
    optimal_x2 = x2.varValue

    print("Optimal Solution:")
    print(f"Product 1 Order Quantity: {optimal_x1}")
    print(f"Product 2 Order Quantity: {optimal_x2}")
    print(f"Optimal Objective Value: {pulp.value(lp_problem.objective)}")
else:
    print("LP problem has no optimal solution or is infeasible.")

if __name__ == "__main__":
    app.run_server(debug=True, port=8060)
