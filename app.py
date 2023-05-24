# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Load dataset
data = pd.read_csv('data/winequality-red.csv')

# Check for missing values
data.isna().sum()

# Remove duplicate data
data.drop_duplicates(keep='first', inplace=True)

# Calculate the correlation matrix
corr_matrix = data.corr()

# Label quality into Good (1) and Bad (0)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.0 else 0)

# Drop the target variable
X = data.drop('quality', axis=1)

# Set the target variable as the label
y = data['quality']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create an instance of the logistic regression model
logreg_model = LogisticRegression()

# Fit the model to the training data
logreg_model.fit(X_train, y_train)

# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Add some style to the dashboard
app.css.append_css({'external_url': 'styles.css'})

# Define the layout of the dashboard
app.layout = html.Div(
    children=[
        html.H1('CO544-2023 Lab 3: Wine Quality Prediction'),

        html.Div([
            html.H3('Exploratory Data Analysis'),
            html.Label('Feature 1 (X-axis)'),
            dcc.Dropdown(
                id='x_feature',
                options=[{'label': col, 'value': col} for col in data.columns],
                value=data.columns[0]
            )
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            html.Label('Feature 2 (Y-axis)'),
            dcc.Dropdown(
                id='y_feature',
                options=[{'label': col, 'value': col} for col in data.columns],
                value=data.columns[1]
            )
        ], style={'width': '30%', 'display': 'inline-block'}),

        dcc.Graph(id='correlation_plot'),

        # Table with 3 columns and 4 rows
        html.H3("Wine Quality Prediction"),
        html.Div([
            html.Table([
                html.Tr([
                    html.Td([html.Label("Fixed Acidity")]),
                    html.Td([dcc.Input(id='fixed_acidity', type='number', required=True)]),
                    html.Td([html.Label("Volatile Acidity")]),
                    html.Td([dcc.Input(id='volatile_acidity', type='number', required=True)]),
                    html.Td([html.Label("Citric Acid")]),
                    html.Td([dcc.Input(id='citric_acid', type='number', required=True)])
                ]),
                html.Tr([
                    html.Td([html.Label("Residual Sugar")]),
                    html.Td([dcc.Input(id='residual_sugar', type='number', required=True)]),
                    html.Td([html.Label("Chlorides")]),
                    html.Td([dcc.Input(id='chlorides', type='number', required=True)]),
                    html.Td([html.Label("Free Sulfur Dioxide")]),
                    html.Td([dcc.Input(id='free_sulfur_dioxide', type='number', required=True)])
                ]),
                html.Tr([
                    html.Td([html.Label("Total Sulfur Dioxide")]),
                    html.Td([dcc.Input(id='total_sulfur_dioxide', type='number', required=True)]),
                    html.Td([html.Label("Density")]),
                    html.Td([dcc.Input(id='density', type='number', required=True)]),
                    html.Td([html.Label("pH")]),
                    html.Td([dcc.Input(id='ph', type='number', required=True)])
                ]),
                html.Tr([
                    html.Td([html.Label("Sulphates")]),
                    html.Td([dcc.Input(id='sulphates', type='number', required=True)]),
                    html.Td([html.Label("Alcohol")]),
                    html.Td([dcc.Input(id='alcohol', type='number', required=True)])
                ])
            ])
        ]),

        html.Div([
            html.Button('Predict', id='predict-button', n_clicks=0),
        ]),

        html.Div([
            html.H4("Predicted Quality"),
            html.Div(id='prediction-output')
        ])
    ]
)


# Define the callback to update the correlation plot
@app.callback(
    Output('correlation_plot', 'figure'),
    [Input('x_feature', 'value'),
     Input('y_feature', 'value')]
)
def update_correlation_plot(x_feature, y_feature):
    fig = px.scatter(data, x=x_feature, y=y_feature, color='quality')
    fig.update_layout(title=f"Correlation between {x_feature} and {y_feature}")
    return fig


# Define the callback function to predict wine quality
@app.callback(
    Output(component_id='prediction-output', component_property='children'),
    [Input('predict-button', 'n_clicks')],
    [State('fixed_acidity', 'value'),
     State('volatile_acidity', 'value'),
     State('citric_acid', 'value'),
     State('residual_sugar', 'value'),
     State('chlorides', 'value'),
     State('free_sulfur_dioxide', 'value'),
     State('total_sulfur_dioxide', 'value'),
     State('density', 'value'),
     State('ph', 'value'),
     State('sulphates', 'value'),
     State('alcohol', 'value')]
)
def predict_quality(n_clicks, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                    free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol):
    # Create input features array for prediction
    input_features = np.array([
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol
    ]).reshape(1, -1)

    # Predict the wine quality (0 = bad, 1 = good)
    prediction = logreg_model.predict(input_features)[0]

    # Return the prediction
    if prediction == 1:
        return 'This wine is predicted to be good quality.'
    else:
        return 'This wine is predicted to be bad quality.'


if __name__ == '__main__':
    app.run_server(debug=False)
