import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import dash_bootstrap_components as dbc

# Path to the CSV file
csv_file_path = '/home/mila/c/cesare.spinoso/RSASumm/data/multioped/test.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = html.Div([
    dbc.Container([
        html.H1("CSV Data Visualization", className='my-4'),
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'whiteSpace': 'nowrap',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'maxWidth': 0,
            },
            style_data_conditional=[
                {
                    'if': {'column_id': c},
                    'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
                } for c in df.columns
            ],
            tooltip_data=[
                {
                    column: {'value': str(value), 'type': 'markdown'}
                    for column, value in row.items()
                } for row in df.to_dict('records')
            ],
            tooltip_duration=None,
        ),
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
