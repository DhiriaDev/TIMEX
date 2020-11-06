import pickle

import dash
import dash_html_components as html

# Initialize Dash app.
app = dash.Dash(__name__)
server = app.server

# Get children fromfile
with open('children.pkl', 'rb') as input_file:
    children = pickle.load(input_file)


print("Serving the layout...")
app.layout = html.Div(children=children)


if __name__ == '__main__':
    app.run_server(port=10000)
