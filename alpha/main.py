# # # Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Input, Output
from dash import dcc, html
import dash_bootstrap_components as dbc

from starter import app
from apps import upload, retrain, archive

# transripts = {'art.csv':'art.wav'}, {'cat.csv': 'cat.wav'} #contains the csv and wav for a transcript
df = "Hello World"

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 5,
    "left": 0,
    "bottom": 0,
    "width": "14rem",
    "padding-left": ".5rem",
    # "padding": "2rem 1rem",
    "background-color": "#18453b",
    "color": "#FFFFFF",
    # "wrap-text": True,
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "2rem",
    "margin-right": "25rem",
    "margin-bottom": "5rem",
    "padding": "2rem 1rem",

}

sidebar = html.Div(

    [
        # html.H2(html.Img(id='img', src='assets/MiDiaries.jpeg', style={'width':'40%',}),"MI Diaries", className="display-4"),
        html.H2("MI Diaries", className="display-4", style={'font-size': "36pt"}),
        html.Hr(),
        html.P(
            html.Img(id='img', src='assets/MiDiaries.jpeg', style={'width': '20%', 'margin-left': '27%'}),
        ),
        dbc.Nav(
            [
                # Unnecessary but currently hold the space (They should be invisible)
                dbc.NavLink("Upload", href="/", active="exact"),
                dbc.NavLink("Train", href="/page-1", active="exact"),
                dbc.NavLink("Transciptions", href="/page-2", active="exact"),
            ],
            horizontal=True,
            pills=True,
            className="nav-link.active"

        ),
    ],
    style=SIDEBAR_STYLE,className="nav-pills",
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

bar = html.Div(id='main-page-content',children=[

    #nav wrapper starts here
    html.Div(
        children=[
            #nav bar
            html.Nav(
                #inside div
                html.Div(
                    children=[
                        html.A(
                            'MI Diaries',
                            className='brand-logo',
                            href='/'
                        ),
                        #ul list components
                        html.Ul(
                            children=[
                               html.Li(html.A('Upload a File', href='/upload')),
                               html.Li(html.A('Select a Model', href='/page2')),
                               html.Li(html.A('Trascript Archive', href='/page3')),
                            ],
                            id='nav-mobile',
                            className='right hide-on-med-and-down'

                        ),
                    ],
                    className='nav-wrapper',
                ),style={'background-color':'#18453b'}),

        ],
        className='navbar-scroll'
    ),
])

app.layout = html.Div([
    bar,
    dcc.Location(id="url", refresh=False),
    sidebar,
    content,
    dcc.Store(id='stored-data', data=df, storage_type='session'),
])


@app.callback(Output("page-content", "children"),
              [Input("url", "pathname")])

def display_page(pathname):
    if pathname == '/':
        return upload.layout
    if pathname == '/page-1':
        return retrain.layout
    if pathname == '/page-2':
        # set up a panda for the table then use dcc.Store
        # use archive to show transcript of the file
        return archive.layout
    else:
        return "404 Page Error! Please choose a link"

if (__name__ == '__main__'):
    app.run_server(debug=True, dev_tools_hot_reload=False)