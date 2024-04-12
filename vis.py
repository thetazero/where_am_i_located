import plotly.express as px
import pandas as pd
import json

with open('predictions.json') as f:
    data = json.load(f)


def reshape_data(data):
    new_data = []
    for data_point in data:
        predicted = data_point['predicted']
        true = data_point['true']
        size = 100
        new_data.append((true[0], true[1], 1, size))
        new_data.append((predicted[0], predicted[1], 0, size))
    return new_data


data = reshape_data(data)


df = pd.DataFrame(data,
                  columns=['lat', 'lon', 'is_true', 'size']
                  )

print(df)


fig = px.scatter_mapbox(df,
                        lat='lat',
                        lon='lon',
                        color='is_true',
                        size='size',
                        zoom=16,
                        height=800,
                        width=800)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()
