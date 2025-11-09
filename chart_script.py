
import plotly.graph_objects as go
import numpy as np

# Create a flowchart-style diagram using plotly
fig = go.Figure()

# Define positions for nodes (x, y coordinates)
# Row 1: Data Loading & Inspection
row1_y = 9
nodes_r1 = [
    {'x': 2, 'y': row1_y, 'text': '📊 Data Loading<br>& Inspection', 'color': '#B3E5EC', 'width': 1.5},
]
sub_r1 = [
    {'x': 0.5, 'y': row1_y-1, 'text': 'Load CA<br>Housing<br>20,640', 'color': '#B3E5EC', 'width': 0.8},
    {'x': 2, 'y': row1_y-1, 'text': 'Check<br>missing<br>values', 'color': '#B3E5EC', 'width': 0.8},
    {'x': 3.5, 'y': row1_y-1, 'text': 'Identify<br>outliers', 'color': '#B3E5EC', 'width': 0.8},
]

# Row 2: EDA
row2_y = 6.5
nodes_r2 = [
    {'x': 2, 'y': row2_y, 'text': '🔍 Exploratory<br>Data Analysis', 'color': '#A5D6A7', 'width': 1.5},
]
sub_r2 = [
    {'x': 0.5, 'y': row2_y-1, 'text': 'Feature<br>distrib.', 'color': '#A5D6A7', 'width': 0.8},
    {'x': 2, 'y': row2_y-1, 'text': 'Correlation<br>analysis', 'color': '#A5D6A7', 'width': 0.8},
    {'x': 3.5, 'y': row2_y-1, 'text': 'Geographic<br>visual.', 'color': '#A5D6A7', 'width': 0.8},
]

# Row 3: Feature Engineering
row3_y = 4
nodes_r3 = [
    {'x': 2, 'y': row3_y, 'text': '🛠️ Feature Eng.', 'color': '#FFEB8A', 'width': 1.5},
]
sub_r3 = [
    {'x': 2, 'y': row3_y-0.8, 'text': 'Create 7 features<br>RoomsPerHouse<br>BedroomsRatio<br>LuxuryScore', 'color': '#FFEB8A', 'width': 1.2},
]

# Row 4: Model Training (two branches)
row4_y = 1.5
nodes_r4 = [
    {'x': 0.7, 'y': row4_y, 'text': '⚙️ Baseline<br>Models', 'color': '#FFCDD2', 'width': 1.0},
    {'x': 3.3, 'y': row4_y, 'text': '⚙️ Engineered<br>Models', 'color': '#FFCDD2', 'width': 1.0},
]
sub_r4 = [
    {'x': 0.2, 'y': row4_y-1, 'text': 'Linear Reg<br>R²≈0.60', 'color': '#FFCDD2', 'width': 0.7},
    {'x': 1.2, 'y': row4_y-1, 'text': 'Random For<br>R²≈0.81', 'color': '#FFCDD2', 'width': 0.7},
    {'x': 2.8, 'y': row4_y-1, 'text': 'Linear Reg<br>R²≈0.64', 'color': '#FFCDD2', 'width': 0.7},
    {'x': 3.8, 'y': row4_y-1, 'text': 'Random For<br>R²≈0.84<br>⭐BEST', 'color': '#FFD700', 'width': 0.7},
]

# Row 5: Evaluation
row5_y = -1
nodes_r5 = [
    {'x': 2, 'y': row5_y, 'text': '📈 Evaluation<br>& Analysis', 'color': '#9FA8B0', 'width': 1.5},
]
sub_r5 = [
    {'x': 0.5, 'y': row5_y-1, 'text': 'RMSE & R²<br>metrics', 'color': '#9FA8B0', 'width': 0.8},
    {'x': 2, 'y': row5_y-1, 'text': 'Feature<br>importance', 'color': '#9FA8B0', 'width': 0.8},
    {'x': 3.5, 'y': row5_y-1, 'text': 'Model<br>comparison', 'color': '#9FA8B0', 'width': 0.8},
]

# Row 6: Outputs
row6_y = -3.5
nodes_r6 = [
    {'x': 0.5, 'y': row6_y, 'text': '💾 Trained<br>Models<br>.pkl files', 'color': '#E1BEE7', 'width': 0.8},
    {'x': 2, 'y': row6_y, 'text': '🌐 Interactive<br>App<br>Streamlit', 'color': '#E1BEE7', 'width': 0.8},
    {'x': 3.5, 'y': row6_y, 'text': '📊 Reports &<br>Visualiz.', 'color': '#E1BEE7', 'width': 0.8},
]

# Function to add a box
def add_box(x, y, text, color, width=1.0, height=0.6):
    fig.add_shape(
        type="rect",
        x0=x-width/2, y0=y-height/2,
        x1=x+width/2, y1=y+height/2,
        line=dict(color="#333333", width=2),
        fillcolor=color,
    )
    fig.add_annotation(
        x=x, y=y,
        text=text,
        showarrow=False,
        font=dict(size=10, color="#13343b"),
        align="center",
    )

# Add all nodes
for node in nodes_r1 + sub_r1 + nodes_r2 + sub_r2 + nodes_r3 + sub_r3 + nodes_r4 + sub_r4 + nodes_r5 + sub_r5 + nodes_r6:
    add_box(node['x'], node['y'], node['text'], node['color'], node['width'])

# Add arrows
arrows = [
    # Row 1 to sub-row 1
    (2, row1_y-0.3, 0.5, row1_y-0.7),
    (2, row1_y-0.3, 2, row1_y-0.7),
    (2, row1_y-0.3, 3.5, row1_y-0.7),
    # Sub-row 1 to row 2
    (0.5, row1_y-1.3, 2, row2_y+0.3),
    (2, row1_y-1.3, 2, row2_y+0.3),
    (3.5, row1_y-1.3, 2, row2_y+0.3),
    # Row 2 to sub-row 2
    (2, row2_y-0.3, 0.5, row2_y-0.7),
    (2, row2_y-0.3, 2, row2_y-0.7),
    (2, row2_y-0.3, 3.5, row2_y-0.7),
    # Sub-row 2 to row 3
    (0.5, row2_y-1.3, 2, row3_y+0.3),
    (2, row2_y-1.3, 2, row3_y+0.3),
    (3.5, row2_y-1.3, 2, row3_y+0.3),
    # Row 3 to sub-row 3
    (2, row3_y-0.3, 2, row3_y-0.4),
    # Sub-row 3 to row 4 branches
    (2, row3_y-1.2, 0.7, row4_y+0.3),
    (2, row3_y-1.2, 3.3, row4_y+0.3),
    # Row 4 to sub-row 4
    (0.7, row4_y-0.3, 0.2, row4_y-0.7),
    (0.7, row4_y-0.3, 1.2, row4_y-0.7),
    (3.3, row4_y-0.3, 2.8, row4_y-0.7),
    (3.3, row4_y-0.3, 3.8, row4_y-0.7),
    # Sub-row 4 to row 5
    (0.2, row4_y-1.3, 2, row5_y+0.3),
    (1.2, row4_y-1.3, 2, row5_y+0.3),
    (2.8, row4_y-1.3, 2, row5_y+0.3),
    (3.8, row4_y-1.3, 2, row5_y+0.3),
    # Row 5 to sub-row 5
    (2, row5_y-0.3, 0.5, row5_y-0.7),
    (2, row5_y-0.3, 2, row5_y-0.7),
    (2, row5_y-0.3, 3.5, row5_y-0.7),
    # Sub-row 5 to row 6
    (0.5, row5_y-1.3, 0.5, row6_y+0.3),
    (2, row5_y-1.3, 2, row6_y+0.3),
    (3.5, row5_y-1.3, 3.5, row6_y+0.3),
]

for x0, y0, x1, y1 in arrows:
    fig.add_annotation(
        x=x1, y=y1,
        ax=x0, ay=y0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#333333",
    )

# Update layout
fig.update_layout(
    title="Housing Price Prediction Workflow",
    xaxis=dict(range=[-0.5, 4.5], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(range=[-4.5, 10], showgrid=False, zeroline=False, visible=False),
    showlegend=False,
    plot_bgcolor='#F3F3EE',
    paper_bgcolor='#F3F3EE',
)

# Save the figure
fig.write_image('housing_workflow.png')
fig.write_image('housing_workflow.svg', format='svg')

print("Flowchart created successfully!")
