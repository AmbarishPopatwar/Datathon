
import plotly.graph_objects as go
import json

# Data
data = {
  "models": ["LR Baseline", "LR Engineered", "RF Baseline", "RF Engineered"],
  "r2_scores": [0.60, 0.64, 0.81, 0.84],
  "rmse_values": [0.73, 0.69, 0.51, 0.47]
}

# Create figure
fig = go.Figure()

# Add R² Score bars (green)
fig.add_trace(go.Bar(
    name='R² (higher better)',
    x=data['models'],
    y=data['r2_scores'],
    marker_color='#2E8B57',
    text=[f'{val:.2f}' for val in data['r2_scores']],
    textposition='outside',
    cliponaxis=False
))

# Add RMSE bars (red)
fig.add_trace(go.Bar(
    name='RMSE (lower better)',
    x=data['models'],
    y=data['rmse_values'],
    marker_color='#DB4545',
    text=[f'{val:.2f}' for val in data['rmse_values']],
    textposition='outside',
    cliponaxis=False
))

# Add horizontal line at R² = 0.80
fig.add_hline(y=0.80, line_dash="dash", line_color="gray", opacity=0.7)

# Update layout
fig.update_layout(
    title='Model Performance: Feature Engineering',
    xaxis_title='Model Type',
    yaxis_title='Metric Value',
    barmode='group',
    yaxis=dict(range=[0, 1.0]),
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('chart.png')
fig.write_image('chart.svg', format='svg')
