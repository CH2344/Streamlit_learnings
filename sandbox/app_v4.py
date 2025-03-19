import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import streamlit as st

st.title('PyTorch Distributions')

mu = st.slider('Mean', -10.0, 10.0, 0.0)
sigma = st.slider('Standard Deviation', 0.1, 10.0, 1.0)

normal_dist = torch.distributions.Normal(mu, sigma)

x_range = torch.linspace(-10, 10, 1000)
pdf = normal_dist.log_prob(x_range).exp()

st.markdown('The **probability density function** of a normal distribution is given by:')
st.latex(r'f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)')


data = {
    'Property': ['Mean', 'Standard Deviation', 'Variance', 'Entropy'],
    'Value': [mu, sigma, sigma**2, normal_dist.entropy().item()]
}

df = pd.DataFrame(data)
st.table(df)

fig = go.Figure(data=go.Scatter(x=x_range,
                                y=pdf,
                                name="Line plot",
                                mode='lines+markers',
                                marker=dict(size=4, color='red'),
                                line=dict(color='blue', width=2)))

fig.update_layout(width=2000,height=600)

st.plotly_chart(fig)
