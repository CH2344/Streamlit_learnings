import streamlit as st
import torch 
import matplotlib.pyplot as plt
st.title('PyTorch Distributions')

mu = st.slider(label='Mean',min_value=-10,max_value=10,value=1)
sigma = st.slider(label='Variance',min_value=0.1,max_value=10.0,value=1.0)
normal_dist = torch.distributions.Normal(mu, sigma)

st.markdown("The PDF of a normal distribution is given by: ")
st.latex(r"f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^-(\frac{x-\mu}{\sigma})^2")
x_range = torch.linspace(-10, 10, 1000)
pdf = normal_dist.log_prob(x_range).exp()
plt.plot(x_range.numpy(), pdf.numpy())
plt.xlabel('x')
plt.ylabel('Probability Density')
st.pyplot(plt)