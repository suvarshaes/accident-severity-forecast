import streamlit as st


def render_home():
    st.title("Welcome to My DTSC691 Project")
    st.subheader("About Me")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(
            """
Hi, I'm Suvarsha. I'm a postgraduate in Computer Science with nearly 12 years of experience in software development, primarily working with ASP.NET and SQL. Throughout my career, I've contributed to scalable enterprise solutions, collaborated with diverse teams, and continuously adapted as technology evolved.

Currently, I'm on a career break to raise my children, and I'm using this time to explore new interests and expand my skill set. I'm particularly passionate about transitioning into Machine Learning and Data Science, and I'm actively learning, experimenting with hands-on projects, and strengthening my foundation in modern AI technologies. This work is part of my final course in Eastern University's Data Science program, where I'm applying everything I've learned through real-world projects.

I chose the U.S. Accidents project because it brings together impactful data science and public safety. Road accidents affect millions of people every year, and I was inspired to explore how predictive modeling can support smarter urban planning, proactive risk identification, and ultimately help communities reduce accident severity. The rich dataset, diverse environmental factors, and practical value of the outcome made it the perfect project for me to challenge myself, learn deeply, and solve a real-world problem with data.

I live in New Jersey with my husband and our two wonderful kids. Outside of work and learning, I enjoy watching movies, cooking, and trying out new recipes.
            """
        )
    with col2:
        st.image("IMG_8491.jpg", width=300)
    