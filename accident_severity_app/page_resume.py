import streamlit as st


def render_resume():
    st.title("Resume")
    st.markdown("### Education")
    st.write(
        """
        - Master’s Degree in Computer Science 
        - Bachelor degree in Chemistry 
        """
    )
    st.markdown("### Work Experience")
    st.write(
        """
          - Ernst & Young Pvt Ltd (December 2013 - October 2021)
          -	EXL Services (February 2012 - December 2013)
          -	Anya Technologies (October 2009 - January 2012)

        """
    )
   
