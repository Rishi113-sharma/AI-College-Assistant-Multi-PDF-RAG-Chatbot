import streamlit as st
from workflow import app


st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered"
)




st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.programme-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)



st.markdown(
    '<div class="title">🎓 College Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about academics, fees, or general topics.</div>',
    unsafe_allow_html=True
)




st.sidebar.title("Student Information")

programme = st.sidebar.selectbox(
    "Select your programme",
    [
        "BCA",
        "BBA",
        "B.COM(H)",
        "B.TECH"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write("### Available Information")
st.sidebar.write("📚 Academic Handbook")
st.sidebar.write("💰 Fee Structure")

st.sidebar.markdown("---")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])




user_query = st.chat_input(
    "Ask your question..."
)


if user_query:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)


    # Run LangGraph
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = app.invoke(
                    {
                        "programme": programme,
                        "messages": [
                            ("human", user_query)
                        ]
                    }
                )

                answer = result["messages"][-1].content

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    f"Something went wrong:\n\n{str(e)}"
                )