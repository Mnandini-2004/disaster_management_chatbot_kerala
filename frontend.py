'''import streamlit as st

st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})'''
import streamlit as st

# Import backend functions
from backend import generate_weather_report
from backend1 import response_query,response_selector,palce_finder,response_tavily,response_from_news,plot_map
from model_predict import predict_class
#process_image

st.title("ChatBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# React to user input
user_query = st.text_input("You:", key="user_input")
if user_query:
    print("Display user message in chat message container")
    #print(predict_class())
    s=response_selector(user_query)
    print("s is ")
    print(s)
    if '0' in s:
        #st.session_state.messages.append({"role": "user", "content": user_query})
        city_name=palce_finder(user_query)
    # Get response from backend
        response =  generate_weather_report(city_name)
    elif "1" in s:
        print("1")    
        response =  response_query(user_query)
    elif "2" in s:
        print("2") 
        response=response_from_news(user_query)
    elif "3" in s:
        print('3')
        response = predict_class()
    elif "4" in s:
        print("cat 4")
        response=plot_map(user_query)    
    else:
        response=response_tavily(user_query)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

#'''# Sidebar for previous chat log
#st.sidebar.title("Chat Log")
#if st.sidebar.button("Show Previous Chats"):
 #   for message in st.session_state.messages:
  #      with st.sidebar.expander(f"{message['role'].title()}"):
   #         st.write(message["content"])
#
# Sidebar for image upload
#st.sidebar.title("Options")
#uploaded_image = st.button("button")
# Handle image upload
#if uploaded_image is not None:
    # Process the uploaded image
 #   processed_image = process_image(uploaded_image)

    # Display the processed image
  #  st.image(processed_image, caption="Uploaded Image", use_column_width=True)'''

# Display current chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



