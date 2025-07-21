# app.py
import streamlit as stl
import requests, json
import uuid

stl.title("GenAI Assistant with Voice")
pmt = stl.text_input("Ask Question")
print("Sesiion_id -",stl.session_state)

if "cid" not in stl.session_state:
    stl.session_state.cid = str(uuid.uuid4())

print("Session ID:", stl.session_state.cid)

if stl.button("Send"):
    response = requests.post("http://localhost:7080/api/agtchat",
                        json={"Question": pmt,
                              "conversation_id": stl.session_state.cid
                              })
    try: 
        response_json = response.json()  # Now it's a Python dict or list
        text_output = response.json()['items'][0]['text']
        stl.write(text_output)  
       
    except ValueError:
       print("RAW BODY:")
       raise   # or handle gracefully   


#if stl.button("submit") and pmt:
 #   with stl.spinner("Guessing ..."):
  #      response = requests.post("http://localhost:7071/api/chat", json={"Question": pmt})
   #     stl.write("Response" , response)
    #    answer = response.json().get("response", "No response")
     #   stl.markdown("||| Answer:")
      #  stl.write("Response" , answer)
       # stl.markdown("|||")
        #stl.markdown("||| Sources")

