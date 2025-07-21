# app.py
import streamlit as stl
import requests

stl.title("GenAI Assistant with Voice")
pmt = stl.text_input("Ask Question")
print("Sesiion_id -",stl.session_state)

if "cid" not in stl.session_state:
    stl.session_state.cid = None

print("Session ID:", stl.session_state.cid)

if stl.button("Send"):
    response = requests.post("http://localhost:7071/api/chat",
                        json={"Question": pmt,
                              "conversation_id": stl.session_state.cid
                              })
    try:
       data = response.json()
       stl.session_state.cid = data["conversation_id"]
       stl.write(data["response"])
       stl.write("Evaluation Score:", data["eval_score"])
    except ValueError:
       print("RAW BODY:", response.text)
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

