// src/App.js
import React, { useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import axios from "axios";

function App() {
  const [question, setQuestion] = useState("");
  const [responseText, setResponseText] = useState("");
  const [conversationId, setConversationId] = useState("");

  useEffect(() => {
    const storedCid = sessionStorage.getItem("cid") || uuidv4();
    sessionStorage.setItem("cid", storedCid);
    setConversationId(storedCid);
    console.log("Conversation ID:", storedCid);
  }, []);

const [isLoading, setIsLoading] = useState(false);

const handleSend = async () => {
    if (!question.trim()) return;
  
    setIsLoading(true);          // Start loading
    setResponseText("");         // Clear previous response
  
    try {
      const res = await axios.post("http://localhost:7080/api/agtchat", {
        Question: question,
        conversation_id: conversationId,
      });

      const text = res.data?.items?.[0]?.text || "No response received.";
      setResponseText(text);
    } catch (error) {
      console.error("Error:", error);
      setResponseText("❌ API call failed. See console for details.");
    } finally {
      setIsLoading(false);       // ✅ Always stop loading
    }
  };
  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-500 text-white font-poppins">
  <div className="max-w-4xl mx-auto py-10 px-6">
    {/* Logo or Title */}
    <div className="flex items-center justify-between mb-8">
      <h1 className="text-4xl font-bold">🤖 CV  Assistant UI</h1>
      <p className="text-center text-white/80 text-lg mb-6 animate-fade-in">
     </p>
      {/* <img src={logo} alt="Logo" className="h-10" /> */}
    </div>

    {/* Input + Button */}
    <div className="flex flex-col sm:flex-row gap-4 mb-6">
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask me anything..."
        className="flex-1 p-4 rounded-lg border border-white/30 bg-white/10 backdrop-blur placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-white"
      />
      <button
        onClick={handleSend}
        className="px-6 py-3 bg-white text-indigo-700 font-bold rounded-lg hover:bg-indigo-200 transition-all duration-300"
      >
        Send 🚀
      </button>
    </div>

    {/* Response */}
    {isLoading && (
  <p className="text-sm text-white/50 animate-pulse mt-2">
    Thinking...
  </p>
  )}
    {responseText && (
      <div className="bg-white/10 border border-white/20 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-2">Response:</h2>
        <p className="whitespace-pre-line">{responseText}</p>
      </div>
    )}
  </div>
</div>
  );
}

export default App;
