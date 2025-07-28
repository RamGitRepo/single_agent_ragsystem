// src/App.js
import React, { useState, useEffect, useRef } from "react";
import { v4 as uuidv4 } from "uuid";
import axios from "axios";
import ReactMarkdown from "react-markdown";

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]); // [{ question, answer }]
  const [conversationId, setConversationId] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef(null);

  // Set or load conversation ID
  useEffect(() => {
    const storedCid = localStorage.getItem("cid") || uuidv4();
    localStorage.setItem("cid", storedCid);
    setConversationId(storedCid);
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSend = async () => {
    if (!question.trim()) return;
  
    const currentQuestion = question;
    setQuestion("");
    setIsLoading(true);
  
    try {
      const res = await axios.post("http://localhost:7080/api/agtchat", {
        Question: currentQuestion,
        conversation_id: conversationId,
      });
  
      const assistantText = res.data?.answer || "No response received.";
  
      // Push Q&A pair to history
      setMessages((prev) => [...prev, { question: currentQuestion, answer: assistantText }]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [...prev, { question: currentQuestion, answer: "❌ API call failed." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-500 text-white font-poppins">
      <div className="max-w-4xl mx-auto py-10 px-6 flex flex-col h-full">
        {/* Title */}
        <h1 className="text-4xl font-bold mb-6">🤖 CV Assistant UI</h1>

        {/* Q&A Response Section */}
        {(messages.length > 0 || isLoading) && (
          <div className="bg-white/10 border border-white/20 rounded-lg p-6 space-y-6 mb-6">
            <h2 className="text-2xl font-bold mb-2"> </h2>

            {messages.map((msg, idx) => (
              <div key={idx} className="bg-white/5 p-4 rounded-lg">
                <p className="text-pink-200 font-semibold">Q: {msg.question}</p>
                <p className="mt-2 text-green-200 font-semibold">A:</p>
                <ReactMarkdown
                  components={{
                    p: ({ node, ...props }) => (
                      <p className="text-white leading-relaxed mb-2" {...props} />
                    ),
                    ul: ({ node, ...props }) => (
                      <ul className="list-disc list-inside mb-2" {...props} />
                    ),
                    li: ({ node, ...props }) => (
                      <li className="text-white mb-1" {...props} />
                    ),
                    a: ({ node, ...props }) => (
                      <a
                        className="text-blue-300 underline"
                        target="_blank"
                        rel="noopener noreferrer"
                        {...props}
                        >
                        {props.children}
                      </a>
                    ),
                    strong: ({ node, ...props }) => (
                      <strong className="text-white font-semibold" {...props} />
                    ),
                  }}
                >
                  {msg.answer.replace(/^Q:\s*/i, "").replace(/^A:\s*/i, "")}
                </ReactMarkdown>
              </div>
            ))}

            {isLoading && (
              <p className="text-white/60 italic animate-pulse">Assistant is thinking...</p>
            )}

            <div ref={bottomRef} />
          </div>
        )}

        {/* Input Area */}
        <div className="flex flex-col sm:flex-row gap-4">
          <textarea
            rows={1}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask me anything..."
            className="flex-1 p-4 rounded-lg border border-white/30 bg-white/10 backdrop-blur placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-white resize-none"
          />
          <button
            onClick={handleSend}
            className="px-6 py-3 bg-white text-indigo-700 font-bold rounded-lg hover:bg-indigo-200 transition-all duration-300"
          >
            Send 🚀
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
