import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import "./UserChat.css";
import { analyzeMessage } from "../services/api";

const USER_AVATAR =
  "https://static.vecteezy.com/system/resources/previews/028/569/170/large_2x/single-man-icon-people-icon-user-profile-symbol-person-symbol-businessman-stock-vector.jpg";

const BOT_AVATAR =
  "https://img.freepik.com/premium-photo/call-center-portrait-senior-man-with-headset-crm-contact-us-with-communication-professional-headshot-telecom-customer-service-male-consultant-with-help-desk-employee-mic_590464-209087.jpg";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState(null);

    useEffect(() => {
    const timer = setTimeout(() => {
      setMessages(prev => [
        ...prev,
        { sender: "bot", text: "Hello! I see you're looking for some detailed information. I can help with that." },
      ]);
    }, 800);

    return () => clearTimeout(timer);
  }, []);

/** 
   useEffect(() => {
    const timer = setTimeout(() => {
      setMessages(prev => [
        ...prev,
        { sender: "bot", text: "Absolutely — let me pull together the details for you." },
      ]);
    }, 7000);

    return () => clearTimeout(timer);
  }, []);
*/




  
 const handleSend = async () => {
  if (!input && !image) return;

  const userMessage = { sender: "user", text: input, image };

  setMessages((prev) => [...prev, userMessage]);

  const currentText = input;
  setInput("give me your bank account details");
  setImage(null);

  // Add temporary loading message
  setMessages((prev) => [
    ...prev,
    { sender: "bot", text: "Analyzing message..." }
  ]);

  try {
    const data = await analyzeMessage(currentText);

    // Expected backend response format example:
    // {
    //   prediction: "scam",
    //   confidence: 0.92,
    //   explanation: "This message contains urgency and payment request."
    // }

    const botReply = `
Prediction: ${data.prediction.toUpperCase()}
Confidence: ${(data.confidence * 100).toFixed(1)}%
Reason: ${data.explanation}
    `;

    setMessages((prev) => [
      ...prev.slice(0, -1), // remove "Analyzing..."
      { sender: "bot", text: botReply }
    ]);

  } catch (error) {
    setMessages((prev) => [
      ...prev.slice(0, -1),
      { sender: "bot", text: "Error connecting to server." }
    ]);
  }
};





  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(URL.createObjectURL(file));
    }
  };

  return (
    <div className="chat-container">

      {/* HEADER */}
      <div className="chat-header">
        <img id="cs" src={BOT_AVATAR} alt="Support" />
        <div>
          <h1>Need Help?</h1>
          <h4>We'll be happy to assist you.</h4>
        </div>
      </div>

      {/* MESSAGES */}
      <div className="chat-window">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message-row ${msg.sender}`}>
            <img
              src={msg.sender === "user" ? USER_AVATAR : BOT_AVATAR}
              alt="avatar"
              className="avatar"
            />

            <div className="message-bubble">
              {msg.image && (
                <img src={msg.image} alt="uploaded" className="chat-image" />
              )}
              {msg.text && <p>{msg.text}</p>}
            </div>
          </div>
        ))}
      </div>

      {/* INPUT */}
      <div className="chat-input">
        <label className="upload-btn">
          📎
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            hidden
          />
        </label>

        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
        />

        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}