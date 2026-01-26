import { useState, useEffect } from "react";
import "./UserChat.css";

const USER_AVATAR =
  "https://static.vecteezy.com/system/resources/previews/028/569/170/large_2x/single-man-icon-people-icon-user-profile-symbol-person-symbol-businessman-stock-vector.jpg";

const BOT_AVATAR =
  "https://img.freepik.com/premium-photo/call-center-portrait-senior-man-with-headset-crm-contact-us-with-communication-professional-headshot-telecom-customer-service-male-consultant-with-help-desk-employee-mic_590464-209087.jpg";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState(null);

  // Bot welcome message
  useEffect(() => {
    const timer = setTimeout(() => {
      setMessages([
        { sender: "bot", text: "Hi! How can I help you today?" },
      ]);
    }, 800);

    return () => clearTimeout(timer);
  }, []);

  const handleSend = () => {
    if (!input && !image) return;

    setMessages((prev) => [
      ...prev,
      { sender: "user", text: input, image },
    ]);

    setInput("");
    setImage(null);
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
