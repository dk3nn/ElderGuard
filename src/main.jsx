import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import Chat from './UserChat.jsx'
import './UserChat.css';


createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Chat />


    <App />
  </StrictMode>,
)
