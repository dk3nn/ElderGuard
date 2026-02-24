import { Routes, Route } from "react-router-dom";
import Chat from "./UserChat";
import SignUp from "./SignUp";


function App() {
  return (
    <Routes>
      <Route path="/" element={<Chat />} />
      <Route path="/signup" element={<SignUp />} />
      
    </Routes>
  );
}

export default App;