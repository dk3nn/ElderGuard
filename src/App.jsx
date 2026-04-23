import { Routes, Route } from "react-router-dom";
import Chat from "./UserChat";
import SignUp from "./SignUp";
import AppDemo from "./AppDemo";


function App() {
  return (
    <Routes>
      <Route path="/" element={<Chat />} />
      <Route path="/signup" element={<SignUp />} />
      <Route path="/demo" element={<AppDemo />} />
    </Routes>
  );
}

export default App;