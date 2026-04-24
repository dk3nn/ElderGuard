import { useState, useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Chat from "./UserChat";
import AppDemo from "./AppDemo";
import Login from "./Login";


function App() {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const currentUser = localStorage.getItem('currentUser');
        if (currentUser) {
            setUser(JSON.parse(currentUser));
        }
        setLoading(false);
    }, []);

    const handleLogout = () => {
        localStorage.removeItem('currentUser');
        setUser(null);
    };

    if (loading) {
        return <div>Loading...</div>;
    }

    return (
        <Routes>
            <Route 
                path="/" 
                element={user ? <Chat user={user} onLogout={handleLogout} /> : <Navigate to="/login" />} 
            />
            <Route 
                path="/login" 
                element={user ? <Navigate to="/" /> : <Login />} 
            />
            <Route path="/demo" element={<AppDemo />} />
        </Routes>
    );
}

export default App;