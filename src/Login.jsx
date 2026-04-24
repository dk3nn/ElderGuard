import { useState } from "react";
import { GoogleOAuthProvider, GoogleLogin } from "@react-oauth/google";
import "./Login.css";

const CLIENT_ID = "1016439236821-6u44uhlkktmvcdk4s9e231mri63pt6kf.apps.googleusercontent.com";

function Login() {
    const [isLogin, setIsLogin] = useState(true);
    const [formData, setFormData] = useState({
        username: "",
        email: "",
        password: "",
        confirmPassword: ""
    });

    const [errors, setErrors] = useState({});
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
        if (errors[name]) {
            setErrors(prev => ({ ...prev, [name]: '' }));
        }
    };

    const toggleMode = () => {
        setIsLogin(!isLogin);
        setFormData({
            username: "",
            email: "",
            password: "",
            confirmPassword: ""
        });
        setErrors({});
    };

    const validate = () => {
        const newErrors = {};

        if (!isLogin && !formData.username.trim()) {
            newErrors.username = "Username is required";
        }

        if (!formData.email) {
            newErrors.email = "Email is required";
        } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
            newErrors.email = "Email is invalid";
        }

        if (!formData.password) {
            newErrors.password = "Password is required";
        } else if (formData.password.length < 6) {
            newErrors.password = "Password must be at least 6 characters";
        }

        if (!isLogin && formData.password !== formData.confirmPassword) {
            newErrors.confirmPassword = "Passwords don't match";
        }

        return newErrors;
    };

    // Submission for both login and signup
    const handleSubmit = async (e) => {
        e.preventDefault();

        const validationErrors = validate();
        if (Object.keys(validationErrors).length > 0) {
            setErrors(validationErrors);
            return;
        }

        setIsSubmitting(true);

        try {
            const endpoint = isLogin ? '/api/login' : '/api/signup';
            const res = await fetch(`http://localhost:5000${endpoint}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    username: formData.username,
                    email: formData.email,
                    password: formData.password
                })
            });

            const data = await res.json();

            if (data.success) {
                localStorage.setItem('currentUser', JSON.stringify(data.user ||
                    {username: formData.username,
                    email: formData.email,
                    authType: 'email'}));
                window.location.reload();
                    }else {
                setErrors({ general: data.message || "Login failed" });
            }

        } catch (error) {
            console.error("Login error:", error);
            setErrors({ general: "An error occurred during login" });
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleGoogleSuccess = async (credentialResponse) => {
        setIsSubmitting(true);
        try {
            const response = await fetch(
                `https://oauth2.googleapis.com/tokeninfo?id_token=${credentialResponse.credential}`
            );
            const userData = await response.json();

            if (userData.email) {
                
                await fetch('http://localhost:5000/api/google-login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: userData.name,
                        email: userData.email,
                        authType: 'google'
                    })
                });

                const user = {
                    username: userData.name,
                    email: userData.email,
                    authType: 'google'
                };

                localStorage.setItem('currentUser', JSON.stringify(user));
                window.location.reload();
            }else {
                setErrors({ general: "Failed to login with Google" });
            }

        } catch (error) {
            console.error("Google login error:", error);
            setErrors({ general: "An error occurred during Google login" });
        } finally {
            setIsSubmitting(false);
        }
    };
     
    return (
        <GoogleOAuthProvider clientId={CLIENT_ID}>
            <div className="login-container">
                <div className="loginCard">
                    <div className="loginHeader">
                        <h1>{isLogin ? 'Log In' : 'Sign Up'}</h1>
                    </div>

                    {errors.general && (
                        <div className="error-message">{errors.general}</div>
                    )}

                    <form className="login-form" onSubmit={handleSubmit}>
                        {!isLogin && (
                            <div className="form-group">
                                <label htmlFor="username">Username</label>
                                <input
                                    type="text"
                                    id="username"
                                    name="username"
                                    value={formData.username}
                                    onChange={handleChange}
                                    className={errors.username ? 'input-error' : ''}
                                />
                                {errors.username && <span className="error">{errors.username}</span>}
                            </div>
                        )}

                        <div className="form-group">
                            <label htmlFor="email">Email</label>
                            <input
                                type="email"
                                id="email"
                                name="email"
                                value={formData.email}
                                onChange={handleChange}
                                className={errors.email ? 'input-error' : ''}
                            />
                            {errors.email && <span className="error">{errors.email}</span>}
                        </div>

                        <div className="form-group">
                            <label htmlFor="password">Password</label>
                            <input
                                type="password"
                                id="password"
                                name="password"
                                value={formData.password}
                                onChange={handleChange}
                                className={errors.password ? 'input-error' : ''}
                            />
                            {errors.password && <span className="error">{errors.password}</span>}
                        </div>

                        {!isLogin && (
                            <div className="form-group">
                                <label htmlFor="confirmPassword">Confirm Password</label>
                                <input
                                    type="password"
                                    id="confirmPassword"
                                    name="confirmPassword"
                                    value={formData.confirmPassword}
                                    onChange={handleChange}
                                    className={errors.confirmPassword ? 'input-error' : ''}
                                />
                                {errors.confirmPassword && (
                                    <span className="error">{errors.confirmPassword}</span>
                                )}
                            </div>
                        )}

                        <button type="submit" className="submit-btn" disabled={isSubmitting}>
                            {isSubmitting ? 'Please wait...' : isLogin ? 'Log In' : 'Sign Up'}
                        </button>

                        <div className="divider">
                            <span>OR</span>
                        </div>

                        <div className="google-login-container">
                            <GoogleLogin
                                onSuccess={handleGoogleSuccess}
                                onError={() => setErrors({ general: 'Google login failed' })}
                                scope="https://www.googleapis.com/auth/gmail.readonly"
                            />
                        </div>

                        <p className="login-link">
                            {isLogin ? "Don't have an account?" : "Already have an account?"}
                            <button type="button" onClick={toggleMode} className="toggle-btn">
                                {isLogin ? 'Sign Up' : 'Log In'}
                            </button>
                        </p>
                    </form>
                </div>
            </div>
        </GoogleOAuthProvider>
    );
}

export default Login
