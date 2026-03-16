import { useState } from "react";
import "./Login.css";

function Login(){
    const [formData, setFormData] = useState({
        email: "",
        password: ""
    });

    const [errors, setErros] = useState({});
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
        if(errors[name]){
            setErros(prev => ({ ...prev, [name]: '' }));
        }
    };

    const validate = () => {
        const newErrors = {};

        if(!formData.email){
            newErrors.email = "Email is required";
        }else if(!/\S+@\S+\.\S+/.test(formData.email)){
            newErrors.email = "Email address is invalid";
        }

        if(!formData.password){
            newErrors.password = "Password is required";
        }else if(formData.password.length < 6){
            newErrors.password = "Password must be at least 6 characters";
        }
        return newErrors;
    };

    const handleSubmit = (e) => {
         e.preventDefault();

        const validationErrors = validate();
            if(Object.keys(validationErrors).length > 0){
                setErrors(validationErrors);
                return;
            }

            setIsSubmitting(true);

            //add api call here


    };

    return(
         <div className="login-container">
    <div className= "loginCard">
    <div className= "loginHeader">
        <h1> Log In</h1>
    </div>

    <form className="login-form" onSubmit={handleSubmit}>
        <div className="form-group">
            <label htmlFor="email">Email</label>
            <input type="email" id="email" name="email" value={formData.email} onChange={handleChange} className={errors.email ? 'input-error' : ''} />
            {errors.email && <span className="error">{errors.email}</span>}
        </div>
        <div className="form-group">
            <label htmlFor="password">Password</label>
            <input type="password" id="password" name="password" value={formData.password} onChange={handleChange} className={errors.password ? 'input-error' : ''} />
            {errors.password && <span className="error">{errors.password}</span>}
        </div>
        <button type="submit" className="submit-btn" onClick={isSubmitting}>Submit</button>

        <p className="login-link">
            Don't have an account? Sign up Here
            <a href="/SignUp">Sign up</a> 
        </p>

    </form>
</div>
</div>
);
}
