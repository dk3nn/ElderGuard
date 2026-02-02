import { useState } from "react";


function SignUp(){

    const [formData, setFormData] = useState({

        firstName: "",
        lastName: "",
        password: "",
        confirmPassword: "",
        email: ""

    });

    const [errors, setErrors] = useState({});
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setFormData(prev => ({ ...prev, [name]: type === "checkbox" ? checked : value }));

        if(errors[name]){
            setErrors(prev => ({ ...prev, [name]: '' }));

        }
    };

    const validate = () => {
        const newErrors = {};

        if(!formData.firstName.trim()){
            newErrors.firstName = "First name is required";
        }

        if(!formData.lastName.trim()){
            newErrors.lastName = "Last name is required";
        }
        
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

        if(!formData.confirmPassword){
            newErrors.confirmPassword = "Confirm password";
        }else if(formData.confirmPassword !== formData.password){
            newErrors.confirmPassword = "Passwords must match";
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


return (

    <div className="signup-container">
    <div className= "signUpCard">
    <div className= "signUpHeader">
        <h1> Sign Up</h1>
    </div>

    <form onSubmit={handleSubmit} className="signUpForm" noValidate>
        <div className="nameFields">
            <div className="form-group">
                <label htmlFor="firstName">First Name</label>
                <input type="text" id="firstName" name="firstName" value={formData.firstName}
                 onChange={handleChange} placeholder="John" className={errors.firstName ? "error" : ""}/>
                 {errors.firstName && (<span className="errorMessage">{errors.firstName}</span>)}
                 </div>
            <div className="form-group">
                <label htmlFor="lastName">Last Name</label>
                <input type="text" id="lastName" name="lastName" value={formData.lastName}
                 onChange={handleChange} placeholder="Doe" className={errors.lastName ? "error" : ""}/>
                 {errors.lastName && (<span className="errorMessage">{errors.lastName}</span>)}
                 </div>
            
            <div className="form-group">
                <label htmlFor="email">Email</label>
                <input type="text" id="email" name="email" value={formData.email}
                 onChange={handleChange} placeholder="JohnDoe@example.com" className={errors.email ? "error" : ""}/>
                 {errors.email && (<span className="errorMessage">{errors.email}</span>)}
                 </div>
            
        <div className="passwordFields">
            <div className="form-group">
                <label htmlFor="password">Password</label>
                <input type="password" id="password" name="password" value={formData.password}
                 onChange={handleChange} placeholder="Must be at least 6 characters" className={errors.password ? "error" : ""}/>
                 {errors.password && (<span className="errorMessage">{errors.password}</span>)}
                 </div>

            <div className="form-group">
                <label htmlFor="confirmPassword">Confirm Password</label>
                <input type="password" id="confirmPassword" name="confirmPassword" value={formData.confirmPassword}
                 onChange={handleChange} placeholder="Re-enter your password" className={errors.confirmPassword ? "error" : ""}/>
                 {errors.confirmPassword && (<span className="errorMessage">{errors.confirmPassword}</span>)}
                 </div>
        </div>
        </div>

        <button type="submit" className="submit-btn" onClick={isSubmitting}>Submit</button>

        <p className="login-link">
            Already have an account? Log in
            <a href="/login">Log in</a> 
        </p>

    </form>
</div>
</div>
);


}
