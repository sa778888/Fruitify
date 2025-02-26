import React from "react";
import { useState,useEffect } from "react";
import Button from "./Button";
import { Link } from "react-router-dom";

const Navbar = ({auth,setAuth}) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  useEffect(() => {
    const token = localStorage.getItem("authToken");
    setIsAuthenticated(!!token);
  }, [auth]);
  const handleLogout = () => {
    localStorage.removeItem("authToken");
    setIsAuthenticated(false);
  };

  return (
    <nav className="sticky top-0 z-50 backdrop-blur-xl bg-black/30 border-b border-white/10">
      <div className="flex flex-wrap justify-between items-center p-4 max-w-7xl mx-auto">
        <Link to="/">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-green-500 text-transparent bg-clip-text">
            Fruitify.ai
          </h1>
        </Link>
        <div className="flex items-center space-x-6">
          <a
            href="#features"
            className="hover:text-green-400 transition-colors duration-300"
          >
            Features
          </a>
          <a
            href="#about"
            className="hover:text-green-400 transition-colors duration-300"
          >
            About
          </a>
          <a
            href="#contact"
            className="hover:text-green-400 transition-colors duration-300"
          >
            Contact
          </a>
          

         { isAuthenticated ? (
    <button onClick={handleLogout} className="bg-red-500 text-white px-4 py-2 rounded">
      Logout
    </button>
  ) : (
    <>
    <a
            href="/login"
            className="text-green-500 hover:text-white hover:underline transition-colors duration-300"
          >
            Log in
          </a>
          
          <Link to="/signup">
          <Button text="Sign up" className="ml-4" />
          </Link></>
    
  )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;