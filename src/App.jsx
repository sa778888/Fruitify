import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import LandingPage from "./pages/LandingPage";
import WebcamAnalysisPage from "./pages/WebCamAnalysisPage";
import { LoginPage, SignupPage } from "./pages/Login";
import { useState } from "react";
import FileUpload from "./pages/Test";

export default function App() {
  const [auth,setAuth]=useState(null)
  return (
    <Router>
      <div className="bg-[#1A1A1A] min-h-screen text-white">
        <Navbar auth={auth} setAuth={setAuth} />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/analyse" element={<WebcamAnalysisPage/>}/>
          <Route path="/login" element={<LoginPage setAuth={setAuth} />} />
          <Route path="/signup" element={<SignupPage setAuth={setAuth} />} />
          <Route path="/test" element={<FileUpload  />} />

        </Routes>
      </div>
    </Router>
  );
}