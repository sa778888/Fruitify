import React from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import Button from "./Button";
import { Link } from "react-router-dom";

const Hero = () => {
  const navigate = useNavigate();

  return (
    <div className="text-center pt-21 pb-45 px-4">
      <motion.h2
        className="text-8xl font-bold mb-15 bg-gradient-to-r from-cyan-200 to-green-600 text-transparent bg-clip-text"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        Revolutionizing Export Fruit Quality Control
      </motion.h2>

      <motion.p
        className="text-[#e0e0e0] mt-4 max-w-2xl mx-auto text-[1.25rem]"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        Meet FruitifyAI, the next-gen automated inspection system designed to enhance quality assessment with AI-driven defect detection, seamless integration, and real-time sorting for optimized export standards.
      </motion.p>

      <motion.div
        className="mt-15 space-x-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <Link to="/analyse">
          <button className="cursor-pointer px-6 py-3 bg-gradient-to-r from-green-500 to-green-700 text-white hover:from-green-700 hover:to-green-500 rounded-3xl font-semibold transition-all duration-300">
            Analyse
          </button>
        </Link>
        <Button text="Learn More" outlined={true} />
      </motion.div>
    </div>
  );
};

export default Hero;
