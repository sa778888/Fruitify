import React from "react";
import { motion } from "framer-motion";
import Hero from "../components/Hero";
import Features from "../components/Features";
import AboutUs from "./AboutUs";
import ContactUs from "./ContactUs";

const sectionVariants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: { opacity: 1, scale: 1, transition: { duration: 1.2, ease: "easeOut" } },
};

const LandingPage = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
      className="overflow-hidden"
    >
      {/* Hero Section - Staggered Entrance */}
      <motion.section
        className="min-h-screen flex items-center justify-center"
        initial={{ opacity: 0, y: -100 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: "easeOut", delay: 0.3 }}
      >
        <Hero />
      </motion.section>

      {/* Features Section - Explosive Pop-up */}
      <motion.section
        className="min-h-screen flex items-center justify-center"
        variants={sectionVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <motion.div
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 1, ease: "easeOut", type: "spring", stiffness: 100 }}
        >
          <Features />
        </motion.div>
      </motion.section>

      {/* About Us Section - 3D Flip-in Animation */}
      <motion.section
        className="min-h-screen flex items-center justify-center"
        initial={{ rotateY: 90, opacity: 0 }}
        whileInView={{ rotateY: 0, opacity: 1 }}
        transition={{ duration: 1.2, ease: "easeOut" }}
        viewport={{ once: true }}
      >
        <AboutUs />
      </motion.section>

      {/* Contact Us Section - Wavy Floating Effect */}
      <motion.section
        className="min-h-screen flex items-center justify-center"
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: "easeOut", delay: 0.3 }}
        viewport={{ once: true }}
      >
        <motion.div
          animate={{ y: [-10, 10, -10] }}
          transition={{ duration: 3, repeat: Infinity, repeatType: "reverse", ease: "easeInOut" }}
          whileHover={{ scale: 1.1 }}
        >
          <ContactUs />
        </motion.div>
      </motion.section>
    </motion.div>
  );
};

export default LandingPage;
