import React from "react";
import { Mail, Phone, MapPin, Github, Linkedin, Twitter } from "lucide-react";

const ContactUs = () => {
  return (
    <div id="contact" className="min-h-screen bg-black text-gray-300">
      <div className="max-w-6xl mx-auto p-6">
        <div className="grid md:grid-cols-2 gap-8">
          {/* Contact Form */}
          <div className="bg-[#1A1F1E] p-8 rounded-lg shadow-lg">
            <h2 className="text-2xl font-bold text-green-400 mb-6">Get in Touch</h2>
            <form className="space-y-6">
              <div>
                <label className="block text-gray-300 mb-2">Name</label>
                <input
                  type="text"
                  className="w-full bg-[#2e3630] text-gray-300 p-3 rounded-lg border border-green-500/30 focus:outline-none focus:border-green-400"
                />
              </div>
              <div>
                <label className="block text-gray-300 mb-2">Email</label>
                <input
                  type="email"
                  className="w-full bg-[#2e3630] text-gray-300 p-3 rounded-lg border border-green-500/30 focus:outline-none focus:border-green-400"
                />
              </div>
              <div>
                <label className="block text-gray-300 mb-2">Message</label>
                <textarea className="w-full bg-[#2e3630] text-gray-300 p-3 rounded-lg border border-green-500/30 focus:outline-none focus:border-green-400 h-32"></textarea>
              </div>
              <button className="w-full bg-green-600 text-white py-3 rounded-lg hover:bg-green-500 transition">
                Send Message
              </button>
            </form>
          </div>

          {/* Contact Information */}
          <div className="space-y-6">
            <div className="bg-[#1A1F1E] p-6 rounded-lg shadow-lg">
              <h2 className="text-xl font-bold text-green-400 mb-4">Contact Information</h2>
              <div className="space-y-4 text-gray-300">
                <div className="flex items-center space-x-3">
                  <Mail className="w-5 h-5 text-green-400" />
                  <span>contact@fruitifyai.com</span>
                </div>
                <div className="flex items-center space-x-3">
                  <Phone className="w-5 h-5 text-green-400" />
                  <span>+1 (555) 123-4567</span>
                </div>
                <div className="flex items-center space-x-3">
                  <MapPin className="w-5 h-5 text-green-400" />
                  <span>123 Tech Avenue, Innovation City</span>
                </div>
              </div>
            </div>

            {/* Social Links */}
            <div className="bg-[#1A1F1E] p-6 rounded-lg shadow-lg">
              <h2 className="text-xl font-bold text-green-400 mb-4">Connect With Us</h2>
              <div className="flex space-x-4">
                <a href="#" className="p-3 bg-green-700 rounded-lg text-white hover:bg-green-600 transition">
                  <Twitter className="w-6 h-6" />
                </a>
                <a href="#" className="p-3 bg-green-700 rounded-lg text-white hover:bg-green-600 transition">
                  <Linkedin className="w-6 h-6" />
                </a>
                <a href="#" className="p-3 bg-green-700 rounded-lg text-white hover:bg-green-600 transition">
                  <Github className="w-6 h-6" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ContactUs;