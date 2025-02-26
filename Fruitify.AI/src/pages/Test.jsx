import React, { useState, useRef } from "react";
import { Camera, StopCircle, Upload } from "lucide-react";

export default function Test() {
  const [message, setMessage] = useState("");
  const [stream, setStream] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const startCamera = async () => {
    try {
      const userStream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });
      setStream(userStream);
      if (videoRef.current) {
        videoRef.current.srcObject = userStream;
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      setMessage("Failed to access camera.");
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const captureImage = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const imageUrl = canvas.toDataURL("image/jpeg");
    setCapturedImage(imageUrl);

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const formData = new FormData();
      formData.append("file", blob, "captured_image.jpg");

      try {
        const response = await fetch("http://127.0.0.1:5000/track", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        if (response.ok) {
          setMessage({
            message: data.message,
            predicted_fruit_freshness: data.predicted_fruit_freshness,
          });
        } else {
          setMessage({
            message: `Error: ${data.message || "Unknown error"}`,
            predicted_fruit_freshness: "",
          });
        }
      } catch (error) {
        setMessage({
          message: "Failed to connect to the backend!",
          predicted_fruit_freshness: "",
        });
        console.error(error);
      }
    }, "image/jpeg");
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center space-y-6 mb-12">
          <h1 className="text-6xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent animate-gradient">
            Fruit Quality Analysis
          </h1>
          <p className="text-lg text-gray-300 max-w-2xl mx-auto">
            Experience real-time fruit quality assessment powered by advanced AI technology.
            Our system provides instant analysis for optimal freshness detection.
          </p>
        </div>

        <div className="bg-gray-800 rounded-3xl shadow-2xl overflow-hidden backdrop-blur-lg border border-gray-700">
          <div className="p-8">
            {/* Adjusted the video container size */}
            <div className="max-w-2xl mx-auto">
              <div className="relative aspect-video rounded-2xl overflow-hidden bg-black mb-6">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="absolute inset-0 w-full h-full object-cover"
                />
                <canvas ref={canvasRef} className="hidden" />
              </div>
            </div>

            <div className="flex flex-wrap gap-4 justify-center mb-8">
              <button
                onClick={startCamera}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 rounded-xl text-white font-medium transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-emerald-500/25"
              >
                <Camera size={20} />
                Start Camera
              </button>
              <button
                onClick={stopCamera}
                disabled={!stream}
                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg ${
                  stream
                    ? "bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 hover:shadow-red-500/25"
                    : "bg-gray-600 cursor-not-allowed"
                }`}
              >
                <StopCircle size={20} />
                Stop Camera
              </button>
              <button
                onClick={captureImage}
                disabled={!stream}
                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-300 transform hover:scale-105 shadow-lg ${
                  stream
                    ? "bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 hover:shadow-blue-500/25"
                    : "bg-gray-600 cursor-not-allowed"
                }`}
              >
                <Upload size={20} />
                Analyze Fruit
              </button>
            </div>

            {(capturedImage || message) && (
              <div className="bg-gray-700/50 rounded-2xl p-6 backdrop-blur-sm border border-gray-600">
                <div className="flex flex-col md:flex-row items-center gap-6">
                  {capturedImage && (
                    <div className="w-32 h-32 rounded-xl overflow-hidden shadow-lg">
                      <img
                        src={capturedImage}
                        alt="Captured Preview"
                        className="w-full h-full object-cover"
                      />
                    </div>
                  )}
                  <div className="flex-1 text-center md:text-left">
                    {message && (
                      <>
                        <p className="text-gray-300 mb-2">{message.message}</p>
                        <p className="text-2xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                          {message.predicted_fruit_freshness}
                        </p>
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}