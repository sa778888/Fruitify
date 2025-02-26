import React, { useState, useRef } from "react";

export default function CameraUpload() {
  const [message, setMessage] = useState("");
  const [stream, setStream] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Start the camera
  const startCamera = async () => {
    try {
      const userStream = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(userStream); // Save stream reference
      if (videoRef.current) {
        videoRef.current.srcObject = userStream;
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      setMessage("Failed to access camera.");
    }
  };

  // Stop the camera
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop()); // Stop all tracks
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null; // Clear video feed
    }
  };

  // Capture an image from the camera
  const captureImage = async () => {
  if (!videoRef.current || !canvasRef.current) return;

  const canvas = canvasRef.current;
  const context = canvas.getContext("2d");

  // Set canvas size to match video
  canvas.width = videoRef.current.videoWidth;
  canvas.height = videoRef.current.videoHeight;

  // Draw the video frame onto the canvas
  context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

  // Convert the image to a Blob
  canvas.toBlob(async (blob) => {
    if (!blob) return;
    const formData = new FormData();
    formData.append("file", blob, "captured_image.jpg");

    try {
      const response = await fetch("http://127.0.0.1:5000/track", {
        method: "POST",
        body: formData,
      });

      const data = await response.json(); // Parse response as JSON
      if (response.ok) {
        setMessage({
          message: data.message,
          predicted_fruit_freshness: data.predicted_fruit_freshness,
        });
      } else {
        setMessage({ message: `Error: ${data.message || "Unknown error"}`, predicted_fruit_freshness: "" });
      }
    } catch (error) {
      setMessage({ message: "Failed to connect to the backend!", predicted_fruit_freshness: "" });
      console.error(error);
    }
  }, "image/jpeg");
};


  return (
    <div className="p-4 border rounded-lg shadow-md max-w-sm mx-auto">
      <h2 className="text-lg font-semibold mb-2">Live Camera Upload</h2>
      <video ref={videoRef} autoPlay playsInline className="w-full mb-2" />
      <canvas ref={canvasRef} className="hidden" />
      <div className="flex gap-2">
        <button
          onClick={startCamera}
          className="bg-green-500 text-white px-4 py-2 rounded-md"
        >
          Start Camera
        </button>
        <button
          onClick={stopCamera}
          className="bg-red-500 text-white px-4 py-2 rounded-md"
          disabled={!stream} // Disable if camera is off
        >
          Stop Camera
        </button>
        <button
          onClick={captureImage}
          className="bg-blue-500 text-white px-4 py-2 rounded-md"
          disabled={!stream} // Disable if camera is off
        >
          Capture & Upload
        </button>
      </div>
      <div>

      {message && <h1 className="mt-2 text-sm text-gray-700">{message.message}</h1>}
      {message && <h1 className="mt-2 text-sm text-gray-700">{message.predicted_fruit_freshness}</h1>}
      </div>
    </div>
  );
}
