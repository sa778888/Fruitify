import React, { useState, useRef } from 'react';
import Webcam from 'react-webcam';
import { Camera, FileText, X } from 'lucide-react';

const WebcamAnalysisPage = () => {
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [report, setReport] = useState(null);
  const webcamRef = useRef(null);

  const videoConstraints = {
    facingMode: 'environment',
    width: 720,
    height: 480
  };

  const startWebcam = () => {
    setIsWebcamActive(true);
  };

  const stopWebcam = () => {
    setIsWebcamActive(false);
  };

  const generateReport = async () => {
    if (webcamRef.current) {
      setIsAnalyzing(true);
      // Capture the current frame
      const imageSrc = webcamRef.current.getScreenshot();
      
      // Simulate analysis delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setReport({
        type: 'success',
        timestamp: new Date().toLocaleString(),
        details: {
          defectsFound: 2,
          quality: 'Good',
          recommendations: 'Minor improvements needed in sorting process'
        }
      });
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="text-center pt-10 pb-45 px-4">
      <h2 className="text-6xl font-bold mb-6 bg-gradient-to-r from-cyan-200 to-green-600 text-transparent bg-clip-text">
        Fruit Quality Analysis
      </h2>
      
      <p className="text-gray-200 mb-10 max-w-xl mx-auto text-lg">
        Real-time fruit quality assessment using advanced AI technology. Start your analysis by enabling the camera.
      </p>

      <div className="mt-20 flex flex-col md:flex-row items-start justify-between max-w-4xl mx-auto gap-8">
        <div className="w-full md:w-1/2 relative aspect-video bg-gray-900 rounded-lg overflow-hidden">
          {isWebcamActive ? (
            <>
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                className="w-full h-full object-cover"
              />
              <button
                onClick={stopWebcam}
                className="absolute top-4 right-4 p-2 bg-gradient-to-r from-red-500 to-red-700 rounded-full hover:from-red-700 hover:to-red-500 transition-all duration-300"
              >
                <X className="w-6 h-6 text-white" />
              </button>
            </>
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <Camera className="w-16 h-16 text-gray-400" />
            </div>
          )}
        </div>

        <div className="w-full md:w-1/2 space-y-6">
          <div className="space-y-4">
            <button
              onClick={startWebcam}
              disabled={isWebcamActive}
              className="w-full cursor-pointer px-6 py-3 bg-gradient-to-r from-green-500 to-green-700 text-white hover:from-green-700 hover:to-green-500 rounded-3xl font-semibold transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <span className="flex items-center justify-center gap-2">
                <Camera className="w-4 h-4" />
                Start Camera
              </span>
            </button>

            <button
              onClick={generateReport}
              disabled={!isWebcamActive || isAnalyzing}
              className="w-full cursor-pointer px-6 py-3 bg-gradient-to-r from-cyan-500 to-cyan-700 text-white hover:from-cyan-700 hover:to-cyan-500 rounded-3xl font-semibold transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <span className="flex items-center justify-center gap-2">
                <FileText className="w-4 h-4" />
                {isAnalyzing ? 'Analyzing...' : 'Generate Report'}
              </span>
            </button>
          </div>

          {report && (
            <div className="mt-6">
              {report.type === 'error' ? (
                <div className="text-red-500 bg-red-50 p-4 rounded-lg">
                  {report.message}
                </div>
              ) : (
                <div className="bg-gradient-to-r from-gray-800 to-gray-900 text-gray-200 p-6 rounded-lg">
                  <h3 className="text-2xl font-bold mb-4 bg-gradient-to-r from-cyan-200 to-green-600 text-transparent bg-clip-text">
                    Analysis Report
                  </h3>
                  <div className="space-y-2 text-lg">
                    <p>Time: {report.timestamp}</p>
                    <p>Defects Found: {report.details.defectsFound}</p>
                    <p>Quality Rating: {report.details.quality}</p>
                    <p>Recommendations: {report.details.recommendations}</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default WebcamAnalysisPage;