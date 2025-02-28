# Fruitify.AI

Fruitify.AI is an AI-powered system that assesses fruit freshness using both hardware and software components. The project combines sensor-based data collection with deep learning models to provide real-time fruit classification and freshness detection.

## Features
### Hardware Component
- **Methane Gas Sensor & Moisture Sensor:** Detects gas emissions and moisture levels to determine fruit freshness.
- **Custom Dataset Creation:** Used Arduino IDE to track sensor values for both fresh and rotten fruit.
- **Machine Learning Model:** Utilized an SVM (Support Vector Machine) model for classification.

### Software Component
- **Fruit Type Detection:** Implemented YOLOv8 to detect the type of fruit.
- **Freshness Detection:** Used a MobileNet + TensorFlow CNN model to classify fruit freshness.
- **Frontend:** Developed an appealing UI/UX using React.js and Tailwind CSS.
- **Backend:** Powered by Flask to handle model inference and data processing.

## Tech Stack
- **Hardware:** Arduino, Methane Gas Sensor, Moisture Sensor
- **Software:** Python, Flask, TensorFlow, YOLOv8, MobileNet, SVM
- **Frontend:** React.js, Tailwind CSS
- **Backend:** Flask


### Software Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/fruitify.ai.git
   cd fruitify.ai
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   npm install
   ```
3. Run the backend:
   ```sh
   python backend.py
   ```
4. Run the frontend:
   ```sh
   npm start
   ```

## Usage
1. Place a fruit in the sensor setup for hardware-based freshness detection.
2. Upload or capture an image for software-based fruit detection.
3. View freshness results and recommendations through the UI.





