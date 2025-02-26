import React from "react";
import Card from "./Card";
const Features = () => {
  return (
    <div id="features" className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl mx-auto px-6 pb-20">
      <Card
        icon="🍏"
        title="Smart Inspection"
        description="AI-powered defect detection and ripeness evaluation for export-quality fruits."
      />
      <Card
        icon="🚀"
        title="Seamless Integration"
        description="Easily attaches to conveyor systems for real-time monitoring and sorting."
      />
      <Card
        icon="📦"
        title="Automated Sorting"
        description="Ensures efficient packaging with minimal manual intervention."
      />
    </div>
  );
};

export default Features;