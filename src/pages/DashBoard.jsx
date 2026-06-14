import { useState } from "react";
import Header from "../components/Header";
import LocationSelector from "../components/LocationSelector";
import SearchBar from "../components/SearchBar";
import UploadCard from "../components/UploadCard";
import DiagnosisCard from "../components/DiagnosisCard";
import FloatingMic from "../components/FloatingMic";

import "../styles/Dashboard.css";

function Dashboard() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [location, setLocation] = useState("Hyderabad");
  const [diagnosisResult, setDiagnosisResult] = useState(null);
  const [isRecording, setIsRecording] = useState(false);

  const handleImageUpload = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target.result);
      // Simulate AI diagnosis
      setTimeout(() => {
        setDiagnosisResult({
          disease: "Leaf Spot Disease",
          confidence: "92%",
          treatment: "Apply fungicide and ensure proper crop rotation",
        });
      }, 1500);
    };
    reader.readAsDataURL(file);
  };

  const handleSearch = (query) => {
    setSearchQuery(query);
    console.log("Searching for:", query, "in location:", location);
  };

  const handleVoiceStart = () => {
    setIsRecording(true);
    console.log("Voice recording started...");
    // Simulate voice recording for 3 seconds
    setTimeout(() => {
      setIsRecording(false);
      setSearchQuery("wheat rust disease");
      console.log("Voice recording stopped - Query set");
    }, 3000);
  };

  const handleLocationChange = (selectedLocation) => {
    setLocation(selectedLocation);
    console.log("Location changed to:", selectedLocation);
  };

  return (
    <div className="dashboard">
      <Header />
      <LocationSelector onLocationChange={handleLocationChange} />
      <SearchBar onSearch={handleSearch} />
      <UploadCard onImageUpload={handleImageUpload} uploadedImage={uploadedImage} />
      <DiagnosisCard result={diagnosisResult} />
      <FloatingMic onRecordStart={handleVoiceStart} isRecording={isRecording} />
    </div>
  );
}

export default Dashboard;