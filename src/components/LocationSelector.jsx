import { useState, useEffect } from "react";
import "../styles/LocationSelector.css";

function LocationSelector({ onLocationChange }) {
  const [selectedLocation, setSelectedLocation] = useState("Hyderabad");
  const [showMap, setShowMap] = useState(false);
  const [currentCoords, setCurrentCoords] = useState(null);

  const locations = [
    { name: "Hyderabad", coords: { lat: 17.3850, lng: 78.4867 } },
    { name: "Vijayawada", coords: { lat: 16.5062, lng: 80.6480 } },
    { name: "Guntur", coords: { lat: 16.3067, lng: 80.4365 } },
    { name: "Warangal", coords: { lat: 17.9689, lng: 79.5941 } },
    { name: "Nellore", coords: { lat: 14.4426, lng: 79.9865 } },
    { name: "Visakhapatnam", coords: { lat: 17.6869, lng: 83.2185 } },
    { name: "Bangalore", coords: { lat: 12.9716, lng: 77.5946 } },
    { name: "Pune", coords: { lat: 18.5204, lng: 73.8567 } },
  ];

  const handleLocationChange = (e) => {
    const location = e.target.value;
    setSelectedLocation(location);
    const coords = locations.find(loc => loc.name === location)?.coords;
    setCurrentCoords(coords);
    onLocationChange?.(location);
  };

  const getCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setCurrentCoords({ lat: latitude, lng: longitude });
          setSelectedLocation("Current Location");
          onLocationChange?.("Current Location");
          console.log("Current location:", latitude, longitude);
        },
        (error) => {
          console.error("Error getting location:", error);
          alert("Unable to get your location. Please enable location services.");
        }
      );
    } else {
      alert("Geolocation is not supported by your browser");
    }
  };

  const openMapLink = () => {
    if (currentCoords) {
      const mapUrl = `https://www.google.com/maps?q=${currentCoords.lat},${currentCoords.lng}`;
      window.open(mapUrl, "_blank");
    }
  };

  return (
    <div className="location-card">
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
        <circle cx="12" cy="10" r="3"></circle>
      </svg>

      <select value={selectedLocation} onChange={handleLocationChange}>
        {locations.map((loc) => (
          <option key={loc.name} value={loc.name}>
            {loc.name}
          </option>
        ))}
      </select>

      <div className="location-actions">
        <button onClick={getCurrentLocation} title="Get current location" className="current-location-btn">
          📍 Current
        </button>
        <button onClick={openMapLink} title="View on map" className="map-btn">
          🗺️ Map
        </button>
      </div>

      {currentCoords && (
        <div className="coords-display">
          <small>
            {currentCoords.lat.toFixed(4)}, {currentCoords.lng.toFixed(4)}
          </small>
        </div>
      )}
    </div>
  );
}

export default LocationSelector;