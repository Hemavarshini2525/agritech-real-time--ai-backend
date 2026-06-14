import { useRef } from "react";
import "../styles/UploadCard.css";

function UploadCard({ onImageUpload, uploadedImage }) {
  const fileInputRef = useRef(null);

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onImageUpload(file);
    }
  };

  return (
    <div className="upload-card">
      {uploadedImage ? (
        <img src={uploadedImage} alt="Uploaded crop" className="uploaded-preview" />
      ) : (
        <div className="upload-icon">📤</div>
      )}

      <h2>Upload Crop Image</h2>

      <p>{uploadedImage ? "Image uploaded!" : "Take or upload crop image"}</p>

      <button onClick={handleButtonClick}>
        {uploadedImage ? "Change Image" : "Choose Image"}
      </button>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />
    </div>
  );
}

export default UploadCard;