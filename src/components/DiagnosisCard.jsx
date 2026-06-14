import "../styles/DiagnosisCard.css";

function DiagnosisCard() {
  return (
    <div className="diagnosis-card">
      <h2>AI Diagnosis Results</h2>

      <div className="result">
        <h3>No Diagnosis Yet</h3>

        <p>
          Upload an image to get AI diagnosis.
        </p>
      </div>
    </div>
  );
}

export default DiagnosisCard;