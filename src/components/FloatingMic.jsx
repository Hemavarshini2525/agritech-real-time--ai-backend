import { useEffect, useRef, useState } from "react";
import "../styles/FloatingMic.css";

function FloatingMic({ onRecordStart, isRecording }) {
  const [hasPermission, setHasPermission] = useState(false);
  const recognitionRef = useRef(null);
  const [transcript, setTranscript] = useState("");

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      console.warn("Speech Recognition not supported");
      return;
    }

    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = false;
    recognitionRef.current.interimResults = true;

    recognitionRef.current.onstart = () => {
      setHasPermission(true);
      onRecordStart?.();
    };

    recognitionRef.current.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          setTranscript(transcript);
          console.log("Final transcript:", transcript);
        } else {
          interim += transcript;
        }
      }
    };

    recognitionRef.current.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
    };

    recognitionRef.current.onend = () => {
      console.log("Recording stopped");
    };
  }, [onRecordStart]);

  const handleMicClick = () => {
    if (recognitionRef.current) {
      if (isRecording) {
        recognitionRef.current.stop();
      } else {
        setTranscript("");
        recognitionRef.current.start();
      }
    }
  };

  return (
    <button 
      className={`mic-btn ${isRecording ? "recording" : ""}`}
      onClick={handleMicClick}
      title={isRecording ? "Stop recording" : "Start voice recording"}
    >
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
        <path d="M12 1a3 3 0 0 0-3 3v12a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
        <line x1="12" y1="19" x2="12" y2="23"></line>
        <line x1="8" y1="23" x2="16" y2="23"></line>
      </svg>
      {isRecording && <span className="pulse"></span>}
    </button>
  );
}

export default FloatingMic;