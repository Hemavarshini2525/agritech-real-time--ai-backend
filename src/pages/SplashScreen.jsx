import "../styles/SplashScreen.css";

function SplashScreen() {
  return (
    <div className="splash">
      <svg
        width="80"
        height="80"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M2 22c1.591 1 7 -1 7 -8V8c0-7-5.408-9-7-8"></path>
        <path d="M22 22c-1.591 1-7-1-7-8V8c0-7 5.408-9 7-8"></path>
      </svg>
      <h1>AgroAI</h1>
      <p>Smart Crop Advisory for Farmers</p>
    </div>
  );
}

export default SplashScreen;